from dataclasses import dataclass
import time
import random

import keras
import numpy as np
import ray
from tqdm import tqdm

from model import build_resnet
from mcts import MCTS
from buffer import ReplayBuffer
from reversi import Reversi
import datetime
from loguru import logger
import os

os.environ["KERAS_BACKEND"] = "jax"

@dataclass
class Sample:
    state: list
    mcts_policy: list
    player: int
    reward: int


@ray.remote(num_cpus=1, num_gpus=0)
def selfplay(weights, num_mcts_simulations, dirichlet_alpha, reversi):

    state = reversi.initialize()

    record = []
    model = build_resnet(n_rows=reversi.n_rows, n_cols=reversi.n_cols, action_space=reversi.n_action_space)
    # model.predict(rvrs.get_nn_state(state, 1))
    model.set_weights(weights)
    mcts = MCTS(model=model, reversi=reversi, alpha=dirichlet_alpha)
    current_player = 1
    done = False
    i = 0

    while not done:
        mcts_policy = mcts.search(
            root_state=state,
            current_player=current_player,
            num_simulations=num_mcts_simulations,
        )

        if i <= 10:
            # For the first 30 moves of each game, the temperature is set to τ = 1;
            # this selects moves proportionally to their visit count in MCTS
            action = np.random.choice(range(reversi.n_action_space), p=mcts_policy)
        else:
            action = random.choice(np.where(np.array(mcts_policy) == max(mcts_policy))[0])

        record.append(Sample(state, mcts_policy, current_player, None))
        next_state, done = reversi.step(state, action, current_player)
        state = next_state
        current_player = -current_player
        i += 1

    #: win: 1, lose: -1, draw: 0
    reward_first, reward_second = reversi.get_result(state)
    for sample in reversed(record):
        sample.reward = reward_first if sample.player == 1 else reward_second
    return record


@ray.remote(num_cpus=1, num_gpus=0)
def testplay(current_weights, num_mcts_simulations, reversi, save_dir,
             dirichlet_alpha=None, n_testplay=24):

    t = time.time()
    win_count = 0

    model = build_resnet(n_rows=reversi.n_rows, n_cols=reversi.n_cols, action_space=reversi.n_action_space)
    model.set_weights(current_weights)

    for n in range(n_testplay):
        alphazero = random.choice([1, -1])
        mcts = MCTS(model=model, alpha=dirichlet_alpha, reversi=reversi)
        state = reversi.initialize()

        current_player = 1
        done = False

        while not done:
            if current_player == alphazero:
                mcts_policy = mcts.search(
                    root_state=state,
                    current_player=current_player,
                    num_simulations=num_mcts_simulations
                )
                action = np.argmax(mcts_policy)
            else:
                action = reversi.get_greedy_action(state, current_player, epsilon=0.3)

            next_state, done = reversi.step(state, action, current_player)

            state = next_state
            current_player = -1 * current_player

        reward_first, reward_second = reversi.get_result(state)
        reward = reward_first if alphazero == 1 else reward_second
        result = "win" if reward == 1 else "lose" if reward == -1 else "draw"

        if reward > 0:
            win_count += 1

        stone_first, stone_second = reversi.count_discs(state)

        if alphazero == 1:
            stone_az, stone_tester = stone_first, stone_second
            color = "black"
        else:
            stone_az, stone_tester = stone_second, stone_first
            color = "white"

        message = f"AlphaZero ({color}) {result}: {stone_az} vs {stone_tester}"

        reversi.save_img(state, os.path.join(save_dir, 'img'), f"test_{n}.png", message)

    elapsed = time.time() - t

    return win_count, win_count / n_testplay, elapsed



def main(num_cpus, n_episodes=10000, buffer_size=40000,
         batch_size=64, epochs_per_update=5,
         num_mcts_simulations=50,
         update_period=300, test_period=300,
         n_testplay=20,
         save_period=3000,
         dirichlet_alpha=0.35):
    
    n_rows = 6
    n_cols = 6

    ray.init(num_cpus=num_cpus, num_gpus=1, local_mode=False)

    str_now = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    save_dir = str_now
    os.makedirs(save_dir, exist_ok=True)

    logger.add(os.path.join(save_dir, 'log.txt'))


    reversi = Reversi(n_rows, n_cols)
    model = build_resnet(n_rows=reversi.n_rows, n_cols=reversi.n_cols, action_space=reversi.n_action_space)
    model.summary()
    current_weights = ray.put(model.get_weights())

    # #: initialize network parameters
    # dummy_state = othello.encode_state(othello.get_initial_state(), 1)
    # network.predict(dummy_state)


    optimizer = keras.optimizers.Adam(learning_rate=1e-3)
    replay = ReplayBuffer(buffer_size=buffer_size, reversi=reversi)
    
    model.compile(
        optimizer,
        loss=[keras.losses.CategoricalCrossentropy(), keras.losses.MeanSquaredError()],
    )

    #: 並列Selfplay
    work_in_progresses = [
        selfplay.remote(current_weights, num_mcts_simulations, dirichlet_alpha, reversi)
        for _ in range(num_cpus - 2)
    ]

    test_in_progress = testplay.remote(
        current_weights, num_mcts_simulations, reversi, save_dir, n_testplay=n_testplay
    )


    n_updates = 0
    n = 0

    loss_m = keras.metrics.Mean()

    while n <= n_episodes:

        for _ in tqdm(range(update_period)):
            #: selfplayが終わったプロセスを一つ取得
            finished, work_in_progresses = ray.wait(work_in_progresses, num_returns=1)
            replay.add_record(ray.get(finished[0]))
            work_in_progresses.extend([
                selfplay.remote(current_weights, num_mcts_simulations, dirichlet_alpha, reversi)
            ])
            n += 1

        #: Update network
        if len(replay) >= 20000:
            num_iters = epochs_per_update * (len(replay) // batch_size)
            for i in range(num_iters):
                states, mcts_policy, rewards = replay.get_minibatch(batch_size=batch_size)

                loss = model.train_on_batch(states, [mcts_policy, rewards])
                loss_m.update_state(loss)
                n_updates += 1

                if i % 100 == 0:
                    logger.info(f'{n_updates=}: {loss_m.result():.4f}')

            current_weights = ray.put(model.get_weights())

        if n % test_period == 0:
            print(f"{n - test_period}: TEST")
            win_count, win_ratio, elapsed_time = ray.get(test_in_progress)
            print(f"SCORE: {win_count}, {win_ratio}, Elapsed: {elapsed_time}")
            test_in_progress = testplay.remote(
                current_weights, num_mcts_simulations, reversi, save_dir, n_testplay=n_testplay
            )

            step = n - test_period
            step_b = n
            buffersize = len(replay)
            logger.info(f'{step=}: {win_count=}')
            logger.info(f'{step=}: {win_ratio=}')
            logger.info(f'{step_b=}: {buffersize=}')

        if n % save_period == 0:
            model.save_weights(os.path.join(save_dir, 'weights'))


if __name__ == "__main__":
    main(num_cpus=28)
