# coding: utf-8
import os
import numpy as np
from tqdm import tqdm
import datetime
import time
import ray
from loguru import logger

os.environ["KERAS_BACKEND"] = "jax"
import keras

from reversi import Reversi
from mcts import MCTS
from buffer import ReplayBuffer, Sample
from model import build_resnet


@ray.remote(num_cpus=1, num_gpus=0)
def selfplay(*, weights, num_mcts_simulations, dirichlet_alpha, reversi, model_fn, model_params):
    """
    SelfPlayを行って棋譜を返す
    """
    model = model_fn(**model_params)
    model.set_weights(weights)
    mcts = MCTS(model=model)

    record = []
    state = reversi.initialize()
    current_player = 1
    done = False
    i = 0
    while not done:
        mcts_policy = mcts.search(
            root_state=state,
            current_player=current_player,
            num_simulations=num_mcts_simulations,
            game=reversi,
            alpha=dirichlet_alpha,
        )

        if i <= 10:
            # 序盤の数手は完全ランダムにする (探索重視)
            action = np.random.choice(range(reversi.action_space_dims), p=mcts_policy)
        else:
            action = np.random.choice(np.where(np.array(mcts_policy) == max(mcts_policy))[0])
        i += 1

        record.append(Sample(state, mcts_policy, current_player, None))
        next_state, done = reversi.step(state, action, current_player)
        state = next_state
        current_player = -current_player

    # 遡ってrewardをセットする
    reward_sente, reward_gote = reversi.get_result(state)
    for sample in reversed(record):
        sample.reward = reward_sente if sample.player == 1 else reward_gote

    return record


@ray.remote(num_cpus=1, num_gpus=0)
def testplay(
    *,
    weights,
    num_mcts_simulations,
    reversi,
    save_dir,
    model_fn,
    model_params,
    dirichlet_alpha=None,
    num_testplay=24,
):
    """
    学習したモデルを使ってGreedyTesterと対局する
    """
    t = time.time()

    model = model_fn(**model_params)
    model.set_weights(weights)

    win_count = 0
    for n in range(num_testplay):
        alpha_zero = np.random.choice([1, -1])  # 手番を決める
        mcts = MCTS(model=model)

        state = reversi.initialize()
        current_player = 1
        done = False
        while not done:
            if current_player == alpha_zero:
                mcts_policy = mcts.search(
                    root_state=state,
                    current_player=current_player,
                    num_simulations=num_mcts_simulations,
                    game=reversi,
                    alpha=dirichlet_alpha,
                )
                action = np.argmax(mcts_policy)
            else:
                action = reversi.get_greedy_action(state, current_player, epsilon=0.3)

            next_state, done = reversi.step(state, action, current_player)
            state = next_state
            current_player = -1 * current_player

        # 結果判定・終了処理
        reward_sente, reward_gote = reversi.get_result(state)
        reward = reward_sente if alpha_zero == 1 else reward_gote
        result = "win" if reward == 1 else "lose" if reward == -1 else "draw"
        if reward > 0:
            win_count += 1

        # 結果を画像として保存
        sente_discs, gote_discs = reversi.count_discs(state)
        if alpha_zero == 1:
            az_discs, tester_discs = sente_discs, gote_discs
            az_color = "black"
        else:
            az_discs, tester_discs = gote_discs, sente_discs
            az_color = "white"
        message = f"AlphaZero ({az_color}) {result}: {az_discs} vs {tester_discs}"
        reversi.save_img(state, os.path.join(save_dir, "img"), f"test_{n}.png", message)

    elapsed = time.time() - t
    return win_count, win_count / num_testplay, elapsed


def main(
    *,
    num_cpus,
    save_dir_prefix,
    num_episodes=10000,
    update_period=300,  # SelfPlayを収集する単位
    buffer_size=40000,  # ReplayBufferのサイズ
    min_buffer_size_for_training=20000,  # 学習を開始する最小のバッファサイズ
    batch_size=64,
    epochs_per_update=5,
    num_mcts_simulations=50,
    dirichlet_alpha=0.35,
    test_period=300,
    num_testplay=20,
    save_period=3000,
):

    num_rows = 8
    num_cols = 8
    model_fn = build_resnet
    model_params = dict(num_rows=num_rows, num_cols=num_cols, action_space_dims=num_rows * num_cols + 1)

    str_now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    save_dir = save_dir_prefix + f"_{num_rows}x{num_cols}_" + str_now
    os.makedirs(save_dir, exist_ok=True)

    logger.add(os.path.join(save_dir, "log.txt"))
    logger.info(f"keras: {keras.__version__}")
    logger.info(f'backend: {os.environ["KERAS_BACKEND"]}')
    logger.info(f"Reversi: {num_rows=}, {num_cols=}")
    logger.info(f"Model: {model_fn=}, {model_params}")

    reversi = Reversi(num_rows, num_rows)
    replay = ReplayBuffer(buffer_size=buffer_size)

    model = model_fn(**model_params)
    model.compile(
        keras.optimizers.Adam(learning_rate=1e-3),
        loss=[keras.losses.CategoricalCrossentropy(), keras.losses.MeanSquaredError()],
    )
    model.summary()

    # 並列処理の準備
    ray.init(num_cpus=num_cpus, num_gpus=1, local_mode=False)
    ray_reversi = ray.put(reversi)
    current_weights = ray.put(model.get_weights())
    work_in_progresses = [
        selfplay.remote(
            weights=current_weights,
            num_mcts_simulations=num_mcts_simulations,
            dirichlet_alpha=dirichlet_alpha,
            reversi=ray_reversi,
            model_fn=model_fn,
            model_params=model_params,
        )
        for _ in range(num_cpus - 2)
    ]
    test_in_progress = testplay.remote(
        weights=current_weights,
        num_mcts_simulations=num_mcts_simulations,
        reversi=ray_reversi,
        save_dir=save_dir,
        model_fn=model_fn,
        model_params=model_params,
        num_testplay=num_testplay,
    )

    num_updates = 0
    episode = 0
    loss_m = keras.metrics.Mean()
    while episode <= num_episodes:
        # 　学習: 棋譜がある程度貯まってから実施
        if len(replay) >= min_buffer_size_for_training:
            num_iters = epochs_per_update * (len(replay) // batch_size)
            for _ in range(num_iters):
                states, mcts_policy, rewards = replay.get_minibatch(batch_size=batch_size, game=reversi)
                loss = model.train_on_batch(states, [mcts_policy, rewards])
                loss_m.update_state(loss)
                num_updates += 1
            logger.info(f"{episode=}: {num_updates=}: loss={loss_m.result():.4f}")
            # 学習後のパラメータをrayに共有
            current_weights = ray.put(model.get_weights())

        # SelfPlay: 棋譜をupdate_periodだけ集める
        for _ in tqdm(range(update_period)):
            finished, work_in_progresses = ray.wait(work_in_progresses, num_returns=1)
            replay.add_record(ray.get(finished[0]))
            episode += 1

            # 1件取ったらすぐに1件投げる
            work_in_progresses.extend(
                [
                    selfplay.remote(
                        weights=current_weights,
                        num_mcts_simulations=num_mcts_simulations,
                        dirichlet_alpha=dirichlet_alpha,
                        model_fn=model_fn,
                        model_params=model_params,
                        reversi=ray_reversi,
                    )
                ]
            )

        # TestPlay(1つ前のweightsがセットされていることに注意)
        if episode % test_period == 0:
            logger.info(f"{episode - test_period}: TEST")
            win_count, win_ratio, elapsed_time = ray.get(test_in_progress)
            logger.info(f"SCORE: {win_count}, {win_ratio:.2f}, Elapsed: {elapsed_time}")
            test_in_progress = testplay.remote(
                weights=current_weights,
                num_mcts_simulations=num_mcts_simulations,
                reversi=ray_reversi,
                save_dir=save_dir,
                model_fn=model_fn,
                model_params=model_params,
                num_testplay=num_testplay,
            )

            step = episode - test_period
            step_b = episode
            buffersize = len(replay)
            logger.info(f"{step=}: {win_count=}")
            logger.info(f"{step=}: {win_ratio=}")
            logger.info(f"{step_b=}: {buffersize=}")

        if episode % save_period == 0:
            model.save(os.path.join(save_dir, "weights.keras"))


if __name__ == "__main__":
    main(num_cpus=26, save_dir_prefix="jax")
