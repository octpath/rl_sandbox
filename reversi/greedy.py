# coding: utf-8
from reversi import Reversi

if __name__ == '__main__':

    while True:
        rvrs = Reversi(n_rows=6, n_cols=6)

        state = rvrs.initialize()

        record = []
        current_player = 1
        done = False


        while not done:
            state_2d = rvrs.get_state_2d(state)
            print('-'*100)
            print(state_2d)

            action = rvrs.get_greedy_action(state, current_player, epsilon=0.1)
            next_state, done = rvrs.step(state, action, current_player)

            state = next_state
            current_player = -current_player

        state_2d = rvrs.get_state_2d(state)
        print('-'*100)
        print(state_2d)

        reward_sente, reward_gote = rvrs.get_result(state)
        print('-'*100)
        print(reward_sente, reward_gote)
        print((state==1).sum(), (state==-1).sum())

        if (state==0).sum() > 0:
            break
