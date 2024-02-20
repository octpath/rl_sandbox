# code: utf-8
import functools
import os

from PIL import Image, ImageDraw
import numpy as np


import json


class Reversi():

    def __init__(self, n_rows=6, n_cols=6):
        self.n_cols = n_cols
        self.n_rows = n_rows
        self.n_action_space = self.n_cols * self.n_rows + 1
        self.action_noop = self.n_action_space - 1
        self.sente = 1
        self.gote = -1

    def initialize(self):
        state = np.zeros((self.n_cols * self.n_rows,), dtype=np.int32)
        state[self._convert_index_2d_to_1d(self.n_rows//2 - 1, self.n_cols//2 - 1)] = self.sente
        state[self._convert_index_2d_to_1d(self.n_rows//2, self.n_cols//2)] = self.sente
        state[self._convert_index_2d_to_1d(self.n_rows//2 - 1, self.n_cols//2)] = self.gote
        state[self._convert_index_2d_to_1d(self.n_rows//2, self.n_cols//2 - 1)] = self.gote
        return state

    def _convert_index_1d_to_2d(self, idx):
        row = idx // self.n_cols
        col = idx % self.n_cols
        return row, col
    
    def _convert_index_2d_to_1d(self, row, col):
        return self.n_cols * row + col

    def get_state_2d(self, state):
        return state.reshape((self.n_rows, self.n_cols))

    def get_nn_state(self, state, current_player):
        state_2d = self.get_state_2d(state)
        return np.stack([
            (state_2d == current_player).astype(np.float32),
            (state_2d == -current_player).astype(np.float32),
        ], axis=-1)

    @functools.lru_cache(maxsize=2048)
    def get_relevant_segments(self, idx):
        _, col = self._convert_index_1d_to_2d(idx)
        return (
            np.arange(idx, -1, -self.n_cols)[1:].astype(np.int32), # top
            np.arange(idx, -1, -self.n_cols+1)[1:self.n_cols-col].astype(np.int32), # top_right
            np.arange(idx+1, idx+self.n_cols-col).astype(np.int32), # right
            np.arange(idx, self.n_rows*self.n_cols, self.n_cols+1)[1:self.n_cols-col].astype(np.int32), # bottom_right
            np.arange(idx, self.n_rows*self.n_cols, self.n_cols)[1:].astype(np.int32), # bottom
            np.arange(idx, self.n_rows*self.n_cols, self.n_cols-1)[1:col+1].astype(np.int32), # bottom_left
            np.arange(idx-col, idx, 1).astype(np.int32)[::-1], # left
            np.arange(idx, -1, -self.n_cols-1)[1:col+1].astype(np.int32), # top_left
        )

    def is_legal_move(self, state, idx, player):

        if state[idx] != 0:
            return False
        
        relevant_segments = self.get_relevant_segments(idx)
        for seg in relevant_segments:
            discs = [state[i_] for i_ in seg]
            if (player in discs) and (-player in discs):
                btw_discs = np.array(discs[:discs.index(player)])
                if (len(btw_discs) > 0) and np.all(btw_discs==-player):
                    return True
        return False

    # def get_legal_moves(self, state, player):
    #     legal_moves = [idx for idx in range(self.n_rows*self.n_cols) if self.is_legal_move(state, idx, player)]
    #     if len(legal_moves) == 0:
    #         legal_moves = [self.action_noop]
    #     return legal_moves

    @functools.lru_cache(maxsize=4096)
    def _get_legal_moves(self, state_str, player):
        state = np.array(json.loads(state_str)).astype(np.int32)
        legal_moves = [idx for idx in range(self.n_rows*self.n_cols) if self.is_legal_move(state, idx, player)]
        if len(legal_moves) == 0:
            legal_moves = [self.action_noop]        
        return legal_moves

    def get_legal_moves(self, state, player):
        state_str = json.dumps([int(s) for s in state])
        return self._get_legal_moves(state_str, player)


    def get_num_opposite_discs(self, state, idx, player):
        # ひっくり返せる枚数をカウントする
        if state[idx] != 0:
            return 0

        rtn = 0
        relevant_segments = self.get_relevant_segments(idx)
        for seg in relevant_segments:
            discs = [state[i_] for i_ in seg]
            if (player in discs) and (-player in discs):
                btw_discs = np.array(discs[:discs.index(player)])
                rtn += len(btw_discs)
        return rtn

    def get_greedy_action(self, state, player, epsilon=0.):
        legal_moves = self.get_legal_moves(state, player)
        if np.random.random() > epsilon:
            if legal_moves == [self.action_noop]:
                return self.action_noop
            best_action_idx = np.argmax([self.get_num_opposite_discs(state, idx, player) for idx in legal_moves])
            z = legal_moves[best_action_idx]
            return z
        else:
            x = np.random.choice(legal_moves)
            return x

    def is_done(self, state):
        if (state==0).sum() == 0:
            # 全部埋まっている
            return True

        sente_moves = self.get_legal_moves(state, self.sente)
        gote_moves = self.get_legal_moves(state, self.gote)
        if (sente_moves == [self.action_noop]) and (sente_moves == gote_moves):
            # お互いにパスしかできない
            return True

        return False
    
    def step(self, state, action, player):
        next_state = state.copy()
        if action == self.action_noop:
            pass
        else:
            relevant_segments = self.get_relevant_segments(action)
            for seg in relevant_segments:
                discs = [state[i_] for i_ in seg]
                if (player in discs) and (-player in discs):
                    found_idx = discs.index(player)
                    btw_discs = np.array(discs[:found_idx])
                    if (len(btw_discs) > 0) and np.all(btw_discs==-player):
                        for idx_ in seg[:found_idx]:
                            next_state[idx_] = player
            next_state[action] = player
        return next_state, self.is_done(next_state)

    def count_discs(self, state):
        num_sente = (state==self.sente).sum()
        num_gote = (state==self.gote).sum()
        return num_sente, num_gote

    def get_result(self, state):
        num_sente, num_gote = self.count_discs(state)
        if num_sente == num_gote:
            return 0, 0
        elif num_sente > num_gote:
            return 1, -1
        else:
            return -1, 1
    
    def save_img(self, state, save_dir, fname, comment):
        os.makedirs(save_dir, exist_ok=True)

        height = 50 * self.n_rows
        width = 50 * self.n_cols

        img = Image.new('RGB', (width, height+30), (47, 79, 79))
        draw = ImageDraw.Draw(img)

        draw.rectangle((0, height, width, height+30), fill="black")
        draw.text((10, height+15), comment)


        for i in range(self.n_cols+1):
            draw.line((0, i*50, width, i*50), fill=(10, 10, 10), width=1)
        for i in range(self.n_rows+1):
            draw.line((i*50, 0, i*50, height), fill=(10, 10, 10), width=1)

        for i in range(self.n_rows * self.n_cols):
            v = state[i]
            row, col = i // self.n_rows, i % self.n_cols
            cy, cx = (50 * row + 5, 50 * col + 5)
            if v == 1:
                draw.ellipse((cx, cy, cx+40, cy+40), fill="black")
            elif v == -1:
                draw.ellipse((cx, cy, cx+40, cy+40), fill="white")

        save_path = os.path.join(save_dir, fname)
        img.save(save_path, quality=95)


# if __name__ == '__main__':
#     state = get_initial_state()
#     save_img(state, "img", "test_1.png", "ALphazero 1: 22 vs 12")
