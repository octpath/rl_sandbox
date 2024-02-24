# coding: utf-8
import json
import numpy as np


class MCTS:

    def __init__(self, model, c_puct=1.0):
        self.model = model
        self.c_puct = c_puct
        self.P = {}  # Prior Probability: [state, action]
        self.N = {}  # Visit Count: [state, action]
        self.W = {}  # Total Action-Value: [state, action]
        self.next_states = {}

    def state_to_str(self, state, player):
        # 状態をhashableにエンコード
        return json.dumps([int(s) for s in state]) + str(player)

    def search(self, root_state, current_player, num_simulations, game, alpha=None):
        s = self.state_to_str(root_state, current_player)
        legal_moves = game.get_legal_moves(root_state, current_player)

        # 初出状態: リーフノードを展開する
        if s not in self.P:
            _ = self._expand(root_state, current_player, game)

        # 必要に応じてPにディリクレノイズを加える (探索)
        if alpha is not None:
            dirichlet_noise = np.random.dirichlet(alpha=[alpha] * len(legal_moves))
            for a, noise in zip(legal_moves, dirichlet_noise):
                self.P[s][a] = (1 - self.eps) * self.P[s][a] + self.eps * noise

        # シミュレーション本体
        for _ in range(num_simulations):
            # PUCTに基づいて行動を選択
            action, next_state = self._select_action_and_next_state(
                root_state, current_player, game
            )

            # 次の状態を評価してW・Nを更新する (相手番なのでvalueの正負を逆転させる)
            v = -self._evaluate(next_state, -current_player, game)
            self.W[s][action] += v
            self.N[s][action] += 1

        # 訪問回数に応じた行動の確率値計算してpiとして返す
        mcts_policy = np.array([n / sum(self.N[s]) for n in self.N[s]])
        return mcts_policy

    def _evaluate(self, state, current_player, game):
        """
        状態の価値を評価して返す
        """
        s = self.state_to_str(state, current_player)

        if game.is_done(state):
            # ゲーム終了: rewardを返す
            reward_sente, reward_gote = game.get_result(state)
            reward = reward_sente if current_player == 1 else reward_gote
            return reward
        elif s not in self.P:
            # ゲーム未終了 かつ 初出状態: リーフを展開してvalueの予測値を返す
            nn_value = self._expand(state, current_player, game)
            return nn_value
        else:
            # ゲーム未終了 かつ 既出状態: 子ノードを評価してvalueを返す
            action, next_state = self._select_action_and_next_state(
                state, current_player, game
            )
            v = -self._evaluate(next_state, -current_player, game)
            self.W[s][action] += v
            self.N[s][action] += 1
            return v

    def _expand(self, state, current_player, game):
        """
        初めて訪れた状態についてリーフノードを展開する
        """
        s = self.state_to_str(state, current_player)
        nn_state = game.get_nn_state(state, current_player)
        legal_moves = game.get_legal_moves(state, current_player)

        # 評価関数でpiとvalueの予測値を得る
        nn_policy, nn_value = self.model.predict(
            nn_state[np.newaxis, :, :, :], verbose=0
        )
        nn_policy, nn_value = nn_policy.tolist()[0], nn_value[0][0]

        # 現在の状態についてP・N・Wを初期化する
        self.P[s] = nn_policy
        self.N[s] = [0] * game.action_space_dims
        self.W[s] = [0] * game.action_space_dims

        # 次に取り得る状態を得る (リーフの展開)
        self.next_states[s] = [
            (
                game.step(state, action, current_player)[0]
                if (action in legal_moves)
                else None
            )
            for action in range(game.action_space_dims)
        ]
        return nn_value

    def _select_action_and_next_state(self, state, current_player, game):
        """
        PUCTに基づいて次の行動を選択する
        """
        s = self.state_to_str(state, current_player)
        legal_moves = game.get_legal_moves(state, current_player)

        # 与えられたstateに対して各actionのU・Qを計算
        U = [
            self.c_puct * self.P[s][a] * np.sqrt(sum(self.N[s])) / (1 + self.N[s][a])
            for a in range(game.action_space_dims)
        ]
        Q = [w / n if n != 0 else 0 for w, n in zip(self.W[s], self.N[s])]
        assert len(U) == len(Q) == game.action_space_dims

        # 合法手以外は-np.infにマスクする
        scores = [u + q for u, q in zip(U, Q)]
        scores = np.array(
            [
                score if action in legal_moves else -np.inf
                for action, score in enumerate(scores)
            ]
        )

        # スコアに基づいて行動する (同値のargmaxで同じものを取らないように乱択)
        action = np.random.choice(np.where(scores == scores.max())[0])
        next_state = self.next_states[s][action]
        return action, next_state
