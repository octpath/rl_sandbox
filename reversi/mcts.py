import math
import random
import json

import numpy as np

class MCTS:

    def __init__(self, model, reversi, alpha, c_puct=1.0, epsilon=0.25):

        self.model = model
        self.alpha = alpha
        self.c_puct = c_puct
        self.eps = epsilon
        #: prior probability
        self.P = {}
        #: visit count
        self.N = {}
        #: W is total action-value and Q is mean action-value
        self.W = {}
        #: cache next states to save computation
        self.next_states = {}
        #: string is hashable
        # self.state_to_str = (
        #     lambda state, player: json.dumps(list(state)) + str(player)
        # )
        self.reversi = reversi

    def state_to_str(self, state, player):
        return json.dumps([int(s) for s in state]) + str(player)

    def search(self, root_state, current_player, num_simulations):

        s = self.state_to_str(root_state, current_player)
        legal_moves = self.reversi.get_legal_moves(root_state, current_player)

        if s not in self.P:
            _ = self._expand(root_state, current_player)

        #: Adding Dirichlet noise to the prior probabilities in the root node
        if self.alpha is not None:
            dirichlet_noise = np.random.dirichlet(alpha=[self.alpha]*len(legal_moves))
            for a, noise in zip(legal_moves, dirichlet_noise):
                self.P[s][a] = (1 - self.eps) * self.P[s][a] + self.eps * noise

        #: MCTS simulation
        for _ in range(num_simulations):

            U = [
                self.c_puct * self.P[s][a] * math.sqrt(sum(self.N[s])) / (1 + self.N[s][a])
                for a in range(self.reversi.n_action_space)
            ]
            Q = [w / n if n != 0 else 0 for w, n in zip(self.W[s], self.N[s])]

            assert len(U) == len(Q) == self.reversi.n_action_space

            scores = [u + q for u, q in zip(U, Q)]

            #: Mask invalid actions
            scores = np.array([score if action in legal_moves else -np.inf
                               for action, score in enumerate(scores)])

            #: np.argmaxでは同値maxで偏るため
            action = random.choice(np.where(scores == scores.max())[0])

            next_state = self.next_states[s][action]

            v = -self._evaluate(next_state, -current_player)

            self.W[s][action] += v

            self.N[s][action] += 1

        mcts_policy = np.array([n / sum(self.N[s]) for n in self.N[s]])

        return mcts_policy

    def _expand(self, state, current_player):

        s = self.state_to_str(state, current_player)

        nn_state = self.reversi.get_nn_state(state, current_player)
        legal_moves = self.reversi.get_legal_moves(state, current_player)

        nn_policy, nn_value = self.model.predict(nn_state[np.newaxis,:,:,:], verbose=0)

        nn_policy, nn_value = nn_policy.tolist()[0], nn_value[0][0]

        self.P[s] = nn_policy
        self.N[s] = [0] * self.reversi.n_action_space
        self.W[s] = [0] * self.reversi.n_action_space

        #: cache valid actions and next state to save computation
        self.next_states[s] = [
            self.reversi.step(state, action, current_player)[0]
            if (action in legal_moves) else None
            for action in range(self.reversi.n_action_space)
        ]

        return nn_value

    def _evaluate(self, state, current_player):

        s = self.state_to_str(state, current_player)

        if self.reversi.is_done(state):
            #: ゲーム終了
            reward_first, reward_second = self.reversi.get_result(state)
            reward = reward_first if current_player == 1 else reward_second
            return reward

        elif s not in self.P:
            #: ゲーム終了していないリーフノードの場合は展開
            nn_value = self._expand(state, current_player)
            return nn_value

        else:
            #: 子ノードをevaluate
            U = [self.c_puct * self.P[s][a] * math.sqrt(sum(self.N[s])) / (1 + self.N[s][a])
                 for a in range(self.reversi.n_action_space)]
            Q = [q / n if n != 0 else q for q, n in zip(self.W[s], self.N[s])]

            assert len(U) == len(Q) == self.reversi.n_action_space

            leval_moves = self.reversi.get_legal_moves(state, current_player)

            scores = [u + q for u, q in zip(U, Q)]
            scores = np.array([score if action in leval_moves else -np.inf
                               for action, score in enumerate(scores)])

            best_action = random.choice(np.where(scores == scores.max())[0])

            next_state = self.next_states[s][best_action]

            v = -self._evaluate(next_state, -current_player)

            self.W[s][best_action] += v
            self.N[s][best_action] += 1

            return v
