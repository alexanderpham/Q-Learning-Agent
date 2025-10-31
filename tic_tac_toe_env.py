from gymnasium import Env
import numpy as np
from typing import Tuple


class TicTacToe(Env):
    """
    A tic-tac-toe game
    You shouldn't need to look at this code much. Focus on the tic_tac_toe_solution script
    """

    def __init__(self):
        self.state = np.zeros(9, dtype=int)
        self._player_turn = 1

    def reset(self, *args) -> Tuple[str, dict]:
        """
        resets the game
        :return: current state, empty dict
        """
        self.state = np.zeros(9, dtype=int)
        self._player_turn = 1
        return self._get_obs(), dict()

    def get_player_turn(self) -> int:
        """
        :return: 1 if it's the X player's turn, 0 if it's the O player's turn
        """
        return (self._player_turn + 1) // 2

    def render(self) -> None:
        """
        prints an ascii representation of the tic-tac-toe board
        """
        print(self._get_state_string())

    def get_available_actions(self, obs=None) -> list:
        """
        gets currently available actions
        :param obs: a board state (as returned by step and reset). defaults to current state
        :return: list of legal numeric actions
        """
        obs = np.array(obs or self.state)
        return list(np.where(obs == 0)[0])

    def step(self, action) -> Tuple[str, float, bool, bool, None]:
        """
        execute an action in the game
        :param action: numeric action to perform (use get_actions() to get a list of currently available actions
        :return: observation, reward, terminated, False, None
        """
        self.state[action] = self._player_turn
        self._player_turn *= -1

        game_won = self._game_won()
        draw = not any(self.state == 0)
        terminated = bool(game_won) or draw

        reward = 0
        if game_won:
            reward = 1

        return self._get_obs(), reward, terminated, False, None

    def _get_state_string(self, obs=None) -> str:
        """
        returns an ascii printout of the tic-tac-toe board
        :param obs: a board state (as returned by step and reset). defaults to current state
        :return: string
        """
        obs = self.state if obs is None else obs
        string = f'{obs[0]}│{obs[1]}│{obs[2]}\n' \
                 f'─┼─┼─\n{obs[3]}│{obs[4]}│{obs[5]}\n' \
                 f'─┼─┼─\n{obs[6]}│{obs[7]}│{obs[8]}\n'
        string = string.replace('0', ' ').replace('-1', 'O').replace('1', 'X')
        return string

    def _get_obs(self) -> str:
        return self._get_state_string(self.state)

    def _game_won(self, obs=None):
        obs = np.array(obs or self.state)
        grid = obs.reshape((3, 3))
        sums = np.concatenate((
            grid.sum(axis=0),
            grid.sum(axis=1),
            [grid.diagonal().sum()],
            [np.fliplr(grid).diagonal().sum()]
        ))
        if any(sums == 3):
            return 1
        if any(sums == -3):
            return -1
        return 0
