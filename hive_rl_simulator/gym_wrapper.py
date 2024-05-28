from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.core import ObsType

from hive_rl_simulator.agent import state_to_reward
from hive_rl_simulator.game import HiveGame, ActionStatus, AnimalType, MAX_PIECES


class GymEnvAdapter(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, game: HiveGame, render_mode: str):
        self.game = HiveGame(
            game.animal_info,
            game.last_player_idx,
            game.turn_num,
            game.board_size
        )
        max_animal_type = max([e.value for e in AnimalType])
        self.observation_space = spaces.Dict(
            {
                "enemy_table": spaces.Box(low=0, high=1, shape=(game.board_size, game.board_size), dtype=int),
                "animal_type_table": spaces.Box(low=0, high=max_animal_type, shape=(game.board_size, game.board_size), dtype=int),
                "animal_idx_table": spaces.Box(low=0, high=MAX_PIECES, shape=(game.board_size, game.board_size),
                                                dtype=int),
                "animal_types": spaces.Box(low=0, high=max_animal_type, shape=(MAX_PIECES,), dtype=int),
            }
        )
        self.action_space = spaces.Dict(
            {
                "animal_idx": spaces.Discrete(MAX_PIECES),
                "point_to": spaces.Box(shape=(2,), low=0, high=game.board_size),
            }
        )

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        super().reset(seed=seed)
        num_ants = np.sum(self.game.animal_info[0, :, 0] == AnimalType.ant.value)
        num_grasshoppers = np.sum(self.game.animal_info[0, :, 0] == AnimalType.grasshopper.value)
        num_spiders = np.sum(self.game.animal_info[0, :, 0] == AnimalType.spider.value)

        self.game = HiveGame.from_setup(
            num_ants=num_ants,
            num_spiders=num_spiders,
            num_grasshoppers=num_grasshoppers,
            board_size=self.game.board_size
        )
        return self._get_obs(), {}

    def step(self, action):
        animal_idx = action["animal_idx"]
        point_to = action["point_to"]
        player_idx = 1 if self.game.last_player_idx == 2 else 2
        action_status = self.game.apply_action(player_idx, animal_idx, point_to)
        winner_state = self.game.get_winner_state()
        reward = state_to_reward(action_status, winner_state, player_idx)
        done = action_status == ActionStatus.success
        terminated = False
        truncated = False
        info = {}
        return self._get_obs(), reward, terminated, truncated, info, done

    def _get_obs(self):
        enemy_table, animal_type_table, animal_idx_table, animal_types = self.game.get_state(self.game.last_player_idx)
        animal_types = np.pad(animal_types, (0, MAX_PIECES - len(animal_types)), mode='constant', constant_values=(0, 0))
        return {
            "enemy_table": enemy_table,
            "animal_type_table": animal_type_table,
            "animal_idx_table": animal_idx_table,
            "animal_types": animal_types
        }
