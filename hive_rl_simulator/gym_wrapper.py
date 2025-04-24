import logging
import math
import pickle
from functools import partial
from pathlib import Path
from typing import Any, Callable, Literal, Optional, Tuple

import gymnasium as gym
import numpy as np
import pygame
import torch
from gymnasium import spaces
from gymnasium.core import ObsType

from hive_rl_simulator.agent import state_to_reward, PlayerUNet, state_to_tensor
from hive_rl_simulator.game import HiveGame, AnimalType, MAX_PIECES, Point, WinnerState, ActionStatus

logger = logging.getLogger(__name__)


class GymEnvAdapter(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, game: HiveGame, render_mode: str = "rgb_array"):
        self.game = HiveGame(
            game.animal_info,
            game.last_player_idx,
            game.turn_num,
            game.board_size,
        )
        max_animal_type = max([e.value for e in AnimalType])
        self.observation_space = spaces.Box(low=0, high=max(max_animal_type, game.board_size), shape=)
        self.observation_space = spaces.Dict(
            {
                "enemy_table": spaces.Box(low=0, high=2, shape=(game.board_size, game.board_size), dtype=int),
                "animal_type_table": spaces.Box(low=0, high=max_animal_type, shape=(game.board_size, game.board_size),
                                                dtype=int),
                "animal_idx_table": spaces.Box(low=0, high=MAX_PIECES, shape=(game.board_size, game.board_size),
                                               dtype=int),
                "animal_types": spaces.Box(low=0, high=max_animal_type, shape=(MAX_PIECES,), dtype=int),
                "action_mask": spaces.Box(low=0, high=1, shape=(MAX_PIECES, game.board_size, game.board_size),
                                          dtype=int)
            }
        )

        # same as animal_info
        # self.observation_space = spaces.Box(low=-1, high=game.board_size-1, shape=(2, MAX_PIECES, 2))
        self.action_space = spaces.Discrete(game.board_size * game.board_size * MAX_PIECES + 1, start=-1)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None
        # The size of a single grid square in pixels
        self.pix_square_size = 100
        self.additional_axis = 4
        self.window_size = (
                int(self.game.board_size + self.additional_axis * 2 + 1)
                *
                int(self.pix_square_size * (1 + math.cos(60 / 360 * 2 * math.pi)))
        )

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
        return self._get_obs(player_idx=1), {}

    def step(self, action):
        raise ValueError("Unsupported")
        # animal_idx = action % MAX_PIECES
        # point_to = Point(action // MAX_PIECES // self.game.board_size, action // MAX_PIECES % self.game.board_size)
        # player_idx = 1 if self.game.last_player_idx == 2 else 2
        # action_status = self.game.apply_action(player_idx, animal_idx, point_to)
        # winner_state = self.game.get_winner_state()
        # reward = state_to_reward(action_status, winner_state, player_idx)
        # # done = action_status == ActionStatus.success
        # terminated = action_status == winner_state != WinnerState.no_termination
        # truncated = False
        # info = {}
        # print("RESET ")
        # return self._get_obs(), reward, terminated, truncated, info

    # def _get_obs(self, player_idx: Optional[int] = None):
    #     player_idx = player_idx or (1 if self.game.last_player_idx == 2 else 2)
    #     opposite_player_idx = 2 if player_idx == 1 else 1
    #     return np.array([
    #         self.game.animal_info[player_idx],
    #         self.game.animal_info[opposite_player_idx]
    #     ]).fillna(-1).astype(int)

    def _get_obs(self, player_idx: Optional[int] = None):
        player_idx = player_idx or (1 if self.game.last_player_idx == 2 else 2)
        enemy_table, animal_type_table, animal_idx_table, animal_types = self.game.get_state(player_idx)
        action_mask = self.game.get_action_mask(player_idx)
        action_mask = np.pad(
            action_mask,
            ((0, MAX_PIECES - len(animal_types)), (0, 0), (0, 0)),
            mode='constant',
            constant_values=(0, 0)
        )
        animal_types = np.pad(animal_types, (0, MAX_PIECES - len(animal_types)), mode='constant',
                              constant_values=(0, 0))

        return {
            "enemy_table": enemy_table,
            "animal_type_table": animal_type_table,
            "animal_idx_table": animal_idx_table,
            "animal_types": animal_types,
            "action_mask": action_mask
        }

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size),
                                                  flags=pygame.SCALED | pygame.RESIZABLE)
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))

        color_per_player = {1: "green", 2: "blue"}
        draw_hex = partial(draw_regular_polygon, vertex_count=6, surface=canvas, radius=self.pix_square_size)

        self._draw_desk(draw_hex)
        self._draw_placed_pieces_on_desk(draw_hex, color_per_player, self.game.animal_info)
        self._draw_not_placed_pieces(draw_hex, color_per_player, self.game.animal_info)

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def _draw_desk(self, draw_hex):
        for i in range(self.game.board_size + self.additional_axis * 2):
            for j in range(self.game.board_size + self.additional_axis * 2):
                if i % 2 == j % 2:
                    self._draw_hex(
                        draw_hex=partial(draw_hex, line_or_fill="line", color="black"),
                        row=j,
                        col=i
                    )

    def _draw_placed_pieces_on_desk(self, draw_hex, color_per_player, animal_info):
        points = [
            (row, col)
            for player_idx in [1, 2]
            for _, row, col in animal_info[player_idx - 1]
            if not np.isnan(col)
        ]
        even = None
        if len(points):
            row, col = points[0]
            even = row % 2 == col % 2

        for player_idx, color in color_per_player.items():
            for animal_type, row, col in animal_info[player_idx - 1]:
                if np.isnan(row):
                    continue
                self._draw_hex(
                    draw_hex=partial(draw_hex, line_or_fill="fill", color=color),
                    row=int(col) + self.additional_axis,
                    col=int(row + (1 if not even else 0)) + self.additional_axis,
                    text=self._render_animal_text(animal_type, player_idx, placed=True)
                )

    def _draw_not_placed_pieces(self, draw_hex, color_per_player, animal_info):
        col_idx_per_player = {1: 0, 2: self.game.board_size - 1 + self.additional_axis * 2}

        for player_idx, color in color_per_player.items():
            row_idx = 0
            col_idx = col_idx_per_player[player_idx]
            for animal_type, row, col in animal_info[player_idx - 1]:
                if not np.isnan(row):
                    continue

                self._draw_hex(
                    draw_hex=partial(draw_hex, line_or_fill="fill", color=color),
                    row=col_idx,
                    col=row_idx,
                    text=self._render_animal_text(animal_type, player_idx, placed=False)
                )

                row_idx += 2

    def _draw_hex(self, draw_hex: Callable, row: int, col: int, text=""):
        x_step = self.pix_square_size * (1 + math.cos(math.pi * 2 * 60 / 360))
        y_step = math.sin(math.pi * 2 * 60 / 360) * self.pix_square_size

        x = x_step * row + self.pix_square_size
        y = y_step * col + y_step
        draw_hex(position=(int(x), int(y)), text=text)

    def _render_animal_text(self, animal_type: int, player_idx: Literal[1, 2], placed: bool):
        if animal_type == AnimalType.bee.value:
            text = "Bee"
        elif animal_type == AnimalType.ant.value:
            text = "Ant"
        elif animal_type == AnimalType.spider.value:
            text = "Spider"
        elif animal_type == AnimalType.grasshopper.value:
            text = "Grasshopper"
        else:
            raise ValueError(f"Unsupported: {animal_type=}")
        text = f"{player_idx}:{text}:{'P' if placed else 'F'}"
        return text


def draw_regular_polygon(surface, color, vertex_count, radius, position, line_or_fill="line", text: str = ""):
    import pygame

    if line_or_fill == "line":
        width = int(radius / 15)
    elif line_or_fill == "fill":
        width = 0
    else:
        raise ValueError(f"Unsupported {line_or_fill=}")

    n, r = vertex_count, radius
    x, y = position
    pygame.draw.polygon(
        surface,
        color, [
            (x + r * math.cos(2 * math.pi * i / n), y + r * math.sin(2 * math.pi * i / n))
            for i in range(n)
        ],
        width=width
    )
    if text != '' and text is not None:
        font = pygame.font.SysFont('Arial', int(radius / 3), bold=True)
        surface.blit(font.render(text, False, (0, 0, 0)), (x - r / 2, y))


class Player:
    def __init__(
            self,
            player: PlayerUNet,
            obs_space: gym.spaces.space.Space,
            action_space: gym.spaces.space.Space,

    ):
        self.player = player
        self.obs_space = obs_space
        self.action_space = action_space

    def get_step(self, obs) -> int:
        with torch.no_grad():
            point_to_per_animal_logits, _ = self.player(*state_to_tensor(
                obs["enemy_table"],
                obs["animal_type_table"],
                obs["animal_idx_table"],
                obs["animal_types"],
                obs["action_mask"],
            ))
            action = np.random.choice(
                list(range(len(point_to_per_animal_logits))),
                p=np.exp(point_to_per_animal_logits.numpy()),
                size=1
            )[0]
            return action


class GymEnvSelfPlayAdapter(GymEnvAdapter):
    def __init__(self, enemy_player: Player, max_episode_steps: int = 48, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.enemy_player = enemy_player
        self.max_episode_steps = max_episode_steps

    def step(self, action):
        animal_idx, point_to = self._parse_action(action)

        action_status_1 = self.game.apply_action(1, animal_idx, point_to)
        if not action_status_1 in [ActionStatus.success, ActionStatus.no_possible_action]:
            path = Path("trace/invalid_action_status_1.pickle")
            path.unlink(missing_ok=True)
            Path("trace").mkdir(parents=True, exist_ok=True)

            with open(str(path), "wb") as f:
                pickle.dump(self, f)
            msg = f"Unsupported {action_status_1=}, PLAYER_IDX={1}, ANIMAL_IDX={animal_idx}, POINT_TO={point_to}, ACTION={action}"
            logger.error(msg)
            raise ValueError(msg)
        winner_state_1 = self.game.get_winner_state()
        logger.info(
            f"STEP={self.game.turn_num}, PLAYER_IDX={1}, ANIMAL_IDX={animal_idx}, POINT_TO={point_to}, ACTION={action}, WINNER_STATE={winner_state_1}")

        terminated = winner_state_1 != WinnerState.no_termination

        # done = action_status == ActionStatus.success

        obs_before_action = self._get_obs(player_idx=2)
        action = self.enemy_player.get_step(obs_before_action)
        animal_idx, point_to = self._parse_action(action)
        action_status_2 = self.game.apply_action(2, animal_idx, point_to)
        if not action_status_2 in [ActionStatus.success, ActionStatus.no_possible_action]:
            action = self.enemy_player.get_step(obs_before_action)
            path = Path("trace/invalid_action_status_2.pickle")
            path.unlink(missing_ok=True)
            Path("trace").mkdir(parents=True, exist_ok=True)
            msg = f"Unsupported {action_status_2=}, PLAYER_IDX={2}, ANIMAL_IDX={animal_idx}, POINT_TO={point_to}, ACTION={action}, "
            logger.error(msg)
            with open(str(path), "wb") as f:
                pickle.dump(self, f)
            raise ValueError()
        assert action_status_2 in [ActionStatus.success, ActionStatus.no_possible_action], f"Invalid {action_status_2=}"
        winner_state_2 = self.game.get_winner_state()
        logger.info(f"PLAYER_IDX={2}, ANIMAL_IDX={animal_idx}, POINT_TO={point_to}, ACTION={action}, WINNER_STATE={winner_state_2}")

        info = {}
        obs = self._get_obs(player_idx=1)
        assert ((obs["enemy_table"] >= 0) & (obs["enemy_table"] <= 2)).all()
        assert ((obs["animal_type_table"] >= 0) & (obs["animal_type_table"] <= 4)).all()
        assert ((obs["animal_idx_table"] >= 0) & (obs["animal_idx_table"] <= MAX_PIECES)).all()
        assert ((obs["action_mask"] >= 0) & (obs["action_mask"] <= 1)).all()

        truncated = False
        if obs["action_mask"].sum() == 0:
            truncated = True
        if self.game.turn_num // 2 > self.max_episode_steps:
            truncated = True

        reward = state_to_reward(
            action_status=action_status_1,
            winner_state=winner_state_2,
            local_player_idx=1,
            truncated=truncated,
            is_player_bee_free=self.game.is_player_bee_free(1),
            is_enemy_bee_locked=~self.game.is_player_bee_free(2),
            num_free_places_around_enemy_bee=self.game.num_free_places_around_bee(2),
            num_free_places_around_player_bee=self.game.num_free_places_around_bee(1),
        )
        if winner_state_1 != WinnerState.no_termination or winner_state_2 != WinnerState.no_termination:
            print(winner_state_1, winner_state_2)

        if self.render_mode == "human":
            self._render_frame()

        return obs, reward, terminated, truncated, info

    def _parse_action(self, action: int) -> Tuple[int, Point]:
        animal_idx, point_to_x, point_to_y = np.unravel_index(
            action,
            (MAX_PIECES, self.game.board_size, self.game.board_size)
        )
        point_to = Point(point_to_x, point_to_y)
        animal_idx += 1
        return animal_idx, point_to


def make_env_args(
        enemy_policy: PlayerUNet,
        num_ants: int = 3,
        num_spiders: int = 3,
        num_grasshoppers: int = 3,
        max_episode_steps: int = 24
):
    game = HiveGame.from_setup(num_ants=num_ants, num_spiders=num_spiders, num_grasshoppers=num_grasshoppers)

    env = GymEnvAdapter(game=game)

    enemy_player = Player(enemy_policy, obs_space=env.observation_space, action_space=env.action_space)
    return dict(
        enemy_player=enemy_player,
        render_mode="rgb_array",
        game=HiveGame.from_setup(num_ants=num_ants, num_spiders=num_spiders, num_grasshoppers=num_grasshoppers),
        max_episode_steps=max_episode_steps
    )
