import math
from functools import partial
from math import cos
from typing import Any, Optional, Callable, Literal

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces
from gymnasium.core import ObsType, RenderFrame
from pygame import Surface

from hive_rl_simulator.agent import state_to_reward
from hive_rl_simulator.game import HiveGame, ActionStatus, AnimalType, MAX_PIECES, Point, WinnerState


class GymEnvAdapter(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, game: HiveGame, render_mode: str):
        self.game = HiveGame(
            game.animal_info,
            game.last_player_idx,
            game.turn_num,
            game.board_size,
        )
        max_animal_type = max([e.value for e in AnimalType])
        self.observation_space = spaces.Dict(
            {
                "enemy_table": spaces.Box(low=0, high=1, shape=(game.board_size, game.board_size), dtype=int),
                "animal_type_table": spaces.Box(low=0, high=max_animal_type, shape=(game.board_size, game.board_size), dtype=int),
                "animal_idx_table": spaces.Box(low=0, high=MAX_PIECES, shape=(game.board_size, game.board_size), dtype=int),
                "animal_types": spaces.Box(low=0, high=max_animal_type, shape=(MAX_PIECES,), dtype=int),
                "action_map": spaces.Box(low=0, high=1, shape=(MAX_PIECES, game.board_size, game.board_size), dtype=int)
            }
        )
        self.action_space = spaces.Discrete(game.board_size * game.board_size * MAX_PIECES)

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
            int(self.pix_square_size * (1+math.cos(60/360*2*math.pi)))
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
        return self._get_obs(), {}

    def step(self, action):
        animal_idx = action % MAX_PIECES
        point_to = Point(action // MAX_PIECES // self.game.board_size, action // MAX_PIECES % self.game.board_size)
        player_idx = 1 if self.game.last_player_idx == 2 else 2
        action_status = self.game.apply_action(player_idx, animal_idx, point_to)
        winner_state = self.game.get_winner_state()
        reward = state_to_reward(action_status, winner_state, player_idx)
        # done = action_status == ActionStatus.success
        terminated = action_status == winner_state != WinnerState.no_termination
        truncated = False
        info = {}
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        enemy_table, animal_type_table, animal_idx_table, animal_types = self.game.get_state(self.game.last_player_idx)
        animal_types = np.pad(animal_types, (0, MAX_PIECES - len(animal_types)), mode='constant', constant_values=(0, 0))

        action_map = self.game.get_action_map(self.game.last_player_idx)

        return {
            "enemy_table": enemy_table,
            "animal_type_table": animal_type_table,
            "animal_idx_table": animal_idx_table,
            "animal_types": animal_types,
            "action_map": action_map
        }

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):

        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size), flags=pygame.SCALED | pygame.RESIZABLE)
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
        x_step = self.pix_square_size * (1 + math.cos(math.pi*2*60/360))
        y_step = math.sin(math.pi*2*60/360) * self.pix_square_size

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


def draw_regular_polygon(surface: Surface, color, vertex_count, radius, position, line_or_fill="line", text: str = ""):
    if line_or_fill == "line":
        width = int(radius/15)
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
        font = pygame.font.SysFont('Arial', int(radius/3), bold=True)
        surface.blit(font.render(text, False, (0, 0, 0)), (x-r/2, y))

