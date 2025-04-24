from typing import List, Tuple, Literal, Optional, NamedTuple

import torch
import torch.nn as nn
import numpy as np
from hive_rl_simulator.game import Table, AnimalType, HiveGame, Point, MAX_PIECES, ActionStatus, WinnerState, BOARD_SIZE
import numpy.typing as npt


class PlayerInput(NamedTuple):
    enemy_table: torch.Tensor
    animal_type_table: torch.Tensor
    animal_idx_table: torch.Tensor
    animal_types: torch.Tensor
    action_mask: torch.Tensor


def state_to_tensor(
        enemy_table: Table,  # current player is 1, enemy player is 2,
        animal_type_table: Table,
        animal_idx_table: Table,
        animal_types: List[AnimalType],
        action_mask: Table
):
    return (
        torch.Tensor(enemy_table),
        torch.Tensor(animal_type_table),
        torch.Tensor(animal_idx_table),
        torch.Tensor(animal_types),
        torch.Tensor(action_mask)
    )


def deserialize_tensor(
        animal_idx_logits: torch.Tensor,
        point_to_logits: torch.Tensor,
        temperature: float = 1
) -> Tuple[List[int], List[Point], List[float]]:
    point_to_proba = torch.nn.functional.softmax(point_to_logits.squeeze() / temperature, dim=0)
    animal_idx_proba = torch.nn.functional.softmax(animal_idx_logits / temperature)
    animal_idx = []
    for probas in list(animal_idx_proba.detach().numpy()):
        local_animal_idx = np.random.choice(np.arange(len(probas)), p=probas)
        animal_idx.append(local_animal_idx)
    animal_idx = np.array(animal_idx)
    animal_idx_proba = animal_idx_proba[:, animal_idx]

    for probas in list(point_to_proba.detach().numpy()):
        point = np.unravel_index(np.random.choice(np.arange(len(probas.flatten())), p=probas.flatten()), probas.shape)

        local_animal_idx = np.random.choice(np.arange(len(probas)), p=probas)
        animal_idx.append(local_animal_idx)

    point_to_proba = point_to_proba.detach().numpy()
    point_to = []
    for i in range(point_to_proba.shape[0]):
        point = np.unravel_index(np.argmax(point_to_proba[i].flatten()), point_to_proba[i].shape)
        point_to.append(Point(*point))
    return animal_idx, point_to, animal_idx_proba


def state_to_reward(
        action_status: ActionStatus,
        winner_state: WinnerState,
        local_player_idx: Literal[1, 2],
        truncated: bool,
        is_player_bee_free: bool,
        is_enemy_bee_locked: bool,
        num_free_places_around_enemy_bee: Optional[int],
        num_free_places_around_player_bee: Optional[int]
) -> float:
    if winner_state == WinnerState.draw_game:
        reward = -100
    elif winner_state == WinnerState.player_1_win and local_player_idx == 1:
        reward = 200
    elif winner_state == WinnerState.player_1_win and local_player_idx == 2:
        reward = -200
    elif winner_state == WinnerState.player_2_win and local_player_idx == 1:
        reward = -200
    elif winner_state == WinnerState.player_2_win and local_player_idx == 2:
        reward = 200
    elif action_status == ActionStatus.success:
        reward = 0
    elif action_status in (
            ActionStatus.invalid_action_ant,
            ActionStatus.invalid_action_spider,
            ActionStatus.invalid_action_grasshopper,
            ActionStatus.invalid_action_bee
    ):
        reward = -2
    elif action_status == ActionStatus.no_possible_action:
        reward = -5
    elif action_status == ActionStatus.selected_animal_doesnt_exist:
        reward = -3
    else:
        raise ValueError(f"Unsupported {action_status=}")

    if action_status != WinnerState.no_termination or truncated:
        reward -= 10 if not is_player_bee_free else 0
        reward -= 10 if not is_enemy_bee_locked else 0
        reward -= (6 - num_free_places_around_player_bee) if num_free_places_around_player_bee is not None else 0
        reward -= num_free_places_around_enemy_bee or 0
    return reward / 200


class PlayerUNetBackBone(nn.Module):
    """Parametrized Policy Network."""

    def __init__(
            self,
            max_piece_nums: int,
            board_size: int,
            board_info_channels: int = 3,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        hidden_space1 = 64
        hidden_space2 = 128
        hidden_space3 = 256

        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=board_info_channels, out_channels=hidden_space1, kernel_size=5, stride=5),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.BatchNorm2d(hidden_space1),
        )

        self.conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_space1, out_channels=hidden_space2, kernel_size=5, stride=5),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.BatchNorm2d(hidden_space2),
        )

        self.conv_3 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_space2, out_channels=hidden_space3, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Dropout(p=0.1),
        )

        self.up_conv_3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=hidden_space3 + max_piece_nums, out_channels=hidden_space2, kernel_size=2,
                               stride=1),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.BatchNorm2d(hidden_space2),
        )

        self.up_conv_2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=hidden_space2 * 2, out_channels=hidden_space1, kernel_size=5, stride=5),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.BatchNorm2d(hidden_space1),
        )

        self.up_conv_1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=hidden_space1 * 2, out_channels=32, kernel_size=5, stride=5),
            nn.ReLU(),
            nn.Dropout(p=0.1),
        )

        self.max_animal_nums = max_piece_nums
        self.board_size = board_size

    def forward_features(
            self,
            enemy_table: torch.Tensor,
            animal_type_table: torch.Tensor,
            animal_idx_table: torch.Tensor,
            animal_types: torch.Tensor,
    ) -> torch.Tensor:
        state = torch.stack((enemy_table, animal_type_table, animal_idx_table), dim=-3).float()
        # forward pass body
        conv_1_res = self.conv_1(state)

        conv_2_res = self.conv_2(conv_1_res)
        conv_3_res = self.conv_3(conv_2_res)

        animal_idx_linear_input = torch.cat(
            (
                torch.reshape(conv_3_res, conv_3_res.shape[:2]),
                animal_types
            ),
            dim=-1
        ).unsqueeze(2).unsqueeze(2)
        up_conv_3_res = self.up_conv_3(animal_idx_linear_input)
        up_conv_2_res = self.up_conv_2(torch.concatenate((up_conv_3_res, conv_2_res), dim=1))
        up_conv_1_res = self.up_conv_1(torch.concatenate((up_conv_2_res, conv_1_res), dim=1))
        return up_conv_1_res


class PlayerUNet(PlayerUNetBackBone):
    def __init__(
            self,
            max_piece_nums: int = MAX_PIECES,
            board_size: int = BOARD_SIZE,
            board_info_channels: int = 3,
            use_action_mask=True,
            *args,
            **kwargs
    ):
        super().__init__(max_piece_nums, board_size, board_info_channels, *args, **kwargs)
        self.use_action_mask = use_action_mask

        self.policy_fn = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=max_piece_nums, kernel_size=1, stride=1),
            nn.Flatten(),
            nn.ReLU(),
            nn.Dropout(p=0.1),
        )
        self.value_fn = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=5, kernel_size=10, stride=10),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.BatchNorm2d(5),

            nn.Conv2d(in_channels=5, out_channels=32, kernel_size=5, stride=5),
            nn.ReLU(),
            nn.Dropout(p=0.1),

            nn.Flatten(),
            nn.Linear(32, 1)
        )

    def forward(
        self,
        enemy_table: torch.Tensor,
        animal_type_table: torch.Tensor,
        animal_idx_table: torch.Tensor,
        animal_types: torch.Tensor,
        action_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if len(enemy_table.shape) == 2:
            batch = False
        elif len(enemy_table.shape) == 3:
            batch = True
        else:
            raise ValueError(f"Unsupported shape: {enemy_table.shape}")
        # add batch dimension for torchrl env.rollout
        if not batch:
            enemy_table = torch.unsqueeze(enemy_table, 0)
            animal_type_table = torch.unsqueeze(animal_type_table, 0)
            animal_idx_table = torch.unsqueeze(animal_idx_table, 0)

            animal_types = torch.unsqueeze(animal_types, 0)
            action_mask = torch.unsqueeze(action_mask, 0)
        if not self.use_action_mask:
            n_batch = enemy_table.shape[0]
            n_rows = enemy_table.shape[2]
            n_cols = enemy_table.shape[3]
            max_animal_nums = self.max_animal_nums
            action_mask = torch.ones((n_batch, max_animal_nums, n_rows, n_cols), device=enemy_table.device)

        up_conv_1_res = self.forward_features(
            enemy_table=enemy_table,
            animal_type_table=animal_type_table,
            animal_idx_table=animal_idx_table,
            animal_types=animal_types
        )

        point_to_per_animal_probs = nn.functional.softmax(self.policy_fn(up_conv_1_res) , dim=1)
        value_out = self.value_fn(up_conv_1_res)

        point_to_per_animal_probs = torch.mul(point_to_per_animal_probs, action_mask.flatten(1)) + 1e-20
        point_to_per_animal_probs = point_to_per_animal_probs / torch.sum(point_to_per_animal_probs, dim=-1, keepdim=True)
        point_to_per_animal_logits = torch.log(point_to_per_animal_probs)  # torch.nn.functional.softmax(up_conv_3_res.squeeze() / temperature, dim=0)

        if not batch:
            point_to_per_animal_logits = torch.squeeze(point_to_per_animal_logits, 0)

        return point_to_per_animal_logits, value_out
