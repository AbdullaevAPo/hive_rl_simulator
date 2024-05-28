from typing import List, Tuple, Literal

import torch
import torch.nn as nn
import numpy as np
from hive_rl_simulator.game import Table, AnimalType, HiveGame, Point, MAX_PIECES, ActionStatus, WinnerState


def state_to_tensor(
        enemy_table: Table,  # current player is 1, enemy player is 2,
        animal_type_table: Table,
        animal_idx_table: Table,
        animal_types: List[AnimalType]
) -> Tuple[torch.Tensor, torch.Tensor]:
    # player_animal_idx_table = np.multiply(animal_idx_table, (enemy_table == 0) | (enemy_table == 2))
    # animal_idx_arr = np.arange(len(animal_types))
    # is_placed = np.in1d(player_animal_idx_table.flatten(), animal_idx_arr)

    return (
        torch.Tensor(np.array((enemy_table, animal_type_table, animal_idx_table))),
        torch.Tensor(np.pad(animal_types, (0, MAX_PIECES - len(animal_types)), constant_values=0))
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


def state_to_reward(action_status: ActionStatus, winner_state: WinnerState, local_player_idx: Literal[1, 2]) -> float:
    if winner_state == WinnerState.draw_game:
        reward = 0
    elif winner_state == WinnerState.player_1_win and local_player_idx == 1:
        reward = 100
    elif winner_state == WinnerState.player_1_win and local_player_idx == 2:
        reward = -100
    elif winner_state == WinnerState.player_2_win and local_player_idx == 1:
        reward = -100
    elif winner_state == WinnerState.player_2_win and local_player_idx == 2:
        reward = 100
    elif action_status != ActionStatus.success:
        reward = 1
    elif action_status in (
            ActionStatus.invalid_action_ant,
            ActionStatus.invalid_action_spider,
            ActionStatus.invalid_action_grasshopper,
            ActionStatus.invalid_action_bee
    ):
        reward = -2
    elif action_status == ActionStatus.no_possible_action:
        reward = -1
    elif action_status == ActionStatus.selected_animal_doesnt_exist:
        reward = -3
    else:
        raise ValueError(f"Unsupported {action_status=}")
    return reward


class PlayerUNet(nn.Module):
    """Parametrized Policy Network."""

    def __init__(self, board_info_channels: int, max_piece_nums: int):
        super().__init__()

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
            nn.Conv2d(in_channels=hidden_space2, out_channels=hidden_space3, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.BatchNorm2d(hidden_space3),
        )
        self.types_to_hidden_space_3 = nn.Sequential(
            nn.Linear(in_features=max_piece_nums, out_features=hidden_space3),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.BatchNorm1d(hidden_space3),
        )

        self.up_conv_3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=hidden_space3*2, out_channels=hidden_space2, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.BatchNorm2d(hidden_space2),
        )

        self.up_conv_2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=hidden_space2*2, out_channels=hidden_space1, kernel_size=5, stride=5),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.BatchNorm2d(hidden_space1),
        )

        self.up_conv_1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=hidden_space1*2, out_channels=max_piece_nums, kernel_size=5, stride=5),
            nn.ReLU(),
            nn.Dropout(p=0.1),
        )

        self.max_animal_nums = max_piece_nums

    def forward(self,
                enemy_table: torch.Tensor,
                animal_type_table: torch.Tensor,
                animal_idx_table: torch.Tensor,
                animal_types: torch.Tensor
    ) -> torch.Tensor:
        state = torch.stack((enemy_table, animal_type_table, animal_idx_table), dim=1).float()

        conv_1_res = self.conv_1(state)
        conv_2_res = self.conv_2(conv_1_res)
        conv_3_res = self.conv_3(conv_2_res)

        animal_idx_linear_input = torch.cat(
            (
                torch.reshape(conv_3_res, conv_3_res.shape[:2]),
                animal_types
            ),
            dim=-1
        )
        up_conv_1_res = self.up_conv_3(animal_idx_linear_input)
        up_conv_2_res = self.up_conv_2(torch.concatenate((up_conv_1_res, conv_2_res), dim=1))
        up_conv_3_res = self.up_conv_1(torch.concatenate((up_conv_2_res, conv_1_res), dim=1))
        point_to_per_animal_logits = up_conv_3_res  # torch.nn.functional.softmax(up_conv_3_res.squeeze() / temperature, dim=0)

        return point_to_per_animal_logits
