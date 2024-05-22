from typing import List, Tuple, Literal

import torch
import torch.nn as nn
import numpy as np
from hive_rl_simulator.game import Table, AnimalType, HiveGame, Point, MAX_PIECES


def state_to_tensor(
        enemy_table: Table,  # current player is 1, enemy player is 2,
        animal_type_table: Table,
        animal_idx_table: Table,
        animal_types: List[AnimalType]
) -> Tuple[torch.Tensor, torch.Tensor]:
    player_animal_idx_table = np.multiply(animal_idx_table, (enemy_table == 0) | (enemy_table == 2))
    animal_idx_arr = np.arange(len(animal_types))
    is_placed = np.in1d(player_animal_idx_table.flatten(), animal_idx_arr)

    return (
        torch.Tensor(np.array((enemy_table, animal_type_table, animal_idx_table))),
        torch.Tensor(np.pad(animal_types, (0, MAX_PIECES - len(animal_types)), constant_values=0))
    )


def deserialize_tensor(
        animal_idx_proba: torch.Tensor,
        point_to_proba: torch.Tensor,
) -> Tuple[List[int], List[Point]]:
    animal_idx = torch.argmax(animal_idx_proba, dim=1).detach().numpy()
    point_to_proba = point_to_proba.detach().numpy()
    point_to = []
    for i in range(point_to_proba.shape[0]):
        point = np.unravel_index(np.argmax(point_to_proba[i].flatten()), point_to_proba[i].shape)
        point_to.append(Point(*point))
    return animal_idx, point_to


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

        self.up_conv_3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=hidden_space3, out_channels=hidden_space2, kernel_size=4, stride=1),
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
            nn.ConvTranspose2d(in_channels=hidden_space1*2, out_channels=1, kernel_size=5, stride=5),
            nn.ReLU(),
            nn.Dropout(p=0.1),
        )

        self.animal_idx_linear = nn.Sequential(
            nn.Linear(hidden_space3 + max_piece_nums, max_piece_nums),
            nn.Softmax()
        )
        self.max_animal_nums = max_piece_nums

    def forward(self, state: torch.Tensor, animal_types: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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
        animal_idx_proba = self.animal_idx_linear(animal_idx_linear_input)
        up_conv_1_res = self.up_conv_3(conv_3_res)
        up_conv_2_res = self.up_conv_2(torch.concatenate((up_conv_1_res, conv_2_res), dim=1))
        up_conv_3_res = self.up_conv_1(torch.concatenate((up_conv_2_res, conv_1_res), dim=1))
        point_to_proba = torch.nn.functional.softmax(up_conv_3_res.squeeze(), dim=0)
        return animal_idx_proba, point_to_proba

        # shared_features = self.shared_net(state.float())
        #
        # action_means = self.policy_mean_net(shared_features)
        # action_stddevs = torch.log(
        #     1 + torch.exp(self.policy_stddev_net(shared_features))
        # )
        #
        # return action_means, action_stddevs

