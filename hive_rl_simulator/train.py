import random
from typing import List, Tuple

import numpy as np
import torch
from torch.distributions import Categorical

from hive_rl_simulator.agent import PlayerUNet, deserialize_tensor, state_to_tensor
from hive_rl_simulator.game import HiveGame, Point, MAX_PIECES


class REINFORCE:
    """REINFORCE algorithm."""

    def __init__(self, game: HiveGame):
        # Hyperparameters
        self.learning_rate = 1e-4  # Learning rate for policy optimization
        self.gamma = 0.99  # Discount factor
        self.eps = 1e-6  # small number for mathematical stability

        self.probs = []  # Stores probability values of the sampled action
        self.rewards = []  # Stores the corresponding rewards

        self.net = PlayerUNet(board_info_channels=2, max_piece_nums=100)
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.learning_rate)
        self.game = game

    def sample_action(self, states: np.ndarray) -> Tuple[List[int], List[Point]]:
        states = torch.tensor(np.array(states))
        animal_idx_proba, point_to_proba = self.net(states)
        animal_idx_proba = animal_idx_proba[:, len(self.game.animal_types)]
        animal_idx = torch.argmax(animal_idx_proba, dim=1)
        point_to = point_to_proba.argmax()
        return animal_idx, point_to

    def update(self):
        """Updates the policy network's weights."""
        running_g = 0
        gs = []

        # Discounted return (backwards) - [::-1] will return an array in reverse
        for R in self.rewards[::-1]:
            running_g = R + self.gamma * running_g
            gs.insert(0, running_g)

        deltas = torch.tensor(gs)

        loss = 0
        # minimize -1 * prob * reward obtained
        for log_prob, delta in zip(self.probs, deltas):
            loss += log_prob.mean() * delta * (-1)

        # Update the policy network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Empty / zero out all episode-centric/related variables
        self.probs = []
        self.rewards = []


def train():
    seed = 42
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Reinitialize agent every seed
    net = PlayerUNet(board_info_channels=3, max_piece_nums=MAX_PIECES)

    games = [
        HiveGame(num_ants=np.random.randint(4)+2, num_spiders=np.random.randint(4)+2, num_grasshoppers=np.random.randint(4)+2)
        for _ in range(50)
    ]
    player_idx_per_game = [1] * len(games)
    for episode in range(100):
        # gymnasium v26 requires users to set seed while resetting the environment
        state_tensor, animal_types = [], []

        for i, game in enumerate(games):
            player_idx = player_idx_per_game[i]
            local_state_tensor, local_animal_types = state_to_tensor(*game.get_state(player_idx))
            state_tensor.append(local_state_tensor)
            animal_types.append(local_animal_types)
        state_tensor = torch.stack(state_tensor)
        animal_types = torch.stack(animal_types)

        animal_idx_proba, point_to_proba = net(state=state_tensor, animal_types=animal_types)

        animal_idx, point_to = deserialize_tensor(animal_idx_proba, point_to_proba)
        for i, game in enumerate(games):
            local_animal_idx = animal_idx[i]
            local_point_to = point_to[i]
            local_player_idx = 2 if game.last_player_idx == 1 else 1
            is_action_applied = game.apply_action(local_player_idx, local_animal_idx, local_point_to)
            print(is_action_applied)


if __name__ == "__main__":
    train()