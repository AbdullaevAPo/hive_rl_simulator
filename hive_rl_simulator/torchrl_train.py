import random
from collections import defaultdict
from typing import List, Tuple

import numpy as np
import torch
from tensordict.nn import TensorDictModule
from torchrl.collectors import SyncDataCollector
from torchrl.data import ReplayBuffer, LazyTensorStorage, SamplerWithoutReplacement
from torchrl.envs import GymWrapper
from torchrl.envs.utils import ExplorationType, set_exploration_type, check_env_specs
from torchrl.modules import ProbabilisticActor, ValueOperator, OneHotCategorical
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from tqdm import tqdm

from hive_rl_simulator.agent import PlayerUNet
from hive_rl_simulator.game import HiveGame, Point, MAX_PIECES
from hive_rl_simulator.gym_wrapper import GymEnvAdapter


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
    env = GymWrapper(
        GymEnvAdapter(
            game=HiveGame.from_setup(
                num_ants=np.random.randint(4)+2,
                num_spiders=np.random.randint(4)+2,
                num_grasshoppers=np.random.randint(4)+2
            ),
            render_mode="human"
        )
    )
    check_env_specs(env)
    env.rollout(10)

    policy_module = TensorDictModule(
        net,
        in_keys=["enemy_table", "animal_type_table", "animal_idx_table", "animal_types"],
        out_keys=["logits"]
    )
    print("Running policy:", policy_module(env.reset().expand(5)))
    print("Running policy:", policy_module(env.reset()))

    policy_module = ProbabilisticActor(
        module=policy_module,
        in_keys=["logits"],
        out_keys=["point_to_per_animal"],
        distribution_class=OneHotCategorical,
        return_log_prob=True,
    )
    value_module = ValueOperator(
        module=net,
        in_keys=["enemy_table", "animal_type_table", "animal_idx_table", "animal_types"],
    )
    print("Running policy:", policy_module(env.reset().expand(5)))
    print("Running value:", value_module(env.reset().expand(5)))

    frames_per_batch = 100
    # For a complete training, bring the number of frames up to 1M
    total_frames = 10_000
    device = (
        torch.device(0)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    max_grad_norm = 1.0

    collector = SyncDataCollector(
        env,
        policy_module,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        split_trajs=False,
        device=device,
    )
    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(max_size=frames_per_batch),
        sampler=SamplerWithoutReplacement(),
    )
    sub_batch_size = 64  # cardinality of the sub-samples gathered from the current data in the inner loop
    num_epochs = 10  # optimisation steps per batch of data collected
    clip_epsilon = (
        0.2  # clip value for PPO loss: see the equation in the intro for more context.
    )
    gamma = 0.99
    lmbda = 0.95
    entropy_eps = 1e-4

    advantage_module = GAE(
        gamma=gamma, lmbda=lmbda, value_network=value_module, average_gae=True
    )

    loss_module = ClipPPOLoss(
        actor_network=policy_module,
        critic_network=value_module,
        clip_epsilon=clip_epsilon,
        entropy_bonus=bool(entropy_eps),
        entropy_coef=entropy_eps,
        # these keys match by default but we set this for completeness
        critic_coef=1.0,
        loss_critic_type="smooth_l1",
    )

    lr = 3e-4
    max_grad_norm = 1.0

    optim = torch.optim.Adam(loss_module.parameters(), lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, total_frames // frames_per_batch, 0.0
    )

    logs = defaultdict(list)
    pbar = tqdm(total=total_frames)
    eval_str = ""

    # We iterate over the collector until it reaches the total number of frames it was
    # designed to collect:
    for i, tensordict_data in enumerate(collector):
        # we now have a batch of data to work with. Let's learn something from it.
        for _ in range(num_epochs):
            # We'll need an "advantage" signal to make PPO work.
            # We re-compute it at each epoch as its value depends on the value
            # network which is updated in the inner loop.
            advantage_module(tensordict_data)
            data_view = tensordict_data.reshape(-1)
            replay_buffer.extend(data_view.cpu())
            for _ in range(frames_per_batch // sub_batch_size):
                subdata = replay_buffer.sample(sub_batch_size)
                loss_vals = loss_module(subdata.to(device))
                loss_value = (
                        loss_vals["loss_objective"]
                        + loss_vals["loss_critic"]
                        + loss_vals["loss_entropy"]
                )

                # Optimization: backward, grad clipping and optimization step
                loss_value.backward()
                # this is not strictly mandatory but it's good practice to keep
                # your gradient norm bounded
                torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
                optim.step()
                optim.zero_grad()

        logs["reward"].append(tensordict_data["next", "reward"].mean().item())
        pbar.update(tensordict_data.numel())
        cum_reward_str = (
            f"average reward={logs['reward'][-1]: 4.4f} (init={logs['reward'][0]: 4.4f})"
        )
        logs["step_count"].append(tensordict_data["step_count"].max().item())
        stepcount_str = f"step count (max): {logs['step_count'][-1]}"
        logs["lr"].append(optim.param_groups[0]["lr"])
        lr_str = f"lr policy: {logs['lr'][-1]: 4.4f}"
        if i % 10 == 0:
            # We evaluate the policy once every 10 batches of data.
            # Evaluation is rather simple: execute the policy without exploration
            # (take the expected value of the action distribution) for a given
            # number of steps (1000, which is our ``env`` horizon).
            # The ``rollout`` method of the ``env`` can take a policy as argument:
            # it will then execute this policy at each step.
            with set_exploration_type(ExplorationType.MEAN), torch.no_grad():
                # execute a rollout with the trained policy
                eval_rollout = env.rollout(1000, policy_module)
                logs["eval reward"].append(eval_rollout["next", "reward"].mean().item())
                logs["eval reward (sum)"].append(
                    eval_rollout["next", "reward"].sum().item()
                )
                logs["eval step_count"].append(eval_rollout["step_count"].max().item())
                eval_str = (
                    f"eval cumulative reward: {logs['eval reward (sum)'][-1]: 4.4f} "
                    f"(init: {logs['eval reward (sum)'][0]: 4.4f}), "
                    f"eval step-count: {logs['eval step_count'][-1]}"
                )
                del eval_rollout
        pbar.set_description(", ".join([eval_str, cum_reward_str, stepcount_str, lr_str]))

        # We're also using a learning rate scheduler. Like the gradient clipping,
        # this is a nice-to-have but nothing necessary for PPO to work.
        scheduler.step()
    #
    # games = [
    #     HiveGame.from_setup(
    #         num_ants=np.random.randint(4)+2,
    #         num_spiders=np.random.randint(4)+2,
    #         num_grasshoppers=np.random.randint(4)+2
    #     )
    #     for _ in range(50)
    # ]
    #
    # player_idx_per_game = [1] * len(games)
    # for episode in range(100):
    #     # gymnasium v26 requires users to set seed while resetting the environment
    #     state_tensor, animal_types = [], []
    #
    #     for i, game in enumerate(games):
    #         player_idx = player_idx_per_game[i]
    #         local_state_tensor, local_animal_types = state_to_tensor(*game.get_state(player_idx))
    #         state_tensor.append(local_state_tensor)
    #         animal_types.append(local_animal_types)
    #     state_tensor = torch.stack(state_tensor)
    #     animal_types = torch.stack(animal_types)
    #
    #
    #     point_to_per_animal_logits = net(state=state_tensor, animal_types=animal_types)
    #
    #     animal_idx, point_to, proba = deserialize_tensor(point_to_per_animal_logits)
    #     for i, game in enumerate(games):
    #         local_animal_idx = animal_idx[i]
    #         local_point_to = point_to[i]
    #         local_player_idx = 2 if game.last_player_idx == 1 else 1
    #         action_status = game.apply_action(local_player_idx, local_animal_idx, local_point_to)
    #         winner_state = game.get_winner_state()
    #         reward = _state_to_reward(action_status, winner_state, local_player_idx)
    #         print(reward)



if __name__ == "__main__":
    train()
