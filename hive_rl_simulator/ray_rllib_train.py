import numpy as np
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import torch.nn as nn
from ray.tune import register_env

from hive_rl_simulator.agent import PlayerUNet, state_to_tensor
from hive_rl_simulator.game import MAX_PIECES, HiveGame
from hive_rl_simulator.gym_wrapper import GymEnvAdapter, GymEnvSelfPlayAdapter, Player


class PlayerUNetRLLibWrapper(TorchModelV2, PlayerUNet):
    def __init__(self, obs_space, act_space, num_outputs, *args, **kwargs):
        TorchModelV2.__init__(self, obs_space, act_space, num_outputs, *args, **kwargs)
        PlayerUNet.__init__(self, board_info_channels=3, max_piece_nums=MAX_PIECES, use_action_mask=True)
        self.value_fn = nn.Sequential(
            nn.Conv2d(self.max_animal_nums, 512, (1, 1)),
            nn.Linear(512, 1)
        )

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]
        state = state_to_tensor(
            obs["enemy_table"],
            obs["animal_type_table"],
            obs["animal_idx_table"],
            obs["animal_types"],
            obs["action_mask"],
        )
        model_out = super(PlayerUNet, self).forward(*state)
        self._value_out = self.value_fn(model_out)
        return model_out, state

    def value_function(self):
        return self._value_out.flatten()


def env_builder(
        enemy_policy: PlayerUNet,
        num_ants: int = np.random.randint(4) + 2,
        num_spiders=np.random.randint(4) + 2,
        num_grasshoppers=np.random.randint(4) + 2
):
    env = GymEnvAdapter(
        game=HiveGame.from_setup(num_ants=num_ants, num_spiders=num_spiders, num_grasshoppers=num_grasshoppers),
        render_mode="rgb_array"
    )
    enemy_player = Player(enemy_policy, obs_space=env.observation_space, action_space=env.action_space)
    return GymEnvSelfPlayAdapter(
        enemy_player,
        render_mode="rgb_array",
        game=HiveGame.from_setup(num_ants=num_ants, num_spiders=num_spiders, num_grasshoppers=num_grasshoppers),
    )


enemy_policy = PlayerUNet(board_info_channels=3, max_piece_nums=MAX_PIECES, use_action_mask=True)
our_policy = PlayerUNet(board_info_channels=3, max_piece_nums=MAX_PIECES, use_action_mask=True)
env = env_builder(enemy_policy)
player = Player(our_policy, obs_space=env.observation_space, action_space=env.action_space)
for i in range(1000):
    print(f"STEP {i}")
    action = player.get_step(env._get_obs())
    env.step(action)

register_env("my_env", env_builder)

ppo_config = (
    PPOConfig()
    # .rl_module(_enable_rl_module_api=False)
    .environment(
        env="my_env",
        env_config={
            "enemy_policy": enemy_policy,
        }
    )
    .rollouts(num_rollout_workers=4, rollout_fragment_length=128)
    .training(
        train_batch_size=512,
        lr=2e-5,
        gamma=0.99,
        lambda_=0.9,
        use_gae=True,
        clip_param=0.4,
        grad_clip=None,
        entropy_coeff=0.1,
        vf_loss_coeff=0.25,
        sgd_minibatch_size=64,
        num_sgd_iter=10,
        model={
            "custom_model": PlayerUNetRLLibWrapper,
            # "custom_model_config": {'num_outputs': None},
            # "vf_share_layers": True,
        }
    )
    .debugging(log_level="DEBUG")
    .framework(framework="torch")
    # .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
)

tune.run(
    "PPO",
    name="PPO",
    stop={"timesteps_total": 100},
    checkpoint_freq=10,
    storage_path="~/ray_results/",
)
