from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import torch.nn as nn

from hive_rl_simulator.agent import PlayerUNet
from hive_rl_simulator.game import MAX_PIECES


class PlayerUNetRLLibWrapper(TorchModelV2, PlayerUNet):
    def __init__(self, obs_space, act_space, num_outputs, *args, **kwargs):
        TorchModelV2.__init__(self, obs_space, act_space, num_outputs, *args, **kwargs)
        PlayerUNet.__init__(self, board_info_channels=3, max_piece_nums=MAX_PIECES, use_action_map = True)
        self.value_fn = nn.Sequential(
            nn.Conv2d(self.max_animal_nums, 512, )
            nn.Linear(512, 1)

    def forward(self, input_dict, state, seq_lens):
        model_out = self.model(input_dict["obs"].permute(0, 3, 1, 2))
        self._value_out = self.value_fn(model_out)
        return self.policy_fn(model_out), state

    def value_function(self):
        return self._value_out.flatten()



ppo_config = (
    PPOConfig()
    # .rl_module(_enable_rl_module_api=False)
    .framework('torch')
    .environment(
        env='HiveGameEnv',
        env_config={
            "dataset_name": "dataset",  # .npy files should be in ./data/dataset/
            "leverage": 2, # leverage for perpetual futures
            "episode_max_len": 168 * 2, # train episode length, 2 weeks
            "lookback_window_len": 168, # 1 week
            "train_start": [2000, 7000, 12000, 17000, 22000],
            "train_end": [6000, 11000, 16000, 21000, 26000],
            "test_start": [6000, 11000, 16000, 21000, 26000],
            "test_end": [7000, 12000, 17000, 22000, 29377-1],
            "order_size": 50, # dollars
            "initial_capital": 1000, # dollars
            "open_fee": 0.12e-2, # taker_fee
            "close_fee": 0.12e-2, # taker_fee
            "maintenance_margin_percentage": 0.012, # 1.2 percent
            "initial_random_allocated": 0, # opened initial random long/short position up to initial_random_allocated $
            "regime": "training",
            "record_stats": False, # True for backtesting
        },
        observation_space=gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(183 * 168,),
            dtype=np.float32
        ),
        action_space=gym.spaces.Discrete(4),
    )
    .training(
        lr=5e-5,
        gamma=0.995, # 1.
        grad_clip=30.,
        entropy_coeff=0.03,
        kl_coeff=0.05,
        kl_target=0.01, # not used if kl_coeff == 0.
        num_sgd_iter=10,
        use_gae=True,
        clip_param=0.3, # larger values for more policy change
        vf_clip_param=10,
        train_batch_size=15 * 8 * 168, # num_rollout_workers * num_envs_per_worker * rollout_fragment_length
        shuffle_sequences=True,
        model={
            "vf_share_layers": False,
            "custom_model": "TransformerModelAdapter",
            "custom_model_config": {
                "d_history_flat": 168 * 183,
                "num_obs_in_history": 168,
                "d_obs": 183,
                "d_time": 2, # hour, day
                "d_account": 2, # unrealized_pnl, available_balance
                "d_candlesticks_btc": 34, # TA indicators
                "d_candlesticks_ftm": 34, # TA indicators
                "d_santiment_btc_1h": 30,
                "d_santiment_btc_1d": 26,
                "d_santiment_ftm_1h": 27,
                "d_santiment_ftm_1d": 28,
                "d_obs_enc": 256,
                "num_attn_blocks": 3,
                "num_heads": 4,
                "dropout_rate": 0.1
            }
        }
    )
    .evaluation(
        evaluation_interval=1,
        evaluation_duration=8,
        evaluation_duration_unit='episodes',
        evaluation_parallel_to_training=False,
        evaluation_config={
            "explore": False,
            "env_config": {
                "regime": "evaluation",
                "record_stats": False, # True for backtesting
                "episode_max_len": 168 * 2, # validation episode length
                "lookback_window_len": 168,
            }
        },
        evaluation_num_workers=4
    )
    .rollouts(
        num_rollout_workers=15,
        num_envs_per_worker=8,
        rollout_fragment_length=168,
        batch_mode='complete_episodes',
        preprocessor_pref=None
    )
    .resources(
        num_gpus=1
    )
    .debugging(
        log_level='WARN'
    )
)

tune.run("PPO", config=ppo_config)
