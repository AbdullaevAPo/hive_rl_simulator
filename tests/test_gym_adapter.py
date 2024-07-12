import numpy as np
import pytest
from PIL import Image as im

from hive_rl_simulator.game import HiveGame, AnimalType
from hive_rl_simulator.gym_wrapper import GymEnvAdapter
from tests.test_game import _simple_game


@pytest.mark.parametrize(
    "game",
    [
        pytest.param(
            _simple_game(),
            id="usual case"
        ),
        pytest.param(
            HiveGame(
                np.array([
                    [
                        (AnimalType.spider.value, 4, 1),
                        (AnimalType.bee.value, 5, 2),
                        (AnimalType.ant.value, 4, 3),
                        (AnimalType.grasshopper.value, 5, 4),
                    ],
                    [
                        (AnimalType.spider.value, 1, 4),
                        (AnimalType.bee.value, 4, 5),
                        (AnimalType.ant.value, 3, 4),
                        (AnimalType.grasshopper.value, 6, 5),
                    ]
                ]),
                last_player_idx=1,
                turn_num=8,
                board_size=50
            ),
            id="usual case 2"
        ),
    ]
)
def test_draw(game: HiveGame):
    game.rescale()
    img = GymEnvAdapter(game, render_mode="rgb_array").render()

    im.fromarray(img).save("asd.png")
