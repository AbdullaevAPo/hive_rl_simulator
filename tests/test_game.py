import copy
from typing import List, Literal, NamedTuple

import numpy as np
import pytest
import numpy.typing as npt
from numpy.testing import assert_array_equal
from contextlib import nullcontext as does_not_raise, AbstractContextManager
from hive_rl_simulator.game import Table, is_graph_component_more_than_1, PointArray, get_available_moves_around_hive, \
    Point, is_movement_locked, move_point_in_table, compute_rescale_args, rescale_tables, AnimalType, \
    rescale_animal_info, HiveGame, ActionStatus, WinnerState


@pytest.mark.parametrize("table, expected", [
    pytest.param(
        np.array([
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0]
        ]),
        False,
        id="case with 1 graph component"
    ),
    pytest.param(
        np.array([
            [0, 0, 0, 1],
            [1, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 0],
        ]),
        True,
        id="case with 2 graph components"
    ),
    pytest.param(
        np.array([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]),
        False,
        id="empty graph case"
    )
])
def test_is_graph_component_more_than_1(table: Table, expected: bool):
    assert is_graph_component_more_than_1(table) == expected


def _some_tbl():
    return np.array([
        [0, 0, 0, 1, 0],
        [0, 0, 1, 0, 1],
        [0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0]
    ])


def _tbl_with_1_lock():
    return np.array([
        [0, 0, 1, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 1, 0, 0],
    ])


@pytest.mark.parametrize(
    "point_from, table, behaviour, expected_dest_points, expected_dest_distance",
    [
        pytest.param(
            (1, 2),
            _some_tbl(),
            does_not_raise(),
            np.array([(2, 1), (4, 1)], dtype=int),
            np.array([1, 2], dtype=int),
            id="usual case"
        ),
        pytest.param(
            (0, 3),
            _some_tbl(),
            does_not_raise(),
            np.array([], dtype=int).reshape((0, 2)),
            np.array([], dtype=int),
            id="no available moves around"
        ),
        pytest.param(
            (3, 2),
            _some_tbl(),
            does_not_raise(),
            np.array([(2, 1), (0, 1), (3, 4), (4, 3)], dtype=int),
            np.array([1, 2, 2, 1], dtype=int),
        )
    ]
)
def test_is_get_available_moves_around_hive(
        point_from: Point,
        table: Table,
        behaviour: AbstractContextManager,
        expected_dest_points: PointArray,
        expected_dest_distance: npt.NDArray[int]
):
    with behaviour:
        actual_dest_points, actual_dest_distance = get_available_moves_around_hive(
            point_from=point_from,
            table=table
        )
        actual_idx = np.lexsort((actual_dest_points[:, 0], actual_dest_points[:, 1]))
        expected_idx = np.lexsort((expected_dest_points[:, 0], expected_dest_points[:, 1]))
        assert_array_equal(
            list(actual_dest_points[actual_idx]),
            list(expected_dest_points[expected_idx])
        )
        assert_array_equal(
            list(actual_dest_distance[actual_idx]),
            list(expected_dest_distance[expected_idx])
        )


@pytest.mark.parametrize(
    "source, dest, table, expected",
    [
        pytest.param(
            (1, 1),
            (2, 2),
            _tbl_with_1_lock(),
            True,
            id="locked case"
        ),
        pytest.param(
            (1, 1),
            (2, 2),
            move_point_in_table(_tbl_with_1_lock(), src_point=Point(3, 1)),
            False,
            id="unlocked case"
        ),
        pytest.param(
            (1, 3),
            (2, 4),
            move_point_in_table(_tbl_with_1_lock(), src_point=Point(3, 1)),
            False,
            id="movement around case"
        )
    ]
)
def test_is_movement_locked(table: Table, source: Point, dest: Point, expected: bool):
    actual = is_movement_locked(
        dest=dest,
        source=source,
        table=table
    )
    assert actual == expected


def _animal_info():
    return np.array([
        [
            (AnimalType.spider.value, np.nan, np.nan),
            (AnimalType.bee.value, 3, 1),
            (AnimalType.ant.value, 4, 2),
        ],
        [
            (AnimalType.spider.value, np.nan, np.nan),
            (AnimalType.bee.value, 3, 3),
            (AnimalType.ant.value, np.nan, np.nan),
        ]
    ])


def _shifted_tbl():
    return np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 1, 0, 0],
    ])


def _expected_animal_info():
    return np.array([
        [
            (AnimalType.spider.value, np.nan, np.nan),
            (AnimalType.bee.value, 2, 1),
            (AnimalType.ant.value, 3, 2),
        ],
        [
            (AnimalType.spider.value, np.nan, np.nan),
            (AnimalType.bee.value, 2, 3),
            (AnimalType.ant.value, np.nan, np.nan),
        ]
    ])


def _expected_shifted_tbl():
    return np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0],
    ])


@pytest.mark.parametrize(
    "animal_info, tables, expected_tables, expected_animal_info",
    [
        pytest.param(
            _animal_info(),
            [_shifted_tbl() * 10, _shifted_tbl()],
            [_expected_shifted_tbl()*10, _expected_shifted_tbl()],
            _expected_animal_info(),
            id="usual case"
        ),
        pytest.param(
            _animal_info()[:][:0],
            [_shifted_tbl() * 0, _shifted_tbl() * 0],
            [_expected_shifted_tbl() * 0, _expected_shifted_tbl() * 0],
            _expected_animal_info()[:][:0],
            id="empty case"
        ),
    ]
)
def test_rescale_to_center(
    animal_info: npt.NDArray,
    tables: List[Table],
    expected_tables: List[Table],
    expected_animal_info: npt.NDArray
):
    central_point = (tables[0].shape[0] // 2, tables[0].shape[1] // 2)
    row_diff, col_diff = compute_rescale_args(tables[0], central_point=central_point)
    actual_tables = rescale_tables(*tables, row_diff=row_diff, col_diff=col_diff)
    actual_animal_info = rescale_animal_info(animal_info, row_diff, col_diff)
    for tbl1, tbl2 in zip(actual_tables, expected_tables):
        assert_array_equal(tbl1, tbl2)
    assert_array_equal(actual_animal_info, expected_animal_info)


def _simple_game():
    # np.array([
    #     [0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0],
    #     [0, 1, 0, 1, 0],
    #     [0, 0, 1, 0, 0],
    #     [0, 0, 0, 0, 0],
    # ])
    return HiveGame(
        np.array([
            [
                (AnimalType.spider.value, np.nan, np.nan),
                (AnimalType.bee.value, 2, 1),
                (AnimalType.ant.value, 3, 2),
            ],
            [
                (AnimalType.spider.value, np.nan, np.nan),
                (AnimalType.bee.value, 2, 3),
                (AnimalType.ant.value, np.nan, np.nan),
            ]
        ]),
        last_player_idx=1,
        turn_num=3,
        shape=(5, 5)
    )


def _game_with_lock():
    # np.array([
    #     [0, 0, 0, 1, 0],
    #     [0, 0, 1, 0, 1],
    #     [0, 0, 0, 0, 0],
    #     [0, 0, 1, 0, 1],
    #     [0, 0, 0, 1, 0],
    # ])
    return HiveGame(
        np.array([
            [
                (AnimalType.spider.value, np.nan, np.nan),
                (AnimalType.bee.value, 1, 2),
                (AnimalType.ant.value, 0, 3),
                (AnimalType.grasshopper.value, 1, 4),
            ],
            [
                (AnimalType.spider.value, np.nan, np.nan),
                (AnimalType.bee.value, 3, 2),
                (AnimalType.ant.value, 4, 3),
                (AnimalType.grasshopper.value, 3, 4),
            ]
        ]),
        last_player_idx=1,
        turn_num=6,
        shape=(5, 5)
    )


@pytest.mark.parametrize("func, game, point_from, player_idx, expected", [
    pytest.param(
        "get_all_possible_dest_points_for_ant",
        _simple_game(),
        Point(3, 2),
        1,
        np.array([], dtype=int).reshape((0, 2)),
        id="no movement case"
    ),
    pytest.param(
        "get_all_possible_dest_points_for_spider",
        _simple_game(),
        Point(2, 1),
        1,
        np.array([(1, 4)]),
        id="spider movement case"
    ),
    pytest.param(
        "get_all_possible_dest_points_for_ant",
        _simple_game(),
        Point(2, 1),
        1,
        np.array([(4, 1), (1, 2), (1, 4), (3, 4), (4, 3), (0, 3)]),
        id="ant movement case"
    ),
    pytest.param(
        "get_all_possible_dest_points_for_spider",
        _simple_game(),
        None,
        1,
        np.array([(0, 1), (1, 0), (3, 0), (4, 1)]),
        id="spider allocation case"
    ),
    pytest.param(
        "get_all_possible_dest_points_for_grasshopper",
        _simple_game(),
        Point(2, 1),
        1,
        np.array([(4, 3)]),
        id="grasshopper movement case"
    ),
    pytest.param(
        "get_all_possible_dest_points_for_grasshopper",
        _simple_game(),
        Point(3, 2),
        1,
        np.array([], dtype=int).reshape((0, 2)),
        id="grasshopper no movement case"
    ),
    pytest.param(
        "get_all_possible_dest_points_for_ant",
        _game_with_lock(),
        Point(0, 3),
        1,
        np.array([], dtype=int).reshape((0, 2)),
        id="ant movement case with lock",
    ),
    pytest.param(
        "get_all_possible_dest_points_for_bee",
        _simple_game(),
        Point(2, 1),
        1,
        np.array([(4, 1), (1, 2)]),
        id="bee movement case",
    ),
])
def test_get_all_possible_dest_points(
        game: HiveGame,
        func: str,
        player_idx: Literal[1, 2],
        point_from: Point,
        expected: PointArray
):
    actual = getattr(game, func)(player_idx=player_idx, point_from=point_from)
    print(actual, expected)
    assert_array_equal(np.array(sorted(actual.tolist())), np.array(sorted(expected.tolist())))


class Action(NamedTuple):
    player_idx: Literal[1, 2]
    animal_idx: int
    point_to: Point


@pytest.mark.parametrize("game, actions, expected_status_seq, expected_game", [
    pytest.param(
        HiveGame(
            np.array([
                [
                    (AnimalType.spider.value, np.nan, np.nan),
                    (AnimalType.bee.value, np.nan, np.nan),
                    (AnimalType.ant.value, np.nan, np.nan),
                    (AnimalType.grasshopper.value, np.nan, np.nan),
                ]
                for player_idx in [1, 2]
            ]),
            last_player_idx=2,
            turn_num=0,
            shape=(5, 5)
        ),
        [
            Action(1, 1, Point(2, 2)),
            Action(2, 1, Point(1, 1)),
            Action(1, 3, Point(3, 3)),
            Action(2, 2, Point(2, 0)),
            Action(1, 4, Point(1, 3)),
        ],
        [ActionStatus.success] * 4 + [ActionStatus.bee_was_not_placed_during_first_3_rounds],
        HiveGame(
            np.array([
                [
                    (AnimalType.spider.value, 2, 2),
                    (AnimalType.bee.value, np.nan, np.nan),
                    (AnimalType.ant.value, 3, 3),
                    (AnimalType.grasshopper.value, np.nan, np.nan),
                ],
                [
                    (AnimalType.spider.value, 1, 1),
                    (AnimalType.bee.value, 2, 0),
                    (AnimalType.ant.value, np.nan, np.nan),
                    (AnimalType.grasshopper.value, np.nan, np.nan),
                ]
            ]),
            last_player_idx=2,
            turn_num=4,
            shape=(5, 5)
        ),
        id="Invalid case where bee is not allocated during first 3 "
    ),
    pytest.param(
        HiveGame(
            np.array([
                [
                    (AnimalType.spider.value, np.nan, np.nan),
                    (AnimalType.bee.value, np.nan, np.nan),
                    (AnimalType.ant.value, np.nan, np.nan),
                    (AnimalType.grasshopper.value, np.nan, np.nan),
                ]
                for player_idx in [1, 2]
            ]),
            last_player_idx=2,
            turn_num=0,
            shape=(8, 8)
        ),
        [
            # allocation
            Action(1, 3, Point(4, 3)),
            Action(2, 3, Point(3, 4)),
            Action(1, 2, Point(5, 2)),
            Action(2, 2, Point(4, 5)),
            Action(1, 1, Point(6, 3)),
            Action(2, 1, Point(5, 6)),
            Action(1, 4, Point(3, 2)),
            Action(2, 4, Point(2, 5)),
            # movement
            Action(1, 4, Point(5, 4)),
            Action(2, 4, Point(6, 5)),
            Action(1, 1, Point(4, 1)),
            Action(2, 1, Point(1, 4)),
        ],
        [ActionStatus.success] * 12,
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
            shape=(8, 8)
        ),
        id="Successful game for all types"
    ),
])
def test_apply_action(game: HiveGame, actions: List[Action], expected_status_seq: List[ActionStatus], expected_game: HiveGame):
    actual_status_seq = []
    for action in actions:
        action_status = game.apply_action(*action, disable_rescale=True)
        actual_status_seq.append(action_status)
        if action_status != ActionStatus.success:
            break

    assert actual_status_seq == expected_status_seq
    assert_array_equal(expected_game.animal_info, game.animal_info)
    assert_array_equal(expected_game.player_table, game.player_table)
    assert_array_equal(expected_game.animal_idx_table, game.animal_idx_table)


@pytest.mark.parametrize(
    "game, expected_state",
    [
        pytest.param(
            HiveGame(
                np.array([
                    [
                        (AnimalType.spider.value, 1, 1),
                        (AnimalType.bee.value, 2, 2),
                        (AnimalType.ant.value, 4, 2),
                        (AnimalType.grasshopper.value, 0, 2),
                    ],
                    [
                        (AnimalType.spider.value, 3, 1),
                        (AnimalType.bee.value, 3, 3),
                        (AnimalType.ant.value, 1, 3),
                        (AnimalType.grasshopper.value, np.nan, np.nan),
                    ],
                ]),
                last_player_idx=2,
                turn_num=8,
                shape=(5, 5)
            ),
            WinnerState.player_2_win,
            id="player_1 lose case"
        ),

    ]
)
def test_get_winner_state(game: HiveGame, expected_state: WinnerState):
    winner_state = game.get_winner_state()
    assert winner_state == expected_state
