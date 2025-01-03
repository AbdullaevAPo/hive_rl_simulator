import enum
import pickle
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Union, NamedTuple, TypeAlias, Tuple, Optional, List, Literal

import cachetools
import numpy.typing as npt
import numpy as np
from numba import jit
from numpy.testing import assert_array_equal

from hive_rl_simulator import utils

MAX_PIECES = 20
BOARD_SIZE = 50
CENTRAL_POINT = (BOARD_SIZE // 2, BOARD_SIZE // 2)


class AnimalType(enum.Enum):
    bee = 1
    ant = 2
    spider = 3
    grasshopper = 4


class Point(NamedTuple):
    row: int
    col: int


PointArray: TypeAlias = npt.NDArray[int]
Table: TypeAlias = npt.NDArray[int]


@jit(nopython=True)
def get_close_coords_vec(points: PointArray, board_size: int = BOARD_SIZE) -> PointArray:
    arr = (
        points + np.array([1, 1]),
        points + np.array([-1, 1]),
        points + np.array([2, 0]),
        points + np.array([-2, 0]),
        points + np.array([1, -1]),
        points + np.array([-1, -1]),
    )
    res = np.vstack(arr)
    mask = (
        (res[:, 0] >= 0) &
        (res[:, 0] < board_size) &
        (res[:, 1] >= 0) &
        (res[:, 1] < board_size)
    )
    res = res[mask, :]
    return res


def get_close_coords_vec_2(point: Union[Point, PointArray], board_size: int = BOARD_SIZE, keep_duplicates: bool = False) -> PointArray:
    row = point[:, 0] if isinstance(point, np.ndarray) else np.array([point[0]])
    col = point[:, 1] if isinstance(point, np.ndarray) else np.array([point[1]])

    res = np.hstack([
        (row + 1, col + 1),
        (row - 1, col + 1),
        (row + 2, col),
        (row - 2, col),
        (row + 1, col - 1),
        (row - 1, col - 1),
    ]).T
    if not keep_duplicates:
        res = np.unique(res, axis=0)
    res = res[(res[:, 0] >= 0) & (res[:, 0] < board_size) & (res[:, 1] >= 0) & (res[:, 1] < board_size)]
    return res


@cachetools.cached(cachetools.LRUCache(maxsize=10000))
def get_close_coords_point(point: Point, board_size: int = BOARD_SIZE) -> PointArray:
    return get_close_coords_vec(np.expand_dims(np.array(point), axis=0), board_size)


def point_to_binary_table(point: Union[Point, PointArray], shape: Tuple[int, int]) -> Table:
    table = np.zeros(shape)
    if isinstance(point, PointArray):
        table[point[:, 0], point[:, 1]] = 1
    else:
        table[point[0], point[1]] = 1
    return table


def safe_x_y(tbl: Table, coord: Point) -> Optional[int]:
    if 0 <= coord[0] < tbl.shape[0] and 0 <= coord[1] < tbl.shape[1]:
        return tbl[coord[0], coord[1]]
    return None


def are_points_valid(x: PointArray, table: Table):
    valid_mask = (x[:, 0] >= 0) & (x[:, 0] < table.shape[0]) & (x[:, 1] >= 0) & (x[:, 1] < table.shape[1])
    filled_mask = np.zeros(len(valid_mask))
    filled_mask[valid_mask] = table[x[:, 0][valid_mask], x[:, 1][valid_mask]]
    return valid_mask & (filled_mask != 0)


# @jit(nopython=True)
def get_movenent_locked_fast(dest: PointArray, source: PointArray, table: Table) -> npt.NDArray[bool]:
    shifts = np.array([
        ((1, 1), (2, 0), (-1, 1)),
        ((1, -1), (2, 0), (-1, -1)),
        ((-1, -1), (-2, 0), (1, -1)),
        ((-1, 1), (-2, 0), (1, 1)),
        ((2, 0), (1, -1), (1, 1)),
        ((-2, 0), (-1, -1), (-1, 1))
    ])

    cnt_points = dest.shape[0]
    cnt_shifts = shifts.shape[0]
    dest = np.repeat(dest, cnt_shifts, axis=0)
    source = np.repeat(source, cnt_shifts, axis=0)
    shifts = np.tile(shifts, (cnt_points, 1, 1))

    matched_shifts = (dest[:, 0] == source[:, 0] + shifts[:, 0, 0]) & (dest[:, 1] == source[:, 1] + shifts[:, 0, 1])
    left_shifts = np.vstack((source[:, 0] + shifts[:, 1, 0], source[:, 1] + shifts[:, 1, 1])).T
    right_shifts = np.vstack((source[:, 0] + shifts[:, 2, 0], source[:, 1] + shifts[:, 2, 1])).T

    valid_left_shifts_mask = are_points_valid(left_shifts, table)
    valid_right_shifts_mask = are_points_valid(right_shifts, table)
    mask = valid_left_shifts_mask & valid_right_shifts_mask & matched_shifts
    mask = (mask.reshape((cnt_points, cnt_shifts)).astype(int).sum(axis=1) != 0)

    return mask


def is_movement_locked(dest: Point, source: Point, table: Table) -> bool:
    for shift, left_shift, right_shift in [
        ((1, 1), (2, 0), (-1, 1)),
        ((1, -1), (2, 0), (-1, -1)),
        ((-1, -1), (-2, 0), (1, -1)),
        ((-1, 1), (-2, 0), (1, 1)),
        ((2, 0), (1, -1), (1, 1)),
        ((-2, 0), (-1, -1), (-1, 1))
    ]:
        if (
            dest[0] == source[0] + shift[0] and dest[1] == source[1] + shift[1]
            and
            (safe_x_y(table, Point(source[0] + left_shift[0], source[1] + left_shift[1])) or 0) != 0
            and
            (safe_x_y(table, Point(source[0] + right_shift[0], source[1] + right_shift[1])) or 0) != 0
        ):
            return True

    return False


def move_point_in_table(
        table: npt.NDArray[int],
        src_point: Optional[Point] = None,
        dst_point: Optional[Point] = None,
        value: Optional = None,
        default_value: int = 0
) -> npt.NDArray[int]:
    table = np.copy(table)
    if dst_point is not None:
        table[dst_point[0], dst_point[1]] = value or table[src_point.row, src_point.col]
    if src_point is not None:
        table[src_point[0], src_point[1]] = default_value
    return table


def is_graph_component_more_than_1(table: Table) -> bool:
    visited_tbl = np.zeros(table.shape)
    start_point = point_where(table != 0)
    if len(start_point) == 0:
        return False
    start_point = start_point[0]
    visited_tbl[start_point[0], start_point[1]] = 1
    while True:
        next_coords = get_close_coords_vec(point_where(visited_tbl == 1), board_size=visited_tbl.shape[0])

        # clean from free points
        next_coords = next_coords[table[next_coords[:, 0], next_coords[:, 1]] != 0]
        # clean from visited points
        next_coords = next_coords[visited_tbl[next_coords[:, 0], next_coords[:, 1]] == 0]
        if len(next_coords) == 0:
            break
        visited_tbl[next_coords[:, 0], next_coords[:, 1]] = 1
    # check matching of visited cells and presented in table cells
    return np.multiply(visited_tbl, table != 0).sum() != (table != 0).sum()


# @jit(nopython=False)
def get_available_moves_around_hive(point_from: Point, table: Table) -> Tuple[PointArray, npt.NDArray[int]]:
    """
    Walks around the hive to find all possible moves without jump over pieces.
    Algorithm is similar to breadth first search.
    """
    if table[point_from[0], point_from[1]] == 0:
        raise ValueError("Point from cannot be empty")
    distance_tbl = np.full(table.shape, fill_value=np.inf)
    distance_tbl[point_from[0], point_from[1]] = 0
    # remove from point
    table = move_point_in_table(table, src_point=Point(point_from[0], point_from[1]), default_value=0)
    have_candidates = True
    while have_candidates:
        have_candidates = False
        for next_point in point_where(distance_tbl != np.inf):
            candidates = get_close_coords_point(tuple(next_point), board_size=table.shape[0])
            candidates_with_filled_neighbours = []
            # clean
            move_locked = get_movenent_locked_fast(source=np.array([next_point]*len(candidates)), dest=candidates, table=table)

            for i, candidate in enumerate(candidates):
                close_to_candidate = get_close_coords_point(tuple(candidate), board_size=table.shape[0])
                has_candidate_filled_neighbours = (table[close_to_candidate[:, 0], close_to_candidate[:, 1]] != 0).any()
                candidates_with_filled_neighbours.append(has_candidate_filled_neighbours and not move_locked[i])

            candidates = candidates[candidates_with_filled_neighbours]
            # keep only free points
            candidates = candidates[table[candidates[:, 0], candidates[:, 1]] == 0]
            # clean from visited points
            candidates = candidates[distance_tbl[candidates[:, 0], candidates[:, 1]] == np.inf]
            if len(candidates) == 0:
                continue
            have_candidates = True
            distance_tbl[candidates[:, 0], candidates[:, 1]] = np.minimum(
                distance_tbl[next_point[0], next_point[1]] + 1,
                distance_tbl[candidates[:, 0], candidates[:, 1]]
            )
    final_points = point_where((distance_tbl >= 1) & (distance_tbl != np.inf))
    return final_points, distance_tbl[final_points[:, 0], final_points[:, 1]]


class WinnerState(enum.Enum):
    no_termination = 0
    player_1_win = 1
    player_2_win = 2
    draw_game = 3


def compute_rescale_args(table: Table, central_point=CENTRAL_POINT) -> Tuple[int, int]:
    non_zero = point_where(table != 0)
    if len(non_zero) == 0:
        return 0, 0

    old_center_row = (np.min(non_zero[:, 0]) + np.max(non_zero[:, 0])) // 2
    old_center_col = (np.min(non_zero[:, 1]) + np.max(non_zero[:, 1])) // 2
    row_diff = central_point[0] - old_center_row
    col_diff = central_point[1] - old_center_col
    return row_diff, col_diff


def rescale_tables(*table_seq: Table, row_diff: int, col_diff: int) -> List[Table]:
    res = []
    for table in table_seq:
        non_zero = point_where(table != 0)
        new_table = np.zeros(table.shape, dtype=table.dtype)
        new_table[non_zero[:, 0] + row_diff, non_zero[:, 1] + col_diff] = table[non_zero[:, 0], non_zero[:, 1]]
        res.append(new_table)
    return res


def rescale_animal_info(animal_info: npt.NDArray, row_diff: int, col_diff: int) -> npt.NDArray:
    animal_info = np.copy(animal_info)
    for player_idx in np.arange(animal_info.shape[0]) + 1:
        animal_info[player_idx - 1][:, 1] += row_diff
        animal_info[player_idx - 1][:, 2] += col_diff
    return animal_info


class ActionStatus(enum.Enum):
    selected_animal_doesnt_exist = 0
    invalid_player_idx = 1
    bee_was_not_placed_during_first_3_rounds = 2
    invalid_action_bee = 3
    invalid_action_grasshopper = 4
    invalid_action_spider = 5
    invalid_action_ant = 6
    success = 7
    no_possible_action = 8


class HiveGame:
    """
    Stateful representation of hive game.
    """

    def __init__(
            self,
            animal_info: npt.NDArray,
            last_player_idx: Literal[1, 2],
            turn_num: int,
            board_size=BOARD_SIZE,
    ):
        assert (animal_info[0][:, 0] == AnimalType.bee.value).any(), "Bee expected to be is placed"
        assert animal_info.shape[0] == 2, f"Num players should equal to 2: {animal_info.shape[0]}"
        assert (animal_info[0][:, 0] == animal_info[1][:, 0]).all(), \
            f"Animal types should be same: {animal_info[0][:, 0]}, {animal_info[1][:, 0]}"
        point_from_0 = animal_info[0][:, [1, 2]]
        point_from_1 = animal_info[1][:, [1, 2]]
        for point_from in [point_from_0, point_from_1]:
            assert_array_equal(np.isnan(point_from) | (point_from < board_size), True)
            assert_array_equal(np.isnan(point_from).sum(axis=1) != 1, True)

        # keep only initialized pieces
        valid_points_0 = ~np.isnan(point_from_0).any(axis=1)
        valid_points_1 = ~np.isnan(point_from_1).any(axis=1)
        point_from_0 = point_from_0[valid_points_0].astype(int)
        point_from_1 = point_from_1[valid_points_1].astype(int)

        assert len(np.vstack((point_from_0, point_from_1))) == \
            len(np.unique(np.vstack((point_from_0, point_from_1)), axis=0))

        animal_idx_table = np.zeros((board_size, board_size), dtype=int)
        player_table = np.zeros((board_size, board_size), dtype=int)

        animal_idx_table[point_from_0[:, 0], point_from_0[:, 1]] = np.arange(animal_info[0].shape[0])[valid_points_0] + 1
        animal_idx_table[point_from_1[:, 0], point_from_1[:, 1]] = np.arange(animal_info[1].shape[0])[valid_points_1] + 1

        player_table[point_from_0[:, 0], point_from_0[:, 1]] = 1
        player_table[point_from_1[:, 0], point_from_1[:, 1]] = 2

        if is_graph_component_more_than_1(player_table):
            raise ValueError("Number of graph components is greater than 1")
        if turn_num > 2:
            for player_idx in [1, 2]:
                cond = animal_info[player_idx - 1][:, 1] == AnimalType.bee
                bee_point = animal_info[player_idx - 1][cond][:, [1, 2]]
                if np.isnan(bee_point).sum() > 0:
                    raise ValueError(f"Bee for player {player_idx} was not placed")

        self.animal_info = animal_info
        self.animal_idx_table = animal_idx_table
        self.player_table = player_table
        self.last_player_idx = last_player_idx
        self.turn_num = turn_num
        self.board_size = board_size
        self.state_log = [deepcopy(self.animal_info)]
        self.action_log = []
        self.disable_cache = False
        self.cache = cachetools.LRUCache(maxsize=1000)
        self.action_mask_cache = cachetools.LRUCache(maxsize=5)

    @staticmethod
    def from_setup(num_ants: int = 3, num_spiders=3, num_grasshoppers=3, board_size=BOARD_SIZE):
        animal_types = []
        # to fill zero position with None
        for animal_type, num in [
            (AnimalType.bee, 1),
            (AnimalType.ant, num_ants),
            (AnimalType.spider, num_spiders),
            (AnimalType.grasshopper, num_grasshoppers)
        ]:
            animal_types += [animal_type.value] * num
        animal_types = np.array(animal_types)
        np.random.shuffle(animal_types)

        animal_info = np.array(
            [
                [
                    (animal_type, None, None)
                    for animal_type in animal_types
                ]
                for player_idx in [1, 2]
            ],
            dtype=float
        )
        return HiveGame(animal_info=animal_info, last_player_idx=2, turn_num=0, board_size=board_size)

    def apply_action(self, player_idx: Literal[1, 2], animal_idx: int, point_to: Point, disable_rescale: bool = False, raise_error: bool = True) -> ActionStatus:
        # approach to play 2 times with same player_idx
        action_state = self.check_action(player_idx=player_idx, animal_idx=animal_idx, point_to=point_to)
        if action_state != ActionStatus.success:
            return action_state

        animal, row_from, col_from = self.animal_info[player_idx - 1][animal_idx - 1]
        point_from = None if np.isnan(row_from) else Point(int(round(row_from)), int(round(col_from)))

        self.animal_idx_table = move_point_in_table(self.animal_idx_table, point_from, point_to, value=animal_idx)
        self.player_table = move_point_in_table(self.player_table, point_from, point_to, value=player_idx)
        self.animal_info[player_idx - 1][animal_idx - 1] = (animal, point_to[0], point_to[1])
        self.last_player_idx = player_idx
        self.turn_num += 1
        self.state_log.append(deepcopy(self.animal_info))
        self.action_log.append({"player_idx": player_idx, "animal_idx": animal_idx, "point_to": point_to, "disable_rescale": disable_rescale})
        # rescale
        if not disable_rescale:
            self.rescale()
        if is_graph_component_more_than_1(self.player_table) and raise_error:
            print(f"INVALID_STATE_AFTER_ACTION: PLAYER_IDX={player_idx}, ANIMAL_IDX={animal_idx}, POINT_TO={point_to}, POINT_FROM={point_from}")
            path = Path("trace/invalid_state_game.pickle")
            path.unlink(missing_ok=True)
            Path("trace").mkdir(parents=True, exist_ok=True)

            with open(str(path), "wb") as f:
                pickle.dump(self, f)
            raise ValueError("Action invalidates game state")
        return action_state

    def check_action(self, player_idx: Literal[1, 2], animal_idx: int, point_to: Point) -> ActionStatus:
        if self.check_no_moves(player_idx) or self.get_winner_state() != WinnerState.no_termination:
            return ActionStatus.no_possible_action
        assert animal_idx is not None and point_to is not None

        # check that animal may be chosen
        if animal_idx < 1 or animal_idx > self.animal_info.shape[1]:
            return ActionStatus.selected_animal_doesnt_exist

        animal, row_from, col_from = self.animal_info[player_idx - 1][animal_idx - 1]
        point_from = None if np.isnan(row_from) else Point(int(round(row_from)), int(round(col_from)))

        if self.turn_num >= 4 and animal != AnimalType.bee.value:
            if not self._is_bee_placed(player_idx):
                return ActionStatus.bee_was_not_placed_during_first_3_rounds

        if animal == AnimalType.ant.value:
            if not (point_to == self.get_all_possible_dest_points_for_ant(player_idx, point_from)).all(axis=1).any():
                return ActionStatus.invalid_action_ant
        elif animal == AnimalType.bee.value:
            if not (point_to == self.get_all_possible_dest_points_for_bee(player_idx, point_from)).all(axis=1).any():
                return ActionStatus.invalid_action_bee
        elif animal == AnimalType.spider.value:
            if not (point_to == self.get_all_possible_dest_points_for_spider(player_idx, point_from)).all(axis=1).any():
                return ActionStatus.invalid_action_spider
        elif animal == AnimalType.grasshopper.value:
            if not (point_to == self.get_all_possible_dest_points_for_grasshopper(player_idx, point_from)).all(axis=1).any():
                return ActionStatus.invalid_action_grasshopper
        else:
            raise ValueError()
        return ActionStatus.success

    def _get_allocation_points(
            self,
            turn_num: int,
            player_idx: Literal[1, 2],
    ) -> PointArray:
        if turn_num == 0:
            return point_where(~np.isnan(self.player_table))
        if turn_num == 1:
            start_point = Point(*point_where(self.player_table != 0)[0, :])
            return get_close_coords_point(start_point, board_size=self.board_size)

        player_points = point_where(self.player_table == player_idx)
        point_to = np.unique(get_close_coords_vec(player_points, board_size=self.board_size), axis=0)

        enemy_table = ~((self.player_table == 0) | (self.player_table == player_idx))
        enemy_and_close_points = np.unique(
            np.concatenate(
                (
                    get_close_coords_vec(point_where(enemy_table), board_size=self.board_size),
                    point_where(enemy_table)
                )
            ), axis=0
        )
        point_to = point_to[self.player_table[point_to[:, 0], point_to[:, 1]] == 0]
        point_to = np.array([
            y for y in list(point_to)
            if not tuple(y) in set(tuple(x) for x in enemy_and_close_points)
        ], dtype=int)
        return point_to

    @cachetools.cachedmethod(
        lambda self: self.cache,
        key=lambda self, player_idx, point_from:
        (self.board_size, self.turn_num, player_idx, point_from)
    )
    def get_all_possible_dest_points_for_bee(self, player_idx: Literal[1, 2], point_from: Optional[Point]) -> PointArray:
        if point_from is None:
            return self._get_allocation_points(self.turn_num, player_idx)

        # removal of current point doesn't cause gap in chain
        if is_graph_component_more_than_1(move_point_in_table(self.player_table, point_from)):
            return np.array([], dtype=int).reshape((0, 2))

        point_to = np.array(get_close_coords_point(point_from, board_size=self.board_size))
        res = self._validate_next_points(point_from, point_to)
        return res

    @cachetools.cachedmethod(
        lambda self: self.cache,
        key=lambda self, player_idx, point_from:
        (self.board_size, self.turn_num, player_idx, point_from)
    )
    def get_all_possible_dest_points_for_ant(self, player_idx: Literal[1, 2], point_from: Point) -> PointArray:
        if point_from is None:
            return self._get_allocation_points(self.turn_num, player_idx)

        # removal of current point doesn't cause gap in chain
        if is_graph_component_more_than_1(move_point_in_table(self.player_table, point_from)):
            return np.array([], dtype=int).reshape((0, 2))

        # get next points
        point_to, distances = get_available_moves_around_hive(point_from, self.player_table)

        res = self._validate_next_points(point_from, point_to)
        return np.array(res)

    @cachetools.cachedmethod(
        lambda self: self.cache,
        key=lambda self, player_idx, point_from:
        (self.board_size, self.turn_num, player_idx, point_from)
    )
    def get_all_possible_dest_points_for_grasshopper(self, player_idx: Literal[1, 2], point_from: Point) -> PointArray:
        if point_from is None:
            return self._get_allocation_points(self.turn_num, player_idx)

        # removal of current point doesn't cause gap in chain
        if is_graph_component_more_than_1(move_point_in_table(self.player_table, point_from)):
            return np.array([], dtype=int).reshape((0, 2))

        # get next points
        point_to = []
        for direction in [(1, 1), (-1, 1), (-1, -1), (1, -1), (-2, 0), (2, 0)]:
            point = point_from
            while (safe_x_y(self.player_table, Point(point[0] + direction[0], point[1] + direction[1])) or 0) != 0:
                point = Point(point[0] + direction[0], point[1] + direction[1])
            if point != point_from:
                point_to.append(Point(point[0] + direction[0], point[1] + direction[1]))
        point_to = np.array(point_to) if point_to else np.array([], dtype=int).reshape((0, 2))
        point_to = self._validate_next_points(point_from, point_to)
        return point_to

    @cachetools.cachedmethod(
        lambda self: self.cache,
        key=lambda self, player_idx, point_from:
        (self.board_size, self.turn_num, player_idx, point_from)
    )
    def get_all_possible_dest_points_for_spider(self, player_idx: Literal[1, 2], point_from: Point) -> PointArray:
        if point_from is None:
            return self._get_allocation_points(self.turn_num, player_idx)

        # removal of current point doesn't cause gap in chain
        if is_graph_component_more_than_1(move_point_in_table(self.player_table, point_from)):
            return np.array([], dtype=int).reshape((0, 2))

        # get next points
        point_to, distances = get_available_moves_around_hive(point_from, self.player_table)
        point_to = point_to[distances == 3]

        res = self._validate_next_points(point_from, point_to)
        return np.array(res)

    def _validate_next_points(self, point_from: Point, point_to: PointArray) -> PointArray:
        # next point is free
        point_to = point_to[self.player_table[point_to[:, 0], point_to[:, 1]] == 0]
        res = []
        for dest in point_to:
            # next_point is outside of hive
            if is_graph_component_more_than_1(move_point_in_table(self.player_table, point_from, dest)):
                continue
            res.append(dest)
        return np.array(res) if res else np.array([], dtype=int).reshape((0, 2))

    def get_winner_state(self) -> WinnerState:
        captured_bees = self._get_captured_bees()
        if len(captured_bees) == 1:
            if captured_bees[0] == 1:
                return WinnerState.player_2_win
            else:
                return WinnerState.player_1_win
        elif len(captured_bees) == 2 or (self.check_no_moves(1) and self.check_no_moves(2)):
            return WinnerState.draw_game
        return WinnerState.no_termination

    def _get_captured_bees(self) -> List[Literal[1, 2]]:
        res = []
        if self.num_free_places_around_bee(1) == 0:
            res.append(1)
        if self.num_free_places_around_bee(2) == 0:
            res.append(2)
        return res

    def get_state(self, player_idx: Literal[1, 2]) -> Tuple[Table, Table, Table, List[AnimalType]]:
        enemy_table = np.zeros(self.player_table.shape)
        enemy_table[(self.player_table == player_idx)] = 1
        enemy_table[(self.player_table != player_idx) & (self.player_table != 0)] = 2

        animal_type_table = np.hstack(([0], self.animal_info[0][:, 0]))[self.animal_idx_table]
        return enemy_table, animal_type_table, self.animal_idx_table, self.animal_info[player_idx - 1][:, 0]

    def check_no_moves(self, player_idx: Literal[1, 2]) -> bool:
        return not self.get_action_mask(player_idx).any()

    @cachetools.cachedmethod(
        lambda self: self.action_mask_cache,
        key=lambda self, player_idx, *args, **kwargs: (self.board_size, self.turn_num, player_idx)
    )
    def get_action_mask(self, player_idx: Literal[1, 2]) -> Table:
        """
        Builds mapping with all available actions for requested player
        """
        animal_info = self.animal_info[player_idx - 1]
        action_mask = np.zeros((animal_info.shape[0], self.board_size, self.board_size))
        for i, (animal_type, row_from, col_from) in enumerate(animal_info):
            point_from = None if np.isnan(row_from) else Point(int(round(row_from)), int(round(col_from)))

            if animal_type == AnimalType.ant.value:
                points = self.get_all_possible_dest_points_for_ant(player_idx, point_from)
            elif animal_type == AnimalType.spider.value:
                points = self.get_all_possible_dest_points_for_spider(player_idx, point_from)
            elif animal_type == AnimalType.grasshopper.value:
                points = self.get_all_possible_dest_points_for_grasshopper(player_idx, point_from)
            elif animal_type == AnimalType.bee.value:
                points = self.get_all_possible_dest_points_for_bee(player_idx, point_from)
            else:
                raise ValueError(f"Unsupported {animal_type=}")
            if len(points.shape) != 2:
                raise ValueError("")
            action_mask[i, points[:, 0], points[:, 1]] = 1

        if self.turn_num >= 4 and not self._is_bee_placed(player_idx):
            bee_idx = np.where(self.animal_info[player_idx - 1][:, 0] == AnimalType.bee.value)[0][0]
            action_mask[np.arange(self.animal_info.shape[1]) != bee_idx] = 0

        return action_mask

    def rescale(self):
        central_point = (self.board_size // 2, self.board_size // 2)
        row_diff, col_diff = compute_rescale_args(self.player_table, central_point=central_point)
        self.player_table, self.animal_idx_table = rescale_tables(
            self.player_table,
            self.animal_idx_table,
            row_diff=row_diff,
            col_diff=col_diff
        )
        self.animal_info = rescale_animal_info(self.animal_info, row_diff, col_diff)

    def set_player_idx(self, last_player_idx: int) -> "HiveGame":
        self.last_player_idx = last_player_idx
        return self

    def set_board_size(self, board_size: int) -> "HiveGame":
        self.board_size = board_size
        return self

    def _is_bee_placed(self, player_idx: Literal[1, 2]) -> bool:
        bee_cond = self.animal_info[player_idx - 1][:, 0] == AnimalType.bee.value
        bee_point = self.animal_info[player_idx - 1][bee_cond, [1, 2]]
        bee_point = None if np.isnan(bee_point[0]) else bee_point
        return bee_point is not None

    def get_free_pieces(self, player_idx: Literal[1, 2]):
        action_mask = self.get_action_mask(player_idx)
        return (action_mask.sum(axis=0) > 0).sum()

    def is_player_bee_free(self, player_idx: Literal[1, 2]) -> bool:
        bee_idx = np.where(self.animal_info[0][:, 0] == AnimalType.bee.value)[0]
        assert len(bee_idx) == 1, f"No bee in animal info: {self.animal_info}"

        action_mask = self.get_action_mask(player_idx)
        return action_mask[bee_idx].sum() != 0

    def num_free_places_around_bee(self, player_idx: Literal[1, 2]) -> Optional[int]:
        bee_idx = np.where(self.animal_info[0][:, 0] == AnimalType.bee.value)[0]
        assert len(bee_idx) == 1, f"No bee in animal info: {self.animal_info}"
        bee_idx = bee_idx[0]

        animal, row_from, col_from = self.animal_info[player_idx - 1][bee_idx]
        point_from = None if np.isnan(row_from) else Point(int(round(row_from)), int(round(col_from)))

        if point_from is None:
            return None

        bee_close_points = get_close_coords_point(Point(*point_from), board_size=self.board_size)
        return np.sum(self.animal_idx_table[bee_close_points[:, 0], bee_close_points[:, 1]] == 0)


@jit(nopython=True)
def point_where(tbl: Table) -> PointArray:
    # assert tbl.dtype == np.bool_
    row, col = np.where(tbl)
    return np.vstack((row, col)).T
