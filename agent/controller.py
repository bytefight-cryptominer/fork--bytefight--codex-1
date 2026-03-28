from collections.abc import Callable, Iterable
from typing import Union

from game import *
import random


class PlayerController:
    """
    Organizer sample: make a random legal cardinal move.
    """

    def __init__(self, player_parity: int, time_left: Callable):
        return

    def bid(self, board: Board, player_parity: int, time_left: Callable) -> int:
        return 0

    def play(
        self,
        board: Board,
        player_parity: int,
        time_left: Callable,
    ) -> Union[Action.Move, Action.Paint, Iterable[Action.Move | Action.Paint]]:
        available = []
        for direction in Direction:
            next_loc = board.get_player(player_parity).loc + direction
            if not board.oob(next_loc) and not board.cells[next_loc.r][next_loc.c].is_wall:
                available.append(Action.Move(direction))

        return random.choice(available)

    def commentate(self, board: Board, player_parity: int, time_left: Callable) -> str:
        return ""
