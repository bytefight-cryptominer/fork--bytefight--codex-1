from collections.abc import Callable, Iterable
from collections import deque
from typing import Union, List, Tuple

from game import *
import random


class PlayerController:
    """
    BFS expansion agent v2: move toward frontier, paint after moving.

    Strategy:
    - BFS from current pos to find nearest unpainted/neutral/hill cell
    - Move toward that target
    - Paint adjacent cells after moving (prioritize hills, neutral cells)
    - Collision avoidance: never step on enemy cell within SAFE_DIST of opponent
    - Stamina management: keep reserve, don't over-paint
    """

    def __init__(self, player_parity: int, time_left: Callable):
        self.parity = player_parity
        self.opp = -player_parity

    def bid(self, board: Board, player_parity: int, time_left: Callable) -> int:
        return 0

    def play(
        self,
        board: Board,
        player_parity: int,
        time_left: Callable,
    ) -> Union[Action.Move, Action.Paint, Iterable[Action.Move | Action.Paint]]:
        me = board.get_player(player_parity)
        opp = board.get_opponent(player_parity)
        rows, cols = board.board_size.r, board.board_size.c
        my_r, my_c = me.loc.r, me.loc.c
        opp_r, opp_c = opp.loc.r, opp.loc.c

        SAFE_DIST = 5
        DIRS = [(-1, 0, Direction.UP), (1, 0, Direction.DOWN),
                (0, -1, Direction.LEFT), (0, 1, Direction.RIGHT)]

        def valid(r, c):
            return 0 <= r < rows and 0 <= c < cols and not board.cells[r][c].is_wall

        def mdist(r1, c1, r2, c2):
            return abs(r1 - r2) + abs(c1 - c2)

        # BFS to find the best target cell to move toward
        # Priority: uncaptured hill > neutral > enemy (far from opp) > reinforce own
        best_target = None
        best_first_dir = None
        best_priority = -1

        visited = set()
        visited.add((my_r, my_c))
        # (r, c, first_direction, depth)
        queue = deque()

        for dr, dc, d_enum in DIRS:
            nr, nc = my_r + dr, my_c + dc
            if not valid(nr, nc):
                continue
            # Safety check
            d_opp = mdist(nr, nc, opp_r, opp_c)
            cell = board.cells[nr][nc]
            if cell.owner_parity == self.opp and d_opp <= SAFE_DIST:
                continue
            if d_opp <= 1 and cell.owner_parity == self.opp:
                continue
            visited.add((nr, nc))
            queue.append((nr, nc, d_enum, 1))

        while queue:
            r, c, first_dir, depth = queue.popleft()
            if depth > 20:
                break

            cell = board.cells[r][c]
            priority = -1

            # Uncaptured hill cell we don't own
            if cell.hill_id and cell.hill_id != 0:
                hill = board.hills[cell.hill_id]
                if hill.controller_parity != player_parity:
                    if cell.owner_parity != player_parity:
                        priority = 1000 - depth * 10
                    else:
                        priority = 500 - depth * 10

            # Neutral cell (expansion)
            if cell.owner_parity == 0 and priority < 0:
                priority = 800 - depth * 15

            # Enemy cell far from opponent
            if cell.owner_parity == self.opp and priority < 0:
                d_opp = mdist(r, c, opp_r, opp_c)
                if d_opp > SAFE_DIST:
                    priority = 200 - depth * 10

            if priority > best_priority:
                best_priority = priority
                best_target = (r, c)
                best_first_dir = first_dir

            # Continue BFS
            if depth < 20:
                for dr, dc, _ in DIRS:
                    nr, nc = r + dr, c + dc
                    if (nr, nc) in visited:
                        continue
                    if not valid(nr, nc):
                        continue
                    # Don't path through enemy territory near opponent
                    d_opp = mdist(nr, nc, opp_r, opp_c)
                    if board.cells[nr][nc].owner_parity == self.opp and d_opp <= SAFE_DIST:
                        continue
                    visited.add((nr, nc))
                    queue.append((nr, nc, first_dir, depth + 1))

        # Fallback: just move somewhere valid
        if best_first_dir is None:
            for dr, dc, d_enum in DIRS:
                nr, nc = my_r + dr, my_c + dc
                if valid(nr, nc):
                    best_first_dir = d_enum
                    break
            if best_first_dir is None:
                return Action.Move(Direction.UP)

        actions: List = []

        # Move first
        actions.append(Action.Move(best_first_dir))

        # Calculate new position after move
        dir_delta = {Direction.UP: (-1, 0), Direction.DOWN: (1, 0),
                     Direction.LEFT: (0, -1), Direction.RIGHT: (0, 1)}
        dr, dc = dir_delta[best_first_dir]
        new_r, new_c = my_r + dr, my_c + dc

        if not valid(new_r, new_c):
            return actions

        # Paint after moving: paint cells adjacent to new position
        # Budget: keep at least 20 stamina in reserve
        stamina_available = me.stamina - 20  # rough estimate
        paint_candidates = []

        for dr2, dc2, _ in DIRS:
            pr, pc = new_r + dr2, new_c + dc2
            if not (0 <= pr < rows and 0 <= pc < cols):
                continue
            pcell = board.cells[pr][pc]
            if pcell.is_wall:
                continue
            if pcell.beacon_parity == player_parity:
                continue
            # Can only paint neutral or own cells
            if pcell.owner_parity != player_parity and pcell.owner_parity != 0:
                continue
            # Skip if already at max paint
            if pcell.owner_parity == player_parity and abs(pcell.paint_value) >= GameConstants.MAX_PAINT_VALUE:
                continue

            # Scoring for paint priority
            pscore = 0
            if pcell.hill_id and pcell.hill_id != 0:
                pscore += 100  # paint hill cells first
            if pcell.owner_parity == 0:
                pscore += 50  # paint neutral cells
            else:
                pscore += 10  # reinforce own cells

            paint_candidates.append((pscore, pr, pc))

        # Sort by priority
        paint_candidates.sort(key=lambda x: -x[0])

        cost = 0
        for _, pr, pc in paint_candidates:
            if cost + GameConstants.PAINT_STAMINA_COST > stamina_available:
                break
            actions.append(Action.Paint(Location(pr, pc)))
            cost += GameConstants.PAINT_STAMINA_COST

        return actions

    def commentate(self, board: Board, player_parity: int, time_left: Callable) -> str:
        return ""
