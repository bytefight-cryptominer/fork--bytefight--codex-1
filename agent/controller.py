from collections.abc import Callable, Iterable
from collections import deque
from typing import Union, List

import numpy as np
import numba

from game import *


# Numba-compiled BFS for ultra-fast deep search
@numba.njit(cache=True)
def _numba_bfs(walls, owner, hill_id, hill_ctrl, powerup,
               my_r, my_c, opp_r, opp_c, rows, cols,
               safe_dist, player_parity, opp_parity, stamina, max_depth):
    """Returns (best_dir_idx, best_priority) where dir_idx is 0-3 for UP/DOWN/LEFT/RIGHT."""
    DRS = np.array([[-1,0],[1,0],[0,-1],[0,1]], dtype=np.int32)

    visited = np.zeros((rows, cols), dtype=numba.boolean)
    visited[my_r, my_c] = True

    # Queue: (r, c, first_dir_idx, depth) stored as int array
    # Max queue size = rows * cols
    q_r = np.zeros(rows * cols, dtype=np.int32)
    q_c = np.zeros(rows * cols, dtype=np.int32)
    q_dir = np.zeros(rows * cols, dtype=np.int32)
    q_depth = np.zeros(rows * cols, dtype=np.int32)
    q_head = 0
    q_tail = 0

    best_dir = -1
    best_priority = -999999

    # Seed neighbors
    for di in range(4):
        nr = my_r + DRS[di, 0]
        nc = my_c + DRS[di, 1]
        if nr < 0 or nr >= rows or nc < 0 or nc >= cols or walls[nr, nc]:
            continue
        d_opp = abs(nr - opp_r) + abs(nc - opp_c)
        # Collision check
        if nr == opp_r and nc == opp_c:
            if owner[nr, nc] != opp_parity:
                return di, 99999  # instant kill!
            continue
        # Safety check (only at depth 1)
        if owner[nr, nc] == opp_parity and d_opp <= safe_dist:
            continue
        visited[nr, nc] = True
        q_r[q_tail] = nr
        q_c[q_tail] = nc
        q_dir[q_tail] = di
        q_depth[q_tail] = 1
        q_tail += 1

    while q_head < q_tail:
        r = q_r[q_head]
        c = q_c[q_head]
        first_dir = q_dir[q_head]
        depth = q_depth[q_head]
        q_head += 1

        if depth > max_depth:
            break

        priority = -999999

        # Hill scoring
        hid = hill_id[r, c]
        if hid != 0:
            ctrl = hill_ctrl[hid] if hid < len(hill_ctrl) else 0
            if ctrl != player_parity:  # uncaptured
                if owner[r, c] != player_parity:
                    priority = 2000 - depth * 20
                else:
                    priority = 1000 - depth * 20
            elif owner[r, c] == 0:
                priority = 800 - depth * 15

        # Powerup
        if powerup[r, c]:
            pup = 1500 - depth * 25
            if stamina < 60:
                pup += 300
            if pup > priority:
                priority = pup

        # Neutral
        if owner[r, c] == 0 and priority < -900:
            priority = 900 - depth * 20

        # Enemy far
        if owner[r, c] == opp_parity and priority < -900:
            d_opp = abs(r - opp_r) + abs(c - opp_c)
            if d_opp > safe_dist:
                priority = 300 - depth * 15

        # Paintable tiebreak
        if priority > -900:
            paintable = 0
            for di2 in range(4):
                pr = r + DRS[di2, 0]
                pc = c + DRS[di2, 1]
                if pr < 0 or pr >= rows or pc < 0 or pc >= cols or walls[pr, pc]:
                    continue
                if owner[pr, pc] == 0:
                    paintable += 2
                elif owner[pr, pc] == player_parity:
                    paintable += 1
            priority += paintable * 2

        if priority > best_priority:
            best_priority = priority
            best_dir = first_dir

        # Expand (allow enemy traversal at depth > 1)
        if depth < max_depth:
            for di in range(4):
                nr = r + DRS[di, 0]
                nc = c + DRS[di, 1]
                if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                    continue
                if walls[nr, nc] or visited[nr, nc]:
                    continue
                visited[nr, nc] = True
                q_r[q_tail] = nr
                q_c[q_tail] = nc
                q_dir[q_tail] = first_dir
                q_depth[q_tail] = depth + 1
                q_tail += 1

    return best_dir, best_priority


class PlayerController:
    """
    v69: Numba JIT-compiled BFS for ultra-deep search (depth 100).
    The JIT compilation makes BFS ~50x faster, allowing full-map coverage.
    """

    def __init__(self, player_parity: int, time_left: Callable):
        self.parity = player_parity
        self.opp = -player_parity
        # Warm up numba JIT (first call triggers compilation)
        _dummy_walls = np.zeros((2,2), dtype=np.bool_)
        _dummy_owner = np.zeros((2,2), dtype=np.int32)
        _dummy_hill = np.zeros((2,2), dtype=np.int32)
        _dummy_ctrl = np.zeros(1, dtype=np.int32)
        _dummy_pup = np.zeros((2,2), dtype=np.bool_)
        _numba_bfs(_dummy_walls, _dummy_owner, _dummy_hill, _dummy_ctrl, _dummy_pup,
                   0, 0, 1, 1, 2, 2, 5, 1, -1, 100, 10)

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
        stamina = me.stamina
        opp_stamina = opp.stamina

        DR = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        DIRS = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]
        DIR_MAP = {(-1, 0): Direction.UP, (1, 0): Direction.DOWN,
                   (0, -1): Direction.LEFT, (0, 1): Direction.RIGHT}
        INV_DIR = {Direction.UP: (-1, 0), Direction.DOWN: (1, 0),
                   Direction.LEFT: (0, -1), Direction.RIGHT: (0, 1)}

        def valid(r, c):
            return 0 <= r < rows and 0 <= c < cols and not board.cells[r][c].is_wall

        def mdist(r1, c1, r2, c2):
            return abs(r1 - r2) + abs(c1 - c2)

        def cell_owner(r, c):
            return board.cells[r][c].owner_parity

        stamina_diff = stamina - opp_stamina
        if stamina_diff > 30:
            SAFE_DIST = 3
        elif stamina_diff < -30:
            SAFE_DIST = 6
        else:
            SAFE_DIST = 5

        # --- Erase step for hill cells with opponent paint ---
        if stamina >= 55:
            for dr, dc in DR:
                nr, nc = my_r + dr, my_c + dc
                if not valid(nr, nc):
                    continue
                ecell = board.cells[nr][nc]
                if (ecell.hill_id and ecell.hill_id != 0 and
                    ecell.owner_parity == self.opp):
                    if nr == opp_r and nc == opp_c:
                        continue
                    return [Action.Move(DIR_MAP[(dr, dc)], move_type=MoveType.ERASE)]

        # --- Convert board to numpy arrays for numba BFS ---
        walls = np.zeros((rows, cols), dtype=np.bool_)
        owner = np.zeros((rows, cols), dtype=np.int32)
        hill_id_arr = np.zeros((rows, cols), dtype=np.int32)
        powerup_arr = np.zeros((rows, cols), dtype=np.bool_)

        for r in range(rows):
            for c in range(cols):
                cell = board.cells[r][c]
                walls[r, c] = cell.is_wall
                owner[r, c] = cell.owner_parity
                hill_id_arr[r, c] = cell.hill_id if cell.hill_id else 0
                powerup_arr[r, c] = bool(cell.powerup)

        # Hill controller array
        max_hid = max(board.hills.keys()) if board.hills else 0
        hill_ctrl = np.zeros(max_hid + 1, dtype=np.int32)
        for hid, hill in board.hills.items():
            if hid < len(hill_ctrl):
                hill_ctrl[hid] = hill.controller_parity

        # --- Numba BFS (depth 100 = full map) ---
        best_dir_idx, best_priority = _numba_bfs(
            walls, owner, hill_id_arr, hill_ctrl, powerup_arr,
            my_r, my_c, opp_r, opp_c, rows, cols,
            SAFE_DIST, player_parity, self.opp, stamina, 100
        )

        if best_dir_idx >= 0 and best_priority == 99999:
            # Collision kill
            return [Action.Move(DIRS[best_dir_idx])]

        best_first_dir = DIRS[best_dir_idx] if best_dir_idx >= 0 else None

        if best_first_dir is None:
            for dr, dc in DR:
                nr, nc = my_r + dr, my_c + dc
                if valid(nr, nc):
                    best_first_dir = DIR_MAP[(dr, dc)]
                    break
            if best_first_dir is None:
                return Action.Move(Direction.UP)

        actions: List = []
        actions.append(Action.Move(best_first_dir))

        ddr, ddc = INV_DIR[best_first_dir]
        new_r, new_c = my_r + ddr, my_c + ddc

        if not valid(new_r, new_c):
            return actions

        reserve = 25 if board.turn_count > 1400 else 10
        paint_budget = stamina - reserve
        paint_spent = 0

        paint_candidates = []
        for dr, dc in DR:
            pr, pc = new_r + dr, new_c + dc
            if not (0 <= pr < rows and 0 <= pc < cols):
                continue
            pcell = board.cells[pr][pc]
            if pcell.is_wall or pcell.beacon_parity == player_parity:
                continue
            if pcell.owner_parity != player_parity and pcell.owner_parity != 0:
                continue
            if pcell.owner_parity == player_parity and abs(pcell.paint_value) >= GameConstants.MAX_PAINT_VALUE:
                continue

            pscore = 0
            if pcell.hill_id and pcell.hill_id != 0:
                pscore += 200
            if pr == my_r and pc == my_c:
                pscore += 150
            if pcell.owner_parity == 0:
                pscore += 100
            else:
                pscore += 10
            paint_candidates.append((pscore, pr, pc))

        paint_candidates.sort(key=lambda x: -x[0])
        for _, pr, pc in paint_candidates:
            if paint_spent + GameConstants.PAINT_STAMINA_COST > paint_budget:
                break
            actions.append(Action.Paint(Location(pr, pc)))
            paint_spent += GameConstants.PAINT_STAMINA_COST

        return actions

    def commentate(self, board: Board, player_parity: int, time_left: Callable) -> str:
        return ""
