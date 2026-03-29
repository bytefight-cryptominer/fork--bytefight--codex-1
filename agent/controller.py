from collections.abc import Callable, Iterable
from collections import deque
from typing import Union, List
from concurrent.futures import ThreadPoolExecutor

from game import *


class PlayerController:
    """
    v110: Multi-threaded evaluation with parallel forecast.
    Uses ThreadPoolExecutor for parallel board state evaluation.
    """

    def __init__(self, player_parity: int, time_left: Callable):
        self.parity = player_parity
        self.opp = -player_parity
        self.executor = ThreadPoolExecutor(max_workers=3)

    def bid(self, board: Board, player_parity: int, time_left: Callable) -> int:
        return 0

    def _eval_direction(self, board, player_parity, direction, my_r, my_c, stamina, opp_r, opp_c, SAFE_DIST):
        """Evaluate a direction using forecast_turn."""
        rows, cols = board.board_size.r, board.board_size.c
        INV_DIR = {Direction.UP: (-1, 0), Direction.DOWN: (1, 0),
                   Direction.LEFT: (0, -1), Direction.RIGHT: (0, 1)}
        DR = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        ddr, ddc = INV_DIR[direction]
        new_r, new_c = my_r + ddr, my_c + ddc
        if not (0 <= new_r < rows and 0 <= new_c < cols) or board.cells[new_r][new_c].is_wall:
            return -999999, None

        # Safety check
        d_opp = abs(new_r - opp_r) + abs(new_c - opp_c)
        if board.cells[new_r][new_c].owner_parity == self.opp and d_opp <= SAFE_DIST:
            return -999999, None
        if d_opp == 0:
            return -999999, None

        # Build actions
        actions = [Action.Move(direction)]
        reserve = 25 if board.turn_count > 1400 else 10
        budget = stamina - reserve
        spent = 0

        candidates = []
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
            if pcell.owner_parity == player_parity:
                if pcell.hill_id and pcell.hill_id != 0:
                    candidates.append((80, pr, pc))
                continue
            pscore = 200 if (pcell.hill_id and pcell.hill_id != 0) else 100
            candidates.append((pscore, pr, pc))

        candidates.sort(key=lambda x: -x[0])
        for _, pr, pc in candidates:
            if spent + GameConstants.PAINT_STAMINA_COST > budget:
                break
            actions.append(Action.Paint(Location(pr, pc)))
            spent += GameConstants.PAINT_STAMINA_COST

        try:
            sim_board, ok = board.forecast_turn(player_parity, actions)
            if not ok:
                return -999999, actions
            
            me = sim_board.get_player(player_parity)
            territory = 0
            for row in sim_board.cells:
                for cell in row:
                    if cell.owner_parity == player_parity:
                        territory += 1
            hills = len(me.controlled_hills) * 100
            local = 0
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    nr, nc = me.loc.r + dr, me.loc.c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        if sim_board.cells[nr][nc].owner_parity == player_parity:
                            local += 2
            return territory + hills + local + me.stamina * 0.1, actions
        except Exception:
            return -999999, actions

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

        def count_paintable(r, c):
            count = 0
            for dr, dc in DR:
                nr, nc = r + dr, c + dc
                if not valid(nr, nc):
                    continue
                o = cell_owner(nr, nc)
                if o == 0:
                    count += 2
                elif o == player_parity and abs(board.cells[nr][nc].paint_value) < GameConstants.MAX_PAINT_VALUE:
                    count += 1
            return count

        # --- Erase step for hill cells ---
        if stamina >= 55:
            for dr, dc in DR:
                nr, nc = my_r + dr, my_c + dc
                if not valid(nr, nc):
                    continue
                ecell = board.cells[nr][nc]
                if (ecell.hill_id and ecell.hill_id != 0 and
                    ecell.owner_parity == self.opp and
                    board.hills[ecell.hill_id].controller_parity != player_parity):
                    if nr == opp_r and nc == opp_c:
                        continue
                    if stamina >= 65:
                        for dr2, dc2 in DR:
                            off_r, off_c = nr + dr2, nc + dc2
                            if valid(off_r, off_c) and cell_owner(off_r, off_c) != self.opp:
                                if mdist(off_r, off_c, opp_r, opp_c) > SAFE_DIST:
                                    return [
                                        Action.Move(DIR_MAP[(dr, dc)], move_type=MoveType.ERASE),
                                        Action.Move(DIR_MAP[(dr2, dc2)]),
                                        Action.Paint(Location(nr, nc))
                                    ]
                    return [Action.Move(DIR_MAP[(dr, dc)], move_type=MoveType.ERASE)]

        # --- Collision pursuit ---
        for dr, dc in DR:
            nr, nc = my_r + dr, my_c + dc
            if valid(nr, nc) and nr == opp_r and nc == opp_c:
                if cell_owner(nr, nc) != self.opp:
                    return [Action.Move(DIR_MAP[(dr, dc)])]

        # --- BFS for long-range hill targeting ---
        bfs_dir = None
        bfs_priority = -999999
        visited = set()
        visited.add((my_r, my_c))
        queue = deque()

        for dr, dc in DR:
            nr, nc = my_r + dr, my_c + dc
            if not valid(nr, nc):
                continue
            d_opp = mdist(nr, nc, opp_r, opp_c)
            if cell_owner(nr, nc) == self.opp and d_opp <= SAFE_DIST:
                continue
            if d_opp == 0:
                continue
            visited.add((nr, nc))
            queue.append((nr, nc, DIR_MAP[(dr, dc)], 1))

        while queue:
            r, c, first_dir, depth = queue.popleft()
            if depth > 35:
                break
            cell = board.cells[r][c]
            priority = -999999

            if cell.hill_id and cell.hill_id != 0:
                hill = board.hills[cell.hill_id]
                if hill.controller_parity == self.opp:
                    if cell.owner_parity != player_parity:
                        priority = 2500 - depth * 20
                    else:
                        priority = 1200 - depth * 20
                elif hill.controller_parity != player_parity:
                    if cell.owner_parity != player_parity:
                        priority = 2000 - depth * 20
                    else:
                        priority = 1000 - depth * 20
                elif cell.owner_parity == 0:
                    priority = 800 - depth * 15

            if cell.powerup:
                stamina_ratio = max(0, min(1, stamina / 100))
                depth_penalty = 25 if stamina > 50 else int(25 + 75 * (1 - stamina_ratio))
                pup_val = 1500 - depth * depth_penalty
                if stamina < 60:
                    pup_val += 300
                priority = max(priority, pup_val)

            if cell.owner_parity == 0 and priority < -900:
                priority = 900 - depth * 20

            if cell.owner_parity == self.opp and priority < -900:
                d_opp = mdist(r, c, opp_r, opp_c)
                if d_opp > SAFE_DIST:
                    priority = 1100 - depth * 20

            if priority > -900:
                priority += count_paintable(r, c) * 2

            if priority > bfs_priority:
                bfs_priority = priority
                bfs_dir = first_dir

            if depth < 35:
                for dr, dc in DR:
                    nr, nc = r + dr, c + dc
                    if (nr, nc) in visited or not valid(nr, nc):
                        continue
                    visited.add((nr, nc))
                    queue.append((nr, nc, first_dir, depth + 1))

        # --- Direction selection: BFS for hills, parallel forecast for territory ---
        if bfs_priority >= 1500 and bfs_dir is not None:
            # High-priority target (hill/powerup) - use BFS direction
            best_first_dir = bfs_dir
        else:
            # Parallel forecast evaluation for territory expansion
            futures = {}
            for d in Direction.cardinals():
                f = self.executor.submit(
                    self._eval_direction, board, player_parity, d,
                    my_r, my_c, stamina, opp_r, opp_c, SAFE_DIST
                )
                futures[d] = f

            best_eval = -999999
            best_first_dir = bfs_dir
            best_actions = None
            for d, f in futures.items():
                try:
                    score, actions = f.result(timeout=0.05)
                    if score > best_eval:
                        best_eval = score
                        best_first_dir = d
                        best_actions = actions
                except Exception:
                    pass

            if best_actions:
                return best_actions

        if best_first_dir is None:
            for dr, dc in DR:
                nr, nc = my_r + dr, my_c + dc
                if valid(nr, nc):
                    best_first_dir = DIR_MAP[(dr, dc)]
                    break
            if best_first_dir is None:
                return Action.Move(Direction.UP)

        # Build actions with simplified paint scoring
        actions = [Action.Move(best_first_dir)]
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
            if pcell.owner_parity == player_parity:
                if pcell.hill_id and pcell.hill_id != 0:
                    paint_candidates.append((80, pr, pc))
                continue
            pscore = 200 if (pcell.hill_id and pcell.hill_id != 0) else 100
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
