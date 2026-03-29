from collections.abc import Callable, Iterable
from collections import deque
from typing import Union, List

from game import *


class PlayerController:
    """
    v111: Minimax agent - considers opponent's best response.
    Uses BFS for hills, minimax for territory expansion.
    """

    def __init__(self, player_parity: int, time_left: Callable):
        self.parity = player_parity
        self.opp = -player_parity

    def bid(self, board, player_parity, time_left):
        return 0

    def _build_actions(self, board, parity, direction, pos_r, pos_c, stam):
        rows, cols = board.board_size.r, board.board_size.c
        INV = {Direction.UP: (-1,0), Direction.DOWN: (1,0), Direction.LEFT: (0,-1), Direction.RIGHT: (0,1)}
        DR = [(-1,0),(1,0),(0,-1),(0,1)]
        ddr, ddc = INV[direction]
        nr, nc = pos_r + ddr, pos_c + ddc
        if not (0 <= nr < rows and 0 <= nc < cols) or board.cells[nr][nc].is_wall:
            return None
        actions = [Action.Move(direction)]
        budget = stam - 10
        spent = 0
        for dr, dc in DR:
            pr, pc = nr+dr, nc+dc
            if not (0 <= pr < rows and 0 <= pc < cols): continue
            pcell = board.cells[pr][pc]
            if pcell.is_wall or pcell.beacon_parity == parity: continue
            if pcell.owner_parity != parity and pcell.owner_parity != 0: continue
            if pcell.owner_parity == parity and abs(pcell.paint_value) >= GameConstants.MAX_PAINT_VALUE: continue
            if pcell.owner_parity == parity:
                if not (pcell.hill_id and pcell.hill_id != 0): continue
            if spent + GameConstants.PAINT_STAMINA_COST > budget: break
            actions.append(Action.Paint(Location(pr, pc)))
            spent += GameConstants.PAINT_STAMINA_COST
        return actions

    def _eval(self, board, parity):
        me = board.get_player(parity)
        rows, cols = board.board_size.r, board.board_size.c
        territory = sum(1 for row in board.cells for cell in row if cell.owner_parity == parity)
        hills = len(me.controlled_hills) * 100
        local = sum(2 for dr in range(-2,3) for dc in range(-2,3)
                    if 0 <= me.loc.r+dr < rows and 0 <= me.loc.c+dc < cols
                    and board.cells[me.loc.r+dr][me.loc.c+dc].owner_parity == parity)
        return territory + hills + local + me.stamina * 0.1

    def play(self, board, player_parity, time_left):
        me = board.get_player(player_parity)
        opp = board.get_opponent(player_parity)
        rows, cols = board.board_size.r, board.board_size.c
        my_r, my_c = me.loc.r, me.loc.c
        opp_r, opp_c = opp.loc.r, opp.loc.c
        stamina = me.stamina
        opp_stamina = opp.stamina

        DR = [(-1,0),(1,0),(0,-1),(0,1)]
        DIR_MAP = {(-1,0): Direction.UP, (1,0): Direction.DOWN, (0,-1): Direction.LEFT, (0,1): Direction.RIGHT}
        INV_DIR = {Direction.UP: (-1,0), Direction.DOWN: (1,0), Direction.LEFT: (0,-1), Direction.RIGHT: (0,1)}

        def valid(r, c): return 0 <= r < rows and 0 <= c < cols and not board.cells[r][c].is_wall
        def mdist(r1,c1,r2,c2): return abs(r1-r2)+abs(c1-c2)
        def cell_owner(r, c): return board.cells[r][c].owner_parity

        sd = stamina - opp_stamina
        SAFE_DIST = 3 if sd > 30 else (6 if sd < -30 else 5)

        def count_paintable(r, c):
            count = 0
            for dr, dc in DR:
                nr2, nc2 = r+dr, c+dc
                if not valid(nr2, nc2): continue
                o = cell_owner(nr2, nc2)
                if o == 0: count += 2
                elif o == player_parity and abs(board.cells[nr2][nc2].paint_value) < GameConstants.MAX_PAINT_VALUE: count += 1
            return count

        # Erase uncaptured hill cells
        if stamina >= 55:
            for dr, dc in DR:
                nr, nc = my_r+dr, my_c+dc
                if not valid(nr, nc): continue
                ecell = board.cells[nr][nc]
                if (ecell.hill_id and ecell.hill_id != 0 and ecell.owner_parity == self.opp and
                    board.hills[ecell.hill_id].controller_parity != player_parity):
                    if nr == opp_r and nc == opp_c: continue
                    if stamina >= 65:
                        for dr2, dc2 in DR:
                            or2, oc2 = nr+dr2, nc+dc2
                            if valid(or2, oc2) and cell_owner(or2, oc2) != self.opp and mdist(or2, oc2, opp_r, opp_c) > SAFE_DIST:
                                return [Action.Move(DIR_MAP[(dr,dc)], move_type=MoveType.ERASE),
                                        Action.Move(DIR_MAP[(dr2,dc2)]), Action.Paint(Location(nr, nc))]
                    return [Action.Move(DIR_MAP[(dr,dc)], move_type=MoveType.ERASE)]

        # Collision pursuit
        for dr, dc in DR:
            nr, nc = my_r+dr, my_c+dc
            if valid(nr, nc) and nr == opp_r and nc == opp_c and cell_owner(nr, nc) != self.opp:
                return [Action.Move(DIR_MAP[(dr,dc)])]

        # BFS
        bfs_dir = None; bfs_priority = -999999
        visited = {(my_r, my_c)}; queue = deque()
        for dr, dc in DR:
            nr, nc = my_r+dr, my_c+dc
            if not valid(nr, nc): continue
            d = mdist(nr, nc, opp_r, opp_c)
            if cell_owner(nr, nc) == self.opp and d <= SAFE_DIST: continue
            if d == 0: continue
            visited.add((nr, nc)); queue.append((nr, nc, DIR_MAP[(dr,dc)], 1))

        while queue:
            r, c, fd, depth = queue.popleft()
            if depth > 35: break
            cell = board.cells[r][c]; p = -999999
            if cell.hill_id and cell.hill_id != 0:
                h = board.hills[cell.hill_id]
                if h.controller_parity == self.opp:
                    p = (2500 if cell.owner_parity != player_parity else 1200) - depth*20
                elif h.controller_parity != player_parity:
                    p = (2000 if cell.owner_parity != player_parity else 1000) - depth*20
                elif cell.owner_parity == 0: p = 800 - depth*15
            if cell.powerup:
                sr = max(0, min(1, stamina/100))
                dp = 25 if stamina > 50 else int(25+75*(1-sr))
                pv = 1500 - depth*dp + (300 if stamina < 60 else 0)
                p = max(p, pv)
            if cell.owner_parity == 0 and p < -900: p = 900 - depth*20
            if cell.owner_parity == self.opp and p < -900:
                if mdist(r, c, opp_r, opp_c) > SAFE_DIST: p = 1100 - depth*20
            if p > -900: p += count_paintable(r, c) * 2
            if p > bfs_priority: bfs_priority = p; bfs_dir = fd
            if depth < 35:
                for dr, dc in DR:
                    nr, nc = r+dr, c+dc
                    if (nr, nc) in visited or not valid(nr, nc): continue
                    visited.add((nr, nc)); queue.append((nr, nc, fd, depth+1))

        # Minimax for low-priority moves (territory expansion)
        if bfs_priority < 1500 and time_left() > 30:
            best_mm = -999999; best_mm_actions = None
            for d in Direction.cardinals():
                actions = self._build_actions(board, player_parity, d, my_r, my_c, stamina)
                if not actions: continue
                ddr, ddc = INV_DIR[d]
                nr, nc = my_r+ddr, my_c+ddc
                if cell_owner(nr, nc) == self.opp and mdist(nr, nc, opp_r, opp_c) <= SAFE_DIST: continue
                if mdist(nr, nc, opp_r, opp_c) == 0: continue
                try:
                    s1, ok1 = board.forecast_turn(player_parity, actions)
                    if not ok1: continue
                    so = s1.get_opponent(player_parity)
                    worst = 999999
                    for od in Direction.cardinals():
                        oa = self._build_actions(s1, self.opp, od, so.loc.r, so.loc.c, so.stamina)
                        if not oa: continue
                        try:
                            s2, ok2 = s1.forecast_turn(self.opp, oa)
                            if ok2:
                                sc = self._eval(s2, player_parity) - self._eval(s2, self.opp) * 0.3
                                worst = min(worst, sc)
                        except: pass
                    if worst == 999999: worst = self._eval(s1, player_parity)
                    if worst > best_mm: best_mm = worst; best_mm_actions = actions
                except: pass
            if best_mm_actions: return best_mm_actions

        best_first_dir = bfs_dir
        if not best_first_dir:
            for dr, dc in DR:
                nr, nc = my_r+dr, my_c+dc
                if valid(nr, nc): best_first_dir = DIR_MAP[(dr,dc)]; break
            if not best_first_dir: return Action.Move(Direction.UP)

        actions = [Action.Move(best_first_dir)]
        ddr, ddc = INV_DIR[best_first_dir]
        new_r, new_c = my_r+ddr, my_c+ddc
        if not valid(new_r, new_c): return actions

        reserve = 25 if board.turn_count > 1400 else 10
        budget = stamina - reserve; spent = 0; cands = []
        for dr, dc in DR:
            pr, pc = new_r+dr, new_c+dc
            if not (0 <= pr < rows and 0 <= pc < cols): continue
            pcell = board.cells[pr][pc]
            if pcell.is_wall or pcell.beacon_parity == player_parity: continue
            if pcell.owner_parity != player_parity and pcell.owner_parity != 0: continue
            if pcell.owner_parity == player_parity and abs(pcell.paint_value) >= GameConstants.MAX_PAINT_VALUE: continue
            if pcell.owner_parity == player_parity:
                if pcell.hill_id and pcell.hill_id != 0: cands.append((80, pr, pc))
                continue
            cands.append((200 if (pcell.hill_id and pcell.hill_id != 0) else 100, pr, pc))
        cands.sort(key=lambda x: -x[0])
        for _, pr, pc in cands:
            if spent + GameConstants.PAINT_STAMINA_COST > budget: break
            actions.append(Action.Paint(Location(pr, pc)))
            spent += GameConstants.PAINT_STAMINA_COST
        return actions

    def commentate(self, board, player_parity, time_left):
        return ""
