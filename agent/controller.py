from collections.abc import Callable, Iterable
from collections import deque
from itertools import combinations
from typing import Union, List

from game import *


class PlayerController:
    """
    v133: v126 + smarter erase exit direction (prefer more paintable neighbors).
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

        # Adaptive safety based on stamina advantage
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

        def local_control_count(sim_board, loc):
            total = 0
            for r_off in range(-2, 3):
                for c_off in range(-2, 3):
                    nr, nc = loc.r + r_off, loc.c + c_off
                    if 0 <= nr < rows and 0 <= nc < cols:
                        if sim_board.cells[nr][nc].owner_parity == player_parity:
                            total += 1
            return total

        def owned_hill_delta(sim_board):
            total = 0
            for r in range(rows):
                for c in range(cols):
                    cell = sim_board.cells[r][c]
                    if not cell.hill_id:
                        continue
                    owner = cell.owner_parity
                    if owner == player_parity:
                        total += 1
                    elif owner == self.opp:
                        total -= 1
            return total

        def hill_control_delta(sim_board):
            total = 0
            for hill in sim_board.hills.values():
                if hill.controller_parity == player_parity:
                    total += 1
                elif hill.controller_parity == self.opp:
                    total -= 1
            return total

        def simulate_turn_prefix(actions):
            sim_board = board.get_copy()
            for action in actions:
                if not sim_board.apply_action(player_parity, action):
                    return None
                if sim_board.get_player(player_parity).is_dead():
                    return None
                if sim_board.get_opponent(player_parity).is_dead():
                    break
            return sim_board

        def score_local_paint_plan(sim_board):
            sim_me = sim_board.get_player(player_parity)
            sim_opp = sim_board.get_opponent(player_parity)
            if sim_opp.is_dead():
                return 10**9
            return (
                hill_control_delta(sim_board) * 4000
                + owned_hill_delta(sim_board) * 250
                + local_control_count(sim_board, sim_me.loc) * 120
                + sim_board.get_territory_count(player_parity) * 15
                + sim_me.stamina * 4
            )

        # --- Erase step for hill cells with opponent paint ---
        # Erase opponent-painted hill cells: both attacking (uncaptured) and defending (ours)
        if stamina >= 50:  # 40 erase + 10 buffer
            for dr, dc in DR:
                nr, nc = my_r + dr, my_c + dc
                if not valid(nr, nc):
                    continue
                ecell = board.cells[nr][nc]
                if (ecell.hill_id and ecell.hill_id != 0 and
                    ecell.owner_parity == self.opp and
                    board.hills[ecell.hill_id].controller_parity == self.opp):
                    if nr == opp_r and nc == opp_c:
                        continue
                    # Erase + move + paint combo when possible
                    if stamina >= 65:
                        # Find best exit direction (most paintable neighbors)
                        best_exit = None
                        best_exit_score = -1
                        for dr2, dc2 in DR:
                            off_r, off_c = nr + dr2, nc + dc2
                            if valid(off_r, off_c) and cell_owner(off_r, off_c) != self.opp:
                                if mdist(off_r, off_c, opp_r, opp_c) > SAFE_DIST:
                                    score = count_paintable(off_r, off_c)
                                    if score > best_exit_score:
                                        best_exit_score = score
                                        best_exit = (dr2, dc2, off_r, off_c)
                        if best_exit:
                            dr2, dc2, off_r, off_c = best_exit
                            combo = [Action.Move(DIR_MAP[(dr, dc)], move_type=MoveType.ERASE)]
                            reserve = 25 if board.turn_count > 1400 else 10
                            extra_budget = stamina - 65 - reserve
                            hill_id = ecell.hill_id
                            for dr3, dc3 in DR:
                                if extra_budget < GameConstants.PAINT_STAMINA_COST:
                                    break
                                pr, pc = nr + dr3, nc + dc3
                                if not valid(pr, pc) or (pr == off_r and pc == off_c):
                                    continue
                                pcell = board.cells[pr][pc]
                                if pcell.is_wall:
                                    continue
                                if (pcell.hill_id == hill_id and
                                    pcell.owner_parity == 0):
                                    combo.append(Action.Paint(Location(pr, pc)))
                                    extra_budget -= GameConstants.PAINT_STAMINA_COST
                            combo.append(Action.Move(DIR_MAP[(dr2, dc2)]))
                            combo.append(Action.Paint(Location(nr, nc)))
                            return combo
                    return [Action.Move(DIR_MAP[(dr, dc)], move_type=MoveType.ERASE)]

        # --- BFS ---
        best_first_dir = None
        best_priority = -999999

        visited = set()
        visited.add((my_r, my_c))
        queue = deque()

        for dr, dc in DR:
            nr, nc = my_r + dr, my_c + dc
            if not valid(nr, nc):
                continue
            d_opp = mdist(nr, nc, opp_r, opp_c)

            # Collision pursuit: mover wins on neutral
            if nr == opp_r and nc == opp_c:
                if cell_owner(nr, nc) == self.opp:
                    continue
                return Action.Move(DIR_MAP[(dr, dc)])

            if cell_owner(nr, nc) == self.opp and d_opp <= SAFE_DIST:
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
                # Dynamic depth penalty: when low stamina, only chase NEARBY powerups
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
                    priority = 1100 - depth * 20  # higher enemy territory priority

            if priority > -900:
                priority += count_paintable(r, c) * 3

            if priority > best_priority:
                best_priority = priority
                best_first_dir = first_dir

            if depth < 35:
                for dr, dc in DR:
                    nr, nc = r + dr, c + dc
                    if (nr, nc) in visited or not valid(nr, nc):
                        continue
                    # Only block opponent cells near opponent at depth 1 (safety)
                    # At deeper depths, allow traversal for reachability
                    # (we won't walk there directly; BFS tracks first_dir from safe seeds)
                    visited.add((nr, nc))
                    queue.append((nr, nc, first_dir, depth + 1))

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

        # Late game conservation
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
                if pcell.hill_id and pcell.hill_id != 0 and abs(pcell.paint_value) < GameConstants.MAX_PAINT_VALUE:
                    paint_candidates.append((80, pr, pc))
                continue
            pscore = 200 if (pcell.hill_id and pcell.hill_id != 0) else 100
            # Forward bias: prefer painting in the movement direction
            if (pr - new_r, pc - new_c) == (ddr, ddc):
                pscore += 15
            paint_candidates.append((pscore, pr, pc))

        paint_candidates.sort(key=lambda x: -x[0])
        max_paints = paint_budget // GameConstants.PAINT_STAMINA_COST
        if len(paint_candidates) >= 2 and max_paints >= 1:
            best_subset = []
            best_subset_score = None
            candidate_locs = [Location(pr, pc) for _, pr, pc in paint_candidates[:4]]
            subset_limit = min(max_paints, len(candidate_locs))

            for subset_size in range(subset_limit + 1):
                for subset in combinations(candidate_locs, subset_size):
                    trial_actions = [actions[0]]
                    for loc in subset:
                        trial_actions.append(Action.Paint(loc))
                    sim_board = simulate_turn_prefix(trial_actions)
                    if sim_board is None:
                        continue
                    score = score_local_paint_plan(sim_board)
                    if best_subset_score is None or score > best_subset_score:
                        best_subset_score = score
                        best_subset = list(subset)

            for loc in best_subset:
                actions.append(Action.Paint(loc))
        else:
            for _, pr, pc in paint_candidates:
                if paint_spent + GameConstants.PAINT_STAMINA_COST > paint_budget:
                    break
                actions.append(Action.Paint(Location(pr, pc)))
                paint_spent += GameConstants.PAINT_STAMINA_COST

        return actions

    def commentate(self, board: Board, player_parity: int, time_left: Callable) -> str:
        return ""
