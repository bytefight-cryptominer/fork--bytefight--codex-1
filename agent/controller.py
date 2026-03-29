import math
from collections import deque
from collections.abc import Callable, Iterable
from typing import List, Union

from game import *


class PlayerController:
    """
    v134: v133 + exact hill-race scoring from hill control counts and distance margins.
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

        drs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        dir_map = {
            (-1, 0): Direction.UP,
            (1, 0): Direction.DOWN,
            (0, -1): Direction.LEFT,
            (0, 1): Direction.RIGHT,
        }
        inv_dir = {
            Direction.UP: (-1, 0),
            Direction.DOWN: (1, 0),
            Direction.LEFT: (0, -1),
            Direction.RIGHT: (0, 1),
        }

        def valid(r, c):
            return 0 <= r < rows and 0 <= c < cols and not board.cells[r][c].is_wall

        def mdist(r1, c1, r2, c2):
            return abs(r1 - r2) + abs(c1 - c2)

        def cell_owner(r, c):
            return board.cells[r][c].owner_parity

        total_hills = len(board.hills)
        domination_goal = (3 * total_hills + 3) // 4 if total_hills else 0
        my_hills = len(me.controlled_hills)
        opp_hills = len(opp.controlled_hills)
        domination_attack = domination_goal > 0 and my_hills + 1 >= domination_goal
        domination_defend = domination_goal > 0 and opp_hills + 1 >= domination_goal

        # Adaptive safety based on stamina advantage.
        stamina_diff = stamina - opp_stamina
        if stamina_diff > 30:
            safe_dist = 3
        elif stamina_diff < -30:
            safe_dist = 6
        else:
            safe_dist = 5

        def count_paintable(r, c):
            count = 0
            for dr, dc in drs:
                nr, nc = r + dr, c + dc
                if not valid(nr, nc):
                    continue
                owner = cell_owner(nr, nc)
                if owner == 0:
                    count += 2
                elif owner == player_parity and abs(board.cells[nr][nc].paint_value) < GameConstants.MAX_PAINT_VALUE:
                    count += 1
            return count

        def hill_control_counts(hill):
            if player_parity > 0:
                my_control = hill.control_positive
                opp_control = -hill.control_negative
            else:
                my_control = -hill.control_negative
                opp_control = hill.control_positive
            threshold = math.ceil(len(hill.cells) * GameConstants.HILL_CONTROL_THRESHOLD)
            return my_control, opp_control, threshold

        def hill_race_bonus(hill, depth, r, c):
            my_control, opp_control, threshold = hill_control_counts(hill)
            my_need = max(0, threshold - my_control)
            opp_need = max(0, threshold - opp_control)
            dist_margin = mdist(opp_r, opp_c, r, c) - depth
            count_bonus = (opp_need - my_need) * 28
            dist_bonus = max(-120, min(180, dist_margin * 26))

            bonus = count_bonus + dist_bonus
            if domination_attack and hill.controller_parity != player_parity:
                bonus += 140
            if domination_defend and hill.controller_parity == self.opp:
                bonus += 180
            if my_need <= 1 and hill.controller_parity != player_parity:
                bonus += 120
            if opp_need <= 1 and hill.controller_parity == self.opp:
                bonus += 120
            return bonus

        # Erase opponent-painted hill cells.
        if stamina >= 50:
            for dr, dc in drs:
                nr, nc = my_r + dr, my_c + dc
                if not valid(nr, nc):
                    continue
                ecell = board.cells[nr][nc]
                if (
                    ecell.hill_id
                    and ecell.hill_id != 0
                    and ecell.owner_parity == self.opp
                    and board.hills[ecell.hill_id].controller_parity == self.opp
                ):
                    if nr == opp_r and nc == opp_c:
                        continue
                    if stamina >= 65:
                        best_exit = None
                        best_exit_score = -1e9
                        hill = board.hills[ecell.hill_id]
                        for dr2, dc2 in drs:
                            off_r, off_c = nr + dr2, nc + dc2
                            if valid(off_r, off_c) and cell_owner(off_r, off_c) != self.opp:
                                if mdist(off_r, off_c, opp_r, opp_c) > safe_dist:
                                    score = count_paintable(off_r, off_c) * 2 + hill_race_bonus(hill, 2, off_r, off_c)
                                    if score > best_exit_score:
                                        best_exit_score = score
                                        best_exit = (dr2, dc2, off_r, off_c)
                        if best_exit:
                            dr2, dc2, off_r, off_c = best_exit
                            combo = [Action.Move(dir_map[(dr, dc)], move_type=MoveType.ERASE)]
                            reserve = 25 if board.turn_count > 1400 else 10
                            extra_budget = stamina - 65 - reserve
                            hill_id = ecell.hill_id
                            for dr3, dc3 in drs:
                                if extra_budget < GameConstants.PAINT_STAMINA_COST:
                                    break
                                pr, pc = nr + dr3, nc + dc3
                                if not valid(pr, pc) or (pr == off_r and pc == off_c):
                                    continue
                                pcell = board.cells[pr][pc]
                                if pcell.is_wall:
                                    continue
                                if pcell.hill_id == hill_id and pcell.owner_parity == 0:
                                    combo.append(Action.Paint(Location(pr, pc)))
                                    extra_budget -= GameConstants.PAINT_STAMINA_COST
                            combo.append(Action.Move(dir_map[(dr2, dc2)]))
                            combo.append(Action.Paint(Location(nr, nc)))
                            return combo
                    return [Action.Move(dir_map[(dr, dc)], move_type=MoveType.ERASE)]

        best_first_dir = None
        best_priority = -999999
        visited = {(my_r, my_c)}
        queue = deque()

        for dr, dc in drs:
            nr, nc = my_r + dr, my_c + dc
            if not valid(nr, nc):
                continue
            d_opp = mdist(nr, nc, opp_r, opp_c)

            # Collision pursuit: mover wins on neutral.
            if nr == opp_r and nc == opp_c:
                if cell_owner(nr, nc) == self.opp:
                    continue
                return Action.Move(dir_map[(dr, dc)])

            if cell_owner(nr, nc) == self.opp and d_opp <= safe_dist:
                continue

            visited.add((nr, nc))
            queue.append((nr, nc, dir_map[(dr, dc)], 1))

        while queue:
            r, c, first_dir, depth = queue.popleft()
            if depth > 35:
                break

            cell = board.cells[r][c]
            priority = -999999

            if cell.hill_id and cell.hill_id != 0:
                hill = board.hills[cell.hill_id]
                race_bonus = hill_race_bonus(hill, depth, r, c)
                if hill.controller_parity == self.opp:
                    if cell.owner_parity != player_parity:
                        priority = 2500 - depth * 20 + race_bonus
                    else:
                        priority = 1200 - depth * 20 + race_bonus
                elif hill.controller_parity != player_parity:
                    if cell.owner_parity != player_parity:
                        priority = 2000 - depth * 20 + race_bonus
                    else:
                        priority = 1000 - depth * 20 + race_bonus
                elif cell.owner_parity == 0:
                    priority = 800 - depth * 15 + race_bonus

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
                if d_opp > safe_dist:
                    priority = 1100 - depth * 20

            if priority > -900:
                priority += count_paintable(r, c) * 3

            if priority > best_priority:
                best_priority = priority
                best_first_dir = first_dir

            if depth < 35:
                for dr, dc in drs:
                    nr, nc = r + dr, c + dc
                    if (nr, nc) in visited or not valid(nr, nc):
                        continue
                    visited.add((nr, nc))
                    queue.append((nr, nc, first_dir, depth + 1))

        if best_first_dir is None:
            for dr, dc in drs:
                nr, nc = my_r + dr, my_c + dc
                if valid(nr, nc):
                    best_first_dir = dir_map[(dr, dc)]
                    break
            if best_first_dir is None:
                return Action.Move(Direction.UP)

        actions: List = [Action.Move(best_first_dir)]
        ddr, ddc = inv_dir[best_first_dir]
        new_r, new_c = my_r + ddr, my_c + ddc
        if not valid(new_r, new_c):
            return actions

        reserve = 25 if board.turn_count > 1400 else 10
        paint_budget = stamina - reserve
        paint_spent = 0

        paint_candidates = []
        for dr, dc in drs:
            pr, pc = new_r + dr, new_c + dc
            if not (0 <= pr < rows and 0 <= pc < cols):
                continue
            pcell = board.cells[pr][pc]
            if pcell.is_wall or pcell.beacon_parity == player_parity:
                continue
            if pcell.owner_parity not in (player_parity, 0):
                continue
            if pcell.owner_parity == player_parity and abs(pcell.paint_value) >= GameConstants.MAX_PAINT_VALUE:
                continue

            if pcell.owner_parity == player_parity:
                if pcell.hill_id and pcell.hill_id != 0 and abs(pcell.paint_value) < GameConstants.MAX_PAINT_VALUE:
                    pscore = 80 + hill_race_bonus(board.hills[pcell.hill_id], 1, pr, pc) // 8
                    paint_candidates.append((pscore, pr, pc))
                continue

            pscore = 200 if (pcell.hill_id and pcell.hill_id != 0) else 100
            if pcell.hill_id and pcell.hill_id != 0:
                pscore += hill_race_bonus(board.hills[pcell.hill_id], 1, pr, pc) // 6
            if (pr - new_r, pc - new_c) == (ddr, ddc):
                pscore += 15
            paint_candidates.append((pscore, pr, pc))

        paint_candidates.sort(key=lambda item: -item[0])
        for _, pr, pc in paint_candidates:
            if paint_spent + GameConstants.PAINT_STAMINA_COST > paint_budget:
                break
            actions.append(Action.Paint(Location(pr, pc)))
            paint_spent += GameConstants.PAINT_STAMINA_COST

        return actions

    def commentate(self, board: Board, player_parity: int, time_left: Callable) -> str:
        return ""
