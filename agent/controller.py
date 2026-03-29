from __future__ import annotations

import math
import random
import time
from collections import deque
from collections.abc import Callable, Iterable
from typing import List, Union

from game import *


class PlayerController:
    """
    v134: v133 heuristic controller + bounded root Monte Carlo search in tactical windows.
    """

    DR = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    DIR_MAP = {
        (-1, 0): Direction.UP,
        (1, 0): Direction.DOWN,
        (0, -1): Direction.LEFT,
        (0, 1): Direction.RIGHT,
    }
    INV_DIR = {
        Direction.UP: (-1, 0),
        Direction.DOWN: (1, 0),
        Direction.LEFT: (0, -1),
        Direction.RIGHT: (0, 1),
    }

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
        tactical_turn = self._select_mcts_turn(board, player_parity, time_left)
        if tactical_turn is not None:
            return tactical_turn
        return self._heuristic_turn(board, player_parity)

    def commentate(self, board: Board, player_parity: int, time_left: Callable) -> str:
        return ""

    def _select_mcts_turn(
        self, board: Board, player_parity: int, time_left: Callable
    ) -> List[Action.Move | Action.Paint] | None:
        budget = self._mcts_budget(board, player_parity, time_left)
        if budget <= 0.0:
            return None

        candidates = self._generate_candidate_turns(board, player_parity)
        if len(candidates) <= 1:
            return None

        valid_turns: List[List[Action.Move | Action.Paint]] = []
        successor_boards: List[Board] = []
        immediate_scores: List[float] = []
        for turn in candidates:
            next_board, ok = board.forecast_turn(player_parity, turn)
            if not ok:
                continue
            valid_turns.append(turn)
            successor_boards.append(next_board)
            immediate_scores.append(self._evaluate_board(next_board, player_parity))

        if len(valid_turns) <= 1:
            return valid_turns[0] if valid_turns else None

        me = board.get_player(player_parity)
        opp = board.get_opponent(player_parity)
        seed = (
            board.turn_count * 10007
            + me.loc.r * 911
            + me.loc.c * 353
            + opp.loc.r * 199
            + opp.loc.c * 131
            + int(me.stamina) * 17
            + (1 if player_parity > 0 else 2)
        )
        rng = random.Random(seed)

        visits = [1 for _ in valid_turns]
        totals = list(immediate_scores)
        end_time = time.perf_counter() + budget
        rollout_count = 0
        max_rollouts = 36 if budget >= 0.08 else 24 if budget >= 0.05 else 16

        # Root-only Monte Carlo search over tactical turn variants.
        while time.perf_counter() < end_time and rollout_count < max_rollouts:
            total_visits = sum(visits)
            best_idx = 0
            best_ucb = float("-inf")
            for idx in range(len(valid_turns)):
                mean_value = totals[idx] / visits[idx]
                explore = 1.35 * math.sqrt(math.log(total_visits + 1.0) / visits[idx])
                ucb = mean_value + explore
                if ucb > best_ucb:
                    best_ucb = ucb
                    best_idx = idx

            value = self._rollout_value(
                successor_boards[best_idx],
                player_parity,
                rng,
            )
            totals[best_idx] += value
            visits[best_idx] += 1
            rollout_count += 1

        best_idx = max(
            range(len(valid_turns)),
            key=lambda idx: (totals[idx] / visits[idx], immediate_scores[idx], visits[idx]),
        )
        return valid_turns[best_idx]

    def _rollout_value(self, board: Board, root_parity: int, rng: random.Random) -> float:
        state = board
        actor = -root_parity
        for depth in range(3):
            winner = state.get_winner()
            if winner is not None:
                break

            turn = self._sample_rollout_turn(state, actor, rng)
            next_state, ok = state.forecast_turn(actor, turn)
            if not ok:
                return -250000.0 if actor == root_parity else 250000.0

            state = next_state
            actor *= -1

            if depth == 1:
                winner = state.get_winner()
                if winner is not None:
                    break

        return self._evaluate_board(state, root_parity)

    def _sample_rollout_turn(
        self, board: Board, player_parity: int, rng: random.Random
    ) -> List[Action.Move | Action.Paint]:
        candidates = self._generate_candidate_turns(board, player_parity)[:4]
        if not candidates:
            return self._heuristic_turn(board, player_parity)
        if len(candidates) == 1:
            return candidates[0]

        scored: List[tuple[float, List[Action.Move | Action.Paint]]] = []
        for turn in candidates:
            next_board, ok = board.forecast_turn(player_parity, turn)
            if not ok:
                continue
            scored.append((self._evaluate_board(next_board, player_parity), turn))

        if not scored:
            return self._heuristic_turn(board, player_parity)

        scored.sort(key=lambda item: item[0], reverse=True)
        if len(scored) == 1 or rng.random() < 0.72:
            return scored[0][1]
        if len(scored) == 2 or rng.random() < 0.75:
            return scored[1][1]
        return scored[min(2, len(scored) - 1)][1]

    def _mcts_budget(self, board: Board, player_parity: int, time_left: Callable) -> float:
        try:
            remaining = float(time_left())
        except Exception:
            remaining = GameConstants.PLAY_TIME_LIMIT

        if remaining < 10.0:
            return 0.0

        me = board.get_player(player_parity)
        opp = board.get_opponent(player_parity)
        distance = self._mdist(me.loc.r, me.loc.c, opp.loc.r, opp.loc.c)

        tactical = False
        if distance <= 7:
            tactical = True
        elif self._generate_erase_turns(board, player_parity):
            tactical = True
        else:
            for hill in board.hills.values():
                if hill.controller_parity == 0:
                    continue
                for loc in hill.cells:
                    if self._mdist(me.loc.r, me.loc.c, loc.r, loc.c) <= 4:
                        tactical = True
                        break
                if tactical:
                    break

        if not tactical:
            return 0.0

        if remaining > 120.0:
            budget = 0.085
        elif remaining > 60.0:
            budget = 0.06
        else:
            budget = 0.04

        if distance <= 4:
            budget += 0.03
        if me.stamina >= 80:
            budget += 0.02
        if self._generate_erase_turns(board, player_parity):
            budget += 0.02

        return min(0.14, budget)

    def _generate_candidate_turns(
        self, board: Board, player_parity: int
    ) -> List[List[Action.Move | Action.Paint]]:
        candidates: List[List[Action.Move | Action.Paint]] = []
        seen: set[tuple] = set()

        def add(turn: Action.Move | Action.Paint | Iterable[Action.Move | Action.Paint]) -> None:
            normalized = self._normalize_turn(turn)
            if not normalized:
                return
            key = self._serialize_turn(normalized)
            if key in seen:
                return
            seen.add(key)
            candidates.append(normalized)

        add(self._heuristic_turn(board, player_parity))

        for _, turn in self._generate_erase_turns(board, player_parity)[:3]:
            add(turn)

        for _, first_dir in self._rank_regular_moves(board, player_parity)[:3]:
            add(self._build_regular_turn(board, player_parity, first_dir, max_paints=None))
            add(self._build_regular_turn(board, player_parity, first_dir, max_paints=1))
            add(self._build_regular_turn(board, player_parity, first_dir, max_paints=0))

        return candidates[:9]

    def _heuristic_turn(
        self, board: Board, player_parity: int
    ) -> List[Action.Move | Action.Paint]:
        me = board.get_player(player_parity)
        opp = board.get_opponent(player_parity)
        rows, cols = board.board_size.r, board.board_size.c
        my_r, my_c = me.loc.r, me.loc.c
        opp_r, opp_c = opp.loc.r, opp.loc.c
        stamina = me.stamina
        opp_stamina = opp.stamina
        safe_dist = self._safe_dist(stamina, opp_stamina)

        erase_turns = self._generate_erase_turns(board, player_parity)
        if erase_turns:
            return erase_turns[0][1]

        best_first_dir = None
        best_priority = -999999.0
        visited = {(my_r, my_c)}
        queue = deque()

        for dr, dc in self.DR:
            nr, nc = my_r + dr, my_c + dc
            if not self._valid(board, nr, nc):
                continue
            d_opp = self._mdist(nr, nc, opp_r, opp_c)

            if nr == opp_r and nc == opp_c:
                if self._cell_owner(board, nr, nc) == self.opp:
                    continue
                return [Action.Move(self.DIR_MAP[(dr, dc)])]

            if self._cell_owner(board, nr, nc) == self.opp and d_opp <= safe_dist:
                continue

            visited.add((nr, nc))
            queue.append((nr, nc, self.DIR_MAP[(dr, dc)], 1))

        while queue:
            r, c, first_dir, depth = queue.popleft()
            if depth > 35:
                break

            priority = self._cell_priority(
                board,
                player_parity,
                r,
                c,
                depth,
                safe_dist,
            )
            if priority > best_priority:
                best_priority = priority
                best_first_dir = first_dir

            if depth < 35:
                for dr, dc in self.DR:
                    nr, nc = r + dr, c + dc
                    if (nr, nc) in visited or not self._valid(board, nr, nc):
                        continue
                    visited.add((nr, nc))
                    queue.append((nr, nc, first_dir, depth + 1))

        if best_first_dir is None:
            for dr, dc in self.DR:
                nr, nc = my_r + dr, my_c + dc
                if self._valid(board, nr, nc):
                    best_first_dir = self.DIR_MAP[(dr, dc)]
                    break
            if best_first_dir is None:
                return [Action.Move(Direction.UP)]

        return self._build_regular_turn(board, player_parity, best_first_dir, max_paints=None)

    def _rank_regular_moves(self, board: Board, player_parity: int) -> List[tuple[float, Direction]]:
        me = board.get_player(player_parity)
        opp = board.get_opponent(player_parity)
        my_r, my_c = me.loc.r, me.loc.c
        opp_r, opp_c = opp.loc.r, opp.loc.c
        safe_dist = self._safe_dist(me.stamina, opp.stamina)

        visited = {(my_r, my_c)}
        queue = deque()
        best_by_dir: dict[Direction, float] = {}

        for dr, dc in self.DR:
            nr, nc = my_r + dr, my_c + dc
            if not self._valid(board, nr, nc):
                continue
            first_dir = self.DIR_MAP[(dr, dc)]
            if nr == opp_r and nc == opp_c and self._cell_owner(board, nr, nc) != self.opp:
                best_by_dir[first_dir] = 100000.0
                continue
            if self._cell_owner(board, nr, nc) == self.opp and self._mdist(nr, nc, opp_r, opp_c) <= safe_dist:
                continue
            visited.add((nr, nc))
            queue.append((nr, nc, first_dir, 1))
            best_by_dir[first_dir] = max(
                best_by_dir.get(first_dir, float("-inf")),
                self._cell_priority(board, player_parity, nr, nc, 1, safe_dist),
            )

        while queue:
            r, c, first_dir, depth = queue.popleft()
            if depth > 35:
                break
            priority = self._cell_priority(board, player_parity, r, c, depth, safe_dist)
            if priority > best_by_dir.get(first_dir, float("-inf")):
                best_by_dir[first_dir] = priority

            if depth < 35:
                for dr, dc in self.DR:
                    nr, nc = r + dr, c + dc
                    if (nr, nc) in visited or not self._valid(board, nr, nc):
                        continue
                    visited.add((nr, nc))
                    queue.append((nr, nc, first_dir, depth + 1))

        ranked = [(score, direction) for direction, score in best_by_dir.items()]
        ranked.sort(key=lambda item: item[0], reverse=True)
        return ranked

    def _generate_erase_turns(
        self, board: Board, player_parity: int
    ) -> List[tuple[float, List[Action.Move | Action.Paint]]]:
        me = board.get_player(player_parity)
        opp = board.get_opponent(player_parity)
        my_r, my_c = me.loc.r, me.loc.c
        opp_r, opp_c = opp.loc.r, opp.loc.c
        stamina = me.stamina
        safe_dist = self._safe_dist(stamina, opp.stamina)

        if stamina < 50:
            return []

        options: List[tuple[float, List[Action.Move | Action.Paint]]] = []
        reserve = 25 if board.turn_count > 1400 else 10

        for dr, dc in self.DR:
            nr, nc = my_r + dr, my_c + dc
            if not self._valid(board, nr, nc):
                continue
            cell = board.cells[nr][nc]
            if (
                not cell.hill_id
                or cell.hill_id == 0
                or cell.owner_parity != self.opp
                or board.hills[cell.hill_id].controller_parity != self.opp
            ):
                continue
            if nr == opp_r and nc == opp_c:
                continue

            base_score = 2200.0 + len(board.hills[cell.hill_id].cells) * 5.0
            if stamina >= 65:
                hill_id = cell.hill_id
                exit_choices: List[tuple[float, tuple[int, int, int, int]]] = []
                for dr2, dc2 in self.DR:
                    off_r, off_c = nr + dr2, nc + dc2
                    if not self._valid(board, off_r, off_c):
                        continue
                    if self._cell_owner(board, off_r, off_c) == self.opp:
                        continue
                    if self._mdist(off_r, off_c, opp_r, opp_c) <= safe_dist:
                        continue
                    score = self._count_paintable(board, player_parity, off_r, off_c)
                    score += self._hill_followup_paint_count(
                        board,
                        hill_id,
                        nr,
                        nc,
                        off_r,
                        off_c,
                    ) * 2
                    exit_choices.append((score, (dr2, dc2, off_r, off_c)))

                exit_choices.sort(key=lambda item: item[0], reverse=True)
                for exit_score, (dr2, dc2, off_r, off_c) in exit_choices[:2]:
                    combo: List[Action.Move | Action.Paint] = [
                        Action.Move(self.DIR_MAP[(dr, dc)], move_type=MoveType.ERASE)
                    ]
                    extra_budget = stamina - 65 - reserve
                    for dr3, dc3 in self.DR:
                        if extra_budget < GameConstants.PAINT_STAMINA_COST:
                            break
                        pr, pc = nr + dr3, nc + dc3
                        if not self._valid(board, pr, pc) or (pr == off_r and pc == off_c):
                            continue
                        pcell = board.cells[pr][pc]
                        if pcell.hill_id == hill_id and pcell.owner_parity == 0:
                            combo.append(Action.Paint(Location(pr, pc)))
                            extra_budget -= GameConstants.PAINT_STAMINA_COST
                    combo.append(Action.Move(self.DIR_MAP[(dr2, dc2)]))
                    combo.append(Action.Paint(Location(nr, nc)))
                    options.append((base_score + exit_score * 12.0, combo))

            options.append(
                (
                    base_score,
                    [Action.Move(self.DIR_MAP[(dr, dc)], move_type=MoveType.ERASE)],
                )
            )

        options.sort(key=lambda item: item[0], reverse=True)
        return options

    def _build_regular_turn(
        self,
        board: Board,
        player_parity: int,
        first_dir: Direction,
        max_paints: int | None,
    ) -> List[Action.Move | Action.Paint]:
        me = board.get_player(player_parity)
        rows, cols = board.board_size.r, board.board_size.c
        ddr, ddc = self.INV_DIR[first_dir]
        new_r, new_c = me.loc.r + ddr, me.loc.c + ddc
        if not self._valid(board, new_r, new_c):
            return [Action.Move(first_dir)]

        actions: List[Action.Move | Action.Paint] = [Action.Move(first_dir)]
        reserve = 25 if board.turn_count > 1400 else 10
        paint_budget = me.stamina - reserve
        paint_spent = 0
        paints_used = 0

        paint_candidates: List[tuple[int, int, int]] = []
        for dr, dc in self.DR:
            pr, pc = new_r + dr, new_c + dc
            if not (0 <= pr < rows and 0 <= pc < cols):
                continue
            pcell = board.cells[pr][pc]
            if pcell.is_wall or pcell.beacon_parity == player_parity:
                continue
            if pcell.owner_parity not in (player_parity, 0):
                continue
            if (
                pcell.owner_parity == player_parity
                and abs(pcell.paint_value) >= GameConstants.MAX_PAINT_VALUE
            ):
                continue

            if pcell.owner_parity == player_parity:
                if pcell.hill_id and pcell.hill_id != 0:
                    paint_candidates.append((80, pr, pc))
                continue

            pscore = 200 if (pcell.hill_id and pcell.hill_id != 0) else 100
            if (pr - new_r, pc - new_c) == (ddr, ddc):
                pscore += 15
            paint_candidates.append((pscore, pr, pc))

        paint_candidates.sort(key=lambda item: -item[0])
        for _, pr, pc in paint_candidates:
            if paint_spent + GameConstants.PAINT_STAMINA_COST > paint_budget:
                break
            if max_paints is not None and paints_used >= max_paints:
                break
            actions.append(Action.Paint(Location(pr, pc)))
            paint_spent += GameConstants.PAINT_STAMINA_COST
            paints_used += 1

        return actions

    def _cell_priority(
        self,
        board: Board,
        player_parity: int,
        r: int,
        c: int,
        depth: int,
        safe_dist: int,
    ) -> float:
        opp = board.get_opponent(player_parity)
        opp_r, opp_c = opp.loc.r, opp.loc.c
        cell = board.cells[r][c]
        priority = -999999.0

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
            me = board.get_player(player_parity)
            stamina_ratio = max(0.0, min(1.0, me.stamina / 100.0))
            depth_penalty = 25 if me.stamina > 50 else int(25 + 75 * (1 - stamina_ratio))
            pup_val = 1500 - depth * depth_penalty
            if me.stamina < 60:
                pup_val += 300
            priority = max(priority, pup_val)

        if cell.owner_parity == 0 and priority < -900:
            priority = 900 - depth * 20

        if cell.owner_parity == self.opp and priority < -900:
            d_opp = self._mdist(r, c, opp_r, opp_c)
            if d_opp > safe_dist:
                priority = 1100 - depth * 20

        if priority > -900:
            priority += self._count_paintable(board, player_parity, r, c) * 3

        return priority

    def _evaluate_board(self, board: Board, player_parity: int) -> float:
        winner = board.get_winner()
        if winner is not None:
            result, _ = winner
            if (result == Result.PLAYER_1 and player_parity > 0) or (
                result == Result.PLAYER_2 and player_parity < 0
            ):
                return 1_000_000.0 - board.turn_count
            if result == Result.TIE:
                return 0.0
            return -1_000_000.0 + board.turn_count

        me = board.get_player(player_parity)
        opp = board.get_opponent(player_parity)

        territory_diff = board.get_territory_count(player_parity) - board.get_territory_count(-player_parity)
        hill_diff = len(me.controlled_hills) - len(opp.controlled_hills)
        local_diff = board._count_adjacent_friendly(player_parity) - board._count_adjacent_friendly(-player_parity)
        stamina_diff = me.stamina - opp.stamina
        max_stamina_diff = me.max_stamina - opp.max_stamina
        distance = self._mdist(me.loc.r, me.loc.c, opp.loc.r, opp.loc.c)

        hill_pressure = 0.0
        for hill in board.hills.values():
            diff = hill.get_control_diff(player_parity)
            threshold = math.ceil(len(hill.cells) * GameConstants.HILL_CONTROL_THRESHOLD)
            if hill.controller_parity == player_parity:
                hill_pressure += 18.0 * diff
            elif hill.controller_parity == -player_parity:
                hill_pressure += 26.0 * diff
            else:
                hill_pressure += 12.0 * diff
            if abs(diff) < threshold:
                hill_pressure += 10.0 * diff

        powerup_bonus = 0.0
        nearest_power = None
        for r in range(board.board_size.r):
            for c in range(board.board_size.c):
                if board.cells[r][c].powerup:
                    dist = self._mdist(me.loc.r, me.loc.c, r, c)
                    nearest_power = dist if nearest_power is None else min(nearest_power, dist)
        if nearest_power is not None:
            powerup_bonus = max(0.0, 60.0 - nearest_power * 10.0)

        collision_pressure = 0.0
        if distance == 1:
            target_owner = self._cell_owner(board, opp.loc.r, opp.loc.c)
            if target_owner != -player_parity:
                collision_pressure += 180.0
            else:
                collision_pressure -= 120.0

        return (
            hill_diff * 420.0
            + territory_diff * 11.0
            + local_diff * 18.0
            + stamina_diff * 4.5
            + max_stamina_diff * 4.0
            + hill_pressure
            + powerup_bonus
            + collision_pressure
            - distance * 2.5
        )

    def _hill_followup_paint_count(
        self,
        board: Board,
        hill_id: int,
        center_r: int,
        center_c: int,
        exit_r: int,
        exit_c: int,
    ) -> int:
        count = 0
        for dr, dc in self.DR:
            pr, pc = center_r + dr, center_c + dc
            if not self._valid(board, pr, pc):
                continue
            if pr == exit_r and pc == exit_c:
                continue
            pcell = board.cells[pr][pc]
            if pcell.hill_id == hill_id and pcell.owner_parity == 0:
                count += 1
        return count

    def _normalize_turn(
        self, turn: Action.Move | Action.Paint | Iterable[Action.Move | Action.Paint]
    ) -> List[Action.Move | Action.Paint]:
        if isinstance(turn, (Action.Move, Action.Paint)):
            return [turn]
        return list(turn)

    def _serialize_turn(self, turn: Iterable[Action.Move | Action.Paint]) -> tuple:
        items = []
        for action in turn:
            if isinstance(action, Action.Move):
                beacon = None
                if action.beacon_target is not None:
                    beacon = (action.beacon_target.r, action.beacon_target.c)
                items.append(
                    (
                        "M",
                        None if action.direction is None else action.direction.name,
                        int(action.move_type),
                        action.place_beacon,
                        beacon,
                    )
                )
            else:
                items.append(("P", action.location.r, action.location.c))
        return tuple(items)

    def _valid(self, board: Board, r: int, c: int) -> bool:
        return 0 <= r < board.board_size.r and 0 <= c < board.board_size.c and not board.cells[r][c].is_wall

    def _cell_owner(self, board: Board, r: int, c: int) -> int:
        return board.cells[r][c].owner_parity

    def _count_paintable(self, board: Board, player_parity: int, r: int, c: int) -> int:
        count = 0
        for dr, dc in self.DR:
            nr, nc = r + dr, c + dc
            if not self._valid(board, nr, nc):
                continue
            owner = self._cell_owner(board, nr, nc)
            if owner == 0:
                count += 2
            elif owner == player_parity and abs(board.cells[nr][nc].paint_value) < GameConstants.MAX_PAINT_VALUE:
                count += 1
        return count

    def _safe_dist(self, stamina: int, opp_stamina: int) -> int:
        stamina_diff = stamina - opp_stamina
        if stamina_diff > 30:
            return 3
        if stamina_diff < -30:
            return 6
        return 5

    def _mdist(self, r1: int, c1: int, r2: int, c2: int) -> int:
        return abs(r1 - r2) + abs(c1 - c2)
