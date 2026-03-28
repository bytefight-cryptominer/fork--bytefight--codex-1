from collections.abc import Callable, Iterable
from collections import deque
from typing import Union, List, Tuple, Optional

from game import *


class PlayerController:
    """
    BFS expansion agent with collision avoidance, hill capture, and stamina management.
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

        # Precompute cell ownership grid for speed
        own_grid = [[0] * cols for _ in range(rows)]
        wall_grid = [[False] * cols for _ in range(rows)]
        for r in range(rows):
            for c in range(cols):
                cell = board.cells[r][c]
                own_grid[r][c] = cell.owner_parity
                wall_grid[r][c] = cell.is_wall

        my_loc = me.loc
        opp_loc = opp.loc

        # Manhattan distance to opponent
        def mdist(r, c, tr, tc):
            return abs(r - tr) + abs(c - tc)

        opp_dist = mdist(my_loc.r, my_loc.c, opp_loc.r, opp_loc.c)

        # Safety: distance threshold for collision avoidance
        SAFE_DIST = 6

        # Direction map
        DIRS = [
            (Direction.UP, -1, 0),
            (Direction.DOWN, 1, 0),
            (Direction.LEFT, 0, -1),
            (Direction.RIGHT, 0, 1),
        ]

        def is_valid(r, c):
            return 0 <= r < rows and 0 <= c < cols and not wall_grid[r][c]

        def is_enemy_cell(r, c):
            return own_grid[r][c] == self.opp

        def is_my_cell(r, c):
            return own_grid[r][c] == player_parity

        def is_neutral(r, c):
            return own_grid[r][c] == 0

        # BFS from current position to find best expansion target
        # Prioritize: hills > neutral cells > reinforcing own territory
        # Avoid: enemy cells near opponent (collision risk)

        best_dir = None
        best_score = -999999

        for d_enum, dr, dc in DIRS:
            nr, nc = my_loc.r + dr, my_loc.c + dc
            if not is_valid(nr, nc):
                continue

            # Hard safety: never step on enemy cell when close to opponent
            dist_to_opp = mdist(nr, nc, opp_loc.r, opp_loc.c)
            if is_enemy_cell(nr, nc) and dist_to_opp <= SAFE_DIST:
                continue

            # Even on neutral, avoid going adjacent to opponent on their territory
            if dist_to_opp <= 1 and is_enemy_cell(nr, nc):
                continue

            score = 0.0

            # Prefer cells that aren't ours (expansion)
            if is_neutral(nr, nc):
                score += 50
            elif is_enemy_cell(nr, nc):
                score += 30  # stepping on enemy removes a layer
            elif is_my_cell(nr, nc):
                score += 0

            # Hill bonus: strongly prefer moving toward/onto hill cells
            cell = board.cells[nr][nc]
            if cell.hill_id and cell.hill_id != 0:
                hill = board.hills[cell.hill_id]
                if hill.controller_parity != player_parity:
                    score += 200  # uncaptured hill cell
                else:
                    score += 20  # already captured

            # BFS lookahead: count expansion opportunities within 4 steps
            expansion_count = self._bfs_expansion_score(
                nr, nc, rows, cols, own_grid, wall_grid, player_parity,
                opp_loc.r, opp_loc.c, SAFE_DIST
            )
            score += expansion_count * 3

            # Bias away from opponent when close
            if opp_dist <= 8:
                score += dist_to_opp * 5

            # Slight randomness to avoid predictable patterns
            import random
            score += random.random() * 2

            if score > best_score:
                best_score = score
                best_dir = d_enum

        if best_dir is None:
            # Fallback: pick any valid direction
            for d_enum, dr, dc in DIRS:
                nr, nc = my_loc.r + dr, my_loc.c + dc
                if is_valid(nr, nc):
                    best_dir = d_enum
                    break

        if best_dir is None:
            return Action.Move(Direction.UP)

        actions: List[Action.Move | Action.Paint] = []

        # Paint before moving: paint adjacent neutral/friendly cells
        paint_actions = self._get_paint_actions(board, player_parity, me)

        # Move
        actions.append(Action.Move(best_dir))

        # Paint after moving
        dr_map = {Direction.UP: (-1, 0), Direction.DOWN: (1, 0),
                  Direction.LEFT: (0, -1), Direction.RIGHT: (0, 1)}
        dr, dc = dr_map[best_dir]
        new_r, new_c = my_loc.r + dr, my_loc.c + dc

        if is_valid(new_r, new_c):
            post_paint = self._get_paint_actions_from(
                board, player_parity, new_r, new_c, me.stamina - 15  # rough estimate after move
            )
            actions.extend(post_paint)

        # Add pre-move paints
        actions = paint_actions + actions

        return actions

    def _bfs_expansion_score(self, sr, sc, rows, cols, own_grid, wall_grid,
                              parity, opp_r, opp_c, safe_dist, max_depth=4):
        """Count reachable non-owned cells within max_depth BFS steps."""
        visited = set()
        visited.add((sr, sc))
        queue = deque([(sr, sc, 0)])
        count = 0
        opp_parity = -parity

        while queue:
            r, c, depth = queue.popleft()
            if depth >= max_depth:
                continue

            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if (nr, nc) in visited:
                    continue
                if not (0 <= nr < rows and 0 <= nc < cols):
                    continue
                if wall_grid[nr][nc]:
                    continue

                visited.add((nr, nc))

                # Don't expand through enemy territory when near opponent
                d_opp = abs(nr - opp_r) + abs(nc - opp_c)
                if own_grid[nr][nc] == opp_parity and d_opp <= safe_dist:
                    continue

                if own_grid[nr][nc] != parity:
                    count += 1

                queue.append((nr, nc, depth + 1))

        return count

    def _get_paint_actions(self, board, player_parity, player) -> List[Action.Paint]:
        """Get paint actions for cells adjacent to current position."""
        actions = []
        loc = player.loc
        rows, cols = board.board_size.r, board.board_size.c
        stamina = player.stamina

        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = loc.r + dr, loc.c + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            cell = board.cells[nr][nc]
            if cell.is_wall:
                continue
            if cell.beacon_parity == player_parity:
                continue
            # Only paint neutral or own cells (can't paint enemy)
            if cell.owner_parity != player_parity and cell.owner_parity != 0:
                continue
            # Don't paint if already at max
            if cell.owner_parity == player_parity and abs(cell.paint_value) >= GameConstants.MAX_PAINT_VALUE:
                continue
            if stamina < GameConstants.PAINT_STAMINA_COST:
                break

            # Prioritize painting hill cells
            priority = 0
            if cell.hill_id and cell.hill_id != 0:
                priority = 1

            actions.append((priority, Action.Paint(Location(nr, nc))))
            stamina -= GameConstants.PAINT_STAMINA_COST

        # Sort by priority (hills first)
        actions.sort(key=lambda x: -x[0])

        # Keep stamina reserve for movement
        min_reserve = 30
        result = []
        cost = 0
        for _, action in actions:
            if player.stamina - cost - GameConstants.PAINT_STAMINA_COST < min_reserve:
                break
            result.append(action)
            cost += GameConstants.PAINT_STAMINA_COST

        return result

    def _get_paint_actions_from(self, board, player_parity, r, c, remaining_stamina) -> List[Action.Paint]:
        """Get paint actions for cells adjacent to a position."""
        actions = []
        rows, cols = board.board_size.r, board.board_size.c
        min_reserve = 20

        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            cell = board.cells[nr][nc]
            if cell.is_wall:
                continue
            if cell.beacon_parity == player_parity:
                continue
            if cell.owner_parity != player_parity and cell.owner_parity != 0:
                continue
            if cell.owner_parity == player_parity and abs(cell.paint_value) >= GameConstants.MAX_PAINT_VALUE:
                continue
            if remaining_stamina < GameConstants.PAINT_STAMINA_COST + min_reserve:
                break

            actions.append(Action.Paint(Location(nr, nc)))
            remaining_stamina -= GameConstants.PAINT_STAMINA_COST

        return actions

    def commentate(self, board: Board, player_parity: int, time_left: Callable) -> str:
        return ""
