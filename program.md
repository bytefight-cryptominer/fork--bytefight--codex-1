# ByteFight Agent

Build a competitive territory-control bot for ByteFight 2026: Paint, evaluated by Glicko-1 rating against a frozen opponent pool.

## Setup

1. **Read the in-scope files**:
   - `agent/controller.py` — your bot's logic. You modify this (and can add modules to `agent/`).
   - `engine/` — the game engine and API. Do not modify.
   - `eval/eval.sh` — submits your agent for evaluation. Do not modify.
   - `docs/rules.md` — complete game rules reference.
2. **Verify the engine exists**: Check that `engine/game/board.py` and `engine/game/game_structs.py` exist.
3. **Initialize results.tsv**: Create `results.tsv` with just the header row.
4. **Run baseline**: `bash eval/eval.sh` to establish the starting score.

## The game

ByteFight is a turn-based 2-player strategy game on a grid (8x8 to 32x32). You control a single agent that moves, paints territory, captures hills, and manages stamina. Win by: stamina collapse, collision kill, or 75%+ hill domination. Matches last up to 2000 turns (1000 per player) with 180 seconds compute time.

Key mechanics:
- **Stamina**: base regen 5 + territory/8 + local control (2 per cell in 5x5). Paint costs 15, erase costs 40, extra moves cost 10/20/30/...
- **Hills**: control 50%+ of hill cells to capture. Each hill gives +40 max stamina.
- **Collisions**: cell owner wins. On neutral cell, the mover wins. Erase steps can't initiate collisions.
- **Beacons**: teleport anchors requiring 2/3 local control to deploy. Paint decays on use.
- **Tiebreak**: hills > cells > compute time remaining.

See `docs/rules.md` for the full specification.

## The agent interface

Your `agent/controller.py` must define a `PlayerController` class:

```python
class PlayerController:
    def __init__(self, player_parity: int, time_left: Callable):
        ...

    def bid(self, board: Board, player_parity: int, time_left: Callable) -> int:
        ...

    def play(self, board: Board, player_parity: int, time_left: Callable) -> Action | Iterable[Action]:
        ...
```

- `player_parity`: 1 or -1 (your identity)
- `board`: full game state (see `engine/game/board.py`)
- `time_left()`: callable returning remaining compute seconds
- `bid`: return stamina to wager for first move (0-100)
- `play`: return one or more `Action.Move` and/or `Action.Paint` per turn

Key imports: `from game import *` gives you `Board`, `Action`, `Direction`, `MoveType`, `Location`, `GameConstants`.

## Experimentation

**What you CAN do:**
- Modify any files inside `agent/` — rewrite `controller.py`, add helper modules, use precomputed data structures.
- Use `numpy`, `numba`, `torch`, `cython` (pre-installed).
- Use up to 3 CPU threads.

**What you CANNOT do:**
- Modify `eval/`, `engine/`, or `docs/`.
- Call external APIs, network services, or LLMs from your controller.
- Exceed 1.5GB RAM.

**The goal: maximize rating.** Rating is a Glicko-1 score from playing against a frozen pool of opponents. Higher is better. The baseline (random mover) scores very low — there is enormous room for improvement.

**Simplicity criterion**: All else being equal, simpler is better.

## Strategy hints

Strong agents typically use:
- BFS-based territory expansion (paint after moving)
- Hill capture and defense
- Collision avoidance (never step on enemy-controlled cells near the opponent)
- Stamina management (don't overspend on paint or multi-moves)
- Positional awareness (bias movement away from opponent when close)

Approaches that tend NOT to work:
- MCTS/lookahead (simulation diverges from actual opponent behavior)
- Pure defensive play (falls behind on territory)

## Output format

```
---
rating:           1523.4
ci:               12.3
wins:             45
losses:           15
ties:             2
games:            62
```

## Logging results

Log each experiment to `results.tsv` (tab-separated):

```
commit	rating	ci	wins	losses	games	status	description
a1b2c3d	1500.0	25.0	30	30	60	keep	baseline random mover
b2c3d4e	1580.0	20.0	40	20	60	keep	added BFS expansion
```

## The experiment loop

LOOP FOREVER:

1. **THINK** — decide what to try next. Review results.tsv. Read game replays if available. Consider what strategies beat the current weaknesses.
2. Modify files in `agent/` with your experimental idea.
3. git commit
4. Run the experiment: `bash eval/eval.sh > run.log 2>&1`
5. Read the results: `grep "^rating:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` for the stack trace and attempt a fix.
7. Record the results in results.tsv (do not commit results.tsv).
8. If rating improved, keep the git commit. If equal or worse, `git reset --hard HEAD~1`.

**Timeout**: If a run exceeds 30 minutes, kill it and treat it as a failure. Eval time varies with server load (~10 min alone, longer when many agents are evaluating concurrently).

**NEVER STOP**: Once the loop begins, do NOT pause to ask the human. You are autonomous. The loop runs until interrupted.
