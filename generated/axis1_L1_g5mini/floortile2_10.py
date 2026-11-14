from collections import deque
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Return the tokens of a PDDL fact string, without surrounding parentheses."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    A domain-dependent heuristic for the secret_domain (w133ormt) that estimates
    the number of actions required to satisfy all tile-color goals (yh6q5j8r).
    The domain's actions allow robots to move along directed tile edges, switch
    held color (z7pcmkeu), and paint adjacent tiles (c1qh1p5g / cadwr4wd). The
    heuristic models the main costs: robot movement (one action per move),
    color changes (one action), and painting (one action).

    Assumptions
    -----------
    - Painting a goal (yh6q5j8r tile color) requires the robot to be at a
      particular tile (a "painting position") from which the static relation
      (xr2t1c1d target pos) or (oh0smogc target pos) holds. Those relations
      are extracted from static facts.
    - Movement between tiles follows directed edges determined by the static
      relations present in the domain (xr2t1c1d, oh0smogc, w224no0l, g7ya35is).
      If the static fact is "(pred A B)" then the robot can move from B -> A
      in one move (this matches the domain operators' parameter order).
    - Changing the robot's held color takes exactly one action (z7pcmkeu), and
      can be done at any robot location as long as the robot already holds some
      color (the constructor/instances provide this).
    - Painting a tile itself takes exactly one action (c1qh1p5g or cadwr4wd).
    - Multiple robots may exist. This heuristic assigns painting tasks to robots
      greedily: at each step the robot-task pair with the lowest incremental
      cost is chosen, updating that robot's position and held color. This gives
      a compact but effective sequential estimate that accounts for multiple
      robots working in the same problem.

    Heuristic Initialization
    ------------------------
    The constructor extracts:
    - self.goals: the set of goal fact strings (task.goals).
    - A directed adjacency graph for tile movement (self.graph). For a static
      fact "(pred A B)" where pred is one of the known movement predicates,
      we add an edge B -> A.
    - A mapping from each target tile to the set of robot positions from which
      it can be painted (self.paint_positions). For any static fact
      "(xr2t1c1d T P)" or "(oh0smogc T P)" we record that P is a painting
      position for target tile T.
    - Known colors (self.colors) from static facts "(cdb9op3o COLOR)".

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given search node/state:
    1. If all goal facts are already present in the state, return 0.
    2. Parse the state to find every robot's current tile (from gk9u3vg3 facts)
       and currently held color (from i4u94rl9 facts). Robots without both
       pieces of information are ignored for assignment.
    3. Build the set of remaining tile goals: every (tile, color) pair that
       appears in the task goals but is not yet in the state as a yh6q5j8r fact.
    4. For each remaining tile goal and for each robot, compute the minimal
       movement distance (number of moves) from the robot's current tile to
       any painting position for that target tile using BFS on the directed
       graph. If no painting positions are available (static info missing),
       treat the target as very costly (but finite).
    5. The incremental cost for a robot to fulfill a single tile goal is:
         movement_distance + (1 if robot_color != target_color else 0) + 1
       where +1 is the painting action itself and +1 for a color-switch if
       needed.
    6. Assign tasks greedily: repeatedly pick the robot-target pair with the
       lowest incremental cost, add that cost to the heuristic total, update
       that robot's location to the painting position chosen and update the
       robot's color (painting leaves the robot holding the same color that
       was used to paint). Remove the assigned tile goal and repeat until no
       tile goals remain.
    7. Return the accumulated cost as the heuristic estimate.

    Notes
    -----
    - The heuristic is not admissible (it is optimistic in various respects),
      but it is efficient to compute and captures the major domain-specific
      trade-offs (movement vs. color changes vs. painting).
    - The code is robust to multiple robots and uses BFS caching to avoid
      repeated full searches from identical robot positions.
    """

    def __init__(self, task):
        # Store goal facts (strings)
        self.goals = set(task.goals)
        # Save static facts
        static_facts = task.static

        # Movement-related predicates appearing in the domain. For any static
        # fact "(pred A B)" we allow movement B -> A (matches operator param order).
        self.movement_preds = {"xr2t1c1d", "oh0smogc", "w224no0l", "g7ya35is"}

        # Painting predicates (from which robot positions can paint a target).
        # The painting operators require either xr2t1c1d or oh0smogc to relate
        # the target tile with the robot position.
        self.painting_preds = {"xr2t1c1d", "oh0smogc"}

        # Build directed graph: node -> set(neighbors)
        self.graph = {}
        # Map target tile -> set of robot positions from which it can be painted
        self.paint_positions = {}
        # Known colors from static
        self.colors = set()

        for fact in static_facts:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "cdb9op3o" and len(parts) >= 2:
                self.colors.add(parts[1])

            if len(parts) >= 3:
                a, b = parts[1], parts[2]
                if pred in self.movement_preds:
                    # movement edge: B -> A
                    self.graph.setdefault(b, set()).add(a)
                    # ensure nodes exist in graph for BFS completeness
                    self.graph.setdefault(a, set())
                if pred in self.painting_preds:
                    # painting: if static fact is (pred target pos) then robot
                    # at "pos" can paint "target".
                    self.paint_positions.setdefault(parts[1], set()).add(parts[2])

        # BFS cache: start_tile -> dict(tile -> distance)
        self._bfs_cache = {}

    def _bfs_distances_from(self, start):
        """
        Return a dict mapping tiles reachable from start to their shortest
        distances (in number of movement actions). Cache results per start.
        """
        if start in self._bfs_cache:
            return self._bfs_cache[start]
        distances = {start: 0}
        q = deque([start])
        while q:
            cur = q.popleft()
            for nbr in self.graph.get(cur, ()):
                if nbr not in distances:
                    distances[nbr] = distances[cur] + 1
                    q.append(nbr)
        self._bfs_cache[start] = distances
        return distances

    def __call__(self, node):
        state = node.state

        # If all goals already in the state, heuristic is 0.
        if self.goals <= state:
            return 0

        # Parse current robots: find their positions and held colors.
        # There may be multiple robots; store only those with both position and color.
        robot_pos = {}   # robot -> tile
        robot_color = {}  # robot -> color

        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "gk9u3vg3" and len(parts) >= 3:
                robot, tile = parts[1], parts[2]
                robot_pos[robot] = tile
            elif pred == "i4u94rl9" and len(parts) >= 3:
                robot, color = parts[1], parts[2]
                robot_color[robot] = color

        # Build list of robots that have at least a position and a color.
        robots = []
        for r in set(list(robot_pos.keys()) + list(robot_color.keys())):
            pos = robot_pos.get(r)
            col = robot_color.get(r)
            if pos is not None and col is not None:
                robots.append({"name": r, "pos": pos, "color": col})
        # If no robots with both pieces of information, heuristic cannot proceed use
        # a fallback: count remaining goals times 2 (paint + color change)
        if not robots:
            remaining = [
                g for g in self.goals if g not in state and get_parts(g)[0] == "yh6q5j8r"
            ]
            return len(remaining) * 2

        # Build remaining tile goals: list of (tile, color)
        remaining = []
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "yh6q5j8r":
                tile, color = parts[1], parts[2]
                # If already satisfied, skip
                if goal in state:
                    continue
                remaining.append((tile, color))

        total_cost = 0
        LARGE = 10**6  # large finite penalty for unreachable targets

        # Greedy assignment: repeatedly pick (robot, tile) with minimal incremental cost.
        # Use BFS distances cached.
        while remaining:
            best = None  # (cost, robot_index, tile_index, chosen_paint_pos)
            # Precompute BFS distances for robots' positions
            bfs_cache = {}
            for ri, r in enumerate(robots):
                bfs_cache[ri] = self._bfs_distances_from(r["pos"])

            for ti, (tile, color) in enumerate(remaining):
                possible_paint_positions = self.paint_positions.get(tile, set())
                if not possible_paint_positions:
                    # No known painting positions -> unreachable via painting operators
                    # penalize heavily but finite
                    continue
                for ri, r in enumerate(robots):
                    distances = bfs_cache[ri]
                    # For this tile, find the painting position reachable with minimal distance
                    min_dist = None
                    chosen_pos = None
                    for ppos in possible_paint_positions:
                        dist = distances.get(ppos)
                        if dist is None:
                            # start might equal ppos -> dist 0 handled, else unreachable
                            # but allow moving through graph nodes that may not be in distances
                            continue
                        if min_dist is None or dist < min_dist:
                            min_dist = dist
                            chosen_pos = ppos
                    if min_dist is None:
                        # robot cannot reach any painting position for this tile
                        continue
                    color_cost = 0 if r["color"] == color else 1
                    cost = min_dist + color_cost + 1  # +1 for painting action
                    # Record best choice
                    if best is None or cost < best[0]:
                        best = (cost, ri, ti, chosen_pos, min_dist, color_cost)

            if best is None:
                # No feasible assignments found (some goals can't be painted given static info).
                # Return a large but finite heuristic: remaining count times a heuristic penalty.
                return total_cost + len(remaining) * LARGE

            cost, ri, ti, chosen_pos, min_dist, color_cost = best
            # Apply assignment: update total cost and robot state accordingly.
            total_cost += cost
            # Update robot's position to the painting position (robot doesn't move during paint,
            # but we assume it moved there using min_dist moves).
            robots[ri]["pos"] = chosen_pos
            # If a color switch was used, robot now holds the chosen color (painting leaves robot color)
            robots[ri]["color"] = remaining[ti][1]  # the goal color
            # Remove assigned tile goal
            remaining.pop(ti)

        return int(total_cost)