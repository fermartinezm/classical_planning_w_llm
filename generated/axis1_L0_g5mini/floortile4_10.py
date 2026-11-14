from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic

INF = 10 ** 6


def get_parts(fact):
    """Return list of tokens inside a PDDL fact string '(pred arg1 arg2...)'."""
    return fact[1:-1].split()


def match(fact, *pattern):
    """Return True if fact matches the given pattern (supports '*' wildcards)."""
    parts = get_parts(fact)
    return all(fnmatch(part, pat) for part, pat in zip(parts, pattern))


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Heuristic for the floortile domain that estimates the number of actions
    required to satisfy all painted (...) goal facts. It reasons about robot
    positions, robot-held colors, and the tile adjacency graph to produce a
    fast, informative estimate used by greedy best-first search.

    Assumptions
    -----------
    - Goals are only of the form (painted tile color). Other goal predicates
      are ignored (this matches the typical floortile instances).
    - Robots can change color to any statically available color with a single
      change_color action.
    - Painting a tile requires:
      * a robot to be on a tile adjacent to the target (one of the neighbor
        tiles according to up/down/left/right relations),
      * the robot to hold the required color,
      * the target tile to be clear. This heuristic is optimistic about
        clearing occupied tiles and does not model detailed interactions
        between multiple robots for clearing; it therefore may underestimate
        the true cost (admissibility is not required).
    - Multiple robots can be used; the heuristic greedily assigns painting
      tasks to robots and updates robot positions and held colors to account
      for sequential actions.

    Heuristic Initialization
    ------------------------
    The constructor extracts and stores:
    - self.goals: the set of required painted facts (strings).
    - self.goal_map: mapping tile -> required_color for all painted goals.
    - self.available_colors: set of statically available colors.
    - self.neighbors: mapping tile -> set(neighbor_tiles) computed from the
      static up/down/left/right relations. The tile graph is treated as
      undirected (neighbors both ways) to compute movement distances.
    - self.tiles: set of tiles known from static facts (used for BFS).

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given state the heuristic proceeds as follows:

    1. If all goal painted facts are already present in the state, return 0.

    2. Parse robot information from the state:
       - For each robot collect its current position and currently held color.
       - If a robot lacks a reported color the heuristic assumes it can change
         to any available color (cost 1 later).

    3. Identify which goal-painted facts are still missing. For each missing
       goal (tile T with color C_goal) determine whether T is already painted
       with C_goal (then skip) or already painted with a different color
       (in which case painting T with the goal color is impossible because the
       domain does not provide an unpaint/remove action). In the impossible
       case this heuristic returns a large cost to strongly discourage the
       planner from exploring this branch.

    4. For each remaining (unpainted) target, compute the cost for each robot:
       - Robot must move to some neighbor tile N of target T. Compute the
         shortest path length (number of move actions) from the robot's
         current tile to N using BFS on the tile adjacency graph.
       - If the robot's current color differs from C_goal, add 1 for the
         change_color action.
       - Add 1 for the paint action itself.
       - If no neighbor is reachable, the cost for that robot is INF.

    5. Assign targets to robots greedily:
       - Repeatedly pick the (robot, target, neighbor) pair with the lowest
         estimated cost.
       - Add that cost to the running heuristic sum.
       - Update the chosen robot's position to the neighbor tile N (robots
         stay where they paint from) and update the robot's held color to
         C_goal (because the robot now holds the color after changing).
       - Remove the painted target from the remaining list and repeat until
         all target goals are assigned.

    6. Return the total summed cost as the heuristic value.

    Notes about approximations and admissibility:
    - The heuristic is intentionally optimistic about tile clearance and
      interactions between robots (it does not plan to move other robots out
      of the way explicitly). It therefore may underestimate the true cost
      (admissibility is not guaranteed), but stays finite for solvable
      states and is efficiently computable.
    """

    def __init__(self, task):
        """
        Extract static facts and goal information.

        task.static contains static facts (frozenset of strings).
        task.goals contains the goal facts (set or frozenset of strings).
        """
        self.goals = set(task.goals)

        # Map tile -> required color for all painted goals.
        self.goal_map = {}
        for g in self.goals:
            parts = get_parts(g)
            if parts and parts[0] == "painted" and len(parts) >= 3:
                tile = parts[1]
                color = parts[2]
                self.goal_map[tile] = color

        # Extract available colors from static.
        self.available_colors = {
            get_parts(fact)[1]
            for fact in task.static
            if match(fact, "available-color", "*")
        }

        # Build neighbors graph (treat edges as undirected for movement).
        self.neighbors = {}
        tiles = set()
        for fact in task.static:
            parts = get_parts(fact)
            if parts:
                pred = parts[0]
                if pred in ("up", "down", "left", "right") and len(parts) == 3:
                    a, b = parts[1], parts[2]
                    tiles.add(a)
                    tiles.add(b)
                    # Add both directions to be robust (static facts often include both,
                    # but we ensure connectivity).
                    self.neighbors.setdefault(a, set()).add(b)
                    self.neighbors.setdefault(b, set()).add(a)

        self.tiles = tiles

    def _bfs_distances(self, start):
        """
        Compute shortest-path distances (number of move actions) from start tile
        to all other tiles in the adjacency graph using BFS.

        Returns dict tile -> distance (int). If start not in graph, returns empty dict.
        """
        if start not in self.neighbors:
            return {}
        dist = {start: 0}
        q = deque([start])
        while q:
            v = q.popleft()
            for nb in self.neighbors.get(v, ()):
                if nb not in dist:
                    dist[nb] = dist[v] + 1
                    q.append(nb)
        return dist

    def __call__(self, node):
        """Compute and return the heuristic estimate (integer)."""
        state = node.state

        # If all goals already satisfied, heuristic is 0.
        if self.goals <= state:
            return 0

        # Parse painted facts from the state.
        painted = {}
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "painted" and len(parts) >= 3:
                painted[parts[1]] = parts[2]

        # If any goal tile is already painted with a different color, the
        # domain cannot change a painted tile to a different color (no unpaint),
        # so treat as impossible (very large cost).
        for tile, req_color in self.goal_map.items():
            if tile in painted and painted[tile] != req_color:
                # large penalty to avoid exploring this branch
                return INF

        # Build list of targets that still need painting: tiles whose required
        # painted fact is not yet present.
        targets = []
        for tile, req_color in self.goal_map.items():
            fact_str = f"(painted {tile} {req_color})"
            if fact_str not in state:
                targets.append((tile, req_color))

        # Parse robots: positions and held colors.
        robots_pos = {}   # robot -> tile
        robots_color = {}  # robot -> color (may be None)
        robots = set()
        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue
            if parts[0] == "robot-at" and len(parts) >= 3:
                robot = parts[1]
                tile = parts[2]
                robots.add(robot)
                robots_pos[robot] = tile
            elif parts[0] == "robot-has" and len(parts) >= 3:
                robot = parts[1]
                color = parts[2]
                robots.add(robot)
                robots_color[robot] = color

        # Ensure every robot has a position and a color (if missing, set None).
        for r in robots:
            robots_pos.setdefault(r, None)
            robots_color.setdefault(r, None)

        # If there are no robots but there are painting goals, it's unsolvable.
        if not robots and targets:
            return INF

        # Precompute BFS distances for each robot's start position.
        robot_dists = {}
        for r, pos in robots_pos.items():
            if pos is None:
                robot_dists[r] = {}
            else:
                robot_dists[r] = self._bfs_distances(pos)

        # Greedy assignment of targets to robots, updating robot positions/colors.
        total_cost = 0
        remaining = list(targets)
        # Copy mutable robot states for updates.
        cur_pos = dict(robots_pos)
        cur_color = dict(robots_color)

        while remaining:
            best = None  # tuple (cost, robot, target_tile, target_color, neighbor_tile)
            # Evaluate cost of assigning each remaining target to each robot.
            for (tile, goal_color) in remaining:
                # The neighbor tiles from which a robot can paint 'tile':
                neighs = list(self.neighbors.get(tile, []))
                if not neighs:
                    # No place to stand and paint this tile => unreachable
                    continue
                for r in robots:
                    start = cur_pos.get(r)
                    # distances from robot's current position to every tile
                    # recompute if robot position changed.
                    dists = self._bfs_distances(start) if start is not None else {}
                    # Determine minimal distance to any neighbor tile.
                    best_neighbor = None
                    best_dist = INF
                    for n in neighs:
                        d = dists.get(n, INF)
                        if d < best_dist:
                            best_dist = d
                            best_neighbor = n
                    if best_dist >= INF:
                        continue
                    # cost components:
                    move_cost = best_dist  # number of move actions to neighbor
                    change_cost = 0 if cur_color.get(r) == goal_color else 1
                    paint_cost = 1
                    cost = move_cost + change_cost + paint_cost
                    # Prefer smaller cost; tie-breaker: smaller move_cost then robot name
                    candidate = (cost, move_cost, r, tile, goal_color, best_neighbor)
                    if best is None or candidate < best:
                        best = candidate

            if best is None:
                # Some target unreachable by any robot
                return INF

            cost, _, chosen_robot, tile, goal_color, neighbor = best
            total_cost += cost

            # Update chosen robot's state: position becomes neighbor; color becomes goal_color
            cur_pos[chosen_robot] = neighbor
            cur_color[chosen_robot] = goal_color

            # Remove target from remaining
            remaining = [(t, c) for (t, c) in remaining if t != tile]

        return total_cost