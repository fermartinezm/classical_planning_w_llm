from collections import deque, defaultdict
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Remove surrounding parentheses and split a PDDL fact into parts."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the secret_domain (w133ormt). The domain
    models robots that can move on a grid of tiles, change the color they
    'hold', and paint adjacent tiles with their current color. Goals are
    predicates of the form (yh6q5j8r tile color) - i.e., a given tile must
    have a given color.

    This heuristic estimates the number of actions required to reach the
    goal by summing, for each goal tile not yet satisfied, an estimate of:
      - the minimum number of moves for some robot to reach a tile adjacent
        to the goal tile (so it can paint it),
      - one painting action,
      - optionally one color-change action if that robot does not already
        hold the desired color.

    The heuristic treats movement cost as the shortest-path distance in a
    directed adjacency graph derived from the static adjacency predicates:
    xr2t1c1d, oh0smogc, w224no0l, g7ya35is. These static facts are used to
    determine from which tile(s) a target tile can be painted (i.e., the
    set of adjacent "painting positions") and what moves are possible in the
    static structure (we ignore dynamic tile-occupancy constraints for the
    purpose of distance computation to keep the heuristic efficient).

    Assumptions
    -----------
    - Painting a target tile T requires the robot to be on some tile S such
      that a static adjacency fact of the form (PRED T S) exists (this is
      directly mirrored from the domain operator preconditions).
    - Changing the robot's held color is possible from any location and
      costs exactly one action.
    - Painting a tile costs exactly one action.
    - Moving from one tile to a neighboring tile costs exactly one action.
    - The heuristic ignores dynamic constraints related to d7ekk7vb (free)
      when computing movement distances (i.e., it assumes tiles are
      traversable along static adjacency edges). This makes the heuristic
      optimistic regarding reachability but keeps it fast to compute.
    - If a tile is already painted with a different color in the current
      state (i.e., (yh6q5j8r tile othercolor) is present and othercolor !=
      desired color), the heuristic treats that goal as effectively
      unreachable and returns a large cost for that goal (so the planner
      will prefer states avoiding that situation).

    Heuristic Initialization
    ------------------------
    In the constructor we extract and pre-process:
      - self.goals: the set of goal facts from the task
      - self.static: static facts from the task
      - adjacency graph (directed) as self.edges mapping: for each static
        adjacency fact (PRED A B) we add an edge B -> A (matching the domain
        operator preconditions which allow movement from B to A).
      - paint_positions: for each tile T, the set of tiles S such that a
        static adjacency fact (PRED T S) exists; these S are valid positions
        from which T can be painted.
      - set of recognized adjacency predicate names used in the domain.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given state:
    1. If the state's facts already satisfy all goals, return 0.
    2. Parse robot facts: collect all robots with their current tile
       (gk9u3vg3) and their held colors (i4u94rl9). There can be multiple
       robots; the heuristic will consider the best robot for painting a
       particular tile.
    3. Parse which tiles are already painted (yh6q5j8r facts). If a tile
       required by the goal is already painted with a different color, treat
       that goal as unreachable and assign a large cost.
    4. For each unsatisfied goal (tile T, color C):
       a. Identify all painting positions S (from static paint_positions[T]).
          If none exist, treat as unreachable (large cost).
       b. For each robot R with location L and current color CR compute:
          - dist = shortest-path distance (in static graph) from L to any S.
            If unreachable, treat dist as infinite for that robot.
          - color_change = 0 if CR == C else 1
          - cost_R = dist + 1 (paint) + color_change
       c. Use the minimal cost_R over all robots as the cost to satisfy that
          particular goal tile.
    5. Sum minimal per-tile costs across all unsatisfied goal tiles and
       return that sum. This sum gives a domain-aware estimate that
       encourages the search to move robots closer to tiles that still need
       painting and to change colors only when required.

    Note on admissibility and planner behavior
    ------------------------------------------
    - The heuristic is not guaranteed to be admissible (it may under- or
      over-estimate because it ignores dynamic occupancy constraints and
      because it treats each tile independently even if multiple tiles could
      be painted during a single traversal with fewer moves).
    - It is designed to be informative and quick to compute to guide
      greedy best-first search towards promising states.
    """

    # adjacency predicate names used in domain to derive edges
    ADJ_PREDICATES = {"xr2t1c1d", "oh0smogc", "g7ya35is", "w224no0l"}

    def __init__(self, task):
        # Goal conditions (set of fact strings)
        self.goals = set(task.goals)
        # Static facts (frozenset of fact strings)
        self.static = set(task.static)

        # Build directed graph edges consistent with domain movement preconditions:
        # for a static fact "(PRED A B)" where PRED in ADJ_PREDICATES,
        # robot can move from B -> A (as movement operator checks (PRED A B)
        # and robot at B to move to A).
        self.edges = defaultdict(set)  # edges[source].add(target) meaning source -> target
        # For painting we want paint_positions[T] = set of S such that (PRED T S) exists.
        self.paint_positions = defaultdict(set)

        for fact in self.static:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred in self.ADJ_PREDICATES and len(parts) >= 3:
                target = parts[1]
                source = parts[2]
                # movement possible from source -> target
                self.edges[source].add(target)
                # painting position: to paint `target` robot must be at `source`
                self.paint_positions[target].add(source)

        # Precompute the set of all tiles known from static adjacency facts and
        # from paint_positions keys.
        self.all_tiles = set(self.edges.keys()) | {t for targets in self.edges.values() for t in targets} | set(self.paint_positions.keys())

        # A large cost to represent unreachable/surprising dead-end goals.
        self.large_cost = 10**6

    def _bfs_distances_from(self, start):
        """
        BFS on the directed static graph from start tile and return a dict
        mapping tile -> distance (number of moves). If start is not in the
        graph, returns an empty dict.
        """
        if start not in self.all_tiles:
            return {}
        dist = {start: 0}
        q = deque([start])
        while q:
            u = q.popleft()
            for v in self.edges.get(u, ()):
                if v not in dist:
                    dist[v] = dist[u] + 1
                    q.append(v)
        return dist

    def __call__(self, node):
        state = set(node.state)  # frozenset -> mutable set for querying

        # If already goal, return 0
        if self.goals <= state:
            return 0

        # Collect robots and their locations/colors
        robots_locations = {}  # robot -> tile
        robots_colors = {}     # robot -> color

        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "gk9u3vg3" and len(parts) >= 3:
                robot = parts[1]
                tile = parts[2]
                robots_locations[robot] = tile
            elif pred == "i4u94rl9" and len(parts) >= 3:
                robot = parts[1]
                color = parts[2]
                robots_colors[robot] = color

        # If there are robots with location but no explicit color fact present,
        # assume they have no color currently (use None). They can change to any color.
        for r in robots_locations:
            if r not in robots_colors:
                robots_colors[r] = None

        if not robots_locations:
            # no robots present - cannot reach painting goals
            return self.large_cost

        # Determine which goals are unsatisfied and their requested (tile, color)
        unsatisfied = []
        for goal in self.goals:
            if goal in state:
                continue
            parts = get_parts(goal)
            if len(parts) >= 3 and parts[0] == "yh6q5j8r":
                tile = parts[1]
                color = parts[2]
                unsatisfied.append((tile, color))
            else:
                # For safety, if a goal is not of expected form, ignore it (or count 1)
                # but keep heuristic finite.
                unsatisfied.append((goal, None))

        # Get currently painted tiles with their colors
        current_painted = {}
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "yh6q5j8r" and len(parts) >= 3:
                current_painted[parts[1]] = parts[2]

        total_cost = 0

        # Cache BFS distances per robot start
        bfs_cache = {}

        # For each unsatisfied goal tile, compute minimal estimated cost across robots
        for tile, desired_color in unsatisfied:
            # If tile is already painted with a different color, mark very costly (likely unsolvable)
            if tile in current_painted and current_painted[tile] != desired_color:
                total_cost += self.large_cost
                continue

            # Determine possible painting positions S for this tile
            candidate_positions = self.paint_positions.get(tile, set())
            if not candidate_positions:
                # No static adjacency allows painting this tile -> unreachable
                total_cost += self.large_cost
                continue

            best_cost_for_tile = self.large_cost

            # Try every robot as candidate painter
            for robot, loc in robots_locations.items():
                # BFS distances from robot start (cached)
                if loc not in bfs_cache:
                    bfs_cache[loc] = self._bfs_distances_from(loc)
                dist_map = bfs_cache[loc]

                # Compute minimal distance to any S in candidate_positions
                min_dist = None
                for s in candidate_positions:
                    # If robot is already at s, dist is 0
                    d = dist_map.get(s)
                    if d is not None:
                        if min_dist is None or d < min_dist:
                            min_dist = d

                if min_dist is None:
                    # This robot cannot reach any painting position for this tile
                    continue

                # color change cost: robot can change color anywhere in one action
                current_color = robots_colors.get(robot)
                color_change = 0 if current_color == desired_color else 1

                # painting action cost is 1
                paint_cost = 1

                cost = min_dist + color_change + paint_cost
                if cost < best_cost_for_tile:
                    best_cost_for_tile = cost

            # If no robot could reach painting positions for this tile, add large cost
            total_cost += best_cost_for_tile

            # Prevent overflow in pathological cases
            if total_cost >= self.large_cost:
                return self.large_cost

        return int(total_cost)