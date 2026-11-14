from collections import deque
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Return the components of a PDDL fact string without surrounding parentheses."""
    # fact is like "(predicate arg1 arg2)". Remove parentheses and split.
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    A domain-dependent heuristic for the Sokoban domain.

    This heuristic estimates the number of actions required to reach the goal by
    combining (1) an estimate of the number of box pushes needed (sum of
    shortest-path distances in the location graph from each box to its goal
    location) and (2) an estimate of the initial robot travel cost to begin
    pushing (distance from the robot to the closest useful adjacent cell of a
    box that still needs to be moved).

    Assumptions
    -----------
    - The domain's static facts include adjacency relations of the form
      (adjacent locA locB dir). We treat adjacency as an undirected graph.
    - Boxes only change location via the push action and are represented by
      facts "(at BOX LOC)" in the state.
    - Goal facts are of the form "(at BOX GOAL_LOC)". The constructor extracts
      goals for all named boxes appearing in the task goals.
    - Deadlocks and irreversibility are not explicitly detected; unreachable
      goals due to deadlocks produce a large heuristic penalty (finite).
    - The heuristic does not try to be admissible; it aims to be informative and
      cheap to compute.

    Heuristic Initialization
    ------------------------
    In the constructor we extract:
    - self.goal_locations: mapping box -> goal location (from task.goals).
    - self.locations_graph: adjacency map location -> set(neighbor locations)
      built from static facts (treating adjacency as undirected).
    No dynamic facts are stored here; they come from node.state at evaluation time.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. Goal check:
       - If all goal facts are already satisfied in the current state, return 0.

    2. Parse the current state to extract:
       - robot_loc: the current location of the robot from the fact "(at-robot ...)".
       - box_locations: mapping box -> current location from facts "(at BOX LOC)".

    3. Preliminaries:
       - Use BFS on the static location graph to compute shortest-path distances
         between locations on demand. If no path exists, treat the distance as
         a large finite penalty (meaning the position is likely unsolvable).

    4. Estimate box work (push cost):
       - For each box that has a goal location, compute the shortest-path
         distance (in edges) from its current location to its goal location.
         Each edge along that path corresponds to one push needed (one push
         moves the box by one adjacent cell). Sum these distances for all
         boxes. If a box has no path to its goal, add a large penalty for that
         box.

    5. Estimate initial robot travel cost:
       - The robot must reach a position from which it can start pushing a box.
         For each box not already at its goal, compute the minimal robot
         distance (in robot moves) to any neighbor cell of the box (i.e., any
         cell adjacent to the box's current location). We take the minimum
         across all such box-neighbor pairs to estimate the robot's initial
         travel cost to begin useful pushing. If the robot is already adjacent
         to a box, this cost can be 0.

    6. Combine:
       - The heuristic value is:
            total_pushes_sum + robot_initial_travel_cost
         where total_pushes_sum is the sum of shortest-path lengths for boxes.
       - This gives an estimate in "actions" where pushes are counted as
         individual actions and the robot's initial non-push moves to reach a
         box are also counted. It is intentionally simple and optimistic about
         inter-box travel and potential re-positioning, but effective and fast.

    Notes on finiteness and usefulness
    ----------------------------------
    - If a required path (box->goal or robot->neighbor) does not exist in the
      static adjacency graph, the heuristic returns a large finite value (a
      configurable penalty). This keeps the heuristic finite even for
      unsolvable-looking states while biasing the search away from them.
    - The heuristic is not admissible but is cheap to compute (BFS on small
      grids) and typically guides greedy best-first search toward solutions by
      focusing on box distances to goals and the robot's readiness to start
      pushing.
    """

    # Penalty used when a required path does not exist in the static graph.
    UNREACHABLE_PENALTY = 1000

    def __init__(self, task):
        """
        Extract static adjacency graph and goal box locations from the task.

        - task.goals: set of goal fact strings
        - task.static: set of static fact strings
        """
        self.goals = task.goals
        static_facts = task.static

        # Extract goal locations for boxes: mapping box -> goal_location
        self.goal_locations = {}
        for fact in self.goals:
            parts = get_parts(fact)
            if parts and parts[0] == "at" and len(parts) >= 3:
                box, loc = parts[1], parts[2]
                # store goal for this box
                self.goal_locations[box] = loc

        # Build adjacency graph from static facts: treat as undirected to be robust.
        self.locations_graph = {}
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "adjacent" and len(parts) >= 3:
                a, b = parts[1], parts[2]
                self.locations_graph.setdefault(a, set()).add(b)
                self.locations_graph.setdefault(b, set()).add(a)

        # A small cache for BFS distances to reduce repeated BFS calls in one
        # heuristic evaluation (keyed by source location).
        self._bfs_cache = {}

    def _bfs_distances_from(self, src):
        """Return dict of distances from src to all reachable locations (BFS)."""
        if src in self._bfs_cache:
            return self._bfs_cache[src]

        distances = {}
        if src not in self.locations_graph:
            # No outgoing edges, only reachable to itself
            distances[src] = 0
            self._bfs_cache[src] = distances
            return distances

        q = deque([src])
        distances[src] = 0
        while q:
            v = q.popleft()
            for nb in self.locations_graph.get(v, ()):
                if nb not in distances:
                    distances[nb] = distances[v] + 1
                    q.append(nb)
        self._bfs_cache[src] = distances
        return distances

    def shortest_distance(self, a, b):
        """Return the shortest distance between locations a and b (number of edges),
        or None if b is not reachable from a."""
        if a == b:
            return 0
        dists = self._bfs_distances_from(a)
        return dists.get(b, None)

    def __call__(self, node):
        """
        Compute the heuristic estimate for the given search node.

        node.state is a frozenset of fact strings like "(at box1 loc_3_5)".
        """
        state = node.state

        # 1) If goals are already satisfied, heuristic is 0.
        if self.goals <= state:
            return 0

        # Clear BFS cache for this evaluation (so we don't grow it across many states)
        # but keep the static graph persistent.
        self._bfs_cache.clear()

        # 2) Extract robot location.
        robot_loc = None
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "at-robot" and len(parts) >= 2:
                robot_loc = parts[1]
                break

        # 3) Extract current box locations.
        box_locations = {}
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "at" and len(parts) >= 3:
                obj, loc = parts[1], parts[2]
                # Only record things that are boxes according to the goals we know,
                # or generally treat any 'at X L' where X appears in goal_locations as a box.
                if obj in self.goal_locations:
                    box_locations[obj] = loc

        # If a box mentioned in goals is missing in the state, treat it as very costly.
        total_pushes = 0
        num_moving_boxes = 0
        for box, goal_loc in self.goal_locations.items():
            cur_loc = box_locations.get(box, None)
            if cur_loc is None:
                # Missing box fact -> penalize heavily
                total_pushes += self.UNREACHABLE_PENALTY
                num_moving_boxes += 1
                continue

            if cur_loc == goal_loc:
                # This box is already at its goal.
                continue

            num_moving_boxes += 1
            dist = self.shortest_distance(cur_loc, goal_loc)
            if dist is None:
                total_pushes += self.UNREACHABLE_PENALTY
            else:
                # Each edge corresponds to one push (box moves one cell per push).
                total_pushes += dist

        # 4) Robot initial travel cost: distance from robot to any neighbor of any box
        # that still needs to be moved. If robot missing, ignore (set to 0).
        robot_travel = 0
        if robot_loc is not None and num_moving_boxes > 0:
            best = None
            for box, cur_loc in box_locations.items():
                # only consider boxes that are not at goal
                goal_loc = self.goal_locations.get(box)
                if goal_loc is not None and cur_loc == goal_loc:
                    continue
                # neighbors of the box where the robot could stand to push
                neighbors = self.locations_graph.get(cur_loc, set())
                if not neighbors:
                    # no neighbors -> unreachable to push this box
                    continue
                # compute minimal distance from robot to any neighbor
                dists_from_robot = self._bfs_distances_from(robot_loc)
                # Check neighbors that are reachable
                min_to_box = None
                for nb in neighbors:
                    if nb in dists_from_robot:
                        d = dists_from_robot[nb]
                        if min_to_box is None or d < min_to_box:
                            min_to_box = d
                if min_to_box is None:
                    # This box has no neighbor reachable from robot; skip but treat as bad later
                    continue
                if best is None or min_to_box < best:
                    best = min_to_box
            if best is None:
                # Robot cannot reach any neighbor of any movable box (likely unsolvable)
                robot_travel = self.UNREACHABLE_PENALTY
            else:
                robot_travel = best

        # 5) Combine components.
        # We count pushes exactly and add the robot travel cost needed to start.
        heuristic_value = total_pushes + robot_travel

        # Small tie-breaker: prefer states with fewer boxes remaining to move by
        # subtracting a tiny value is not allowed if heuristic must be integer, so
        # we add nothing. Keep integer-valued heuristic.
        return int(heuristic_value)