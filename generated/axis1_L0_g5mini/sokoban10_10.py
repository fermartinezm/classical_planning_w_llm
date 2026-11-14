from collections import deque, defaultdict
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Remove surrounding parentheses and split a fact into parts."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    A domain-dependent heuristic for the Sokoban domain.

    The heuristic estimates the remaining number of actions (robot moves + pushes)
    needed to place all boxes on their goal locations. It combines (1) a lower-level
    estimate of the number of pushes required to move each box to its dedicated goal
    (computed from the static adjacency graph) and (2) an estimate of the robot's
    travel cost to reach the first box that must be pushed. Successive pushes of the
    same box are assumed to be contiguous (which matches the domain effect where the
    robot moves into the box's previous cell after a push), so only the initial travel
    to the box is explicitly counted.

    Assumptions
    -----------
    - Each goal in the task is of the form "(at <box> <location>)" and specifies a
      unique goal location for that particular box.
    - The static adjacency facts define the traversable graph between locations via
      directed adjacency facts "(adjacent l1 l2 dir)". The heuristic uses those
      adjacency links as edges in a directed graph.
    - The heuristic ignores dynamic blocking by other boxes when computing shortest
      paths (i.e., it assumes free movement along the static graph). This makes the
      estimate optimistic with respect to robot movement, but still effective for
      guiding greedy search.
    - If some locations are not connected according to the static adjacency graph,
      unreachable distances are mapped to a large finite penalty to keep the heuristic
      finite.

    Heuristic Initialization
    ------------------------
    In the constructor:
    - Extract mapping from each box to its goal location from task.goals.
    - Build a directed adjacency graph from the static "(adjacent ...)" facts.
    - Collect all locations appearing in static facts, initial state, and goals.
    - Precompute shortest-path distances between all pairs of locations using BFS.
      The result is a dictionary self.distances[loc][other_loc] => shortest number of
      traversals (edges). If no path exists, the pair is absent and later treated
      as 'large cost'.

    Step-By-Step Thinking for Computing Heuristic
    --------------------------------------------
    When called on a planner node (state):
    1. If all goal facts are already true in the state, return 0.
    2. Find the robot's current location from the state fact "(at-robot <loc>)".
    3. For each box that is not yet at its goal:
       a. Look up the precomputed shortest-path distance from the box's current
          location to its goal location. Each step along this path translates to
          exactly one push (because each push moves the box to an adjacent location).
          Sum these distances across all non-goal boxes to obtain total_pushes.
       b. Identify candidate robot approach positions for the box (i.e., all
          locations that are adjacent to the box's current location in the static
          graph). The robot must reach one of these to perform the first push.
       c. Using precomputed distances, compute the minimal distance from the robot's
          current location to any of those candidate approach positions.
    4. From all boxes that still require pushing, take the minimal such robot travel
       distance (this is the optimistic estimate of how far the robot has to walk to
       start pushing). Add that to total_pushes.
    5. Return total_pushes + initial_robot_travel as the heuristic value.

    Rationale
    ---------
    - Counting pushes directly reflects the number of push actions required to move
      boxes to their exact goal cells (one push per adjacent move).
    - Counting only the minimal initial robot travel to reach any box approximates
      the additional walking cost required to begin pushing. Subsequent moves incurred
      by a sequence of pushes are captured implicitly (the robot ends up in the
      correct relative location after each push), so they do not need to be added.
    - The heuristic is not admissible but is fast to compute and informative for
      guiding greedy best-first search in Sokoban problems.
    """

    def __init__(self, task):
        """
        Initialize static information and precompute all-pairs shortest path
        distances over the directed adjacency graph given in task.static.
        """
        # Goals: mapping box -> goal_location (assume goals are of the form "(at box loc)")
        self.goal_locations = {}
        for fact in task.goals:
            parts = get_parts(fact)
            if parts and parts[0] == "at" and len(parts) >= 3:
                box, loc = parts[1], parts[2]
                self.goal_locations[box] = loc

        # Build adjacency graph from static facts.
        self.adj = defaultdict(list)  # directed adjacency: node -> [neighbors]
        self.pred = defaultdict(list)  # predecessors: node -> [nodes with edges to node]
        locations = set()

        for fact in task.static:
            parts = get_parts(fact)
            if parts and parts[0] == "adjacent" and len(parts) >= 3:
                l1, l2 = parts[1], parts[2]
                self.adj[l1].append(l2)
                self.pred[l2].append(l1)
                locations.add(l1)
                locations.add(l2)

        # Also include locations mentioned in the initial state and in goals
        # (some problems may not list all adjacency pairs for isolated nodes).
        for fact in task.initial_state:
            parts = get_parts(fact)
            if parts:
                if parts[0] == "at-robot" and len(parts) >= 2:
                    locations.add(parts[1])
                if parts[0] == "at" and len(parts) >= 3:
                    locations.add(parts[2])

        for loc in self.goal_locations.values():
            locations.add(loc)

        # Ensure we have an entry for all detected locations in adjacency structures.
        for loc in locations:
            self.adj.setdefault(loc, [])
            self.pred.setdefault(loc, [])

        # Precompute all-pairs shortest path distances (directed) using BFS from each location.
        self.distances = {}  # distances[src] = {dst: dist}
        for src in locations:
            dist = {}
            queue = deque([(src, 0)])
            visited = {src}
            while queue:
                node, d = queue.popleft()
                dist[node] = d
                for nbr in self.adj.get(node, ()):
                    if nbr not in visited:
                        visited.add(nbr)
                        queue.append((nbr, d + 1))
            self.distances[src] = dist

        # Large finite penalty for unreachable locations (keeps heuristic finite).
        # Scaled by number of locations to avoid excessive domination of the heuristic.
        self.INF_COST = 1000 * max(1, len(locations))

    def __call__(self, node):
        """
        Compute a heuristic estimate for the given state (node).

        Returns an integer estimate of the number of actions (moves + pushes)
        needed to reach the goal from node.state.
        """
        state = node.state

        # If already at goal, heuristic is 0.
        if hasattr(self, "goal_locations"):
            # Build goal facts as strings to compare against state quickly.
            # We could also rely on task.goal_reached external check but the heuristic
            # must be consistent if goals are present in the state's facts.
            goal_facts = set(f"(at {box} {loc})" for box, loc in self.goal_locations.items())
            if goal_facts <= state:
                return 0

        # Extract robot location from state: find fact "(at-robot <loc>)"
        robot_loc = None
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "at-robot" and len(parts) >= 2:
                robot_loc = parts[1]
                break

        # If robot location is not found (should not happen), assume a worst-case penalty.
        if robot_loc is None:
            return self.INF_COST

        # Extract current locations of relevant boxes (those that appear in goals).
        box_current_locations = {}
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "at" and len(parts) >= 3:
                obj, loc = parts[1], parts[2]
                if obj in self.goal_locations:
                    box_current_locations[obj] = loc

        # Sum of pushes required for boxes not at their goals.
        total_pushes = 0
        boxes_to_move = []

        for box, goal_loc in self.goal_locations.items():
            cur_loc = box_current_locations.get(box)
            if cur_loc is None:
                # Box not present in state (unexpected), assign high cost.
                total_pushes += self.INF_COST
                boxes_to_move.append((box, None))
                continue
            if cur_loc == goal_loc:
                continue
            # Distance (in edges) from box current location to its goal.
            # Each such edge corresponds to one push of that box.
            dist_map = self.distances.get(cur_loc, {})
            push_dist = dist_map.get(goal_loc, None)
            if push_dist is None:
                # unreachable according to static graph: large cost
                push_dist = self.INF_COST
            total_pushes += push_dist
            boxes_to_move.append((box, cur_loc))

        # If there are no boxes to move, return 0 (should have been caught earlier).
        if not boxes_to_move:
            return 0

        # Compute minimal robot travel cost to reach any approach position adjacent to
        # any box that still needs to be moved. For a box at B, the robot must reach
        # some neighbor of B to perform the first push. We consider all predecessors
        # (nodes with an adjacency edge into B) as valid approach positions.
        best_robot_travel = self.INF_COST
        for box, box_loc in boxes_to_move:
            if box_loc is None:
                continue
            # Candidate approach positions are predecessors of the box's current location.
            preds = self.pred.get(box_loc, [])  # may be empty
            if not preds:
                # If there are no predecessors (isolated), assign large penalty.
                candidate_cost = self.INF_COST
            else:
                robot_dist_map = self.distances.get(robot_loc, {})
                # distance from robot to an approach position (pred)
                candidate_cost = min(
                    (robot_dist_map.get(pred, None) if pred in robot_dist_map else None)
                    for pred in preds
                )
                # If all preds unreachable (None), set to INF
                if candidate_cost is None:
                    candidate_cost = self.INF_COST
            if candidate_cost < best_robot_travel:
                best_robot_travel = candidate_cost

        # Final heuristic: pushes (each counted as one action) plus the minimal initial
        # robot travel (each step counts as one move action). This is an optimistic
        # but informative estimate.
        return total_pushes + best_robot_travel