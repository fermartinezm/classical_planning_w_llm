from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at box1 loc_3_5)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    Summary:
    ----------
    This heuristic estimates the number of actions required to solve a Sokoban state by approximating 
    the effort to push all boxes from their current locations to their goal locations. It combines:
    - The minimal sum of shortest distances to push each box to a goal location (assuming the robot can position itself optimally),
    - The minimal movement cost for the robot to get into pushing positions for each box,
    - And an approximate cost of pushing actions based on distances.

    It is not admissible but aims to be informative and computationally efficient to guide greedy best-first search effectively.

    Assumptions:
    -----------
    - The layout graph (locations and adjacencies) is static.
    - There is exactly one robot and one or more boxes.
    - Each box must be moved to exactly one goal location.
    - Boxes and robot locations are given as discrete locations.
    - Robot can move (move actions) on clear locations.
    - Boxes are pushed by the robot only into adjacent clear locations.
    - The heuristic ignores complex obstruction and deadlock detection but accounts for shortest distances in the layout.
    - The heuristic estimate always returns 0 if the state is a goal state.
    - The heuristic value is finite for reachable states.

    Heuristic Initialization:
    -------------------------
    - Parse static facts to build:
      - A graph representing the grid layout from adjacent location facts.
      - A list of all goal locations extracted from the task goals on box positions.
    - Precompute shortest distances between all locations using BFS on the adjacency graph.
    - Store goal locations for boxes from the task goals.

    Step-By-Step Thinking for Computing Heuristic:
    ----------------------------------------------
    1. Check if the current state satisfies all goal conditions; if so, return 0.
    2. Extract current robot location and the locations of all boxes.
    3. For each box current location, compute distances to all goal locations using precomputed shortest paths.
    4. Find a minimal matching of boxes to goal locations minimizing sum of distances (greedy matching).
       (Since usually few boxes, this is tractable; the heuristic aims to assign each box to the closest unmatched goal.)
    5. For each box-goal pair:
       - Estimate pushing cost as the shortest path distance from the box's current location to the goal location.
         Each push moves the box one step; so the push cost roughly equals the path length.
       - Estimate robot traveling cost to get behind the box in the right direction to push it one step closer to goal.
         Since precise push directions are complex to estimate, approximate the robot cost as the minimal cost for the robot 
         to reach a location adjacent to the box that is required for pushing towards the goal location.
    6. Sum all pushing costs plus an estimate for the robot movements.
       Since pushing is dominant, pushing distances weighted with a small factor for robot travel is used.
    7. Return the sum as heuristic estimate.

    This heuristic balances informativeness and computational cost by using shortest path distances on the static adjacency graph
    and a greedy box-to-goal assignment strategy without performing complex deadlock or robot path feasibility checks.
    """

    def __init__(self, task):
        """Initialize heuristic by parsing static facts and goals, building adjacency graph and shortest paths."""
        self.goals = task.goals
        self.static = task.static

        # Build adjacency graph: for each location, maintain a set of reachable adjacent locations.
        self.adjacency = {}
        for fact in self.static:
            if match(fact, "adjacent", "*", "*", "*"):
                parts = get_parts(fact)
                loc1, loc2 = parts[1], parts[2]
                self.adjacency.setdefault(loc1, set()).add(loc2)
                # Note: adjacency is directed; sokoban adjacency usually bidirectional so adj facts in both directions exist.

        # Extract all goals that are of form (at box location)
        # We store the goal locations only, ignoring which box is assigned since any box can go to any goal in sokoban.
        self.goal_locations = set()
        for goal_fact in self.goals:
            if match(goal_fact, "at", "*", "*"):
                parts = get_parts(goal_fact)
                location = parts[2]
                self.goal_locations.add(location)

        # Precompute shortest distances between all locations using BFS

        # Cache distances in a dict of dict: distance[src][dest] = shortest path length
        self.distance = {}

        # Compute distances from every location to every other location
        for loc in self.adjacency:
            self.distance[loc] = self._bfs_distances(loc)

    def _bfs_distances(self, start):
        """Compute shortest distance from start to all reachable locations using BFS on adjacency graph."""
        dist = {start: 0}
        queue = [start]
        while queue:
            current = queue.pop(0)
            current_dist = dist[current]
            for neighbor in self.adjacency.get(current, []):
                if neighbor not in dist:
                    dist[neighbor] = current_dist + 1
                    queue.append(neighbor)
        return dist

    def __call__(self, node):
        """Compute heuristic estimate for the given node's state."""
        state = node.state

        # Quick goal check - if all goal facts hold, heuristic is 0.
        if self.goals <= state:
            return 0

        # Extract current robot location
        robot_loc = None
        for fact in state:
            if match(fact, "at-robot", "*"):
                robot_loc = get_parts(fact)[1]
                break

        # Extract box locations as a dict: box -> location
        boxes = {}
        for fact in state:
            if match(fact, "at", "*", "*"):
                parts = get_parts(fact)
                if parts[0] == "at" and parts[1].startswith("box"):
                    boxes[parts[1]] = parts[2]

        box_locations = list(boxes.values())
        goal_locations = list(self.goal_locations)

        # If no boxes or no goals, heuristic is zero (should only happen for trivial or goal states)
        if not box_locations or not goal_locations:
            return 0

        # Compute minimal sum of distances matching boxes to goals using greedy assignment by nearest neighbor.
        # Note: This is a heuristic approximation and does not guarantee global minimal matching.
        unmatched_goals = set(goal_locations)
        box_to_goal_pairs = []
        for box_loc in box_locations:
            # Find nearest unmatched goal location for this box
            best_goal = None
            best_dist = None
            dist_from_box = self.distance.get(box_loc, {})
            for goal_loc in unmatched_goals:
                dist = dist_from_box.get(goal_loc)
                if dist is not None:
                    if best_dist is None or dist < best_dist:
                        best_goal = goal_loc
                        best_dist = dist
            if best_goal is None:
                # If there is no reachable goal from this box's location,
                # return a large number (heuristic is not infinity but large)
                best_dist = 1000
                best_goal = next(iter(unmatched_goals))  # just pick one arbitrarily
            unmatched_goals.remove(best_goal)
            box_to_goal_pairs.append((box_loc, best_goal, best_dist))

        # Estimate pushing and robot costs
        # Each push moves the box one step closer to the goal at a cost of 1 action.
        # Robot movement to pushing position approximated as min distance from robot to adjacent location behind box for pushing.
        # Here we approximate robot move cost per box as:
        # minimal distance from robot location to any location adjacent to the box from which a push can be made.

        total_push_cost = 0
        total_robot_move_cost = 0

        dist_robot = self.distance.get(robot_loc, {}) if robot_loc else {}

        for box_loc, goal_loc, push_dist in box_to_goal_pairs:
            total_push_cost += push_dist
            # To estimate the robot positioning cost:
            # Find all locations adjacent to box_loc where robot can stand and push box towards goal_loc.
            # This means finding directions from box_loc to goal_loc, then robot must stand on the opposite side.
            # Since directions are not explicitly encoded, approximate by checking all adjacency from box_loc
            # and testing if next location towards goal_loc.

            # Get neighbors of box_loc
            neighbors = self.adjacency.get(box_loc, set())

            # Candidate robot positions are those adjacent to box_loc such that box can be pushed towards goal
            # That means the goal location must be reachable from box_loc by moving the box.
            # We consider for each neighbor if robot can push the box from that side towards the box_loc.

            # We'll approximate by considering neighbors of box_loc:
            # For a neighbor n of box_loc, if goal_loc is reachable from n via the box moving one step away from n,
            # then robot can stand at n to push the box towards goal_loc.

            # So find all neighbors n of box_loc that satisfy that box can be pushed to a location closer to goal.
            # Since Sokoban adjacency is bidirectional, we check if box_loc is adjacent to n (which it is),
            # and if from box_loc to goal_loc shortest distance is push_dist,
            # then pushing from n to box_loc means box moves from box_loc to a neighbor of box_loc.
            #
            # As we don't know exact push directions, we approximate:
            # For all neighbors n of box_loc, compute distance from n to goal_loc,
            # the robot standing at n pushes box from box_loc to n, i.e. box moves from box_loc to n.
            # So robot must stand at n to push box from box_loc to n.

            # We find minimal distance from robot_loc to those candidate robot positions n.

            dist_box = self.distance.get(box_loc, {})
            candidate_positions = []
            for n in neighbors:
                # The box would be pushed into neighbor n:
                # Check if distance from n to goal_loc is less than distance from box_loc to goal_loc,
                # meaning pushing box from box_loc to n is a step closer to goal.
                dist_n_to_goal = self.distance.get(n, {}).get(goal_loc)
                if dist_n_to_goal is not None and dist_n_to_goal < push_dist:
                    candidate_positions.append(n)

            if not candidate_positions:
                # If no suitable pushing position found, fallback: robot moves directly to box location (not valid push),
                # but still add distance
                candidate_positions = [box_loc]

            # Find minimal robot distance to candidate push positions
            min_robot_dist = min((dist_robot.get(pos, 1000) for pos in candidate_positions), default=1000)

            total_robot_move_cost += min_robot_dist

        # Weight the pushing cost stronger than robot movement cost.
        # Pushing each box step requires 1 action,
        # robot movements to pushing positions add overhead but are generally smaller.
        #
        # We use a weighting factor to balance influence on heuristic.
        ROBOT_MOVE_WEIGHT = 0.5

        heuristic_value = total_push_cost + ROBOT_MOVE_WEIGHT * total_robot_move_cost
        # Ensure heuristic is at least 1 if not goal and nonzero costs estimated to prevent zero heuristic in non-goal states.
        if heuristic_value < 1:
            heuristic_value = 1

        return heuristic_value
