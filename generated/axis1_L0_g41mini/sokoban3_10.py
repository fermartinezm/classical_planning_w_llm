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

    Summary
    -------
    This heuristic estimates the number of actions required to push all boxes onto their goal locations.
    The heuristic guides a greedy best-first search by combining:
    - The minimal distances (in terms of moves) for each box from its current location to any goal location.
    - The estimated cost of moving the robot from its current location to push each box in turn.

    The heuristic is non-admissible but aims to be informative and efficiently computable.

    Assumptions
    -----------
    - Each box must be moved to a specific goal location where it is required.
    - The robot can only push boxes, not pull them.
    - Moving to a position adjacent to a box, then pushing it towards a goal, are the main cost contributors.
    - The environment is static except for movable boxes and the robot.
    - It is sufficient to consider Manhattan-like distances on the graph induced by 'adjacent' relations for estimating movement costs.
    - The heuristic does not try to detect deadlocks or other unsolvable configurations.

    Heuristic Initialization
    ------------------------
    - Parse all static `(adjacent loc1 loc2 dir)` facts to build an undirected graph (adjacency list) of locations.
    - Extract all goal locations for boxes from the goal conditions `(at box loc)`.
    - Identify the boxes in the task.
    - Precompute shortest paths between all locations using BFS on the adjacency graph for efficient distance queries.
      (This is done once to speed up heuristic calls.)

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given state:
    1) Identify the robot's current location.
    2) Identify each box's current location.
    3) For each box:
       a) Find the goal location assigned to that box from the goal conditions.
       b) Compute the shortest path distance from the box's current location to its goal location.
          This estimates the minimal number of pushes needed for that box.
    4) Estimate the robot cost to reach each box to push it:
       a) For each box, compute the shortest distance from the robot's location to a location adjacent to the box
          from which the robot can push the box towards its goal. The pushing direction must be valid according to adjacency.
       b) To keep it efficient, consider the minimum distance to any adjacent location of the box that lies on the shortest path
          from the box to its goal (to ensure pushing towards goal).
    5) Sum for all boxes an estimate = distance_box_to_goal (pushes) + distance_robot_to_push_position (moves).
       (Moving the robot costs actions; each push also requires a robot to be in pushing position.)
    6) Return this sum as the heuristic value.
    7) If the state is a goal state (all boxes at their goal), return 0.

    This heuristic attempts to capture pressing factors influencing progress:
    where the boxes are in relation to their goals and how far the robot must reach them to push.
    """

    def __init__(self, task):
        """Initialize heuristic data structures using static facts and goal conditions."""
        self.goals = task.goals
        static_facts = task.static

        # Build adjacency graph of locations: undirected edges for quick path searches.
        self.adj_graph = {}
        for fact in static_facts:
            if match(fact, "adjacent", "*", "*", "*"):
                parts = get_parts(fact)
                loc1 = parts[1]
                loc2 = parts[2]
                # Add edges both ways (though adjacency seems bidirectional explicitly, better safe)
                self.adj_graph.setdefault(loc1, set()).add(loc2)
                self.adj_graph.setdefault(loc2, set()).add(loc1)

        # Extract box to goal location mapping from goals: (at box loc).
        self.box_goals = {}
        for goal_fact in self.goals:
            if match(goal_fact, "at", "*", "*"):
                parts = get_parts(goal_fact)
                box = parts[1]
                loc = parts[2]
                self.box_goals[box] = loc

        # Gather all locations from adjacency graph for distance computations.
        self.locations = set(self.adj_graph.keys())
        for neighbors in self.adj_graph.values():
            self.locations |= neighbors

        # Precompute shortest distances between all locations using BFS.
        # Store distances as a dict of dict: dist[loc1][loc2] = shortest distance
        self.distances = {}
        for loc in self.locations:
            self.distances[loc] = self._bfs_distances(loc)

    def _bfs_distances(self, start):
        """Return dict of shortest path distances from start to all reachable locations."""
        distances = {start: 0}
        queue = [start]
        head = 0
        while head < len(queue):
            current = queue[head]
            head += 1
            for neighbor in self.adj_graph.get(current, ()):
                if neighbor not in distances:
                    distances[neighbor] = distances[current] + 1
                    queue.append(neighbor)
        return distances

    def __call__(self, node):
        """Compute heuristic value estimating actions to reach the goal from the given state."""
        state = node.state

        # Check if goal reached: all box goal facts hold in the state.
        if self.goals <= state:
            return 0

        # Extract robot location
        robot_loc = None
        for fact in state:
            if match(fact, "at-robot", "*"):
                robot_loc = get_parts(fact)[1]
                break
        if robot_loc is None:
            # No robot location found - should not happen
            return float('inf')

        # Extract current box locations
        box_locations = {}
        for fact in state:
            if match(fact, "at", "*", "*"):
                parts = get_parts(fact)
                obj = parts[1]
                loc = parts[2]
                if obj in self.box_goals:
                    box_locations[obj] = loc

        # If any box missing location => invalid state => return infinite heuristic
        if len(box_locations) < len(self.box_goals):
            return float('inf')

        total_cost = 0

        # For each box compute:
        # - distance from box current location to box goal (pushes)
        # - minimal distance for robot to reach a pushing position to push box towards the goal
        for box, goal_loc in self.box_goals.items():
            box_loc = box_locations[box]

            # Distance box to goal: number of pushes at least
            dist_box_goal = self.distances.get(box_loc, {}).get(goal_loc, None)
            if dist_box_goal is None:
                # Box cannot reach goal location, unsolvable -> infinite heuristic
                return float('inf')

            # We want to find a location adjacent to box_loc from which robot can push towards goal_loc
            # Find all neighbors of box_loc and check if pushing from neighbor moves box closer to goal

            min_robot_dist = float('inf')

            neighbors_of_box = self.adj_graph.get(box_loc, set())
            # For each neighbor, check if box_loc to goal_loc distance decreases by moving box in direction from neighbor
            for robot_pos in neighbors_of_box:
                # Check if robot can push box from robot_pos to box_loc->pushing_direction
                # For pushing, box moves from box_loc to some neighbor in direction from robot_pos to box_loc.
                # So find location loc_push_to which is neighbor of box_loc (other than robot_pos) where pushing would move box.
                # Actually, pushing means robot is at robot_pos and box is at box_loc, push moves box to loc_push_to adjacent to box_loc.
                # So loc_push_to must be adjacent to box_loc, and robot_pos must be adjacent to box_loc on opposite side.
                # Since adjacency is bidirectional and we have adjacency graph, we check if pushing to neighbor would decrease box-to-goal distance.

                # Candidate push locations are neighbors of box_loc (different from robot_pos)
                for push_to in neighbors_of_box:
                    if push_to == robot_pos:
                        continue
                    # Check if pushing box from box_loc to push_to gets closer to goal:
                    dist_before = self.distances.get(box_loc, {}).get(goal_loc, float('inf'))
                    dist_after = self.distances.get(push_to, {}).get(goal_loc, float('inf'))
                    # Valid push if dist_after < dist_before and robot_pos and push_to adjacent to box_loc
                    # Also robot_pos must be adjacent to box_loc (already given) and push_to is adjacent to box_loc.
                    # Actually, from domain: pushing direction must be the same as adjacent robot_pos->box_loc and box_loc->push_to.
                    # We have adjacency graph but no directions stored here, so approximate by adjacency.

                    # We only want to consider pushes that move box closer to goal:
                    if dist_after < dist_before:
                        # Compute robot distance to robot_pos
                        robot_dist = self.distances.get(robot_loc, {}).get(robot_pos, float('inf'))
                        if robot_dist < min_robot_dist:
                            min_robot_dist = robot_dist

            # If no adjacent robot position found with valid pushing direction, fallback to just robot distance to box location
            if min_robot_dist == float('inf'):
                # Robot must go to box location to start pushing (somehow)
                min_robot_dist = self.distances.get(robot_loc, {}).get(box_loc, float('inf'))
                if min_robot_dist == float('inf'):
                    return float('inf')

            # Each push requires 1 push action, and each step robot takes action (move).
            # Heuristic sum: push costs plus robot movement costs to pushing position
            total_cost += dist_box_goal + min_robot_dist

        return total_cost
