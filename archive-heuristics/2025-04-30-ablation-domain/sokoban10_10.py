from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque

# Helper functions (included as they are used by the class)
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    if not fact or not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# Heuristic class
class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to solve a Sokoban puzzle by summing the
    shortest path distances of each box to its goal location and adding the
    shortest path distance from the robot to the nearest box that needs to be moved.

    # Assumptions
    - The environment is represented as a graph of locations connected by
      adjacent predicates.
    - Shortest path distances between locations can be precomputed.
    - The cost of moving a box is primarily related to the distance it needs
      to travel, plus the robot's effort to reach a box to initiate pushing.
    - This heuristic is non-admissible; it does not guarantee finding the
      optimal solution but aims to guide a greedy search efficiently.

    # Heuristic Initialization
    - Parses the goal conditions to map each box to its specific goal location.
    - Extracts all locations from the initial state and static adjacent facts.
    - Builds an undirected graph representing the connectivity of locations
      based on adjacent facts.
    - Computes all-pairs shortest paths between all locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    1. Get the current locations of the robot and all boxes from the state.
    2. Identify which boxes are not currently at their designated goal locations.
    3. If all boxes are at their goals, the state is a goal state, and the
       heuristic value is 0.
    4. For each box that is not at its goal, find the shortest path distance
       from its current location to its goal location using the precomputed
       distances. Sum these distances. This represents a lower bound on the
       number of pushes required for all boxes.
    5. Find the minimum shortest path distance from the robot's current location
       to the location of any box that needs to be moved. This estimates the
       robot's cost to engage with a box.
    6. The total heuristic value is the sum of the total box-to-goal distance
       (step 4) and the minimum robot-to-box distance (step 5).
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, building the
        location graph, and computing all-pairs shortest paths.
        """
        self.goals = task.goals
        self.static_facts = task.static
        self.initial_state = task.initial_state

        # Map boxes to their goal locations
        self.goal_locations = {}
        for goal_fact in self.goals:
            parts = get_parts(goal_fact)
            if parts and parts[0] == 'at' and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                # Assuming objects in 'at' goals are the boxes we care about
                self.goal_locations[obj] = loc

        # Collect all locations from initial state and static facts
        all_locations = set()
        for fact in self.initial_state:
             parts = get_parts(fact)
             # Locations can appear in at-robot, at, or clear predicates
             if parts and parts[0] in ['at-robot', 'at', 'clear'] and len(parts) >= 2:
                 # The location is always the last argument
                 all_locations.add(parts[-1])
        for fact in self.static_facts:
            parts = get_parts(fact)
            # Locations appear in adjacent predicates
            if parts and parts[0] == 'adjacent' and len(parts) == 4:
                all_locations.add(parts[1])
                all_locations.add(parts[2])

        # Build the undirected location graph
        self.graph = {loc: [] for loc in all_locations}
        for fact in self.static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == 'adjacent' and len(parts) == 4:
                loc1, loc2, direction = parts[1], parts[2], parts[3]
                # Add bidirectional edges for distance calculation
                if loc2 not in self.graph[loc1]:
                    self.graph[loc1].append(loc2)
                if loc1 not in self.graph[loc2]:
                    self.graph[loc2].append(loc1)

        # Compute all-pairs shortest paths
        self.dist = {}
        for start_loc in all_locations:
            self.dist[start_loc] = self._bfs(start_loc, all_locations)

    def _bfs(self, start_loc, all_locations):
        """Helper function to perform BFS from a start location."""
        distances = {loc: float('inf') for loc in all_locations}
        distances[start_loc] = 0
        queue = deque([(start_loc, 0)])
        visited = {start_loc}

        while queue:
            curr_loc, dist = queue.popleft()

            # Ensure location exists in graph keys and has neighbors
            for neighbor in self.graph.get(curr_loc, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    distances[neighbor] = dist + 1
                    queue.append((neighbor, dist + 1))
        return distances


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Find robot location
        robot_loc = None
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == 'at-robot' and len(parts) == 2:
                robot_loc = parts[1]
                break

        # Find box locations for boxes that have goals
        box_locs = {}
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == 'at' and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                if obj in self.goal_locations:
                     box_locs[obj] = loc

        # Identify boxes not at their goal
        boxes_to_move = [box for box, loc in box_locs.items() if loc != self.goal_locations[box]]

        # If all boxes are at goal, heuristic is 0
        if not boxes_to_move:
            return 0

        # Calculate sum of box-to-goal distances
        sum_box_dist = 0
        for box in boxes_to_move:
            current_loc = box_locs[box]
            goal_loc = self.goal_locations[box]
            # Ensure locations are valid and distance is computed
            if current_loc in self.dist and goal_loc in self.dist.get(current_loc, {}):
                 dist = self.dist[current_loc][goal_loc]
                 if dist == float('inf'):
                     # Box is in a disconnected component from its goal
                     return float('inf')
                 sum_box_dist += dist
            else:
                 # Should not happen if init is correct, but safeguard
                 # This implies a box or goal location wasn't in the graph
                 return float('inf')

        # Calculate minimum robot-to-box distance for boxes needing movement
        min_robot_box_dist = float('inf')
        # Ensure robot location was found and is in the graph
        if robot_loc is None or robot_loc not in self.dist:
             return float('inf')

        for box in boxes_to_move:
            box_loc = box_locs[box]
            # Ensure box location is valid and distance is computed
            if box_loc in self.dist.get(robot_loc, {}):
                dist = self.dist[robot_loc][box_loc]
                min_robot_box_dist = min(min_robot_box_dist, dist)
            else:
                 # Should not happen if init is correct, but safeguard
                 # This implies a box location wasn't in the graph
                 return float('inf')

        # The heuristic value is the sum of box distances and robot distance to nearest box
        return sum_box_dist + min_robot_box_dist
