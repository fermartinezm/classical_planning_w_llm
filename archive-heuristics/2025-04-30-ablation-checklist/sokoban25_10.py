import sys
from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic

# Set a large value to represent infinity for unreachable locations
INF = float('inf')

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at-robot loc_1_1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure the number of parts matches the number of args, unless args contains wildcards
    if len(parts) != len(args) and '*' not in args:
         return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def bfs(start_node, graph):
    """
    Performs Breadth-First Search starting from start_node to find shortest
    distances to all reachable nodes in the graph.

    Args:
        start_node: The node to start the BFS from.
        graph: An adjacency list representation of the graph (dict: node -> set of neighbors).

    Returns:
        A dictionary mapping reachable nodes to their shortest distance from start_node.
        Nodes not reachable will not be in the dictionary.
    """
    distances = {start_node: 0}
    queue = deque([start_node])
    visited = {start_node}

    while queue:
        current_node = queue.popleft()
        current_dist = distances[current_node]

        if current_node in graph: # Handle nodes with no outgoing edges
            for neighbor in graph[current_node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    distances[neighbor] = current_dist + 1
                    queue.append(neighbor)

    return distances

class SokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing two components:
    1. The sum of shortest path distances for each box from its current location
       to its goal location (ignoring obstacles and robot).
    2. The minimum shortest path distance for the robot from its current location
       to any location adjacent to any misplaced box.

    This heuristic is not admissible as it ignores obstacles and the specific
    pushing mechanics (which require clear squares). It also doesn't account
    for potential deadlocks. However, it provides a reasonable estimate for
    greedy best-first search by prioritizing states where boxes are closer
    to their goals and the robot is positioned to potentially push a box.

    # Heuristic Initialization
    - Parses goal conditions to map each box to its goal location.
    - Builds an adjacency graph of locations based on `adjacent` facts.
    - Pre-computes all-pairs shortest paths on this location graph using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the current location of the robot and all boxes.
    2. Identify which boxes are not yet at their goal locations.
    3. Calculate the sum of shortest path distances for each misplaced box
       from its current location to its goal location using the pre-computed distances.
       This estimates the minimum number of 'box movements' needed.
    4. Calculate the minimum shortest path distance for the robot from its
       current location to *any* location that is adjacent to *any* misplaced box.
       This estimates the robot's cost to get into a position to potentially
       start pushing a box.
    5. The heuristic value is the sum of the total box distance and the minimum
       robot access distance.
    6. If any goal location is unreachable from a box's current location, or
       if the robot cannot reach any location adjacent to a misplaced box,
       the heuristic returns infinity, indicating an likely unsolvable state
       or a state far from the goal.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and building
        and analyzing the location graph.
        """
        super().__init__(task)

        # Store goal locations for each box.
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                box, location = args
                self.goal_locations[box] = location

        # Build the adjacency graph from static 'adjacent' facts.
        self.adj_list = {}
        all_locations = set()
        for fact in task.static:
            if match(fact, "adjacent", "*", "*", "*"):
                _, loc1, loc2, _ = get_parts(fact)
                all_locations.add(loc1)
                all_locations.add(loc2)
                if loc1 not in self.adj_list:
                    self.adj_list[loc1] = set()
                self.adj_list[loc1].add(loc2)

        self.all_locations = list(all_locations) # Store as list for consistent ordering

        # Pre-compute all-pairs shortest paths using BFS.
        # self.distances[l1][l2] will store the shortest distance from l1 to l2.
        self.distances = {}
        for start_loc in self.all_locations:
            self.distances[start_loc] = bfs(start_loc, self.adj_list)

    def get_distance(self, loc1, loc2):
        """Safely retrieve pre-computed distance, returning INF if unreachable."""
        if loc1 not in self.distances or loc2 not in self.distances[loc1]:
            return INF
        return self.distances[loc1][loc2]

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Find current location of the robot.
        robot_location = None
        for fact in state:
            if match(fact, "at-robot", "*"):
                robot_location = get_parts(fact)[1]
                break

        if robot_location is None:
             # This should not happen in a valid Sokoban state, but handle defensively
             return INF

        # Find current locations of all boxes.
        box_locations = {}
        for fact in state:
            if match(fact, "at", "*", "*"):
                obj_type = get_parts(fact)[1] # Need to check if it's a box
                # A more robust way would be to get object types from the task,
                # but assuming 'box' objects start with 'box' based on examples.
                if obj_type.startswith('box'):
                     box, location = get_parts(fact)[1:]
                     box_locations[box] = location

        total_box_distance = 0
        misplaced_boxes = []

        # Calculate sum of distances for misplaced boxes.
        for box, goal_location in self.goal_locations.items():
            current_location = box_locations.get(box) # Use .get() in case box isn't in state (shouldn't happen)

            if current_location is None:
                 # Box is missing from state, problem likely unsolvable from here
                 return INF

            if current_location != goal_location:
                misplaced_boxes.append(box)
                dist = self.get_distance(current_location, goal_location)
                if dist == INF:
                    # Box goal is unreachable in the static grid
                    return INF
                total_box_distance += dist

        # If all boxes are at their goals, the heuristic is 0.
        if not misplaced_boxes:
            return 0

        # Calculate minimum robot distance to a location adjacent to any misplaced box.
        min_robot_access_dist = INF
        robot_can_reach_any_adjacent = False

        for box in misplaced_boxes:
            box_loc = box_locations[box]
            # Consider all locations adjacent to the box's current location
            if box_loc in self.adj_list:
                for adjacent_loc in self.adj_list[box_loc]:
                    # Check if the robot can reach this adjacent location
                    dist_to_adjacent = self.get_distance(robot_location, adjacent_loc)
                    if dist_to_adjacent != INF:
                        min_robot_access_dist = min(min_robot_access_dist, dist_to_adjacent)
                        robot_can_reach_any_adjacent = True

        # If the robot cannot reach any location adjacent to any misplaced box,
        # the state is likely unsolvable or very difficult.
        if not robot_can_reach_any_adjacent:
             return INF

        # The heuristic is the sum of box distances and the minimum robot access distance.
        # Add 1 to the robot distance to account for the push action itself,
        # which involves the robot moving into the box's spot.
        # This is a simplification; a push is one action, but it enables box movement.
        # A simple sum of box moves + robot access seems reasonable for a non-admissible heuristic.
        # Let's just sum the distances for simplicity as a first attempt.
        return total_box_distance + min_robot_access_dist

