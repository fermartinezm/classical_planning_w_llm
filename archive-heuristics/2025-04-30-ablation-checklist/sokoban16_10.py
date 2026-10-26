import collections
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at ball1 rooma)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def bfs(graph, start_node):
    """
    Performs a Breadth-First Search to find shortest distances from a start node
    to all reachable nodes in an undirected graph.

    Args:
        graph: An adjacency list representation of the graph (dict: node -> list of neighbors).
        start_node: The node to start the BFS from.

    Returns:
        A dictionary mapping each reachable node to its shortest distance from the start_node.
    """
    distances = {start_node: 0}
    queue = collections.deque([start_node])

    while queue:
        current_node = queue.popleft()
        distance = distances[current_node]

        if current_node in graph: # Handle potential nodes with no adjacent facts
            for neighbor in graph[current_node]:
                if neighbor not in distances:
                    distances[neighbor] = distance + 1
                    queue.append(neighbor)
    return distances

class SokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost by summing the shortest path distances
    for each box to its goal location, plus the shortest path distance
    for the robot to reach the closest box that needs to be moved.

    # Assumptions
    - The grid structure is defined by `adjacent` predicates.
    - Each box has a specific goal location.
    - The cost of moving a box is primarily determined by the distance it needs to travel.
    - The robot must reach a box to push it.

    # Heuristic Initialization
    - Builds an undirected graph of locations based on `adjacent` facts.
    - Pre-calculates all-pairs shortest paths between locations using BFS.
    - Extracts the goal location for each box from the task goals.

    # Step-by-Step Thinking for Computing the Heuristic Value
    1. Identify the current location of the robot.
    2. Identify the current location of each box.
    3. For each box:
       - Check if it is at its goal location.
       - If not, add the pre-calculated shortest distance from its current location
         to its goal location to a running total (sum of box distances).
       - Keep track of which boxes still need to be moved.
    4. If all boxes are at their goals, the heuristic is 0.
    5. If boxes still need to be moved, find the minimum pre-calculated shortest
       distance from the robot's current location to the current location of
       any box that still needs to be moved.
    6. The total heuristic value is the sum of box distances plus this minimum
       robot-to-box distance.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting:
        - Goal locations for each box.
        - Static facts (`adjacent` relationships).
        - Pre-calculating all-pairs shortest paths.
        """
        self.goals = task.goals
        static_facts = task.static

        # Build the location graph from adjacent facts.
        # Treat adjacent relations as undirected for distance calculation.
        self.graph = collections.defaultdict(list)
        all_locations = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if match(fact, "adjacent", "*", "*", "*"):
                loc1, loc2, _ = parts[1:]
                self.graph[loc1].append(loc2)
                self.graph[loc2].append(loc1) # Add reverse edge for undirected graph
                all_locations.add(loc1)
                all_locations.add(loc2)

        # Pre-calculate all-pairs shortest paths using BFS.
        self.distances = {}
        for start_loc in all_locations:
            self.distances[start_loc] = bfs(self.graph, start_loc)

        # Store goal locations for each box.
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                box, location = args
                self.goal_locations[box] = location

    def get_distance(self, loc1, loc2):
        """Helper to get pre-calculated distance, returns infinity if unreachable."""
        if loc1 not in self.distances or loc2 not in self.distances[loc1]:
            return float('inf')
        return self.distances[loc1][loc2]

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Find robot location
        robot_loc = None
        for fact in state:
            if match(fact, "at-robot", "*"):
                robot_loc = get_parts(fact)[1]
                break
        if robot_loc is None:
             # This should not happen in a valid Sokoban state, but handle defensively
             return float('inf') # Robot not found? Problematic state.

        # Find current box locations
        current_box_locations = {}
        for fact in state:
            if match(fact, "at", "*", "*"):
                obj_name, loc_name = get_parts(fact)[1:]
                # Check if the object is a box (assuming all objects starting with 'box' are boxes)
                # A more robust way would be to parse types from the domain, but this is simpler
                # given the example instance names.
                if obj_name.startswith("box"):
                     current_box_locations[obj_name] = loc_name

        total_box_distance = 0
        boxes_to_move = []

        # Calculate sum of distances for boxes not at goal
        for box_name, goal_loc in self.goal_locations.items():
            current_loc = current_box_locations.get(box_name) # Use .get for safety

            if current_loc is None:
                 # Box not found in state? Problematic state.
                 return float('inf')

            if current_loc != goal_loc:
                dist = self.get_distance(current_loc, goal_loc)
                if dist == float('inf'):
                    # Box goal is unreachable
                    return float('inf')
                total_box_distance += dist
                boxes_to_move.append(box_name)

        # If all boxes are at their goals, heuristic is 0
        if not boxes_to_move:
            return 0

        # Calculate minimum distance from robot to any box that needs moving
        min_robot_dist = float('inf')
        for box_name in boxes_to_move:
            box_loc = current_box_locations[box_name]
            dist = self.get_distance(robot_loc, box_loc)
            min_robot_dist = min(min_robot_dist, dist)

        # If robot cannot reach any box that needs moving
        if min_robot_dist == float('inf'):
             return float('inf')

        # Total heuristic is sum of box distances + minimum robot distance to a box
        return total_box_distance + min_robot_dist

