import collections
import math
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

def bfs_shortest_path(graph, start_node):
    """
    Performs Breadth-First Search to find the shortest path distance
    from a start node to all other reachable nodes in a graph.

    Args:
        graph: An adjacency list representation of the graph (dict: node -> set of neighbors).
        start_node: The node to start the BFS from.

    Returns:
        A dictionary mapping reachable nodes to their shortest distance from start_node.
        Returns an empty dictionary if start_node is not in the graph.
    """
    if start_node not in graph:
        return {}

    distances = {start_node: 0}
    queue = collections.deque([start_node])

    while queue:
        current_node = queue.popleft()

        # Check if current_node is in graph before accessing neighbors
        if current_node in graph:
            for neighbor in graph[current_node]:
                if neighbor not in distances:
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)

    return distances

class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the number of actions needed to reach a goal state
    by summing the shortest path distances of each box to its goal location
    and adding the shortest path distance of the robot to the nearest box
    that needs to be moved.

    # Assumptions:
    - The grid structure is defined by `adjacent` predicates.
    - Shortest path distances on this grid approximate the cost of movement.
    - The cost of getting the robot into position to push a box is implicitly
      covered by the robot-to-box distance and the box-to-goal distance.
    - Interactions between boxes (blocking each other) and complex robot
      maneuvering (getting around boxes, clearing paths) are largely ignored
      for computational efficiency, making the heuristic non-admissible.
    - All goal locations are assumed to be reachable from the initial box/robot
      locations in solvable problems. Unreachable goals/boxes will result in
      a very large heuristic value.

    # Heuristic Initialization
    - Builds an undirected graph from `adjacent` facts.
    - Precomputes all-pairs shortest paths using BFS.
    - Extracts goal locations for each box.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the current location of the robot and all boxes.
    2. Identify which boxes are not yet at their goal locations.
    3. If all boxes are at their goals, the heuristic is 0.
    4. Calculate the shortest path distance from the robot's current location
       to the location of *each* box that needs to be moved. Find the minimum
       of these distances. This estimates the cost for the robot to reach
       a box it needs to interact with.
    5. For each box that needs to be moved, calculate the shortest path distance
       from its current location to its goal location. This estimates the
       minimum number of pushes required for that box (ignoring obstacles).
    6. Sum the box-to-goal distances for all boxes that need moving.
    7. The total heuristic value is the minimum robot-to-box distance (step 4)
       plus the sum of box-to-goal distances (step 6).
    8. Use precomputed shortest paths (from __init__) for distance lookups.
    9. Handle cases where locations are unreachable by returning a large value (infinity).
    """

    def __init__(self, task):
        """
        Initialize the heuristic by building the location graph,
        precomputing shortest paths, and extracting goal locations.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # Build the location graph from adjacent facts.
        # The graph is undirected for distance calculation purposes.
        self.graph = collections.defaultdict(set)
        all_locations = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "adjacent":
                loc1, loc2, direction = parts[1:]
                self.graph[loc1].add(loc2)
                self.graph[loc2].add(loc1) # Add reverse edge for undirected graph
                all_locations.add(loc1)
                all_locations.add(loc2)

        # Ensure all locations mentioned in the problem (even if isolated) are keys in the graph
        # This prevents errors if a location exists but has no adjacencies.
        for loc in all_locations:
             if loc not in self.graph:
                 self.graph[loc] = set()


        # Precompute all-pairs shortest paths using BFS from every location.
        self.distances = {}
        for start_loc in self.graph:
            self.distances[start_loc] = bfs_shortest_path(self.graph, start_loc)

        # Store goal locations for each box.
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                box, location = args
                self.goal_locations[box] = location

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Find robot and box locations in the current state.
        robot_loc = None
        box_locations = {}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at-robot":
                robot_loc = parts[1]
            elif parts[0] == "at" and parts[1] in self.goal_locations: # Only track boxes that have a goal
                 box, loc = parts[1:]
                 box_locations[box] = loc

        # Identify boxes that are not at their goal locations.
        boxes_to_move = {
            box for box, loc in box_locations.items()
            if box in self.goal_locations and loc != self.goal_locations[box]
        }

        # If all relevant boxes are at their goals, the heuristic is 0.
        if not boxes_to_move:
            return 0

        # Calculate minimum distance from robot to any box that needs moving.
        min_robot_dist = math.inf
        robot_reachable = False
        if robot_loc in self.distances:
            for box in boxes_to_move:
                box_loc = box_locations[box]
                if box_loc in self.distances[robot_loc]:
                    min_robot_dist = min(min_robot_dist, self.distances[robot_loc][box_loc])
                    robot_reachable = True

        # If the robot cannot reach any box that needs moving, the state is likely a deadlock or unsolvable.
        if not robot_reachable:
             return math.inf # Or a very large number

        # Calculate the sum of distances for each box to its goal.
        sum_box_distances = 0
        for box in boxes_to_move:
            box_loc = box_locations[box]
            goal_loc = self.goal_locations[box]

            # Check if box_loc and goal_loc are in the precomputed distances table
            if box_loc in self.distances and goal_loc in self.distances[box_loc]:
                 sum_box_distances += self.distances[box_loc][goal_loc]
            else:
                 # If a box cannot reach its goal, the state is likely a deadlock or unsolvable.
                 return math.inf # Or a very large number

        # The heuristic is the cost to get the robot to a relevant box
        # plus the estimated cost to move all boxes to their goals.
        # This is non-admissible as it ignores robot movement costs between pushing different boxes
        # and the cost of maneuvering the robot around boxes.
        return min_robot_dist + sum_box_distances

