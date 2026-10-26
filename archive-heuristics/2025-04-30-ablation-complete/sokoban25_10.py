# Required imports
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque

# Helper functions from Logistics example (adapted for general use)
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact string or malformed fact
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at-robot loc_1_1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure the number of parts in the fact matches the number of arguments in the pattern
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# BFS function to compute shortest path distances
def bfs(start_node, graph):
    """
    Computes shortest path distances from a start_node to all reachable nodes
    in a graph using Breadth-First Search.

    Args:
        start_node: The node to start the BFS from.
        graph: A dictionary representing the graph where keys are nodes
               and values are lists of adjacent nodes (successors).

    Returns:
        A dictionary where keys are reachable nodes and values are their
        shortest distance from the start_node. Returns an empty dictionary
        if start_node is not in the graph.
    """
    if start_node not in graph:
        return {} # Start node not in graph

    distances = {start_node: 0}
    queue = deque([start_node])
    visited = {start_node}

    while queue:
        current_node = queue.popleft()

        # Check if current_node exists in graph keys before iterating neighbors
        # This check is technically redundant if start_node was in graph, but safe.
        if current_node in graph:
            for neighbor in graph[current_node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)

    return distances

class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal state by summing,
    for each box not at its goal location, the shortest path distance
    from the box's current location to its goal location, plus the shortest
    path distance from the robot's current location to the box's current location.
    The distances are computed on the location graph defined by the 'adjacent'
    predicates, representing possible robot movement paths.

    # Assumptions
    - The location graph defined by 'adjacent' predicates is connected (or at least,
      all locations mentioned in problem instances are within a connected component).
    - The shortest path distance on this graph is a reasonable, albeit non-admissible,
      proxy for the number of pushes required for a box (as each push moves a box
      one step along an adjacent path) and the number of moves required for the robot
      to reach a box.
    - The goal only specifies the locations of boxes. The robot's final position
      does not affect goal satisfaction.

    # Heuristic Initialization
    - Extracts the goal locations for each box from the task's goal conditions.
    - Builds the location graph based on 'adjacent' static facts. This graph
      includes all locations mentioned in 'adjacent' facts and edges for every
      specified adjacency.
    - Precomputes all-pairs shortest path distances on the built location graph
      using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Check if the current state is the goal state by comparing the state facts
       against the task's goal facts. If it is the goal state, return 0.
    2. If not the goal state, initialize the total heuristic cost to 0.
    3. Find the current location of the robot by searching for the `(at-robot ?l)` fact in the state. If the robot's location cannot be found, return infinity (as the state is likely invalid or a dead end).
    4. Find the current location of each box by searching for `(at ?b ?l)` facts in the state.
    5. Iterate through each box that has a specified goal location:
       a. Determine the box's current location from the state and its goal location from the precomputed goals.
       b. If the box's current location is known and is not the same as its goal location:
          i. Look up the precomputed shortest path distance from the box's current location to its goal location. Check if the box's current location is a valid node in the graph. If not, or if no path exists to the goal location (distance is None), this box cannot reach its goal, and the state is likely a dead end. Return infinity. Otherwise, add this distance to the total cost. This estimates the minimum number of pushes required for this box.
          ii. Look up the precomputed shortest path distance from the robot's current location to the box's current location. Check if the robot's current location is a valid node in the graph. If not, or if no path exists to the box's current location (distance is None), the robot cannot reach the box, and the state is likely a dead end. Return infinity. Otherwise, add this distance to the total cost. This estimates the robot movement needed to reach this box to start pushing it.
    6. After iterating through all goal boxes, return the accumulated total heuristic cost. This value will be greater than 0 for any non-goal state where boxes needing movement are reachable by the box and the robot.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and building
        the location graph for distance calculations.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # Store goal locations for each box.
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            # Goal facts are typically '(at box_name loc_name)'
            if predicate == "at" and len(args) == 2:
                obj_name, loc_name = args
                # Assume objects in 'at' goals are boxes
                self.goal_locations[obj_name] = loc_name

        # Build the location graph from adjacent facts.
        # The graph represents possible moves for the robot and possible single-step
        # movements for boxes when pushed.
        self.location_graph = {}
        all_locations = set()

        # First pass: collect all locations mentioned in adjacent facts
        for fact in static_facts:
            if match(fact, "adjacent", "*", "*", "*"):
                _, loc1, loc2, _ = get_parts(fact)
                all_locations.add(loc1)
                all_locations.add(loc2)

        # Initialize graph with all collected locations
        for loc in all_locations:
            self.location_graph[loc] = []

        # Second pass: add directed edges based on adjacency
        for fact in static_facts:
            if match(fact, "adjacent", "*", "*", "*"):
                _, loc1, loc2, _ = get_parts(fact)
                # Add directed edge from loc1 to loc2. This edge exists if robot
                # can move loc1 -> loc2, or box can be pushed from some loc0
                # through loc1 to loc2. For simple graph distance, we just use
                # the adjacency relation.
                self.location_graph[loc1].append(loc2)

        # Precompute all-pairs shortest path distances using BFS
        self.all_distances = {}
        for start_loc in self.location_graph:
            self.all_distances[start_loc] = bfs(start_loc, self.location_graph)

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # 1. Check for goal state
        if self.goals <= state:
            return 0

        # 2. Initialize cost
        total_cost = 0

        # 3. Find current robot location
        robot_location = None
        for fact in state:
            if match(fact, "at-robot", "*"):
                robot_location = get_parts(fact)[1]
                break

        if robot_location is None:
             # Robot location is essential for progress. If not found, state is invalid/dead end.
             return float('inf')

        # 4. Find current box locations
        current_box_locations = {}
        for fact in state:
            if match(fact, "at", "*", "*"):
                obj_name, loc_name = get_parts(fact)[1:]
                # Assume any object with 'at' predicate that is not the robot is a box
                # This relies on robot not being an object of type 'box' and
                # not having an '(at obj robot_loc)' fact. The domain confirms this.
                current_box_locations[obj_name] = loc_name

        # 5. Iterate through goal boxes and sum costs
        for box, goal_location in self.goal_locations.items():
            current_location = current_box_locations.get(box)

            # If box location is unknown or box is already at goal, cost is 0 for this box's goal component
            if current_location is None or current_location == goal_location:
                continue

            # Calculate box distance to goal
            # Check if current_location is a valid node in the graph before accessing distances
            if current_location not in self.all_distances:
                 # Location exists in state but not in graph (e.g., isolated location). Dead end.
                 return float('inf')

            box_dist_to_goal = self.all_distances[current_location].get(goal_location)

            if box_dist_to_goal is None:
                # Box goal is unreachable from its current location on the graph. Dead end.
                return float('inf')

            total_cost += box_dist_to_goal

            # Calculate robot distance to box
            # Check if robot_location is a valid node in the graph before accessing distances
            if robot_location not in self.all_distances:
                 # Robot location exists in state but not in graph. Dead end.
                 return float('inf')

            robot_dist_to_box = self.all_distances[robot_location].get(current_location)

            if robot_dist_to_box is None:
                 # Robot cannot reach the box's location on the graph. Dead end.
                 return float('inf')

            total_cost += robot_dist_to_box

        # 6. Return total cost
        # If we reached here and total_cost is 0, it means all boxes are at their goals,
        # but the initial goal state check failed. This shouldn't happen with the
        # provided domain structure where goals are only box locations.
        # If it could happen (e.g., goal included robot position), we'd need to add
        # robot goal cost here. But based on examples, this is not needed.
        # The heuristic is > 0 for non-goal states where boxes need moving and are reachable.
        return total_cost
