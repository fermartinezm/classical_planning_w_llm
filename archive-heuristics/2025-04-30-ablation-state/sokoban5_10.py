from fnmatch import fnmatch
from collections import deque

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Assumes fact is a string like "(predicate arg1 arg2)"
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at ball1 rooma)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Use zip to handle potential length differences gracefully
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def bfs_distance(start_node, end_node, graph):
    """
    Finds the shortest path distance between start_node and end_node
    in the given graph using BFS.
    Graph is represented as an adjacency dictionary: node -> list of neighbors.
    Returns float('inf') if no path exists.
    """
    if start_node == end_node:
        return 0

    queue = deque([(start_node, 0)])
    visited = {start_node}
    distances = {start_node: 0}

    while queue:
        current_node, current_dist = queue.popleft()

        # Optimization: If we reached the target, return distance immediately
        if current_node == end_node:
            return current_dist

        # Check if current_node exists in the graph keys
        if current_node in graph:
            for neighbor in graph[current_node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    distances[neighbor] = current_dist + 1
                    queue.append((neighbor, current_dist + 1))

    # If BFS completes without finding the end_node
    return float('inf')

def bfs_distances_from_start(start_node, graph):
    """
    Finds the shortest path distance from start_node to all reachable nodes
    in the given graph using BFS.
    Graph is represented as an adjacency dictionary: node -> list of neighbors.
    Returns a dictionary: node -> distance.
    """
    queue = deque([(start_node, 0)])
    visited = {start_node}
    distances = {start_node: 0}

    # Ensure start_node is in graph keys, even if it has no neighbors
    if start_node not in graph:
        # This case should ideally be handled during graph construction in __init__
        # to ensure all known locations are keys, but this adds robustness.
        graph[start_node] = []

    queue.append((start_node, 0))

    while queue:
        current_node, current_dist = queue.popleft()

        # Check if current_node exists in the graph keys
        if current_node in graph:
            for neighbor in graph[current_node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    distances[neighbor] = current_dist + 1
                    queue.append((neighbor, current_dist + 1))

    return distances


class sokobanHeuristic: # Inherit from Heuristic if applicable in the target environment, e.g., class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal state.
    It is calculated as the sum of the shortest path distances for each box
    from its current location to its goal location, plus the shortest path
    distance from the robot to the nearest box that is not yet at its goal.
    Distances are computed using Breadth-First Search (BFS) on the graph
    of locations defined by the 'adjacent' facts.

    # Assumptions
    - The goal state specifies the target location for each box using the
      '(at ?b ?l)' predicate.
    - The locations form a connected graph (or relevant parts are connected).
    - The 'adjacent' facts define the traversable paths for both the robot
      (via 'move' action) and for pushing boxes (via 'push' action).
    - The heuristic does not account for potential deadlocks or complex
      interactions between multiple boxes or obstacles.
    - The cost of a 'move' action and a 'push' action is implicitly 1.

    # Heuristic Initialization
    - The goal locations for each box are extracted from the task's goal conditions.
    - A graph representing the locations and their adjacencies is built from
      the static 'adjacent' facts. This graph is used for BFS distance calculations.
    - A set of all location names is stored (though not strictly used in __call__ currently).

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current location of the robot.
    2. Identify the current location of each box whose goal is known.
    3. Initialize the total heuristic value to 0.
    4. Identify which boxes are not currently at their respective goal locations.
    5. For each box that is not at its goal:
       a. Get the box's current location and its goal location.
       b. Calculate the shortest path distance between the box's current location
          and its goal location using BFS on the pre-computed location graph.
       c. Add this distance to the total heuristic value. This represents the
          minimum number of pushes required for this box in isolation.
          If the goal is unreachable for any box, the heuristic is infinity.
    6. If there are any boxes not at their goals:
       a. Calculate the shortest path distances from the robot's current location
          to all reachable locations using a single BFS.
       b. Find the minimum distance among these results to the current location
          of any box that is not at its goal.
       c. Add this minimum distance to the total heuristic value. This represents
          the cost for the robot to reach a box it needs to move.
          If no such box is reachable by the robot, the heuristic is infinity.
    7. Return the total heuristic value. If all boxes are at their goals, the
       heuristic is 0.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and building
        the location graph from static facts.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # Store goal locations for each box.
        # Assuming goal is a conjunction of (at box loc) facts.
        self.goal_locations = {}
        for goal in self.goals:
            # Check if the goal fact is an 'at' predicate
            if match(goal, "at", "*", "*"):
                predicate, box, location = get_parts(goal)
                self.goal_locations[box] = location

        # Build the location graph from 'adjacent' facts.
        # Graph is represented as an adjacency dictionary: location -> list of neighbors.
        self.location_graph = {}
        self.all_locations = set() # Keep track of all location names

        for fact in static_facts:
            # Check if the fact is an 'adjacent' predicate
            if match(fact, "adjacent", "*", "*", "*"):
                predicate, loc1, loc2, direction = get_parts(fact)
                self.all_locations.add(loc1)
                self.all_locations.add(loc2)

                # Add directed edge loc1 -> loc2
                if loc1 not in self.location_graph:
                    self.location_graph[loc1] = []
                self.location_graph[loc1].append(loc2)

                # Add directed edge loc2 -> loc1 to make it undirected
                if loc2 not in self.location_graph:
                     self.location_graph[loc2] = []
                self.location_graph[loc2].append(loc1)

        # Ensure all locations mentioned in the graph are keys in the graph dict,
        # even if they have no outgoing edges (e.g., dead ends).
        for loc in self.all_locations:
             if loc not in self.location_graph:
                  self.location_graph[loc] = []


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state (frozenset of facts).

        # Find robot's current location
        robot_location = None
        for fact in state:
            if match(fact, "at-robot", "*"):
                predicate, loc = get_parts(fact)
                robot_location = loc
                break # Assuming only one robot

        if robot_location is None:
             # This should not happen in a valid Sokoban state, but handle defensively
             return float('inf') # Robot location unknown or not found

        # Find current location of each box relevant to the goal
        current_box_locations = {}
        for fact in state:
            if match(fact, "at", "*", "*"):
                 predicate, obj, loc = get_parts(fact)
                 # Only consider objects that are boxes and have a goal location
                 if obj in self.goal_locations:
                    current_box_locations[obj] = loc

        total_cost = 0
        boxes_not_at_goal = []

        # Calculate box-to-goal distances
        for box, goal_loc in self.goal_locations.items():
            current_loc = current_box_locations.get(box) # Use .get for safety

            if current_loc is None:
                 # Box location unknown - problem state is invalid or incomplete
                 # Assuming 'at' fact exists for all boxes in state.
                 return float('inf') # Should not happen in valid states

            if current_loc != goal_loc:
                boxes_not_at_goal.append(box)
                # Calculate distance from current box location to goal location
                dist = bfs_distance(current_loc, goal_loc, self.location_graph)
                if dist == float('inf'):
                    # Box is in a location from which its goal is unreachable
                    return float('inf') # This state is likely a dead end or unsolvable
                total_cost += dist

        # Calculate robot-to-nearest-box distance if there are boxes not at goal
        if boxes_not_at_goal:
            # Use multi-target BFS from robot location to find distances to all reachable nodes
            robot_distances = bfs_distances_from_start(robot_location, self.location_graph)

            min_robot_to_box_dist = float('inf')
            for box in boxes_not_at_goal:
                box_loc = current_box_locations[box]
                # Get distance from the pre-computed distances.
                # If box_loc is not in robot_distances, it's unreachable from the robot.
                dist = robot_distances.get(box_loc, float('inf'))
                min_robot_to_box_dist = min(min_robot_to_box_dist, dist)

            if min_robot_to_box_dist == float('inf'):
                 # Robot cannot reach any box that needs moving
                 return float('inf') # This state is likely a dead end or unsolvable

            total_cost += min_robot_to_box_dist

        # The heuristic is the sum of box distances + robot distance.
        # If total_cost is 0, it means sum of box distances is 0 (all boxes at goal)
        # and either there were no boxes not at goal (so robot distance part skipped)
        # or min_robot_to_box_dist was 0 (robot is at a box location, but this only happens if boxes_not_at_goal is not empty).
        # If boxes_not_at_goal is not empty, min_robot_to_box_dist is added.
        # If all boxes are at goal, boxes_not_at_goal is empty, total_cost remains 0.
        # If not all boxes are at goal, total_cost > 0 (assuming reachability).
        # So, h=0 iff goal is reached.

        return total_cost
