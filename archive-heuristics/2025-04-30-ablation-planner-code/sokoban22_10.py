from fnmatch import fnmatch
from collections import deque
import math

# Assume Heuristic base class is available in the environment
# from heuristics.heuristic_base import Heuristic

# Define a dummy Heuristic base class for standalone testing if needed
# In a real planning environment, this would be provided by the framework.
class Heuristic:
    def __init__(self, task):
        pass
    def __call__(self, node):
        raise NotImplementedError


# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Use the simple zip-based check as seen in example heuristics
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def loc_str_to_tuple(loc_str):
    """Converts a location string like 'loc_R_C' to a tuple (R, C)."""
    parts = loc_str.split('_')
    if len(parts) == 3 and parts[0] == 'loc':
        try:
            return (int(parts[1]), int(parts[2]))
        except ValueError:
            pass # Not a valid loc_R_C format
    return None # Handle cases that don't match expected format

def loc_tuple_to_str(loc_tuple):
    """Converts a location tuple (R, C) back to a string like 'loc_R_C'."""
    return f'loc_{loc_tuple[0]}_{loc_tuple[1]}'

def bfs(start_node, goal_node, graph, is_traversable_func):
    """
    Performs BFS on the graph to find the shortest path distance.
    graph: dict mapping node to set of adjacent nodes.
    is_traversable_func: function that takes a node and returns True if traversable.
    Returns distance or float('inf') if goal is unreachable.
    """
    if start_node == goal_node:
        return 0

    # Ensure start_node is in the graph. Traversability of start_node itself
    # is handled by the caller (e.g., robot's current location doesn't need to be clear).
    if start_node not in graph:
         return float('inf')

    queue = deque([(start_node, 0)])
    visited = {start_node}

    while queue:
        current_node, dist = queue.popleft()

        neighbors = graph.get(current_node, set())
        for neighbor in neighbors:
            # Check if neighbor is in the graph, traversable, and not visited
            if neighbor in graph and is_traversable_func(neighbor) and neighbor not in visited:
                if neighbor == goal_node:
                    return dist + 1
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))

    return float('inf') # Goal not reachable


class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing two components:
    1. The minimum number of pushes required for each misplaced box to reach its goal,
       calculated as the shortest path distance on the static grid graph (ignoring obstacles).
    2. The minimum number of robot moves required to reach any location adjacent
       to any misplaced box, calculated as the shortest path distance on the
       currently traversable grid graph (considering clear locations).

    # Assumptions
    - The grid structure is implied by the 'loc_R_C' naming convention.
    - Adjacency defined by 'adjacent' facts corresponds to grid movement and is symmetric.
    - The minimum number of pushes for a box is its shortest path distance on the static graph.
    - The robot needs to reach a location adjacent to a box to push it.
    - Action costs are uniform (implicitly 1).

    # Heuristic Initialization
    - Extracts goal locations for each box.
    - Builds the static adjacency graph from 'adjacent' facts, mapping location strings to (row, col) tuples.
    - Creates mappings between location strings and (row, col) tuples for all relevant locations.

    # Step-By-Step Thinking for Computing Heuristic
    1. Parse the current state to find the robot's location and the location of each box.
    2. Convert location strings to (row, col) tuples using the precomputed map.
    3. Identify which boxes are not at their goal locations (misplaced boxes).
    4. If no boxes are misplaced, the heuristic is 0.
    5. For each misplaced box:
       - Calculate the shortest path distance from its current location to its goal location
         using BFS on the *static* adjacency graph (ignoring current obstacles). Sum these distances.
       - If any box is unreachable from its goal on the static graph, the state is likely
         unsolvable, return infinity.
    6. Find the set of all locations that are adjacent to any of the misplaced boxes
       (based on the static graph). These are potential robot push positions.
    7. Calculate the minimum shortest path distance from the robot's current location
       to any of these potential robot push positions using a multi-goal BFS on the
       graph considering only locations marked as 'clear' in the current state as
       traversable for the robot.
       - If the robot cannot reach any push position, return infinity.
    8. The total heuristic value is the sum of the total box distance (step 5) and
       the minimum robot distance to a push position (step 7).
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and static facts.
        Builds the static graph of locations and location mappings.
        """
        self.goals = task.goals
        static_facts = task.static

        # Collect all locations mentioned in adjacent facts and goal facts
        all_loc_strs = set()
        for fact in static_facts:
            if match(fact, "adjacent", "*", "*", "*"):
                _, loc1_str, loc2_str, _ = get_parts(fact)
                all_loc_strs.add(loc1_str)
                all_loc_strs.add(loc2_str)
        for goal in self.goals:
             if match(goal, "at", "*", "*"):
                 _, _, loc_str = get_parts(goal)
                 all_loc_strs.add(loc_str)

        # Create mappings for all relevant locations
        self.loc_str_to_tuple_map = {}
        self.loc_tuple_to_str_map = {}
        self.graph_static = {}
        for loc_str in sorted(list(all_loc_strs)): # Sort for consistent mapping
             loc_tuple = loc_str_to_tuple(loc_str)
             if loc_tuple:
                 self.loc_str_to_tuple_map[loc_str] = loc_tuple
                 self.loc_tuple_to_str_map[loc_tuple] = loc_str
                 self.graph_static[loc_tuple] = set() # Initialize graph nodes

        # Populate the static graph with edges (assuming adjacency is symmetric)
        for fact in static_facts:
            if match(fact, "adjacent", "*", "*", "*"):
                _, loc1_str, loc2_str, _ = get_parts(fact)
                loc1_tuple = self.loc_str_to_tuple_map.get(loc1_str)
                loc2_tuple = self.loc_str_to_tuple_map.get(loc2_str)
                if loc1_tuple and loc2_tuple:
                    self.graph_static[loc1_tuple].add(loc2_tuple)
                    self.graph_static[loc2_tuple].add(loc1_tuple) # Add reverse edge

        # Store goal locations for each box as tuples
        self.box_goals_tuple = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                box, location_str = args
                loc_tuple = self.loc_str_to_tuple_map.get(location_str)
                if loc_tuple:
                    self.box_goals_tuple[box] = loc_tuple


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Parse current state facts into strings first
        robot_loc_str = None
        box_locations_str = {}
        clear_locs_str = set()

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at-robot":
                robot_loc_str = parts[1]
            elif parts[0] == "at" and parts[1] in self.box_goals_tuple: # Only track boxes we care about
                 box = parts[1]
                 loc_str = parts[2]
                 box_locations_str[box] = loc_str
            elif parts[0] == "clear":
                 loc_str = parts[1]
                 clear_locs_str.add(loc_str)

        # Convert location strings to tuples using the precomputed map
        robot_loc_tuple = self.loc_str_to_tuple_map.get(robot_loc_str) if robot_loc_str else None
        box_locations_tuple = {
            box: self.loc_str_to_tuple_map.get(loc_str)
            for box, loc_str in box_locations_str.items()
            if self.loc_str_to_tuple_map.get(loc_str) is not None
        }
        clear_locs_tuple = {
            self.loc_str_to_tuple_map.get(loc_str)
            for loc_str in clear_locs_str
            if self.loc_str_to_tuple_map.get(loc_str) is not None
        }

        # Check if robot location was found and is a known location
        if robot_loc_tuple is None or robot_loc_tuple not in self.graph_static:
             return float('inf') # Robot is in an unknown or invalid location

        # Identify misplaced boxes and calculate total static box distance
        total_box_dist = 0
        misplaced_boxes = []

        # Function for static BFS (all locations in graph are traversable for box distance)
        is_box_traversable_static = lambda loc: loc in self.graph_static

        for box, goal_loc_tuple in self.box_goals_tuple.items():
            current_loc_tuple = box_locations_tuple.get(box)

            # If box is not found or not at goal
            if current_loc_tuple is None or current_loc_tuple != goal_loc_tuple:
                misplaced_boxes.append(box)
                if current_loc_tuple is None or current_loc_tuple not in self.graph_static:
                     # Box is missing or in an unknown/invalid location
                     return float('inf')

                # Calculate static distance for the box
                dist = bfs(current_loc_tuple, goal_loc_tuple, self.graph_static, is_box_traversable_static)

                if dist == float('inf'):
                    # Box cannot reach its goal even on the static graph
                    return float('inf')
                total_box_dist += dist

        # If all boxes are at their goals, the heuristic is 0
        if not misplaced_boxes:
            return 0

        # Calculate minimum robot distance to a push position
        min_robot_dist = float('inf')
        potential_push_locs_tuple = set()

        # Find all locations adjacent to any misplaced box (in the static graph)
        for box in misplaced_boxes:
            box_loc_tuple = box_locations_tuple[box]
            # Add all neighbors of the box location in the static graph
            potential_push_locs_tuple.update(self.graph_static.get(box_loc_tuple, set()))

        # Function for robot BFS (only clear locations are traversable)
        is_robot_traversable_state = lambda loc: loc in clear_locs_tuple

        # Calculate minimum distance from robot to any potential push location
        # Use a multi-goal BFS starting from the robot
        # The robot's current location doesn't need to be clear to start BFS from it.
        queue = deque([(robot_loc_tuple, 0)])
        visited = {robot_loc_tuple}
        found_min_robot_dist = float('inf')

        while queue:
            current_loc, dist = queue.popleft()

            # If the current location is one of the potential push locations
            if current_loc in potential_push_locs_tuple:
                 found_min_robot_dist = dist
                 break # Found the minimum distance to *any* push location

            neighbors = self.graph_static.get(current_loc, set())
            for neighbor in neighbors:
                 # Robot can move to neighbor if it's in the static graph, is clear in the state, and not visited
                 if neighbor in self.graph_static and is_robot_traversable_state(neighbor) and neighbor not in visited:
                     visited.add(neighbor)
                     queue.append((neighbor, dist + 1))

        min_robot_dist = found_min_robot_dist

        if min_robot_dist == float('inf'):
            # Robot cannot reach any location adjacent to any misplaced box
            return float('inf')

        # Total heuristic is sum of box distances + minimum robot distance to get started
        return total_box_dist + min_robot_dist

