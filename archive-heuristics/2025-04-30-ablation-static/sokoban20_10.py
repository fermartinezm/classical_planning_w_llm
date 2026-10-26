from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at ball1 room1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Basic check for number of parts if no wildcards are used in args
    if len(parts) != len(args) and '*' not in args:
         return False
    # Check if each part matches the corresponding arg pattern
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def parse_location_name(loc_str):
    """Parses a location string like 'loc_R_C' into a tuple (R, C)."""
    parts = loc_str.split('_')
    if len(parts) == 3 and parts[0] == 'loc':
        try:
            row = int(parts[1])
            col = int(parts[2])
            return (row, col)
        except ValueError:
            # Handle cases where R or C are not valid integers
            return None
    # Handle cases where the string format is unexpected
    return None

def bfs_distance(start_loc_tuple, end_loc_tuple, graph):
    """
    Calculates the shortest path distance between two location tuples using BFS.
    Returns float('inf') if the end location is unreachable within the graph.
    """
    if start_loc_tuple == end_loc_tuple:
        return 0

    # Ensure start and end locations are in the graph
    if start_loc_tuple not in graph or end_loc_tuple not in graph:
        return float('inf')

    queue = deque([start_loc_tuple])
    visited = {start_loc_tuple}
    distance = {start_loc_tuple: 0}

    while queue:
        current_loc = queue.popleft()

        if current_loc == end_loc_tuple:
            return distance[current_loc]

        # Check if current_loc has neighbors in the graph
        if current_loc in graph:
            for neighbor in graph[current_loc]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
                    distance[neighbor] = distance[current_loc] + 1

    return float('inf') # End location is unreachable from start location


class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the number of actions required by summing the
    shortest path distances for each box from its current location to its
    goal location. The distance is calculated on the grid graph defined by
    the 'adjacent' predicates. This heuristic ignores the robot's position
    and obstacles (other boxes or walls) when calculating box distances,
    making it non-admissible but potentially useful for greedy search.

    # Assumptions
    - Locations are named in the format 'loc_R_C' allowing parsing into (row, col) tuples.
    - The 'adjacent' predicates define a graph of reachable locations.
    - The goal is defined solely by the final locations of all boxes.
    - The heuristic value is 0 only when all boxes are at their goal locations.

    # Heuristic Initialization
    - Parses location names from 'loc_R_C' format into (row, col) tuples for all locations
      mentioned in 'adjacent' facts.
    - Builds an undirected adjacency graph of locations based on 'adjacent' facts,
      using the (row, col) tuples as nodes.
    - Extracts the goal location string for each box from the task's goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    1. Check if the current state is a goal state by verifying if all goal predicates
       (which are assumed to be 'at' facts for boxes) are present in the state. If it is,
       return 0.
    2. If the state is not a goal state, find the current location string for each box
       by iterating through the state facts.
    3. Initialize a total distance counter to 0.
    4. For each box that has a goal location defined:
       a. Get the box's current location string and its goal location string.
       b. If the box is already at its goal location, skip it.
       c. Convert the current and goal location strings into (row, col) tuples using
          the mapping created during initialization.
       d. If either the current or goal location tuple is not found in the graph
          (meaning it's not a location connected by 'adjacent' facts), the goal is
          unreachable from this location within the defined grid structure; return
          infinity.
       e. Calculate the shortest path distance between the box's current (row, col)
          and its goal (row, col) using BFS on the pre-built location graph.
       f. If the BFS returns infinity (goal is unreachable from the current location
          on the graph), return infinity for the heuristic value.
       g. Add the calculated distance to the total distance counter.
    5. Return the total sum of distances for all misplaced boxes.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by building the location graph and extracting goal locations.
        """
        self.goals = task.goals  # Goal conditions (frozenset of facts)
        static_facts = task.static  # Static facts (frozenset of facts)

        # Map location strings to (row, col) tuples
        self.loc_str_to_tuple = {}

        # Build the graph: {(r, c): [(r_adj1, c_adj1), ...]}
        self.graph = {}

        # Extract all locations mentioned in adjacent facts
        all_loc_strs = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'adjacent' and len(parts) == 4:
                loc1_str, loc2_str, direction = parts[1:]
                all_loc_strs.add(loc1_str)
                all_loc_strs.add(loc2_str)

        # Parse all relevant location strings into tuples and build the string-to-tuple map
        for loc_str in all_loc_strs:
             loc_tuple = parse_location_name(loc_str)
             if loc_tuple:
                 self.loc_str_to_tuple[loc_str] = loc_tuple

        # Populate the graph using tuples, only for locations found and parsed
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'adjacent' and len(parts) == 4:
                loc1_str, loc2_str, direction = parts[1:]
                loc1_tuple = self.loc_str_to_tuple.get(loc1_str)
                loc2_tuple = self.loc_str_to_tuple.get(loc2_str)

                # Only add edges if both locations were successfully parsed and mapped
                if loc1_tuple and loc2_tuple:
                    if loc1_tuple not in self.graph:
                        self.graph[loc1_tuple] = []
                    if loc2_tuple not in self.graph:
                        self.graph[loc2_tuple] = []
                    # Add undirected edge
                    self.graph[loc1_tuple].append(loc2_tuple)
                    self.graph[loc2_tuple].append(loc1_tuple)

        # Remove duplicate neighbors in graph (adj facts might be redundant)
        for loc_tuple in self.graph:
            self.graph[loc_tuple] = list(set(self.graph[loc_tuple]))

        # Store goal locations for each box
        # Assuming goals are only (at box location) facts
        self.goal_locations = {} # {'box1': 'loc_2_4', ...}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "at" and len(parts) == 3:
                obj, location = parts[1:]
                # Assuming objects in goals are boxes
                self.goal_locations[obj] = location
            # Ignore other potential goal types if any exist (though none in example)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state (frozenset of facts)

        # Check if the current state is the goal state
        if self.goals.issubset(state):
             return 0 # Heuristic is 0 only for goal states

        # Find current location of each box and the robot
        current_box_locations = {} # {'box1': 'loc_4_4', ...}
        robot_location_str = None

        # First pass to find robot location
        for fact in state:
             parts = get_parts(fact)
             if parts[0] == "at-robot" and len(parts) == 2:
                  robot_location_str = parts[1]
                  break # Found robot, no need to search further

        # Second pass to find box locations, excluding the robot's location
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at" and len(parts) == 3:
                 obj, location = parts[1:]
                 # If the object is not the robot (by checking its location), it's a box
                 # This relies on the assumption that only boxes and the robot are 'at' locations
                 if location != robot_location_str:
                     current_box_locations[obj] = location

        total_distance = 0

        # Calculate sum of distances for misplaced boxes
        for box, goal_l_str in self.goal_locations.items():
            current_l_str = current_box_locations.get(box)

            # If box is not found in the current state or is already at its goal, skip it.
            # Note: If a box is not found in the state but is in the goals, it's an invalid state representation.
            if current_l_str is None or current_l_str == goal_l_str:
                continue

            # Convert location strings to tuples
            current_loc_tuple = self.loc_str_to_tuple.get(current_l_str)
            goal_loc_tuple = self.loc_str_to_tuple.get(goal_l_str)

            # If either location is not in our graph (derived from adjacent facts),
            # it means the box or goal is in an isolated part of the grid, likely unreachable.
            if current_loc_tuple is None or goal_loc_tuple is None:
                 return float('inf') # Indicate unsolvable or highly problematic state

            # Calculate shortest path distance using BFS
            dist = bfs_distance(current_loc_tuple, goal_loc_tuple, self.graph)

            # If BFS returns infinity, the goal is unreachable from the current location on the graph
            if dist == float('inf'):
                return float('inf') # Indicate unsolvable state

            total_distance += dist

        # The total distance is the sum of shortest path distances for all misplaced boxes.
        # This is the heuristic value.
        return total_distance

