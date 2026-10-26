from collections import deque
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper functions for PDDL parsing
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at obj loc)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def parse_location(loc_str):
    """Parses a location string like 'loc_X_Y' into a tuple (X, Y)."""
    # Assuming location format is always 'loc_row_col'
    parts = loc_str.split('_')
    if len(parts) == 3 and parts[0] == 'loc':
        try:
            return (int(parts[1]), int(parts[2]))
        except ValueError:
            return None # Indicate parsing failure
    return None # Indicate parsing failure


class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing the shortest
    path distances for each box to its goal location and adding the shortest
    path distance for the robot to the nearest box that still needs to be moved.

    # Assumptions
    - The locations form a grid-like structure defined by adjacent facts, parsable as 'loc_row_col'.
    - Adjacency is symmetric.
    - The location graph is connected for all relevant locations (robot, boxes, goals) in a solvable problem.
    - The heuristic is non-admissible and designed for greedy best-first search.

    # Heuristic Initialization
    The constructor performs the following steps:
    1. Parses static facts to build a graph representing the locations and their adjacencies.
    2. Creates mappings between location strings (e.g., 'loc_1_1') and coordinate tuples (e.g., (1, 1)).
    3. Calculates all-pairs shortest paths between all locations using BFS, storing the results in a dictionary.
    4. Extracts the goal location for each box from the task's goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current location of the robot and all boxes.
    2. Determine which boxes are not yet at their goal locations.
    3. If all boxes are at their goals, the heuristic is 0.
    4. If there are boxes not at their goals:
       a. Calculate the sum of the shortest path distances for each such box from its current location to its goal location (using pre-calculated distances). This estimates the minimum number of pushes required in an ideal scenario, ignoring obstacles and robot positioning.
       b. Calculate the shortest path distance from the robot's current location to the nearest box that still needs to be moved (using pre-calculated distances). This estimates the robot's cost to reach a relevant box.
       c. The total heuristic value is the sum of the total box distance and the minimum robot-to-box distance.
    """

    def __init__(self, task):
        """Initialize the heuristic by building the location graph and pre-calculating distances."""
        super().__init__(task)
        self.goals = task.goals
        static_facts = task.static

        # 1. Build location graph and coordinate mappings
        self.graph = {}
        self.locations = set()
        self.loc_to_coords = {}
        self.coords_to_loc = {}

        all_loc_strs = set()
        for fact in static_facts:
            if match(fact, "adjacent", "*", "*", "*"):
                _, loc1_str, loc2_str, _ = get_parts(fact)
                all_loc_strs.add(loc1_str)
                all_loc_strs.add(loc2_str)
                self.graph.setdefault(loc1_str, []).append(loc2_str)
                self.graph.setdefault(loc2_str, []).append(loc1_str) # Assuming symmetric adjacency

        # Also collect locations from initial state and goals to ensure all relevant locations are included
        for fact in task.initial_state:
             if match(fact, "at-robot", "*"):
                 all_loc_strs.add(get_parts(fact)[1])
             elif match(fact, "at", "*", "*"):
                 _, _, loc_str = get_parts(fact)
                 all_loc_strs.add(loc_str)

        for goal in self.goals:
             if match(goal, "at", "*", "*"):
                 _, _, loc_str = get_parts(goal)
                 all_loc_strs.add(loc_str)


        for loc_str in all_loc_strs:
            coords = parse_location(loc_str)
            if coords is not None:
                self.locations.add(loc_str)
                self.loc_to_coords[loc_str] = coords
                self.coords_to_loc[coords] = loc_str
                # Ensure all locations added to self.locations are keys in the graph, even if isolated
                self.graph.setdefault(loc_str, [])


        # 2. Calculate all-pairs shortest paths
        self.all_dists = {}
        for start_loc in self.locations:
            self.all_dists[start_loc] = self._bfs(start_loc)

        # 3. Extract goal locations for boxes
        self.goal_locations = {}
        for goal in self.goals:
            if match(goal, "at", "*", "*"):
                _, box, loc = get_parts(goal)
                self.goal_locations[box] = loc

    def _bfs(self, start_loc):
        """Performs BFS from a start location to find distances to all other locations."""
        dists = {loc: float('inf') for loc in self.locations}
        if start_loc not in self.locations:
             # Start location is not in the known graph/locations set
             return dists # Return distances indicating unreachable

        dists[start_loc] = 0
        q = deque([start_loc])

        while q:
            curr_loc = q.popleft()
            curr_d = dists[curr_loc]

            # Get neighbors from the graph, handle locations with no adjacencies
            for neighbor in self.graph.get(curr_loc, []):
                if neighbor in self.locations and dists[neighbor] == float('inf'):
                    dists[neighbor] = curr_d + 1
                    q.append(neighbor)
        return dists

    def __call__(self, node):
        """Compute the domain-dependent heuristic value for the given state."""
        state = node.state

        # 1. Identify robot and box locations
        robot_loc = None
        box_locations = {}
        for fact in state:
            if match(fact, "at-robot", "*"):
                robot_loc = get_parts(fact)[1]
            elif match(fact, "at", "*", "*"):
                _, box, loc = get_parts(fact)
                box_locations[box] = loc

        # Ensure robot location is found and is a known location
        if robot_loc is None or robot_loc not in self.locations:
             return float('inf')

        # 2. Determine boxes not at goal
        boxes_to_move = []
        for box, goal_loc in self.goal_locations.items():
            current_loc = box_locations.get(box)
            # Check if box exists, its location is known, and is not at its goal
            if current_loc is not None and current_loc in self.locations and current_loc != goal_loc:
                boxes_to_move.append(box)
            elif current_loc is None or current_loc not in self.locations:
                 # Box is missing or in an unknown location
                 return float('inf') # Problem likely unsolvable

        # 3. If all boxes are at goal, heuristic is 0
        if not boxes_to_move:
            return 0

        # 4a. Calculate sum of box-goal distances
        sum_box_dist = 0
        for box in boxes_to_move:
            current_loc = box_locations[box]
            goal_loc = self.goal_locations[box]
            # Use pre-calculated distance, handle potential missing locations gracefully
            dist = self.all_dists.get(current_loc, {}).get(goal_loc, float('inf'))
            if dist == float('inf'):
                 # Indicates a box or goal is in an unreachable part of the map
                 return float('inf') # Problem likely unsolvable from here
            sum_box_dist += dist

        # 4b. Calculate robot distance to nearest box needing move
        min_robot_dist_to_box = float('inf')
        for box in boxes_to_move:
            current_loc = box_locations[box]
            # Use pre-calculated distance, handle potential missing locations gracefully
            dist = self.all_dists.get(robot_loc, {}).get(current_loc, float('inf'))
            if dist == float('inf'):
                 # Robot cannot reach this box
                 return float('inf') # Problem likely unsolvable from here
            min_robot_dist_to_box = min(min_robot_dist_to_box, dist)

        # 4c. Total heuristic
        h = sum_box_dist + min_robot_dist_to_box

        return h
