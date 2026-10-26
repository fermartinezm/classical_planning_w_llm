# Assuming heuristics.heuristic_base provides the base class
from heuristics.heuristic_base import Heuristic

from collections import deque
from fnmatch import fnmatch
import math # For math.inf

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
    # Ensure we don't go out of bounds if pattern is longer than fact parts
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def parse_location(loc_str):
    """Parses 'loc_R_C' string into (R, C) tuple."""
    try:
        parts = loc_str.split('_')
        # PDDL locations are typically 1-indexed.
        return (int(parts[1]), int(parts[2]))
    except (ValueError, IndexError):
        # Handle cases where the string is not in expected format
        return None, None # Indicate failure

def location_to_str(row, col):
    """Converts (R, C) tuple back to 'loc_R_C' string."""
    return f"loc_{row}_{col}"


class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the number of actions required to solve the Sokoban puzzle.
    It combines the estimated effort to move each box to its goal location with the
    estimated effort for the robot to reach a position to push a box. It also includes
    a penalty for potential deadlocks (boxes in corners or against multiple walls).

    # Assumptions
    - The grid structure is defined by 'adjacent' facts.
    - Location names follow the 'loc_R_C' format, where R and C are integers.
    - Each box has a unique goal location specified in the task goals.
    - The graph derived from 'adjacent' facts represents the walkable areas for the robot and pushable paths for boxes (ignoring dynamic obstacles).
    - The goal state is defined solely by the locations of the boxes.

    # Heuristic Initialization
    - Parses goal facts to map each box to its target location.
    - Builds an undirected graph representing the grid connectivity from 'adjacent' facts. This graph is used for shortest path calculations.
    - Infers grid dimensions and creates mappings between location strings and (row, col) coordinates. This is used for detecting wall neighbors for deadlock checks. It collects all location names mentioned in the initial state and static facts to determine the grid boundaries.

    # Step-By-Step Thinking for Computing Heuristic
    1. Check if the current state is a goal state. A state is considered a goal state by this heuristic if all boxes are currently located at their respective goal locations specified in the task goals. If it is a goal state, the heuristic value is 0.
    2. Identify all boxes that are not currently at their goal locations. These are the boxes that need to be moved.
    3. Calculate the total estimated cost for moving the boxes:
       - For each box that needs to be moved, calculate the shortest path distance on the grid graph (derived from 'adjacent' facts) from its current location to its goal location using Breadth-First Search (BFS). This distance represents a lower bound on the number of push actions required for that box, ignoring dynamic obstacles. Sum these distances for all boxes that need moving. If any box's goal is unreachable on the walkable graph, the state is likely unsolvable, and the heuristic returns infinity.
    4. Calculate the estimated cost for the robot to position itself:
       - Find the shortest path distance on the grid graph from the robot's current location to the location of the *nearest* box that needs to be moved. This distance represents the minimum number of move actions the robot needs to take to reach a box's vicinity. Add this minimum distance to the total heuristic. If the robot cannot reach any box that needs moving on the walkable graph, the state is likely unsolvable, and the heuristic returns infinity.
    5. Add a penalty for potential deadlocks:
       - For each box that is not at its goal location, check if its current location is adjacent to two or more "wall" locations. A "wall" location is defined as a grid coordinate neighbor (based on inferred grid dimensions) whose location name exists in the problem's set of all location names but is not a walkable location (i.e., not a node in the graph built from 'adjacent' facts). Locations adjacent to two or more walls are typically corners or enclosed spaces where a box can easily become permanently stuck if it's not its goal location. Add a significant penalty (e.g., 100) to the total heuristic for each such box.
    6. The final heuristic value is the sum of the box-goal distances, the minimum robot-box distance, and the deadlock penalty. The value is always non-negative.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and static facts.
        Builds the grid graph and location-coordinate mappings.
        """
        # Assume task object has .goals and .static attributes
        self.goals = task.goals
        static_facts = task.static

        # Store goal locations for each box
        self.box_goals = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                box, location = args
                self.box_goals[box] = location

        # Build the adjacency graph from adjacent facts
        self.graph = {}
        all_loc_names = set() # Collect all location names mentioned in static facts

        for fact in static_facts:
            parts = get_parts(fact)
            if match(fact, "adjacent", "*", "*", "*"):
                _, loc1, loc2, _ = parts
                all_loc_names.add(loc1)
                all_loc_names.add(loc2)
                if loc1 not in self.graph:
                    self.graph[loc1] = []
                if loc2 not in self.graph:
                    self.graph[loc2] = []
                # Add bidirectional edges
                self.graph[loc1].append(loc2)
                self.graph[loc2].append(loc1)

        # Remove duplicates from adjacency lists
        for loc in self.graph:
            self.graph[loc] = list(set(self.graph[loc]))

        # Collect location names from initial state as well to ensure we have all possible locations
        for fact in task.initial_state:
             parts = get_parts(fact)
             # Check for locations in predicates like (at-robot loc) or (at box loc)
             if len(parts) > 1 and parts[1].startswith('loc_'):
                  all_loc_names.add(parts[1])
             if len(parts) > 2 and parts[2].startswith('loc_'):
                  all_loc_names.add(parts[2])

        # Build location-coordinate mappings and find grid dimensions
        self.loc_to_coords = {}
        self.coords_to_loc = {}
        max_row = 0
        max_col = 0
        for loc_str in all_loc_names:
            r, c = parse_location(loc_str)
            if r is not None and c is not None:
                self.loc_to_coords[loc_str] = (r, c)
                self.coords_to_loc[(r, c)] = loc_str
                max_row = max(max_row, r)
                max_col = max(max_col, c)
        # Store max row/col index (assuming 1-based indexing from PDDL names)
        self.grid_dims = (max_row, max_col)

        # Define possible coordinate offsets for neighbors (relative to a cell)
        # (dr, dc) -> (0, 1) right, (0, -1) left, (1, 0) down, (-1, 0) up
        self.neighbor_offsets = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        self.all_loc_names = all_loc_names # Store all known location names


    def get_distance(self, start_loc, end_loc):
        """
        Performs BFS on the graph to find the shortest distance between two locations.
        Returns -1 if no path exists.
        """
        if start_loc == end_loc:
            return 0
        # Ensure locations exist in the graph (are walkable)
        if start_loc not in self.graph or end_loc not in self.graph:
             return -1 # Cannot reach if start/end is a wall or non-existent

        queue = deque([(start_loc, 0)])
        visited = {start_loc}

        while queue:
            current_loc, dist = queue.popleft()

            if current_loc == end_loc:
                return dist

            # Check if current_loc is a valid node in the graph before accessing neighbors
            if current_loc in self.graph:
                for neighbor in self.graph[current_loc]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, dist + 1))

        return -1 # No path found


    def __call__(self, node):
        """
        Computes the heuristic value for the given state.
        """
        state = node.state

        # Get current locations of robot and boxes
        robot_location = None
        box_locations = {}
        # clear_locations = set() # Not directly used in this heuristic calculation

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at-robot":
                robot_location = parts[1]
            elif parts[0] == "at" and parts[1] in self.box_goals: # Only track boxes we care about
                 box_locations[parts[1]] = parts[2]
            # elif parts[0] == "clear":
            #      clear_locations.add(parts[1]) # Not used

        # Check if goal is reached (all boxes at their goal locations)
        # We check against the box_goals dictionary derived from task.goals
        all_boxes_at_goal = True
        # Iterate through all boxes that *should* be at a goal location
        for box, goal_loc in self.box_goals.items():
            # Check if the box is currently in the state and at the correct goal location
            # If a box from box_goals is not even in box_locations, it's not at its goal.
            if box not in box_locations or box_locations[box] != goal_loc:
                all_boxes_at_goal = False
                break

        if all_boxes_at_goal:
             return 0 # Goal state reached

        total_heuristic = 0
        boxes_to_move = []

        # Calculate box-goal distances
        for box, goal_loc in self.box_goals.items():
            current_loc = box_locations.get(box)
            # Ensure the box is in the state and not already at its goal
            if current_loc and current_loc != goal_loc:
                boxes_to_move.append(box)
                # Add box-goal distance (shortest path on walkable graph)
                dist = self.get_distance(current_loc, goal_loc)
                if dist == -1:
                    # Box is in a location disconnected from its goal on the walkable graph.
                    # This indicates an unsolvable state portion (deadlock).
                    return math.inf # Return infinity for unsolvable states

                # Each unit of distance represents at least one push action
                total_heuristic += dist

        # If no boxes need moving, but we didn't return 0, something is wrong.
        # This case should be covered by the initial all_boxes_at_goal check.
        # Adding a defensive check just in case.
        if not boxes_to_move:
             # This implies all boxes are at their goals, but the initial check failed?
             # Or maybe some boxes from the task are not in the current state?
             # Assuming valid states always contain all task objects.
             # If we reach here, it must be a goal state based on box positions.
             # The initial check should have caught this.
             # Let's return 0, although this path should ideally not be taken
             # if the initial all_boxes_at_goal check is correct.
             return 0


        # Calculate robot-to-nearest-box distance
        min_robot_box_dist = math.inf
        robot_reachable_to_any_box = False

        # The robot needs to reach a location adjacent to the box to push it.
        # For simplicity, we calculate the distance from the robot's current location
        # to the location of each box that needs moving. The minimum of these distances
        # is an estimate of the robot's approach cost.
        # This ignores obstacles (other boxes, walls) and the requirement to reach
        # a *clear* adjacent cell. It's a relaxation.

        # Ensure robot_location is not None and is a valid node in the graph
        if robot_location and robot_location in self.graph:
            for box in boxes_to_move:
                box_loc = box_locations[box]
                # Ensure box_loc is a valid node in the graph
                if box_loc in self.graph:
                    dist = self.get_distance(robot_location, box_loc)
                    if dist != -1:
                         min_robot_box_dist = min(min_robot_box_dist, dist)
                         robot_reachable_to_any_box = True # At least one box is reachable by robot

        # If robot cannot reach any box that needs moving (on the walkable graph),
        # it's likely unsolvable.
        if not robot_reachable_to_any_box:
             return math.inf # Return infinity if robot is trapped or isolated

        total_heuristic += min_robot_box_dist

        # Add deadlock penalty
        deadlock_penalty = 0
        for box in boxes_to_move:
            box_loc_str = box_locations[box]
            goal_loc_str = self.box_goals[box]

            # Penalty applies only if the box is not at its goal
            if box_loc_str != goal_loc_str:
                box_r, box_c = self.loc_to_coords.get(box_loc_str, (None, None))

                # Ensure location coordinates were parsed correctly
                if box_r is not None and box_c is not None:
                    wall_neighbors = 0
                    # Check potential neighbors based on grid structure
                    for dr, dc in self.neighbor_offsets:
                        neighbor_r, neighbor_c = box_r + dr, box_c + dc
                        neighbor_loc_str = location_to_str(neighbor_r, neighbor_c)

                        # A potential neighbor location is a "wall" if:
                        # 1. The location name exists in the problem (i.e., it's a real grid cell).
                        # 2. It is not a walkable location (not a node in the graph).
                        # We assume all_loc_names contains all grid locations mentioned in the problem.
                        if neighbor_loc_str in self.all_loc_names and neighbor_loc_str not in self.graph:
                             wall_neighbors += 1

                    # Penalize boxes trapped in corners (>= 2 wall neighbors)
                    # This is a strong indicator of a deadlock if not at the goal.
                    # The penalty value is arbitrary but should be large enough
                    # to make these states less preferable.
                    if wall_neighbors >= 2:
                         deadlock_penalty += 100 # Arbitrary large penalty

        total_heuristic += deadlock_penalty

        # The heuristic should be non-negative.
        # Given the components (distances >= 0, penalty >= 0), the sum is >= 0.
        # math.inf is handled correctly.
        return total_heuristic
