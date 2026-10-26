from collections import deque
from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact."""
    # Assumes fact is a string like '(predicate arg1 arg2)'
    # Returns ['predicate', 'arg1', 'arg2']
    return fact[1:-1].split()

# Helper functions for location names and coordinates
def parse_location(loc_name):
    """Parses 'loc_row_col' into (row, col)."""
    # Assumes loc_name is a string like 'loc_1_1' or 'loc_10_12'
    parts = loc_name.split('_')
    # Expects at least 3 parts: 'loc', row, col
    if len(parts) != 3 or parts[0] != 'loc':
        raise ValueError(f"Unexpected location format: {loc_name}")
    try:
        row = int(parts[1])
        col = int(parts[2])
        return (row, col)
    except ValueError:
        raise ValueError(f"Could not parse row/col from location name: {loc_name}")


def format_location(row, col):
    """Formats (row, col) into 'loc_row_col'."""
    return f"loc_{row}_{col}"

class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing the grid
    distances for each box to its goal location and adding the minimum grid
    distance from the robot's current location to a valid push position
    for any box that needs to be moved.

    # Assumptions
    - Locations are named in the format 'loc_row_col'.
    - The grid structure is defined by 'adjacent' facts, forming a connected graph.
    - The heuristic calculates distances on the grid graph ignoring dynamic
      obstacles (other boxes).
    - The heuristic assumes solvable instances; it may return infinity for
      unsolvable states (e.g., disconnected grid components).

    # Heuristic Initialization
    - Parses 'adjacent' facts to build a graph representation of the grid.
    - Maps location names to (row, col) coordinates based on the 'loc_row_col' format.
    - Computes all-pairs shortest paths (grid distance) on the grid graph
      using BFS.
    - Extracts goal locations for each box from the task goals.

    # Step-By-Step Thinking for Computing Heuristic
    1. Check if the current state is a goal state by verifying if all goal facts are present. If yes, return 0.
    2. Identify the robot's current location from the state facts.
    3. Identify the current location for each box from the state facts.
    4. Initialize `box_distance_sum` to 0 and `min_robot_distance_to_push_pos` to infinity.
    5. Iterate through each box and its goal location defined in the task goals:
       a. Get the box's current location from the state.
       b. If the box is not at its goal location:
          i. Add the precomputed grid distance from the box's current location to its goal location to `box_distance_sum`.
          ii. Determine the set of potential robot push locations for this box. These are locations adjacent to the box's current location, on the side opposite to a push direction that moves the box closer to its goal (based on coordinate comparison). Only include locations that exist in the grid.
          iii. For each potential push location, find the precomputed grid distance from the robot's current location to this push location.
          iv. Update `min_robot_distance_to_push_pos` with the minimum distance found in the previous step (across all boxes not at their goals).
    6. If no boxes needed moving (which should be caught by step 1), the heuristic is 0.
    7. If boxes need moving but the robot cannot reach any potential push position (min_robot_distance_to_push_pos remains infinity), the state is likely a dead end, return infinity.
    8. Otherwise, the heuristic value is `box_distance_sum` + `min_robot_distance_to_push_pos`.
    """

    def __init__(self, task):
        """Initialize the heuristic."""
        self.goals = task.goals
        self.static_facts = task.static

        # 1. Collect all locations from static facts (assuming adjacent facts cover all relevant locations)
        self.locations = set()
        for fact in self.static_facts:
            parts = get_parts(fact)
            if parts[0] == 'adjacent':
                _, l1, l2, _ = parts
                self.locations.add(l1)
                self.locations.add(l2)

        # 2. Map location names to coordinates and vice versa
        self.loc_to_coords = {}
        self.coords_to_loc = {}
        for loc in self.locations:
            try:
                coords = parse_location(loc)
                self.loc_to_coords[loc] = coords
                self.coords_to_loc[coords] = loc
            except ValueError as e:
                # Log or handle error if location format is unexpected
                # print(f"Warning: Could not parse location '{loc}'. {e}")
                # This location might be excluded from graph/distances
                pass # Skip locations that don't match expected format

        # 3. Build grid graph from adjacent facts
        self.grid_graph = {loc: [] for loc in self.locations}
        for fact in self.static_facts:
            parts = get_parts(fact)
            if parts[0] == 'adjacent':
                _, l1, l2, _ = parts
                # Only add edge if both locations were successfully parsed and added to self.locations
                if l1 in self.grid_graph and l2 in self.grid_graph:
                    self.grid_graph[l1].append(l2)
                    # PDDL adjacent facts are typically symmetric, but adding both directions
                    # explicitly ensures graph is undirected for BFS distance calculation.
                    # self.grid_graph[l2].append(l1) # Redundant if PDDL is symmetric

        # 4. Compute all-pairs shortest paths (grid distance) using BFS
        self.grid_distances = {}
        for start_node in self.locations:
            self.grid_distances[start_node] = self._bfs(start_node)

        # 5. Extract goal locations for each box
        self.goal_locations = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "at":
                # Goal is (at box location)
                box, location = parts[1:]
                self.goal_locations[box] = location

    def _bfs(self, start_node):
        """Performs BFS from start_node to find distances to all reachable nodes."""
        distances = {node: float('inf') for node in self.locations}
        if start_node not in self.locations:
             # Start node is not in the grid graph (e.g., parsing error)
             return distances # All distances remain inf

        distances[start_node] = 0
        queue = deque([start_node])

        while queue:
            current_node = queue.popleft()

            # Check if current_node exists in the graph (handles potential parsing errors)
            if current_node not in self.grid_graph:
                 continue

            for neighbor in self.grid_graph.get(current_node, []):
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)
        return distances

    def _get_potential_push_locations(self, box_loc, goal_loc):
        """
        Identifies potential robot locations adjacent to box_loc
        that allow pushing the box towards goal_loc based on coordinates.
        Returns a set of location names.
        """
        potential_push_locs = set()

        # Ensure both locations were successfully parsed into coordinates
        if box_loc not in self.loc_to_coords or goal_loc not in self.loc_to_coords:
             return potential_push_locs # Cannot determine push direction without coords

        r_b, c_b = self.loc_to_coords[box_loc]
        r_g, c_g = self.loc_to_coords[goal_loc]

        # Determine required push directions based on Manhattan distance reduction
        # If r_b > r_g, need to push up (towards smaller row index). Robot needs to be below (larger row index).
        if r_b > r_g:
            push_loc_coords = (r_b + 1, c_b)
            push_loc_name = format_location(*push_loc_coords)
            # Check if the calculated location name exists in our known locations
            if push_loc_name in self.locations:
                 potential_push_locs.add(push_loc_name)

        # If r_b < r_g, need to push down (towards larger row index). Robot needs to be above (smaller row index).
        if r_b < r_g:
            push_loc_coords = (r_b - 1, c_b)
            push_loc_name = format_location(*push_loc_coords)
            if push_loc_name in self.locations:
                 potential_push_locs.add(push_loc_name)

        # If c_b > c_g, need to push left (towards smaller col index). Robot needs to be right (larger col index).
        if c_b > c_g:
            push_loc_coords = (r_b, c_b + 1)
            push_loc_name = format_location(*push_loc_coords)
            if push_loc_name in self.locations:
                 potential_push_locs.add(push_loc_name)

        # If c_b < c_g, need to push right (towards larger col index). Robot needs to be left (smaller col index).
        if c_b < c_g:
            push_loc_coords = (r_b, c_b - 1)
            push_loc_name = format_location(*push_loc_coords)
            if push_loc_name in self.locations:
                 potential_push_locs.add(push_loc_name)

        return potential_push_locs


    def __call__(self, node):
        """Compute the heuristic value for the given state."""
        state = node.state

        # 1. Check for goal state
        if self.goals <= state:
            return 0

        # 2. Find robot location
        robot_loc = None
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at-robot':
                robot_loc = parts[1]
                break

        # If robot location isn't found or isn't in our known locations, state is likely invalid
        if robot_loc is None or robot_loc not in self.locations:
             return float('inf')

        # 3. Find box locations
        box_locations = {}
        # Only consider boxes that are part of the goal
        goal_boxes = set(self.goal_locations.keys())
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at' and parts[1] in goal_boxes:
                 box, loc = parts[1:]
                 box_locations[box] = loc

        box_distance_sum = 0
        min_robot_distance_to_push_pos = float('inf')
        found_box_to_move = False # Flag to check if any box needs moving

        # 5. Iterate through each box and its goal location
        for box, goal_loc in self.goal_locations.items():
            current_loc = box_locations.get(box)

            # If a goal box is not found in the state, it's an invalid state
            if current_loc is None or current_loc not in self.locations:
                 return float('inf') # Should not happen in valid states

            # a. If the box is not at its goal location
            if current_loc != goal_loc:
                found_box_to_move = True

                # i. Add box distance to sum
                # Ensure locations are in our precomputed distances table
                if current_loc in self.grid_distances and goal_loc in self.grid_distances[current_loc]:
                    box_distance_sum += self.grid_distances[current_loc][goal_loc]
                else:
                    # Locations not connected or not found during init parsing
                    return float('inf') # State is likely unsolvable from here

                # ii. Find potential push locations for this box
                potential_push_locs = self._get_potential_push_locations(current_loc, goal_loc)

                # iii. Find minimum robot distance to a push location for this box
                min_dist_for_this_box = float('inf')
                for push_loc in potential_push_locs:
                    # Ensure robot_loc and push_loc are in our precomputed distances table
                    if robot_loc in self.grid_distances and push_loc in self.grid_distances[robot_loc]:
                         dist = self.grid_distances[robot_loc][push_loc]
                         min_dist_for_this_box = min(min_dist_for_this_box, dist)

                # iv. Update overall minimum robot distance
                # Only update if we found at least one reachable push location for this box
                if min_dist_for_this_box != float('inf'):
                    min_robot_distance_to_push_pos = min(min_robot_distance_to_push_pos, min_dist_for_this_box)
                # Note: If a box needs moving but robot cannot reach *any* push pos for *that specific box*,
                # it doesn't make the whole state infinite *yet*. It only makes the state infinite
                # if the robot cannot reach *any* push position for *any* box that needs moving.

        # 6. If no boxes needed moving, return 0 (already handled by initial goal check)
        # This check is redundant if goal check is correct, but harmless.
        if not found_box_to_move:
             return 0

        # 7. If boxes need moving but robot cannot reach any push position for any of them
        if min_robot_distance_to_push_pos == float('inf'):
             return float('inf') # State is likely unsolvable

        # 8. Otherwise, return the sum
        return box_distance_sum + min_robot_distance_to_push_pos
