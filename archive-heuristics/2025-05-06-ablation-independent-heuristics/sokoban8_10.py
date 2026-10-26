# Assume Heuristic and Task classes are imported from the planner environment
# from heuristics.heuristic_base import Heuristic
# from task import Task, Operator, Node # Node might be needed for type hinting, but we only use node.state

import re
import math
from collections import deque # More efficient queue for BFS

# Helper function to parse location string 'loc_R_C' into (R, C) tuple
def parse_location_string(loc_str):
    """Parses a location string like 'loc_R_C' into an (R, C) tuple."""
    match = re.match(r'loc_(\d+)_(\d+)', loc_str)
    if match:
        return (int(match.group(1)), int(match.group(2)))
    return None # Should not happen in valid PDDL

# Helper function to format (R, C) tuple back to location string
def format_location_string(r, c):
    """Formats an (R, C) tuple back into a location string like 'loc_R_C'."""
    return f'loc_{r}_{c}'

# Helper function to parse a PDDL fact string
def parse_fact(fact_str):
    """Parses a PDDL fact string into its predicate and arguments."""
    # Removes leading/trailing parentheses and splits by space
    # Handles cases like '(at box1 loc_4_4)' -> ['at', 'box1', 'loc_4_4']
    # Handles cases like '(at-robot loc_6_4)' -> ['at-robot', 'loc_6_4']
    # Handles cases like '(clear loc_2_4)' -> ['clear', 'loc_2_4']
    # Handles cases like '(adjacent loc_4_2 loc_4_3 right)' -> ['adjacent', 'loc_4_2', 'loc_4_3', 'right']
    parts = fact_str[1:-1].split()
    if not parts:
        return None, [] # Handle empty fact string case, though unlikely
    predicate = parts[0]
    args = parts[1:]
    return predicate, args

# Helper function to get opposite direction
def get_opposite_direction(direction):
    """Returns the opposite direction."""
    if direction == 'up': return 'down'
    if direction == 'down': return 'up'
    if direction == 'left': return 'right'
    if direction == 'right': return 'left'
    return None # Should not happen

# Helper function to get direction from loc1 to loc2 if adjacent
def get_direction(loc1_str, loc2_str, graph):
    """Returns the direction from loc1 to loc2 if they are adjacent in the graph."""
    for direction, adj_loc_str in graph.get(loc1_str, {}).items():
        if adj_loc_str == loc2_str:
            return direction
    return None # Not adjacent

# Helper function to get adjacent location in a specific direction
def get_adjacent_in_direction(loc_str, direction, graph):
    """Returns the location adjacent to loc_str in the given direction."""
    return graph.get(loc_str, {}).get(direction)


class sokobanHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Sokoban domain.

    Summary:
    Estimates the cost to reach the goal state by summing two components:
    1.  The minimum total distance required to move all boxes to their
        assigned goal locations. This is calculated using a greedy matching
        between current box locations and goal locations based on shortest
        path distances on the grid.
    2.  The minimum distance required for the robot to reach a position
        from which it can push any box that is not yet at its goal, towards
        its assigned goal location, provided such a push reduces the box's
        distance to its goal.
    The heuristic returns infinity if any box is in a precomputed dead-end
    location (a location from which no goal is reachable). It returns 0
    if the state is a goal state.

    Assumptions:
    -   The PDDL domain uses 'loc_R_C' format for locations, where R and C are integers.
    -   The static facts include all 'adjacent' predicates defining the grid connectivity.
    -   The goal state consists of '(at box loc)' facts for all boxes.
    -   The number of boxes equals the number of goal locations. (The greedy matching handles potential mismatches by returning infinity if not all boxes/goals can be matched or if the best match is infinite).
    -   The grid is connected (or at least, all relevant locations and goals are in the same connected component).

    Heuristic Initialization:
    1.  Parses all location strings from task facts and static facts to map them
        to (row, column) tuples and build a graph representation (adjacency list)
        based on 'adjacent' predicates.
    2.  Precomputes all-pairs shortest path distances between all locations
        using BFS, storing them in a dictionary `self.distances[loc1][loc2]`.
        This is a relaxation ignoring dynamic obstacles.
    3.  Identifies goal locations from `task.goals`.
    4.  Identifies simple dead-end locations: locations from which no goal
        location is reachable via the precomputed shortest paths.

    Step-By-Step Thinking for Computing Heuristic:
    1.  Check if the current state is the goal state using `self.task.goal_reached(state)`. If yes, return 0.
    2.  Parse the current state to find the robot's location (`robot_loc_str`) and the location
        of each box (`box_loc_map`).
    3.  For each box, check if its current location is a precomputed dead-end
        location (`self.dead_end_locations`). If any box is in a dead-end, return `float('inf')`.
    4.  Calculate the box-to-goal cost (`h_boxes`):
        a.  Get sets of current box locations and goal locations.
        b.  Create a list of (distance, box_loc, goal_loc) tuples for every
            possible pairing of a box location and a goal location, using the
            precomputed shortest path distances (`self.distances`).
        c.  Sort this list by distance in ascending order.
        d.  Iterate through the sorted list. Maintain sets of matched box
            locations and matched goal locations, and a dictionary for box assignments (`box_assignments`).
            For each tuple (dist, b_loc, g_loc):
            If `b_loc` has not been matched and `g_loc` has not been matched:
                If `dist` is `float('inf')`, it means remaining boxes cannot reach remaining goals, so return `float('inf')`.
                Add `dist` to `h_boxes`.
                Mark `b_loc` and `g_loc` as matched.
                Find the box name at `b_loc` and store the assignment `box_assignments[box_name] = g_loc`.
        e.  If, after iterating through all pairs, the number of assignments
            (`len(box_assignments)`) does not equal the number of boxes
            (`len(box_loc_map)`), it indicates a matching failure (e.g., unequal
            counts or unmatchable pairs), return `float('inf')`.
    5.  Calculate the robot-to-box cost (`h_robot`):
        a.  Initialize `h_robot` to `float('inf')`.
        b.  Identify boxes that are not yet at their assigned goal locations
            (`boxes_to_move`). If this list is empty, set `h_robot` to 0.
        c.  For each box (`box_name`) needing movement, get its current location
            (`b_loc_str`) and assigned goal location (`g_loc_str`) from `box_loc_map`
            and `box_assignments`.
        d.  Calculate the current shortest distance from `b_loc_str` to `g_loc_str`.
        e.  Iterate through all locations (`p_loc_str`) adjacent to `b_loc_str`.
            `p_loc_str` is a potential position for the robot to push from.
        f.  Determine the direction of the potential push (`b_loc_str` to `next_loc_str`)
            which is the opposite of the direction from `p_loc_str` to `b_loc_str`.
        g.  Find the location `next_loc_str` adjacent to `b_loc_str` in the push direction.
        h.  If `next_loc_str` exists and the shortest distance from `next_loc_str`
            to `g_loc_str` is strictly less than the distance from `b_loc_str` to `g_loc_str`:
            i.  `p_loc_str` is a valid pushing position for a distance-reducing push.
            ii. Calculate the shortest distance from the robot's current location
                (`robot_loc_str`) to `p_loc_str` using `self.distances`.
            iii. Update `h_robot` with the minimum distance found so far across
                 all valid pushing positions for all boxes needing movement.
        i.  If, after checking all boxes needing movement and all potential
            pushing positions, `h_robot` is still `float('inf')` (and there were
            boxes needing movement), it means the robot cannot reach a position
            for a distance-reducing push for any box. Return `float('inf')`.
    6.  The total heuristic value is `h_boxes + h_robot`.
    """

    def __init__(self, task):
        super().__init__(task)
        self.task = task # Store task for goal_reached check

        # 1. Build graph and location mapping
        self.location_coords = {} # loc_str -> (r, c)
        self.graph = {} # loc_str -> {direction: adj_loc_str}
        all_locations = set()

        # Collect all locations from facts and static facts
        # Locations appear in (at-robot loc), (at box loc), (clear loc), (adjacent loc1 loc2 dir)
        for fact_str in task.facts | task.static:
             predicate, args = parse_fact(fact_str)
             if predicate in ['at-robot', 'clear']:
                 if args: all_locations.add(args[0])
             elif predicate == 'at':
                 if len(args) > 1: all_locations.add(args[1]) # (at box loc)
             elif predicate == 'adjacent':
                 if len(args) > 1: all_locations.add(args[0])
                 if len(args) > 1: all_locations.add(args[1])


        for loc_str in all_locations:
            coords = parse_location_string(loc_str)
            if coords:
                self.location_coords[loc_str] = coords
                self.graph[loc_str] = {} # Initialize adjacency list entry

        # Populate graph with adjacent facts
        for fact_str in task.static:
            predicate, args = parse_fact(fact_str)
            if predicate == 'adjacent' and len(args) == 3:
                loc1_str, loc2_str, direction = args
                if loc1_str in self.graph and loc2_str in self.graph:
                     self.graph[loc1_str][direction] = loc2_str

        self.locations = list(self.graph.keys()) # Ordered list of locations

        # 2. Precompute All-Pairs Shortest Paths (APSP)
        self.distances = {}
        for start_loc in self.locations:
            self.distances[start_loc] = self._bfs(start_loc)

        # 3. Identify goal locations
        self.goal_locations = set()
        # Assuming goals are always (at box loc)
        for goal_fact_str in task.goals:
            predicate, args = parse_fact(goal_fact_str)
            if predicate == 'at' and len(args) == 2:
                self.goal_locations.add(args[1])

        # 4. Identify simple dead-end locations
        # A location is a dead-end if no goal is reachable from it
        self.dead_end_locations = set()
        for loc_str in self.locations:
            # A goal location is never a dead-end in this definition
            if loc_str not in self.goal_locations:
                is_dead_end = True
                if not self.goal_locations: # Handle case with no goals (shouldn't happen in Sokoban)
                    is_dead_end = False # Or maybe True? Depends on domain semantics. Assume solvable means goals exist.
                else:
                    for goal_loc_str in self.goal_locations:
                        # Check if distance is finite
                        if self.distances.get(loc_str, {}).get(goal_loc_str, math.inf) != math.inf:
                            is_dead_end = False
                            break
                if is_dead_end:
                    self.dead_end_locations.add(loc_str)


    def _bfs(self, start_loc_str):
        """Performs BFS from a start location to find distances to all reachable locations."""
        distances = {loc: math.inf for loc in self.locations}
        if start_loc_str not in self.locations:
             return distances # Start location might not be in graph if parsing failed or task is weird

        distances[start_loc_str] = 0
        queue = deque([start_loc_str]) # Use deque for efficient popping
        visited = {start_loc_str}

        while queue:
            current_loc_str = queue.popleft()

            # BFS on the graph ignores dynamic obstacles (clear facts).
            # It finds shortest path assuming all locations in the graph are traversable.
            for direction, neighbor_loc_str in self.graph.get(current_loc_str, {}).items():
                 if neighbor_loc_str not in visited:
                    visited.add(neighbor_loc_str)
                    distances[neighbor_loc_str] = distances[current_loc_str] + 1
                    queue.append(neighbor_loc_str)

        return distances

    def __call__(self, node):
        state = node.state

        # 1. Check if goal is reached
        if self.task.goal_reached(state):
            return 0

        # Parse current state
        robot_loc_str = None
        box_loc_map = {} # box_name -> loc_str
        # current_clear_locations = set() # Not strictly needed for this heuristic's calculation logic

        for fact_str in state:
            predicate, args = parse_fact(fact_str)
            if predicate == 'at-robot' and args:
                robot_loc_str = args[0]
            elif predicate == 'at' and len(args) == 2:
                box_name, loc_str = args
                box_loc_map[box_name] = loc_str
            # elif predicate == 'clear' and args:
            #      current_clear_locations.add(args[0])

        # Ensure we found robot and all boxes mentioned in goals
        goal_boxes = {parse_fact(g)[1][0] for g in self.task.goals if parse_fact(g)[0] == 'at'}
        if robot_loc_str is None or len(box_loc_map) != len(goal_boxes):
             # This state is likely invalid or represents a failure state not covered
             # by simple dead-ends. Return infinity.
             return math.inf

        # 2. Check for simple dead-ends for boxes
        for box_name, box_loc_str in box_loc_map.items():
            if box_loc_str in self.dead_end_locations:
                return math.inf # Box is in a location from which no goal is reachable

        # Get lists/sets for matching
        current_box_locations_list = list(box_loc_map.values())
        goal_locations_list = list(self.goal_locations)

        # Handle case where number of boxes != number of goals (shouldn't happen in standard Sokoban)
        if len(current_box_locations_list) != len(goal_locations_list):
             # This heuristic assumes a 1:1 matching. If counts differ, it's likely unsolvable
             # or requires a different heuristic approach. Return infinity.
             return math.inf

        # 4. Calculate box-to-goal cost (h_boxes) using greedy matching
        box_goal_pairs = []
        for b_loc in current_box_locations_list:
            for g_loc in goal_locations_list:
                dist = self.distances.get(b_loc, {}).get(g_loc, math.inf)
                box_goal_pairs.append((dist, b_loc, g_loc))

        box_goal_pairs.sort() # Sort by distance

        h_boxes = 0
        matched_b_locs = set()
        matched_g_locs = set()
        box_assignments = {} # box_name -> goal_loc_str

        # Perform greedy matching
        for dist, b_loc, g_loc in box_goal_pairs:
            if b_loc not in matched_b_locs and g_loc not in matched_g_locs:
                if dist == math.inf:
                     # If the shortest remaining distance is inf, remaining boxes cannot reach remaining goals
                     return math.inf
                h_boxes += dist
                matched_b_locs.add(b_loc)
                matched_g_locs.add(g_loc)

                # Find which box is at b_loc to store assignment
                assigned_box_name = None
                # Iterate through box_loc_map to find the box name for this location
                for name, loc in box_loc_map.items():
                    if loc == b_loc:
                        # Found a box at this location. Assign the goal to it.
                        # Assuming unique box locations in state for matching purposes.
                        # If multiple boxes were at b_loc, this would pick one arbitrarily.
                        assigned_box_name = name
                        break # Found the box name for this location

                if assigned_box_name:
                     box_assignments[assigned_box_name] = g_loc
                # Note: If multiple boxes are at the same location, this greedy matching
                # might behave unexpectedly or assign the goal to an arbitrary box at that location.
                # Standard Sokoban has unique box locations. We proceed assuming this.


        # Check if all boxes were matched (should be if #boxes == #goals and all reachable)
        if len(box_assignments) != len(box_loc_map):
             # This indicates a problem with matching (e.g., unequal counts, or unreachable pairs)
             # If dist == math.inf check above didn't catch it, this will.
             return math.inf


        # 5. Calculate robot-to-box cost (h_robot)
        h_robot = math.inf
        boxes_to_move = [(name, loc) for name, loc in box_loc_map.items() if loc != box_assignments.get(name)]

        if not boxes_to_move:
             h_robot = 0 # All boxes are at their assigned goals

        for box_name, b_loc_str in boxes_to_move:
            g_loc_str = box_assignments.get(box_name) # Get assigned goal

            # Find valid pushing positions for this box towards its goal
            current_box_dist_to_goal = self.distances.get(b_loc_str, {}).get(g_loc_str, math.inf)

            # Iterate through all locations adjacent to the box's current location
            # These are potential robot positions (p_loc_str)
            for p_loc_str in self.graph.get(b_loc_str, {}).values():
                # The push direction is from p_loc_str to b_loc_str
                push_direction = get_direction(p_loc_str, b_loc_str, self.graph)
                if push_direction:
                     # The box would move in the opposite direction
                     box_move_direction = get_opposite_direction(push_direction)
                     next_loc_str = get_adjacent_in_direction(b_loc_str, box_move_direction, self.graph)

                     if next_loc_str:
                         # Check if this push reduces the distance to the goal
                         dist_after_push = self.distances.get(next_loc_str, {}).get(g_loc_str, math.inf)

                         if dist_after_push < current_box_dist_to_goal:
                             # This is a valid pushing position p_loc_str for a distance-reducing push
                             # Calculate robot distance to this required position
                             robot_dist = self.distances.get(robot_loc_str, {}).get(p_loc_str, math.inf)
                             h_robot = min(h_robot, robot_dist)

        # If h_robot is still inf and there are boxes to move, it means the robot
        # cannot reach *any* position that allows a push which immediately reduces
        # the distance of *any* box to its assigned goal.
        if h_robot == math.inf and boxes_to_move:
             return math.inf

        # 6. Total heuristic
        # If h_boxes is inf, we already returned inf.
        # If h_robot is inf here, it means boxes_to_move was empty, and h_robot was set to 0.
        # So, h_robot should be finite unless boxes_to_move was non-empty and no valid push pos was found.
        return h_boxes + h_robot
