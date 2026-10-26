import math
from fnmatch import fnmatch
from collections import deque
# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

# Helper function to parse location string like 'loc_r_c'
def parse_location(loc_str):
    """Parses a location string 'loc_r_c' into a tuple (r, c)."""
    parts = loc_str.split('_')
    if len(parts) == 3 and parts[0] == 'loc':
        try:
            r = int(parts[1])
            c = int(parts[2])
            return (r, c)
        except ValueError:
            pass
    return None

# Helper function to get opposite direction
def get_opposite_direction(direction):
    """Returns the opposite direction."""
    if direction == 'up': return 'down'
    if direction == 'down': return 'up'
    if direction == 'left': return 'right'
    if direction == 'right': return 'left'
    return None # Should not happen with valid directions

# BFS function to find shortest path distance
def bfs(start_loc, target_locs, adjacency_graph, occupied_locations):
    """
    Performs BFS to find the shortest path distance from start_loc to any target_loc in target_locs,
    avoiding occupied_locations.
    """
    if not target_locs:
        return float('inf')

    q = deque([(start_loc, 0)])
    visited = {start_loc}

    while q:
        curr_loc, dist = q.popleft()

        if curr_loc in target_locs:
            return dist

        neighbors = adjacency_graph.get(curr_loc, set())

        for next_loc in neighbors:
            # Check if the next location is in the set of locations the robot *cannot* enter.
            if next_loc not in visited and next_loc not in occupied_locations:
                visited.add(next_loc)
                q.append((next_loc, dist + 1))

    return float('inf')

# Assuming Heuristic base class is defined elsewhere and imported
# from heuristics.heuristic_base import Heuristic

class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach a goal state by summing, for each box not at its goal:
    1. The shortest path distance for the box to its goal location on the grid (ignoring obstacles).
    2. The shortest path distance for the robot to reach a location adjacent to the box from which it can push the box towards the goal (avoiding other boxes and non-clear cells).

    # Assumptions
    - The grid structure is defined by `adjacent` facts.
    - Locations are named `loc_r_c` allowing coordinate extraction.
    - Each box has a specific goal location defined in the task goals. (Assumes a one-to-one mapping or picks one goal per box).
    - The cost of a 'move' action is 1.
    - The cost of a 'push' action is 1.
    - The heuristic estimates the number of actions (moves + pushes).

    # Heuristic Initialization
    - Parses `task.static` to build:
        - An undirected adjacency graph of locations.
        - A directed adjacency map including directions.
        - Mappings between location strings (`loc_r_c`) and coordinate tuples (`(r, c)`).
    - Stores the set of all locations in the graph.
    - Parses `task.goals` to create a mapping from each box to its goal location.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current location of the robot, all boxes, and all clear locations.
    2. Create a set of locations currently occupied by boxes.
    3. Initialize the total heuristic value `h` to 0.
    4. For each box specified in the goal conditions:
       a. Determine the box's current location (`box_l`) and its goal location (`goal_l`).
       b. If `box_l` is the same as `goal_l`, this box is already satisfied; continue to the next box.
       c. Calculate the estimated number of pushes needed for this box: This is the shortest path distance from `box_l` to `goal_l` on the undirected grid graph, ignoring all obstacles. Use BFS for this (`dist_box`). If unreachable, the state is likely unsolvable; return infinity.
       d. Identify potential robot push locations: These are locations `P` such that `P` is adjacent to `box_l` and pushing from `P` towards `box_l` moves the box towards `goal_l`. Determine the required push directions (up, down, left, or right) based on the relative coordinates of `box_l` and `goal_l`. For each required direction `D`, find the location `P` such that `(adjacent P box_l D)` is true. Collect these into a set `target_push_locations`.
       e. Calculate the estimated cost for the robot to reach one of these push locations: This is the shortest path distance from the robot's current location (`robot_l`) to any location in `target_push_locations` on the undirected grid graph. The robot cannot move into locations occupied by *any* box or locations that are not clear. Use BFS for this (`dist_robot`). If unreachable, return infinity.
       f. Add `dist_box + dist_robot` to the total heuristic `h`.
    5. After processing all boxes, check if all goal conditions are met in the current state. If yes, the heuristic is 0. Otherwise, return the calculated `h`.

    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        self.adj_map_directed = {} # loc -> { (neighbor_loc, direction), ... }
        self.adj_map_undirected = {} # loc -> { neighbor_loc, ... }
        self.loc_to_coords = {} # loc_r_c -> (r, c)

        # Build adjacency graphs and location-coordinate maps
        all_locations = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == 'adjacent' and len(parts) == 4:
                loc1, loc2, direction = parts[1], parts[2], parts[3]
                all_locations.add(loc1)
                all_locations.add(loc2)

                # Directed graph
                if loc1 not in self.adj_map_directed:
                    self.adj_map_directed[loc1] = set()
                self.adj_map_directed[loc1].add((loc2, direction))

                # Undirected graph
                if loc1 not in self.adj_map_undirected:
                    self.adj_map_undirected[loc1] = set()
                self.adj_map_undirected[loc1].add(loc2)
                if loc2 not in self.adj_map_undirected:
                    self.adj_map_undirected[loc2] = set()
                self.adj_map_undirected[loc2].add(loc1) # Add reverse connection for undirected graph

        # Build location-coordinate maps
        for loc in all_locations:
            coords = parse_location(loc)
            if coords:
                self.loc_to_coords[loc] = coords

        # Store all locations in the graph for quick lookup
        self.all_locations_in_graph = set(self.adj_map_undirected.keys())


        # Store goal locations for each box
        self.box_goals = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == 'at' and len(parts) == 3:
                box, location = parts[1], parts[2]
                # Assuming one goal location per box for simplicity
                # If a box appears multiple times in goals, the last one parsed wins.
                # If multiple boxes target the same location, this map stores it correctly.
                self.box_goals[box] = location

    def get_neighbor_by_direction(self, loc, direction):
        """Finds the neighbor location from loc in the given direction using the directed graph."""
        if loc not in self.adj_map_directed:
            return None
        for neighbor_loc, dir in self.adj_map_directed.get(loc, set()):
            if dir == direction:
                return neighbor_loc
        return None

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Check if goal is reached first for efficiency
        if self.goals.issubset(state):
             return 0

        # Find robot location, current box locations, and clear locations
        robot_l = None
        current_box_locations = {}
        clear_locations = set()
        for fact in state:
            parts = get_parts(fact)
            if parts:
                if parts[0] == 'at-robot' and len(parts) == 2:
                    robot_l = parts[1]
                elif parts[0] == 'at' and len(parts) == 3 and parts[1] in self.box_goals:
                    box, location = parts[1], parts[2]
                    current_box_locations[box] = location
                elif parts[0] == 'clear' and len(parts) == 2:
                    clear_locations.add(parts[1])

        if robot_l is None:
             # Should not happen in a valid state
             return float('inf')

        # Set of locations occupied by boxes
        box_occupied_locations = set(current_box_locations.values())

        total_cost = 0

        # Calculate cost for each box not at its goal
        for box, goal_l in self.box_goals.items():
            box_l = current_box_locations.get(box)

            if box_l is None:
                 # Box defined in goal is not found in the current state facts.
                 # This state might be invalid or represents a situation where the box is lost.
                 # Treat as unreachable.
                 return float('inf')

            if box_l == goal_l:
                continue # Box is already at its goal

            # 1. Estimate box movement cost (pushes needed)
            # BFS for box path, ignoring all obstacles. This is the minimum grid distance.
            dist_box = bfs(box_l, {goal_l}, self.adj_map_undirected, set())

            if dist_box == float('inf'):
                # Box goal is unreachable on the grid itself
                return float('inf')

            # 2. Estimate robot movement cost to reach a push position
            target_push_locations = set()
            box_coords = self.loc_to_coords.get(box_l)
            goal_coords = self.loc_to_coords.get(goal_l)

            if box_coords is None or goal_coords is None:
                 # Location coordinates not parsed correctly
                 return float('inf')

            box_r, box_c = box_coords
            goal_r, goal_c = goal_coords

            # Determine required push directions from box_l towards goal_l
            required_push_dirs = set()
            if box_r > goal_r: required_push_dirs.add('up')
            if box_r < goal_r: required_push_dirs.add('down')
            if box_c > goal_c: required_push_dirs.add('left')
            if box_c < goal_c: required_push_dirs.add('right')

            # Find the locations P adjacent to box_l such that pushing from P moves box_l
            # in a required direction. This means (adjacent P box_l D) where D is a required_push_dir.
            # Iterate through neighbors of box_l in the *undirected* graph,
            # and for each neighbor P, check if the directed edge from P to box_l exists
            # and has a direction in required_push_dirs.

            if box_l in self.adj_map_undirected:
                for potential_push_loc in self.adj_map_undirected[box_l]:
                    if potential_push_loc in self.adj_map_directed:
                         for neighbor_of_potential, dir_from_potential in self.adj_map_directed.get(potential_push_loc, set()):
                             if neighbor_of_potential == box_l and dir_from_potential in required_push_dirs:
                                 target_push_locations.add(potential_push_loc)

            if not target_push_locations:
                 # No valid push position exists for this box towards the goal.
                 # The box might be trapped or the goal is unreachable from here.
                 return float('inf')

            # BFS for robot path to any target push location.
            # Robot cannot move into locations occupied by *any* box.
            # Robot also cannot move into locations that are *not* clear.
            # The set of locations the robot *cannot* move into are those NOT in clear_locations OR occupied by boxes.
            # This is equivalent to (all_locations - clear_locations) | box_occupied_locations
            # i.e., non-clear locations OR box locations.

            # Locations the robot cannot enter during BFS traversal
            occupied_for_robot_bfs = box_occupied_locations | (self.all_locations_in_graph - clear_locations)

            dist_robot = bfs(robot_l, target_push_locations, self.adj_map_undirected, occupied_for_robot_bfs)

            if dist_robot == float('inf'):
                # Robot cannot reach any valid push position for this box
                return float('inf')

            # Add estimated costs for this box
            # dist_box is minimum pushes. dist_robot is cost to get to *a* push position.
            # Total cost for this box = pushes + robot_cost_to_first_push_pos
            total_cost += dist_box + dist_robot

        # If we reached here, it means goals are not fully met, and no box/robot path was infinite.
        # total_cost is the sum of estimates for all boxes not at their goal.
        return total_cost
