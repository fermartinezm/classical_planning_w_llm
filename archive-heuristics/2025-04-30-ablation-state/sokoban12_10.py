from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque
import math # Import math for infinity

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

# BFS implementation to find shortest distance
def bfs(start_loc, target_locs, graph, obstacles):
    """
    Finds the shortest path distance from start_loc to any target_loc,
    avoiding obstacles. Returns distance or float('inf') if unreachable.
    """
    if start_loc in obstacles:
        return math.inf

    queue = deque([(start_loc, 0)])
    visited = {start_loc}

    while queue:
        current_loc, dist = queue.popleft()

        if current_loc in target_locs:
            return dist

        # Locations not in the graph are not traversable
        if current_loc not in graph:
             continue

        # Explore neighbors
        for direction, neighbor_loc in graph[current_loc].items():
            if neighbor_loc not in visited and neighbor_loc not in obstacles:
                visited.add(neighbor_loc)
                queue.append((neighbor_loc, dist + 1))

    return math.inf # Target not reachable

# BFS to find shortest path (sequence of locations)
def bfs_shortest_path(start_loc, target_loc, graph):
    """
    Finds the shortest path (list of locations) from start_loc to target_loc
    on the graph. Returns None if unreachable.
    """
    queue = deque([(start_loc, [start_loc])])
    visited = {start_loc}

    while queue:
        current_loc, path = queue.popleft()

        if current_loc == target_loc:
            return path

        # Locations not in the graph are not traversable
        if current_loc not in graph:
            continue

        # Explore neighbors
        for direction, neighbor_loc in graph[current_loc].items():
            if neighbor_loc not in visited:
                visited.add(neighbor_loc)
                queue.append((neighbor_loc, path + [neighbor_loc]))

    return None # Target not reachable

# Function to find the direction from loc1 to loc2
def find_direction(loc1, loc2, graph):
    """Finds the direction from loc1 to loc2 if they are adjacent."""
    if loc1 in graph:
        for direction, neighbor in graph[loc1].items():
            if neighbor == loc2:
                return direction
    return None # Not adjacent or loc1 not in graph

# Function to find the location adjacent to loc in the opposite direction
def find_opposite_location(loc, direction, graph):
    """Finds the location adjacent to `loc` in the direction opposite to `direction`."""
    opposite_dir = None
    if direction == 'up': opposite_dir = 'down'
    elif direction == 'down': opposite_dir = 'up'
    elif direction == 'left': opposite_dir = 'right'
    elif direction == 'right': opposite_dir = 'left'

    # The robot needs to be at a location `robot_push_pos` such that
    # `adjacent(robot_push_pos, current_loc, push_dir)`.
    # This means `current_loc` is adjacent to `robot_push_pos` in `push_dir`.
    # So, we need to find a location `l` such that `graph[l]` contains `push_dir`
    # and `graph[l][push_dir] == loc`.

    push_dir = direction # The direction the box moves is the 'dir' in adjacent(rloc, bloc, dir)

    # Iterate through all locations in the graph to find the one adjacent to loc in push_dir
    for potential_rloc in graph:
        if potential_rloc in graph and push_dir in graph[potential_rloc] and graph[potential_rloc][push_dir] == loc:
             return potential_rloc # Found the location where robot must be

    return None # No location exists from which to push in that direction


class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing, for each box
    not at its goal, the shortest path distance for the box to its goal plus
    the shortest path distance for the robot to reach a position from which
    it can push the box one step along a shortest path towards its goal.

    # Assumptions
    - The goal state specifies the target location for each box using the `(at ?box ?location)` predicate.
    - There is a one-to-one mapping between boxes and goal locations specified in the goal.
    - The grid structure and traversable locations are defined by the `adjacent` predicates.
    - Locations that are defined as 'location' objects in the problem, but are not initially clear
      and are not occupied by the robot or a box, are considered permanent obstacles (walls).

    # Heuristic Initialization
    - Extracts all defined 'location' objects from the task facts.
    - Builds a graph representation of the traversable locations based on `adjacent` facts.
    - Extracts the goal location for each box from the task's goal conditions.
    - Identifies permanent obstacles (walls) from the initial state and the list of all locations.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current location of the robot and each box that has a goal.
    2. Identify locations currently occupied by other boxes (not the one being considered) and permanent obstacles. These are obstacles for the robot's movement.
    3. Initialize the total heuristic cost to 0.
    4. For each box that is not currently at its designated goal location:
        a. Calculate the shortest path distance for the box from its current location to its goal location. This BFS is performed on the traversable location graph, ignoring other boxes and the robot as obstacles for the box itself. This distance represents the minimum number of pushes required for this box assuming a clear path.
        b. If the box cannot reach its goal (no path exists on the traversable graph), the state is likely unsolvable; return infinity.
        c. If the box is already at the goal (distance is 0), continue to the next box.
        d. If the box needs to move (distance > 0), find the location adjacent to the box's current location that is the first step on *a* shortest path towards the goal.
        e. Determine the required push direction: this is the direction from the box's current location to the next location found in step 4d.
        f. Find the location where the robot must be situated to perform this push. According to the PDDL `push` action, the robot must be adjacent to the box in the direction the box is being pushed.
        g. Check if this required robot push position is occupied by another box or is a permanent obstacle. If so, the robot cannot get there, and the state is likely unsolvable or requires complex unblocking; return infinity.
        h. Calculate the shortest path distance for the robot from its current location to this required pushing position. This BFS considers other boxes (not the one being pushed) and permanent obstacles as blocked locations.
        i. If the robot cannot reach the required pushing position avoiding obstacles, the state is likely unsolvable; return infinity.
        j. Add the box's shortest path distance (minimum pushes) and the robot's shortest path distance (to get into position for the first push) to the total heuristic cost.
    5. Return the total heuristic cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static
        initial_state = task.initial_state # Need initial state to find permanent obstacles
        all_facts = task.facts # Contains all ground facts, including type predicates

        # Get all possible locations from task.facts
        self.all_locations = {get_parts(fact)[1] for fact in all_facts if get_parts(fact)[0] == 'location'}

        # Build the graph from adjacent facts (only includes traversable edges)
        self.graph = self.build_location_graph(static_facts)

        # Identify permanent obstacles (locations from the full list that are not clear and not occupied initially)
        self.permanent_obstacles = self.identify_permanent_obstacles(initial_state, self.all_locations)

        # Store goal locations for each box
        self.goal_locations = self.extract_goal_locations(self.goals)


    def build_location_graph(self, static_facts):
        """Builds a graph where nodes are locations and edges are adjacencies."""
        graph = {} # Map loc -> {direction: loc}
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'adjacent':
                loc1, loc2, direction = parts[1], parts[2], parts[3]
                if loc1 not in graph:
                    graph[loc1] = {}
                graph[loc1][direction] = loc2
        return graph

    def identify_permanent_obstacles(self, initial_state, all_locations):
        """Identifies locations that are not clear initially and not occupied by robot/box."""
        occupied_init = {get_parts(fact)[1] for fact in initial_state if get_parts(fact)[0] in ['at-robot', 'at']}
        clear_init = {get_parts(fact)[1] for fact in initial_state if get_parts(fact)[0] == 'clear'}
        # Locations that are in the domain (all_locations) but are not clear
        # and not occupied initially are considered permanent obstacles.
        permanent_obstacles = {loc for loc in all_locations if loc not in clear_init and loc not in occupied_init}
        return permanent_obstacles


    def extract_goal_locations(self, goals):
        """Extracts the target location for each box from the goal conditions."""
        goal_locations = {}
        # Goals can be a single fact or an 'and' of facts
        if isinstance(goals, frozenset): # Multiple goals
             goal_facts = goals
        else: # Single goal fact
             goal_facts = {goals}

        for goal in goal_facts:
            parts = get_parts(goal)
            if parts[0] == "at" and len(parts) == 3: # (at ?box ?location)
                box, location = parts[1], parts[2]
                # Assuming the goal specifies locations for objects of type 'box'
                goal_locations[box] = location
            # Ignore other goal types if any

        return goal_locations


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # 1. Get current robot and box locations
        loc_robot = None
        box_locations = {}
        # Locations currently occupied by boxes (excluding the one being considered)
        # and permanent obstacles are obstacles for the robot.
        current_box_occupied = set()

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at-robot':
                loc_robot = parts[1]
            elif parts[0] == 'at' and len(parts) == 3 and parts[1] in self.goal_locations: # Check if it's one of the goal boxes
                 box, loc = parts[1], parts[2]
                 box_locations[box] = loc
                 current_box_occupied.add(loc) # Add to occupied set for robot path calculation

        if loc_robot is None:
             # Should not happen in a valid state
             return math.inf

        total_cost = 0

        # 2. Calculate cost for each box not at its goal
        for box, goal_loc in self.goal_locations.items():
            current_loc = box_locations.get(box)
            if current_loc is None:
                 # This box is not in the state? Should not happen.
                 return math.inf

            if current_loc == goal_loc:
                continue # Box is already home

            # Calculate shortest path for the box (ignoring other boxes/robot as obstacles for the box itself)
            # The box path BFS uses the traversable graph (self.graph). Permanent obstacles
            # are implicitly handled if they are not part of the graph or not adjacent.
            box_path = bfs_shortest_path(current_loc, goal_loc, self.graph)

            if box_path is None:
                # Box cannot reach its goal on the traversable graph
                return math.inf # Unsolvable state

            box_dist = len(box_path) - 1 # Distance is number of steps/pushes

            if box_dist == 0: # Should be caught by current_loc == goal_loc
                 continue

            # Find the required push position for the first step of the box path
            next_loc_b = box_path[1]
            # The direction the box needs to move
            push_dir = find_direction(current_loc, next_loc_b, self.graph)

            if push_dir is None:
                 # Should not happen if box_path is valid and > 0
                 return math.inf

            # The robot needs to be at a location `robot_push_pos` such that
            # `adjacent(robot_push_pos, current_loc, push_dir)`.
            # This means `current_loc` is adjacent to `robot_push_pos` in `push_dir`.
            # So, robot_push_pos is adjacent to current_loc in the direction the box moves.
            robot_push_pos = find_opposite_location(current_loc, push_dir, self.graph)


            if robot_push_pos is None:
                 # No location exists from which the box can be pushed in the required direction.
                 # This box is likely stuck against a wall or boundary.
                 return math.inf # Unsolvable state

            # Calculate shortest path for the robot to reach the push position
            # Obstacles for the robot are other boxes and permanent obstacles.
            robot_obstacles = set(self.permanent_obstacles)
            for b, loc in box_locations.items():
                if b != box: # Other boxes are obstacles for the robot
                    robot_obstacles.add(loc)

            # Check if the required robot_push_pos is itself an obstacle
            if robot_push_pos in robot_obstacles:
                 # Robot cannot reach the required push position because another box or wall is there.
                 return math.inf # Unsolvable state

            robot_dist = bfs(loc_robot, {robot_push_pos}, self.graph, robot_obstacles)

            if robot_dist == math.inf:
                # Robot cannot reach the push position avoiding obstacles
                return math.inf # Unsolvable state

            # Heuristic contribution for this box:
            # Cost = (distance box needs to travel) + (distance robot needs to travel to get ready for the *first* push)
            # Cost = box_dist + robot_dist
            # This sums the minimum pushes required for the box and the cost for the robot
            # to get into position for the first push for this box.

            total_cost += box_dist + robot_dist

        # Heuristic is 0 if all boxes are at their goals.
        return total_cost
