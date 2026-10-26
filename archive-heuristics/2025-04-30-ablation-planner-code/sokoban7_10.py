from collections import deque
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Utility function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

# BFS for shortest path on the static grid graph
def bfs_dist(start_loc, graph, locations):
    """
    Computes shortest path distances from start_loc to all other locations
    in the static graph. Returns a dictionary {location: distance}.
    """
    distances = {loc: float('inf') for loc in locations}
    if start_loc not in locations:
        # Start location is not part of the known locations (e.g., invalid state)
        return distances

    distances[start_loc] = 0
    queue = deque([start_loc])

    while queue:
        current_loc = queue.popleft()
        current_dist = distances[current_loc]

        if current_loc in graph: # Check if location has neighbors in the graph
            for neighbor in graph[current_loc]:
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = current_dist + 1
                    queue.append(neighbor)

    return distances

# Get the location behind the box relative to the push direction
def get_required_robot_pos(box_loc, next_box_loc, loc_pair_to_dir, adj_graph):
    """
    Given a box moving from box_loc to next_box_loc, find the location
    behind box_loc where the robot must be.
    """
    # Find the direction of the box movement (from box_loc to next_box_loc)
    move_dir = loc_pair_to_dir.get((box_loc, next_box_loc))
    if move_dir is None:
        # Should not happen if next_box_loc is a valid neighbor in the graph
        return None

    # Find the opposite direction
    dir_to_opposite = {'up': 'down', 'down': 'up', 'left': 'right', 'right': 'left'}
    # The robot must be at a location R such that moving from R in 'move_dir' leads to box_loc.
    # This means R is 'opposite_dir' from box_loc.
    required_dir_from_box_loc = dir_to_opposite.get(move_dir)
    if required_dir_from_box_loc is None:
         # Should not happen with standard directions
         return None

    # Find the neighbor of box_loc that is in the required direction (opposite of move_dir)
    if box_loc in adj_graph:
        # We need the location L such that loc_pair_to_dir[(L, box_loc)] == move_dir
        # This is equivalent to finding L such that loc_pair_to_dir[(box_loc, L)] == required_dir_from_box_loc
        # So we look for a neighbor L of box_loc such that the direction from box_loc to L is required_dir_from_box_loc
        for neighbor in adj_graph[box_loc]:
             if loc_pair_to_dir.get((box_loc, neighbor)) == required_dir_from_box_loc:
                 return neighbor

    # No location found behind the box in the required direction (e.g., wall)
    return None


class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing, for each box
    not at its goal, the shortest distance the robot needs to move to get into
    position for the first push towards the goal, plus the shortest distance
    the box needs to be pushed to reach the goal. Distances are calculated
    on the static grid graph, ignoring dynamic obstacles.

    # Assumptions
    - Each box has a unique goal location specified in the goal conditions.
    - The grid structure and adjacency are defined by the 'adjacent' facts.
    - Robot and box movement costs are simplified approximations based on static distances.
    - Robot repositioning cost between pushes along a box's path is ignored or approximated within the box distance term.
    - The heuristic returns infinity if a box cannot reach its goal or the robot cannot reach the required push position on the static graph, indicating a likely unsolvable state or deadlock.

    # Heuristic Initialization
    - Parses static 'adjacent' facts to build a graph representing the grid connectivity.
    - Builds mappings for location pairs to directions and directions to opposite directions.
    - Extracts goal locations for each box from the task goals.
    - Precomputes all-pairs shortest paths on the static grid graph using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current location of the robot and all boxes.
    2. Initialize the total heuristic cost to 0.
    3. For each box that is not currently at its specific goal location:
        a. Calculate the shortest path distance (number of pushes) for the box from its current location to its goal location on the static grid graph (ignoring dynamic obstacles). Let this be `box_dist`. If the goal is unreachable on the static graph, the state is likely unsolvable from here, return infinity.
        b. Find a neighbor location (`next_box_loc`) of the box's current location that lies on a shortest path towards the goal location on the static graph. This location is one step closer to the goal.
        c. Determine the required location (`robot_push_pos`) where the robot must be, adjacent to the box's current location, to push the box towards `next_box_loc`. This position is behind the box relative to the push direction.
        d. Calculate the shortest path distance for the robot from its current location to `robot_push_pos` on the static grid graph (ignoring dynamic obstacles). Let this be `robot_dist`. If the position is unreachable on the static graph, return infinity.
        e. Add `robot_dist + box_dist` to the total heuristic cost for this box.
    4. The total heuristic value for the state is the sum of these costs for all off-goal boxes.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting static facts and goal conditions."""
        self.goals = task.goals
        static_facts = task.static
        objects = task.objects # Need objects to get all location names

        # Collect all location names
        self.locations = {get_parts(obj)[1] for obj in objects if get_parts(obj)[0] == 'location'}

        # Build adjacency graph and location pair to direction map
        # adj_graph[l1] contains l2 if there is an adjacent fact (l1, l2, dir)
        self.adj_graph = {}
        # loc_pair_to_dir[(l1, l2)] stores the direction from l1 to l2
        self.loc_pair_to_dir = {}
        self.dir_to_opposite_dir = {'up': 'down', 'down': 'up', 'left': 'right', 'right': 'left'}

        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'adjacent':
                loc1, loc2, direction = parts[1], parts[2], parts[3]
                if loc1 not in self.adj_graph:
                    self.adj_graph[loc1] = []
                self.adj_graph[loc1].append(loc2)
                self.loc_pair_to_dir[(loc1, loc2)] = direction

        # Precompute all-pairs shortest paths on the static graph
        # dist_map[start_loc][end_loc] stores the shortest distance
        self.dist_map = {}
        for start_loc in self.locations:
             self.dist_map[start_loc] = bfs_dist(start_loc, self.adj_graph, self.locations)

        # Store goal locations for each box
        self.goal_locations = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "at":
                box, location = parts[1], parts[2]
                self.goal_locations[box] = location

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Get current robot location
        robot_loc = None
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at-robot':
                robot_loc = parts[1]
                break
        if robot_loc is None:
             # Robot location not found in state - should not happen in a valid Sokoban state
             return float('inf')

        # Get current box locations
        box_locs = {}
        for fact in state:
            parts = get_parts(fact)
            # Check if the fact is an 'at' predicate for an object that is a box
            # We identify boxes by checking if they are in our goal_locations map
            if parts[0] == 'at' and len(parts) == 3 and parts[1] in self.goal_locations:
                box, location = parts[1], parts[2]
                box_locs[box] = location

        total_cost = 0

        # Calculate cost for each box not at its goal
        for box_name, goal_loc in self.goal_locations.items():
            current_box_loc = box_locs.get(box_name)

            if current_box_loc is None:
                 # Box location not found in state - should not happen
                 return float('inf')

            if current_box_loc == goal_loc:
                continue # Box is already at its goal

            # 1. Box distance to goal (k) using precomputed static distances
            box_dist = self.dist_map.get(current_box_loc, {}).get(goal_loc, float('inf'))
            if box_dist == float('inf'):
                # Box cannot reach its goal on the static graph - likely deadlock
                return float('inf')

            # 2. Find the first step location (l_1) on a shortest path for the box
            # We look for a neighbor of current_box_loc that is one step closer to the goal
            next_box_loc = None
            if current_box_loc in self.adj_graph:
                for neighbor in self.adj_graph[current_box_loc]:
                    # Check if neighbor is in the dist_map for goal_loc
                    dist_from_neighbor_to_goal = self.dist_map.get(neighbor, {}).get(goal_loc, float('inf'))
                    if dist_from_neighbor_to_goal == box_dist - 1:
                        next_box_loc = neighbor
                        break # Found one shortest path step

            if next_box_loc is None and box_dist > 0:
                 # If box_dist > 0, there must be a neighbor one step closer on a shortest path
                 # If we didn't find one, something is wrong with the graph or distance map
                 return float('inf') # Error or unexpected graph structure

            # 3. Determine the required robot position (r_0) for the first push
            required_robot_pos = get_required_robot_pos(current_box_loc, next_box_loc, self.loc_pair_to_dir, self.adj_graph)
            if required_robot_pos is None:
                # Cannot get behind the box in the required direction on the static graph - likely deadlock
                return float('inf')

            # 4. Robot distance to the required push position using precomputed static distances
            robot_dist = self.dist_map.get(robot_loc, {}).get(required_robot_pos, float('inf'))
            if robot_dist == float('inf'):
                # Robot cannot reach the required push position on the static graph - likely deadlock
                return float('inf')

            # Add cost for this box: robot moves to push pos + box pushes
            # Using robot_dist + box_dist as the heuristic contribution for this box
            total_cost += robot_dist + box_dist

        return total_cost
