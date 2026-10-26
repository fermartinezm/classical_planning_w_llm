# Required imports
from heuristics.heuristic_base import Heuristic
from collections import deque
import math # For float('inf')

# Helper function to parse PDDL facts
def get_parts(fact):
    """Splits a PDDL fact string into its components."""
    # Remove surrounding parentheses and split by spaces
    return fact[1:-1].split()

class sokobanHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Sokoban domain.

    Summary:
    This heuristic estimates the cost to reach the goal state by summing two components
    for each box that is not yet at its goal location:
    1. The shortest path distance (in terms of grid cells) from the box's current
       location to its goal location. This represents the minimum number of pushes
       required for the box itself, ignoring the robot and other obstacles.
    2. The shortest path distance (in terms of grid cells) from the robot's current
       location to the specific location from which it can perform the *first* push
       required to move the box along a shortest path towards its goal.

    The total heuristic value is the sum of these costs over all misplaced boxes.
    This heuristic is non-admissible as it does not account for the robot's movement
    costs for subsequent pushes for the same box, potential conflicts between boxes,
    or the cost of moving obstacles out of the way. It aims to guide the search
    by prioritizing states where boxes are closer to their goals and the robot
    is well-positioned to start pushing them.

    Assumptions:
    - The domain uses 'loc_row_col' naming convention for locations, but connectivity
      is strictly defined by 'adjacent' facts. The heuristic relies solely on the
      'adjacent' facts to build the graph.
    - Goal states consist of 'at' predicates for boxes at specific locations.
    - The graph defined by 'adjacent' facts is undirected (if l1 is adjacent to l2,
      l2 is adjacent to l1 in the reverse direction). The initialization code
      explicitly adds reverse edges.
    - The heuristic assumes a unique goal location for each box specified in the task goals.
    - All locations mentioned in 'adjacent' facts are valid locations.

    Heuristic Initialization:
    The constructor performs the following steps:
    1. Parses the task goals to identify the target location for each box and collect box names.
    2. Parses the static facts ('adjacent' predicates) to build an adjacency graph
       representing the connections between locations. The graph stores, for each
       location, a mapping from direction to the adjacent location. Reverse
       connections are added explicitly using a predefined mapping for reverse directions.
    3. Stores the set of all locations identified from the graph.

    Step-By-Step Thinking for Computing Heuristic:
    For a given state:
    1. Identify the robot's current location by finding the fact '(at-robot ?l)' in the state.
       If the robot location cannot be found, the state is considered invalid/unsolvable,
       and the heuristic returns infinity.
    2. Identify the current location of each box that has a goal location defined,
       by finding facts '(at ?b ?l)' where ?b is a known box name.
       If a box with a goal is not found in the state, the state is considered invalid/unsolvable,
       and the heuristic returns infinity.
    3. Initialize the total heuristic value `h` to 0.
    4. For each box that is not currently at its goal location:
        a. Calculate the shortest path distance and retrieve a shortest path from the
           box's current location to its goal location using BFS on the adjacency graph.
           The `get_path_info` helper function is used for this.
           If the goal is unreachable for the box (BFS returns infinity distance),
           the state is considered unsolvable, and the heuristic returns infinity.
        b. Add the shortest path distance (which represents the minimum number of pushes
           required for the box itself) to `h`.
        c. If the box needs to move (the distance calculated in 4a is greater than 0):
            i. Determine the location `next_loc_b` that the box should move to in the
               first step along the shortest path towards the goal. This is the second
               location in the shortest path list returned by `get_path_info`.
            ii. Determine the direction `dir_to_next` from the box's current location
                to `next_loc_b` using the `get_direction` helper function and the graph.
            iii. Determine the required robot location `rloc` from which the robot
                 can push the box from its current location (`current_loc`) towards
                 `next_loc_b`. Based on the PDDL `push` action, the robot must be
                 adjacent to `current_loc` in the direction `dir_to_next`. This means
                 `current_loc` is adjacent to `rloc` in the reverse direction. `rloc`
                 is found by looking up `current_loc` in the graph using the reverse
                 of `dir_to_next`.
                 If the required robot location cannot be determined (e.g., no location
                 is adjacent in the reverse direction), the state is considered
                 unsolvable, and the heuristic returns infinity.
            iv. Calculate the shortest path distance from the robot's current location
                to the required robot location `rloc` using BFS (`get_path_info`).
                If `rloc` is unreachable for the robot, the state is considered
                unsolvable, and the heuristic returns infinity.
            v. Add this distance (robot movement cost for the first push) to `h`.
    5. Return the total calculated heuristic value `h`.
    """
    def __init__(self, task):
        super().__init__(task)
        self.goals = task.goals
        static_facts = task.static

        # 1. Parse goals to get box goal locations and box names
        self.goal_locations = {}
        self.box_names = set()
        for goal in self.goals:
            parts = get_parts(goal)
            # Assuming goal predicates are (at box loc)
            if parts[0] == 'at' and len(parts) == 3:
                box_name, goal_loc = parts[1], parts[2]
                self.goal_locations[box_name] = goal_loc
                self.box_names.add(box_name)

        # 2. Build adjacency graph from static facts
        self.graph = {}
        self.reverse_directions = {'up': 'down', 'down': 'up', 'left': 'right', 'right': 'left'}

        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'adjacent' and len(parts) == 4:
                loc1, loc2, direction = parts[1], parts[2], parts[3]
                if loc1 not in self.graph:
                    self.graph[loc1] = {}
                self.graph[loc1][direction] = loc2

                # Add reverse connection
                if loc2 not in self.graph:
                    self.graph[loc2] = {}
                reverse_dir = self.reverse_directions.get(direction)
                if reverse_dir:
                     self.graph[loc2][reverse_dir] = loc1

        self.all_locations = set(self.graph.keys())


    # Helper function to get location from fact string
    def get_location_from_fact(self, fact_string):
        """Extracts the location from an (at-robot) or (at box) fact."""
        parts = get_parts(fact_string)
        if parts[0] == 'at-robot' and len(parts) == 2:
            return parts[1]
        if parts[0] == 'at' and len(parts) == 3 and parts[1] in self.box_names:
             return parts[2]
        return None # Should not happen for relevant facts

    # Helper function to get box name from fact string
    def get_box_name_from_fact(self, fact_string):
        """Extracts the box name from an (at box) fact."""
        parts = get_parts(fact_string)
        if parts[0] == 'at' and len(parts) == 3 and parts[1] in self.box_names:
             return parts[1]
        return None # Should not happen for relevant facts

    # Helper function to perform BFS and get distance and path
    def get_path_info(self, start_loc, end_loc):
        """
        Performs BFS to find the shortest path distance and path
        from start_loc to end_loc on the adjacency graph.
        Returns (distance, path) or (float('inf'), None) if unreachable.
        """
        if start_loc == end_loc:
            return (0, [start_loc])

        # Ensure start and end locations are in the graph
        if start_loc not in self.graph or end_loc not in self.graph:
             return (math.inf, None)

        queue = deque([(start_loc, 0)])
        visited = {start_loc}
        parent = {start_loc: None}

        while queue:
            current_loc, dist = queue.popleft()

            if current_loc == end_loc:
                # Reconstruct path
                path = []
                node = end_loc
                while node is not None:
                    path.append(node)
                    node = parent[node]
                path.reverse()
                return (dist, path)

            # Check if current_loc has neighbors in the graph
            if current_loc in self.graph:
                for neighbor_loc in self.graph[current_loc].values():
                    if neighbor_loc not in visited:
                        visited.add(neighbor_loc)
                        parent[neighbor_loc] = current_loc
                        queue.append((neighbor_loc, dist + 1))

        return (math.inf, None) # End location unreachable

    # Helper function to get direction between adjacent locations
    def get_direction(self, from_loc, to_loc):
        """
        Given two adjacent locations, returns the direction from from_loc to to_loc.
        Assumes to_loc is directly reachable from from_loc in one step.
        """
        if from_loc in self.graph:
            for direction, adj_loc in self.graph[from_loc].items():
                if adj_loc == to_loc:
                    return direction
        return None # Should not happen for adjacent locations

    def __call__(self, node):
        state = node.state

        # 1. Get robot location
        robot_loc = None
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at-robot':
                robot_loc = parts[1]
                break
        if robot_loc is None:
             # This state is likely invalid or represents a failure
             return math.inf

        # 2. Get box locations
        box_locations = {}
        for fact in state:
            box_name = self.get_box_name_from_fact(fact)
            if box_name:
                box_locations[box_name] = self.get_location_from_fact(fact)

        # 3. Calculate heuristic
        h = 0
        for box in self.goal_locations:
            current_loc = box_locations.get(box)
            goal_loc = self.goal_locations[box]

            # Handle case where box is not in the state (shouldn't happen in valid states)
            if current_loc is None:
                 return math.inf

            if current_loc != goal_loc:
                # Cost for box movement (minimum pushes)
                dist_box_goal, path_box_goal = self.get_path_info(current_loc, goal_loc)

                if dist_box_goal == math.inf:
                    # Box cannot reach its goal
                    return math.inf

                h += dist_box_goal # Add minimum pushes for the box

                # Cost for robot to get into position for the first push
                if dist_box_goal > 0: # Box needs to move
                    # The location the box moves to in the first step
                    next_loc_b = path_box_goal[1]
                    # The direction of the first push
                    dir_to_next = self.get_direction(current_loc, next_loc_b)

                    # The required robot location (adjacent to current_loc in dir_to_next)
                    # This is the location R such that adjacent(R, current_loc, dir_to_next)
                    # which is equivalent to adjacent(current_loc, R, reverse(dir_to_next))
                    reverse_dir = self.reverse_directions.get(dir_to_next)
                    if reverse_dir is None:
                         # Should not happen if dir_to_next is a valid direction
                         return math.inf

                    required_robot_loc = self.graph.get(current_loc, {}).get(reverse_dir)

                    if required_robot_loc is None:
                         # This indicates a problem with the graph or path logic, or an unsolvable state
                         # where the required pushing position doesn't exist (e.g., wall behind box).
                         return math.inf

                    # Distance for robot to reach the required pushing location
                    dist_robot_rloc, _ = self.get_path_info(robot_loc, required_robot_loc)

                    if dist_robot_rloc == math.inf:
                        # Robot cannot reach the required pushing position
                        return math.inf

                    h += dist_robot_rloc # Add cost for robot to reach the first pushing position

        return h
