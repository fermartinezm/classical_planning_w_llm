import math
from collections import deque

# Assuming the Operator and Task classes are available in the environment
# where this heuristic is used. No need to include their code here.
# from task import Operator, Task

class sokobanHeuristic:
    """
    Domain-dependent heuristic for the Sokoban domain.

    Summary:
        This heuristic estimates the cost to reach a goal state by summing
        the grid distances of each box to its goal location and adding the
        minimum distance the robot needs to travel through clear squares
        to reach a position adjacent to any box.

    Assumptions:
        - The problem grid is defined by 'adjacent' facts.
        - Each box has a unique name and is assigned a specific goal location
          in the goal state.
        - The grid is connected, allowing BFS to find distances.

    Heuristic Initialization:
        In the constructor (__init__), the heuristic pre-processes static
        information from the task:
        1. Builds a graph representation of the grid connectivity based on
           'adjacent' facts.
        2. Identifies all unique locations in the grid.
        3. Pre-calculates all-pairs shortest path distances between any two
           locations on the full grid graph using BFS. This represents the
           minimum number of steps (moves or pushes) between locations
           ignoring dynamic obstacles.
        4. Parses the goal state to determine the target location for each box.

    Step-By-Step Thinking for Computing Heuristic:
        For a given state (in the __call__ method):
        1. Extract the current location of the robot and each box from the state facts.
        2. Identify the set of locations that are currently 'clear' (empty)
           according to the state facts. Add the robot's current location
           to this set, as the robot can move from its current spot.
        3. Calculate the total 'box-to-goal' distance: For each box, find the
           pre-calculated grid distance from its current location to its goal
           location. Sum these distances. If this sum is 0, all boxes are
           at their goals, and the heuristic is 0.
        4. If the total box-to-goal distance is greater than 0, calculate the
           minimum distance for the robot to reach a 'push position' adjacent
           to any box:
           a. Build a robot movement graph where nodes are the 'clear' locations
              (including the robot's current location) and edges connect
              adjacent clear locations.
           b. Perform a BFS starting from the robot's current location on this
              robot movement graph to find shortest path distances to all
              reachable clear locations.
           c. For each box, find all locations adjacent to it in the full grid.
           d. Find the minimum distance from the robot's current location
              (calculated in step 4b) to any of these adjacent locations
              that are reachable by the robot (i.e., are in the robot movement graph).
              This is the 'min_robot_push_dist'. If no such location is reachable,
              it implies the robot is blocked from reaching any box via clear squares,
              and the heuristic returns infinity.
        5. The final heuristic value is the sum of the total box-to-goal grid
           distance (step 3) and the minimum robot distance to a push position
           (step 4d).
    """

    def __init__(self, task):
        self.task = task
        self.grid_adj = {}
        self.locations = set()
        self.box_goals = {}
        self.grid_distances = {}
        self.opposite_direction = {'up': 'down', 'down': 'up', 'left': 'right', 'right': 'left'}

        # 1. Build grid graph and collect locations from static facts
        for fact_string in task.static:
            pred, args = self._parse_fact(fact_string)
            if pred == 'adjacent':
                if len(args) == 3:
                    loc1, loc2, direction = args
                    self.locations.add(loc1)
                    self.locations.add(loc2)
                    if loc1 not in self.grid_adj:
                        self.grid_adj[loc1] = {}
                    if loc2 not in self.grid_adj:
                        self.grid_adj[loc2] = {}
                    self.grid_adj[loc1][direction] = loc2
                    # Add reverse direction
                    if direction in self.opposite_direction:
                         reverse_dir = self.opposite_direction[direction]
                         self.grid_adj[loc2][reverse_dir] = loc1

        # Ensure all locations mentioned in initial/goal states are included
        # This is important if some locations exist but have no adjacencies listed
        # (e.g., isolated islands, though unlikely in typical Sokoban).
        # Also ensures locations with only robot/box/clear facts are known.
        all_facts = set(task.initial_state) | set(task.goals)
        for fact_string in all_facts:
             pred, args = self._parse_fact(fact_string)
             if pred in ('at-robot', 'clear'):
                 if args: self.locations.add(args[0])
             elif pred == 'at':
                 # args is ['box_name', 'loc_name']
                 if len(args) > 1: self.locations.add(args[1])

        # Ensure all locations are keys in grid_adj even if they have no adjacencies listed (e.g., isolated spots)
        for loc in self.locations:
            if loc not in self.grid_adj:
                self.grid_adj[loc] = {}

        # 2. Pre-calculate all-pairs shortest paths on the full grid
        self._precompute_grid_distances()

        # 3. Parse goal state to find box goals
        for goal_fact in task.goals:
            pred, args = self._parse_fact(goal_fact)
            if pred == 'at':
                # args is ['box_name', 'loc_name']
                if len(args) > 1:
                    box_name, goal_loc = args
                    self.box_goals[box_name] = goal_loc
                # Handle potential goal facts that are not 'at' predicates if necessary,
                # but Sokoban goals are typically just box locations.

    def _parse_fact(self, fact_string):
        """Helper to parse a PDDL fact string."""
        # Remove surrounding brackets and split by space
        parts = fact_string.strip('()').split()
        if not parts:
            return None, [] # Handle empty string case
        return parts[0], parts[1:]

    def _bfs(self, start_node, graph, valid_nodes=None):
        """Performs BFS to find distances from start_node on a potentially restricted graph."""
        distances = {}
        queue = deque([(start_node, 0)])
        visited = {start_node}

        # Check if the start node is valid for this specific BFS graph
        if valid_nodes is not None and start_node not in valid_nodes:
             # This BFS cannot start from here. Return empty distances.
             return {}

        while queue:
            (curr_node, dist) = queue.popleft()
            distances[curr_node] = dist

            # Neighbors are values in the adjacency dictionary for curr_node
            # Use graph.get(curr_node, {}) to handle nodes with no entries
            neighbors = graph.get(curr_node, {}).values()

            for neighbor_node in neighbors:
                # Check if neighbor is in the set of valid nodes for this BFS
                if valid_nodes is not None and neighbor_node not in valid_nodes:
                    continue # Skip if neighbor is not a valid node

                if neighbor_node not in visited:
                    visited.add(neighbor_node)
                    queue.append((neighbor_node, dist + 1))

        return distances


    def __call__(self, state):
        """
        Computes the heuristic value for the given state.
        """
        robot_loc = None
        box_locations = {} # {box_name: loc}
        clear_locations = set() # {loc}
        # occupied_locations = set() # {loc} # Not strictly needed for this heuristic logic

        # Extract dynamic information from the state
        for fact_string in state:
            pred, args = self._parse_fact(fact_string)
            if pred == 'at-robot':
                if args:
                    robot_loc = args[0]
                    # occupied_locations.add(robot_loc) # Not needed
            elif pred == 'at':
                # args is ['box_name', 'loc_name']
                if len(args) > 1:
                    box_name, loc = args
                    box_locations[box_name] = loc
                    # occupied_locations.add(loc) # Not needed
            elif pred == 'clear':
                if args:
                    clear_locations.add(args[0])

        # Ensure robot_loc is known
        if robot_loc is None:
             # This state is malformed, robot location is unknown.
             # Should not happen in a valid Sokoban state.
             return float('inf') # Or handle as error

        # 1. Calculate total box-to-goal grid distance
        total_box_grid_dist = 0
        all_boxes_at_goal = True
        # Iterate through boxes found in the current state
        for box_name, current_loc in box_locations.items():
            goal_loc = self.box_goals.get(box_name)
            if goal_loc is None:
                 # Goal location for this box is unknown. Should not happen if goals are parsed correctly.
                 # Or maybe a box exists in state but not in goals? Assume goals list all relevant boxes.
                 continue # Skip boxes not in goals

            if current_loc != goal_loc:
                all_boxes_at_goal = False
                # Use pre-calculated grid distance
                dist = self.grid_distances.get((current_loc, goal_loc), float('inf'))
                if dist == float('inf'):
                    # A box is in a location unreachable from its goal in the full grid.
                    # This state is likely unsolvable.
                    return float('inf')
                total_box_grid_dist += dist

        # If all boxes are at their goals, the heuristic is 0
        if all_boxes_at_goal:
            return 0

        # 2. Calculate minimum robot distance to a push position for any box
        min_robot_push_dist = float('inf')

        # Build the robot movement graph for the current state
        # Nodes are locations the robot can potentially be on (clear locations + robot's location)
        # Edges exist between adjacent nodes if both are in the set of movable locations.
        # A location is movable by the robot if it is clear or is the robot's current location.
        robot_movable_locations = set(clear_locations)
        robot_movable_locations.add(robot_loc) # Robot can move from its current spot

        robot_movement_graph = {}
        for loc in robot_movable_locations:
             robot_movement_graph[loc] = {}
             # Check neighbors from the full grid graph
             for direction, neighbor in self.grid_adj.get(loc, {}).items():
                  # An edge exists in the robot graph if the neighbor is also movable
                  if neighbor in robot_movable_locations:
                       robot_movement_graph[loc][direction] = neighbor

        # Calculate robot shortest paths from current location on the robot movement graph
        # The start node (robot_loc) is guaranteed to be in robot_movable_locations
        robot_clear_path_distances = self._bfs(robot_loc, robot_movement_graph, valid_nodes=robot_movable_locations)

        # Find minimum distance to any location adjacent to any box
        found_reachable_push_pos = False
        for box_name, box_loc in box_locations.items():
            # Find locations adjacent to the box in the full grid
            adjacent_to_box = self.grid_adj.get(box_loc, {}).values()
            for adj_loc in adjacent_to_box:
                # Check if the robot can reach this adjacent location via clear squares
                if adj_loc in robot_clear_path_distances:
                    dist = robot_clear_path_distances[adj_loc]
                    min_robot_push_dist = min(min_robot_push_dist, dist)
                    found_reachable_push_pos = True

        # If robot cannot reach any location adjacent to any box via clear squares
        if not found_reachable_push_pos:
             # This state might be a deadlock or require complex moves to free path
             # Return infinity as a strong penalty for greedy search
             return float('inf')


        # 3. Combine distances
        # The total heuristic is the sum of box distances and the cost for the robot
        # to get into position for the *first* push.
        # This is a simplification; each push requires robot positioning.
        # However, for a non-admissible, efficient heuristic, this is a common approach.
        # It captures the main components: box progress and robot accessibility.

        return total_box_grid_dist + min_robot_push_dist
