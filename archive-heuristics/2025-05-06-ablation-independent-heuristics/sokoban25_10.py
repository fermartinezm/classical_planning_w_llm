from heuristics.heuristic_base import Heuristic
from task import Task
from collections import deque
from scipy.optimize import linear_sum_assignment
import math # For float('inf')

class sokobanHeuristic(Heuristic):
    """
    Summary:
        Domain-dependent heuristic for Sokoban. Estimates the cost by summing
        the minimum cost to move all boxes that are not at their goal locations
        to their respective goal locations (using shortest path distances and
        minimum weight matching) and the minimum cost for the robot to reach
        a position from which it can push any of these boxes.

    Assumptions:
        - The domain uses 'at-robot', 'at', 'clear', and 'adjacent' predicates.
        - Locations are represented as strings (e.g., 'loc_R_C').
        - Adjacency is symmetric (if A is adjacent to B, B is adjacent to A).
        - The graph formed by 'adjacent' facts might be disconnected.
        - Each box mentioned in a goal fact has a unique goal location specified
          by a single '(at box goal_loc)' fact in the goals.
        - The number of boxes that need to reach a goal location equals the
          number of goal locations that are not yet satisfied.
        - Scipy library is available for linear sum assignment (Hungarian algorithm).

    Heuristic Initialization:
        1. Collect all unique location names mentioned in the task definition.
        2. Build an adjacency graph based on the 'adjacent' facts.
        3. Precompute shortest path distances between all pairs of locations using BFS.
        4. Identify the goal location for each box from the task goals, storing
           them in a dictionary mapping box names to goal location names.

    Step-By-Step Thinking for Computing Heuristic:
        1. Check if the current state is a goal state by verifying if all goal
           facts are present in the state. If yes, return 0.
        2. Extract the robot's current location, the current location of each box,
           and the set of clear locations from the state. If the robot location
           is missing or unknown, return infinity.
        3. Identify unsatisfied goal facts, which are the '(at box goal_loc)'
           facts from the task goals that are not present in the current state.
        4. From the unsatisfied goal facts, determine the set of boxes that need
           to reach a goal and their corresponding target goal locations. Ensure
           target locations are known.
        5. Get the current locations of these boxes from the state. If any box
           needed for a goal is not found in the state or is in an unknown location,
           return infinity.
        6. If there are no boxes that need to reach a goal (i.e., no unsatisfied
           goal facts), the goal must be reached (checked in step 1), return 0.
        7. Verify that the number of boxes needing a goal matches the number of
           target goal locations. If they don't match, return infinity as the
           state is likely unsolvable or malformed for this heuristic's assumptions.
        8. Create a cost matrix where entry (i, j) is the shortest path distance
           from the current location of the i-th box needing a goal to the j-th
           target goal location. Use infinity if no path exists.
        9. Use the Hungarian algorithm (linear_sum_assignment) on the cost matrix
           to find the minimum cost perfect matching between the current box
           locations and the target goal locations. The sum of costs in this
           matching is the 'box_goal_distance'. If the minimum matching cost is
           infinity, return infinity.
        10. Calculate the minimum distance for the robot to reach *any* location
            that is adjacent to *any* box needing a goal and is currently clear.
            This is the 'min_robot_distance'. Iterate through all boxes needing
            a goal, find their adjacent locations, check if the adjacent location
            is clear in the current state, and find the shortest path distance
            from the robot's current location to that clear adjacent location.
            Take the minimum distance found. If no such clear adjacent location
            is reachable, return infinity.
        11. The final heuristic value is the sum of 'box_goal_distance' and
            'min_robot_distance'.
    """

    def __init__(self, task):
        super().__init__()
        self.goals = task.goals
        self.static = task.static

        # 1. Collect all unique location names from all ground facts
        self.all_locations = set()
        # task.facts contains all possible ground facts defined in the domain/problem
        for fact_string in task.facts:
            predicate, args = self.parse_fact(fact_string)
            if predicate in ('at-robot', 'clear'):
                if args: self.all_locations.add(args[0])
            elif predicate == 'at':
                if len(args) > 1: self.all_locations.add(args[1]) # Location argument is the second one
            elif predicate == 'adjacent':
                if len(args) > 1: self.all_locations.add(args[0])
                if len(args) > 2: self.all_locations.add(args[1]) # Locations are first two args

        # 2. Build adjacency graph
        self.graph = {loc: set() for loc in self.all_locations}
        for fact_string in self.static:
            predicate, args = self.parse_fact(fact_string)
            if predicate == 'adjacent' and len(args) > 1:
                loc1, loc2 = args[0], args[1]
                # Ensure locations are in our collected set before adding to graph
                if loc1 in self.graph and loc2 in self.graph:
                    self.graph[loc1].add(loc2)
                    self.graph[loc2].add(loc1) # Assuming symmetric adjacency

        # 3. Precompute all-pairs shortest paths
        self.distances = {}
        for start_loc in self.all_locations:
            dist_map = self.bfs(start_loc, self.graph)
            for end_loc, dist in dist_map.items():
                 self.distances[(start_loc, end_loc)] = dist

        # 4. Identify goal location for each box (Mapping box name to its goal location)
        self.box_goals = {} # {box_name: goal_loc}
        for goal_fact_string in self.goals:
            predicate, args = self.parse_fact(goal_fact_string)
            if predicate == 'at' and len(args) > 1:
                box_name, goal_loc = args[0], args[1]
                self.box_goals[box_name] = goal_loc

    def parse_fact(self, fact_string):
        # Helper to parse a PDDL fact string like '(predicate arg1 arg2)'
        if not isinstance(fact_string, str) or not fact_string.startswith('(') or not fact_string.endswith(')'):
             return None, []
        parts = fact_string[1:-1].split()
        if not parts:
            return None, []
        predicate = parts[0]
        args = parts[1:]
        return predicate, args

    def bfs(self, start_loc, graph):
        # Breadth-First Search to find shortest distances from start_loc
        distances = {loc: float('inf') for loc in graph}
        if start_loc not in graph:
             # Start location is not in the graph (e.g., isolated or malformed fact).
             # Distances to all other nodes remain infinity.
             return distances

        distances[start_loc] = 0
        queue = deque([start_loc])

        while queue:
            curr_loc = queue.popleft()
            current_dist = distances[curr_loc]

            # Check if curr_loc is in graph (should be if initialized correctly)
            if curr_loc in graph:
                for neighbor in graph.get(curr_loc, set()): # Use .get for safety
                    # Check if neighbor is a known location before accessing its distance
                    if neighbor in distances and distances[neighbor] == float('inf'):
                        distances[neighbor] = current_dist + 1
                        queue.append(neighbor)
        return distances

    def __call__(self, node):
        state = node.state

        # 1. Check if goal is reached
        if self.goals <= state:
            return 0

        # 2. Extract robot, box, clear locations from the current state
        robot_loc = None
        current_box_locs = {} # {box_name: loc}
        clear_locs = set()

        for fact_string in state:
            predicate, args = self.parse_fact(fact_string)
            if predicate == 'at-robot' and args:
                robot_loc = args[0]
            elif predicate == 'at' and len(args) > 1:
                box_name, box_loc = args[0], args[1]
                current_box_locs[box_name] = box_loc
            elif predicate == 'clear' and args:
                clear_locs.add(args[0])

        # If robot location is missing or unknown, return infinity (invalid state)
        if robot_loc is None or robot_loc not in self.all_locations:
             return float('inf')

        # 3. Identify unsatisfied goal facts and 4. determine boxes/targets
        unsatisfied_goal_targets = {} # {box_name: goal_loc}
        for goal_fact_string in self.goals:
             if goal_fact_string not in state:
                 predicate, args = self.parse_fact(goal_fact_string)
                 if predicate == 'at' and len(args) > 1:
                     box_name, goal_loc = args[0], args[1]
                     # Ensure goal_loc is a known location
                     if goal_loc in self.all_locations:
                         unsatisfied_goal_targets[box_name] = goal_loc
                     else:
                         # Goal location is unknown, likely unsolvable
                         return float('inf')


        # The boxes that need moving are the keys in unsatisfied_goal_targets.
        boxes_needing_goal = list(unsatisfied_goal_targets.keys())
        # The target goal locations are the values in unsatisfied_goal_targets.
        target_goal_locs = list(unsatisfied_goal_targets.values())

        # Get the current locations of the boxes that need to reach a goal
        current_locs_of_boxes_needing_goal = []
        for box_name in boxes_needing_goal:
             current_loc = current_box_locs.get(box_name)
             if current_loc is not None and current_loc in self.all_locations:
                  current_locs_of_boxes_needing_goal.append(current_loc)
             else:
                  # A box required by the goal is not in the state or in an unknown location. Unsolvable.
                  return float('inf')

        # 6. If no boxes need moving (all goals satisfied), return 0.
        if not boxes_needing_goal:
             return 0 # Should be caught by the initial goal check, but double check

        # 7. Verify counts match for matching
        num_boxes_to_match = len(current_locs_of_boxes_needing_goal)
        num_target_goals = len(target_goal_locs)

        # These two numbers *must* be equal for a perfect matching interpretation
        # of the Hungarian algorithm, which is standard for N boxes to N goals.
        # If they differ, the problem structure is non-standard or the state is weird.
        # Return infinity in such cases.
        if num_boxes_to_match != num_target_goals:
             # print(f"Warning: Mismatch in boxes to move ({num_boxes_to_match}) and target goals ({num_target_goals})")
             return float('inf')


        # 8. Build cost matrix (N x N)
        cost_matrix = []
        for i in range(num_boxes_to_match):
            row = []
            for j in range(num_target_goals):
                box_loc = current_locs_of_boxes_needing_goal[i]
                goal_loc = target_goal_locs[j]
                # Get distance, default to infinity if locations are unknown or path doesn't exist
                dist = self.distances.get((box_loc, goal_loc), float('inf'))
                row.append(dist)
            cost_matrix.append(row)

        # 9. Use Hungarian algorithm
        # linear_sum_assignment works on rectangular matrices, but we expect square here.
        # It returns row and column indices for the optimal assignment.
        try:
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
        except ValueError:
             # This might happen if cost_matrix is empty or malformed, though checks above should prevent it.
             return float('inf')


        box_goal_dist = 0
        for i, j in zip(row_ind, col_ind):
            cost = cost_matrix[i][j]
            if cost == float('inf'):
                 # If any assignment in the optimal matching has infinite cost,
                 # it means a box cannot reach its assigned goal.
                 return float('inf')
            box_goal_dist += cost

        # 10. Calculate robot distance
        min_robot_dist = float('inf')

        # Need to find a clear location adjacent to ANY box that needs to move
        # The boxes that need to move are those in 'boxes_needing_goal'
        for box_name in boxes_needing_goal:
            box_loc = current_box_locs.get(box_name)
            # Ensure box is in state and its location is known
            if box_loc is not None and box_loc in self.all_locations:
                # Find locations adjacent to box_loc
                adjacent_to_box = self.graph.get(box_loc, set())
                for adj_loc in adjacent_to_box:
                    # Check if the adjacent location is clear AND is a known location
                    if f'(clear {adj_loc})' in state and adj_loc in self.all_locations:
                        # Calculate distance from robot to this clear adjacent location
                        dist = self.distances.get((robot_loc, adj_loc), float('inf'))
                        min_robot_dist = min(min_robot_dist, dist)

        # If robot cannot reach any clear location adjacent to any box that needs moving
        if min_robot_dist == float('inf'):
             # This state might be a dead end for moving boxes
             return float('inf')

        # 11. Return total heuristic value
        return box_goal_dist + min_robot_dist
