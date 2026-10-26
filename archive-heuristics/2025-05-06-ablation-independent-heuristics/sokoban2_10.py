from heuristics.heuristic_base import Heuristic
from task import Task
from collections import deque
import math # For float('inf')

class sokobanHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Sokoban domain.

    Summary:
    Estimates the cost to reach the goal by summing two components:
    1. Box-Goal Cost: The sum of Manhattan distances for each box to its
       closest goal location. This provides a lower bound on the number
       of push actions required (ignoring robot movement and obstacles).
    2. Robot Positioning Cost: The shortest path distance for the robot
       to reach a location adjacent to a box, from where it can push the
       box towards its closest goal. This estimates the robot movement cost.
       The box chosen is the one currently closest (in Manhattan distance)
       to any of its goal locations.

    Assumptions:
    - Location names are in the format 'loc_row_col' where row and col are integers.
    - The grid structure is defined by 'adjacent' facts in the static information.
    - Goals are specified as '(at boxX loc_Y_Z)' facts.
    - The heuristic is non-admissible and designed for greedy best-first search.
    - The grid defined by 'adjacent' facts is connected for relevant locations.

    Heuristic Initialization:
    - Parses 'adjacent' facts from task.static to build an adjacency map
      representing the grid graph and a set of all valid location names.
    - Parses goal facts from task.goals to identify the target location(s)
      for each box, stored in self.box_goals.

    Step-By-Step Thinking for Computing Heuristic:
    1. Check if the current state is a goal state using task.goal_reached(state).
       If true, return 0.
    2. Extract the robot's current location ('loc_robot') from the state.
    3. Extract the current location of each box ('box_locations' dictionary)
       from the state.
    4. Identify the set of locations currently occupied by boxes ('box_occupied_locations').
    5. Calculate the Box-Goal Cost:
       - Initialize total_box_distance = 0.
       - Initialize variables to track the box closest to its goal:
         min_dist_to_goal_for_any_box = infinity, box_to_push = None,
         loc_b_to_push = None, loc_g_closest_for_push_box = None.
       - Iterate through each box and its current location in box_locations:
         - Check if the box is already at one of its goal locations. If yes, skip it.
         - If not at a goal, find the minimum Manhattan distance from its current
           location to any of its goal locations defined in self.box_goals.
         - Add this minimum distance to total_box_distance.
         - If this minimum distance is less than min_dist_to_goal_for_any_box,
           update the tracking variables for the box_to_push.
       - Box_Goal_Cost = total_box_distance.
    6. Calculate the Robot Positioning Cost:
       - If all boxes are at their goals (box_to_push is None), Robot_Positioning_Cost is 0.
       - Otherwise, use the box_to_push identified in step 5.
       - Find the closest goal location ('loc_g_closest_for_push_box') for box_to_push.
       - Determine all possible required robot locations adjacent to box_to_push
         from where it could be pushed one step closer (reducing Manhattan distance)
         to loc_g_closest_for_push_box. This yields a list 'possible_push_locs'.
       - If possible_push_locs is empty (box cannot be pushed towards its closest goal
         in any distance-reducing direction from here), return infinity (potential deadlock).
       - Calculate the shortest path distance from the robot's current location
         ('loc_robot') to each location in possible_push_locs using BFS.
         The BFS considers a location traversable if it is the robot's start location,
         OR if it is marked as clear in the state AND not occupied by any box.
       - Robot_Positioning_Cost = the minimum of these BFS distances.
       - If the minimum BFS distance is infinity (robot cannot reach any required
         push location), return infinity (likely unsolvable state).
    7. The final heuristic value is Box_Goal_Cost + Robot_Positioning_Cost.
    """

    def __init__(self, task):
        super().__init__(task)
        self.task = task
        self.goals = task.goals

        # Parse adjacent facts to build adjacency map and get all locations
        self.adj = {} # {loc_name: {direction: adjacent_loc_name, ...}, ...}
        self.all_locations = set()
        for fact_str in task.static:
            if fact_str.startswith('(adjacent '):
                parts = fact_str.replace('(', '').replace(')', '').split()
                # Expected format: (adjacent loc1 loc2 direction)
                if len(parts) == 4:
                    _, loc1, loc2, direction = parts
                    if loc1 not in self.adj:
                        self.adj[loc1] = {}
                    self.adj[loc1][direction] = loc2
                    self.all_locations.add(loc1)
                    self.all_locations.add(loc2)

        # Parse goal facts to get goal locations for each box
        self.box_goals = {} # {box_name: [loc_name, ...]}
        for goal_fact in self.goals:
            if goal_fact.startswith('(at box '):
                parts = goal_fact.replace('(', '').replace(')', '').split()
                # Expected format: (at boxName locName)
                if len(parts) == 3:
                    _, box_name, loc_name = parts
                    if box_name not in self.box_goals:
                        self.box_goals[box_name] = []
                    self.box_goals[box_name].append(loc_name)

    def parse_location(self, loc_str):
        """Parses a location string 'loc_row_col' into a (row, col) tuple."""
        # Assumes loc_str is already just the location name, e.g., 'loc_3_5'
        parts = loc_str.split('_')
        if len(parts) == 3 and parts[0] == 'loc':
            try:
                row = int(parts[1])
                col = int(parts[2])
                return (row, col)
            except ValueError:
                pass
        # If it doesn't match the expected format, it's an error for this function
        # print(f"Error: Invalid location string format for parsing: {loc_str}")
        return None # Indicate parsing failure

    def manhattan_distance(self, loc1_str, loc2_str):
        """Calculates Manhattan distance between two location strings."""
        p1 = self.parse_location(loc1_str)
        p2 = self.parse_location(loc2_str)
        if p1 is None or p2 is None:
            return float('inf') # Cannot calculate distance if parsing fails
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def bfs_distance(self, start_loc, target_loc, state):
        """
        Calculates the shortest path distance from start_loc to target_loc
        on the grid defined by self.adj, considering locations occupied by
        boxes as obstacles.
        """
        if start_loc == target_loc:
            return 0

        queue = deque([(start_loc, 0)])
        visited = {start_loc}

        # Identify locations occupied by boxes and clear locations from the current state
        box_occupied_locations = {loc for fact in state if fact.startswith('(at box ')}
        clear_locations = {loc.replace('(clear ', '').replace(')', '') for fact in state if fact.startswith('(clear ')}

        while queue:
            current_loc, dist = queue.popleft()

            neighbors_info = self.adj.get(current_loc, {})
            for direction, neighbor_loc in neighbors_info.items():
                # A location is traversable by the robot if it's the start location,
                # OR it is marked as clear AND not occupied by a box.
                is_neighbor_occupied_by_box = neighbor_loc in box_occupied_locations
                is_neighbor_clear = neighbor_loc in clear_locations

                is_neighbor_traversable = (neighbor_loc == start_loc) or (is_neighbor_clear and not is_neighbor_occupied_by_box)

                if is_neighbor_traversable and neighbor_loc not in visited:
                     if neighbor_loc == target_loc:
                         return dist + 1 # Found target
                     visited.add(neighbor_loc)
                     queue.append((neighbor_loc, dist + 1))

        return float('inf') # Target not reachable

    def get_required_push_locations(self, box_loc_str, goal_loc_str):
        """
        Finds locations adjacent to box_loc_str from where the robot can push
        the box one step towards goal_loc_str (reducing Manhattan distance).
        Returns a list of such required robot locations.
        """
        box_p = self.parse_location(box_loc_str)
        goal_p = self.parse_location(goal_loc_str)

        if box_p is None or goal_p is None:
            return []

        current_dist = self.manhattan_distance(box_loc_str, goal_loc_str)
        possible_push_locs = []

        directions = ['up', 'down', 'left', 'right']
        opposite_dirs = {'up': 'down', 'down': 'up', 'left': 'right', 'right': 'left'}

        # Iterate through all 4 possible push directions
        for push_dir in directions:
            # Check if the box can be pushed in this direction (i.e., is there an adjacent location?)
            if box_loc_str in self.adj and push_dir in self.adj[box_loc_str]:
                loc_target = self.adj[box_loc_str][push_dir]
                target_dist = self.manhattan_distance(loc_target, goal_loc_str)

                # Check if pushing in this direction reduces the distance to the goal
                if target_dist < current_dist:
                    # Find the required robot location to push in this direction
                    # This is the location adjacent to the box in the opposite direction
                    robot_req_loc = self.adj[box_loc_str].get(opposite_dirs[push_dir])

                    # Check if the required robot location exists in the grid
                    if robot_req_loc:
                        possible_push_locs.append(robot_req_loc)

        return possible_push_locs


    def __call__(self, node):
        state = node.state

        # 1. Check if goal is reached
        if self.task.goal_reached(state):
            return 0

        # Extract robot and box locations from the current state
        loc_robot = None
        box_locations = {} # {box_name: loc_name}
        # Identify locations occupied by boxes
        box_occupied_locations = set()

        for fact in state:
            if fact.startswith('(at-robot '):
                loc_robot = fact.replace('(at-robot ', '').replace(')', '')
            elif fact.startswith('(at box '):
                parts = fact.replace('(', '').replace(')', '').split()
                if len(parts) == 3:
                    box_name, loc_name = parts[1], parts[2]
                    box_locations[box_name] = loc_name
                    box_occupied_locations.add(loc_name)

        if loc_robot is None:
             # Robot location not found, should not happen in valid states
             return float('inf')

        # 5. Calculate Box-Goal Cost and find box to push
        total_box_distance = 0
        min_dist_to_goal_for_any_box = float('inf')
        box_to_push = None
        loc_b_to_push = None # Location of the box_to_push
        loc_g_closest_for_push_box = None # Closest goal location for box_to_push

        boxes_not_at_goal = []
        for box_name, loc_b in box_locations.items():
            # Check if this box is already at one of its goal locations
            is_at_goal = False
            if box_name in self.box_goals:
                 for goal_loc in self.box_goals[box_name]:
                     if loc_b == goal_loc:
                         is_at_goal = True
                         break

            if not is_at_goal:
                boxes_not_at_goal.append(box_name)
                min_dist_b_goal = float('inf')
                closest_goal_loc_for_this_box = None

                if box_name in self.box_goals:
                    for goal_loc in self.box_goals[box_name]:
                        dist = self.manhattan_distance(loc_b, goal_loc)
                        if dist < min_dist_b_goal:
                            min_dist_b_goal = dist
                            closest_goal_loc_for_this_box = goal_loc

                # If a box has no goals defined, its distance is inf.
                # This shouldn't happen in valid problems, but handle defensively.
                if min_dist_b_goal == float('inf'):
                    # This box has no goal location defined in the task
                    return float('inf') # Unsolvable

                total_box_distance += min_dist_b_goal

                # Find the box that is currently closest to *any* of its goals
                # This is a simple greedy choice for which box to focus on for robot movement
                if min_dist_b_goal < min_dist_to_goal_for_any_box:
                     min_dist_to_goal_for_any_box = min_dist_b_goal
                     box_to_push = box_name
                     loc_b_to_push = loc_b
                     loc_g_closest_for_push_box = closest_goal_loc_for_this_box


        Box_Goal_Cost = total_box_distance

        # 6. Calculate Robot Positioning Cost
        Robot_Positioning_Cost = 0
        if box_to_push is not None:
            # Get all possible robot locations from where box_to_push can be pushed
            # towards its closest goal (loc_g_closest_for_push_box)
            possible_push_locs = self.get_required_push_locations(loc_b_to_push, loc_g_closest_for_push_box)

            if not possible_push_locs:
                 # The chosen box cannot be pushed towards its closest goal from its current location
                 # in any direction that reduces Manhattan distance. This likely indicates a deadlock
                 # for this box relative to its closest goal.
                 # A more sophisticated heuristic might check if it can be pushed towards *another* goal,
                 # or if another box is pushable. For simplicity, we treat this as a high cost.
                 # Returning infinity helps prune states where the chosen box is stuck.
                 return float('inf')

            # Find the minimum BFS distance from the robot to any of the possible push locations
            min_robot_dist = float('inf')
            for push_loc in possible_push_locs:
                 # BFS needs the current state to know about clear/occupied locations
                 dist = self.bfs_distance(loc_robot, push_loc, state)
                 min_robot_dist = min(min_robot_dist, dist)

            Robot_Positioning_Cost = min_robot_dist

            # If the robot cannot reach any valid push location, state is likely unsolvable
            if Robot_Positioning_Cost == float('inf'):
                 return float('inf')


        # 7. Total heuristic value
        # The heuristic is the sum of the total box distance to goals
        # and the robot's effort to get into position for the "most promising" push.
        # This is a non-admissible estimate.
        return Box_Goal_Cost + Robot_Positioning_Cost

