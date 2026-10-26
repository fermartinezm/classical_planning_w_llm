import collections
import logging
from heuristics.heuristic_base import Heuristic
from task import Operator, Task

# Helper function to parse PDDL fact strings
def parse_fact(fact_string):
    """Parses a PDDL fact string like '(predicate arg1 arg2)'."""
    # Remove parentheses and split by space
    parts = fact_string[1:-1].split()
    if not parts: # Handle empty string case
        return None, []
    predicate = parts[0]
    args = parts[1:]
    return predicate, args

# Helper function for Breadth-First Search
def bfs(start_node, get_neighbors_func, target_nodes=None):
    """
    Performs BFS to find shortest distance from start_node to any node in target_nodes.
    Returns distance or float('inf') if no target is reachable.
    If target_nodes is None, returns distances to all reachable nodes (not used in this heuristic).
    """
    # Check if start_node is valid (exists in the graph structure used by get_neighbors_func)
    # A simple check is if get_neighbors_func returns a list for it.
    if not get_neighbors_func(start_node):
         # If start_node has no neighbors in the relevant graph, it's isolated or invalid.
         # This might not be strictly correct if a node exists but has no outgoing edges,
         # but for grid-like structures, it's a reasonable proxy.
         # A more robust check would be against a set of all valid nodes.
         # Let's assume start_node is valid if it's in the adj_list keys used by the lambdas.
         # The lambdas handle missing keys by returning [].
         pass # Proceed assuming start_node might be valid but isolated

    queue = collections.deque([(start_node, 0)])
    visited = {start_node}

    while queue:
        current_node, dist = queue.popleft()

        if target_nodes is not None and current_node in target_nodes:
            return dist # Found a target

        # Get neighbors using the provided function (which handles state/grid)
        for neighbor in get_neighbors_func(current_node):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))

    return float('inf') # No target found

class sokobanHeuristic(Heuristic):
    """
    Domain-dependent heuristic for Sokoban.

    Summary:
    Estimates the cost to reach the goal by summing two components:
    1.  The sum of minimum grid distances for each box to reach any goal location.
    2.  The robot's shortest path distance to reach a clear location adjacent
        to the box that is currently closest to any goal, from which a push
        towards a clear location is possible.

    Assumptions:
    - The grid structure is defined by 'adjacent' facts in the static information.
    - Locations are named 'loc_row_col' (though parsing to (row, col) is not strictly used by BFS).
    - The heuristic is non-admissible and designed for greedy best-first search.

    Heuristic Initialization:
    - Parses all location names from the task facts (initial state, goals, static).
    - Builds an adjacency list representation of the grid graph based on 'adjacent' static facts.
    - Builds a directed adjacency map (location -> direction -> neighbor) for easier push logic.
    - Stores the mapping between directions and their opposites.
    - Identifies all box names and goal locations for boxes from the task definition.

    Step-By-Step Thinking for Computing Heuristic:
    1.  Check if the current state is a goal state using `self.task.goal_reached(state)`. If yes, return 0.
    2.  Parse the current state to find the robot's location, each box's location, and all clear locations.
    3.  Identify boxes that are not currently at any goal location.
    4.  If all boxes are at goals, the heuristic is 0 (already handled in step 1).
    5.  Calculate the 'box_component':
        - Initialize `h_boxes = 0`.
        - Initialize `min_dist_to_any_goal_for_any_box = float('inf')` and variables to track the target box (`target_box_name`, `target_box_loc`).
        - Define a neighbor function `grid_bfs_neighbors` that uses the full grid adjacency list (`self.adj_list`).
        - For each box `b` at location `b_loc` that is not at a goal:
            - Compute the shortest path distance (using BFS with `grid_bfs_neighbors`) from `b_loc` to the *set* of `self.goal_locations`. This BFS returns the minimum distance to *any* goal.
            - If the distance is infinity, the box cannot reach any goal, so the state is unsolvable. Return `float('inf')`.
            - Add this minimum distance to `h_boxes`.
            - If this minimum distance is less than `min_dist_to_any_goal_for_any_box`, update `min_dist_to_any_goal_for_any_box` and set `target_box_name` and `target_box_loc` to this box's details. The target box is the one closest to *any* goal.
    6.  Calculate the 'robot_component':
        - Initialize `h_robot = 0`.
        - If a `target_box_loc` was identified (meaning there's at least one box not at a goal):
            - Find potential robot target locations: These are clear locations `rloc` adjacent to `target_box_loc` such that there exists a location `floc` adjacent to `target_box_loc` (in the push direction opposite `rloc`) and `floc` is also clear. This ensures a push is possible from `rloc`.
            - Iterate through directions (`push_dir`) from `target_box_loc` to a neighbor `floc` using `self.location_adj_by_dir`.
            - If `floc` is clear, find the required robot location `rloc` adjacent to `target_box_loc` in the `reverse_direction` of `push_dir`.
            - If `rloc` exists and is clear, add `rloc` to the set of `potential_robot_targets`.
            - If `potential_robot_targets` is empty, the robot cannot get into a position to push the target box towards any clear spot. Set `h_robot` to `float('inf')`.
            - Otherwise, define a neighbor function `robot_bfs_neighbors` that uses the grid adjacency list (`self.adj_list`) but only allows movement to locations present in `clear_locations`.
            - Compute the robot's shortest path distance (using BFS with `robot_bfs_neighbors`) from `robot_loc` to the set of `potential_robot_targets`. This BFS returns the minimum distance to any valid target location. Set `h_robot` to this distance.
    7.  The total heuristic value is the sum of `h_boxes` and `h_robot`. If either component is `float('inf')`, the total is `float('inf')`. Return the total.
    """

    def __init__(self, task):
        super().__init__()
        self.task = task # Store task for goal check

        self.adj_list = {}
        self.location_adj_by_dir = {}
        self.all_locations = set()
        self.box_names = set()
        self.goal_locations = set() # Goal locations for boxes

        self.reverse_direction = {'up': 'down', 'down': 'up', 'left': 'right', 'right': 'left'}

        # Collect all locations and box names from initial state, goals, and static facts
        facts_to_parse = task.initial_state | task.goals | task.static

        for fact_string in facts_to_parse:
            predicate, args = parse_fact(fact_string)
            for arg in args:
                 if isinstance(arg, str) and arg.startswith('loc_'):
                     self.all_locations.add(arg)

        # Collect box names from initial state and goals
        for fact_string in task.initial_state | task.goals:
             predicate, args = parse_fact(fact_string)
             if predicate == 'at' and len(args) == 2:
                 box_name, loc = args
                 # Assume anything that appears as the first arg of 'at' is a box
                 self.box_names.add(box_name)
                 if isinstance(loc, str) and loc.startswith('loc_'): self.all_locations.add(loc)

        # Collect goal locations for boxes
        for goal_fact in task.goals:
             predicate, args = parse_fact(goal_fact)
             if predicate == 'at' and len(args) == 2 and args[0] in self.box_names:
                 goal_loc = args[1]
                 if isinstance(goal_loc, str) and goal_loc.startswith('loc_'):
                     self.goal_locations.add(goal_loc)
                     self.all_locations.add(goal_loc)

        # Build adjacency list and directed adjacency map
        for loc_string in self.all_locations:
            if isinstance(loc_string, str) and loc_string.startswith('loc_'):
                self.adj_list[loc_string] = []
                self.location_adj_by_dir[loc_string] = {}

        # Build adjacency list and directed adjacency map from static facts
        for fact_string in task.static:
            predicate, args = parse_fact(fact_string)
            if predicate == 'adjacent' and len(args) == 3:
                l1, l2, direction = args
                if isinstance(l1, str) and l1.startswith('loc_') and \
                   isinstance(l2, str) and l2.startswith('loc_') and \
                   l1 in self.adj_list and l2 in self.adj_list: # Ensure locations are known
                     self.adj_list[l1].append(l2)
                     self.location_adj_by_dir[l1][direction] = l2

        # Ensure goal_locations only contains valid locations
        self.goal_locations = {loc for loc in self.goal_locations if loc in self.all_locations}

    def __call__(self, node):
        state = node.state

        # 1. Check for goal state
        if self.task.goal_reached(state):
            return 0

        # 2. Parse current state
        robot_loc = None
        box_locations = {} # {box_name: location}
        clear_locations = set()

        for fact_string in state:
            predicate, args = parse_fact(fact_string)
            if predicate == 'at-robot' and len(args) == 1:
                robot_loc = args[0]
            elif predicate == 'at' and len(args) == 2:
                box_name, loc = args
                if box_name in self.box_names: # Only track known boxes
                    box_locations[box_name] = loc
            elif predicate == 'clear' and len(args) == 1:
                clear_locations.add(args[0])

        # Ensure robot_loc is valid and exists in our graph
        if robot_loc is None or robot_loc not in self.all_locations:
             # This state is likely invalid or unreachable
             logging.warning(f"Robot location {robot_loc} not found or invalid.")
             return float('inf')

        # 3. Identify boxes not at goals
        boxes_not_at_goal = {b for b, loc in box_locations.items() if loc not in self.goal_locations}

        # 4. If all boxes are at goals (should be caught by step 1, but double check)
        if not boxes_not_at_goal:
             return 0 # Should be goal state

        # 5. Calculate box_component and find target box for robot
        h_boxes = 0
        min_dist_to_any_goal_for_any_box = float('inf')
        target_box_name = None
        target_box_loc = None

        # BFS function for grid distance (ignores dynamic state)
        grid_bfs_neighbors = lambda loc: self.adj_list.get(loc, [])

        if not self.goal_locations:
             # No goals defined for boxes? Should not happen in valid problem.
             # Treat as unsolvable if boxes need goals.
             logging.warning("No goal locations defined for boxes.")
             return float('inf')

        for box_name in boxes_not_at_goal:
            b_loc = box_locations.get(box_name) # Use .get for safety
            if b_loc is None or b_loc not in self.all_locations:
                 # Box location is unknown or invalid
                 logging.warning(f"Location for box {box_name} ({b_loc}) not found or invalid.")
                 return float('inf')

            # Find min grid distance from this box to any goal
            dist_to_goals = bfs(b_loc, grid_bfs_neighbors, target_nodes=self.goal_locations)

            if dist_to_goals == float('inf'):
                 # This box cannot reach any goal location on the grid
                 return float('inf') # State is unsolvable

            min_dist_to_any_goal_for_this_box = dist_to_goals # BFS returns min dist to any target

            h_boxes += min_dist_to_any_goal_for_this_box

            # Track the box that is closest to any goal (for robot heuristic)
            if min_dist_to_any_goal_for_this_box < min_dist_to_any_goal_for_any_box:
                 min_dist_to_any_goal_for_any_box = min_dist_to_any_goal_for_this_box
                 target_box_name = box_name
                 target_box_loc = b_loc

        # If h_boxes is still 0, it means all boxes are at goals (already handled)
        # If h_boxes is infinity, it means at least one box cannot reach any goal (handled above)

        # 6. Calculate robot_component
        h_robot = 0 # Default if no boxes need moving (should be caught earlier)
        if target_box_loc: # target_box_loc is None if boxes_not_at_goal was empty
            # Find potential robot target locations: clear locations adjacent to the target box
            # such that a push is possible from there.
            potential_robot_targets = set()
            box_loc = target_box_loc

            # Iterate through directions from the box location
            # Check if box_loc exists in the directed adjacency map
            if box_loc in self.location_adj_by_dir:
                for push_dir, floc in self.location_adj_by_dir[box_loc].items():
                    # floc is the location the box would move *into*
                    # Check if floc is a valid location and is clear
                    if floc in self.all_locations and floc in clear_locations:
                        # Find the location rloc where the robot needs to be to push in this direction
                        # rloc is adjacent to box_loc in the reverse direction
                        reverse_dir = self.reverse_direction.get(push_dir)
                        if reverse_dir:
                             # Find the location adjacent to box_loc in the reverse direction
                             rloc = self.location_adj_by_dir.get(box_loc, {}).get(reverse_dir)
                             # Check if rloc is a valid location and is clear
                             if rloc in self.all_locations and rloc in clear_locations:
                                 # rloc is a valid push-ready location for this push_dir
                                 potential_robot_targets.add(rloc)

            if not potential_robot_targets:
                # No location exists where the robot can stand AND push the box into a clear spot.
                h_robot = float('inf')
            else:
                # Calculate robot's shortest path to any potential target location
                # BFS on the grid graph, only traversing clear locations
                def robot_bfs_neighbors(loc):
                    # Robot can move to adjacent locations if they are clear and valid locations
                    return [n for n in self.adj_list.get(loc, []) if n in clear_locations and n in self.all_locations]

                h_robot = bfs(robot_loc, robot_bfs_neighbors, target_nodes=potential_robot_targets)

        # 7. Total Heuristic
        if h_boxes == float('inf') or h_robot == float('inf'):
            return float('inf')
        else:
            return h_boxes + h_robot

