from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque # For BFS

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to solve a Sokoban puzzle by summing the minimum number of pushes required for each box to reach its goal and the minimum number of robot moves required to reach a position from which it can make the first push for any box that needs moving.

    # Assumptions
    - The grid structure and adjacency are defined by `adjacent` facts.
    - Each box has a specific goal location defined in the task goals.
    - Pathfinding for heuristic calculation ignores dynamic obstacles (other boxes, robot's current position) for efficiency.
    - Assumes solvable puzzles have paths between relevant locations in the static adjacency graph.
    - Assumes box names start with 'box'.

    # Heuristic Initialization
    - Stores the task goals and static facts.
    - Builds an undirected graph from `adjacent` facts for shortest path calculations (robot and box distances ignoring dynamic obstacles).
    - Builds directed and reverse-directed adjacency maps from `adjacent` facts to find required robot push positions.
    - Stores the goal location for each box.

    # Step-By-Step Thinking for Computing Heuristic
    1. Extract the current location of the robot and all boxes from the state.
    2. Check if all boxes are already at their goal locations. If yes, the state is a goal state, return 0.
    3. Initialize `total_box_distance` to 0 and `min_robot_cost_to_help` to infinity.
    4. Initialize a flag `can_push_any_box_towards_goal` to False.
    5. Iterate through each box and its goal location defined in the task goals:
       a. Get the box's current location from the state. If the box is not found or is already at its goal, skip this box.
       b. Calculate the shortest path from the box's current location to its goal location using the pre-computed undirected graph (ignoring dynamic obstacles).
       c. If no path exists, the state is likely unsolvable; return infinity.
       d. The minimum number of pushes required for this box is the length of the shortest path minus 1. Add this value to `total_box_distance`.
       e. If the shortest path has more than one location (i.e., the box is not already at the goal), determine the first step of the path (from `current_loc` to `next_loc`).
       f. Find the required robot position to push the box from `current_loc` to `next_loc`. This position is the location `p` such that `(adjacent p current_loc direction)` where `(adjacent current_loc next_loc direction)`. Use the reverse-directed adjacency map to find `p`.
       g. If such a required robot position `p` exists:
          i. Calculate the shortest path distance from the robot's current location to `p` using the undirected graph (ignoring dynamic obstacles).
          ii. If the push position is reachable by the robot (distance is not infinity), update `min_robot_cost_to_help` with the minimum of its current value and the calculated robot distance. Set `can_push_any_box_towards_goal` to True.
    6. After iterating through all boxes, if the state is not a goal state (checked in step 2) but `can_push_any_box_towards_goal` is still False, it means no reachable push position was found for the first step of any box towards its goal. This indicates an unsolvable state; return infinity.
    7. The heuristic value is `total_box_distance + min_robot_cost_to_help`. This sums the minimum pushes needed for all boxes and the cost for the robot to get into position for the first useful push.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static_facts = task.static

        # Build adjacency graph (undirected for shortest path)
        self.adj_undirected = {}
        # Build directed adjacency map (for finding push directions)
        self.adj_directed = {} # { loc1: { direction: loc2, ... }, ... }
        # Build reverse directed adjacency map (for finding robot push positions)
        # This maps a location and a direction to the location *behind* it in that direction.
        # E.g., if (adjacent loc_A loc_B right), then adj_reverse_directed[loc_B]['right'] = loc_A.
        self.adj_reverse_directed = {} # { loc2: { direction: loc1, ... }, ... }
        self.reverse_dir = {'up': 'down', 'down': 'up', 'left': 'right', 'right': 'left'}

        for fact in self.static_facts:
            parts = get_parts(fact)
            if parts[0] == 'adjacent':
                l1, l2, direction = parts[1], parts[2], parts[3]

                # Undirected graph
                if l1 not in self.adj_undirected: self.adj_undirected[l1] = []
                if l2 not in self.adj_undirected: self.adj_undirected[l2] = []
                self.adj_undirected[l1].append(l2)
                self.adj_undirected[l2].append(l1)

                # Directed graph
                if l1 not in self.adj_directed: self.adj_directed[l1] = {}
                self.adj_directed[l1][direction] = l2

                # Reverse directed graph
                if l2 not in self.adj_reverse_directed: self.adj_reverse_directed[l2] = {}
                self.adj_reverse_directed[l2][direction] = l1

        # Remove duplicates from undirected graph
        for loc in self.adj_undirected:
            self.adj_undirected[loc] = list(set(self.adj_undirected[loc]))

        # Store goal locations for boxes
        self.goal_locations = {}
        for goal in self.goals:
            parts = get_parts(goal)
            # Assuming goal predicates are (at box_name location_name)
            if parts[0] == 'at' and len(parts) == 3 and parts[1].startswith('box'):
                box, location = parts[1], parts[2]
                self.goal_locations[box] = location

    def shortest_path(self, start_loc, end_loc):
        """Find the shortest path (list of locations) between two locations using BFS on the undirected graph."""
        if start_loc == end_loc:
            return [start_loc]
        queue = deque([(start_loc, [start_loc])])
        visited = {start_loc}
        while queue:
            current_loc, path = queue.popleft()
            if current_loc == end_loc:
                return path
            if current_loc in self.adj_undirected: # Check if location exists in graph
                for neighbor in self.adj_undirected[current_loc]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, path + [neighbor]))
        return None # No path found

    def shortest_path_dist(self, start_loc, end_loc):
        """Find the shortest path distance between two locations using BFS on the undirected graph."""
        if start_loc == end_loc:
            return 0
        queue = deque([(start_loc, 0)])
        visited = {start_loc}
        while queue:
            current_loc, dist = queue.popleft()
            if current_loc == end_loc:
                return dist
            if current_loc in self.adj_undirected: # Check if location exists in graph
                for neighbor in self.adj_undirected[current_loc]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, dist + 1))
        return float('inf') # No path found

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Find robot location
        robot_loc = None
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at-robot':
                robot_loc = parts[1]
                break
        # If robot location is not found, something is wrong with the state representation.
        # Return infinity as it's likely an invalid or unsolvable state.
        if robot_loc is None:
            return float('inf')

        # Find box locations
        box_locations = {}
        for fact in state:
            parts = get_parts(fact)
            # Assuming box predicates are (at box_name location_name)
            if parts[0] == 'at' and len(parts) == 3 and parts[1].startswith('box'):
                box, loc = parts[1], parts[2]
                box_locations[box] = loc

        # Check if all boxes are at goals
        all_boxes_at_goal = True
        for box, goal_loc in self.goal_locations.items():
            current_loc = box_locations.get(box)
            # If a box is not found in the state, it's an invalid state.
            # If a box is found but not at its goal, goal is not reached.
            if current_loc is None or current_loc != goal_loc:
                all_boxes_at_goal = False
                break

        if all_boxes_at_goal:
            return 0 # Goal state reached

        # Calculate heuristic for non-goal states
        total_box_distance = 0
        min_robot_cost_to_help = float('inf')
        can_push_any_box_towards_goal = False # Flag to check if any box can be pushed towards goal

        for box, goal_loc in self.goal_locations.items():
            current_loc = box_locations.get(box)
            # Skip boxes already at their goal or not found (shouldn't happen if all_boxes_at_goal is False)
            if current_loc is None or current_loc == goal_loc:
                continue

            # Calculate shortest path for the box to its goal
            path = self.shortest_path(current_loc, goal_loc)

            if path is None:
                # Box cannot reach goal location on the static graph. Unsolvable.
                return float('inf')

            # Add box distance (minimum pushes required)
            total_box_distance += len(path) - 1

            # Estimate robot cost for this box's first push
            if len(path) > 1: # Box is not at goal, so path length is at least 2
                first_step_start = path[0] # box_loc
                first_step_end = path[1]   # next_loc

                # Find the direction from first_step_start to first_step_end
                push_direction = None
                if first_step_start in self.adj_directed:
                    for direction, loc in self.adj_directed[first_step_start].items():
                        if loc == first_step_end:
                            push_direction = direction
                            break

                # Find the required robot position (location behind first_step_start in push_direction)
                # This is the location P such that (adjacent P first_step_start push_direction)
                required_robot_pos = None
                if push_direction and first_step_start in self.adj_reverse_directed and push_direction in self.adj_reverse_directed[first_step_start]:
                     required_robot_pos = self.adj_reverse_directed[first_step_start][push_direction]

                if required_robot_pos:
                     # Calculate robot distance to the required push position
                     robot_to_push_pos_dist = self.shortest_path_dist(robot_loc, required_robot_pos)

                     # If the push position is reachable by the robot, update min_robot_cost_to_help
                     if robot_to_push_pos_dist != float('inf'):
                         min_robot_cost_to_help = min(min_robot_cost_to_help, robot_to_push_pos_dist)
                         can_push_any_box_towards_goal = True # Found at least one box that can potentially be pushed

        # If there are boxes not at goals (checked by all_boxes_at_goal flag),
        # but we couldn't find any reachable push position for the first step of any of them,
        # it suggests an unsolvable state.
        if not can_push_any_box_towards_goal:
             # This case is reached if all_boxes_at_goal was False, but the loop
             # did not find any box for which a reachable push position exists
             # towards its goal.
             return float('inf')

        # The heuristic is the sum of minimum pushes for all boxes
        # plus the minimum cost for the robot to get into position
        # for the first push of any box that needs moving.
        return total_box_distance + min_robot_cost_to_help
