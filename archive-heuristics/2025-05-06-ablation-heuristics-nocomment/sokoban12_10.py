from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque
import math # For float('inf')

# Helper function to parse PDDL fact strings
def get_parts(fact):
    """Splits a PDDL fact string into its predicate and arguments."""
    # Remove surrounding parentheses and split by space
    return fact[1:-1].split()

# Helper function to parse location names like 'loc_R_C'
def parse_location(loc_str):
    """Parses a location string 'loc_R_C' into (row, column) integers."""
    try:
        parts = loc_str.split('_')
        # Ensure there are enough parts and the last two are digits
        if len(parts) == 3 and parts[0] == 'loc' and parts[1].isdigit() and parts[2].isdigit():
             return int(parts[1]), int(parts[2])
        else:
             # If format is unexpected, return None
             return None, None
    except (ValueError, IndexError):
        # If parsing fails, return None
        return None, None

# Helper function to format location names
def format_location(r, c):
     """Formats row and column integers back into a location string 'loc_R_C'."""
     return f'loc_{r}_{c}'

class sokobanHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Sokoban domain.

    Summary:
        This heuristic estimates the cost to reach the goal state by combining
        two components:
        1. The sum of Manhattan distances for each box from its current location
           to its goal location. This estimates the minimum number of pushes
           required for all boxes if they could be moved independently.
        2. The shortest path distance for the robot from its current location
           to any valid "pushing spot" for any box that is not yet at its goal.
           A pushing spot is a location adjacent to a box from which the robot
           can push the box towards its goal, and which is currently clear
           (or the robot's current location). The robot's path must avoid
           locations occupied by other boxes and only traverse clear locations.
        The total heuristic value is the sum of these two components. It is
        non-admissible but aims to guide the search towards states where boxes
        are closer to their goals and the robot is well-positioned to make
        the next push.

    Assumptions:
        - The domain uses the 'loc_R_C' naming convention for locations, where
          R and C are integers representing row and column.
        - The 'adjacent' facts define a grid-like connectivity between locations.
        - The goal state specifies the target location for each box using the
          '(at box_name goal_location)' predicate.
        - The heuristic does not perform complex deadlock detection. States
          where boxes are in unresolvable positions might receive a finite
          heuristic value, or infinity if the robot cannot reach a pushing spot.

    Heuristic Initialization:
        The constructor `__init__` performs the following steps:
        1. Stores the goal predicates from the task.
        2. Extracts the target goal location for each box from the goal predicates
           and stores them in `self.goal_locations`.
        3. Builds an adjacency list representation of the grid graph from the
           static 'adjacent' facts. This graph represents all possible movements
           between locations. Both directions of adjacency are added (e.g., if
           A is adjacent to B, B is also added as adjacent to A).
        4. Stores a set of all location names present in the adjacency list.

    Step-By-Step Thinking for Computing Heuristic (`__call__` method):
        1. Check if the current state is a goal state. If all goal predicates
           are satisfied, the heuristic value is 0.
        2. Extract the robot's current location, the current location of each
           relevant box (those with a goal), and the set of locations that
           are currently clear from the input state facts.
        3. Identify locations currently occupied by boxes.
        4. Calculate the sum of Manhattan distances for all boxes that are
           not currently at their respective goal locations. This value
           represents the minimum number of pushes needed for the boxes
           in an ideal scenario. If any location parsing fails, return infinity.
        5. If the sum of Manhattan distances is 0, it means all boxes are
           at their goals, so the heuristic is 0 (this check is redundant
           if step 1 is done, but adds robustness).
        6. If boxes still need to reach their goals (sum > 0):
           a. Identify the set of all "potential pushing spots". A location `L`
              is a potential pushing spot if it is adjacent to a box `B`
              (that needs pushing) at `loc_b`, and pushing `B` from `L`
              towards `loc_b` would result in `B` moving to a location `loc_next`
              that is strictly closer (by Manhattan distance) to `B`'s goal `loc_g`.
           b. Filter the potential pushing spots to find "reachable pushing spots".
              These are spots from the potential set that are currently clear
              or are the robot's current location. The robot can only move into
              clear spots.
           c. If there are no reachable pushing spots for any box that needs
              pushing, it implies the robot cannot currently get into a position
              to make progress. The state is likely unsolvable or requires
              complex unblocking. Return `float('inf')` (or a large value)
              to penalize this state heavily.
           d. Calculate the shortest path distance for the robot from its
              current location to *any* location within the set of reachable
              pushing spots. This is done using a Breadth-First Search (BFS)
              on the grid graph. The BFS must only traverse locations that are
              currently clear (or the robot's starting location) and must avoid
              locations occupied by other boxes (as the robot cannot move into
              these spots).
           e. If the BFS finds no path to any reachable pushing spot, return
              `float('inf')`.
           f. The robot's BFS distance represents the estimated cost for the
              robot to get into position for the *next* push.
           g. The total heuristic value is the sum of the total box Manhattan
              distance and the robot's BFS distance to a pushing spot.
        7. Return the calculated heuristic value.
    """
    def __init__(self, task):
        super().__init__(task)
        self.goals = task.goals
        static_facts = task.static

        # Extract goal locations for each box
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at" and len(args) == 2:
                box, location = args
                self.goal_locations[box] = location
            # Ignore other goal types if any

        # Build adjacency list for the grid graph from static facts
        self.adj_list = {}
        self.all_locations = set()
        for fact in static_facts:
            predicate, *args = get_parts(fact)
            if predicate == "adjacent" and len(args) == 3:
                l1, l2, direction = args[0], args[1], args[2]
                if l1 not in self.adj_list:
                    self.adj_list[l1] = []
                if l2 not in self.adj_list:
                    self.adj_list[l2] = []
                # Add bidirectional edges for movement
                self.adj_list[l1].append(l2)
                self.adj_list[l2].append(l1)
                self.all_locations.add(l1)
                self.all_locations.add(l2)
            # Ignore other static fact types

        # Ensure all locations mentioned in goals are in the adj_list keys
        # This handles cases where a goal location might be isolated (shouldn't happen in valid sokoban)
        for loc in self.goal_locations.values():
             if loc not in self.adj_list:
                  self.adj_list[loc] = []
                  self.all_locations.add(loc)


    def __call__(self, node):
        state = node.state

        # Check if goal is reached
        if self.goals <= state:
             return 0

        # Extract current robot and box locations and clear locations
        robot_loc = None
        box_locations = {} # {box_name: location_name}
        clear_locations = set()
        occupied_by_boxes = set()

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip empty facts if any

            predicate = parts[0]
            if predicate == "at-robot" and len(parts) == 2:
                robot_loc = parts[1]
            elif predicate == "at" and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                if obj in self.goal_locations: # Only track boxes that have a goal
                    box_locations[obj] = loc
                    occupied_by_boxes.add(loc)
            elif predicate == "clear" and len(parts) == 2:
                clear_locations.add(parts[1])

        # Robot must exist in a valid state
        if robot_loc is None:
            return float('inf')

        # --- Heuristic Calculation ---

        total_box_manhattan_distance = 0
        boxes_to_push = [] # List of (box_name, current_loc, goal_loc) for boxes not at goal

        for box, current_loc in box_locations.items():
            goal_loc = self.goal_locations.get(box) # Use .get for safety

            if goal_loc is None:
                 # This box doesn't have a goal specified, ignore it for heuristic
                 continue

            if current_loc == goal_loc:
                continue # This box is already at its goal

            boxes_to_push.append((box, current_loc, goal_loc))

            # Calculate Manhattan distance for the box
            r1, c1 = parse_location(current_loc)
            r2, c2 = parse_location(goal_loc)

            # Handle potential parsing errors
            if r1 is None or c1 is None or r2 is None or c2 is None:
                 # If location parsing failed, we cannot compute distance.
                 # This state might be problematic or unsolvable. Return infinity.
                 return float('inf')

            box_manhattan_dist = abs(r1 - r2) + abs(c1 - c2)
            total_box_manhattan_distance += box_manhattan_dist

        # If all relevant boxes are at their goals, heuristic is 0 (already checked at start)
        # If total_box_manhattan_distance is 0, it means boxes are at goals.

        # If boxes need pushing, calculate robot cost
        robot_bfs_dist_to_push_spot = float('inf')

        if total_box_manhattan_distance > 0:
            all_potential_pushing_spots = set()

            for box, current_loc, goal_loc in boxes_to_push:
                br, bc = parse_location(current_loc)
                gr, gc = parse_location(goal_loc)

                # Iterate through neighbors of the box location to find potential pushing spots
                if current_loc in self.adj_list:
                    for neighbor_loc in self.adj_list[current_loc]:
                        nr, nc = parse_location(neighbor_loc)

                        # Check if neighbor_loc parsing was successful
                        if nr is None or nc is None:
                             continue # Skip if neighbor location is malformed

                        # Calculate the location the box would move to if pushed from neighbor_loc
                        # The direction of push is from neighbor_loc towards current_loc
                        # The box moves from current_loc in that same direction
                        # Vector from neighbor to current: (br-nr, bc-nc)
                        # Box moves from (br, bc) by this vector
                        next_r = br + (br - nr)
                        next_c = bc + (bc - nc)
                        next_loc_str = format_location(next_r, next_c)

                        # Check if next_loc is a valid location in the grid
                        if next_loc_str in self.all_locations:
                             # Check if moving to next_loc reduces Manhattan distance to goal
                             next_dist_to_goal = abs(next_r - gr) + abs(next_c - gc)
                             current_dist_to_goal = abs(br - gr) + abs(bc - gc)

                             # Use a small epsilon for floating point comparison if needed, but distances are integers here
                             if next_dist_to_goal < current_dist_to_goal:
                                 # This neighbor_loc is a valid spot to push from to reduce distance
                                 all_potential_pushing_spots.add(neighbor_loc)

            # Filter potential pushing spots: Robot must be able to reach them.
            # Robot can only move into clear locations or its current location.
            # A pushing spot must be clear for the robot to enter it.
            # The location the box moves *to* must also be clear for the push action to be applicable.
            # The heuristic focuses on getting the robot into position, so we only need the pushing spot (neighbor_loc) to be reachable by the robot.
            # The BFS will traverse clear locations + robot_loc.
            # The target pushing spot must be one of these traversable locations.
            robot_traversable_locations = clear_locations | {robot_loc}
            reachable_pushing_spots = all_potential_pushing_spots & robot_traversable_locations

            # If no reachable pushing spot exists for any box that needs pushing
            if not reachable_pushing_spots:
                 # This state is likely a dead end or requires complex setup
                 return float('inf')

            # Perform BFS from robot_loc to find the minimum distance to any reachable pushing spot
            # Obstacles for robot movement are locations occupied by *other* boxes.
            # The robot can move from its current location even if it's occupied by a box (e.g., after a push).
            # The robot cannot move *into* a location occupied by a box.
            bfs_obstacles = occupied_by_boxes - {robot_loc}

            queue = deque([(robot_loc, 0)])
            visited = {robot_loc}

            while queue:
                current_r_loc, dist = queue.popleft()

                # Check if we reached a target pushing spot
                if current_r_loc in reachable_pushing_spots:
                    robot_bfs_dist_to_push_spot = dist
                    break # Found shortest path to *a* pushing spot

                # Get neighbors from the pre-computed adjacency list
                if current_r_loc not in self.adj_list:
                     # Should not happen if all_locations and adj_list are built correctly
                     continue

                for neighbor_r_loc in self.adj_list[current_r_loc]:
                    # Robot can only move to traversable locations that are not obstacles
                    # Traversable: clear_locations | {robot_loc}
                    # Obstacles: occupied_by_boxes - {robot_loc}
                    is_traversable = neighbor_r_loc in clear_locations or neighbor_r_loc == robot_loc
                    is_obstacle = neighbor_r_loc in bfs_obstacles

                    if is_traversable and not is_obstacle and neighbor_r_loc not in visited:
                        visited.add(neighbor_r_loc)
                        queue.append((neighbor_r_loc, dist + 1))

            # If BFS finished without finding a path to any reachable pushing spot
            if robot_bfs_dist_to_push_spot == float('inf'):
                 # This case should ideally be caught by the 'if not reachable_pushing_spots' check,
                 # but this provides a fallback if the BFS graph is disconnected.
                 return float('inf')

            # The heuristic is the sum of box distances and the robot distance to get into position
            heuristic_value = total_box_manhattan_distance + robot_bfs_dist_to_push_spot

        else:
             # total_box_manhattan_distance is 0, meaning all boxes are at goals.
             # This case is handled by the initial check `if self.goals <= state: return 0`.
             # If we reach here, it means total_box_manhattan_distance was calculated as 0,
             # but the initial goal check somehow failed. Assuming standard sokoban goals,
             # this branch implies goal reached.
             heuristic_value = 0 # Should not be reached if initial check is correct

        return heuristic_value
