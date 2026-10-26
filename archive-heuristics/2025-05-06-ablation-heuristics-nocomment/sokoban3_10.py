from heuristics.heuristic_base import Heuristic
import collections
import math
import re

# Helper function to parse PDDL facts
def get_parts(fact):
    """Removes surrounding parentheses and splits by space."""
    # Handle cases like '(at-robot loc_1_1)' -> ['at-robot', 'loc_1_1']
    # Handle cases like '(adjacent loc_1_1 loc_1_2 right)' -> ['adjacent', 'loc_1_1', 'loc_1_2', 'right']
    # Remove leading/trailing parentheses and split by space
    return fact.strip('()').split()

# Helper function to parse location names like 'loc_row_col'
def get_coords(location_name):
    """Parses location name 'loc_row_col' into (row, col) tuple."""
    # Use regex for more robust parsing
    match = re.match(r'loc_(\d+)_(\d+)', location_name)
    if match:
        return (int(match.group(1)), int(match.group(2)))
    # Fallback for simpler formats if necessary, or raise error
    # For this domain, 'loc_r_c' seems standard based on examples.
    raise ValueError(f"Unexpected location format: {location_name}")

# Helper function to calculate Manhattan distance
def md(loc1_name, loc2_name):
    """Calculates Manhattan distance between two locations."""
    r1, c1 = get_coords(loc1_name)
    r2, c2 = get_coords(loc2_name)
    return abs(r1 - r2) + abs(c1 - c2)

# Helper function to get opposite direction
def opposite_dir(direction):
    """Returns the opposite direction."""
    if direction == 'up': return 'down'
    if direction == 'down': return 'up'
    if direction == 'left': return 'right'
    if direction == 'right': return 'left'
    return None # Should not happen in this domain

class sokobanHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Sokoban domain.

    Summary:
    Estimates the cost to reach the goal state by summing the Manhattan
    distances of all off-goal boxes to their respective goal locations
    and adding the shortest path distance for the robot to reach a
    position from which it can push any off-goal box towards its goal.

    Assumptions:
    - Location names follow the 'loc_row_col' format.
    - The 'adjacent' facts define a connected grid-like structure.
    - Every box in the initial state has a corresponding goal location
      specified in the task goals.
    - The PDDL domain correctly uses the 'clear' predicate to indicate
      empty locations.
    - The heuristic is non-admissible and designed for greedy best-first search.
    - For solvable states, there is always a path for the robot to reach
      a position to push an off-goal box towards its goal.

    Heuristic Initialization:
    - Stores the task's goals.
    - Builds an adjacency list representation of the grid graph
      from the 'adjacent' static facts. This graph is used for
      robot pathfinding. The graph is bidirectional.
    - Extracts the goal location for each box from the task goals.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by processing static task information.
        """
        self.task = task
        self.goals = task.goals
        self.adj = collections.defaultdict(list)

        # Build adjacency list from static facts
        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == 'adjacent':
                l1, l2, direction = parts[1], parts[2], parts[3]
                self.adj[l1].append((l2, direction))
                # Add reverse adjacency as well
                self.adj[l2].append((l1, opposite_dir(direction)))

        # Extract box goal locations
        self.box_goals = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == 'at' and len(parts) == 3:
                box, location = parts[1], parts[2]
                self.box_goals[box] = location

    def __call__(self, node):
        """
        Computes the heuristic value for the given state.

        Step-By-Step Thinking for Computing Heuristic:
        1. Check if the current state is a goal state using the task's goal_reached method.
           If yes, the heuristic is 0.
        2. Parse the current state to find the robot's location and the location
           of each box. Also identify locations occupied by boxes.
        3. Calculate the total Manhattan distance for all boxes that are not
           at their goal locations. This is the sum of Manhattan distances
           from each off-goal box's current location to its goal location.
           This represents a lower bound on the number of push actions needed
           if there were no obstacles.
        4. Identify potential push target locations for the robot. A location `push_l`
           is a potential push target if the robot can stand there and push an
           off-goal box `b` at `box_l` towards its goal `goal_l`.
           Specifically, `push_l` must be adjacent to `box_l`, and pushing from
           `push_l` through `box_l` must lead to a location `next_l` that is
           one step closer to `goal_l` in Manhattan distance, and `next_l` must
           be clear in the current state (checked by verifying the '(clear next_l)'
           fact is present).
        5. If no such push target location exists (no off-goal box can be pushed
           towards its goal right now in a way that reduces Manhattan distance
           and has a clear target square), the state might be a dead end or require
           complex setup. Return a large heuristic value (infinity) to discourage
           the search from exploring this path further.
        6. If push targets exist, calculate the shortest path distance from the
           robot's current location to any of these push targets using Breadth-First
           Search (BFS) on the grid graph defined by 'adjacent' facts. The BFS
           treats locations occupied by *any* box as obstacles, preventing
           the robot from moving through them.
        7. The heuristic value is the sum of the total box-to-goal Manhattan
           distance (step 3) and the minimum robot distance to a push position
           (step 6).
        """
        state = node.state

        # 1. Check for goal state
        if self.task.goal_reached(state):
            return 0

        # 2. Parse state
        robot_l = None
        box_locations = {}
        occupied_by_box = set() # Locations occupied by boxes

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at-robot':
                robot_l = parts[1]
            elif parts[0] == 'at' and len(parts) == 3:
                box, location = parts[1], parts[2]
                box_locations[box] = location
                occupied_by_box.add(location)

        # 3. Calculate total box-to-goal Manhattan distance
        box_goal_distance = 0
        off_goal_boxes = []
        for box, current_loc in box_locations.items():
            goal_loc = self.box_goals.get(box) # Get goal for this box
            # Only consider boxes that have a goal and are not there yet
            if goal_loc and current_loc != goal_loc:
                box_goal_distance += md(current_loc, goal_loc)
                off_goal_boxes.append(box)

        # If no boxes are off-goal, but goal_reached was False,
        # it implies other goal conditions exist (not in this domain)
        # or the initial state was a goal state already.
        # Given the domain, this case should ideally not be reached if goal_reached is false.
        # However, if it is reached, and no boxes need moving, the heuristic is 0.
        if not off_goal_boxes:
             return 0 # Should be caught by task.goal_reached(state)

        # 4. Find potential push target locations for the robot
        push_targets = set()
        for box in off_goal_boxes:
            box_l = box_locations[box]
            goal_l = self.box_goals[box]

            current_md_to_goal = md(box_l, goal_l)

            # Iterate through neighbors of the box location (potential push_l)
            for push_l, dir_to_box in self.adj.get(box_l, []):
                 # Find the location next_l if pushed from box_l in dir_to_box
                 # dir_from_box is the direction the box moves
                 dir_from_box = dir_to_box

                 # Find next_l: the location adjacent to box_l in direction dir_from_box
                 next_l = None
                 for neighbor_l, d in self.adj.get(box_l, []):
                     if d == dir_from_box:
                         next_l = neighbor_l
                         break

                 if next_l:
                     # Check if pushing towards goal reduces Manhattan distance
                     next_md_to_goal = md(next_l, goal_l)

                     if next_md_to_goal < current_md_to_goal: # Check if distance decreases
                         # Check if next_l is clear (precondition for push)
                         # A location is clear if the fact '(clear L)' is in the state.
                         if f'(clear {next_l})' in state:
                             push_targets.add(push_l)

        # 5. Handle case where no push is possible towards goal
        if not push_targets:
            # This state might be a dead end or require complex setup moves.
            # Return infinity to indicate a very high cost/unsolvable path.
            return math.inf

        # 6. Calculate minimum robot distance to a push position using BFS
        min_robot_dist_to_push_pos = math.inf

        # Obstacles for robot movement: locations occupied by *any* box
        # The robot cannot move into a square occupied by a box.
        robot_obstacles = occupied_by_box # Use the set collected earlier

        # BFS
        queue = collections.deque([(robot_l, 0)])
        visited = {robot_l}

        while queue:
            curr_l, dist = queue.popleft()

            # If we reached a push target, we found the shortest path to *a* target
            if curr_l in push_targets:
                 min_robot_dist_to_push_pos = dist
                 break # Stop BFS once the first target is reached

            # Explore neighbors
            for neighbor_l, dir in self.adj.get(curr_l, []):
                # Robot cannot move into a location occupied by a box
                if neighbor_l not in robot_obstacles and neighbor_l not in visited:
                    visited.add(neighbor_l)
                    queue.append((neighbor_l, dist + 1))

        # If BFS finished without reaching any target (should not happen in solvable states
        # where push_targets was not empty)
        if min_robot_dist_to_push_pos == math.inf:
             # This means no push target is reachable by the robot.
             # This state is likely unsolvable or requires moving blocking boxes first.
             # Return infinity.
             return math.inf

        # 7. Combine components
        return box_goal_distance + min_robot_dist_to_push_pos
