from collections import deque
from fnmatch import fnmatch
# Assuming Heuristic base class is available at this path
# from heuristics.heuristic_base import Heuristic

# Define a dummy Heuristic base class if not running within the planner environment
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    class Heuristic:
        def __init__(self, task):
            self.task = task
        def __call__(self, node):
            raise NotImplementedError
        def __str__(self):
            return self.__class__.__name__
        def __repr__(self):
            return f"<{self.__class__.__name__}>"


# Helper functions to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and starts/ends with parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        # Handle unexpected fact format, maybe return empty list or raise error
        # For this context, assuming valid fact strings.
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at obj1 loc1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Check if the number of parts matches the number of pattern arguments
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the number of actions required to move all boxes
    to their goal locations. It sums the shortest path distances for each box
    to its goal (representing minimum pushes) and adds the minimum distance
    for the robot to reach a position from which it can make the first push
    for any box that needs moving.

    # Assumptions
    - The grid structure and connectivity are defined by `adjacent` facts.
    - Distances are shortest paths on this static grid graph.
    - The heuristic ignores dynamic obstacles (other boxes, robot position)
      for box paths and robot paths, except for checking reachability on the
      static graph. The robot's current position is considered for its initial
      movement cost.
    - Each push action moves a box one step towards its goal along a shortest path.
    - The robot needs to be adjacent to the box in the push direction to push it.

    # Heuristic Initialization
    - Build the grid graph from `adjacent` facts, storing neighbors and directions.
    - Determine the opposite direction mapping (e.g., up <-> down).
    - Identify goal locations for each box from the task goals.
    - Precompute shortest path distances and the next step on the path
      from every location towards every goal location using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Extract Relevant Information:
       - Identify the current robot location from the state.
       - Identify the current location of every box from the state.

    2. Check for Goal State:
       - If all boxes are already at their respective goal locations, the heuristic is 0.

    3. Initialize Heuristic Components:
       - Initialize the total heuristic value `h` to 0.
       - Initialize the minimum robot distance to any required push position `min_robot_dist_to_any_push_pos` to infinity.

    4. Compute Robot Reachability:
       - Run a Breadth-First Search (BFS) starting from the robot's current location on the static grid graph. This computes the shortest path distance from the robot to all reachable locations. Store these distances in a map (`robot_dist_map`). This step is performed for each state evaluation because the robot's position changes.

    5. Iterate Through Boxes:
       - For each box defined in the task goals:
         a. Get the box's current location (`l_b`) and its goal location (`g_b`).
         b. If the box is already at its goal (`l_b == g_b`), skip this box.
         c. Calculate the shortest path distance (`box_dist`) for the box from `l_b` to `g_b` using the precomputed distances (`self.dist_to_goal`). If `g_b` is unreachable from `l_b` on the static grid, the state is likely unsolvable; return infinity.
         d. Add `box_dist` to the total heuristic `h`. This `box_dist` represents the minimum number of push actions required for this box, ignoring dynamic obstacles.
         e. If `box_dist` is greater than 0 (meaning the box needs to move):
            i. Find the next location (`l_b_next`) on a shortest path from `l_b` towards `g_b` using the precomputed next steps (`self.next_step_towards_goal`). If this lookup fails (which shouldn't happen if `box_dist > 0` and reachable), return infinity.
            ii. Determine the direction (`dir`) of the push required to move the box from `l_b` to `l_b_next` using `self.dir_map`. If this lookup fails, return infinity.
            iii. Determine the required robot location (`rloc_required`) adjacent to `l_b` in the direction `dir` (based on the PDDL `push` action definition, this is the location such that `(adjacent rloc_required l_b dir)` holds). This is equivalent to the neighbor of `l_b` in the opposite direction of `dir`. Use `self.neighbor_in_dir` and `self.opposite_dir`. If this required location doesn't exist in the grid, return infinity.
            iv. Get the shortest path distance (`robot_dist`) from the robot's current location to `rloc_required` using the `robot_dist_map` computed in step 4. If `rloc_required` is unreachable by the robot on the static grid, return infinity.
            v. Update `min_robot_dist_to_any_push_pos = min(min_robot_dist_to_any_push_pos, robot_dist)`.

    6. Final Heuristic Value:
       - If the total box distance `h` is greater than 0 (meaning at least one box needed moving), add `min_robot_dist_to_any_push_pos` to `h`. This accounts for the robot's initial effort to get into position for the first push of *any* box.
       - Return the calculated heuristic value `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by building the grid graph and precomputing
        distances and next steps towards goal locations.
        """
        self.task = task
        self.locations = set()
        # Undirected graph for distance calculation (adjacency list)
        self.graph = {}
        # Map (location, neighbor_location) -> direction
        self.dir_map = {}
        # Map location -> direction -> neighbor_location
        self.neighbor_in_dir = {}
        # Map direction -> opposite_direction
        self.opposite_dir = {
            'up': 'down', 'down': 'up', 'left': 'right', 'right': 'left'
        }

        # Build the graph and direction maps from static adjacent facts
        for fact in task.static:
            parts = get_parts(fact)
            if parts and parts[0] == 'adjacent' and len(parts) == 4:
                l1, l2, direction = parts[1], parts[2], parts[3]
                self.locations.add(l1)
                self.locations.add(l2)

                self.graph.setdefault(l1, []).append(l2)
                self.graph.setdefault(l2, []).append(l1) # Add reverse edge for undirected graph

                self.dir_map[(l1, l2)] = direction
                # Ensure opposite direction exists before adding reverse mapping
                if direction in self.opposite_dir:
                    self.dir_map[(l2, l1)] = self.opposite_dir[direction]
                else:
                    # Handle unexpected direction, maybe log a warning
                    pass # Or raise an error if directions must be one of the four

                self.neighbor_in_dir.setdefault(l1, {})[direction] = l2
                if direction in self.opposite_dir:
                     self.neighbor_in_dir.setdefault(l2, {})[self.opposite_dir[direction]] = l1


        # Identify goal locations for each box
        self.box_goals = {}
        # We assume goals are (at box_name location_name) facts
        for goal in task.goals:
            parts = get_parts(goal)
            # Check if it's an 'at' predicate with two arguments after the predicate name
            if parts and parts[0] == 'at' and len(parts) == 3:
                box_name, goal_location = parts[1], parts[2]
                self.box_goals[box_name] = goal_location

        # Precompute shortest path distances and next steps towards each goal location
        # dist_to_goal[location][goal_location] = distance
        # next_step_towards_goal[location][goal_location] = next_location_on_path
        self.dist_to_goal = {}
        self.next_step_towards_goal = {}

        # Run BFS from each unique goal location
        unique_goal_locations = set(self.box_goals.values())
        for goal_loc in unique_goal_locations:
            if goal_loc not in self.locations:
                 # Goal location is not in the graph, problem is likely unsolvable
                 # We can represent this by not populating distances for this goal.
                 # The heuristic will return infinity later if a box needs to reach here.
                 continue

            q = deque([goal_loc])
            d = {goal_loc: 0}
            p = {goal_loc: None} # Parent pointer for path reconstruction
            visited = {goal_loc}

            while q:
                curr = q.popleft()

                # Store distance from curr to goal_loc
                self.dist_to_goal.setdefault(curr, {})[goal_loc] = d[curr]

                # Store the next step from curr towards goal_loc (which is its parent in the BFS tree rooted at goal_loc)
                if p[curr] is not None:
                    self.next_step_towards_goal.setdefault(curr, {})[goal_loc] = p[curr]

                # Explore neighbors
                if curr in self.graph: # Ensure curr is in graph (might not be if goal_loc was not)
                    for neighbor in self.graph[curr]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            d[neighbor] = d[curr] + 1
                            p[neighbor] = curr
                            q.append(neighbor)


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state

        # Find robot location
        robot_location = None
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == 'at-robot' and len(parts) == 2:
                robot_location = parts[1]
                break

        if robot_location is None or robot_location not in self.locations:
             # Robot location not found or not in the grid graph
             return float('inf') # Should not happen in valid states

        # Find box locations
        box_locations = {}
        for fact in state:
            parts = get_parts(fact)
            # Check if it's an 'at' predicate with two arguments after the predicate name
            if parts and parts[0] == 'at' and len(parts) == 3 and parts[1] in self.box_goals:
                box_name, loc = parts[1], parts[2]
                box_locations[box_name] = loc

        # Check if all boxes are at goals
        all_goals_reached = True
        for box, goal_loc in self.box_goals.items():
            # If a box is not in box_locations, it's not 'at' any location, which is invalid.
            # We assume valid states have all boxes 'at' some location.
            if box not in box_locations or box_locations[box] != goal_loc:
                all_goals_reached = False
                break

        if all_goals_reached:
            return 0 # Goal state reached

        # --- Heuristic Calculation ---
        total_box_distance = 0
        min_robot_dist_to_any_push_pos = float('inf')

        # Run BFS from current robot location to find distances to all reachable locations
        # This BFS is on the static grid graph, ignoring dynamic obstacles ('clear' predicate)
        # for the robot's path calculation itself, but it starts from the robot's
        # actual current location.
        robot_dist_map = {}
        q = deque([robot_location])
        robot_dist_map[robot_location] = 0
        visited_robot_bfs = {robot_location}

        while q:
            curr = q.popleft()
            if curr in self.graph: # Ensure curr is in graph
                for neighbor in self.graph[curr]:
                    # We calculate robot distance on the static grid, ignoring dynamic obstacles
                    # A more complex heuristic could check 'clear' predicate here, but that
                    # makes the BFS state space larger or requires checking state facts inside BFS.
                    if neighbor not in visited_robot_bfs:
                        visited_robot_bfs.add(neighbor)
                        robot_dist_map[neighbor] = robot_dist_map[curr] + 1
                        q.append(neighbor)


        # Iterate through boxes that need to move
        boxes_to_move = [box for box, goal_loc in self.box_goals.items() if box_locations.get(box) != goal_loc]

        if not boxes_to_move:
             # This case should be covered by all_goals_reached check, but defensive
             return 0

        for box in boxes_to_move:
            l_b = box_locations[box]
            g_b = self.box_goals[box]

            # Get box distance to goal from precomputed table
            if l_b not in self.dist_to_goal or g_b not in self.dist_to_goal[l_b]:
                 # Box goal is unreachable from current box location on static grid
                 return float('inf') # Unsolvable state according to static graph

            box_dist = self.dist_to_goal[l_b][g_b]
            total_box_distance += box_dist

            # Calculate required robot position for the first push towards goal
            if box_dist > 0:
                # Get the next step location for the box towards the goal
                # This lookup should succeed if box_dist > 0 and reachable
                if l_b not in self.next_step_towards_goal or g_b not in self.next_step_towards_goal[l_b]:
                     return float('inf') # Consistency check failed
                l_b_next = self.next_step_towards_goal[l_b][g_b]

                # Determine the direction of the push (l_b -> l_b_next)
                if (l_b, l_b_next) not in self.dir_map:
                     return float('inf') # Consistency check failed
                dir = self.dir_map[(l_b, l_b_next)]

                # Determine the required robot location to push l_b in direction dir
                # Robot must be adjacent to l_b in direction dir according to PDDL (adjacent ?rloc ?bloc ?dir)
                # This means l_b is adjacent to robot in opposite_dir[dir]
                opp_dir = self.opposite_dir.get(dir)
                if opp_dir is None or l_b not in self.neighbor_in_dir or opp_dir not in self.neighbor_in_dir[l_b]:
                     # Required robot position doesn't exist (e.g., pushing from edge where robot can't be)
                     return float('inf') # Box cannot be pushed this way from here

                rloc_required = self.neighbor_in_dir[l_b][opp_dir]

                # Get robot distance to the required push position from the robot BFS
                if rloc_required not in robot_dist_map:
                     # Robot cannot reach the required push position on static grid
                     return float('inf') # Unsolvable state according to static graph

                robot_dist = robot_dist_map[rloc_required]
                min_robot_dist_to_any_push_pos = min(min_robot_dist_to_any_push_pos, robot_dist)

        # Add the minimum robot distance if any box needs moving
        # total_box_distance > 0 check is implicit because we only iterate boxes_to_move
        # and return inf if any step fails. If we reach here and boxes_to_move is not empty,
        # min_robot_dist_to_any_push_pos must be finite.
        if boxes_to_move:
             heuristic_value = total_box_distance + min_robot_dist_to_any_push_pos
        else:
             # Should be covered by all_goals_reached check, but fallback
             heuristic_value = 0


        return heuristic_value

