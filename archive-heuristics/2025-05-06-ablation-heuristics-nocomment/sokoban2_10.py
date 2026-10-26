# Assuming Heuristic base class is available in the execution environment
# from heuristics.heuristic_base import Heuristic

from collections import deque

def get_parts(fact):
    """Helper to split a PDDL fact string into predicate and arguments."""
    # Remove parentheses and split by space
    return fact[1:-1].split()

class sokobanHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Sokoban domain.

    Summary:
        This heuristic estimates the cost to reach the goal by summing two components
        for each box that is not yet at its goal location:
        1. The shortest path distance (number of pushes) required to move the box
           from its current location to its goal location, ignoring obstacles other
           than the grid structure itself.
        2. The shortest path distance the robot needs to travel from its current
           location to a position directly behind the box, ready to perform the
           first push towards the box's goal location.

        The total heuristic value is the sum of these costs for all misplaced boxes.
        This heuristic is non-admissible as it sums costs independently for each box
        and simplifies the robot's movement cost.

    Assumptions:
        - The grid structure and connectivity are fully defined by the 'adjacent'
          static facts.
        - Location names are unique strings.
        - Goal states are defined by '(at boxX locY)' facts for some objects,
          and these objects are the boxes that need to be moved.
        - The grid is structured such that shortest paths and adjacent locations
          can be determined from the 'adjacent' facts.
        - All locations involved in the problem (initial robot/box locations,
          goal locations, and intermediate locations on shortest paths) are
          reachable from each other within the defined adjacency graph. Unreachable
          goals or push positions will result in a very large heuristic value,
          guiding the search away from likely unsolvable states.

    Heuristic Initialization:
        In the constructor (__init__), the heuristic precomputes the following:
        - `box_goals`: A dictionary mapping each box object (string) to its target
          goal location (string) extracted from the task's goal state. Objects
          appearing in '(at <obj> <loc>)' goal facts are assumed to be the boxes.
        - `adj_info`: An adjacency list represented as a dictionary. Keys are
          location strings, and values are lists of tuples `(neighbor_location, direction)`,
          derived from the 'adjacent' static facts. This represents the directed
          connections defined by 'adjacent'.
        - `rev_adj_info`: A reverse adjacency list, mapping a location `l2` to a list
          of tuples `(l1, direction)` where `l1` is adjacent to `l2` in the given direction.
          This helps find the location 'behind' a box (`l1`) given the box's location (`l2`)
          and the direction of the push (`direction`).
        - `opp_dir`: A dictionary mapping each direction string to its opposite.
        - `distances`: A dictionary storing the shortest path distance between any
          pair of locations in the grid. This is computed using an All-Pairs Shortest
          Path algorithm (specifically, running BFS from every location) on the
          *undirected* graph derived from `adj_info` (treating all adjacent connections
          as bidirectional for distance calculation).

    Step-By-Step Thinking for Computing Heuristic:
        In the __call__ method, for a given state:
        1. Identify the current location of the robot (`L_r`) and each box (`L_b`)
           by examining the state facts.
        2. Initialize the total heuristic value `h` to 0.
        3. Iterate through each box and its goal location (`G_b`) stored in `self.box_goals`.
        4. For a box `b` currently at `L_b`:
           a. If `L_b` is the same as `G_b`, the box is already at its goal, so it
              contributes 0 to the heuristic for this box. Continue to the next box.
           b. If `L_b` is different from `G_b`:
              i. Calculate the shortest path distance between `L_b` and `G_b` using
                 the precomputed `self.distances`. Let this be `dist_box_goal`.
                 Add `dist_box_goal` to `h`. This represents the minimum number of
                 pushes required for the box itself along a shortest path in the grid.
                 If `G_b` is unreachable from `L_b`, `dist_box_goal` will be infinite,
                 and the heuristic will reflect this, indicating a likely unsolvable state.
              ii. Determine the location `L_next` which is a neighbor of `L_b` and
                  lies on a shortest path from `L_b` to `G_b`. This is a neighbor
                  `L_next` such that `dist(L_next, G_b) == dist(L_b, G_b) - 1`.
                  Find the direction `push_dir` from `L_b` to `L_next` using `self.adj_info`.
                  If `dist_box_goal > 0` but no such `L_next` is found, it indicates
                  an inconsistency or unreachable goal, and the heuristic returns infinity.
              iii. Find the opposite direction `approach_dir` using `self.opp_dir`.
              iv. Find the location `L_r_push` such that `L_r_push` is adjacent to `L_b`
                  in the `approach_dir`. This is the location the robot must occupy
                  to push the box from `L_b` towards `L_next`. This is done using
                  `self.rev_adj_info`.
              v. Calculate the shortest path distance between the robot's current
                 location `L_r` and `L_r_push` using `self.distances`. Let this be
                 `dist_robot_approach`. Add `dist_robot_approach` to `h`. This
                 represents the minimum robot movement needed to initiate the
                 first push towards the goal for this box. If `L_r_push` is
                 unreachable from `L_r`, `dist_robot_approach` will be infinite,
                 indicating a likely unsolvable state. If `L_r_push` cannot be
                 found (e.g., box is against a wall where robot cannot get behind),
                 and `dist_box_goal > 0`, it also indicates a likely unsolvable
                 state (deadlock or requires complex maneuvers), and the heuristic
                 returns infinity.
        5. Return the total accumulated heuristic value `h`.
    """
    def __init__(self, task):
        # The Heuristic base class constructor might do something,
        # but the provided example heuristics don't call super().__init__.
        # Let's follow the example structure.
        self.goals = task.goals
        static_facts = task.static

        self.box_goals = {}
        # Assuming goals of the form (at <obj> <loc>) where <obj> is a box
        # We identify objects in '(at ...)' goal facts as the boxes we need to move.
        for goal in self.goals:
            parts = get_parts(goal)
            # Check if the goal fact is of the form (at <obj> <location>)
            if parts[0] == 'at' and len(parts) == 3:
                 obj_name = parts[1]
                 goal_location = parts[2]
                 self.box_goals[obj_name] = goal_location

        self.adj_info = {}
        self.rev_adj_info = {}
        location_names = set()

        # Build adjacency information including directions
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'adjacent' and len(parts) == 4:
                loc1, loc2, direction = parts[1], parts[2], parts[3]
                location_names.add(loc1)
                location_names.add(loc2)

                if loc1 not in self.adj_info:
                    self.adj_info[loc1] = []
                self.adj_info[loc1].append((loc2, direction))

                if loc2 not in self.rev_adj_info:
                    self.rev_adj_info[loc2] = []
                # Store the location loc1 and the direction from loc1 to loc2
                # when looking up loc1 from loc2. This is correct for finding
                # the location *behind* loc2 relative to the direction dir.
                self.rev_adj_info[loc2].append((loc1, direction))


        self.location_names = list(location_names) # Convert to list for consistent iteration order

        self.opp_dir = {'up': 'down', 'down': 'up', 'left': 'right', 'right': 'left'}

        # Compute all-pairs shortest paths using BFS
        self.distances = {}
        # Create a simple adjacency list for BFS (undirected graph for distance)
        # We use the union of adj_info and rev_adj_info connections to build the undirected graph
        simple_adj_list = {}

        # Add all locations to the simple_adj_list keys initially
        for loc in self.location_names:
             simple_adj_list[loc] = []

        # Populate the simple_adj_list based on adj_info (implicitly adding reverse edges)
        for loc1, neighbors in self.adj_info.items():
             for loc2, _ in neighbors:
                 # Add connection loc1 -> loc2 if not already present
                 if loc2 not in simple_adj_list[loc1]:
                     simple_adj_list[loc1].append(loc2)
                 # Add reverse connection loc2 -> loc1 if not already present
                 if loc1 not in simple_adj_list[loc2]:
                     simple_adj_list[loc2].append(loc1)


        for start_loc in self.location_names:
            # Run BFS from start_loc
            q = deque([(start_loc, 0)])
            visited = {start_loc}
            loc_distances = {start_loc: 0}

            while q:
                current_loc, dist = q.popleft()

                # Check if current_loc exists in the simple_adj_list (should always be true now)
                if current_loc not in simple_adj_list:
                     continue # Should not happen with pre-population

                for neighbor in simple_adj_list[current_loc]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        loc_distances[neighbor] = dist + 1
                        q.append((neighbor, dist + 1))

            # Store distances from start_loc to all reachable locations
            for end_loc in self.location_names:
                 # If end_loc was not visited, it's unreachable from start_loc. Store inf distance.
                 self.distances[(start_loc, end_loc)] = loc_distances.get(end_loc, float('inf'))


    def __call__(self, node):
        state = node.state

        # Find current robot and box locations
        robot_location = None
        current_box_locations = {} # box_name -> location

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at-robot' and len(parts) == 2:
                robot_location = parts[1]
            elif parts[0] == 'at' and len(parts) == 3: # Check for (at <obj> <loc>)
                 obj_name = parts[1]
                 location = parts[2]
                 # If this object is one of the boxes we care about (i.e., in our goals)
                 if obj_name in self.box_goals:
                    current_box_locations[obj_name] = location

        # If robot location is not found, the state is likely invalid or terminal
        if robot_location is None:
             return float('inf')

        total_heuristic = 0

        # Calculate heuristic for each misplaced box
        for box, goal_location in self.box_goals.items():
            current_location = current_box_locations.get(box) # Get box's current location

            # If box location is not found in the state, it's an issue (e.g., box not present)
            if current_location is None:
                 # This shouldn't happen in valid Sokoban states, but handle defensively
                 return float('inf')

            if current_location != goal_location:
                # Box is misplaced

                # 1. Box-Goal Distance (minimum pushes)
                dist_box_goal = self.distances.get((current_location, goal_location), float('inf'))
                if dist_box_goal == float('inf'):
                    # Box goal is unreachable from its current location
                    return float('inf') # Problem is likely unsolvable from here

                total_heuristic += dist_box_goal

                # 2. Robot-Approach Distance for the first push
                # Find a neighbor L_next of current_location that is one step closer to goal_location
                L_next = None
                push_dir = None

                # Check if current_location is in adj_info (should be if it's a valid location)
                if current_location in self.adj_info:
                    for neighbor, direction in self.adj_info[current_location]:
                        # Check if neighbor is in distances keys (reachable from current_location)
                        # and if it's one step closer to the goal
                        if (neighbor, goal_location) in self.distances and \
                           self.distances[(neighbor, goal_location)] == dist_box_goal - 1:
                            L_next = neighbor
                            push_dir = direction
                            break # Found a suitable next step

                # If dist_box_goal > 0 but no L_next found, it implies an inconsistency
                # in the graph or distance calculation, or a state where the box
                # cannot move towards the goal along a shortest path (e.g., blocked).
                # Given BFS guarantees finding such a neighbor if one exists and is reachable,
                # not finding one means either dist_box_goal was 0 (already at goal, handled)
                # or the goal is unreachable (handled by inf dist_box_goal), or there's
                # a bug. Assuming the distance calculation is correct, if dist_box_goal > 0,
                # L_next must be found. If it's not, it's an error state or unreachable goal.
                # We already return inf if dist_box_goal is inf. If dist_box_goal is finite > 0
                # and L_next is None:
                if dist_box_goal > 0 and L_next is None:
                     # This indicates a state from which the box cannot move towards the goal
                     # along a shortest path according to the grid graph. Likely unsolvable.
                     return float('inf')


                # Find the location L_r_push behind current_location relative to push_dir
                L_r_push = None
                # Only proceed if L_next and push_dir were found (which they should be if dist_box_goal > 0)
                if push_dir is not None:
                    approach_dir = self.opp_dir.get(push_dir)
                    # Check if current_location is in rev_adj_info (should be if it's a valid location)
                    if approach_dir is not None and current_location in self.rev_adj_info:
                        for potential_r_loc, direction in self.rev_adj_info[current_location]:
                            # We are looking for a location potential_r_loc such that
                            # (adjacent potential_r_loc current_location approach_dir) is true.
                            # The rev_adj_info[current_location] stores (loc1, dir) where
                            # (adjacent loc1 current_location dir) is true.
                            # So we need to find (potential_r_loc, approach_dir) in rev_adj_info[current_location].
                            if direction == approach_dir:
                                L_r_push = potential_r_loc
                                break # Found the location behind the box

                # If L_r_push is found, add robot travel distance
                if L_r_push is not None:
                    dist_robot_approach = self.distances.get((robot_location, L_r_push), float('inf'))
                    if dist_robot_approach == float('inf'):
                         # Robot cannot reach the required push position
                         return float('inf') # Problem likely unsolvable

                    total_heuristic += dist_robot_approach
                else:
                    # If dist_box_goal > 0 but we couldn't find L_r_push, it means there's
                    # no location adjacent to the box in the opposite direction of the
                    # intended push. This implies the box is against a wall or edge
                    # where the robot cannot get behind it to push in that direction.
                    # This is a strong indicator of a deadlock or a state requiring
                    # complex maneuvers not captured by this simple heuristic.
                    # Return infinity to prune this path.
                    if dist_box_goal > 0:
                         return float('inf')


        return total_heuristic

# Note: The Heuristic base class is assumed to be provided by the environment.
# The import 'from heuristics.heuristic_base import Heuristic' is commented out
# as per the instruction to provide only the Python code of the heuristic.
# The class definition 'class sokobanHeuristic(Heuristic):' assumes 'Heuristic'
# is a known base class in the execution environment.
