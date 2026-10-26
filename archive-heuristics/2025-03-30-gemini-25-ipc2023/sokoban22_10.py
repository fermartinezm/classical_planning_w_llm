import math
from collections import deque
from fnmatch import fnmatch
# Assuming the planner infrastructure provides Heuristic base class
# and that the task object is passed correctly.
# If running standalone, define a placeholder for Heuristic.
try:
    # Attempt to import the base class from the expected location
    from heuristics.heuristic_base import Heuristic
except ImportError:
    # Define a placeholder if the import fails (e.g., for standalone testing)
    class Heuristic:
        """Placeholder for the Heuristic base class."""
        def __init__(self, task):
            """Initializes the heuristic with the planning task."""
            self.task = task # Store task info if needed by base class or for debugging
        def __call__(self, node):
            """Calculates the heuristic value for a given search node."""
            raise NotImplementedError("Heuristic calculation not implemented.")

# Helper function to parse PDDL facts "(pred obj1 obj2 ...)" -> ["pred", "obj1", "obj2", ...]
def get_parts(fact):
    """
    Extracts the components of a PDDL fact string.
    Removes parentheses and splits the string by spaces.
    Returns an empty list if the fact is malformed.
    """
    if not isinstance(fact, str) or len(fact) < 2 or fact[0] != '(' or fact[-1] != ')':
        return [] # Return empty list for malformed or non-string facts
    return fact[1:-1].split()

class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal state in a Sokoban problem.
    It calculates the sum of the shortest path distances for each box from its
    current location to its target goal location, based on the empty grid topology.
    Additionally, it adds the shortest path distance from the robot's current
    location to the nearest misplaced box. It includes a check for simple
    dead-end states (corners and certain corridor traps) where a box cannot be
    moved towards any goal. The heuristic is designed to be informative for
    greedy best-first search and is not necessarily admissible.

    # Assumptions
    - The cost of moving the robot ('move' action) and pushing a box ('push' action) is 1.
    - Distances are calculated based on the shortest path in the grid topology
      defined by 'adjacent' facts, ignoring dynamic obstacles (robot, other boxes).
      This provides a lower bound on movement/pushing cost in an empty grid.
    - Simple dead ends (like corners or corridors blocked on both ends without
      a way to push out) are detected based on the static layout (adjacency info).
      More complex deadlocks involving multiple boxes are not detected.

    # Heuristic Initialization
    - Parses the goal conditions (`task.goals`) to identify the target location for each box.
    - Builds data structures representing the map layout from static 'adjacent' facts:
        - `self.adj`: An adjacency list for efficient BFS ({loc: {neighbor1, neighbor2}}).
        - `self.location_directions`: Maps locations to their neighbors by direction
          ({loc: {direction: neighbor}}), used for dead-end checks.
    - Computes all-pairs shortest path distances between reachable locations using BFS,
      stored in `self.distances`. Unreachable pairs have infinite distance.
    - Identifies simple "dead end" locations based on the static map geometry (corners,
      inescapable corridors) and stores them in `self.dead_ends`. Goal locations
      are never considered dead ends.

    # Step-By-Step Thinking for Computing Heuristic
    1.  **Parse Current State:** Extract the robot's location (`robot_loc`) and the
        location of each box (`box_locations`) from the `node.state`.
    2.  **Check Goal State:** Verify if all boxes listed in `self.goal_locations` are
        currently at their respective goal positions. If yes, the goal is reached,
        return 0.
    3.  **Check for Dead Ends:** Iterate through each box `b` that has a defined goal.
        If its current location `bloc` is in `self.dead_ends` and `bloc` is not the
        goal location `gloc` for `b`, then this state is considered a dead end.
        Return `math.inf`.
    4.  **Calculate Sum of Box Distances:**
        a. Initialize `sum_box_dist = 0`.
        b. For each box `b` with a goal location `gloc`:
           i. Get its current location `bloc`. If the box is missing from the state,
              return `math.inf` (invalid state).
           ii. If `bloc != gloc`:
               - Retrieve the pre-calculated shortest path distance `dist(bloc, gloc)`
                 from `self.distances`.
               - If `dist == math.inf` (goal statically unreachable), return `math.inf`.
               - Add `dist` to `sum_box_dist`.
    5.  **Calculate Minimum Robot-to-Box Distance:**
        a. Initialize `min_robot_dist = math.inf`.
        b. Initialize `misplaced_boxes_exist = False`.
        c. For each box `b` with goal `gloc`:
           i. Get current location `bloc`.
           ii. If `bloc != gloc`:
               - Set `misplaced_boxes_exist = True`.
               - Retrieve `dist(robot_loc, bloc)` from `self.distances`.
               - If `dist == math.inf` (robot cannot reach box statically), return `math.inf`.
               - Update `min_robot_dist = min(min_robot_dist, dist)`.
    6.  **Combine Costs:**
        a. If `misplaced_boxes_exist` is False (and it's not a goal state, checked earlier),
           it indicates an inconsistency. Return `math.inf`.
        b. If `misplaced_boxes_exist` is True but `min_robot_dist` is still `math.inf`,
           the robot cannot reach any misplaced box. Return `math.inf`.
        c. The final heuristic value is `sum_box_dist + min_robot_dist`.
    7.  **Return Value:** Return the calculated heuristic value, ensuring it's non-negative.
        For non-goal states, the value should be greater than 0.
    """

    def __init__(self, task):
        super().__init__(task) # Initialize base class
        self.goals = task.goals
        static_facts = task.static
        self.locations = set()
        # Adjacency for BFS (undirected edges based on symmetric 'adjacent')
        self.adj = {}
        # Adjacency with directions (potentially directed if 'adjacent' isn't symmetric)
        self.location_directions = {} # {loc: {direction: neighbor}}

        # Parse static facts to build map representation
        for fact in static_facts:
            parts = get_parts(fact)
            # Ensure the fact is an 'adjacent' predicate with correct arity
            if not parts or parts[0] != 'adjacent' or len(parts) != 4:
                continue
            l1, l2, direction = parts[1], parts[2], parts[3]
            self.locations.add(l1)
            self.locations.add(l2)
            # Build adjacency list for BFS (undirected)
            if l1 not in self.adj: self.adj[l1] = set()
            if l2 not in self.adj: self.adj[l2] = set()
            self.adj[l1].add(l2)
            self.adj[l2].add(l1)
            # Store directed adjacency with direction info
            if l1 not in self.location_directions: self.location_directions[l1] = {}
            self.location_directions[l1][direction] = l2

        # Compute all-pairs shortest paths using BFS
        self.distances = {}
        for start_node in self.locations:
            # Initialize distances from start_node to all other locations as infinity
            self.distances[start_node] = {loc: math.inf for loc in self.locations}
            # Distance from start_node to itself is 0
            if start_node in self.locations: # Check if start_node is a valid location
                 self.distances[start_node][start_node] = 0
            else:
                 continue # Skip if start_node isn't in the set of locations

            queue = deque([start_node])
            visited = {start_node}

            while queue:
                current_node = queue.popleft()
                current_dist = self.distances[start_node][current_node]

                # Explore neighbors using the undirected adjacency list self.adj
                for neighbor in self.adj.get(current_node, set()):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        self.distances[start_node][neighbor] = current_dist + 1
                        queue.append(neighbor)

        # Parse goal locations for boxes
        self.goal_locations = {} # {box_name: goal_loc}
        self.goal_location_set = set() # Set of locations that are goals
        for goal in self.goals:
            parts = get_parts(goal)
            # Ensure the goal fact is 'at' with correct arity
            if not parts or parts[0] != 'at' or len(parts) != 3:
                continue
            # Assuming the second argument of 'at' in goal is a box name
            box, loc = parts[1], parts[2]
            self.goal_locations[box] = loc
            self.goal_location_set.add(loc) # Add location to the set of goal squares

        # --- Identify dead end locations ---
        self.dead_ends = set()
        # Define pairs of perpendicular directions for corner checking
        dir_pairs = [('up', 'left'), ('up', 'right'), ('down', 'left'), ('down', 'right')]

        for loc in self.locations:
            # A location cannot be a dead end if it's a goal location for any box
            if loc in self.goal_location_set:
                continue

            loc_dirs = self.location_directions.get(loc, {}) # Directions available *from* loc

            # Check simple corners: No exit in two perpendicular directions
            is_corner = False
            for d1, d2 in dir_pairs:
                if d1 not in loc_dirs and d2 not in loc_dirs:
                    is_corner = True
                    break
            if is_corner:
                self.dead_ends.add(loc)
                continue # Found a corner dead end, no need for further checks on this loc

            # Check corridor traps: Blocked on opposite sides & no way to push out
            # Check horizontal trap (blocked left/right)
            if 'left' not in loc_dirs and 'right' not in loc_dirs:
                 # Check if pushing out vertically is possible (requires space behind)
                 can_push_out_up = ('up' in loc_dirs and
                                    'down' in loc_dirs and # Need location below...
                                    self.location_directions[loc]['down'] in self.location_directions and # ...that has directions defined...
                                    'up' in self.location_directions[self.location_directions[loc]['down']]) # ...and can move back up to 'loc'

                 can_push_out_down = ('down' in loc_dirs and
                                      'up' in loc_dirs and # Need location above...
                                      self.location_directions[loc]['up'] in self.location_directions and # ...that has directions defined...
                                      'down' in self.location_directions[self.location_directions[loc]['up']]) # ...and can move back down to 'loc'

                 if not can_push_out_up and not can_push_out_down:
                     self.dead_ends.add(loc)
                     continue # Found horizontal corridor trap

            # Check vertical trap (blocked up/down)
            if 'up' not in loc_dirs and 'down' not in loc_dirs:
                 # Check if pushing out horizontally is possible (requires space behind)
                 can_push_out_left = ('left' in loc_dirs and
                                      'right' in loc_dirs and # Need location right...
                                      self.location_directions[loc]['right'] in self.location_directions and # ...that has directions defined...
                                      'left' in self.location_directions[self.location_directions[loc]['right']]) # ...and can move back left to 'loc'

                 can_push_out_right = ('right' in loc_dirs and
                                       'left' in loc_dirs and # Need location left...
                                       self.location_directions[loc]['left'] in self.location_directions and # ...that has directions defined...
                                       'right' in self.location_directions[self.location_directions[loc]['left']]) # ...and can move back right to 'loc'

                 if not can_push_out_left and not can_push_out_right:
                     self.dead_ends.add(loc)
                     # Continue to next location (no need for further checks on this one)
        # --- End of dead end identification ---


    def __call__(self, node):
        """Calculates the heuristic value for the given search node."""
        state = node.state
        robot_loc = None
        box_locations = {} # {box_name: location}

        # Parse current state to find robot and box locations
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            if parts[0] == 'at-robot' and len(parts) == 2:
                robot_loc = parts[1]
            elif parts[0] == 'at' and len(parts) == 3:
                # Check if the object 'at' this location is one of the goal boxes
                if parts[1] in self.goal_locations:
                     box_locations[parts[1]] = parts[2]

        # If robot location is not found in the state, it's invalid
        if robot_loc is None:
             return math.inf

        # Check if the current state satisfies all goal conditions
        is_goal = True
        # Handle case where no goals are defined (technically always a goal state)
        if not self.goal_locations:
             is_goal = True
        else:
            for box, gloc in self.goal_locations.items():
                # If a box required by the goal is not found in the current state,
                # or if it's not at its designated goal location, it's not a goal state.
                if box not in box_locations or box_locations[box] != gloc:
                    is_goal = False
                    break
        # If it's a goal state, heuristic value is 0
        if is_goal:
            return 0

        sum_box_dist = 0
        min_robot_dist_to_misplaced_box = math.inf
        misplaced_boxes_exist = False

        # Iterate through all boxes defined in the goal to check dead ends and calculate distances
        for box, gloc in self.goal_locations.items():
            bloc = box_locations.get(box)
            # If a box required by the goal is not present in the state, return infinity (invalid state)
            if bloc is None:
                 return math.inf

            # Check dead end condition only if the box is not currently at its goal location
            if bloc != gloc:
                # If the box is in a pre-calculated dead end location, return infinity
                if bloc in self.dead_ends:
                    return math.inf

                misplaced_boxes_exist = True
                # Get the shortest path distance for the box to its goal
                dist = self.distances.get(bloc, {}).get(gloc, math.inf)
                # If the goal location is statically unreachable from the box's current location
                if dist == math.inf:
                    return math.inf
                sum_box_dist += dist

                # Calculate the shortest path distance from the robot to this misplaced box
                robot_dist_to_box = self.distances.get(robot_loc, {}).get(bloc, math.inf)
                # If the robot cannot statically reach the box's location
                if robot_dist_to_box == math.inf:
                     return math.inf
                # Update the minimum distance found so far from the robot to any misplaced box
                min_robot_dist_to_misplaced_box = min(min_robot_dist_to_misplaced_box, robot_dist_to_box)

        # If we are in a non-goal state, but no boxes were detected as misplaced
        # (e.g., if goal_locations was empty or goals somehow match state but is_goal was false),
        # this indicates an inconsistency. Return infinity.
        if not misplaced_boxes_exist and not is_goal:
             # This path should ideally not be reached if the goal check logic is sound.
             return math.inf

        # If misplaced boxes exist, but the robot cannot reach any of them (min distance is still infinity)
        if misplaced_boxes_exist and min_robot_dist_to_misplaced_box == math.inf:
             return math.inf

        # Final heuristic value: Sum of box-goal distances + robot distance to nearest misplaced box
        # If no boxes are misplaced, this part is skipped due to the initial goal check returning 0.
        heuristic_value = sum_box_dist + min_robot_dist_to_misplaced_box

        # The heuristic value should be non-negative by construction.
        # For non-goal states, it should be > 0 because either sum_box_dist > 0 or min_robot_dist >= 0
        # (and at least one box is misplaced).
        return heuristic_value
