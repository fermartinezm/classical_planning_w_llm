import sys
from collections import deque

# Helper function to parse PDDL facts
def parse_fact(fact_string):
    """Removes surrounding brackets and splits the fact string."""
    parts = fact_string.strip("()'").split()
    if not parts: # Handle empty string or just "()"
        return None, []
    predicate = parts[0]
    objects = parts[1:]
    return predicate, objects

class sokobanHeuristic:
    """
    Domain-dependent heuristic for the Sokoban planning domain.

    Summary:
        This heuristic estimates the cost to reach the goal state by summing
        two components:
        1. The sum of the shortest path distances for each box from its current
           location to its goal location. This estimates the minimum number
           of pushes required for all boxes, assuming clear paths.
        2. The shortest path distance for the robot from its current location
           to any location that is adjacent to a box that is not yet at its goal.
           This estimates the cost for the robot to get into a position to
           start pushing one of the remaining boxes, considering only movement
           through currently clear cells.

    Assumptions:
        - The PDDL instance defines a grid-like structure using `adjacent`
          predicates.
        - The goal state specifies the target location for each box using `(at box_name goal_location)` facts.
        - There is a one-to-one mapping between boxes in the initial state
          (that need moving) and goal locations specified for boxes.
        - The grid defined by `adjacent` predicates is mostly connected.
        - Location names can be uniquely identified and used as keys in dictionaries.

    Heuristic Initialization:
        The heuristic constructor `__init__` precomputes static information
        from the planning task:
        - It builds an adjacency list representation (`self.adj_list`) of the
          grid graph based on the `adjacent` predicates in the static facts.
        - It identifies all unique location names.
        - It computes the all-pairs shortest path distances between all locations
          on the grid graph (`self.location_distances`) using BFS. This is used
          to estimate the minimum number of pushes required for a box, ignoring
          dynamic obstacles.
        - It extracts the goal location for each box from the task's goal
          conditions and stores them in `self.box_goals`.

    Step-By-Step Thinking for Computing Heuristic:
        The heuristic function `__call__(self, state, task)` computes the
        heuristic value for a given state:
        1. Parse the current `state` to find the robot's location (`robot_loc`),
           the current location of each box (`box_locations`), and the set
           of clear locations (`clear_locations`).
        2. Initialize `total_box_dist` to 0 and create a list `boxes_to_move_locs`
           to store the locations of boxes that are not yet at their goal.
        3. Iterate through the boxes found in the current state. For each box:
           - Get its current location (`current_loc`) and its goal location
             (`goal_loc`) from the precomputed `self.box_goals`.
           - If the box has a goal location and is not at its goal (`current_loc != goal_loc`):
             - Look up the precomputed shortest path distance between
               `current_loc` and `goal_loc` in `self.location_distances`.
             - If the distance is found and is finite, add it to `total_box_dist`.
               This represents the estimated pushes needed for this box.
             - Add `current_loc` to `boxes_to_move_locs`.
           - If a distance lookup fails (e.g., goal location not found or
             unreachable in the precomputed graph), return `float('inf')`
             as the state is likely unsolvable.
        4. If `total_box_dist` is 0, it means all boxes are at their goals,
           so the state is a goal state. Return 0.
        5. If there are boxes to move (`total_box_dist > 0`), calculate the
           robot's contribution to the heuristic:
           - Identify the set of `target_robot_locations`. These are all
             locations that are adjacent to any box location listed in
             `boxes_to_move_locs`.
           - Perform a Breadth-First Search (BFS) starting from the robot's
             current location (`robot_loc`).
           - The BFS explores only locations that are marked as `clear` in the
             current `state`.
           - The BFS aims to find the shortest path to *any* location within
             the `target_robot_locations` set.
           - The distance found by this BFS is `robot_dist`.
        6. If the robot BFS cannot reach any of the `target_robot_locations`
           (e.g., robot is trapped or all adjacent locations to boxes are blocked),
           `robot_dist` remains `float('inf')`. Return `float('inf')` as the
           state is likely unsolvable.
        7. Otherwise, return the sum `total_box_dist + robot_dist`.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by precomputing static information.

        @param task: The planning task object.
        """
        self.adj_list = {}
        locations = set()

        # Build adjacency list from static adjacent facts
        for fact in task.static:
            pred, objs = parse_fact(fact)
            if pred == 'adjacent':
                l1, l2, direction = objs
                locations.add(l1)
                locations.add(l2)
                if l1 not in self.adj_list:
                    self.adj_list[l1] = []
                self.adj_list[l1].append((l2, direction))

        self.location_names = sorted(list(locations))

        # Compute all-pairs shortest paths for locations (for box distance)
        # This graph ignores dynamic facts like 'clear' or 'at'
        self.location_distances = {}
        for start_loc in self.location_names:
            self.location_distances[start_loc] = {}
            queue = deque([(start_loc, 0)])
            visited = {start_loc}
            while queue:
                curr_loc, dist = queue.popleft()
                self.location_distances[start_loc][curr_loc] = dist
                # Traverse using the adjacency list
                for next_loc, _ in self.adj_list.get(curr_loc, []):
                    if next_loc not in visited:
                        visited.add(next_loc)
                        queue.append((next_loc, dist + 1))

        # Parse goals for box destinations
        self.box_goals = {}
        # Assuming goal facts are (at box_name goal_location)
        for goal_fact in task.goals:
             pred, objs = parse_fact(goal_fact)
             if pred == 'at':
                 box_name, goal_loc = objs
                 self.box_goals[box_name] = goal_loc

    def __call__(self, state, task):
        """
        Computes the heuristic value for the given state.

        @param state: The current state (frozenset of facts).
        @param task: The planning task object.
        @return: The heuristic value (non-negative integer or float('inf')).
        """
        robot_loc = None
        box_locations = {} # box_name -> current_loc
        clear_locations = set()

        # Parse current state facts
        for fact in state:
            pred, objs = parse_fact(fact)
            if pred == 'at-robot':
                robot_loc = objs[0]
            elif pred == 'at':
                box_name, current_loc = objs
                box_locations[box_name] = current_loc
            elif pred == 'clear':
                clear_locations.add(objs[0])

        total_box_dist = 0
        boxes_to_move_locs = [] # Locations of boxes not at goal

        # Calculate sum of box-to-goal distances
        for box_name, current_loc in box_locations.items():
            goal_loc = self.box_goals.get(box_name)
            # Only consider boxes that have a goal specified
            if goal_loc is not None and current_loc != goal_loc:
                # Get precomputed distance between current box location and its goal
                if current_loc in self.location_distances and goal_loc in self.location_distances[current_loc]:
                     dist = self.location_distances[current_loc][goal_loc]
                     if dist == float('inf'): # Should not happen if precomputation was correct and grid connected
                         return float('inf') # Box goal unreachable
                     total_box_dist += dist
                     boxes_to_move_locs.append(current_loc)
                else:
                     # This means current_loc or goal_loc was not in the precomputed graph
                     # (e.g., disconnected component). Problem likely unsolvable.
                     return float('inf')

        # If all boxes are at their goals, the heuristic is 0
        if total_box_dist == 0:
            return 0

        # Calculate robot distance to nearest location adjacent to a box that needs moving
        target_robot_locations = set()
        for box_loc_to_move in boxes_to_move_locs:
            # Find locations adjacent to this box location
            for adj_loc, _ in self.adj_list.get(box_loc_to_move, []):
                target_robot_locations.add(adj_loc)

        # If there are boxes to move but no locations adjacent to them (e.g., box in 1x1 room)
        # This might indicate an unsolvable state if the box cannot be reached.
        # However, the box_to_goal distance would likely be inf already in such a case.
        # Let's proceed with BFS. If target_robot_locations is empty, BFS won't find anything.
        if not target_robot_locations:
             # This case is unlikely if boxes_to_move_locs is not empty and adj_list is built correctly
             # but handle defensively. If no adjacent locations exist, robot can't push.
             return float('inf')


        # Robot BFS to find shortest path from robot_loc to any target_robot_location
        # Robot can only move through clear cells.
        queue = deque([(robot_loc, 0)])
        visited = {robot_loc}
        robot_dist = float('inf')

        while queue:
            curr_loc, dist = queue.popleft()

            if curr_loc in target_robot_locations:
                robot_dist = dist
                break # Found shortest path to one target location

            # Explore neighbors reachable by robot (must be clear)
            for next_loc, _ in self.adj_list.get(curr_loc, []):
                # Robot can move to next_loc if it is clear and not visited
                if next_loc in clear_locations and next_loc not in visited:
                    visited.add(next_loc)
                    queue.append((next_loc, dist + 1))

        # If robot cannot reach any target location adjacent to a box
        if robot_dist == float('inf'):
            # This state might be a deadlock for the robot or unsolvable
            return float('inf')

        # The heuristic is the sum of box distances and robot distance
        return total_box_dist + robot_dist
