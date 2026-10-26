from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic

# Helper functions to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at obj loc)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure the number of parts is at least the number of args and matches the pattern
    if len(parts) < len(args) or not all(fnmatch(part, arg) for part, arg in zip(parts, args)):
         return False
    # If args is shorter than parts, we only check the prefix match, which is intended
    return True


class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal state by summing the
    shortest path distance for each box to its goal location and adding the
    minimum shortest path distance for the robot to reach a location from
    which it can push *any* box that is not yet at its goal.

    # Assumptions
    - The grid structure is defined by `adjacent` predicates.
    - Boxes need to reach specific goal locations.
    - The heuristic assumes box paths are clear of other boxes (non-admissible relaxation).
    - The heuristic assumes the robot can reach a push position if a path exists,
      ignoring temporary obstacles it could move (non-admissible relaxation).

    # Heuristic Initialization
    - Parses the goal conditions to map each box to its goal location.
    - Parses the static facts (`adjacent` predicates) to build a graph
      representation of the locations and their adjacencies, including directions.
      Stores this as an adjacency map: `location -> {direction: neighbor_location}`.
    - Stores the mapping of directions to their opposites.

    # Step-By-Step Thinking for Computing Heuristic
    1. Check if the current state satisfies all goal conditions (all boxes are at their goal locations as specified in `self.box_goals`). If yes, the heuristic is 0.
    2. Parse the current state to identify the robot's location (`loc_r`) and the current location of each box (`box_locations`).
    3. Initialize `total_box_distance = 0` and `min_robot_distance_to_push_pos = infinity`.
    4. Create a set of locations currently occupied by boxes (`robot_obstacles`). These locations are obstacles for the robot's movement BFS.
    5. Initialize a flag `boxes_to_move = False`.
    6. Iterate through each box defined in the goal conditions (`self.box_goals`):
       a. Get the box's current location (`loc_b`) from `box_locations`. If the box is not found in the state (should not happen in valid states), treat as unsolvable and return infinity.
       b. Get the box's goal location (`goal_location`) from `self.box_goals`.
       c. If `loc_b` is the same as `goal_location`, this box is already at its goal; continue to the next box.
       d. Set `boxes_to_move = True`.
       e. Calculate the shortest path distance for the box from `loc_b` to `goal_location` using BFS on the location graph (`self.adj_map`). Obstacles for the box path are considered only implicit walls (locations not present in `self.adj_map`). If the goal is unreachable for the box, the state is likely unsolvable; return infinity. Add this distance to `total_box_distance`.
       f. For the current box, find the minimum shortest path distance for the robot to reach *any* location `push_loc` from which it can push the box towards *any* adjacent location (`target_loc`). A `push_loc` is adjacent to `loc_b` in the opposite direction of the push `loc_b -> target_loc`. Calculate the robot's shortest path from its current location `loc_r` to `push_loc` using BFS, avoiding locations in `robot_obstacles`. Update `min_robot_distance_to_push_pos` with the minimum distance found across all possible push locations for *this* box.
    7. After iterating through all boxes:
       a. If `boxes_to_move` is still `False` (meaning all boxes were at their goals), return 0. This case should ideally be caught by the initial check, but serves as a safeguard.
       b. If `min_robot_distance_to_push_pos` is still infinity (robot cannot reach any push position for any box that needs moving), return infinity.
       c. Otherwise, the heuristic value is `total_box_distance + min_robot_distance_to_push_pos`.

    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and static facts.
        """
        super().__init__(task) # Call base class constructor if needed

        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # Store goal locations for each box.
        self.box_goals = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                # Goal is (at box_name location_name)
                box_name, location_name = args
                self.box_goals[box_name] = location_name

        # Build adjacency map: location -> {direction: neighbor_location}
        self.adj_map = {}
        for fact in static_facts:
            predicate, *args = get_parts(fact)
            if predicate == "adjacent":
                # Fact is (adjacent loc1 loc2 dir)
                loc1_name, loc2_name, dir_name = args
                self.adj_map.setdefault(loc1_name, {})[dir_name] = loc2_name

        # Define opposite directions
        self.opposite_dirs = {'up': 'down', 'down': 'up', 'left': 'right', 'right': 'left'}

    def get_adjacent_location(self, loc, dir):
        """Returns the location adjacent to `loc` in `dir`, or None if none exists."""
        return self.adj_map.get(loc, {}).get(dir)

    def bfs(self, start, end, adj_map, obstacles):
        """
        Performs Breadth-First Search to find the shortest path distance.

        Args:
            start (str): The starting location.
            end (str): The target location.
            adj_map (dict): The adjacency map of the graph.
            obstacles (set): A set of locations that cannot be visited.

        Returns:
            int: The shortest distance, or float('inf') if unreachable.
        """
        if start == end:
            return 0
        # Cannot start inside an obstacle unless start == end (distance 0)
        if start in obstacles:
             return float('inf')

        queue = deque([(start, 0)])
        visited = {start}

        while queue:
            current_loc, dist = queue.popleft()

            # Check neighbors from the adjacency map
            # Iterate over neighbor locations directly
            for neighbor_loc in adj_map.get(current_loc, {}).values():
                if neighbor_loc == end:
                    return dist + 1
                if neighbor_loc not in obstacles and neighbor_loc not in visited:
                    visited.add(neighbor_loc)
                    queue.append((neighbor_loc, dist + 1))

        return float('inf') # End not reachable

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state  # Current world state (frozenset of fact strings).

        # Parse current state
        loc_r = None
        box_locations = {} # {box_name: location_name}

        for fact in state:
            predicate, *args = get_parts(fact)
            if predicate == "at-robot":
                loc_r = args[0]
            elif predicate == "at" and len(args) == 2: # (at box location)
                box_name, location_name = args
                box_locations[box_name] = location_name

        # Check if goal is reached
        is_goal_state = True
        for box, goal_location in self.box_goals.items():
            if box not in box_locations or box_locations[box] != goal_location:
                is_goal_state = False
                break

        if is_goal_state:
            return 0

        # Compute heuristic components
        total_box_dist = 0
        min_robot_dist_to_any_push_pos = float('inf')

        # Locations occupied by boxes (obstacles for the robot)
        robot_obstacles = set(box_locations.values())

        boxes_to_move = False

        for box, goal_location in self.box_goals.items():
            loc_b = box_locations.get(box) # Get current location of this box

            if loc_b is None:
                 # This box is not in the state? Indicates an invalid state or parsing issue.
                 # Treat as unsolvable.
                 return float('inf')

            if loc_b == goal_location:
                continue # This box is already at its goal

            boxes_to_move = True

            # 1. Box distance component
            # BFS for box path, obstacles are implicit walls (locations not in adj_map)
            # Note: Box BFS should technically avoid locations occupied by *other* boxes,
            # but for a non-admissible relaxation, we ignore other boxes as obstacles here.
            box_dist = self.bfs(loc_b, goal_location, self.adj_map, set())
            if box_dist == float('inf'):
                # Box cannot reach its goal location
                return float('inf')
            total_box_dist += box_dist

            # 2. Robot distance component for this box
            min_robot_dist_for_this_box = float('inf')

            # Iterate through all possible directions the box could be pushed from loc_b
            # A push from loc_b to target_loc requires robot at push_loc.
            # push_loc is adjacent to loc_b in the opposite direction of loc_b -> target_loc.
            for push_dir, target_loc in self.adj_map.get(loc_b, {}).items():
                 opposite_dir = self.opposite_dirs.get(push_dir)
                 if opposite_dir:
                     push_loc = self.get_adjacent_location(loc_b, opposite_dir)

                     if push_loc: # Check if such a push_loc exists
                         # BFS for robot path from loc_r to push_loc
                         # Obstacles for robot are locations occupied by *any* box
                         robot_dist = self.bfs(loc_r, push_loc, self.adj_map, robot_obstacles)
                         min_robot_dist_for_this_box = min(min_robot_dist_for_this_box, robot_dist)

            # Update the minimum robot distance needed for *any* box that needs moving
            min_robot_dist_to_any_push_pos = min(min_robot_dist_to_any_push_pos, min_robot_dist_for_this_box)


        # If no boxes needed moving, the initial goal check should have returned 0.
        # This case should only be reached if box_goals was empty or all boxes were at goals.
        if not boxes_to_move:
             return 0

        # If we need to move boxes but the robot cannot reach any push position for any of them
        if min_robot_dist_to_any_push_pos == float('inf'):
             return float('inf')

        # The heuristic is the sum of box distances plus the minimum robot distance
        # to get into position for *one* of the boxes that needs moving.
        return total_box_dist + min_robot_dist_to_any_push_pos
