# Add necessary imports
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at-robot loc_1_1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure the number of parts matches the number of args, unless args contains wildcards
    # A simpler check: just zip and compare, fnmatch handles wildcards
    return len(parts) == len(args) and all(fnmatch(part, arg) for part, arg in zip(parts, args))

def shortest_path_distance(start, end, graph, obstacles=None):
    """
    Finds the shortest path distance between two locations in a graph.

    Args:
        start (str): The starting location.
        end (str): The target location.
        graph (dict): The graph representation {location: {direction: adjacent_location}}.
        obstacles (set, optional): A set of locations that cannot be traversed. Defaults to None.

    Returns:
        int or float('inf'): The shortest distance, or infinity if no path exists.
    """
    if start == end:
        return 0

    queue = deque([(start, 0)])
    visited = {start}

    obstacles_set = set(obstacles) if obstacles else set()

    # Cannot end inside an obstacle unless start is the end
    if end in obstacles_set and start != end:
        return float('inf')

    while queue:
        (current, dist) = queue.popleft()

        # Graph stores {location: {direction: adjacent_location}}
        # We need to iterate over adjacent locations regardless of direction for BFS
        neighbors = graph.get(current, {}).values()

        for neighbor in neighbors:
            if neighbor not in visited:
                # Cannot traverse through an obstacle unless it's the target
                if neighbor in obstacles_set and neighbor != end:
                    continue

                visited.add(neighbor)
                if neighbor == end:
                    return dist + 1
                queue.append((neighbor, dist + 1))

    return float('inf') # No path found

def bfs_path(start, end, graph, obstacles=None):
    """
    Finds a shortest path between two locations in a graph.

    Args:
        start (str): The starting location.
        end (str): The target location.
        graph (dict): The graph representation {location: {direction: adjacent_location}}.
        obstacles (set, optional): A set of locations that cannot be traversed. Defaults to None.

    Returns:
        list or None: A list of locations representing the path, or None if no path exists.
    """
    if start == end:
        return [start]

    queue = deque([(start, [start])])
    visited = {start}

    obstacles_set = set(obstacles) if obstacles else set()

    # Cannot end inside an obstacle unless start is the end
    if end in obstacles_set and start != end:
        return None

    while queue:
        (current, path) = queue.popleft()

        # Graph stores {location: {direction: adjacent_location}}
        neighbors = graph.get(current, {}).values()

        for neighbor in neighbors:
            if neighbor not in visited:
                 # Cannot traverse through an obstacle unless it's the target
                 if neighbor in obstacles_set and neighbor != end:
                    continue

                 visited.add(neighbor)
                 if neighbor == end:
                     return path + [neighbor]
                 queue.append((neighbor, path + [neighbor]))

    return None # No path found


class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing, for each box
    not at its goal, the minimum number of pushes required for that box plus
    the minimum number of robot moves to get into position for the first push
    of that box.

    # Assumptions
    - The grid structure is defined by `adjacent` predicates.
    - Boxes can only be moved by pushing.
    - The heuristic relaxes the constraint that the target location of a push
      must be clear (except for the goal location check).
    - The heuristic relaxes the constraint that the robot's path to the push
      position must be clear of the box being pushed (it only considers *other*
      boxes as obstacles for robot movement).
    - The heuristic assumes a box can be pushed from location A to B if A and B
      are adjacent and there is a location C adjacent to A in the same direction
      as A to B (where the robot would stand).

    # Heuristic Initialization
    - Extracts goal locations for each box from the goal state.
    - Builds an adjacency graph of locations from `adjacent` facts.
    - Builds a reverse adjacency graph to find the robot's required push position.
    - Builds a box-movable graph indicating possible box movements via push actions.
    - Defines a mapping for opposite directions.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the current location of the robot and all boxes.
    2. Identify the goal location for each box.
    3. Check if the current state is the goal state (all boxes at goal). If yes, return 0.
    4. Check for simple impossible states: if any goal location is occupied by a *different* box, return infinity.
    5. Initialize total heuristic cost to 0.
    6. For each box that is not at its goal location:
        a. Find the shortest path for the box from its current location to its goal location using the precomputed box-movable graph. This path represents the sequence of locations the box must occupy. If no path exists, the state is likely unsolvable or requires complex unblocking, so return infinity.
        b. The length of this path minus one is the minimum number of pushes required for this box.
        c. Determine the direction of the *first* push along this shortest path.
        d. Calculate the required location for the robot to stand *before* this first push (adjacent to the box's current location in the direction of the push). This is found using the reverse adjacency graph.
        e. Calculate the shortest path distance for the robot from its current location to this required push position. The robot's path is considered blocked by *other* boxes (those not currently being considered). If no path exists for the robot, return infinity.
        f. Add the robot's distance to the push position and the box's minimum pushes to the total heuristic cost.
    7. Return the total heuristic cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # Store goal locations for each box.
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                box_name, location = args
                self.goal_locations[box_name] = location

        # Build adjacency graph: location -> {direction_str -> adjacent_location}
        self.adjacency_graph = {}
        # Build reverse adjacency graph: location -> {direction_str -> adjacent_location}
        # If (adjacent l1 l2 dir), then l2 is adjacent to l1 in opp_dir.
        # This graph maps l2 -> {opp_dir -> l1}
        self.reverse_adjacency_graph = {}
        # Build box-movable graph: location -> {direction_str -> adjacent_location}
        # An edge l1 -> l2 in dir exists if (adjacent l1 l2 dir) AND robot can stand behind l1
        # to push towards l2. Robot stands at rloc where (adjacent rloc l1 dir).
        self.box_movable_graph = {}

        # Mapping for opposite directions
        self.opposite_direction = {
            'down': 'up',
            'up': 'down',
            'left': 'right',
            'right': 'left'
        }

        # First pass to build basic adjacency graphs
        for fact in static_facts:
            predicate, *args = get_parts(fact)
            if predicate == "adjacent":
                l1, l2, dir_str = args

                if l1 not in self.adjacency_graph:
                    self.adjacency_graph[l1] = {}
                self.adjacency_graph[l1][dir_str] = l2

                # Reverse graph: l2 is adjacent to l1 in the opposite direction
                opp_dir_str = self.opposite_direction.get(dir_str)
                if opp_dir_str:
                    if l2 not in self.reverse_adjacency_graph:
                        self.reverse_adjacency_graph[l2] = {}
                    self.reverse_adjacency_graph[l2][opp_dir_str] = l1 # Store l1 as adjacent to l2 in opp_dir

        # Second pass to build box-movable graph
        # A box can move from l1 to l2 in direction dir if (adjacent l1 l2 dir)
        # AND there exists rloc such that (adjacent rloc l1 dir).
        # This rloc is the location the robot stands. rloc is adjacent to l1 in direction dir.
        # This means l1 is adjacent to rloc in the opposite direction.
        # So, rloc = self.reverse_adjacency_graph[l1][opp_dir].
        for fact in static_facts:
             predicate, *args = get_parts(fact)
             if predicate == "adjacent":
                 l1, l2, dir_str = args
                 opp_dir_str = self.opposite_direction.get(dir_str)

                 if opp_dir_str: # Ensure opposite direction exists
                     # Check if there is a location behind l1 in the push direction (l1 -> l2)
                     # This location is adjacent to l1 in the opposite direction (l1 <- rloc)
                     # So, rloc = reverse_adjacency_graph[l1][opp_dir_str]
                     required_robot_pos_behind = self.reverse_adjacency_graph.get(l1, {}).get(opp_dir_str)

                     if required_robot_pos_behind is not None:
                         # Yes, there is a spot behind l1 where the robot can stand
                         if l1 not in self.box_movable_graph:
                             self.box_movable_graph[l1] = {}
                         self.box_movable_graph[l1][dir_str] = l2


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Get robot location
        robot_loc = None
        for fact in state:
            if match(fact, "at-robot", "*"):
                robot_loc = get_parts(fact)[1]
                break
        if robot_loc is None:
             # Should not happen in valid Sokoban states
             return float('inf')

        # Get current box locations
        current_box_locations = {}
        for fact in state:
            if match(fact, "at", "*", "*"):
                obj, loc = get_parts(fact)[1:]
                if obj in self.goal_locations: # Only track boxes that have a goal
                    current_box_locations[obj] = loc

        # Check for goal state
        is_goal = True
        for box, goal_loc in self.goal_locations.items():
            if current_box_locations.get(box) != goal_loc:
                is_goal = False
                break
        if is_goal:
            return 0

        # Check for impossible state: any goal location is occupied by a *different* box
        goal_locations_set = set(self.goal_locations.values())
        for box, loc in current_box_locations.items():
             if loc in goal_locations_set:
                 # This location is a goal location for *some* box
                 goal_for_box = None
                 for b, g in self.goal_locations.items():
                      if g == loc:
                           goal_for_box = b
                           break
                 if goal_for_box is not None and box != goal_for_box:
                      # This goal location is occupied by the wrong box
                      # This state is likely unsolvable in standard Sokoban
                      return float('inf')


        total_heuristic = 0

        boxes_to_move = [box for box, loc in current_box_locations.items() if loc != self.goal_locations[box]]

        for box in boxes_to_move:
            current_loc = current_box_locations[box]
            goal_loc = self.goal_locations[box]

            # Calculate box distance to goal using the box-movable graph
            box_path = bfs_path(current_loc, goal_loc, self.box_movable_graph)

            if box_path is None:
                # Box cannot reach its goal even ignoring other boxes and robot position
                return float('inf') # State is likely unsolvable

            dist_box_goal = len(box_path) - 1

            # If box is already at goal, its contribution is 0 (handled by boxes_to_move list)
            if dist_box_goal == 0:
                 continue

            # Find the required robot position for the first push
            next_loc = box_path[1] # The location the box moves to in the first step

            # Find the direction of the first push (current_loc -> next_loc)
            push_dir = None
            for direction, adj_loc in self.box_movable_graph.get(current_loc, {}).items():
                 if adj_loc == next_loc:
                      push_dir = direction
                      break

            if push_dir is None:
                 # Should not happen if box_path exists, but safety check
                 # This implies the first step of the box path is not a valid box move
                 return float('inf')

            # The robot needs to be at rloc such that (adjacent rloc current_loc push_dir)
            # This means current_loc is adjacent to rloc in the opposite direction
            opp_dir_str = self.opposite_direction.get(push_dir)
            if opp_dir_str is None:
                 # Should not happen with standard directions
                 return float('inf')

            # Find the required robot position (rloc) using the reverse adjacency graph
            # rloc = self.reverse_adjacency_graph[current_loc][opp_dir_str]
            required_robot_pos = self.reverse_adjacency_graph.get(current_loc, {}).get(opp_dir_str)

            if required_robot_pos is None:
                 # The grid structure doesn't allow pushing from this side
                 # This implies the box_movable_graph edge was invalid or problem definition is strange
                 # Or it requires moving a wall (impossible) or another box (not considered by box_movable_graph BFS)
                 # For this heuristic, we treat this as impossible via simple pushes
                 return float('inf')

            # Calculate robot distance to the required push position
            # Robot cannot move through other boxes
            occupied_by_other_boxes = {
                loc for b, loc in current_box_locations.items()
                if b != box # Robot can move to the location occupied by the box it will push
            }
            # The required_robot_pos might be occupied by the robot itself, or clear, or occupied by another box.
            # The BFS handles the case where the target is an obstacle.
            # We need to make sure the robot's current location is not considered an obstacle.
            # The BFS handles this by adding start to visited initially.

            dist_robot_to_push_pos = shortest_path_distance(
                robot_loc, required_robot_pos, self.adjacency_graph, occupied_by_other_boxes
            )

            if dist_robot_to_push_pos == float('inf'):
                 # Robot cannot reach the position to push this box
                 return float('inf') # State is likely unsolvable or requires complex unblocking

            # Add contribution for this box
            total_heuristic += dist_robot_to_push_pos + dist_box_goal

        return total_heuristic
