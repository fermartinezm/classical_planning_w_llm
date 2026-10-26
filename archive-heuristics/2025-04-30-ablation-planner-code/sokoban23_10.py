from collections import deque
# Assuming Heuristic base class is available in the environment
# from heuristics.heuristic_base import Heuristic

# If running standalone for testing, uncomment the mock Heuristic class
# class Heuristic:
#     def __init__(self, task):
#         self.goals = task.goals
#         self.static = task.static
#
#     def __call__(self, node):
#         pass

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact strings or malformed facts gracefully
    if not fact or not isinstance(fact, str) or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()

def bfs_distance(start, end, adj_map, obstacles):
    """
    Find the shortest path distance between start and end locations using BFS.

    Args:
        start (str): The starting location name.
        end (str): The target location name.
        adj_map (dict): Adjacency map {loc: {dir: neighbor_loc, ...}}.
        obstacles (list): A list of location names that cannot be traversed.

    Returns:
        int or float('inf'): The shortest distance or infinity if unreachable.
    """
    if start == end:
        return 0
    queue = deque([(start, 0)])
    visited = {start}
    obstacle_set = set(obstacles)

    while queue:
        current_loc, dist = queue.popleft()

        # Check neighbors from adj_map
        # adj_map might not contain current_loc if it's an isolated point (shouldn't happen in Sokoban grid)
        neighbors_dict = adj_map.get(current_loc)
        if neighbors_dict is None:
             continue # Should not happen in a connected grid

        neighbors = neighbors_dict.values()

        for neighbor in neighbors:
            if neighbor == end:
                return dist + 1
            # Cannot traverse through obstacle locations
            if neighbor not in visited and neighbor not in obstacle_set:
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))

    return float('inf') # Unreachable

class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the number of actions required to move all boxes
    to their goal locations. It sums the minimum pushes needed for each box
    (approximated by grid distance) and the minimum robot moves to reach
    the closest box that needs moving.

    # Assumptions
    - The grid structure is defined by 'adjacent' facts.
    - Boxes can only be pushed.
    - The heuristic uses BFS distance on the grid graph.
    - Robot movement is blocked by boxes.
    - Box movement distance is calculated ignoring other boxes as obstacles
      (optimistic estimate of pushes).

    # Heuristic Initialization
    - Parses goal facts to store the target location for each box.
    - Parses static 'adjacent' facts to build an adjacency map representing
      the grid graph, including reverse directions.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the current location of the robot and all boxes from the state.
    2. Determine which boxes are not yet at their designated goal locations.
    3. If all boxes are at their goals, the state is a goal state, return 0.
    4. For each box that needs to be moved, calculate the shortest path distance
       from its current location to its goal location using BFS on the grid graph.
       This BFS considers all locations traversable (ignoring other boxes or
       robot as obstacles for the box's path calculation). Sum these distances
       to get the total estimated minimum pushes required for all boxes.
    5. Calculate the shortest path distance from the robot's current location
       to the location of the closest box that needs moving. This BFS considers
       locations occupied by other boxes (that need moving) as obstacles for
       the robot's path.
    6. The heuristic value is the sum of the total estimated box pushes
       (from step 4) and the robot's distance to the closest box (from step 5).
    7. If any required path (box to goal or robot to box) is unreachable,
       return infinity to indicate a likely dead end.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and static facts.

        Args:
            task: An object containing task information, including goals and static facts.
                  Assumed to have attributes `goals` and `static`.
        """
        # Assuming task object has .goals and .static attributes
        self.goals = task.goals
        static_facts = task.static

        # Build adjacency map from static 'adjacent' facts
        self.adj_map = {}
        reverse_dir = {'up': 'down', 'down': 'up', 'left': 'right', 'right': 'left'}

        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == 'adjacent' and len(parts) == 4:
                loc1, loc2, direction = parts[1], parts[2], parts[3]
                if loc1 not in self.adj_map:
                    self.adj_map[loc1] = {}
                self.adj_map[loc1][direction] = loc2

                # Add reverse mapping
                if direction in reverse_dir:
                    rev_direction = reverse_dir[direction]
                    if loc2 not in self.adj_map:
                        self.adj_map[loc2] = {}
                    self.adj_map[loc2][rev_direction] = loc1
                # Note: Assuming only 'up', 'down', 'left', 'right' directions exist.


        # Store goal locations for each box
        self.box_goals = {}
        for goal in self.goals:
            parts = get_parts(goal)
            # Goal facts are typically (at box loc)
            if parts and parts[0] == 'at' and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                # Assume any object in a goal (at obj loc) is a box if it's not the robot
                # A more robust approach would use task.objects and task.types
                # but assuming simple goals like (at box1 loc)
                self.box_goals[obj] = loc
            # Handle potential 'and' in goals like (and (at box1 loc1) (at box2 loc2))
            elif parts and parts[0] == 'and':
                 for sub_goal_str in parts[1:]:
                     sub_parts = get_parts(sub_goal_str)
                     if sub_parts and sub_parts[0] == 'at' and len(sub_parts) == 3:
                         obj, loc = sub_parts[1], sub_parts[2]
                         self.box_goals[obj] = loc


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.

        Args:
            node: The current node in the search tree. Assumed to have a `state` attribute
                  which is a frozenset of PDDL fact strings.

        Returns:
            int or float('inf'): The estimated heuristic cost to reach a goal state.
        """
        state = node.state
        robot_loc = None
        current_box_locations = {}

        # Extract robot and box locations from the current state
        for fact in state:
            parts = get_parts(fact)
            if parts:
                if parts[0] == 'at-robot' and len(parts) == 2:
                    robot_loc = parts[1]
                elif parts[0] == 'at' and len(parts) == 3:
                    obj, loc = parts[1], parts[2]
                    # Only track locations for objects that are boxes and have goals
                    if obj in self.box_goals:
                         current_box_locations[obj] = loc

        # Check if all boxes are at their goals
        all_boxes_at_goal = True
        # Need to check if all boxes *defined in the goals* are at their goals
        # and if there are no other boxes not mentioned in goals (assuming not).
        # Also need to check if the number of boxes in state matches the number in goals.
        # For simplicity, assume only boxes mentioned in goals exist and need moving.
        if len(current_box_locations) != len(self.box_goals):
             all_boxes_at_goal = False # Should not happen in valid states?

        if all_boxes_at_goal:
            for box, goal_loc in self.box_goals.items():
                if box not in current_box_locations or current_box_locations[box] != goal_loc:
                    all_boxes_at_goal = False
                    break # Exit loop early if any box is not at its goal

        if all_boxes_at_goal:
            return 0 # Goal state

        boxes_to_move = {
            box: loc for box, loc in current_box_locations.items()
            if box in self.box_goals and current_box_locations[box] != self.box_goals[box]
        }

        # Calculate total box distance (minimum pushes)
        total_box_distance = 0
        for box, loc in boxes_to_move.items():
            goal_loc = self.box_goals[box]
            # BFS for box path ignores obstacles (other boxes, robot)
            dist = bfs_distance(loc, goal_loc, self.adj_map, [])
            if dist == float('inf'):
                # Box cannot reach its goal location
                return float('inf') # Indicates a dead end
            total_box_distance += dist

        # Calculate robot distance to the closest box that needs moving
        robot_to_closest_box_distance = float('inf')
        # Obstacles for robot are locations occupied by *any* box
        all_box_locations = list(current_box_locations.values())

        for box, loc in boxes_to_move.items():
            # When calculating distance to 'loc', 'loc' itself is the target, not an obstacle
            # The obstacles are locations of *other* boxes.
            obstacles_for_this_robot_path = [b_loc for b_loc in all_box_locations if b_loc != loc]
            # Robot cannot move into its own current location either, but BFS starts there.
            dist = bfs_distance(robot_loc, loc, self.adj_map, obstacles_for_this_robot_path)
            robot_to_closest_box_distance = min(robot_to_closest_box_distance, dist)

        if robot_to_closest_box_distance == float('inf'):
             # Robot cannot reach any box that needs moving
             return float('inf') # Indicates a dead end

        # Combine distances: total box pushes + robot moves to get to the first box
        # This is a simple sum. Each push action costs 1. Robot moves cost 1.
        # The total box distance is a lower bound on pushes.
        # The robot needs to reach the box (robot_to_closest_box_distance)
        # Then it needs to perform pushes (total_box_distance).
        # Each push moves the robot too.
        # A simple sum is a reasonable non-admissible estimate.
        heuristic_value = total_box_distance + robot_to_closest_box_distance

        return heuristic_value
