from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at ball1 rooma)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Simple check: if not using wildcards, parts and args must have same length
    if '*' not in args and len(parts) != len(args):
         return False
    # Check if each part matches the corresponding arg pattern
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost by summing the minimum number of pushes
    required for each box to reach its goal (ignoring other boxes as obstacles
    for the box's path) and the minimum number of robot moves required to reach
    a clear location adjacent to any box that is not yet at its goal.

    # Heuristic Initialization
    - Builds a graph representing the grid connectivity from `adjacent` facts.
    - Extracts the goal location for each box.

    # Step-by-Step Thinking for Computing Heuristic
    1. Identify the current location of the robot and all boxes.
    2. Identify locations occupied by boxes, as these are obstacles for the robot's path.
    3. For each box not at its goal location:
       - Calculate the shortest path distance (minimum number of pushes) for the box
         to reach its goal on the grid graph, ignoring other boxes as obstacles
         for the box's path. Sum these distances.
    4. If all boxes are at their goals, the heuristic is 0.
    5. If boxes remain outside their goals, calculate the minimum distance
       for the robot to reach *any* location `L` such that `L` is adjacent
       to a box location `B` (where `B` is a box not at its goal) and `L` is
       currently clear (or is the robot's current location). The robot's path
       must avoid locations occupied by boxes.
    6. The heuristic value is the sum of the total box push distances and the
       minimum robot distance calculated in step 5.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting:
        - The grid graph from adjacent facts.
        - Goal locations for each box.
        """
        self.goals = task.goals
        static_facts = task.static

        # Build the grid graph from adjacent facts
        # graph[loc1] = {loc2, loc3, ...} where loc2, loc3 are adjacent to loc1
        self.graph = {}
        for fact in static_facts:
            if match(fact, "adjacent", "*", "*", "*"):
                _, loc1, loc2, _ = get_parts(fact)
                if loc1 not in self.graph:
                    self.graph[loc1] = set()
                self.graph[loc1].add(loc2)
                # Assuming adjacency is symmetric for movement purposes
                if loc2 not in self.graph:
                    self.graph[loc2] = set()
                self.graph[loc2].add(loc1)

        # Store goal locations for each box.
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                box, location = args
                self.goal_locations[box] = location

    def shortest_path_distance(self, start_node, end_node, obstacles=None):
        """
        Computes the shortest path distance between two nodes in the graph
        using BFS, optionally avoiding obstacle nodes.
        Returns float('inf') if the end node is unreachable.
        """
        if start_node == end_node:
            return 0

        # Ensure start and end nodes are in the graph
        if start_node not in self.graph or end_node not in self.graph:
             return float('inf')

        queue = deque([(start_node, 0)])
        visited = {start_node}

        # Convert obstacles to a set for faster lookup
        obstacle_set = set(obstacles) if obstacles else set()

        while queue:
            current_node, dist = queue.popleft()

            # Check neighbors from the graph
            for neighbor in self.graph.get(current_node, []):
                if neighbor == end_node:
                    return dist + 1
                # Cannot move into a location occupied by an obstacle
                if neighbor not in visited and neighbor not in obstacle_set:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))

        # If end_node is not reachable
        return float('inf')

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Find robot location
        robot_loc = None
        for fact in state:
            if match(fact, "at-robot", "*"):
                robot_loc = get_parts(fact)[1]
                break
        if robot_loc is None:
             # Should not happen in a valid state
             return float('inf')

        # Find box locations and identify obstacles for robot movement
        box_locations = {}
        robot_obstacles = set() # Locations occupied by boxes are obstacles for robot movement
        for fact in state:
            if match(fact, "at", "*", "*"):
                obj_name = get_parts(fact)[1]
                loc = get_parts(fact)[2]
                # Check if it's a box using the goal_locations keys
                if obj_name in self.goal_locations:
                     box_locations[obj_name] = loc
                     robot_obstacles.add(loc) # Boxes are obstacles for robot movement

        # Heuristic component 1: Sum of box-to-goal distances (minimum pushes)
        # For box distance, we assume the box can move along the grid graph
        # This ignores other boxes as obstacles for the box's path, which is a simplification.
        box_distance_sum = 0
        boxes_not_at_goal = []
        for box, current_loc in box_locations.items():
            goal_loc = self.goal_locations.get(box)
            # Only consider boxes that have a goal and are not there yet
            if goal_loc and current_loc != goal_loc:
                # Calculate distance for the box on the grid graph (ignoring other boxes)
                dist = self.shortest_path_distance(current_loc, goal_loc)
                if dist == float('inf'):
                    # If a box cannot reach its goal even without obstacles, it's unsolvable
                    return float('inf')
                box_distance_sum += dist
                boxes_not_at_goal.append(box)

        # If all boxes are at their goals, the heuristic is 0.
        if not boxes_not_at_goal:
            return 0

        # Heuristic component 2: Robot cost
        # Calculate the minimum distance for the robot to reach *any* clear location
        # adjacent to *any* box that is not at its goal.
        min_robot_to_push_pos_dist = float('inf')
        found_potential_push_pos = False

        for box in boxes_not_at_goal:
            box_loc = box_locations[box]
            # Find locations adjacent to the box location
            for adj_loc in self.graph.get(box_loc, set()):
                 # Check if adj_loc is clear in the current state
                 # A location is clear if the state contains (clear adj_loc)
                 # or if the robot is already at adj_loc (cost 0)
                 is_clear_in_state = f"(clear {adj_loc})" in state or adj_loc == robot_loc

                 if is_clear_in_state:
                      found_potential_push_pos = True
                      # Calculate robot distance to this potential push position
                      # Robot path must avoid locations occupied by boxes (robot_obstacles)
                      dist = self.shortest_path_distance(robot_loc, adj_loc, obstacles=robot_obstacles)
                      min_robot_to_push_pos_dist = min(min_robot_to_push_pos_dist, dist)

        # If there are boxes not at goal, but no reachable clear location adjacent to any of them, it's stuck.
        # This check is important. If found_potential_push_pos is False, it means no adjacent clear spot exists.
        # If found_potential_push_pos is True, but min_robot_to_push_pos_dist is inf, it means robot cannot reach any of them.
        if boxes_not_at_goal and (not found_potential_push_pos or min_robot_to_push_pos_dist == float('inf')):
             return float('inf')

        # Final heuristic value: Sum of box pushes + robot cost to get into position for the first push.
        # This is a non-admissible estimate.
        return box_distance_sum + min_robot_to_push_pos_dist

