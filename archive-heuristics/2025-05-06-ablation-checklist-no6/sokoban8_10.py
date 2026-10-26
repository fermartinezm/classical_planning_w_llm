from fnmatch import fnmatch
from collections import deque
# Assume Heuristic base class is available
# from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact."""
    # Ensure fact is a string and remove surrounding parentheses
    if isinstance(fact, str) and fact.startswith('(') and fact.endswith(')'):
        return fact[1:-1].split()
    # Fallback for unexpected formats, though string facts are expected
    return str(fact)[1:-1].split()


# Helper function to check if a PDDL fact matches a given pattern.
# Pattern can include wildcards (*).
def match(fact, *args):
    """Check if a PDDL fact matches a given pattern."""
    parts = get_parts(fact)
    # Ensure the number of parts matches the number of pattern arguments
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# Assume Heuristic base class is defined elsewhere and imported
# If running standalone or without the framework, you might need a dummy base class:
# class Heuristic:
#     def __init__(self, task): pass
#     def __call__(self, node): raise NotImplementedError

# Uncomment the appropriate class definition depending on the environment
# class sokobanHeuristic(Heuristic): # Use this line if Heuristic base class is available
class sokobanHeuristic: # Use this line if Heuristic base class is NOT available
    """
    A domain-dependent heuristic for the Sokoban domain.

    Estimates the cost based on box-goal distances and robot-to-push-position distances.
    H = Sum(shortest_path_dist(box_i, goal_i, ignore_dynamic_obstacles))
        + shortest_path_dist(robot, any_location_adjacent_to_any_misplaced_box, avoid_boxes)

    This heuristic is not admissible but aims to guide a greedy best-first search
    efficiently by prioritizing states where boxes are closer to their goals
    and the robot is closer to a position from which it can push a misplaced box.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and building the grid graph.
        """
        # If inheriting from Heuristic and it requires task in its __init__,
        # uncomment the following line:
        # super().__init__(task)

        self.goals = task.goals
        self.static_facts = task.static

        # Store goal locations for each box
        # We assume goals are only of the form (at boxX locY)
        self.goal_locations = {}
        for goal in self.goals:
            # Goal facts are typically strings like '(at box1 loc_2_4)'
            predicate, *args = get_parts(goal)
            if predicate == "at" and len(args) == 2: # Ensure it's an (at obj loc) fact
                obj_name, loc = args[0], args[1]
                # Assume any object mentioned in an 'at' goal is a box
                self.goal_locations[obj_name] = loc

        # Build the grid graph from adjacent facts
        # The graph represents traversable locations and their direct connections.
        # Locations not in this graph are effectively walls.
        self.graph = {} # location_str -> set(neighbor_location_str)

        for fact in self.static_facts:
            # Facts are strings like '(adjacent loc_1_1 loc_1_2 right)'
            if match(fact, "adjacent", "*", "*", "*"):
                _, loc1, loc2, direction = get_parts(fact)
                # Add loc1 and loc2 to the graph if not present
                if loc1 not in self.graph: self.graph[loc1] = set()
                if loc2 not in self.graph: self.graph[loc2] = set()
                # Add adjacency in both directions (assuming grid is traversable both ways)
                self.graph[loc1].add(loc2)
                self.graph[loc2].add(loc1)

    def _bfs_distance(self, start_loc, end_loc, obstacles):
        """
        Computes the shortest path distance between start_loc and end_loc
        on the grid graph, avoiding locations in the obstacles set.
        Returns float('inf') if no path exists.
        """
        # If start or end is the same, distance is 0
        if start_loc == end_loc:
            return 0

        # If start location is an obstacle or not a valid location in the graph,
        # it's unreachable.
        if start_loc in obstacles or start_loc not in self.graph:
             return float('inf')

        queue = deque([(start_loc, 0)])
        visited = {start_loc}

        while queue:
            current_loc, dist = queue.popleft()

            # current_loc is guaranteed to be in self.graph if it was added to the queue

            # Explore neighbors of the current location
            # Use .get() with an empty set as default for safety if a location somehow
            # exists in visited/queue but not the graph (shouldn't happen with current logic).
            for neighbor_loc in self.graph.get(current_loc, set()):
                # Avoid obstacles and already visited locations
                if neighbor_loc not in visited and neighbor_loc not in obstacles:
                    # If the neighbor is the target, we found the shortest path
                    if neighbor_loc == end_loc:
                        return dist + 1
                    # Otherwise, add neighbor to visited and queue
                    visited.add(neighbor_loc)
                    queue.append((neighbor_loc, dist + 1))

        # If the queue is empty and the target was not reached, it's unreachable
        return float('inf')

    def _bfs_distance_to_set(self, start_loc, target_locs, obstacles):
        """
        Computes the shortest path distance from start_loc to any location
        in the target_locs set, avoiding locations in the obstacles set.
        Returns float('inf') if no location in the set is reachable.
        """
        # If start location is an obstacle or not a valid location in the graph,
        # it's unreachable.
        if start_loc in obstacles or start_loc not in self.graph:
             return float('inf')

        # Check if start is already in the target set (and not an obstacle)
        # This check is technically redundant due to the first line, but harmless.
        if start_loc in target_locs:
            return 0

        queue = deque([(start_loc, 0)])
        visited = {start_loc}

        while queue:
            current_loc, dist = queue.popleft()

            # current_loc is guaranteed to be in self.graph if it was added to the queue

            # Explore neighbors of the current location
            for neighbor_loc in self.graph.get(current_loc, set()):
                # Avoid obstacles and already visited locations
                if neighbor_loc not in visited and neighbor_loc not in obstacles:
                    # If the neighbor is in the target set, we found the shortest path
                    if neighbor_loc in target_locs:
                        return dist + 1
                    # Otherwise, add neighbor to visited and queue
                    visited.add(neighbor_loc)
                    queue.append((neighbor_loc, dist + 1))

        return float('inf') # No location in the set found

    def _bfs_distance_to_any_adjacent_to_set(self, start_loc, target_locs, obstacles):
        """
        Computes the shortest path distance from start_loc to any location L
        such that L is adjacent to any location in target_locs, avoiding locations
        in the obstacles set.
        Returns float('inf') if no such location is reachable.
        """
        # If start location is an obstacle or not a valid location in the graph,
        # it's unreachable.
        if start_loc in obstacles or start_loc not in self.graph:
             return float('inf')

        queue = deque([(start_loc, 0)])
        visited = {start_loc}

        # Build a set of all locations adjacent to any target location
        adjacent_to_targets = set()
        for target_loc in target_locs:
            # Ensure target_loc is a valid grid cell before checking neighbors
            if target_loc in self.graph:
                for neighbor_loc in self.graph[target_loc]:
                    adjacent_to_targets.add(neighbor_loc)

        # If robot starts adjacent to a target (and not on an obstacle), distance is 0
        # This check is technically redundant due to the first line, but harmless.
        if start_loc in adjacent_to_targets:
             return 0

        while queue:
            current_loc, dist = queue.popleft()

            # current_loc is guaranteed to be in self.graph if it was added to the queue

            # Explore neighbors of the current location
            for neighbor_loc in self.graph.get(current_loc, set()):
                # Avoid obstacles and already visited locations
                if neighbor_loc not in visited and neighbor_loc not in obstacles:
                    # If the neighbor is adjacent to a target, we found the shortest path
                    if neighbor_loc in adjacent_to_targets:
                        return dist + 1
                    # Otherwise, add neighbor to visited and queue
                    visited.add(neighbor_loc)
                    queue.append((neighbor_loc, dist + 1))

        return float('inf') # No location adjacent to a target found

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state

        # Check if goal is reached. If so, heuristic is 0.
        if self.goals <= state:
            return 0

        # Extract current locations of robot and boxes from the state
        robot_loc = None
        box_locations = {} # Map box_name -> location_str

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip any potentially empty parsed facts
            predicate = parts[0]
            if predicate == "at-robot" and len(parts) == 2:
                robot_loc = parts[1]
            # Check for 'at' facts for objects that are boxes (i.e., have a goal location)
            elif predicate == "at" and len(parts) == 3:
                obj_name, loc = parts[1], parts[2]
                if obj_name in self.goal_locations:
                     box_locations[obj_name] = loc

        # Identify boxes that are not currently at their goal locations
        misplaced_boxes = {
            box for box, loc in box_locations.items()
            if self.goal_locations.get(box) != loc
        }

        # If there are no misplaced boxes, the goal must be reached (already checked)
        # This case should ideally not be reached if the initial goal check is correct.
        # If it is reached, it means all boxes are at goals, so heuristic is 0.
        if not misplaced_boxes:
             return 0

        # --- Heuristic Calculation ---

        # Component 1: Sum of Box-to-Goal Distances
        # Calculate the shortest path distance for each misplaced box to its assigned goal.
        # For simplicity and efficiency (non-admissible heuristic), we ignore dynamic
        # obstacles (other boxes, robot) for this calculation. Obstacles are only
        # static walls (locations not in the graph).
        box_goal_distance_sum = 0
        for box in misplaced_boxes:
            current_box_loc = box_locations[box]
            goal_box_loc = self.goal_locations[box]

            # Calculate distance using BFS, ignoring dynamic obstacles (empty set)
            dist = self._bfs_distance(current_box_loc, goal_box_loc, set())

            # If any box cannot reach its goal location even ignoring obstacles,
            # the state is likely unsolvable. Return infinity.
            if dist == float('inf'):
                 return float('inf')

            box_goal_distance_sum += dist

        # Component 2: Robot-to-Push Position Distance
        # The robot needs to reach a location adjacent to a misplaced box to push it.
        # Calculate the minimum shortest path distance from the robot's current location
        # to *any* location that is adjacent to *any* misplaced box location.
        # Obstacles for the robot's movement are the locations occupied by boxes.
        misplaced_box_locations = {box_locations[box] for box in misplaced_boxes}

        # Obstacles for robot movement are all box locations (robot cannot move onto a box)
        obstacles_for_robot = set(box_locations.values())

        # Calculate distance from robot_loc to any location adjacent to a misplaced box location,
        # avoiding locations occupied by boxes.
        robot_to_push_pos_dist = self._bfs_distance_to_any_adjacent_to_set(
            robot_loc,
            misplaced_box_locations,
            obstacles_for_robot
        )

        # If the robot cannot reach any push position, the state is likely unsolvable.
        # This might happen if all misplaced boxes are surrounded by walls/other boxes
        # such that no adjacent location is reachable by the robot.
        if robot_to_push_pos_dist == float('inf'):
             return float('inf')

        # --- Combine Components ---
        # The total heuristic is the sum of the box-goal distances and the robot-to-push distance.
        # This is a common pattern: cost to achieve subgoals (boxes at goals) + cost to enable action (robot reaching a push spot).
        # Each unit of box-goal distance roughly corresponds to one push action.
        # The robot-to-push distance is the cost (in moves) to get the robot
        # into a position to perform the first push on any relevant box.
        # This is a non-admissible estimate of the total number of actions (moves + pushes).

        total_heuristic = box_goal_distance_sum + robot_to_push_pos_dist

        # The heuristic value is non-negative. It is 0 only if the goal is reached.
        # It is finite for solvable states and infinity for detected unsolvable states.

        return total_heuristic
