from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque

# Helper function to parse PDDL fact strings
def get_parts(fact):
    """Removes parentheses and splits the fact string into parts."""
    return fact[1:-1].split()

# Helper function to match PDDL fact strings
def match(fact, *args):
    """Checks if a fact matches a pattern using fnmatch."""
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# Helper function to parse location string like 'loc_row_col' into (row, col) tuple
def parse_location(loc_str):
    """Converts a location string 'loc_row_col' to a (row, col) tuple."""
    parts = loc_str.split('_')
    # Assuming format is always loc_row_col
    if len(parts) != 3 or parts[0] != 'loc':
        # Handle unexpected format, maybe return None or raise error
        # For this problem, assume format is correct.
        raise ValueError(f"Unexpected location format: {loc_str}")
    try:
        row = int(parts[1])
        col = int(parts[2])
        return (row, col)
    except ValueError:
        raise ValueError(f"Could not parse row/col from location string: {loc_str}")


# Helper function to calculate Manhattan distance between two location tuples
def manhattan_distance(loc1_tuple, loc2_tuple):
    """Calculates the Manhattan distance between two (row, col) tuples."""
    r1, c1 = loc1_tuple
    r2, c2 = loc2_tuple
    return abs(r1 - r2) + abs(c1 - c2)

class sokobanHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Sokoban domain.

    Summary:
    Estimates the cost to reach the goal state by summing the Manhattan
    distances of each box to its goal location and adding the shortest
    path distance for the robot to reach any location adjacent
    to any box that is not yet at its goal, considering other boxes as obstacles.

    Assumptions:
    - The PDDL instance uses location names in the format 'loc_row_col'.
    - The grid defined by 'adjacent' facts is connected.
    - The goal state is defined by the 'at' predicate for all boxes.
    - The heuristic is non-admissible and designed for greedy best-first search.

    Heuristic Initialization:
    - Parses static 'adjacent' facts to build a graph representation of the grid.
      The graph maps (row, col) tuples to lists of adjacent (row, col) tuples.
    - Parses goal 'at' facts to store the target location (as a (row, col) tuple)
      for each box.

    Step-By-Step Thinking for Computing Heuristic:
    1. Check if the current state is the goal state (all boxes at their goal locations). If yes, return 0.
    2. Extract the current location of the robot and all boxes from the state.
       Convert all relevant location strings to (row, col) tuples.
    3. Calculate the sum of Manhattan distances for each box from its current location
       to its goal location. This provides a lower bound on the number of pushes required
       for all boxes combined, ignoring obstacles and robot position.
    4. Identify potential target locations for the robot's movement. These are any
       locations that are adjacent to a box that is not yet at its goal. The robot
       needs to reach one of these locations to be able to push a box.
    5. Identify obstacles for the robot's movement. These are any locations currently
       occupied by a box. The robot cannot move into these locations.
    6. Perform a Breadth-First Search (BFS) on the grid graph starting from the
       robot's current location. The BFS should only traverse to neighbor locations
       that are not in the set of obstacles. The goal of the BFS is to find the
       shortest path distance from the robot's current location to *any* location
       in the set of potential robot target locations.
    7. If the BFS finds a path, the distance is the minimum number of robot moves
       required to get into a position from which it might be able to push a box.
    8. If the BFS cannot reach any target robot location (e.g., robot is trapped
       or all adjacent spots to boxes are unreachable), the state is likely unsolvable
       or requires significant backtracking. Return infinity in this case.
    9. The total heuristic value is the sum of the box-goal Manhattan distances
       (step 3) and the minimum robot approach distance calculated by the BFS (step 7).
    """
    def __init__(self, task):
        super().__init__(task) # Call parent constructor

        self.goal_locations = {} # box_name -> (row, col) tuple
        for goal in task.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                box, location = args
                self.goal_locations[box] = parse_location(location)

        self.graph = {} # (row, col) tuple -> list of (row, col) tuples

        # Build the adjacency graph from static 'adjacent' facts
        for fact in task.static:
            if match(fact, "adjacent", "*", "*", "*"):
                _, loc1_str, loc2_str, _ = get_parts(fact)
                loc1_tuple = parse_location(loc1_str)
                loc2_tuple = parse_location(loc2_tuple)
                # Add directed edges. Assuming grid is undirected, add both ways.
                self.graph.setdefault(loc1_tuple, []).append(loc2_tuple)
                self.graph.setdefault(loc2_tuple, []).append(loc1_tuple)


    def bfs_distance_to_set(self, start_loc_tuple, target_loc_tuples_set, obstacle_loc_tuples_set):
        """
        Performs BFS on the grid graph to find the shortest path distance
        from start_loc_tuple to any location in target_loc_tuples_set,
        avoiding locations in obstacle_loc_tuples_set.

        Returns the minimum distance found, or None if no target is reachable.
        """
        # If the start location is already one of the targets, distance is 0
        if start_loc_tuple in target_loc_tuples_set:
            return 0

        # If start location is an obstacle, it's unreachable (shouldn't happen for robot start)
        if start_loc_tuple in obstacle_loc_tuples_set:
             return None

        queue = deque([(start_loc_tuple, 0)])
        visited = {start_loc_tuple}

        while queue:
            current_loc_tuple, dist = queue.popleft()

            # Explore neighbors
            for neighbor_loc_tuple in self.graph.get(current_loc_tuple, []):
                # Cannot move into an obstacle location
                if neighbor_loc_tuple in obstacle_loc_tuples_set:
                    continue

                # Avoid revisiting locations
                if neighbor_loc_tuple not in visited:
                    # Check if this neighbor is one of the targets
                    if neighbor_loc_tuple in target_loc_tuples_set:
                        return dist + 1 # Found a target location, return distance

                    # Add neighbor to visited and queue for further exploration
                    visited.add(neighbor_loc_tuple)
                    queue.append((neighbor_loc_tuple, dist + 1))

        # If the queue is empty and no target was reached
        return None


    def __call__(self, node):
        state = node.state

        # Extract dynamic information from the current state
        robot_loc_str = None
        box_locations_str = {} # box_name -> loc_str

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at-robot":
                robot_loc_str = parts[1]
            elif parts[0] == "at" and len(parts) == 3: # (at box loc)
                 box_locations_str[parts[1]] = parts[2]

        # Convert locations to tuples
        if robot_loc_str is None:
             # Robot location not found in state, should not happen in valid states
             return float('inf') # Indicate invalid/unsolvable state

        robot_loc_tuple = parse_location(robot_loc_str)
        box_locations_tuple = {box: parse_location(loc) for box, loc in box_locations_str.items()}

        # Check if goal is reached
        goal_reached = True
        # Iterate through required goals
        for box, goal_loc_tuple in self.goal_locations.items():
             # Check if the box exists in the current state and is at the goal location
             current_loc_tuple = box_locations_tuple.get(box)
             if current_loc_tuple is None or current_loc_tuple != goal_loc_tuple:
                 goal_reached = False
                 break

        if goal_reached:
            return 0

        # Calculate sum of box-goal Manhattan distances
        box_goal_distance_sum = 0
        boxes_not_at_goal = []
        for box, goal_loc_tuple in self.goal_locations.items():
             current_loc_tuple = box_locations_tuple.get(box)
             # Only consider boxes that exist and are not at their goal
             if current_loc_tuple is not None and current_loc_tuple != goal_loc_tuple:
                 box_goal_distance_sum += manhattan_distance(current_loc_tuple, goal_loc_tuple)
                 boxes_not_at_goal.append(box)

        # If no boxes need moving (should be caught by goal_reached check, but safety)
        if not boxes_not_at_goal:
             return 0 # Should not be reached if goal_reached is False

        # Identify potential target locations for robot BFS: any location adjacent to a non-goal box
        target_robot_locations_tuple = set()
        for box in boxes_not_at_goal:
            current_loc_tuple = box_locations_tuple[box]
            # Ensure current_loc_tuple is in graph (should be if it's a valid location)
            for neighbor_loc_tuple in self.graph.get(current_loc_tuple, []):
                 # Any location adjacent to a non-goal box is a potential target for the robot
                 target_robot_locations_tuple.add(neighbor_loc_tuple)

        # If there are no potential target locations (e.g., no adjacent spots for any non-goal box)
        # This could happen if a box is in a location with no defined adjacencies in the static facts.
        # In a well-formed grid, this shouldn't happen for non-goal boxes unless they are stuck.
        # If no targets, the robot cannot approach any box needing a push.
        # The heuristic should reflect this difficulty. Returning infinity seems reasonable.
        if not target_robot_locations_tuple:
             return float('inf')


        # Identify obstacles for robot BFS: locations occupied by boxes
        # The robot cannot move *into* a location occupied by a box.
        robot_obstacles_tuple = set(box_locations_tuple.values())

        # Calculate shortest path distance for robot to reach any target location, avoiding obstacles
        robot_approach_distance = self.bfs_distance_to_set(
            robot_loc_tuple,
            target_robot_locations_tuple,
            robot_obstacles_tuple
        )

        # If robot cannot reach any target location, return infinity
        if robot_approach_distance is None:
            return float('inf')

        # Total heuristic is sum of box distances + robot distance to get into position
        return box_goal_distance_sum + robot_approach_distance
