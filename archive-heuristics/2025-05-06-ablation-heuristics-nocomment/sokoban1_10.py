# Required imports
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque

# Helper function to parse PDDL facts
def get_parts(fact):
    """Removes parentheses and splits a PDDL fact string into parts."""
    return fact[1:-1].split()

# Helper function to match PDDL fact parts using fnmatch patterns
def match(fact, *args):
    """Checks if a PDDL fact matches a sequence of patterns."""
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# BFS function to calculate shortest path distance
def bfs_distance(start, end, adj_list, obstacles):
    """
    Calculates the shortest path distance between start and end locations
    using BFS on an undirected graph represented by adj_list, avoiding obstacles.

    Args:
        start (str): The starting location name.
        end (str): The target location name.
        adj_list (dict): Adjacency list where keys are location names and values
                         are sets of neighbor location names.
        obstacles (set): A set of location names that cannot be traversed.

    Returns:
        int or float('inf'): The shortest distance, or infinity if end is not reachable.
    """
    if start == end:
        return 0

    queue = deque([(start, 0)])
    visited = {start}

    while queue:
        current_loc, dist = queue.popleft()

        # Get neighbors from the set
        neighbors = adj_list.get(current_loc, set())

        for neighbor_loc in neighbors:
            if neighbor_loc == end:
                return dist + 1
            # Check if the neighbor is an obstacle
            if neighbor_loc in obstacles:
                 continue # Cannot traverse through this location

            if neighbor_loc not in visited:
                visited.add(neighbor_loc)
                queue.append((neighbor_loc, dist + 1))

    return float('inf') # Not reachable


class sokobanHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Sokoban domain.

    Summary:
        Estimates the cost to reach the goal state by summing the minimum number
        of pushes required for each misplaced box to reach its goal location
        (calculated considering other boxes as obstacles) and the minimum number
        of robot moves required to reach the location of that box (calculated
        while avoiding other boxes).

    Assumptions:
        - Locations are named in a way that allows building an adjacency graph
          from `adjacent` facts.
        - The grid is connected such that BFS is meaningful.
        - The heuristic simplifies the push mechanic: it assumes a box can be
          pushed along any path on the grid not blocked by *other* boxes.
          It doesn't check for complex deadlocks or required intermediate
          robot positions beyond reaching the box's location.
        - The robot can reach any location occupied by a box it wants to push,
          avoiding other boxes.
        - The heuristic is non-admissible.

    Heuristic Initialization:
        - Parses the goal facts to create a mapping from each box to its goal location.
        - Parses the static `adjacent` facts to build an adjacency list representation
          of the location graph. This graph is treated as undirected for BFS.

    Step-By-Step Thinking for Computing Heuristic:
        1. Check if the current state is the goal state using `self.goals <= state`. If yes, return 0.
        2. Identify the current location of the robot from the state facts. If the robot location is not found, return infinity (or a large penalty).
        3. Identify the current location for each box that has a goal. Create a set of locations currently occupied by *all* boxes found in the state.
        4. Initialize the total heuristic value to 0. Define a large penalty value (e.g., 1000000) for unreachable situations.
        5. Iterate through each box that has a defined goal location (based on the goals parsed during initialization).
        6. For the current box, get its current location from the locations found in step 3, and its goal location from the initialized goal mapping. If the box's current location is not found in the state (e.g., box missing), return infinity (or penalty).
        7. If the box is already at its goal location, its contribution to the heuristic is 0. Continue to the next box.
        8. If the box is not at its goal location:
            a. Determine the set of obstacles for the box's path: these are the locations occupied by *other* boxes (excluding the current box's location).
            b. Calculate the minimum number of pushes required for this box to reach its goal location. This is estimated as the shortest path distance between the box's current location and its goal location using BFS on the location graph, avoiding the obstacles identified in step 8a. Let this be `box_distance`.
            c. If `box_distance` is infinite, it means the box goal is unreachable with the current configuration of other boxes. Assign the large penalty value to the total heuristic and return it immediately, as the state is likely unsolvable.
            d. Determine the set of obstacles for the robot's path: these are also the locations occupied by *other* boxes (excluding the current box's location). The robot can move to the square occupied by the box it wants to push.
            e. Calculate the minimum number of moves required for the robot to reach the box's current location. This is estimated as the shortest path distance between the robot's current location and the box's current location using BFS on the location graph, avoiding the obstacles identified in step 8d. Let this be `robot_distance`.
            f. If `robot_distance` is infinite, it means the robot cannot reach the box. Assign the large penalty value to the total heuristic and return it immediately.
            g. Otherwise, the cost for this misplaced box is `box_distance + robot_distance`. Add this cost to the total heuristic value.
        9. After iterating through all misplaced boxes, if the loop completed without returning a penalty, the total heuristic value is the sum of costs.
        10. Return the calculated `total_heuristic`. (Note: Step 1 already handled the case where total_heuristic would be 0).

    """
    def __init__(self, task):
        self.goals = task.goals
        static_facts = task.static

        self.goal_locations = {}
        # Parse goals to find box goal locations
        for goal in self.goals:
            # Goals are typically (at boxX loc_Y_Z)
            parts = get_parts(goal)
            if parts[0] == 'at' and len(parts) == 3:
                box_name = parts[1]
                location_name = parts[2]
                self.goal_locations[box_name] = location_name

        # Adjacency list for undirected graph: loc -> set of neighbor_loc
        self.adj_list = {}
        self.locations = set()
        # Parse adjacent facts to build the graph
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'adjacent' and len(parts) == 4:
                loc1, loc2, direction = parts[1], parts[2], parts[3]
                self.locations.add(loc1)
                self.locations.add(loc2)
                if loc1 not in self.adj_list:
                    self.adj_list[loc1] = set()
                if loc2 not in self.adj_list:
                    self.adj_list[loc2] = set()
                self.adj_list[loc1].add(loc2)
                self.adj_list[loc2].add(loc1) # Add reverse edge assuming symmetry

    def __call__(self, node):
        state = node.state

        # Check if goal is reached first for efficiency and correctness (h=0 iff goal)
        if self.goals <= state:
             return 0

        # Find robot location
        robot_location = None
        for fact in state:
            if match(fact, "at-robot", "*"):
                robot_location = get_parts(fact)[1]
                break
        if robot_location is None:
             # Should not happen in a valid Sokoban state
             return float('inf') # Robot location unknown

        # Find current box locations
        current_box_locations = {}
        locations_of_all_boxes = set()
        for fact in state:
            if match(fact, "at", "*", "*"):
                parts = get_parts(fact)
                box_name = parts[1]
                location_name = parts[2]
                current_box_locations[box_name] = location_name
                locations_of_all_boxes.add(location_name)

        total_heuristic = 0
        large_penalty = 1000000 # Use a large number for unreachable states

        # Calculate heuristic for each box
        for box_name, goal_location in self.goal_locations.items():
            current_location = current_box_locations.get(box_name)

            if current_location is None:
                 # Box is missing from state facts - problem?
                 return float('inf') # Indicate problem

            if current_location == goal_location:
                continue # Box is already at goal

            # Box is misplaced
            # Obstacles for BFS for this box are locations of *other* boxes
            obstacles_for_box_path = locations_of_all_boxes - {current_location}

            # 1. Cost for the box to reach its goal (BFS avoiding other boxes)
            box_distance = bfs_distance(current_location, goal_location, self.adj_list, obstacles=obstacles_for_box_path)

            if box_distance == float('inf'):
                 # Box goal is unreachable with current box configuration
                 return large_penalty # Indicate unsolvable/deadlock

            # 2. Cost for the robot to reach the box (BFS avoiding other boxes)
            # Obstacles for robot BFS are locations of *other* boxes
            obstacles_for_robot_path = locations_of_all_boxes - {current_location}

            robot_distance = bfs_distance(robot_location, current_location, self.adj_list, obstacles=obstacles_for_robot_path)

            if robot_distance == float('inf'):
                 # Robot cannot reach the box
                 return large_penalty # Indicate unsolvable/deadlock

            # Total cost for this box: box_distance (pushes) + robot_distance (moves)
            # This is a simple sum, non-admissible.
            total_heuristic += box_distance + robot_distance

        # If we reach here, it's not a goal state (checked at the start)
        # and calculation was finite for all boxes.
        # The total_heuristic will be > 0 because there's at least one misplaced box.
        # If any penalty was applied during the loop, we would have returned already.
        return total_heuristic
