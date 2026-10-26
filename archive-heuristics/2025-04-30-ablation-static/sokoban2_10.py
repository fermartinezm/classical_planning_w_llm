from collections import deque
# Assuming Heuristic base class is available in the environment
# from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

# BFS helper function
def bfs_distance(start, targets, allowed_to_step_on, graph):
    """
    Find the shortest path distance from start to any target location
    using only locations in allowed_to_step_on for intermediate steps.

    Args:
        start (str): The starting location.
        targets (set): A set of target location strings.
        allowed_to_step_on (set): A set of location strings the path can traverse *through*.
                                  The start node itself doesn't need to be in this set.
        graph (dict): Adjacency list mapping location string to list of adjacent location strings.

    Returns:
        int or float('inf'): The shortest distance, or infinity if no path exists.
    """
    # If start is already a target, distance is 0.
    if start in targets:
        return 0

    queue = deque([(start, 0)])
    visited = {start}

    while queue:
        current_loc, dist = queue.popleft()

        # Explore neighbors from the graph
        for neighbor in graph.get(current_loc, []):
            if neighbor not in visited:
                # If the neighbor is a target, we found a path.
                if neighbor in targets:
                    return dist + 1

                # If the neighbor is not a target, it must be an intermediate step.
                # It must be in the set of locations the agent is allowed to step on.
                if neighbor in allowed_to_step_on:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))

    return float('inf') # No path found

# Define the heuristic class
class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing two components for each misplaced box:
    1. The minimum number of pushes required to move the box from its current location to its goal location (calculated as the shortest path distance on the full grid graph).
    2. The minimum number of robot moves required to reach a location adjacent to the box's current location (calculated as the shortest path distance on the graph of currently clear locations).

    # Assumptions
    - The grid structure and adjacency are defined by the 'adjacent' facts.
    - The robot can only move into 'clear' locations.
    - The heuristic assumes that moving the robot close to a box and then pushing it along a shortest path is a reasonable strategy, ignoring potential deadlocks or complex robot repositioning needed for specific pushes.
    - Action costs are uniform (cost 1 per move or push).
    - The heuristic is non-admissible and designed for greedy best-first search.

    # Heuristic Initialization
    - Extracts the goal location for each box from the task's goal conditions.
    - Builds an adjacency list graph representing the connections between all locations based on the 'adjacent' facts. This graph is used for box-to-goal distance calculation.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Initialize the total heuristic cost `total_h` to 0.
    2. Extract the current location of the robot from the state (fact `(at-robot ?l)`).
    3. Extract the current location of each box from the state (facts `(at ?b ?l)` for boxes defined in the goal). Store these in a dictionary mapping box name to location.
    4. Extract the set of currently 'clear' locations from the state (facts `(clear ?l)`). This set will be used for calculating robot movement distances.
    5. Iterate through each box and its corresponding goal location stored during initialization (`self.goal_locations`).
    6. For the current box:
       a. Get its current location from the dictionary created in step 3. If the box is not found or is already at its goal location, skip to the next box.
       b. Calculate the first component of the heuristic for this box: The minimum number of pushes required to move the box from its current location to its goal location. This is computed as the shortest path distance between the box's current location and its goal location using BFS on the *full* graph of all locations (ignoring obstacles like other boxes or the robot for this relaxed distance).
       c. Calculate the second component of the heuristic for this box: The minimum number of robot moves required to reach a location adjacent to the box's current location. This is computed as the shortest path distance from the robot's current location to *any* location adjacent to the box's current location, using BFS on the graph where only currently 'clear' locations are traversable.
       d. If the BFS for robot movement returns infinity (meaning the robot cannot reach any location adjacent to the box), the state might be a deadlock or unsolvable. In this case, return `float('inf')` immediately as the heuristic value for the state.
       e. Add the two calculated distances (box-to-goal distance and robot-to-adjacent-box distance) to `total_h`.
    7. After iterating through all goal boxes, return the final `total_h`. If all goal boxes were already at their goals, `total_h` will be 0, correctly indicating a goal state.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal locations and building the adjacency graph.
        """
        # The set of facts that must hold in goal states.
        self.goals = task.goals
        # Static facts are not affected by actions.
        static_facts = task.static

        # Store goal locations for each box.
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                box, location = args
                self.goal_locations[box] = location

        # Build the adjacency list graph from static facts.
        self.adjacency_list = {}
        # Collect all locations first
        all_locations = set()
        for fact in static_facts:
             parts = get_parts(fact)
             if parts[0] == 'adjacent':
                 l1, l2, d = parts[1:]
                 all_locations.add(l1)
                 all_locations.add(l2)

        # Initialize adjacency list for all locations found
        for loc in all_locations:
            self.adjacency_list[loc] = []

        # Populate adjacency list
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'adjacent':
                l1, l2, d = parts[1:]
                # Add directed edges for both directions if they exist in facts
                if l2 not in self.adjacency_list.get(l1, []): # Use .get for safety, though init should cover
                    self.adjacency_list[l1].append(l2)
                if l1 not in self.adjacency_list.get(l2, []):
                     self.adjacency_list[l2].append(l1)

        # Ensure any goal locations not found in adjacent facts are still in the graph structure
        # (though they won't have neighbors unless defined by adjacent facts)
        for loc in self.goal_locations.values():
             if loc not in self.adjacency_list:
                 self.adjacency_list[loc] = []


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Extract current state information
        robot_loc = None
        box_locations = {} # Map box name to location
        clear_locations = set() # Set of clear location strings

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at-robot':
                robot_loc = parts[1]
            elif parts[0] == 'at' and parts[1] in self.goal_locations: # Only track boxes we care about
                 box_locations[parts[1]] = parts[2]
            elif parts[0] == 'clear':
                 clear_locations.add(parts[1])

        total_h = 0

        # Calculate heuristic contribution for each misplaced box
        for box, goal_loc in self.goal_locations.items():
            box_loc = box_locations.get(box) # Get current location of the box

            # If box is not found in state (shouldn't happen in valid Sokoban) or already at goal, skip
            if box_loc is None or box_loc == goal_loc:
                continue

            # Component 1: Box distance to goal (minimum pushes)
            # BFS on the full graph of locations. Allowed locations are all locations in the graph.
            all_locations_set = set(self.adjacency_list.keys())
            dist_box_to_goal = bfs_distance(box_loc, {goal_loc}, all_locations_set, self.adjacency_list)

            # If box goal is unreachable on the full graph, the problem is likely unsolvable
            if dist_box_to_goal == float('inf'):
                 return float('inf')

            # Component 2: Robot distance to a location adjacent to the box
            # Find locations adjacent to the box's current location
            adjacent_to_box_locs = set(self.adjacency_list.get(box_loc, []))

            # If the box has no adjacent locations (isolated?), this shouldn't happen in valid Sokoban.
            # If it did, robot couldn't push it. Let's assume adjacent_to_box_locs is not empty for misplaced boxes.
            # If adjacent_to_box_locs is empty, dist_robot_to_adjacent_box will be inf, which is correct.

            # BFS from robot_loc to any adjacent_to_box_loc, only traversing through clear locations.
            # The set of locations the robot is allowed to step on is clear_locations.
            dist_robot_to_adjacent_box = bfs_distance(robot_loc, adjacent_to_box_locs, clear_locations, self.adjacency_list)

            # If robot cannot reach any adjacent location, it's likely a deadlock or unsolvable state
            if dist_robot_to_adjacent_box == float('inf'):
                 return float('inf') # Indicate unsolvable/deadlock

            # Add costs for this box
            total_h += dist_box_to_goal + dist_robot_to_adjacent_box

        # The heuristic is 0 if and only if all boxes are at their goal locations.
        # This is handled by the loop skipping boxes already at the goal.
        # If all boxes are at goals, the loop doesn't add anything, total_h remains 0.

        return total_h
