from collections import deque

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

# BFS function to compute distances from a start node
def bfs_distances(start_node, graph):
    """
    Computes shortest path distances from a start_node to all other nodes
    in the graph using Breadth-First Search.
    """
    # Initialize distances to infinity for all nodes in the graph
    distances = {node: float('inf') for node in graph}

    # If the start_node is not even in the graph keys (e.g., an isolated location
    # mentioned in init/goal but not adjacent facts), add it.
    if start_node not in distances:
         distances[start_node] = float('inf')

    distances[start_node] = 0
    queue = deque([start_node])

    while queue:
        current_node = queue.popleft()

        # Check if current_node has neighbors in the graph
        if current_node in graph:
            for neighbor in graph[current_node]:
                if distances[neighbor] == float('inf'): # If not visited
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)
    return distances

# Define the heuristic class.
# It should have an __init__ method that takes a task object
# and a __call__ method that takes a node object.
class sokobanHeuristic:
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal state by summing two components:
    1. The sum of shortest path distances for each box from its current location to its goal location.
    2. The shortest path distance from the robot's current location to the nearest box that is not yet at its goal.

    # Assumptions
    - The grid structure and valid movements are defined solely by the `adjacent` predicates.
    - Shortest path distances on this graph are a reasonable estimate for the minimum number of pushes (for boxes) and moves (for the robot).
    - Ignoring potential deadlocks (states where a box is trapped).
    - Ignoring the specific orientation/pushing position required for the robot, using distance to the box location as a proxy.

    # Heuristic Initialization
    - Parses the static facts (`adjacent` predicates) to build a graph representation of the locations.
    - Precomputes all-pairs shortest path distances between all locations using BFS.
    - Extracts the goal locations for each box from the task's goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the robot's current location.
    2. Identify the current location of each box.
    3. Initialize `sum_box_distances` to 0 and `min_robot_to_box_distance` to infinity.
    4. Iterate through each box and its goal location defined in the task goals:
       a. If the box is already at its goal location, skip it.
       b. If the box is not at its goal:
          i. Add the shortest path distance from the box's current location to its goal location to `sum_box_distances`. If the goal is unreachable, the heuristic is infinity.
          ii. Calculate the shortest path distance from the robot's current location to the box's current location. If the robot is unreachable by the box's location (or vice versa), the heuristic is infinity.
          iii. Update `min_robot_to_box_distance` with the minimum distance found so far from the robot to any box not at its goal.
    5. If all boxes are at their goals, the heuristic is 0.
    6. Otherwise, the heuristic value is `sum_box_distances + min_robot_to_box_distance`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by building the location graph, precomputing
        distances, and storing goal locations.
        """
        self.goals = task.goals
        static_facts = task.static

        # Build the location graph from adjacent facts
        self.graph = {}
        self.all_locations = set()

        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'adjacent':
                l1, l2, direction = parts[1], parts[2], parts[3]
                if l1 not in self.graph:
                    self.graph[l1] = []
                if l2 not in self.graph:
                    self.graph[l2] = []
                self.graph[l1].append(l2)
                self.graph[l2].append(l1) # Assuming adjacency is symmetric
                self.all_locations.add(l1)
                self.all_locations.add(l2)

        # Add any locations mentioned in goals or initial state that might not have adjacencies
        for fact in task.initial_state:
             parts = get_parts(fact)
             if parts[0] in ['at-robot', 'at']:
                 # parts[1] is robot/box name, parts[2] is location for 'at'
                 # parts[1] is location for 'at-robot'
                 loc = parts[1] if parts[0] == 'at-robot' else parts[2]
                 self.all_locations.add(loc)
        for goal in task.goals:
             parts = get_parts(goal)
             if parts[0] == 'at':
                 box, loc = parts[1], parts[2]
                 self.all_locations.add(loc)

        # Ensure all locations from all_locations are keys in the graph, even if they have no neighbors
        # This is important so BFS can compute distance 0 from a node to itself even if isolated.
        for loc in self.all_locations:
             if loc not in self.graph:
                 self.graph[loc] = []

        # Precompute all-pairs shortest path distances
        self.distances = {}
        for start_loc in self.all_locations:
             self.distances[start_loc] = bfs_distances(start_loc, self.graph)

        # Store goal locations for each box
        self.goal_locations = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "at":
                box, location = parts[1], parts[2]
                self.goal_locations[box] = location

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Find robot location
        robot_location = None
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at-robot':
                robot_location = parts[1]
                break

        # Find box locations
        current_box_locations = {}
        for fact in state:
            parts = get_parts(fact)
            # Check if the fact is an 'at' predicate for a box that has a goal
            if parts[0] == 'at' and parts[1] in self.goal_locations:
                box = parts[1]
                location = parts[2]
                current_box_locations[box] = location

        sum_box_distances = 0
        min_robot_to_box_distance = float('inf')
        any_box_not_at_goal = False

        # Iterate through each box that has a goal defined
        for box, goal_location in self.goal_locations.items():
            current_location = current_box_locations.get(box)

            # If the box's current location is not found in the state
            # (shouldn't happen in valid Sokoban states but check for robustness)
            # or if it's already at the goal
            if current_location is None or current_location == goal_location:
                continue

            any_box_not_at_goal = True

            # Distance from box to its goal
            # Use .get() defensively, although all relevant locations should be in self.distances
            box_to_goal_dist = self.distances.get(current_location, {}).get(goal_location, float('inf'))

            # If box cannot reach goal, return infinity (unsolvable state)
            if box_to_goal_dist == float('inf'):
                return float('inf')

            sum_box_distances += box_to_goal_dist

            # Distance from robot to the box
            # Use .get() defensively
            robot_to_box_dist = self.distances.get(robot_location, {}).get(current_location, float('inf'))

            # If robot cannot reach the box, return infinity (unsolvable state)
            if robot_to_box_dist == float('inf'):
                 return float('inf')

            min_robot_to_box_distance = min(min_robot_to_box_distance, robot_to_box_dist)

        # If all boxes are at their goals, the heuristic is 0.
        if not any_box_not_at_goal:
            return 0

        # If there were boxes not at goals, min_robot_to_box_distance must have been updated
        # to a finite value unless the robot cannot reach *any* box not at its goal,
        # which would have resulted in float('inf') being returned earlier.
        # This check is a final safeguard.
        if min_robot_to_box_distance == float('inf'):
             # This implies any_box_not_at_goal is True, but robot_to_box_dist was inf for all such boxes.
             return float('inf')


        return sum_box_distances + min_robot_to_box_distance
