import collections
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

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
    # Ensure the number of parts matches the number of arguments in the pattern
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def bfs(graph, start):
    """
    Performs BFS to find shortest distances from a start node to all reachable nodes.
    Returns a dictionary {node: distance}.
    """
    distances = {start: 0}
    queue = collections.deque([start])
    while queue:
        current = queue.popleft()
        # Check if current node exists in the graph keys before iterating neighbors
        if current in graph:
            for neighbor in graph[current]:
                if neighbor not in distances:
                    distances[neighbor] = distances[current] + 1
                    queue.append(neighbor)
    return distances

class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal state.
    It calculates the sum of shortest path distances for each box from its current
    location to its goal location (estimating the number of pushes needed).
    Additionally, it adds the shortest path distance from the robot's current
    location to the location of the box that is currently closest to the robot
    among those not yet at their goal. This estimates the robot's travel cost
    to start interacting with a box.

    # Assumptions
    - Each box has a unique goal location specified in the task goals.
    - The grid connectivity is defined by the `adjacent` facts, and movement
      is possible between any two locations connected by an `adjacent` fact
      (assuming symmetry).
    - The cost of a robot move action is 1.
    - The cost of a push action (which moves the box one step and the robot
      into the box's previous location) is 1.
    - This heuristic is non-admissible; it does not guarantee a lower bound
      on the true cost. It simplifies robot movement costs (only considers
      reaching the first box) and does not detect complex deadlocks (e.g.,
      boxes pushed into corners where they cannot be moved further towards
      the goal, unless the goal itself is unreachable on the grid).

    # Heuristic Initialization
    - Parses `adjacent` facts from the static information to build an undirected
      graph representing the grid connectivity.
    - Precomputes shortest path distances between all pairs of locations using
      Breadth-First Search (BFS) on the constructed graph.
    - Extracts and stores the goal location for each box from the task goals.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current location of the robot.
    2. Identify the current location of each box that has a goal location.
    3. Initialize the total heuristic cost to 0.
    4. Create a list of boxes that are not currently at their respective goal locations.
    5. Iterate through the list of boxes not at their goal:
       - For each such box, retrieve its current location and its goal location.
       - Calculate the shortest path distance between the box's current location
         and its goal location using the precomputed distances. This distance
         represents the minimum number of push actions required for this box.
       - Add this distance to the total heuristic cost.
       - If the goal location is unreachable from the box's current location
         (distance is infinity), the state is likely a deadlock or unsolvable;
         return infinity.
    6. If there are any boxes not at their goal locations:
       - Find the box among these that is located closest to the robot's current
         location, using the precomputed distances.
       - Calculate the shortest path distance from the robot's current location
         to the location of this closest box. This estimates the robot's initial
         movement cost to reach a box it needs to push.
       - Add this distance to the total heuristic cost.
       - If the closest box is unreachable by the robot (distance is infinity),
         return infinity.
    7. Return the calculated total heuristic cost. If no boxes need moving, the
       cost is 0 (as calculated in step 3 and not increased in steps 5 or 6).
    """

    def __init__(self, task):
        """
        Initialize the heuristic by building the graph, precomputing distances,
        and storing goal locations.
        """
        self.goals = task.goals
        static_facts = task.static

        # Build the graph from adjacent facts
        self.graph = collections.defaultdict(list)
        locations = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'adjacent':
                loc1, loc2 = parts[1], parts[2]
                # Assuming adjacency is symmetric for robot movement
                self.graph[loc1].append(loc2)
                self.graph[loc2].append(loc1)
                locations.add(loc1)
                locations.add(loc2)

        # Ensure all locations mentioned in adjacent facts are in the graph keys
        # even if they have no outgoing adjacent facts listed.
        for loc in locations:
             if loc not in self.graph:
                 self.graph[loc] = []

        # Precompute all-pairs shortest paths using BFS from each location
        self.all_pairs_distances = {}
        # Only compute for locations that are part of the graph (connected locations)
        for start_node in self.graph:
            self.all_pairs_distances[start_node] = bfs(self.graph, start_node)

        # Store goal locations for each box
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                box, location = args
                self.goal_locations[box] = location

    def get_distance(self, loc1, loc2):
        """
        Helper to get precomputed distance between two locations.
        Returns float('inf') if loc1 or loc2 are not in the precomputed distances
        (e.g., isolated locations or invalid state facts) or if unreachable.
        """
        if loc1 not in self.all_pairs_distances or loc2 not in self.all_pairs_distances.get(loc1, {}):
             # This can happen if a location in the state is not part of the
             # connected graph defined by adjacent facts. Treat as unreachable.
             return float('inf')
        return self.all_pairs_distances[loc1][loc2]

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions
        to reach the goal state.
        """
        state = node.state

        # Find current robot location
        robot_location = None
        for fact in state:
            if match(fact, "at-robot", "*"):
                robot_location = get_parts(fact)[1]
                break

        # If robot location is not found, the state is invalid/unreachable
        if robot_location is None or robot_location not in self.graph:
             return float('inf')

        # Find current box locations for boxes relevant to the goal
        box_locations = {}
        for fact in state:
            if match(fact, "at", "*", "*"):
                 obj, loc = get_parts(fact)[1], get_parts(fact)[2]
                 # Only consider objects that are boxes and have a goal
                 if obj in self.goal_locations:
                     box_locations[obj] = loc

        total_cost = 0
        boxes_not_at_goal = []

        # Calculate cost component for each box not at its goal
        for box, goal_loc in self.goal_locations.items():
            current_loc = box_locations.get(box)

            # If a box is missing from the state or its location is not in the graph,
            # it's an invalid/unreachable state.
            if current_loc is None or current_loc not in self.graph:
                 return float('inf')

            if current_loc != goal_loc:
                # Add distance from box to goal (minimum pushes required)
                box_dist = self.get_distance(current_loc, goal_loc)
                if box_dist == float('inf'):
                    # Box cannot reach goal location on the grid. Deadlock.
                    return float('inf')
                total_cost += box_dist
                boxes_not_at_goal.append(box)

        # If all boxes are at goal, heuristic is 0
        if not boxes_not_at_goal:
            return 0

        # Add robot movement cost to reach the closest box that needs moving
        min_robot_to_box_dist = float('inf')

        for box in boxes_not_at_goal:
            box_loc = box_locations[box]
            dist = self.get_distance(robot_location, box_loc)
            min_robot_to_box_dist = min(min_robot_to_box_dist, dist)

        # If the robot cannot reach any box that needs moving, it's a deadlock.
        if min_robot_to_box_dist == float('inf'):
             return float('inf')

        total_cost += min_robot_to_box_dist

        # Return the estimated cost to goal state.
        return total_cost

