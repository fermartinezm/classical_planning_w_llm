from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

# Helper function to match PDDL facts
def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at ball1 rooma)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# BFS function for shortest path on the location graph
def bfs_distance(graph, start_loc, end_loc):
    """
    Calculates the shortest path distance between two locations using BFS.
    Assumes graph keys and values are location names (strings).
    Returns float('inf') if end_loc is unreachable from start_loc.
    """
    if start_loc == end_loc:
        return 0
    # Check if start or end locations are not in the graph (isolated or invalid)
    if start_loc not in graph or end_loc not in graph:
        return float('inf')

    queue = deque([(start_loc, 0)])
    visited = {start_loc}
    while queue:
        current_loc, dist = queue.popleft()

        # current_loc is guaranteed to be in graph keys by the initial check

        for neighbor_loc in graph[current_loc]:
            if neighbor_loc == end_loc:
                return dist + 1
            if neighbor_loc not in visited:
                visited.add(neighbor_loc)
                queue.append((neighbor_loc, dist + 1))
    # If end_loc is unreachable from start_loc
    return float('inf')

class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal state.
    It is calculated as the sum of shortest path distances for each box to its goal
    location, plus the shortest path distance from the robot to the closest box
    that needs to be moved.

    # Assumptions
    - The problem is defined on a grid graph where connections are given by
      `adjacent` predicates.
    - The cost of moving the robot one step is 1.
    - The cost of pushing a box one step is approximately 1 (ignoring the robot
      positioning cost for subsequent pushes).
    - The heuristic ignores potential blockages by other boxes or walls
      during path calculation for both the robot and the boxes.
    - The heuristic assumes any box can be pushed to its goal if a path exists
      on the grid graph, ignoring the robot's specific positioning requirement
      relative to the box for initiating a push.
    - All objects appearing in `(at ?obj ?loc)` goal facts are boxes.

    # Heuristic Initialization
    - Builds a graph representation of the locations and their adjacencies
      from the `adjacent` static facts.
    - Extracts the goal location for each object that is required to be at a
      specific location in the goal state. These objects are assumed to be boxes.

    # Step-By-Step Thinking for Computing Heuristic
    1. Parse the current state to find the robot's current location and the
       current location of every object that is a box (i.e., has a goal location).
    2. Identify the set of boxes that are not currently at their respective
       goal locations.
    3. If this set of boxes is empty, the current state is a goal state, and
       the heuristic value is 0.
    4. If there are boxes not at their goals:
       a. Initialize a sum for box distances to 0.
       b. For each box that needs to be moved, calculate the shortest path
          distance from its current location to its goal location using BFS
          on the pre-computed location graph. Add this distance to the sum.
          This sum estimates the minimum number of push actions required for
          all boxes combined, ignoring obstacles and robot positioning.
          If any box's goal is unreachable on the graph, the problem is
          unsolvable from this state, and the heuristic is infinity.
       c. Initialize a minimum robot distance to infinity.
       d. Find the robot's current location. If the robot's location is unknown
          or unreachable, the problem is unsolvable, and the heuristic is infinity.
       e. For each box that needs to be moved, calculate the shortest path
          distance from the robot's current location to the box's current
          location using BFS on the location graph. Update the minimum robot
          distance found so far. This estimates the minimum robot movement
          needed to reach *any* box that requires moving. If any box is
          unreachable by the robot, the problem is unsolvable, and the
          heuristic is infinity.
       f. The total heuristic value is the sum of the total box distance
          (from step 4b) and the minimum robot distance (from step 4e).
    """

    def __init__(self, task):
        """
        Initialize the heuristic by building the location graph and extracting
        box goal locations.
        """
        self.goals = task.goals
        static_facts = task.static

        # Build the location graph from adjacent facts
        self.location_graph = {}
        # Collect all locations mentioned in adjacent facts first
        all_locations = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "adjacent":
                loc1, loc2, direction = parts[1:]
                all_locations.add(loc1)
                all_locations.add(loc2)

        # Initialize graph with all found locations
        self.location_graph = {loc: [] for loc in all_locations}

        # Populate adjacency lists
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "adjacent":
                loc1, loc2, direction = parts[1:]
                # Add directed edge; BFS on this graph finds path length
                # The PDDL defines symmetric adjacent facts, so this builds an undirected graph representation
                self.location_graph[loc1].append(loc2)


        # Store goal locations for each object that appears in an 'at' goal
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at" and len(args) == 2:
                obj, location = args
                self.goal_locations[obj] = location

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state (frozenset of fact strings).

        # Find robot location
        robot_location = None
        for fact in state:
            if match(fact, "at-robot", "*"):
                robot_location = get_parts(fact)[1]
                break

        # If robot location is not found, the state is likely invalid or unsolvable
        if robot_location is None or robot_location not in self.location_graph:
             return float('inf')

        # Find current box locations (only for objects that have a goal)
        current_box_locations = {}
        for fact in state:
            if match(fact, "at", "*", "*"):
                 obj, loc = get_parts(fact)[1:]
                 if obj in self.goal_locations: # Only track objects that are goals (assumed to be boxes)
                    current_box_locations[obj] = loc

        # Identify boxes not at their goal
        boxes_to_move = {
            box for box, loc in current_box_locations.items()
            if box in self.goal_locations and loc != self.goal_locations[box]
        }

        # If all boxes are at their goal, heuristic is 0
        if not boxes_to_move:
            return 0

        # Calculate sum of box distances to goals
        box_distance_sum = 0
        for box in boxes_to_move:
            current_loc = current_box_locations.get(box) # Use .get for safety
            goal_loc = self.goal_locations[box]

            # If a box location is not found in the state or not in the graph, it's unsolvable
            if current_loc is None or current_loc not in self.location_graph:
                 return float('inf')

            dist = bfs_distance(self.location_graph, current_loc, goal_loc)
            # If a box's goal is unreachable, the problem is unsolvable from this state
            if dist == float('inf'):
                 return float('inf')
            box_distance_sum += dist

        # Calculate robot distance to the closest box that needs moving
        robot_distance_to_closest_box = float('inf')
        for box in boxes_to_move:
            box_loc = current_box_locations[box] # Location is guaranteed to be in graph by check above
            dist = bfs_distance(self.location_graph, robot_location, box_loc)
            # If any box is unreachable by the robot, the problem is unsolvable
            if dist == float('inf'):
                 return float('inf')
            robot_distance_to_closest_box = min(robot_distance_to_closest_box, dist)

        # Total heuristic is sum of box distances and robot distance to closest box
        # This estimates the total pushes needed + the cost to get the robot to the first push.
        return box_distance_sum + robot_distance_to_closest_box
