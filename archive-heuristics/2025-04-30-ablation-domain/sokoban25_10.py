from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque
import math

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at obj loc)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# BFS to find all-pairs shortest paths
def all_pairs_bfs(graph, locations):
    """
    Computes shortest path distances between all pairs of locations in the graph.
    Graph is an adjacency list: {loc_name: [neighbor_loc_name, ...]}
    Returns a dictionary: {(loc1, loc2): distance}
    """
    distances = {}
    for start_loc in locations:
        distances[(start_loc, start_loc)] = 0
        queue = deque([(start_loc, 0)])
        visited = {start_loc}

        while queue:
            current_loc, dist = queue.popleft()

            if current_loc in graph: # Ensure current_loc is a valid node in the graph
                for neighbor_loc in graph[current_loc]:
                    if neighbor_loc not in visited:
                        visited.add(neighbor_loc)
                        distances[(start_loc, neighbor_loc)] = dist + 1
                        queue.append((neighbor_loc, dist + 1))
    return distances


class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the number of actions required to move all boxes
    to their goal locations. It considers the shortest path distance for each
    misplaced box to its assigned goal and the distance for the robot to reach
    a position to start pushing. For multiple boxes and goals, it assumes a
    fixed pairing between boxes and goals as defined in the problem's goal state.

    # Assumptions
    - The grid structure and connectivity are defined by `adjacent` predicates.
    - All locations mentioned in `adjacent` predicates are traversable by the robot
      if clear.
    - Boxes can only be pushed, not pulled.
    - The cost of moving the robot between pushes is approximated by the distance
      to the nearest misplaced box.
    - There is a fixed one-to-one pairing between boxes and goal locations as
      specified in the problem's goal state.

    # Heuristic Initialization
    - Extracts all location objects mentioned in `adjacent` facts and builds a
      graph representing connectivity.
    - Precomputes all-pairs shortest path distances between these locations using BFS.
    - Stores the goal location for each box based on the `(at box goal_loc)`
      predicates in the task definition's goal state.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Identify the current location of the robot.
    2. Identify the current location of each box that has a corresponding goal
       defined in the task.
    3. Identify which of these boxes are not currently at their goal locations.
    4. If no such boxes are misplaced, the heuristic is 0 (goal state).
    5. Compute the estimated box-movement cost:
       - For each box that is not at its goal, find the shortest path distance
         from its current location to its *assigned* goal location (as defined
         in the task's goal state).
       - The box-movement cost is the sum of these shortest path distances
         for all misplaced boxes. If any box cannot reach its goal, the cost
         is infinite.
    6. Compute the estimated robot-movement cost:
       - Find the nearest misplaced box.
       - Calculate the shortest path distance from the robot's current location
         to the location of this nearest misplaced box. This estimates the
         initial effort for the robot to engage with a box. If the robot cannot
         reach any misplaced box, the cost is infinite.
    7. The total heuristic value is the sum of the estimated box-movement cost
       and the estimated robot-movement cost. Each unit of cost represents a
       simplified estimate of an action (either a push or a robot move).
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting graph structure and goal locations.
        """
        self.goals = task.goals
        static_facts = task.static

        # Build the graph from adjacent facts
        self.graph = {}
        locations_set = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'adjacent':
                loc1, loc2, direction = parts[1], parts[2], parts[3]
                locations_set.add(loc1)
                locations_set.add(loc2)
                if loc1 not in self.graph:
                    self.graph[loc1] = []
                if loc2 not in self.graph:
                    self.graph[loc2] = []
                # Add edges in both directions
                self.graph[loc1].append(loc2)
                self.graph[loc2].append(loc1)

        self.locations = list(locations_set)

        # Remove duplicates from adjacency lists (BFS doesn't strictly need this, but good practice)
        for loc in self.graph:
             self.graph[loc] = list(set(self.graph[loc]))

        # Precompute all-pairs shortest paths
        self.distances = all_pairs_bfs(self.graph, self.locations)

        # Store goal locations for each box
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                box, location = args
                self.goal_locations[box] = location

    def get_distance(self, loc1, loc2):
        """Helper to get precomputed distance, handling cases where locs might not be in graph."""
        # Ensure both locations are in the set of known locations from the graph
        if loc1 not in self.locations or loc2 not in self.locations:
             # This might happen if a box or robot is on a location not connected
             # by any 'adjacent' facts. Treat as unreachable.
             return float('inf')

        # Look up the precomputed distance
        # BFS guarantees finding the shortest path if one exists.
        # If no path exists, the pair won't be in self.distances (except for (loc, loc): 0)
        # Check if the key exists before accessing
        if (loc1, loc2) in self.distances:
            return self.distances[(loc1, loc2)]
        else:
            # No path found between loc1 and loc2
            return float('inf')


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Find robot location
        robot_location = None
        for fact in state:
            if match(fact, "at-robot", "*"):
                robot_location = get_parts(fact)[1]
                break

        # Find current box locations for boxes that have a goal
        current_box_locations = {}
        for fact in state:
            if match(fact, "at", "*", "*"):
                obj, loc = get_parts(fact)[1], get_parts(fact)[2]
                # Only track locations for objects that are boxes with defined goals
                if obj in self.goal_locations:
                     current_box_locations[obj] = loc

        # Identify misplaced boxes and their current locations
        misplaced_boxes_info = [] # List of (box_name, current_loc, goal_loc)
        for box, goal_loc in self.goal_locations.items():
            # A box is misplaced if it's not at its goal location OR if its current
            # location is unknown (e.g., not mentioned in 'at' facts, though this
            # shouldn't happen in valid states).
            if box not in current_box_locations or current_box_locations[box] != goal_loc:
                 # We only care about boxes whose current location is known and is not the goal
                 if box in current_box_locations:
                    misplaced_boxes_info.append((box, current_box_locations[box], goal_loc))
                 # else: The box location is unknown, treat as unreachable/infinite cost?
                 # Assuming valid states always specify box locations.

        # If all boxes are at their goals, heuristic is 0
        if not misplaced_boxes_info:
            return 0

        # --- Box Movement Cost (sum of distances to assigned goals) ---
        box_movement_cost = 0
        for box, current_loc, goal_loc in misplaced_boxes_info:
             dist = self.get_distance(current_loc, goal_loc)
             if dist == float('inf'):
                 # If a box cannot reach its goal, the state is likely a dead end.
                 return float('inf')
             box_movement_cost += dist


        # --- Robot Movement Cost ---
        # Estimate the cost for the robot to get to a position to push a box.
        # A simple estimate is the distance to the nearest misplaced box.
        robot_movement_cost = float('inf')
        # Need to ensure robot_location is valid (in the graph)
        if robot_location not in self.locations:
             # Robot is in an isolated location, cannot reach anything in the graph
             return float('inf')

        reachable_misplaced_box_locations = [
            loc for box, loc, goal in misplaced_boxes_info if loc in self.locations
        ]

        # If there are misplaced boxes, but none of them are in locations connected
        # to the graph where the robot is, the robot cannot reach any of them.
        can_robot_reach_any_box_location = False
        for box_loc in reachable_misplaced_box_locations:
            if self.get_distance(robot_location, box_loc) != float('inf'):
                can_robot_reach_any_box_location = True
                break

        if not can_robot_reach_any_box_location:
             return float('inf')

        # Find the minimum distance from the robot to any reachable misplaced box location
        for box_loc in reachable_misplaced_box_locations:
             dist = self.get_distance(robot_location, box_loc)
             robot_movement_cost = min(robot_movement_cost, dist)


        # If robot cannot reach any misplaced box, state is likely a dead end
        # This check is now redundant due to the explicit check above, but kept for clarity.
        if robot_movement_cost == float('inf'):
             return float('inf')

        # Total heuristic value
        # The sum of box distances estimates the number of pushes.
        # The robot distance estimates the initial robot movement.
        # This doesn't account for robot movement *between* pushes.
        # A simple sum is a common non-admissible heuristic.
        h_value = box_movement_cost + robot_movement_cost

        return h_value
