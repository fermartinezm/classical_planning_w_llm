from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque
import math

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
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def reverse_direction(direction):
    """Returns the opposite direction."""
    if direction == 'down': return 'up'
    if direction == 'up': return 'down'
    if direction == 'left': return 'right'
    if direction == 'right': return 'left'
    return None # Should not happen for valid directions

class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal state.
    It is calculated as the sum of the shortest path distances for each misplaced box
    to its goal location (on the static graph of locations), plus the shortest
    distance for the robot to reach any location adjacent to any misplaced box
    (on the graph of currently clear locations).

    # Assumptions
    - The location graph is defined by `adjacent` facts and is static.
    - Boxes can only be pushed one step at a time into a clear adjacent location.
    - The robot can only move into a location `l` if the fact `(clear l)` is true in the state.
    - The heuristic does not consider complex interactions like moving obstacles
      or potential deadlocks (e.g., pushing a box into a corner).

    # Heuristic Initialization
    - Parses goal conditions to map each box to its goal location.
    - Builds the static graph of locations and their adjacencies based on
      `adjacent` facts. Stores directions for edges.
    - Computes all-pairs shortest path distances on this static graph using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current location of the robot and all boxes.
    2. Identify the set of locations `L_clear` for which the fact `(clear l)` is true in the state.
    3. Compute the shortest path distances for the robot from its current location
       to all other locations, considering only paths through locations in `L_clear`.
       This is done using BFS on the dynamic graph restricted to `L_clear`.
    4. Initialize the heuristic value `h` to 0.
    5. Initialize `min_robot_dist_to_neighbor_of_box` to infinity.
    6. Iterate through each box that has a goal location:
       - If the box is not at its goal location:
         - Add the precomputed static shortest path distance from the box's current
           location to its goal location to `h`. If the goal is statically unreachable,
           return infinity.
         - For each neighbor `neighbor_of_b` of the box's current location (on the static graph):
           - If the robot can reach this neighbor location via a path through locations in `L_clear`
             (checked using the robot BFS results):
             - Update `min_robot_dist_to_neighbor_of_box` with the minimum
               distance found so far.
    7. If there are misplaced boxes and `min_robot_dist_to_neighbor_of_box` is still
       infinity (robot cannot reach any neighbor of any misplaced box on a clear path),
       return infinity.
    8. Otherwise, add `min_robot_dist_to_neighbor_of_box` to `h`.
    9. Return `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and building
        the static location graph and distances.
        """
        self.goals = task.goals
        static_facts = task.static

        # Store goal locations for each box.
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                box, location = args # Correct order: (at ?o - box ?l - location)
                self.goal_locations[box] = location

        # Build the static graph of locations with directions.
        self.graph = {}
        self.all_locations = set()
        for fact in static_facts:
            if match(fact, "adjacent", "*", "*", "*"):
                _, loc1, loc2, direction = get_parts(fact)
                self.all_locations.add(loc1)
                self.all_locations.add(loc2)
                if loc1 not in self.graph:
                    self.graph[loc1] = {}
                if loc2 not in self.graph:
                    self.graph[loc2] = {}
                self.graph[loc1][loc2] = direction
                # Assuming adjacency is symmetric, add the reverse edge
                self.graph[loc2][loc1] = reverse_direction(direction)

        # Compute all-pairs shortest path distances on the static graph using BFS.
        self.distances = {}
        for start_loc in self.all_locations:
            self.distances[start_loc] = {}
            q = deque([(start_loc, 0)])
            visited = {start_loc}
            self.distances[start_loc][start_loc] = 0

            while q:
                current_loc, d = q.popleft()

                # Check if current_loc is in graph (should be if in all_locations)
                if current_loc not in self.graph:
                    continue

                for neighbor in self.graph[current_loc]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        self.distances[start_loc][neighbor] = d + 1
                        q.append((neighbor, d + 1))

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Get robot location.
        robot_loc = None
        for fact in state:
            if match(fact, "at-robot", "*"):
                robot_loc = get_parts(fact)[1]
                break
        if robot_loc is None:
             # Should not happen in a valid Sokoban state, but handle defensively
             return math.inf

        # Get current box locations.
        box_locs = {}
        for fact in state:
            if match(fact, "at", "*", "*"):
                predicate, box, loc = get_parts(fact)
                if box in self.goal_locations: # Only track boxes relevant to goals
                    box_locs[box] = loc

        # Identify locations for which the (clear l) fact is true in the state.
        # The robot can move to any of these locations.
        clear_facts_locs = {get_parts(fact)[1] for fact in state if match(fact, "clear", "*")}

        # Compute robot distances from robot_loc on the graph restricted to clear_facts_locs.
        # The robot's current location is the starting point, even if it's not "clear" for a box.
        robot_distances_from_r = {}
        q = deque([(robot_loc, 0)])
        visited_robot = {robot_loc}
        robot_distances_from_r[robot_loc] = 0

        while q:
            curr, d = q.popleft()

            if curr not in self.graph: # Should not happen if all_locations is correct
                 continue

            for neighbor in self.graph[curr]: # Iterate through static adjacencies
                # Robot can move to neighbor if neighbor has the (clear neighbor) fact in state.
                if neighbor in clear_facts_locs and neighbor not in visited_robot:
                     visited_robot.add(neighbor)
                     robot_distances_from_r[neighbor] = d + 1
                     q.append((neighbor, d + 1))

        h = 0
        min_robot_dist_to_neighbor_of_box = math.inf
        misplaced_boxes_exist = False

        # Calculate box-to-goal distances and find minimum robot distance to a box neighbor.
        for box, goal_loc in self.goal_locations.items():
            current_loc = box_locs.get(box) # Get box's current location

            # If box location is unknown (shouldn't happen) or box is already at goal, skip.
            if current_loc is None or current_loc == goal_loc:
                continue

            misplaced_boxes_exist = True

            # Add box-to-goal distance (minimum pushes) using static graph distances.
            # Check if the goal is statically reachable from the box's current location.
            if current_loc not in self.distances or goal_loc not in self.distances[current_loc]:
                 # Goal is unreachable for this box on the static graph. Problem unsolvable.
                 return math.inf
            h += self.distances[current_loc][goal_loc]

            # Find minimum robot distance to any neighbor of this misplaced box.
            # Iterate through all neighbors of the box's current location on the static graph.
            if current_loc in self.graph:
                for neighbor_of_b in self.graph[current_loc]:
                    # Check if the neighbor location is reachable by the robot on a clear path.
                    # The neighbor must be in the set of locations the robot can move to.
                    if neighbor_of_b in robot_distances_from_r:
                         robot_dist = robot_distances_from_r[neighbor_of_b]
                         min_robot_dist_to_neighbor_of_box = min(min_robot_dist_to_neighbor_of_box, robot_dist)

        # If no boxes are misplaced, the heuristic is 0.
        if not misplaced_boxes_exist:
            return 0

        # If there are misplaced boxes but the robot cannot reach any neighbor
        # of any misplaced box on a clear path, the state is likely a dead end.
        if min_robot_dist_to_neighbor_of_box == math.inf:
            return math.inf

        # Add the minimum robot distance to the heuristic.
        h += min_robot_dist_to_neighbor_of_box

        return h
