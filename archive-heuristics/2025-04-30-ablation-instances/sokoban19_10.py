from fnmatch import fnmatch
from collections import deque
import math # Import math for infinity

# Assuming Heuristic base class is available from heuristics.heuristic_base
# If not, a dummy class would be needed for standalone execution, but the problem
# implies it will be used within a framework providing this base class.
# from heuristics.heuristic_base import Heuristic

# Define a dummy Heuristic base class if running standalone for testing
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    class Heuristic:
        def __init__(self, task):
            pass
        def __call__(self, node):
            pass


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and handle potential empty facts or malformed strings gracefully
    if not isinstance(fact, str) or len(fact) < 2:
        return []
    # Remove surrounding parentheses and split by whitespace
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at box1 loc_1_1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Check if the number of parts matches the number of arguments in the pattern
    if len(parts) != len(args):
        return False
    # Use fnmatch for pattern matching on each part
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the number of actions required to solve the Sokoban puzzle by summing the shortest path distances of each misplaced box to its goal location and the shortest path distance of the robot to the nearest misplaced box. The shortest paths are computed on the empty grid defined by the `adjacent` facts.

    # Assumptions
    - Each box has a unique goal location specified in the task goals.
    - The grid structure and connectivity are static and defined by `adjacent` facts.
    - Shortest path distances are computed on the empty grid, ignoring the current positions of boxes and the robot as temporary obstacles. This is a relaxation.
    - The cost of moving a box one step is at least 1 (one push action).
    - The cost for the robot to get into position to push a box is approximated by the shortest path distance to the box's location.

    # Heuristic Initialization
    - Extract the goal locations for each box from the task goals.
    - Build an undirected graph representing the grid connectivity from the `adjacent` static facts.
    - Compute all-pairs shortest path distances on this grid graph using Breadth-First Search (BFS). These distances represent the minimum number of moves between any two locations on the empty grid.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current location of the robot by finding the fact `(at-robot ?l)`.
    2. Identify the current location of each box by finding facts `(at ?b ?l)` for all boxes `?b` defined in the problem.
    3. Determine which boxes are not currently at their assigned goal locations. These are the "misplaced boxes".
    4. If the set of misplaced boxes is empty, the state is a goal state, and the heuristic value is 0.
    5. If there are misplaced boxes:
       a. Calculate the sum of the shortest path distances for each misplaced box. For each misplaced box `b`, find the precomputed shortest path distance from its current location to its goal location. Sum these distances. This estimates the total number of push actions needed for all boxes, ignoring robot movement and obstacles.
       b. Find the misplaced box that is nearest to the robot. Calculate the shortest path distance from the robot's current location to the location of each misplaced box using the precomputed distances. Find the minimum of these distances. This estimates the cost for the robot to reach a box it needs to move.
       c. The total heuristic value is the sum of the total box-to-goal distance (from step 5a) and the minimum robot-to-nearest-misplaced-box distance (from step 5b). If any required distance lookup fails (e.g., location not in graph), return infinity, indicating a potentially unreachable or deadlocked state.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and building
        the location graph for shortest path computations.
        """
        self.goals = task.goals
        static_facts = task.static

        # Store goal locations for each box.
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at" and len(args) == 2:
                box, location = args
                self.goal_locations[box] = location

        # Build the location graph from adjacent facts.
        self.location_graph = {}
        all_locations = set()
        for fact in static_facts:
            if match(fact, "adjacent", "*", "*", "*"):
                _, loc1, loc2, _ = get_parts(fact)
                all_locations.add(loc1)
                all_locations.add(loc2)
                # Add bidirectional edges
                if loc1 not in self.location_graph:
                    self.location_graph[loc1] = []
                if loc2 not in self.location_graph:
                    self.location_graph[loc2] = []
                self.location_graph[loc1].append(loc2)
                self.location_graph[loc2].append(loc1)

        # Remove duplicates from adjacency lists (if any)
        for loc in self.location_graph:
             self.location_graph[loc] = list(set(self.location_graph[loc]))

        # Compute all-pairs shortest paths using BFS.
        self.shortest_paths = {}
        for start_node in all_locations:
            self.shortest_paths[start_node] = self._bfs(start_node, all_locations)

    def _bfs(self, start_node, all_nodes):
        """
        Perform BFS from a start node to find distances to all reachable nodes.
        Returns a dictionary {node: distance}.
        """
        distances = {node: math.inf for node in all_nodes}
        if start_node in distances: # Ensure start_node is a known location
            distances[start_node] = 0
            queue = deque([start_node])

            while queue:
                current_node = queue.popleft()

                # Only process if the node exists in the graph (should be true if from all_nodes)
                if current_node in self.location_graph:
                    for neighbor in self.location_graph[current_node]:
                        if distances[neighbor] == math.inf:
                            distances[neighbor] = distances[current_node] + 1
                            queue.append(neighbor)

        return distances

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state

        # Find robot location
        robot_loc = None
        for fact in state:
            if match(fact, "at-robot", "*"):
                robot_loc = get_parts(fact)[1]
                break

        # If robot location isn't found, something is wrong with the state representation
        if robot_loc is None:
             # This state is likely invalid or unreachable in a standard problem
             return math.inf # Or a large constant

        # Find box locations for boxes we care about (those with goals)
        box_locations = {}
        for fact in state:
            if match(fact, "at", "*", "*"):
                obj, loc = get_parts(fact)
                # Only consider objects that are boxes with defined goals
                if obj in self.goal_locations:
                     box_locations[obj] = loc

        # Identify misplaced boxes
        misplaced_boxes = {
            box for box, loc in box_locations.items()
            if box in self.goal_locations and loc != self.goal_locations[box]
        }

        # If all boxes are at goals, heuristic is 0
        if not misplaced_boxes:
            return 0

        # Calculate sum of box-to-goal distances
        box_goal_distance_sum = 0
        for box in misplaced_boxes:
            current_box_loc = box_locations.get(box) # Use .get for safety
            goal_box_loc = self.goal_locations.get(box) # Use .get for safety

            # Ensure both locations are valid and reachable in the precomputed graph
            if current_box_loc is None or goal_box_loc is None or \
               current_box_loc not in self.shortest_paths or \
               goal_box_loc not in self.shortest_paths.get(current_box_loc, {}):
                 # This box is in an unexpected location or its goal is unreachable
                 return math.inf # Indicate unsolvable/unreachable

            dist = self.shortest_paths[current_box_loc][goal_box_loc]
            if dist == math.inf:
                 # Goal is unreachable from current box location on the empty grid
                 return math.inf # Indicate unsolvable/unreachable

            box_goal_distance_sum += dist


        # Calculate minimum robot-to-misplaced-box distance
        min_robot_box_distance = math.inf
        robot_loc_in_graph = robot_loc in self.shortest_paths

        if not robot_loc_in_graph:
             # Robot is in an unexpected location not in the graph
             return math.inf # Indicate unsolvable/unreachable

        for box in misplaced_boxes:
            box_loc = box_locations.get(box) # Use .get for safety

            if box_loc is not None and box_loc in self.shortest_paths[robot_loc]:
                 dist = self.shortest_paths[robot_loc][box_loc]
                 min_robot_box_distance = min(min_robot_box_distance, dist)
            # else: box_loc might be None or not reachable from robot_loc, handled by initial inf

        # If robot cannot reach any misplaced box, this state might be a deadlock
        # or unreachable. A large value is appropriate.
        if min_robot_box_distance == math.inf:
             # This might happen if the robot is in a disconnected component
             # from all misplaced boxes.
             return math.inf # Indicate unsolvable/unreachable

        # The heuristic is the sum of box distances and robot distance to nearest box
        return box_goal_distance_sum + min_robot_box_distance

