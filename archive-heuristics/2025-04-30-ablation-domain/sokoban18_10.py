from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and has parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        # This should not happen with valid PDDL facts from the parser,
        # but as a safeguard, return an empty list.
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at box1 loc_1_1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal state by summing two components:
    1. The minimum number of pushes required for each misplaced box to reach its goal location, calculated as the shortest path distance on the static grid graph.
    2. The minimum number of robot moves required to reach any of the misplaced boxes, calculated as the shortest path distance on the static grid graph.

    # Assumptions
    - The grid structure and connectivity are defined solely by the static `adjacent` facts.
    - The heuristic calculates distances on this static grid graph, ignoring dynamic obstacles (other boxes, the robot's current position relative to pushing direction, and the current `clear` status of locations) for simplicity and computational efficiency. This is a key source of non-admissibility.
    - Locations that are unreachable from each other on the static grid graph are considered infinitely distant. If a box's goal is unreachable or the robot cannot reach any misplaced box on the static grid, the heuristic returns infinity.
    - The cost of any action (move or push) is assumed to be 1 for distance calculation purposes within the static grid BFS.

    # Heuristic Initialization
    - Extracts the goal location for each box from the task's goal conditions.
    - Builds an undirected graph representing the static grid connectivity based on `adjacent` facts.
    - Computes all-pairs shortest path distances on this static grid graph using BFS from every location. These distances are stored for quick lookup during heuristic computation.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the robot's current location.
    2. Identify the current location of each box.
    3. Determine which boxes are not currently at their designated goal locations (misplaced boxes).
    4. If there are no misplaced boxes, the state is a goal state, and the heuristic value is 0.
    5. If there are misplaced boxes:
        a. Calculate the sum of the shortest path distances (on the static grid graph, precomputed in `__init__`) for each misplaced box from its current location to its goal location. This represents the minimum number of pushes needed for all boxes if they could move independently on the empty grid. If any box's goal is unreachable, the total heuristic is infinity.
        b. Calculate the minimum shortest path distance (on the static grid graph) from the robot's current location to the location of *any* of the misplaced boxes. This represents the minimum robot movement cost to get close to a box that needs pushing. If the robot cannot reach any misplaced box, the total heuristic is infinity.
        c. The total heuristic value for the state is the sum of the total box-goal distance sum (from step 5a) and the minimum robot-to-box distance (from step 5b).
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal locations and precomputing
        shortest path distances on the static grid.
        """
        self.goals = task.goals
        static_facts = task.static

        # Store goal locations for each box
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                box, location = args
                self.goal_locations[box] = location

        # Build the static grid graph from adjacent facts
        self.graph = {}
        locations = set()
        for fact in static_facts:
            if match(fact, "adjacent", "*", "*", "*"):
                _, loc1, loc2, _ = get_parts(fact)
                locations.add(loc1)
                locations.add(loc2)
                if loc1 not in self.graph:
                    self.graph[loc1] = []
                if loc2 not in self.graph:
                    self.graph[loc2] = []
                # Add edges in both directions as movement is bidirectional
                self.graph[loc1].append(loc2)
                self.graph[loc2].append(loc1)

        # Compute All-Pairs Shortest Paths on the static graph using BFS
        self.distances = {}
        for start_loc in locations:
            dist_map = self._bfs(start_loc, self.graph)
            for end_loc, dist in dist_map.items():
                self.distances[(start_loc, end_loc)] = dist

    def _bfs(self, start_node, graph):
        """
        Performs a Breadth-First Search starting from start_node to find
        shortest distances to all reachable nodes in the graph.
        """
        distances = {node: float('inf') for node in graph}
        distances[start_node] = 0
        queue = deque([start_node])

        while queue:
            current_node = queue.popleft()

            # Check if current_node exists in the graph keys (handles isolated nodes if any)
            if current_node not in graph:
                 continue

            for neighbor in graph[current_node]:
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)
        return distances

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions
        to reach a goal state from the current state.
        """
        state = node.state

        robot_loc = None
        box_locations = {}
        # Use a set for efficient lookups if needed, though iteration is fine here
        # current_facts = set(state)

        # Find robot and box locations in the current state
        for fact in state:
            predicate, *args = get_parts(fact)
            if predicate == "at-robot":
                robot_loc = args[0]
            elif predicate == "at" and args and args[0].startswith("box"):
                box, loc = args
                box_locations[box] = loc

        total_box_dist = 0
        misplaced_boxes = []

        # Identify misplaced boxes and calculate sum of box-goal distances
        for box, goal_loc in self.goal_locations.items():
            box_loc = box_locations.get(box)

            # If a box expected by the goal is not found in the state,
            # something is wrong or the state is unreachable/invalid.
            # Return infinity.
            if box_loc is None:
                 return float('inf')

            if box_loc != goal_loc:
                misplaced_boxes.append((box, box_loc))
                # Get static distance from current box location to goal location
                dist_box_goal = self.distances.get((box_loc, goal_loc), float('inf'))

                # If goal is unreachable on the static grid, return infinity
                if dist_box_goal == float('inf'):
                    return float('inf')
                total_box_dist += dist_box_goal

        # If no boxes are misplaced, the goal is reached
        if not misplaced_boxes:
            return 0

        # Calculate minimum robot distance to any misplaced box
        min_robot_dist = float('inf')
        for box, box_loc in misplaced_boxes:
             # Get static distance from robot location to the box location
             dist_robot_box = self.distances.get((robot_loc, box_loc), float('inf'))
             min_robot_dist = min(min_robot_dist, dist_robot_box)

        # If robot cannot reach any misplaced box on the static grid, return infinity
        if min_robot_dist == float('inf'):
             return float('inf')

        # The heuristic is the sum of total box push distance and minimum robot approach distance
        total_heuristic = total_box_dist + min_robot_dist

        return total_heuristic

