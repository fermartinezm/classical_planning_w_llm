from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic

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
    # Ensure the number of parts matches the number of arguments
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal state.
    It is the sum of two components:
    1. The sum of shortest path distances for each box from its current location
       to its goal location.
    2. The shortest path distance for the robot from its current location to the
       location of the closest box that is not yet at its goal.

    # Assumptions
    - The grid structure is defined by `adjacent` predicates.
    - Each box has a specific goal location defined in the problem file.
    - Distances are shortest paths on the grid graph.
    - The heuristic is non-admissible, designed for greedy best-first search.

    # Heuristic Initialization
    - Parses the static facts to build the grid graph (adjacency list).
    - Computes all-pairs shortest paths on the grid graph using BFS.
    - Parses the goal conditions to map each box to its target location.

    # Step-by-Step Thinking for Computing the Heuristic Value
    For a given state:
    1. Identify the current location of the robot.
    2. Identify the current location of each box.
    3. For each box that is not yet at its goal location:
       a. Calculate the shortest path distance from the box's current location
          to its goal location using the precomputed distances. Sum these distances.
       b. Calculate the shortest path distance from the robot's current location
          to the box's current location. Find the minimum of these distances
          over all boxes not at their goals.
    4. The heuristic value is the sum of the total box-to-goal distance and the
       minimum robot-to-box distance.
    5. If all boxes are at their goals, the heuristic is 0.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by building the grid graph, computing distances,
        and extracting goal locations.
        """
        self.goals = task.goals
        static_facts = task.static

        # Build the grid graph from adjacent facts
        self.graph = {}
        self.locations = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "adjacent":
                loc1, loc2, direction = parts[1], parts[2], parts[3]
                self.locations.add(loc1)
                self.locations.add(loc2)
                if loc1 not in self.graph:
                    self.graph[loc1] = []
                if loc2 not in self.graph:
                    self.graph[loc2] = []
                # Add edges in both directions since adjacent is symmetric
                # (although PDDL lists both directions explicitly)
                if loc2 not in self.graph[loc1]:
                     self.graph[loc1].append(loc2)
                if loc1 not in self.graph[loc2]:
                     self.graph[loc2].append(loc1)

        # Compute all-pairs shortest paths using BFS
        self.distances = {}
        for start_loc in self.locations:
            self.distances[start_loc] = self._bfs(start_loc)

        # Store goal locations for each box
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                box, location = args
                self.goal_locations[box] = location

    def _bfs(self, start_node):
        """
        Performs Breadth-First Search from a start node to find distances
        to all reachable nodes in the graph.
        """
        distances = {loc: float('inf') for loc in self.locations}
        distances[start_node] = 0
        queue = deque([start_node])

        while queue:
            current_loc = queue.popleft()

            if current_loc in self.graph: # Ensure current_loc has neighbors in the graph
                for neighbor in self.graph[current_loc]:
                    if distances[neighbor] == float('inf'):
                        distances[neighbor] = distances[current_loc] + 1
                        queue.append(neighbor)

        return distances

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Find robot location
        robot_loc = None
        for fact in state:
            if match(fact, "at-robot", "*"):
                robot_loc = get_parts(fact)[1]
                break

        # Find current box locations
        box_locations = {}
        for fact in state:
            if match(fact, "at", "*", "*"):
                obj, loc = get_parts(fact)[1:]
                # Only consider objects that are boxes and have a goal
                if obj in self.goal_locations:
                     box_locations[obj] = loc

        total_box_dist = 0
        min_robot_to_box_dist = float('inf')
        boxes_not_at_goal = False

        # Calculate heuristic components for boxes not at their goals
        for box, goal_loc in self.goal_locations.items():
            current_box_loc = box_locations.get(box)

            # Ensure the box exists in the current state and is not at its goal
            if current_box_loc and current_box_loc != goal_loc:
                boxes_not_at_goal = True
                # Add box-to-goal distance
                # Handle cases where a location might not be in the precomputed distances
                # (e.g., due to disconnected graph, though unlikely in Sokoban)
                if current_box_loc in self.distances and goal_loc in self.distances[current_box_loc]:
                    box_dist = self.distances[current_box_loc][goal_loc]
                    # If distance is infinity, it's unreachable, return infinity
                    if box_dist == float('inf'):
                        return float('inf')
                    total_box_dist += box_dist
                else:
                    # Should not happen in typical Sokoban grids, but handle defensively
                    return float('inf') # Goal is unreachable

                # Find minimum robot-to-box distance
                if robot_loc in self.distances and current_box_loc in self.distances[robot_loc]:
                     robot_dist = self.distances[robot_loc][current_box_loc]
                     min_robot_to_box_dist = min(min_robot_to_box_dist, robot_dist)
                else:
                    # Robot cannot reach the box, return infinity
                    return float('inf')


        # If all boxes are at their goals, heuristic is 0
        if not boxes_not_at_goal:
            return 0

        # If robot cannot reach any box that needs moving, return infinity
        if min_robot_to_box_dist == float('inf'):
             return float('inf')

        # The heuristic is the sum of box distances and the minimum robot distance
        return total_box_dist + min_robot_to_box_dist

