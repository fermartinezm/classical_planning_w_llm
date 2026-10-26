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
    # Ensure the number of parts matches the number of arguments
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal state by summing:
    1. The shortest path distance for each box from its current location to its goal location.
    2. The shortest path distance for the robot from its current location to the location of the closest box that needs to be moved.

    # Assumptions
    - The grid structure is defined by `adjacent` predicates.
    - Goal specifies the final location for each specific box.
    - The heuristic uses shortest path distances on the grid, ignoring dynamic obstacles (other boxes, robot) and static obstacles (non-clear locations) for distance calculation, but these are handled by the planner's state representation.

    # Heuristic Initialization
    - Parses the static `adjacent` facts to build a graph representation of the grid.
    - Computes all-pairs shortest paths on this grid using BFS.
    - Extracts the goal locations for each box from the task goals.

    # Step-by-Step Thinking for Computing the Heuristic Value
    1. Precomputation (`__init__`):
       - Build an adjacency graph from `adjacent` predicates.
       - Compute shortest path distances between all pairs of locations using BFS. Store these in a dictionary `self.distances[loc1][loc2]`.
       - Extract the target location for each box from the goal state. Store in `self.goal_locations`.

    2. Heuristic Calculation (`__call__`):
       - Get the current state.
       - Find the robot's current location.
       - Find the current location of each box.
       - Initialize `total_box_distance = 0`.
       - Initialize `min_robot_to_box_distance = infinity`.
       - Identify which boxes are not yet at their goal locations.
       - For each box not at its goal:
         - Add the precomputed shortest distance from its current location to its goal location to `total_box_distance`.
         - Calculate the precomputed shortest distance from the robot's current location to this box's current location. Update `min_robot_to_box_distance` with the minimum such distance found so far.
       - If all boxes are at their goals (`total_box_distance == 0`), the heuristic is 0.
       - Otherwise, the heuristic is `total_box_distance + min_robot_to_box_distance`.

    This heuristic is non-admissible as it sums distances for boxes independently and simplifies the robot's movement cost, but it provides a reasonable estimate for greedy search.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by precomputing distances and goal locations.
        """
        self.goals = task.goals
        self.static = task.static

        # 1. Build the graph from adjacent facts
        self.graph = collections.defaultdict(list)
        locations = set()
        for fact in self.static:
            if match(fact, "adjacent", "*", "*", "*"):
                _, loc1, loc2, _ = get_parts(fact)
                self.graph[loc1].append(loc2)
                locations.add(loc1)
                locations.add(loc2)

        # 2. Compute all-pairs shortest paths using BFS
        self.distances = {}
        for start_node in locations:
            self.distances[start_node] = self._bfs(start_node, self.graph)

        # 3. Extract goal locations for each box
        self.goal_locations = {}
        for goal in self.goals:
            # Assuming goals are of the form (at box_name loc_name)
            if match(goal, "at", "*", "*"):
                _, box, location = get_parts(goal)
                self.goal_locations[box] = location

    def _bfs(self, start_node, graph):
        """
        Performs Breadth-First Search from a start node to find distances
        to all reachable nodes in the graph.
        """
        distances = {start_node: 0}
        queue = collections.deque([start_node])

        while queue:
            current_node = queue.popleft()

            # Check if current_node exists in the graph keys (it might not if it's a location only mentioned in goals/init but not adjacent facts)
            if current_node not in graph:
                 continue

            for neighbor in graph[current_node]:
                if neighbor not in distances:
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)

        return distances

    def __call__(self, node):
        """
        Compute the heuristic value for the given state.
        """
        state = node.state

        # Find robot's current location
        robot_loc = None
        for fact in state:
            if match(fact, "at-robot", "*"):
                robot_loc = get_parts(fact)[1]
                break

        if robot_loc is None:
             # This should not happen in a valid Sokoban state, but handle defensively
             return float('inf') # Or some large value indicating a problematic state

        # Find current box locations
        current_box_locations = {}
        for fact in state:
            if match(fact, "at", "*", "*"):
                obj, loc = get_parts(fact)[1:3]
                if obj in self.goal_locations: # Only track boxes that have a goal
                    current_box_locations[obj] = loc

        total_box_distance = 0
        min_robot_to_box_distance = float('inf')
        boxes_to_move = []

        # Calculate sum of box distances to goals and find boxes that need moving
        for box, goal_loc in self.goal_locations.items():
            current_loc = current_box_locations.get(box) # Use .get for safety

            # If a box from the goal state is not found in the current state, something is wrong
            if current_loc is None:
                 # This indicates an invalid state representation or task definition
                 # For a heuristic, we might return infinity or a large value
                 return float('inf')

            if current_loc != goal_loc:
                boxes_to_move.append(box)
                # Add distance from current box location to its goal location
                # Ensure locations exist in precomputed distances (they should if graph is built correctly)
                if current_loc in self.distances and goal_loc in self.distances[current_loc]:
                    total_box_distance += self.distances[current_loc][goal_loc]
                else:
                    # This might happen if goal location is unreachable from current location
                    # or if locations weren't in the adjacent facts (shouldn't happen in valid PDDL)
                    return float('inf') # Indicate unreachable goal for this box

        # If all boxes are at their goals, heuristic is 0
        if total_box_distance == 0:
            return 0

        # Calculate minimum robot distance to any box that needs moving
        for box in boxes_to_move:
            current_loc = current_box_locations[box]
            # Add distance from robot location to current box location
            if robot_loc in self.distances and current_loc in self.distances[robot_loc]:
                 min_robot_to_box_distance = min(min_robot_to_box_distance, self.distances[robot_loc][current_loc])
            else:
                 # Robot location or box location not in graph (unreachable?)
                 return float('inf')

        # If min_robot_to_box_distance is still infinity, it means there are boxes to move
        # but the robot cannot reach any of them.
        if min_robot_to_box_distance == float('inf'):
             return float('inf')

        # The heuristic is the sum of box distances plus the minimum robot distance to a box
        return total_box_distance + min_robot_to_box_distance

# Example usage (assuming you have a Task object 'task')
# heuristic = sokobanHeuristic(task)
# h_value = heuristic(current_node)
