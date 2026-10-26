# Assuming heuristics.heuristic_base is available in the environment
from heuristics.heuristic_base import Heuristic
from fnmatch import fnmatch
from collections import deque

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    if not fact or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def bfs_shortest_path(graph, start, end):
    """
    Find the shortest path distance between two nodes in a graph using BFS.
    Graph is represented as {node: {neighbor: edge_data, ...}, ...}
    Returns the distance (number of edges) or None if unreachable.
    """
    if start == end:
        return 0
    if start not in graph or end not in graph:
        return None

    queue = deque([(start, 0)])
    visited = {start}

    while queue:
        current_loc, current_dist = queue.popleft()

        if current_loc == end:
             return current_dist

        if current_loc in graph:
            for neighbor in graph[current_loc]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, current_dist + 1))

    return None # Target not reachable

class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal state.
    It is calculated as the sum of the shortest path distances from each box to its
    goal location, plus the shortest path distance from the robot to the nearest
    box that is not yet at its goal. The distances are computed on the grid graph
    defined by the 'adjacent' facts, ignoring obstacles (other boxes or walls).

    # Assumptions
    - The grid structure is defined by 'adjacent' facts.
    - Location names follow the pattern 'loc_r_c'.
    - Each box has a single goal location specified in the task goals.
    - The grid is connected such that all locations mentioned in initial/goal states
      and static facts are reachable from each other (or unreachable parts are dead ends).
    - The heuristic is non-admissible.

    # Heuristic Initialization
    - Extracts the goal location for each box from the task goals.
    - Builds a graph representation of the grid connectivity from the static 'adjacent' facts.
    - Stores location coordinates by parsing location names (though coordinates are not strictly used in this version of the heuristic, the parsing logic is kept).

    # Step-By-Step Thinking for Computing Heuristic
    1. Check if the current state is the goal state using `self.task.goal_reached(state)`. If yes, return 0.
    2. Identify the current location of the robot by iterating through the state facts.
    3. Identify the current location of each box that has a goal location defined in the task.
    4. Initialize the total heuristic value to 0.
    5. Create a list of locations for boxes that are not yet at their goal.
    6. For each box that is not at its goal location:
       - Get its current location and its goal location.
       - Calculate the shortest path distance between the box's current location and its goal location using BFS on the grid graph (`self.shortest_path_distance`). This represents the minimum number of pushes needed for this box in an empty grid.
       - If the goal is unreachable for this box (BFS returns None), the state is likely a dead end or unsolvable; return infinity (`float('inf')`).
       - Add this distance to the total heuristic.
    7. If there are boxes not at their goal (i.e., `ungoaled_box_locations` is not empty):
       - Calculate the shortest path distance from the robot's current location to each location of an ungoaled box.
       - Find the minimum of these distances. This represents the minimum number of moves for the robot to reach *any* box that needs pushing.
       - If the robot cannot reach any ungoaled box (all distances are None), return infinity (`float('inf')`).
       - Add this minimum distance to the total heuristic.
    8. Return the total heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and static facts.
        """
        self.task = task

        # Store goal locations for each box.
        self.box_goals = {}
        for goal in self.task.goals:
            if match(goal, "at", "*", "*"):
                _, box, goal_loc = get_parts(goal)
                self.box_goals[box] = goal_loc

        # Build the graph from adjacent facts and store location coordinates.
        self.graph = {}  # {loc: {neighbor: dir}}
        self.location_coords = {} # {loc: (r, c)}
        self.all_locations = set()

        self.opposite_direction = {'up': 'down', 'down': 'up', 'left': 'right', 'right': 'left'}

        for fact in self.task.static:
            if match(fact, "adjacent", "*", "*", "*"):
                _, loc1, loc2, dir1 = get_parts(fact)
                dir2 = self.opposite_direction.get(dir1)
                if dir2 is None:
                     continue

                self.graph.setdefault(loc1, {})[loc2] = dir1
                self.graph.setdefault(loc2, {})[loc1] = dir2

                self.all_locations.add(loc1)
                self.all_locations.add(loc2)

        # Parse coordinates from location names like 'loc_r_c'
        for loc in self.all_locations:
            try:
                parts = loc.split('_')
                if len(parts) == 3 and parts[0] == 'loc':
                    row = int(parts[1])
                    col = int(parts[2])
                    self.location_coords[loc] = (row, col)
                else:
                    self.location_coords[loc] = None
            except (ValueError, IndexError):
                 self.location_coords[loc] = None


    def shortest_path_distance(self, start, end):
        """
        Find the shortest path distance between two locations on the grid graph.
        Uses the global bfs_shortest_path function.
        Returns the distance or None if unreachable.
        """
        return bfs_shortest_path(self.graph, start, end)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        if self.task.goal_reached(state):
            return 0

        robot_location = None
        for fact in state:
            if match(fact, "at-robot", "*"):
                _, robot_location = get_parts(fact)
                break

        if robot_location is None:
             return float('inf')

        box_locations = {}
        for fact in state:
            if match(fact, "at", "*", "*"):
                obj, loc = get_parts(fact)
                if obj in self.box_goals:
                    box_locations[obj] = loc

        if len(box_locations) != len(self.box_goals):
             return float('inf')

        total_heuristic = 0
        ungoaled_box_locations = []

        for box_name, goal_location in self.box_goals.items():
            current_box_location = box_locations[box_name]

            if current_box_location != goal_location:
                ungoaled_box_locations.append(current_box_location)
                box_to_goal_dist = self.shortest_path_distance(current_box_location, goal_location)

                if box_to_goal_dist is None:
                    return float('inf')

                total_heuristic += box_to_goal_dist

        if ungoaled_box_locations:
            min_robot_to_box_dist = float('inf')
            robot_can_reach_any_box = False

            for box_loc in ungoaled_box_locations:
                dist = self.shortest_path_distance(robot_location, box_loc)
                if dist is not None:
                    min_robot_to_box_dist = min(min_robot_to_box_dist, dist)
                    robot_can_reach_any_box = True

            if not robot_can_reach_any_box:
                 return float('inf')

            total_heuristic += min_robot_to_box_dist

        return total_heuristic
