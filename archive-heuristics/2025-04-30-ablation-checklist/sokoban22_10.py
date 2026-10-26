from fnmatch import fnmatch
from collections import deque
# Assuming the Heuristic base class is available in the environment
# from heuristics.heuristic_base import Heuristic

# Helper functions from the example Logistics heuristic
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
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def bfs(graph, start, end):
    """
    Performs Breadth-First Search to find the shortest path distance
    between start and end nodes in a graph.

    Args:
        graph: A dictionary representing the adjacency list of the graph.
               {location: [adjacent_locations]}
        start: The starting location.
        end: The target location.

    Returns:
        The shortest distance (number of edges) from start to end,
        or float('inf') if end is unreachable from start.
    """
    if start == end:
        return 0

    # Ensure start and end are valid locations in the graph structure
    if start not in graph or end not in graph:
        return float('inf')

    queue = deque([(start, 0)])
    visited = {start}

    while queue:
        current_loc, dist = queue.popleft()

        if current_loc == end:
            return dist

        # Check if current_loc has neighbors in the graph
        if current_loc in graph:
            for neighbor in graph[current_loc]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))

    return float('inf') # Target not reachable

class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    Estimates the cost as the sum of shortest path distances from each box
    to its goal location, plus the shortest path distance from the robot
    to the nearest box that is not yet at its goal.

    This heuristic is non-admissible. It calculates distances on the static
    adjacency graph, ignoring dynamic obstacles (other boxes, robot) and
    the specific push mechanics (robot needs to be on a specific side).
    It aims to guide a greedy best-first search efficiently by prioritizing
    states where boxes are closer to their goals and the robot is closer
    to a box that needs moving.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by building the adjacency graph and
        extracting box-goal mappings.
        """
        self.goals = task.goals
        self.static = task.static

        # Build the adjacency graph from static 'adjacent' facts
        self.graph = self._build_graph(self.static)

        # Extract box-goal location mapping from goal conditions
        self.goal_locations = self._extract_goal_locations(self.goals)

    def _build_graph(self, static_facts):
        """Builds an adjacency list graph from 'adjacent' facts."""
        graph = {}
        all_locations = set()

        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'adjacent' and len(parts) == 4:
                loc1, loc2 = parts[1], parts[2]
                all_locations.add(loc1)
                all_locations.add(loc2)

                # Add edge loc1 -> loc2
                if loc1 not in graph:
                    graph[loc1] = []
                graph[loc1].append(loc2)
                # The PDDL includes both directions explicitly, so we rely on that
                # rather than assuming symmetry and adding reverse edges here.

        # Ensure all locations mentioned in adjacent facts are keys in the graph,
        # even if they have no outgoing edges listed. This prevents KeyError in BFS.
        for loc in all_locations:
            if loc not in graph:
                graph[loc] = []

        return graph

    def _extract_goal_locations(self, goals):
        """Extracts the mapping from box objects to their goal locations."""
        goal_locations = {}
        for goal in goals:
            parts = get_parts(goal)
            # Goal facts are typically (at <object> <location>)
            if parts[0] == 'at' and len(parts) == 3:
                obj_name, loc_name = parts[1], parts[2]
                # In Sokoban, 'at' goals usually apply to boxes.
                # We can identify boxes by their type or naming convention.
                # Assuming objects starting with 'box' are the relevant ones for 'at' goals.
                if obj_name.startswith('box'):
                     goal_locations[obj_name] = loc_name
        return goal_locations


    def __call__(self, node):
        """
        Compute the heuristic value for the given state.

        The heuristic is the sum of shortest path distances from each box's
        current location to its goal location, plus the shortest path distance
        from the robot to the nearest box that is not yet at its goal.
        """
        state = node.state

        # Find current locations of the robot and all boxes
        current_locations = {}
        robot_loc = None
        box_locations = {}

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at-robot' and len(parts) == 2:
                robot_loc = parts[1]
                current_locations['robot'] = robot_loc
            elif parts[0] == 'at' and len(parts) == 3:
                 obj_name, loc_name = parts[1], parts[2]
                 if obj_name.startswith('box'): # Identify boxes
                     box_locations[obj_name] = loc_name
                     current_locations[obj_name] = loc_name

        # If robot location is unknown, state is invalid or unreachable
        if robot_loc is None:
             return float('inf')

        total_box_goal_distance = 0
        boxes_needing_move = []

        # Calculate distance for each box not at its goal
        for box, goal_loc in self.goal_locations.items():
            if box in box_locations: # Ensure the box exists in the current state
                current_loc = box_locations[box]
                if current_loc != goal_loc:
                    boxes_needing_move.append(box)
                    dist = bfs(self.graph, current_loc, goal_loc)
                    if dist == float('inf'):
                         # If any box cannot reach its goal on the static graph,
                         # this state is likely a dead end or requires complex
                         # maneuvers not captured by simple distance.
                         return float('inf')
                    total_box_goal_distance += dist
            # else:
                # A box defined in the goal is not present in the state's 'at' facts.
                # This indicates an invalid state or an unhandled case.
                # For typical Sokoban, all boxes should have an 'at' fact.
                # Treat as unreachable.
                # return float('inf')


        # If all boxes are at their goals, the heuristic is 0
        if not boxes_needing_move:
             return 0

        # Calculate the minimum distance from the robot to any box that needs moving
        min_robot_box_distance = float('inf')
        for box in boxes_needing_move:
             box_loc = box_locations[box]
             dist_robot = bfs(self.graph, robot_loc, box_loc)
             min_robot_box_distance = min(min_robot_box_distance, dist_robot)

        # If the robot cannot reach any box that needs moving, return infinity
        if min_robot_box_distance == float('inf'):
             return float('inf')

        # The heuristic is the sum of box-goal distances plus the distance
        # from the robot to the nearest box that needs moving.
        # We add 1 to the robot distance to account for the first push action
        # after reaching the box's vicinity. This is a common trick in Sokoban
        # heuristics to slightly inflate the cost and differentiate states.
        # However, the problem asks for an estimate of *actions*. Reaching the box
        # takes moves, pushing takes a push action. Let's just add the distance.
        # The robot needs to reach a square *adjacent* to the box, not the box's square itself.
        # The distance to the box is a proxy for this.

        # A simpler and often effective non-admissible heuristic is just the sum
        # of box-goal distances. Let's use that first as it's less prone to
        # weird interactions between robot and multiple boxes.
        # If that's too weak, we can revisit adding the robot distance.

        # Reverting to the simpler heuristic: Sum of box-goal distances only.
        # This is a common relaxation where boxes can move freely to their goals.
        # It ignores the robot and push mechanics, but is fast and goal-directed.
        return total_box_goal_distance

        # Alternative: Sum of box-goal distances + min robot-box distance
        # return total_box_goal_distance + min_robot_box_distance

        # Alternative: Sum of box-goal distances + sum of robot-box distances (less common)
        # total_robot_box_distance = 0
        # for box in boxes_needing_move:
        #      box_loc = box_locations[box]
        #      dist_robot = bfs(self.graph, robot_loc, box_loc)
        #      if dist_robot == float('inf'): return float('inf') # Robot cannot reach this box
        #      total_robot_box_distance += dist_robot
        # return total_box_goal_distance + total_robot_box_distance


# Note: The Heuristic base class is assumed to be provided externally.
# If running this code standalone, you would need a definition like:
# class Heuristic:
#     def __init__(self, task): pass
#     def __call__(self, node): raise NotImplementedError
