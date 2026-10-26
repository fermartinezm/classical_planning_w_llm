import collections
from fnmatch import fnmatch
# Assuming Heuristic base class is available in this path
# from heuristics.heuristic_base import Heuristic

# Define a dummy Heuristic base class for standalone testing if needed
# In the actual environment, this import will work.
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    class Heuristic:
        def __init__(self, task):
            pass
        def __call__(self, node):
            raise NotImplementedError

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact strings or malformed facts defensively
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at obj loc)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure the number of parts matches the number of args for a valid match attempt
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# BFS function
def bfs_distance(graph, start, end):
    """
    Find the shortest path distance between start and end nodes in a graph.
    Returns float('inf') if end is unreachable from start or if start/end are not in graph.
    """
    if start == end:
        return 0
    # Ensure start and end are valid nodes in the graph structure
    # A node might exist in the graph keys but have no neighbors (isolated)
    if start not in graph or end not in graph:
         return float('inf')

    queue = collections.deque([(start, 0)])
    visited = {start}
    while queue:
        current_loc, dist = queue.popleft()

        # Check if current_loc is a valid key in the graph dictionary
        # This check is technically redundant if start is already checked,
        # but harmless.
        if current_loc in graph:
            for neighbor in graph[current_loc]:
                if neighbor == end:
                    return dist + 1
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))

    return float('inf') # Target not reachable

class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing the
    shortest path distances for each box to its goal location and adding
    the shortest path distance from the robot to the nearest box that
    still needs to be moved. Distances are calculated on the graph of
    locations defined by the 'adjacent' predicates.

    # Assumptions
    - The world is represented as a graph of locations connected by
      adjacent predicates.
    - Shortest path distance on this graph is a reasonable estimate
      of movement cost for both the robot and the minimum pushes for a box
      (ignoring obstacles other than walls/non-adjacent locations).
    - The heuristic does not account for complex interactions like
      boxes blocking each other or dead-end states (except implicitly
      if goal is unreachable via graph).
    - Assumes a one-to-one mapping between boxes and goal locations
      defined in the goal state.
    - Assumes locations involved in 'at', 'at-robot', and 'adjacent'
      predicates are nodes in the same connected graph structure (or
      at least reachable within their relevant subgraphs).

    # Heuristic Initialization
    - Builds an undirected graph of locations based on the 'adjacent'
      predicates found in the static facts.
    - Extracts the goal location for each box from the task's goal state.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the current location of the robot and all boxes from the state.
    2. For each box, determine if it is at its goal location.
    3. If all boxes are at their goal locations, the heuristic is 0.
    4. Otherwise, initialize total_box_distance = 0 and min_robot_distance = infinity.
    5. Create a list of boxes that are not yet at their goal locations, storing their
       current and goal locations.
    6. Iterate through the list of boxes that need moving:
       - For each box, calculate the shortest path distance from its current location
         to its goal location using BFS on the location graph. Add this distance
         to total_box_distance.
    7. If total_box_distance is infinity (meaning at least one box goal is unreachable),
       the state is likely unsolvable, return infinity.
    8. Iterate again through the list of boxes that need moving:
       - For each box, calculate the shortest path distance from the robot's current
         location to the box's current location using BFS on the location graph.
         Update min_robot_distance if this distance is smaller.
    9. If min_robot_distance is infinity (meaning the robot cannot reach any box
       that needs moving), the state is likely unsolvable, return infinity.
    10. The heuristic value is total_box_distance + min_robot_distance.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by building the location graph and
        extracting box goal locations.
        """
        self.goals = task.goals
        static_facts = task.static

        # Build the location graph from adjacent facts
        self.graph = collections.defaultdict(set)
        all_locations = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "adjacent":
                # adjacent ?loc1 ?loc2 ?dir
                if len(parts) == 4:
                    loc1, loc2 = parts[1], parts[2]
                    self.graph[loc1].add(loc2)
                    self.graph[loc2].add(loc1) # Treat as undirected for distance
                    all_locations.add(loc1)
                    all_locations.add(loc2)

        # Extract goal locations for each box
        self.goal_locations = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "at" and len(parts) == 3:
                box, location = parts[1], parts[2]
                self.goal_locations[box] = location
                all_locations.add(location) # Add goal locations to potential nodes

        # Add any locations found (from adjacent or goals) to graph keys if not already present,
        # with empty adjacency set. This ensures BFS doesn't error if start/end is a valid
        # location name but has no adjacencies listed in static facts (e.g., isolated).
        for loc in all_locations:
             if loc not in self.graph:
                  self.graph[loc] = set()


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Find current robot location
        robot_loc = None
        # Iterate through state facts to find robot location
        for fact in state:
            if match(fact, "at-robot", "*"):
                parts = get_parts(fact)
                if len(parts) == 2:
                    robot_loc = parts[1]
                    break

        # If robot location is not found, state is invalid or unsolvable
        if robot_loc is None:
             return float('inf')

        # Find current box locations
        box_locations = {}
        # Iterate through state facts to find box locations
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "at" and len(parts) == 3:
                 obj_name = parts[1]
                 # Only consider objects that are identified as boxes (i.e., have a goal location)
                 if obj_name in self.goal_locations:
                    box_locations[obj_name] = parts[2]

        boxes_to_move = []
        # Identify boxes that are not at their goal
        for box, goal_loc in self.goal_locations.items():
            current_box_loc = box_locations.get(box)

            # If box location is unknown (e.g., box not in state facts), state is likely malformed/unsolvable
            if current_box_loc is None:
                 return float('inf')

            if current_box_loc != goal_loc:
                boxes_to_move.append((box, current_box_loc, goal_loc))

        # If no boxes need moving, heuristic is 0
        if not boxes_to_move:
            return 0

        total_box_distance = 0
        min_robot_distance = float('inf')

        # Calculate sum of box-to-goal distances
        for box, current_box_loc, goal_loc in boxes_to_move:
            dist = bfs_distance(self.graph, current_box_loc, goal_loc)
            # If any box goal is unreachable, the state is unsolvable
            if dist == float('inf'):
                 return float('inf')
            total_box_distance += dist

        # Calculate robot-to-nearest-box distance
        for box, current_box_loc, goal_loc in boxes_to_move:
             dist = bfs_distance(self.graph, robot_loc, current_box_loc)
             min_robot_distance = min(min_robot_distance, dist)

        # If robot cannot reach any box that needs moving, the state is unsolvable
        if min_robot_distance == float('inf'):
             return float('inf')

        # The heuristic is the sum of box distances plus the robot's distance to the nearest box
        return total_box_distance + min_robot_distance
