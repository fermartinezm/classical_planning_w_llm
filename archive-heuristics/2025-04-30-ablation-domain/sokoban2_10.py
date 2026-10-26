# Assuming heuristics.heuristic_base exists and has a Heuristic base class
# from heuristics.heuristic_base import Heuristic

# If the base class is not available in this environment, define a dummy one
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    class Heuristic:
        def __init__(self, task):
            pass
        def __call__(self, node):
            raise NotImplementedError

from fnmatch import fnmatch
from collections import deque
import sys # Import sys for float('inf')

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at robot loc_1_1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure the number of parts matches the number of args if no wildcards in args count
    # This simple check is okay for fixed predicate arities like in Sokoban
    if len(parts) != len(args) and '*' not in args:
         return False
    # Check if each part matches the corresponding argument pattern
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


def bfs(graph, start_node):
    """
    Performs BFS to find shortest distances from start_node to all reachable nodes.
    Returns a dictionary mapping node -> distance.
    """
    distances = {start_node: 0}
    queue = deque([start_node])
    while queue:
        current_node = queue.popleft()
        current_dist = distances[current_node]
        # Check if current_node exists in the graph keys before iterating neighbors
        if current_node in graph:
            for neighbor in graph[current_node]:
                if neighbor not in distances:
                    distances[neighbor] = current_dist + 1
                    queue.append(neighbor)
    return distances


class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing the shortest
    path distances for each box to its assigned goal location and adding the
    shortest path distance from the robot to the closest box that needs to be moved.
    The path distances are computed on the grid graph defined by 'adjacent' predicates.

    # Assumptions
    - The grid connectivity is defined by 'adjacent' predicates, forming an undirected graph.
    - Each box has a specific goal location defined in the problem.
    - The cost of moving the robot one step is 1.
    - The cost of pushing a box one step is implicitly related to the path distance,
      as each step in the path corresponds to a push action.
    - The heuristic assumes paths exist between relevant locations in solvable problems.
    - Objects starting with 'box' are the boxes that need to be moved to goal locations.

    # Heuristic Initialization
    - Builds an undirected graph of locations based on 'adjacent' facts.
    - Precomputes all-pairs shortest path distances on this graph using BFS.
    - Stores the goal location for each box (identified by having a goal predicate).

    # Step-By-Step Thinking for Computing Heuristic
    1. **Extract State Information:** Identify the current location of the robot and all boxes from the state facts.
    2. **Identify Boxes to Move:** Compare the current location of each box with its designated goal location (precomputed during initialization). Create a list of boxes that are not yet at their goals.
    3. **Goal Check:** If the list of boxes to move is empty, the current state is a goal state, and the heuristic value is 0.
    4. **Calculate Total Box Distance:** For each box identified in step 2, find the shortest path distance from its current location to its goal location using the precomputed distances. Sum these distances. This sum represents the minimum total number of 'push' steps required for all boxes, ignoring the robot's positioning effort for each push.
    5. **Calculate Robot Distance to Closest Box:** Find the shortest path distance from the robot's current location to the location of the closest box that needs to be moved (from the list in step 2). This estimates the initial cost for the robot to reach a box it needs to interact with.
    6. **Combine Distances:** The total heuristic value is the sum of the total 'box distance' (step 4) and the 'robot distance to closest box' (step 5). Return this sum. Handle cases where distances are infinite (indicating unreachable locations) by returning infinity.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by building the location graph, precomputing
        distances, and storing goal locations.
        """
        self.goals = task.goals
        static_facts = task.static

        # Build the location graph from adjacent facts
        self.graph = {}
        locations = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'adjacent':
                loc1, loc2, direction = parts[1], parts[2], parts[3]
                locations.add(loc1)
                locations.add(loc2)
                # Add undirected edges
                self.graph.setdefault(loc1, []).append(loc2)
                self.graph.setdefault(loc2, []).append(loc1)

        # Remove duplicates from adjacency lists (graph is undirected)
        for loc in self.graph:
            self.graph[loc] = list(set(self.graph[loc]))

        # Precompute all-pairs shortest paths
        self.distances = {}
        all_locations = list(locations) # Ensure consistent order if needed, though BFS doesn't require it
        for start_loc in all_locations:
            # Use BFS to find distances from start_loc to all others
            dist_from_start = bfs(self.graph, start_loc)
            for end_loc, dist in dist_from_start.items():
                self.distances[(start_loc, end_loc)] = dist

        # Store goal locations for each box
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                # Goal is (at ?box ?location)
                box, location = args
                self.goal_locations[box] = location

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state (frozenset of fact strings)

        # Get current robot location
        robot_loc = None
        for fact in state:
            if match(fact, "at-robot", "*"):
                robot_loc = get_parts(fact)[1]
                break
        # If robot location is not found, the state is likely invalid or
        # represents an unsolvable branch.
        if robot_loc is None:
             return float('inf')

        # Get current box locations and identify boxes not at goal
        box_locations = {}
        boxes_to_move = []
        # Iterate through goal locations to find relevant boxes
        for box in self.goal_locations.keys():
             found_at_fact = False
             for fact in state:
                 # Check for the fact (at box current_loc)
                 if match(fact, "at", box, "*"):
                     current_loc = get_parts(fact)[2]
                     box_locations[box] = current_loc
                     found_at_fact = True
                     if current_loc != self.goal_locations[box]:
                         boxes_to_move.append(box)
                     break # Found location for this box, move to next box
             # If a box from the goal is not found in the state, it's an invalid state
             # (A box required by the goal is missing from the state)
             if not found_at_fact:
                 return float('inf')


        # If all boxes are at their goals, the heuristic is 0
        if not boxes_to_move:
            return 0

        total_box_distance = 0
        min_robot_to_box_distance = float('inf')

        for box in boxes_to_move:
            current_loc = box_locations[box]
            goal_loc = self.goal_locations[box]

            # Calculate box-to-goal distance (push distance)
            # Use .get() with default float('inf') to handle potential unreachable locations
            box_to_goal_dist = self.distances.get((current_loc, goal_loc), float('inf'))
            if box_to_goal_dist == float('inf'):
                 # Goal is unreachable from current box location on the graph
                 # This state is likely part of an unsolvable path or the problem is unsolvable
                 return float('inf')
            total_box_distance += box_to_goal_dist

            # Calculate robot-to-box distance
            robot_to_box_dist = self.distances.get((robot_loc, current_loc), float('inf'))
            if robot_to_box_dist == float('inf'):
                 # Robot cannot reach the box location on the graph
                 # This state is likely part of an unsolvable path or the problem is unsolvable
                 return float('inf')
            min_robot_to_box_distance = min(min_robot_to_box_distance, robot_to_box_dist)

        # The heuristic is the sum of box distances plus the robot's distance
        # to the nearest box it needs to interact with.
        # This is non-admissible as it doesn't account for robot movement
        # *between* pushing different boxes or getting into position for
        # subsequent pushes on the same box.
        return total_box_distance + min_robot_to_box_distance
