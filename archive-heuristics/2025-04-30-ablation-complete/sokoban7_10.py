# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

# If running standalone for testing, uncomment this mock:
# class Heuristic:
#     def __init__(self, task):
#         pass
#     def __call__(self, node):
#         # node is assumed to have a 'state' attribute
#         pass

from fnmatch import fnmatch
from collections import deque
import sys # Import sys for float('inf')

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle empty fact string or invalid format defensively
    if not fact or not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
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
    # Ensure we don't go out of bounds if fact has fewer parts than args
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def bfs(graph, start_node):
    """
    Performs Breadth-First Search to find shortest distances from a start node
    to all reachable nodes in a graph.

    Args:
        graph: Adjacency list representation of the graph (dict: node -> list of neighbors).
        start_node: The starting node for the BFS.

    Returns:
        A dictionary mapping each reachable node to its shortest distance from the start_node.
        Returns an empty dictionary if start_node is not in the graph.
    """
    if start_node not in graph:
        return {}

    distances = {start_node: 0}
    queue = deque([start_node])
    visited = {start_node}

    while queue:
        current_node = queue.popleft()

        # Check if current_node has neighbors in the graph
        if current_node in graph:
            for neighbor in graph[current_node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)

    return distances

class sokobanHeuristic: # Inherit from Heuristic if available
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal state.
    It is calculated as the sum of the shortest path distances for each misplaced
    box to its goal location, plus the shortest path distance from the robot's
    current location to the location of the nearest misplaced box. The distances
    are calculated on the static grid defined by the 'adjacent' predicates,
    ignoring dynamic obstacles (other boxes, robot) in the path calculation itself.

    # Assumptions
    - The grid structure is defined by 'adjacent' predicates and is static.
    - Shortest path distances on this static grid are a reasonable approximation
      of movement costs for both the robot and boxes (when pushed).
    - The heuristic is non-admissible and aims to guide a greedy best-first search.
    - The goal state is defined by the locations of the boxes.
    - The graph defined by 'adjacent' predicates is connected for all relevant locations (robot, boxes, goals) in solvable problems.

    # Heuristic Initialization
    - Build a graph representation of the grid from the 'adjacent' facts in the
      static information.
    - Precompute shortest path distances between all pairs of locations using BFS.
    - Extract the goal location for each box from the task's goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    1. Check if the current state satisfies all goal conditions. If yes, the
       heuristic value is 0.
    2. Identify the robot's current location from the state.
    3. Identify the current location of each box from the state.
    4. Determine which boxes are not at their goal locations (misplaced boxes)
       by comparing current box locations with the precomputed goal locations.
    5. If there are no misplaced boxes, the state is a goal state (already handled in step 1),
       return 0.
    6. Calculate the sum of shortest path distances for each misplaced box from its
       current location to its goal location. Use the precomputed distances.
       If any goal location is unreachable from a box's current location in the
       static graph, the state is likely unsolvable or represents a deadlock;
       return infinity.
       This sum represents a minimum number of pushes required for all boxes,
       ignoring robot positioning.
    7. Find the misplaced box that is closest to the robot (in terms of shortest
       path distance). Calculate this minimum distance.
       If any misplaced box location is unreachable from the robot's current location
       in the static graph, the state is likely unsolvable or represents a deadlock;
       return infinity.
       This estimates the robot's effort to reach a box it needs to push.
    8. The heuristic value is the sum of the total box-to-goal distances (from step 6)
       and the minimum robot-to-nearest-misplaced-box distance (from step 7).
    """

    def __init__(self, task):
        """
        Initialize the heuristic by building the grid graph, precomputing distances,
        and extracting goal locations.
        """
        # Assuming task object has 'goals' and 'static' attributes
        self.goals = task.goals
        static_facts = task.static

        # Build the graph from adjacent facts
        self.graph = {}
        locations = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "adjacent":
                # adjacent predicate has 3 arguments: loc1, loc2, direction
                if len(parts) == 4:
                    loc1, loc2, direction = parts[1:]
                    locations.add(loc1)
                    locations.add(loc2)
                    # Add edges in both directions for undirected distance calculation
                    self.graph.setdefault(loc1, []).append(loc2)
                    self.graph.setdefault(loc2, []).append(loc1)
                # else: ignore malformed adjacent fact

        # Remove duplicates from adjacency lists
        for loc in self.graph:
            self.graph[loc] = list(set(self.graph[loc]))

        # Precompute all-pairs shortest paths
        self.distances = {}
        # Get all unique locations from the graph keys and values
        all_locations = set(self.graph.keys())
        for neighbors in self.graph.values():
             all_locations.update(neighbors)
        all_locations = list(all_locations) # Convert to list if needed, set is fine for iteration

        for start_loc in all_locations:
             self.distances[start_loc] = bfs(self.graph, start_loc)

        # Store goal locations for each box
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at" and len(args) == 2:
                box, location = args
                self.goal_locations[box] = location
            # else: ignore other goal types if any

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # state is a frozenset of strings

        # Check if goal is reached
        if self.goals <= state:
            return 0

        # Extract current robot and box locations
        robot_location = None
        current_box_locations = {}
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "at-robot" and len(parts) == 2:
                robot_location = parts[1]
            elif parts and parts[0] == "at" and len(parts) == 3:
                box, location = parts[1:]
                current_box_locations[box] = location
            # else: ignore other fact types

        # If robot location is not found, something is wrong with the state
        if robot_location is None:
             # This shouldn't happen in a valid Sokoban state, but handle defensively
             return float('inf')

        # Identify misplaced boxes and calculate sum of box-to-goal distances
        misplaced_boxes = []
        sum_box_dist = 0
        for box, goal_loc in self.goal_locations.items():
            # If a box is required by the goal but not found in the state, it's an issue
            if box not in current_box_locations:
                 return float('inf') # Invalid state

            current_loc = current_box_locations[box]

            if current_loc != goal_loc:
                misplaced_boxes.append(box)
                # Get distance from precomputed table
                # Check if current_loc or goal_loc are in the distances table and reachable
                if current_loc not in self.distances or goal_loc not in self.distances.get(current_loc, {}):
                     # If goal is unreachable from current box location in static graph
                     return float('inf')

                box_dist = self.distances[current_loc][goal_loc]
                sum_box_dist += box_dist

        # If no boxes are misplaced, the goal must be reached (handled by initial check)
        # This case should logically not be reached if the initial goal check failed.
        if not misplaced_boxes:
             return 0 # Should already be 0 from the initial check

        # Calculate minimum robot-to-misplaced-box distance
        min_robot_box_dist = float('inf')
        for box in misplaced_boxes:
            box_loc = current_box_locations[box]
            # Check if robot_location or box_loc are in the distances table and reachable
            if robot_location not in self.distances or box_loc not in self.distances.get(robot_location, {}):
                 # Robot cannot reach this box in static graph
                 return float('inf')

            robot_box_dist = self.distances[robot_location][box_loc]
            min_robot_box_dist = min(min_robot_box_dist, robot_box_dist)

        # Total heuristic is sum of box distances + minimum robot distance
        # This is a non-admissible heuristic.
        return sum_box_dist + min_robot_box_dist
