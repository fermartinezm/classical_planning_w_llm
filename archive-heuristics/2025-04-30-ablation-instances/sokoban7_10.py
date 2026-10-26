# Add the required import for the base class if available
# from heuristics.heuristic_base import Heuristic

# Dummy Heuristic base class for standalone testing if the actual one is not provided
# class Heuristic:
#     def __init__(self, task):
#         self.goals = task.goals
#         self.static = task.static
#     def __call__(self, node):
#         raise NotImplementedError

from fnmatch import fnmatch
from collections import deque
import sys # Needed for float('inf') if not using builtins

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at obj loc)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Use zip to iterate up to the length of the shorter sequence.
    # fnmatch handles '*' correctly.
    # The original example didn't enforce length match, let's follow that.
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


def bfs_shortest_path(start_node, graph):
    """
    Computes shortest path distances from a start_node to all other nodes
    in a graph using Breadth-First Search.

    Args:
        start_node: The node to start the BFS from.
        graph: An adjacency list representation of the graph (dict: node -> list of neighbors).

    Returns:
        A dictionary mapping each reachable node to its shortest distance from start_node.
        Nodes not reachable will have distance float('inf').
    """
    # Initialize distances to infinity for all nodes in the graph
    distances = {node: float('inf') for node in graph}
    # Distance from start node to itself is 0
    distances[start_node] = 0
    # Use a deque for efficient queue operations
    queue = deque([(start_node, 0)])
    # Keep track of visited nodes to avoid cycles and redundant processing
    visited = {start_node}

    while queue:
        # Get the next node and its distance from the start
        current_node, current_dist = queue.popleft() # Use popleft() for deque

        # Explore neighbors
        if current_node in graph:
            for neighbor in graph[current_node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    # Distance to neighbor is one more than current node's distance
                    distances[neighbor] = current_dist + 1
                    # Add neighbor to the queue
                    queue.append((neighbor, current_dist + 1))

    return distances


class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the number of actions required to move all boxes
    to their goal locations. It sums the shortest path distance for each box
    to its goal and adds the shortest path distance for the robot to reach
    a location adjacent to any box that needs moving.

    # Assumptions:
    - Each box that needs to be moved has a unique goal location specified in the task goals.
    - The grid structure and adjacency relations between locations are static and defined by 'adjacent' facts.
    - Shortest path distance on the location graph is a reasonable estimate
      of movement cost for both boxes (when pushed by robot) and the robot.
    - Ignores potential deadlocks (states where a box cannot be moved further due to walls or other boxes).
    - Ignores the 'clear' predicate constraint for heuristic calculation,
      except implicitly through the graph structure (walls/impassable areas are not adjacent).

    # Heuristic Initialization
    - Extract the goal location for each box from the task goals.
    - Build an adjacency map for locations based on static 'adjacent' facts. This map represents the traversable graph for the robot and boxes (when pushed).
    - Collect all unique locations mentioned in static facts and goals.
    - Compute all-pairs shortest paths between all collected locations using BFS
      on the built graph and store these distances.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the current location of the robot and all boxes from the state.
    2. Initialize the total heuristic value to 0.
    3. Initialize the minimum distance from the robot to a 'push-ready' location
       (a location adjacent to a box that needs moving) to infinity.
    4. Get the robot's current location. If the robot's location is not in the precomputed distance map (e.g., isolated location not in the graph), return infinity.
    5. For each box that has a goal location defined in the task goals:
       a. Determine its current location from the state. If the box's current location is not found in the state or is not in the precomputed distance map, return infinity.
       b. Get the box's goal location. If the goal location is not in the precomputed distance map, return infinity.
       c. If the box is not at its goal location:
          i. Look up the precomputed shortest path distance from the box's current
             location to its goal location. If unreachable (distance is infinity), return infinity.
          ii. Add this distance to the total heuristic. This estimates
              the minimum number of pushes required for this box.
          iii. Find all locations adjacent to the box's current location using
              the precomputed adjacency map. These are potential 'push-ready'
              locations for the robot.
          iv. For each potential 'push-ready' location:
              - Look up the precomputed shortest path distance from the robot's
                current location to this adjacent location. If unreachable (distance is infinity), return infinity.
              - Update the minimum robot distance found so far with the minimum
                distance calculated in the previous step (across all adjacent
                locations of the current box and across all boxes needing movement).
    6. After iterating through all boxes, check if any box needed movement (`needs_movement` flag).
    7. If no boxes needed movement, the heuristic is 0.
    8. If boxes needed movement but the minimum robot distance to a push position is still infinity (meaning the robot cannot reach any location adjacent to any box that needs moving), the problem is likely unsolvable from this state, so return infinity.
    9. Otherwise (boxes need movement and robot can reach a push position), add the minimum robot distance to the total heuristic. This estimates the cost for the robot to reach a position where it can start pushing *any* box that needs moving.
    10. Return the total heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # Store goal locations for each box.
        self.goal_locations = {}
        # Collect all unique locations mentioned in static facts and goals
        all_locations = set()

        for goal in self.goals:
            predicate, *args = get_parts(goal)
            # Assuming 'at' predicates in goals refer to boxes and their target locations
            if predicate == "at" and len(args) == 2:
                obj_name, loc_name = args
                self.goal_locations[obj_name] = loc_name
                all_locations.add(loc_name) # Add goal locations to our set of known locations

        # Build adjacency map: location -> list of adjacent_locations
        self.adjacency_map = {}
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'adjacent' and len(parts) == 4:
                loc1, loc2, direction = parts[1], parts[2], parts[3]
                all_locations.add(loc1) # Add locations from adjacent facts
                all_locations.add(loc2)
                if loc1 not in self.adjacency_map:
                    self.adjacency_map[loc1] = []
                if loc2 not in self.adjacency_map:
                    self.adjacency_map[loc2] = []
                # Store just the adjacent location for graph traversal
                self.adjacency_map[loc1].append(loc2)
                self.adjacency_map[loc2].append(loc1) # Assuming adjacency is symmetric

        # Ensure all collected locations are keys in the adjacency map, even if they have no adjacencies listed
        # This is important so BFS covers all relevant locations.
        for loc in all_locations:
             if loc not in self.adjacency_map:
                 self.adjacency_map[loc] = []

        # Compute all-pairs shortest paths
        self.distance_map = {}
        for start_loc in self.adjacency_map:
             # BFS needs a graph where values are just neighbors, not (neighbor, direction)
             # The adjacency_map is already in this format (list of neighbors)
             self.distance_map[start_loc] = bfs_shortest_path(start_loc, self.adjacency_map)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Find robot location
        robot_location = None
        for fact in state:
            if match(fact, "at-robot", "*"):
                robot_location = get_parts(fact)[1]
                break

        # If robot location is not in our graph, it's an unreachable state or invalid problem
        if robot_location is None or robot_location not in self.distance_map:
             return float('inf')

        # Find box locations for boxes that have goals
        box_locations = {}
        for fact in state:
            if match(fact, "at", "*", "*"):
                 obj_name, loc_name = get_parts(fact)[1:]
                 # Only track locations for boxes that are in our goal list
                 if obj_name in self.goal_locations:
                      box_locations[obj_name] = loc_name

        total_box_distance = 0
        min_robot_distance_to_push_pos = float('inf')
        needs_movement = False # Flag to check if any box needs moving

        # Calculate cost for each box not at its goal
        for box, goal_location in self.goal_locations.items():
            current_location = box_locations.get(box)

            # If box is not found in state or its current location is not in our graph, problem is likely unsolvable
            if current_location is None or current_location not in self.distance_map:
                 return float('inf')

            # If goal location is not in our graph, problem is likely unsolvable
            if goal_location not in self.distance_map:
                 return float('inf')

            if current_location != goal_location:
                needs_movement = True # At least one box needs to move

                # Add box distance to goal
                dist_box_to_goal = self.distance_map[current_location].get(goal_location, float('inf'))
                if dist_box_to_goal == float('inf'):
                     # Goal is unreachable for this box from its current location
                     return float('inf')
                total_box_distance += dist_box_to_goal

                # Find minimum robot distance to a location adjacent to this box
                # Check if the box's current location has adjacent locations in our graph
                if current_location in self.adjacency_map:
                    for adjacent_loc in self.adjacency_map[current_location]:
                        # Check if the adjacent location is in the distance map from the robot's location
                        dist_robot_to_adj = self.distance_map[robot_location].get(adjacent_loc, float('inf'))
                        # If robot cannot reach *any* adjacent location of *any* box needing movement,
                        # min_robot_distance_to_push_pos will remain inf.
                        min_robot_distance_to_push_pos = min(min_robot_distance_to_push_pos, dist_robot_to_adj)
                # else: If a box is at a location with no adjacencies and needs moving,
                # dist_box_to_goal would likely be inf unless the goal is the same location.
                # If goal is same location, needs_movement is false. If goal is different, dist_box_to_goal is inf.
                # So this case doesn't need explicit handling here.


        # If no boxes needed movement, the heuristic is 0.
        # Otherwise, add the minimum robot travel cost.
        if not needs_movement:
             return 0
        elif min_robot_distance_to_push_pos == float('inf'):
             # Boxes need movement, but robot cannot reach *any* adjacent location
             # of *any* box that needs movement.
             return float('inf')
        else:
             return total_box_distance + min_robot_distance_to_push_pos
