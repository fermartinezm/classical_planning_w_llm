# Need to import Heuristic from a specific path relative to where this file will be placed.
# Assuming it's in a 'heuristics' directory and heuristic_base is in the same directory.
# If the structure is different, the import path might need adjustment.
# The problem description implies 'heuristics.heuristic_base', so let's use that.
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque
import sys # Import sys for float('inf') comparison safety, though float('inf') is standard

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
    # Ensure the number of parts matches the number of args, unless args has wildcards at the end
    # A simple zip check is sufficient for the patterns used here.
    if len(parts) != len(args):
        return False # Or handle variable length facts if needed, but not in this domain
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def bfs(start_loc, graph):
    """
    Performs BFS to find shortest distances from start_loc to all reachable locations.
    Returns a dictionary {location: distance}.
    """
    # Initialize distances to infinity for all nodes in the graph
    distances = {loc: float('inf') for loc in graph}
    # Set distance to start_loc to 0
    distances[start_loc] = 0
    # Use a deque for efficient queue operations
    queue = deque([start_loc])
    # Keep track of visited nodes to avoid cycles and redundant processing
    visited = {start_loc}

    while queue:
        # Get the next location to process from the queue
        current_loc = queue.popleft()

        # Iterate over neighbors of the current location
        # Ensure current_loc is actually a key in the graph before accessing neighbors
        # This handles cases where a location might be in the 'locations' set but not have adjacent facts
        if current_loc in graph:
            for neighbor in graph[current_loc]:
                # If the neighbor hasn't been visited yet
                if neighbor not in visited:
                    visited.add(neighbor)
                    distances[neighbor] = distances[current_loc] + 1
                    queue.append(neighbor)

    return distances

class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal state
    by summing the minimum number of pushes needed for each misplaced box to reach
    its goal, and adding the minimum number of robot movements required to reach
    the vicinity of the closest misplaced box.

    # Assumptions
    - The location graph defined by 'adjacent' predicates is connected for all
      relevant locations (robot, boxes, goals). If not, unreachable goals result
      in infinite heuristic value.
    - The cost of a 'move' action is 1.
    - The cost of a 'push' action is 1.
    - The heuristic ignores potential deadlocks or complex box interactions beyond
      simple distance. It assumes boxes can be moved along shortest paths if the
      robot can reach them.

    # Heuristic Initialization
    - Builds the location graph based on 'adjacent' facts.
    - Computes all-pairs shortest paths between locations using BFS.
    - Extracts the goal location for each box from the task goals.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the current location of the robot from the state facts.
    2. Identify the current location of each box from the state facts.
    3. Initialize `total_box_distance` to 0 and `min_robot_to_box_distance` to infinity.
    4. Initialize a counter for `misplaced_boxes_count` to 0.
    5. Iterate through each box that has a goal location defined in the task:
       a. Get the box's current location.
       b. Get the box's goal location.
       c. If the current location is different from the goal location:
          i. Increment `misplaced_boxes_count`.
          ii. Look up the precomputed shortest path distance from the box's current
              location to its goal location. If the distance is infinity (unreachable),
              return infinity for the heuristic.
          iii. Add this distance to `total_box_distance`.
          iv. Look up the precomputed shortest path distance from the robot's current
              location to the box's current location. If the distance is infinity
              (unreachable), return infinity for the heuristic.
          v. Update `min_robot_to_box_distance` with the minimum of its current value
             and the robot-to-box distance just calculated.
    6. If `misplaced_boxes_count` is 0, it means all boxes are at their goals,
       so return 0.
    7. Otherwise, the heuristic value is `total_box_distance + min_robot_to_box_distance`.
       Ensure `min_robot_to_box_distance` is not still infinity (which would happen
       if there were misplaced boxes but the robot couldn't reach any of them).
       If it's infinity, return infinity.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by building the location graph, computing
        shortest paths, and extracting goal locations.
        """
        self.goals = task.goals
        static_facts = task.static

        # Build the location graph from adjacent facts
        self.graph = {}
        locations = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "adjacent":
                loc1, loc2, direction = parts[1], parts[2], parts[3]
                locations.add(loc1)
                locations.add(loc2)
                # Add edges for both directions as adjacency is symmetric
                self.graph.setdefault(loc1, []).append(loc2)
                self.graph.setdefault(loc2, []).append(loc1)

        # Ensure all locations mentioned in adjacent facts are in the graph keys
        # even if they have no outgoing edges (though unlikely in Sokoban).
        for loc in locations:
             self.graph.setdefault(loc, [])

        # Compute all-pairs shortest paths
        self.distances = {}
        all_locations = list(self.graph.keys()) # Use keys to get all nodes
        for start_loc in all_locations:
            self.distances[start_loc] = bfs(start_loc, self.graph)

        # Store goal locations for each box
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                box, location = args
                self.goal_locations[box] = location
            # Note: Sokoban goals are typically only (at box location).
            # If other goal types existed, they would need handling or ignoring.


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Find robot location
        robot_location = None
        for fact in state:
            if match(fact, "at-robot", "*"):
                robot_location = get_parts(fact)[1]
                break
        # If robot location isn't found, the state is likely malformed or unsolvable.
        # Return infinity in such cases.
        if robot_location is None:
             return float('inf')


        # Find current box locations for boxes that have goals
        current_box_locations = {}
        for fact in state:
            if match(fact, "at", "*", "*"):
                obj, loc = get_parts(fact)[1:3]
                # Only consider objects that are boxes with defined goals
                if obj in self.goal_locations:
                     current_box_locations[obj] = loc

        total_box_distance = 0
        min_robot_to_box_distance = float('inf')
        misplaced_boxes_count = 0

        # Calculate heuristic components for each misplaced box
        for box, goal_location in self.goal_locations.items():
            current_location = current_box_locations.get(box)

            # If a box with a goal isn't found in the state's 'at' facts,
            # it's an invalid state for this heuristic's assumptions.
            if current_location is None:
                 return float('inf')

            if current_location != goal_location:
                misplaced_boxes_count += 1

                # Distance for the box to reach its goal
                # Check if locations exist in the distance map (should if graph was built correctly)
                if current_location not in self.distances or goal_location not in self.distances.get(current_location, {}):
                     # This indicates a location mentioned in state/goal wasn't in the graph
                     return float('inf')

                dist_box_to_goal = self.distances[current_location][goal_location]
                if dist_box_to_goal == float('inf'):
                    # Goal is unreachable for this box from its current location
                    return float('inf')

                total_box_distance += dist_box_to_goal

                # Distance for the robot to reach this box
                if robot_location not in self.distances or current_location not in self.distances.get(robot_location, {}):
                     # This indicates robot location or box location wasn't in the graph
                     return float('inf')

                dist_robot_to_this_box = self.distances[robot_location][current_location]
                if dist_robot_to_this_box == float('inf'):
                     # Robot cannot reach this box
                     return float('inf')

                min_robot_to_box_distance = min(min_robot_to_box_distance, dist_robot_to_this_box)

        # If all boxes are at their goals, heuristic is 0
        if misplaced_boxes_count == 0:
            return 0

        # If there are misplaced boxes but the robot cannot reach any of them, return infinity
        if min_robot_to_box_distance == float('inf'):
             return float('inf')

        # The heuristic is the sum of total box movement and robot movement to the closest box
        return total_box_distance + min_robot_to_box_distance
