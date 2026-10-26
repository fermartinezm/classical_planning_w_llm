from fnmatch import fnmatch
from collections import deque
import math # For float('inf')

# Assume Heuristic base class is available from the environment
# from heuristics.heuristic_base import Heuristic

# Utility functions to parse PDDL facts
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
    # Handle cases where fact has fewer parts than args
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing the shortest
    path distances of each misplaced box to its goal location and adding the
    shortest path distance from the robot to the closest misplaced box.

    # Assumptions
    - The goal state specifies the required location for each box using the `(at ?box ?location)` predicate.
    - The grid structure and adjacency are defined by the `(adjacent ?l1 ?l2 ?dir)` facts in the static information.
    - Shortest path distance on the adjacency graph is a reasonable proxy for movement cost for both boxes (if they could move freely) and the robot.
    - The graph of locations defined by `adjacent` facts is connected for all relevant locations (robot, boxes, goals).

    # Heuristic Initialization
    - Extract the goal location for each box from the task goals. This creates a mapping from box name to its target location.
    - Build a graph representation of all locations mentioned in `adjacent` facts. The graph nodes are locations, and edges connect adjacent locations.
    - Pre-compute all-pairs shortest path distances between all pairs of locations in the graph using Breadth-First Search (BFS). These distances are stored in a dictionary.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the current location of the robot and each box from the current state facts.
    2. Initialize the total heuristic value to 0.
    3. Create a list to keep track of boxes that are not currently at their designated goal locations.
    4. Iterate through each box that has a specified goal location:
       - Get the box's current location from the state.
       - Get the box's goal location (stored during initialization).
       - If the current location is different from the goal location:
         - Add the pre-computed shortest path distance from the box's current location to its goal location to the total heuristic value. This estimates the minimum number of "box steps" needed.
         - Add the box to the list of misplaced boxes.
       - If the distance between the box's current location and goal location was not found (indicating they are in disconnected parts of the graph), return infinity as the state is likely unsolvable.
    5. If there are any misplaced boxes:
       - Find the misplaced box whose current location is closest to the robot's current location (using pre-computed distances).
       - Add the shortest path distance from the robot's current location to this closest misplaced box's location to the total heuristic value. This estimates the robot's effort to get to where it can start pushing a box.
       - If the distance between the robot's current location and any misplaced box location was not found, return infinity.
    6. The final heuristic value is the calculated total. If no boxes are misplaced, the heuristic is 0.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and building the location graph."""
        self.goals = task.goals
        static_facts = task.static

        # Store goal locations for each box.
        self.goal_locations = {}
        for goal in self.goals:
            # Goal facts are typically (at ?box ?location)
            if match(goal, "at", "*", "*"):
                _, box, location = get_parts(goal)
                self.goal_locations[box] = location

        # Build the graph of locations based on adjacent facts.
        self.location_graph = {}
        all_locations = set()

        # First pass to collect all unique locations mentioned in adjacent facts
        for fact in static_facts:
             if match(fact, "adjacent", "*", "*", "*"):
                 _, loc1, loc2, _ = get_parts(fact)
                 all_locations.add(loc1)
                 all_locations.add(loc2)

        # Initialize graph with all locations
        for loc in all_locations:
            self.location_graph[loc] = set()

        # Second pass to add edges based on adjacency
        for fact in static_facts:
            if match(fact, "adjacent", "*", "*", "*"):
                _, loc1, loc2, _ = get_parts(fact)
                # Add bidirectional edges as adjacency is symmetric
                self.location_graph[loc1].add(loc2)
                self.location_graph[loc2].add(loc1)

        # Pre-compute all-pairs shortest path distances.
        self.distances = {}
        for start_node in self.location_graph:
            self._bfs(start_node)

    def _bfs(self, start_node):
        """Performs BFS starting from start_node to compute distances to all reachable nodes."""
        q = deque([(start_node, 0)])
        visited = {start_node}
        self.distances[(start_node, start_node)] = 0

        while q:
            current_node, dist = q.popleft()

            # Ensure current_node exists in the graph before accessing neighbors
            if current_node not in self.location_graph:
                 continue # Should not happen if all_locations were collected correctly

            for neighbor in self.location_graph[current_node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    self.distances[(start_node, neighbor)] = dist + 1
                    q.append((neighbor, dist + 1))

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Find current location of robot and boxes
        robot_location = None
        current_box_locations = {}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at-robot":
                robot_location = parts[1]
            elif parts[0] == "at" and len(parts) == 3: # Check for (at ?obj ?loc)
                 obj, loc = parts[1], parts[2]
                 # Only track objects that are boxes and have a goal location
                 if obj in self.goal_locations:
                    current_box_locations[obj] = loc

        # If robot location is not found, the state is likely invalid or unreachable
        if robot_location is None:
             return float('inf')

        total_box_distance = 0
        misplaced_boxes = []

        # Calculate sum of box-goal distances for misplaced boxes
        for box, goal_loc in self.goal_locations.items():
            current_loc = current_box_locations.get(box)

            # If a box with a goal isn't found in the state, it's unreachable
            if current_loc is None:
                 return float('inf')

            if current_loc != goal_loc:
                # Look up the pre-computed distance
                dist_key = (current_loc, goal_loc)
                if dist_key in self.distances:
                    total_box_distance += self.distances[dist_key]
                    misplaced_boxes.append(box)
                else:
                    # Goal location is unreachable from the box's current location
                    return float('inf')

        # If all boxes are at their goals, the heuristic is 0.
        if not misplaced_boxes:
            return 0

        # Calculate distance from robot to the closest misplaced box
        min_robot_to_box_distance = float('inf')
        for box in misplaced_boxes:
            box_loc = current_box_locations[box]
            dist_key = (robot_location, box_loc)
            if dist_key in self.distances:
                 min_robot_to_box_distance = min(min_robot_to_box_distance, self.distances[dist_key])
            else:
                 # Robot cannot reach this box
                 return float('inf')

        # If min_robot_to_box_distance is still inf, it means robot cannot reach any misplaced box.
        if min_robot_to_box_distance == float('inf'):
             return float('inf')

        # The heuristic is the sum of box-goal distances and the robot-to-closest-misplaced-box distance.
        # This is non-admissible but provides a reasonable estimate for greedy search.
        return total_box_distance + min_robot_to_box_distance
