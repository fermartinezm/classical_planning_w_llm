from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic

# Helper functions to parse PDDL facts represented as strings
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
    # Ensure the number of parts matches the number of arguments for a meaningful match
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal state by considering:
    1. The sum of the shortest path distances for each box from its current location
       to its goal location.
    2. The shortest path distance for the robot from its current location to the
       nearest box that is not yet at its goal.

    The total heuristic value is the sum of these two components.

    # Assumptions
    - The grid structure is defined by `adjacent` facts.
    - Movement is possible between any two adjacent locations for the robot (ignoring
      `clear` predicates for heuristic calculation speed) and for boxes (ignoring
      obstacles and robot position for heuristic calculation speed).
    - There is a one-to-one mapping between boxes and goal locations.
    - The shortest path distances are calculated on the full grid graph defined by
      `adjacent` facts, ignoring dynamic obstacles (other boxes, robot, walls not
      defined by lack of adjacency). This makes the heuristic non-admissible but
      fast to compute.

    # Heuristic Initialization
    - Build the grid graph from `adjacent` facts.
    - Pre-compute all-pairs shortest path distances on this graph using BFS.
    - Parse goal conditions to map each box to its specific goal location.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the current location of the robot.
    2. Identify the current location of each box that has a goal location.
    3. Initialize `total_box_distance = 0`.
    4. Initialize `min_robot_to_box_distance = infinity`.
    5. Iterate through each box and its corresponding goal location:
       - If the box is not at its goal location:
         - Add the pre-computed shortest distance from the box's current location
           to its goal location to `total_box_distance`.
         - Calculate the pre-computed shortest distance from the robot's current
           location to the box's current location.
         - Update `min_robot_to_box_distance` with the minimum of its current
           value and the robot-to-box distance just calculated.
    6. If no boxes are off-goal, the heuristic is 0.
    7. Otherwise, the heuristic is `total_box_distance + min_robot_to_box_distance`.
       If any required distance was infinite (locations are disconnected), the
       heuristic should be infinite.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting:
        - Goal locations for each box.
        - Static facts (`adjacent` relationships).
        - Pre-compute distances on the grid graph.
        """
        self.goals = task.goals  # Goal conditions.
        self.static = task.static  # Facts that are not affected by actions.

        # Build the grid graph from adjacent facts
        self.graph = {}
        locations = set()
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == 'adjacent' and len(parts) == 4:
                loc1, loc2, direction = parts[1], parts[2], parts[3]
                locations.add(loc1)
                locations.add(loc2)
                # Add directed edge
                self.graph.setdefault(loc1, []).append(loc2)
                # Add reverse edge as movement is bidirectional in Sokoban grid
                self.graph.setdefault(loc2, []).append(loc1)

        self.locations = list(locations) # Store locations list

        # Pre-compute all-pairs shortest paths using BFS
        self.distances = {}
        for start_loc in self.locations:
            self.distances[start_loc] = self._bfs(start_loc)

        # Parse goal locations for each box
        self.goal_locations = {}
        for goal in self.goals:
            parts = get_parts(goal)
            # Assuming goal is (at box location)
            if parts[0] == 'at' and len(parts) == 3:
                box, location = parts[1], parts[2]
                self.goal_locations[box] = location

    def _bfs(self, start_node):
        """Perform BFS from start_node to find distances to all reachable nodes."""
        distances = {node: float('inf') for node in self.locations}
        distances[start_node] = 0
        queue = deque([start_node])

        while queue:
            current_node = queue.popleft()

            # Check if current_node exists in the graph keys (it should if it's in self.locations)
            if current_node in self.graph:
                for neighbor in self.graph[current_node]:
                    if distances[neighbor] == float('inf'):
                        distances[neighbor] = distances[current_node] + 1
                        queue.append(neighbor)
        return distances

    def get_distance(self, loc1, loc2):
        """Get pre-computed distance between two locations."""
        # Handle cases where locations might not be in the pre-computed distances (e.g., invalid state)
        if loc1 in self.distances and loc2 in self.distances[loc1]:
             return self.distances[loc1][loc2]
        # If locations are not in the pre-computed graph, they are unreachable
        return float('inf')

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Find robot location
        robot_location = None
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at-robot' and len(parts) == 2:
                robot_location = parts[1]
                break

        # If robot location isn't found, the state is likely invalid or terminal (goal)
        # If it's not the goal state, it's likely unsolvable from here.
        if robot_location is None:
             # Check if goal is reached. If not, it's an invalid state.
             if not self.goals <= state:
                 return float('inf')
             else:
                 return 0 # Goal reached

        # Find box locations for boxes that have a goal
        box_locations = {}
        for fact in state:
            parts = get_parts(fact)
            # Check for (at box location) facts where box is one of the goal boxes
            if parts[0] == 'at' and len(parts) == 3 and parts[1] in self.goal_locations:
                box_locations[parts[1]] = parts[2]

        total_box_distance = 0
        min_robot_to_box_distance = float('inf')
        off_goal_boxes_exist = False

        # Calculate heuristic components for each box that needs to reach a goal
        for box, goal_loc in self.goal_locations.items():
            current_loc = box_locations.get(box) # Get current location of this specific box

            # If a goal box is not found in the state, it's an invalid state
            if current_loc is None:
                return float('inf')

            if current_loc != goal_loc:
                off_goal_boxes_exist = True

                # Distance for the box to its goal
                box_dist = self.get_distance(current_loc, goal_loc)
                if box_dist == float('inf'):
                    # If a box cannot reach its goal, the state is unsolvable
                    return float('inf')
                total_box_distance += box_dist

                # Distance for the robot to the box
                robot_to_box_dist = self.get_distance(robot_location, current_loc)
                if robot_to_box_dist == float('inf'):
                     # If robot cannot reach this off-goal box, consider it for min distance
                     # but the state might still be solvable if robot can reach *another* box
                     pass # Keep min_robot_to_box_distance as inf if no box is reachable
                else:
                    min_robot_to_box_distance = min(min_robot_to_box_distance, robot_to_box_dist)


        # If all boxes are at their goals, the heuristic is 0
        if not off_goal_boxes_exist:
            return 0

        # If there are off-goal boxes but the robot cannot reach *any* of them
        if min_robot_to_box_distance == float('inf'):
             return float('inf')

        # The heuristic is the sum of box distances to goals plus the robot's distance
        # to the nearest off-goal box.
        return total_box_distance + min_robot_to_box_distance

