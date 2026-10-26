import collections
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

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
    if len(parts) < len(args) or (len(parts) > len(args) and args[-1] != '*'):
         return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing the shortest
    path distances for each box to its goal location and adding the shortest
    path distance from the robot to the nearest box that is not yet at its goal.

    # Assumptions
    - The heuristic uses shortest path distances on the static adjacency graph,
      ignoring dynamic obstacles (other boxes, robot) during distance calculation.
    - It assumes a fixed mapping between boxes and goals based on the problem's
      goal state definition.
    - It does not detect dead-end states where a box is pushed into a location
      from which it cannot be moved further or reach its goal.

    # Heuristic Initialization
    - Builds a graph representation of the locations based on `adjacent` facts.
    - Extracts the goal location for each box from the task's goal conditions.

    # Step-by-Step Thinking for Computing the Heuristic Value
    1.  **Build the Location Graph:** During initialization, parse all `adjacent`
        facts to create an adjacency list representation of the locations. This
        graph represents the possible movements for the robot and boxes (when pushed).
    2.  **Identify Box Goals:** During initialization, parse the goal conditions
        to find the target location for each box.
    3.  **Find Current State Information:** In the `__call__` method, extract the
        current location of the robot and each box from the given state.
    4.  **Calculate Box-to-Goal Distances:** For each box that is *not* currently
        at its goal location, calculate the shortest path distance from its current
        location to its goal location using BFS on the pre-built graph. Sum these
        distances. This represents a lower bound on the number of pushes required
        for each box individually, ignoring robot positioning.
    5.  **Calculate Robot-to-Nearest-Box Distance:** Find the box that is not yet
        at its goal and is closest to the robot (shortest path distance on the graph).
        Calculate the shortest path distance from the robot's current location to
        this closest box's location. This estimates the cost for the robot to get
        into a position to start pushing a box.
    6.  **Combine Distances:** The heuristic value is the sum of the total box-to-goal
        distance (from step 4) and the robot-to-nearest-box distance (from step 5).
        If all boxes are at their goals, the heuristic is 0. If any required location
        (box goal, or a box location from the robot) is unreachable in the graph,
        return infinity.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by building the location graph and extracting
        box goal locations.
        """
        self.goals = task.goals
        static_facts = task.static

        # Build the graph from adjacent facts
        self.graph = collections.defaultdict(set)
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "adjacent":
                loc1, loc2, direction = parts[1:]
                # Add bidirectional edges for movement
                self.graph[loc1].add(loc2)
                self.graph[loc2].add(loc1) # Assuming adjacency is symmetric

        # Store goal locations for each box
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at" and len(args) == 2:
                obj, location = args
                # Assuming goals are only for boxes
                if obj.startswith("box"): # Simple check for box type
                     self.goal_locations[obj] = location

        # Pre-calculate all-pairs shortest paths if graph is small enough,
        # or just use BFS on demand. BFS on demand is simpler and often
        # sufficient for typical Sokoban grid sizes. Let's use BFS on demand.

    def shortest_path_distance(self, start, end):
        """
        Calculates the shortest path distance between two locations using BFS.
        Returns float('inf') if the end is unreachable from the start.
        """
        if start == end:
            return 0
        if start not in self.graph or end not in self.graph:
             # One of the locations doesn't exist in the graph
             return float('inf')

        queue = collections.deque([(start, 0)])
        visited = {start}

        while queue:
            current_loc, dist = queue.popleft()

            if current_loc == end:
                return dist

            for neighbor in self.graph.get(current_loc, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))

        return float('inf') # End not reachable

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state

        # Find robot location
        robot_location = None
        for fact in state:
            if match(fact, "at-robot", "*"):
                robot_location = get_parts(fact)[1]
                break

        if robot_location is None:
             # Should not happen in a valid Sokoban state, but handle defensively
             return float('inf')

        # Find current box locations
        current_box_locations = {}
        for fact in state:
            if match(fact, "at", "box*", "*"):
                box, loc = get_parts(fact)[1:]
                current_box_locations[box] = loc

        total_box_distance = 0
        min_robot_to_box_distance = float('inf')
        boxes_not_at_goal = []

        # Calculate total distance for boxes not at goal and find nearest box
        for box, goal_location in self.goal_locations.items():
            current_box_location = current_box_locations.get(box)

            # If a box expected in the goal state is not in the current state,
            # or its location is unknown, this state is likely invalid or very far.
            if current_box_location is None:
                 return float('inf')

            if current_box_location != goal_location:
                boxes_not_at_goal.append(box)

                # Distance from box to its goal
                box_dist = self.shortest_path_distance(current_box_location, goal_location)
                if box_dist == float('inf'):
                    # Box cannot reach its goal - likely a dead end or unsolvable state
                    return float('inf')
                total_box_distance += box_dist

                # Distance from robot to this box
                robot_to_this_box_dist = self.shortest_path_distance(robot_location, current_box_location)
                if robot_to_this_box_dist == float('inf'):
                    # Robot cannot reach this box - likely an unsolvable state
                    return float('inf')
                min_robot_to_box_distance = min(min_robot_to_box_distance, robot_to_this_box_dist)

        # If all boxes are at their goals, the heuristic is 0
        if not boxes_not_at_goal:
            return 0

        # The heuristic is the sum of box-to-goal distances plus the robot's
        # distance to the nearest box that needs moving.
        # This is non-admissible as it doesn't account for robot maneuvering
        # around boxes or other complexities, but provides a reasonable estimate.
        return total_box_distance + min_robot_to_box_distance

