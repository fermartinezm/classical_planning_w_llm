from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque
import math

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

# Helper function to match PDDL facts
def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at obj loc)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing the shortest
    path distance for each box to its goal location and adding the shortest
    path distance from the robot to the nearest box that is not yet at its goal.
    Distances are computed on the graph of locations defined by adjacency relations.

    # Assumptions
    - The goal state specifies the target location for each box using `(at boxX goal_locX)` facts.
    - The environment is represented as a graph of locations connected by `adjacent` facts.
    - Adjacency is symmetric (if A is adjacent to B, B is adjacent to A).
    - Shortest path distances between locations can be computed on this graph.
    - The heuristic ignores potential dead-ends for boxes (locations from which a box cannot be pushed to its goal) and complex robot maneuvering around obstacles (other boxes).
    - All boxes listed in the goal are present in the state.

    # Heuristic Initialization
    - Build a graph of locations based on `adjacent` facts.
    - Precompute shortest path distances between all pairs of locations using BFS on the location graph.
    - Extract the goal location for each box from the task's goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current location of the robot by finding the fact `(at-robot ?l)`.
    2. Identify the current location of each box by finding facts `(at ?b ?l)` for all boxes `?b` that appear in the goal conditions.
    3. Initialize the total heuristic cost `h` to 0.
    4. Initialize the minimum distance from the robot to any misplaced box `min_robot_to_box_dist` to infinity.
    5. Initialize a counter for misplaced boxes `misplaced_boxes_count` to 0.
    6. For each box `b` and its corresponding goal location `G_b` identified during initialization:
       a. Get the current location `L_b` of box `b` from the state.
       b. If `L_b` is not equal to `G_b`:
          i. Increment `misplaced_boxes_count`.
          ii. Get the precomputed shortest path distance `d_box` from `L_b` to `G_b` using the precomputed distances. If `d_box` is infinity, the goal is unreachable for this box, return infinity immediately.
          iii. Add `d_box` to the total cost `h`.
          iv. Get the precomputed shortest path distance `d_robot_to_box` from the robot's current location `L_r` to `L_b`. If `d_robot_to_box` is infinity, the robot cannot reach this box, return infinity immediately.
          v. Update `min_robot_to_box_dist = min(min_robot_to_box_dist, d_robot_to_box)`.
    7. If `misplaced_boxes_count` is greater than 0:
       a. Add `min_robot_to_box_dist` to the total cost `h`.
    8. Return the total cost `h`.
    """

    def __init__(self, task):
        """Initialize the heuristic by building the location graph and precomputing distances."""
        self.goals = task.goals
        static_facts = task.static

        # Build the location graph from adjacent facts
        self.location_graph = {}
        locations = set()
        for fact in static_facts:
            if match(fact, "adjacent", "*", "*", "*"):
                _, loc1, loc2, _ = get_parts(fact)
                locations.add(loc1)
                locations.add(loc2)
                if loc1 not in self.location_graph:
                    self.location_graph[loc1] = set()
                if loc2 not in self.location_graph:
                    self.location_graph[loc2] = set()
                # Add edges in both directions assuming adjacency is symmetric for movement
                self.location_graph[loc1].add(loc2)
                self.location_graph[loc2].add(loc1)

        self.locations = list(locations) # List of all locations

        # Precompute all-pairs shortest paths using BFS
        self.distances = {}
        for start_loc in self.locations:
            self.distances[start_loc] = self._bfs(start_loc)

        # Store goal locations for each box
        self.goal_locations = {}
        for goal in self.goals:
            # Goal facts are typically (at boxX goal_locX)
            if match(goal, "at", "*", "*"):
                 _, box, location = get_parts(goal)
                 self.goal_locations[box] = location
            # Note: This heuristic assumes goals only specify box locations.
            # If robot location is also specified in the goal, this needs adjustment.

    def _bfs(self, start_location):
        """Perform BFS from a start location to find distances to all other reachable locations."""
        distances = {loc: math.inf for loc in self.locations}
        if start_location not in self.locations:
             # Start location is not in the graph, no reachable locations except itself (dist 0)
             # This might happen if the initial state has the robot/box in an isolated location.
             # The heuristic will return inf if a goal is unreachable.
             return distances # All distances remain inf

        distances[start_location] = 0
        queue = deque([start_location])

        while queue:
            current_loc = queue.popleft()

            if current_loc in self.location_graph: # Check if current_loc has neighbors
                for neighbor in self.location_graph[current_loc]:
                    if distances[neighbor] == math.inf:
                        distances[neighbor] = distances[current_loc] + 1
                        queue.append(neighbor)
        return distances

    def get_distance(self, loc1, loc2):
        """Get the precomputed shortest path distance between two locations."""
        # Handle cases where locations might not be in the precomputed distances dict
        # (e.g., isolated locations not part of any 'adjacent' fact)
        if loc1 not in self.distances or loc2 not in self.distances[loc1]:
             if loc1 == loc2:
                 # Distance to self is 0, even if isolated
                 return 0
             # If either location is not in the graph or unreachable, distance is infinity
             return math.inf
        return self.distances[loc1][loc2]

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Find robot location
        robot_location = None
        for fact in state:
            if match(fact, "at-robot", "*"):
                _, robot_location = get_parts(fact)
                break
        # If robot_location is None, the state is likely invalid for this domain.
        # Returning infinity indicates an unreachable or invalid state.
        if robot_location is None:
             return math.inf

        # Find box locations for boxes relevant to the goal
        box_locations = {}
        for fact in state:
            if match(fact, "at", "*", "*"):
                _, obj, loc = get_parts(fact)
                # Only consider objects that are boxes and have a goal location
                if obj in self.goal_locations:
                    box_locations[obj] = loc

        total_cost = 0
        min_robot_to_box_dist = math.inf
        misplaced_boxes_count = 0

        # Calculate cost for each box that is not at its goal
        for box, goal_location in self.goal_locations.items():
            current_location = box_locations.get(box)

            # If a box from the goal is not found in the state, it's an invalid state.
            # Return infinity.
            if current_location is None:
                 return math.inf

            if current_location != goal_location:
                misplaced_boxes_count += 1
                # Add box distance to goal
                box_dist = self.get_distance(current_location, goal_location)
                if box_dist == math.inf:
                    # Box is in a location from which goal is unreachable
                    return math.inf

                total_cost += box_dist

                # Calculate robot distance to this box
                robot_to_box_dist = self.get_distance(robot_location, current_location)
                if robot_to_box_dist == math.inf:
                     # Robot cannot reach the box
                     return math.inf

                min_robot_to_box_dist = min(min_robot_to_box_dist, robot_to_box_dist)

        # Add robot distance if there are misplaced boxes
        if misplaced_boxes_count > 0:
             # min_robot_to_box_dist must be finite here because we checked inside the loop
             total_cost += min_robot_to_box_dist

        # The heuristic is 0 if and only if misplaced_boxes_count is 0,
        # which means all boxes are at their goal locations.
        return total_cost
