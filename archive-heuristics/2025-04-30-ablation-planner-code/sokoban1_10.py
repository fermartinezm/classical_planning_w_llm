from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque

# Helper functions to parse PDDL facts
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
    # Ensure the number of parts matches the number of args, unless args contains wildcards
    # A simpler check: just zip and compare, fnmatch handles different lengths implicitly
    # by stopping at the shortest sequence. This is usually fine for PDDL facts.
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal state
    by summing two components for each box that is not yet at its goal location:
    1. The shortest path distance from the box's current location to its goal location.
       This estimates the minimum number of push actions needed for the box itself.
    2. The shortest path distance from the robot's current location to the box's current location.
       This estimates the minimum number of move actions needed for the robot to reach the box.

    # Assumptions
    - The heuristic is a relaxation that ignores dynamic obstacles (other boxes, robot)
      when calculating shortest paths. It assumes free movement on the underlying grid graph.
    - It ignores the specific positioning required for the robot to push a box (i.e., being
      on the correct side). It only considers the distance to the box's location.
    - It does not detect or penalize dead-end states for boxes.
    - Action costs for move and push are assumed to be 1.

    # Heuristic Initialization
    - Extracts the goal locations for each box from the task's goal conditions.
    - Builds an undirected graph representing the grid connectivity based on the `adjacent`
      static facts.
    - Precomputes the shortest path distances between all pairs of locations in the graph
      using Breadth-First Search (BFS).

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current location of the robot.
    2. Identify the current location of each box.
    3. Initialize the total heuristic value to 0.
    4. For each box specified in the goal conditions:
       a. Get the box's current location and its goal location (stored during initialization).
       b. If the box exists in the state and its current location is not the same as its goal location:
          i. Calculate the shortest path distance from the box's current location to its goal location
             using the precomputed distances. Add this distance to the total heuristic.
          ii. Calculate the shortest path distance from the robot's current location to the box's
              current location using the precomputed distances. Add this distance to the total heuristic.
    5. Return the total heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and building the location graph.
        """
        super().__init__(task)

        # Store goal locations for each box
        self.goal_locations = {}
        for goal in task.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                box, location = args
                self.goal_locations[box] = location

        # Build the graph from adjacent facts
        self.graph = {}
        all_locations = set()

        # Collect all locations mentioned in adjacent facts
        for fact in task.static:
            if match(fact, "adjacent", "*", "*", "*"):
                _, loc1, loc2, _ = get_parts(fact)
                all_locations.add(loc1)
                all_locations.add(loc2)
                self.graph.setdefault(loc1, []).append(loc2)
                # Assuming adjacency is symmetric, add the reverse edge
                self.graph.setdefault(loc2, []).append(loc1)

        # Collect all locations mentioned in initial state facts
        for fact in task.initial_state:
             if match(fact, "at-robot", "*"):
                 _, loc = get_parts(fact)
                 all_locations.add(loc)
                 self.graph.setdefault(loc, []) # Add node even if no edges yet
             elif match(fact, "at", "*", "*"):
                 _, _, loc = get_parts(fact)
                 all_locations.add(loc)
                 self.graph.setdefault(loc, []) # Add node even if no edges yet

        # Collect all locations mentioned in goal state facts
        for goal in task.goals:
             if match(goal, "at", "*", "*"):
                 _, _, loc = get_parts(goal)
                 all_locations.add(loc)
                 self.graph.setdefault(loc, []) # Add node even if no edges yet

        # Compute all-pairs shortest paths using BFS
        self.distances = {}
        for start_node in all_locations:
            self.distances[start_node] = self._bfs(start_node, all_locations)

    def _bfs(self, start_node, all_locations):
        """
        Performs BFS from a start node to find distances to all reachable nodes.
        Returns a dictionary {location: distance}.
        """
        distances = {loc: float('inf') for loc in all_locations}
        distances[start_node] = 0
        queue = deque([start_node])

        while queue:
            current_node = queue.popleft()

            # Get neighbors from the graph, handle nodes with no edges
            neighbors = self.graph.get(current_node, [])

            for neighbor in neighbors:
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)

        return distances

    def __call__(self, node):
        """
        Compute the domain-dependent heuristic value for the given state.
        """
        state = node.state

        # Find robot location
        robot_location = None
        for fact in state:
            if match(fact, "at-robot", "*"):
                _, robot_location = get_parts(fact)
                break

        # If robot location is not found, the state is invalid or unsolvable
        if robot_location is None:
             return float('inf') # Indicate unsolvable state

        # Find current box locations
        current_box_locations = {}
        for fact in state:
            if match(fact, "at", "*", "*"):
                _, box, location = get_parts(fact)
                current_box_locations[box] = location

        total_heuristic = 0

        # Sum distances for each misplaced box
        for box, goal_location in self.goal_locations.items():
            current_location = current_box_locations.get(box)

            # If the box exists in the state and is not at its goal location
            if current_location and current_location != goal_location:
                # Add box-to-goal distance
                # Check if current_location and goal_location are in the precomputed distances
                if current_location not in self.distances or goal_location not in self.distances[current_location]:
                     # This implies a location exists but wasn't in the graph construction, or goal is unreachable
                     return float('inf') # Indicate unsolvable state

                box_dist = self.distances[current_location][goal_location]

                if box_dist == float('inf'):
                     # If goal is unreachable for the box, this state is likely unsolvable
                     return float('inf') # Or a large constant

                total_heuristic += box_dist

                # Add robot-to-box distance
                # Check if robot_location and current_location are in the precomputed distances
                if robot_location not in self.distances or current_location not in self.distances[robot_location]:
                     # This implies a location exists but wasn't in the graph construction, or box is unreachable from robot
                     return float('inf') # Indicate unsolvable state

                robot_dist = self.distances[robot_location][current_location]

                if robot_dist == float('inf'):
                     # If robot cannot reach the box, this state is likely unsolvable
                     return float('inf') # Or a large constant

                total_heuristic += robot_dist

        return total_heuristic
