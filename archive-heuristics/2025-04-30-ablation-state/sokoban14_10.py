from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque

# Helper functions (defined outside the class)
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
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# Define the heuristic class
class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing two components for each box not yet at its goal:
    1. The shortest path distance from the box's current location to its goal location.
    2. The shortest path distance from the robot's current location to the box's current location.
    The distances are calculated on the static grid graph defined by `adjacent` predicates, ignoring dynamic obstacles.

    # Assumptions
    - The grid structure and connectivity are defined by the `adjacent` predicates.
    - Shortest path distances are calculated on this static grid graph, ignoring dynamic obstacles (`clear` predicate).
    - All locations mentioned in `adjacent` facts are part of the traversable grid.
    - Goal locations for boxes are reachable from their initial locations in solvable problems.
    - The robot location is always present in a valid state.
    - Boxes are always at a location (not carried).

    # Heuristic Initialization
    - Extracts the goal locations for each box from the task goals.
    - Builds a graph representing the grid connectivity based on `adjacent` facts.
    - Computes all-pairs shortest path distances between all locations in the graph using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    1. Get the current location of the robot by finding the fact `(at-robot ?l)` in the state.
    2. Get the current location of each box by finding facts `(at ?b ?l)` in the state.
    3. Initialize the total heuristic value `total_cost` to 0.
    4. Iterate through each box that has a specified goal location (`self.goal_locations`).
       a. Get the box name and its goal location.
       b. Find the box's current location in the state using the `box_locations` map.
       c. If the box's current location is the same as its goal location, this box contributes 0 to the heuristic. Continue to the next box.
       d. If the box is not at its goal:
          i. Calculate the shortest path distance from the box's current location to its goal location using the precomputed `self.distances`. If the goal is unreachable from the box in the static graph, return `float('inf')` as the state is likely unsolvable. Add this distance to `total_cost`.
          ii. Calculate the shortest path distance from the robot's current location to the box's current location using the precomputed `self.distances`. If the box is unreachable from the robot in the static graph, return `float('inf')` as the state is likely unsolvable. Add this distance to `total_cost`.
    5. Return the final `total_cost`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and building the grid graph
        to precompute all-pairs shortest path distances.
        """
        # The set of facts that must hold in goal states.
        self.goals = task.goals
        # Static facts are not affected by actions.
        static_facts = task.static

        # Store goal locations for each box.
        # We assume goals are of the form (at boxX locY)
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                box, location = args
                self.goal_locations[box] = location

        # Build the adjacency graph from static 'adjacent' facts.
        # graph[loc1] = [(loc2, dir), ...]
        self.graph = {}
        # Collect all unique locations first from adjacent facts
        all_locations = set()
        for fact in static_facts:
            if match(fact, "adjacent", "*", "*", "*"):
                _, loc1, loc2, _ = get_parts(fact)
                all_locations.add(loc1)
                all_locations.add(loc2)

        # Initialize graph with all locations found
        for loc in all_locations:
            self.graph[loc] = []

        # Add edges based on adjacent facts
        for fact in static_facts:
            if match(fact, "adjacent", "*", "*", "*"):
                _, loc1, loc2, dir = get_parts(fact)
                self.graph[loc1].append((loc2, dir))

        # Compute all-pairs shortest path distances using BFS.
        self.distances = {}
        for start_loc in self.graph:
            self.distances[start_loc] = {}
            q = deque([(start_loc, 0)])
            visited = {start_loc}

            while q:
                current_loc, d = q.popleft()
                self.distances[start_loc][current_loc] = d

                # Iterate through neighbors in the graph
                for neighbor_loc, _ in self.graph.get(current_loc, []):
                    if neighbor_loc not in visited:
                        visited.add(neighbor_loc)
                        q.append((neighbor_loc, d + 1))

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Find robot location
        robot_location = None
        for fact in state:
            if match(fact, "at-robot", "*"):
                _, robot_location = get_parts(fact)
                break
        # Assuming robot location is always present in a valid state and is one of the locations in the graph

        # Find current box locations
        box_locations = {}
        for fact in state:
            if match(fact, "at", "*", "*"):
                _, box, location = get_parts(fact)
                box_locations[box] = location

        total_cost = 0  # Initialize action cost counter.

        # Calculate heuristic for each box that needs to reach a goal location
        for box, goal_location in self.goal_locations.items():
            current_box_location = box_locations.get(box)

            # If the box is not found in the current state (shouldn't happen in Sokoban)
            # or is already at the goal, it contributes 0 to the heuristic.
            if current_box_location is None or current_box_location == goal_location:
                continue

            # Add box-to-goal distance
            # Check if goal is reachable from box location in the static graph
            if current_box_location not in self.distances or goal_location not in self.distances[current_box_location]:
                 # If goal is unreachable from the box's current location,
                 # this state is likely a dead end or unsolvable.
                 return float('inf') # Return infinity or a very large number

            total_cost += self.distances[current_box_location][goal_location]

            # Add robot-to-box distance
            # Check if box is reachable from robot location in the static graph
            if robot_location not in self.distances or current_box_location not in self.distances[robot_location]:
                 # If the box is unreachable from the robot's current location,
                 # the robot cannot interact with it. This state is likely unsolvable.
                 return float('inf') # Return infinity or a very large number

            total_cost += self.distances[robot_location][current_box_location]

        return total_cost
