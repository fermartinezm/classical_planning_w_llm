from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from itertools import product
import heapq

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at p1 l1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the transport domain.

    # Summary
    This heuristic estimates the number of actions needed to transport all packages to their respective goal locations.

    # Assumptions:
    - Packages can be either on the ground or inside a vehicle.
    - Vehicles can move between connected locations.
    - Each package must be transported independently, considering loading and unloading actions.

    # Heuristic Initialization
    - Extracts goal locations for each package.
    - Constructs a graph representation of the road network for efficient distance calculation.

    # Step-by-Step Thinking for Computing Heuristic Value
    1. **Extract Goal Information**: Identify the goal location for each package.
    2. **Current State Analysis**: For each package, determine its current location and whether it is inside a vehicle.
    3. **Distance Calculation**: Use Dijkstra's algorithm to find the shortest path between locations.
    4. **Action Counting**: For each package, calculate the number of actions required to move it to the goal location, considering loading, unloading, and transportation actions.
    5. **Summing Actions**: Sum the actions for all packages to get the total heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting:
        - Goal locations for each package.
        - Static facts (road connections) to build a graph representation.
        """
        self.goals = task.goals  # Goal conditions
        static_facts = task.static  # Static facts (road connections)

        # Build graph representation of the road network
        self.graph = {}
        for fact in static_facts:
            if match(fact, "road", "*", "*"):
                loc1, loc2 = get_parts(fact)[1], get_parts(fact)[2]
                if loc1 not in self.graph:
                    self.graph[loc1] = []
                if loc2 not in self.graph:
                    self.graph[loc2] = []
                self.graph[loc1].append(loc2)
                self.graph[loc2].append(loc1)

        # Precompute all pairs shortest paths using Dijkstra's algorithm
        self.distances = {}
        for loc in self.graph:
            self.distances[loc] = {}
            heap = []
            heapq.heappush(heap, (0, loc))
            self.distances[loc][loc] = 0
            visited = set()
            while heap:
                dist, u = heapq.heappop(heap)
                if u in visited:
                    continue
                visited.add(u)
                for v in self.graph[u]:
                    if v not in self.distances[loc]:
                        new_dist = dist + 1
                        if new_dist < self.distances[loc].get(v, float('inf')):
                            self.distances[loc][v] = new_dist
                            heapq.heappush(heap, (new_dist, v))

        # Store goal locations for each package
        self.goal_locations = {}
        for goal in self.goals:
            predicate, package, location = get_parts(goal)
            if predicate == "at":
                self.goal_locations[package] = location

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state  # Current world state

        # Track where packages and vehicles are currently located
        current_locations = {}
        for fact in state:
            predicate, *args = get_parts(fact)
            if predicate in ["at", "in"]:
                obj, location = args
                current_locations[obj] = location

        total_cost = 0  # Initialize action cost counter

        # Process each package
        for package in self.goal_locations:
            goal_location = self.goal_locations[package]

            # Skip if already at goal
            if package in current_locations and current_locations[package] == goal_location:
                continue

            # Find the current location of the package
            current_location = None
            for fact in state:
                if match(fact, "at", package, "*"):
                    current_location = get_parts(fact)[2]
                    break
            if current_location is None:
                continue  # Package not present in state (should not happen)

            # Check if the package is inside a vehicle
            in_vehicle = any(fact for fact in state if match(fact, "in", package, "*"))

            # Calculate the distance between current and goal locations
            if current_location not in self.distances or goal_location not in self.distances[current_location]:
                distance = float('inf')
            else:
                distance = self.distances[current_location].get(goal_location, float('inf'))

            # If the package is in a vehicle, it needs to be unloaded first
            if in_vehicle:
                total_cost += 1  # Unloading action

            # Calculate the number of moves required
            if distance == float('inf'):
                # No path exists (should not happen in solvable states)
                continue
            else:
                total_cost += distance  # Driving actions

            # If the package is not already at the goal, add loading action
            if current_locations.get(package, None) != goal_location:
                total_cost += 1  # Loading action

        return total_cost
