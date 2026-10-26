# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic
# If running standalone or in a different environment, you might need a dummy definition:
# class Heuristic:
#     def __init__(self, task):
#         self.goals = task.goals
#         self.static = task.static
#     def __call__(self, node):
#         raise NotImplementedError

from fnmatch import fnmatch
from collections import deque

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    if not fact or not isinstance(fact, str) or fact[0] != '(' or fact[-1] != ')':
        # Handle cases that are not standard PDDL facts represented as strings
        # print(f"Warning: get_parts received non-standard fact: {fact}") # Optional: for debugging
        return []
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at package1 location1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions required to move each package
    from its current location to its goal location, summing the costs for all
    packages not yet at their goal. It considers the cost of picking up, dropping,
    and driving.

    # Assumptions
    - The road network defined by `(road l1 l2)` facts is undirected and connected
      for all locations relevant to package movements in solvable problems.
    - Vehicle capacity constraints are ignored. It is assumed that a suitable
      vehicle is available for pickup/dropoff when needed at the required location.
    - The cost of each action (drive, pick-up, drop) is 1.
    - The task goal consists primarily of `(at package location)` facts.

    # Heuristic Initialization
    - Extracts the goal location for each package from the task's goal conditions.
    - Builds the road network graph from the static `(road l1 l2)` facts.
    - Computes the shortest path distance between all pairs of locations in the
      road network using Breadth-First Search (BFS). These distances represent
      the minimum number of `drive` actions required to travel between locations.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current location or container (vehicle) for every package
       and the current location for every vehicle by examining the state's facts (`(at ...)` and `(in ...)`).
    2. Initialize the total heuristic cost to 0.
    3. For each package that has a defined goal location (extracted during initialization):
       a. Get the package's goal location.
       b. Get the package's current status (location if on ground, or vehicle name if inside).
       c. If the package is currently at its goal location on the ground, its contribution to the heuristic is 0. Continue to the next package.
       d. If the package is on the ground at `loc_p_current` (which is not the goal location):
          - The estimated cost for this package is 1 (pick-up) + shortest_path_distance(`loc_p_current`, `loc_p_goal`) (drive) + 1 (drop).
       e. If the package is inside vehicle `v`, and vehicle `v` is currently at `loc_v_current`:
          - The estimated cost for this package is shortest_path_distance(`loc_v_current`, `loc_p_goal`) (drive) + 1 (drop).
          - Note: If `loc_v_current` is already the goal location, the drive cost is 0, and the cost is just 1 (drop).
       f. If the shortest path distance required in steps d or e is infinite (meaning the goal location is unreachable from the current location), the state is likely a dead end, and the heuristic should return infinity.
       g. Add the calculated cost for this package to the total heuristic cost.
    4. Return the total heuristic cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and building
        the road network for shortest path calculations.
        """
        self.goals = task.goals
        self.static = task.static

        # Store goal locations for each package.
        self.package_goals = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            # Assuming goal facts are primarily (at package location)
            if predicate == "at" and len(args) == 2:
                 # We assume the first arg is the object (package) and the second is the location.
                 obj, location = args
                 self.package_goals[obj] = location


        # Build the road network graph from static facts.
        # The graph will be an adjacency list: {location: [neighbor1, neighbor2, ...]}
        self.road_graph = {}
        all_locations = set() # Collect all locations mentioned in road facts

        for fact in self.static:
            predicate, *args = get_parts(fact)
            if predicate == "road" and len(args) == 2:
                l1, l2 = args
                # Roads are typically bidirectional in this domain
                self.road_graph.setdefault(l1, []).append(l2)
                self.road_graph.setdefault(l2, []).append(l1) # Assuming bidirectional roads
                all_locations.add(l1)
                all_locations.add(l2)

        # Ensure all locations mentioned are keys in the graph, even if they have no roads
        # (though road facts usually come in pairs, this makes it robust)
        for loc in all_locations:
             self.road_graph.setdefault(loc, [])

        # Compute all-pairs shortest paths on the road network.
        # Store as {(loc1, loc2): distance}
        self.shortest_paths = self._compute_all_pairs_shortest_paths(self.road_graph)

    def _compute_all_pairs_shortest_paths(self, graph):
        """Computes shortest path distances between all pairs of nodes in the graph."""
        all_paths = {}
        # Iterate over all locations that are keys in the graph
        for start_node in graph.keys():
            paths_from_start = self._bfs(graph, start_node)
            for end_node, dist in paths_from_start.items():
                if dist != float('inf'):
                     all_paths[(start_node, end_node)] = dist
        return all_paths

    def _bfs(self, graph, start_node):
        """Computes shortest path distances from start_node to all other nodes."""
        # Initialize distances for all locations that are keys in the graph
        distances = {node: float('inf') for node in graph.keys()}

        if start_node not in distances:
             # If the start_node is not a key in the graph (e.g., an isolated location
             # not mentioned as a source in any road fact, but maybe as a destination),
             # we can't start BFS from it to find paths to others.
             # Distance to itself is 0, others remain inf.
             # This case should be rare in well-formed transport problems.
             # Add it to distances if it's not there.
             distances[start_node] = 0
             # No neighbors to explore from an isolated node not in graph keys.
             return distances


        distances[start_node] = 0
        queue = deque([start_node])

        while queue:
            current_node = queue.popleft()

            # Check if current_node has neighbors in the graph
            if current_node in graph:
                for neighbor in graph[current_node]:
                    # Only update if we found a shorter path (always true with BFS on unweighted graph)
                    # and if the neighbor is a known location (present in distances keys)
                    if neighbor in distances and distances[neighbor] == float('inf'):
                        distances[neighbor] = distances[current_node] + 1
                        queue.append(neighbor)
        return distances


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Track where packages and vehicles are currently located or contained.
        # {object_name: location_name_or_vehicle_name}
        current_locations = {}
        # We don't strictly need vehicle_contents for this heuristic,
        # just the location of the vehicle if a package is inside it.

        for fact in state:
            predicate, *args = get_parts(fact)
            if predicate == "at" and len(args) == 2:
                 # (at ?x - locatable ?v - location)
                 obj, location = args
                 current_locations[obj] = location
            elif predicate == "in" and len(args) == 2:
                 # (in ?x - package ?v - vehicle)
                 package, vehicle = args
                 current_locations[package] = vehicle # Package is inside a vehicle


        total_cost = 0  # Initialize action cost counter.

        # Iterate through packages that have a goal location defined
        for package, goal_location in self.package_goals.items():
            # Check if the package exists in the current state's locations/containers
            if package not in current_locations:
                 # This package is not mentioned in any 'at' or 'in' fact.
                 # This shouldn't happen in valid states for goal packages.
                 # Treat as unreachable or error.
                 # print(f"Warning: Goal package {package} not found in state.") # Optional: for debugging
                 return float('inf') # Or a large constant

            current_status = current_locations[package]

            # If the package is already at its goal location on the ground, it contributes 0.
            if current_status == goal_location:
                continue

            # Package is not at its goal location. Calculate cost to move it.
            package_cost = 0

            # Case 1: Package is on the ground at current_status (which is a location)
            # Check if current_status is a known location in our road graph
            if current_status in self.road_graph:
                loc_p_current = current_status
                # Cost: Pick-up (1) + Drive (dist) + Drop (1)
                drive_cost = self.shortest_paths.get((loc_p_current, goal_location), float('inf'))

                if drive_cost == float('inf'):
                    # Goal location is unreachable from current location.
                    # This state is likely a dead end for this package.
                    # A high heuristic value is appropriate.
                    return float('inf') # Or a large constant

                package_cost = 1 + drive_cost + 1

            # Case 2: Package is inside a vehicle (current_status is a vehicle name)
            else: # current_status is a vehicle name
                vehicle_name = current_status
                # Find the vehicle's current location
                if vehicle_name not in current_locations:
                    # Vehicle is not 'at' any location? This shouldn't happen in valid states.
                    # Treat as unreachable or error.
                     # print(f"Warning: Vehicle {vehicle_name} containing package {package} not found at any location.") # Optional: for debugging
                     return float('inf') # Or a large constant

                loc_v_current = current_locations[vehicle_name]

                # Cost: Drive (dist) + Drop (1)
                drive_cost = self.shortest_paths.get((loc_v_current, goal_location), float('inf'))

                if drive_cost == float('inf'):
                    # Goal location is unreachable from vehicle's current location.
                    return float('inf') # Or a large constant

                package_cost = drive_cost + 1

            total_cost += package_cost

        # The heuristic is the sum of costs for packages not at their goal.
        # If all packages in self.package_goals are at their goal, total_cost will be 0.
        # This assumes the task goal is *only* that these packages are at these locations.
        # If the task goal includes other conditions, this heuristic might not be 0 at the goal state.
        # However, for typical transport problems, package delivery is the main goal.
        # Let's assume the goal is satisfied if and only if all packages in self.package_goals
        # are at their respective goal locations. In this case, total_cost == 0 iff goal is reached.

        return total_cost
