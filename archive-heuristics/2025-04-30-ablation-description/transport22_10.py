from fnmatch import fnmatch
from collections import deque
# Assuming heuristic_base is available in the environment
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
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
    This heuristic estimates the cost to reach the goal by summing the estimated costs
    for each package that is not yet at its goal location. The cost for each package
    is estimated independently, ignoring vehicle capacity and assuming a vehicle is
    always available to pick up/drop off the package and transport it along the
    shortest path.

    # Assumptions
    - The road network is undirected (if road l1 l2 exists, road l2 l1 exists).
    - Shortest path distances between locations can be precomputed.
    - Vehicle capacity constraints are ignored.
    - Vehicle availability and coordination are ignored.
    - Packages are either on the ground at a location or inside a vehicle.

    # Heuristic Initialization
    - Extracts the goal location for each package from the task goals.
    - Builds the road network graph from static 'road' facts.
    - Computes all-pairs shortest path distances between locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    For each package that is not at its goal location:
    1. Determine the package's current status by searching the state: Is it on the ground at a location `L_current`, or inside a vehicle `V`?
    2. If the package is on the ground at location `L_current`:
       - It needs to be picked up (cost 1).
       - It needs to be transported from `L_current` to its goal location `L_goal`. The estimated cost is the shortest path distance between `L_current` and `L_goal`.
       - It needs to be dropped off at `L_goal` (cost 1).
       - Total estimated cost for this package: 1 (pick) + distance(L_current, L_goal) (drive) + 1 (drop).
    3. If the package is inside a vehicle `V`:
       - Find the vehicle's current location `L_vehicle` by searching the state.
       - It needs to be transported from `L_vehicle` to its goal location `L_goal`. The estimated cost is the shortest path distance between `L_vehicle` and `L_goal`.
       - It needs to be dropped off at `L_goal` (cost 1).
       - Total estimated cost for this package: distance(L_vehicle, L_goal) (drive) + 1 (drop).
    4. If the package is already at its goal location on the ground, the cost for this package is 0.
    5. If at any point a required location (package current, vehicle current, or package goal) is not found in the precomputed distances (meaning it wasn't in the road graph from static facts), or if a vehicle location cannot be found, the state is likely invalid or unreachable, and the heuristic returns infinity.
    6. The total heuristic value is the sum of the estimated costs for all packages.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and building
        the road network graph to compute distances.
        """
        # The set of facts that must hold in goal states.
        self.goals = task.goals
        # Static facts are facts that are true in every state.
        static_facts = task.static

        # 1. Extract goal locations for packages
        self.goal_locations = {}
        for goal in self.goals:
            # Goal facts are typically (at ?p ?l)
            if match(goal, "at", "*", "*"):
                predicate, package, location = get_parts(goal)
                self.goal_locations[package] = location

        # 2. Build the road network graph
        self.road_graph = {}
        locations = set()
        for fact in static_facts:
            if match(fact, "road", "*", "*"):
                predicate, loc1, loc2 = get_parts(fact)
                locations.add(loc1)
                locations.add(loc2)
                self.road_graph.setdefault(loc1, set()).add(loc2)
                self.road_graph.setdefault(loc2, set()).add(loc1) # Assuming roads are bidirectional

        # Store all unique locations found in the road graph
        self.locations = list(locations)

        # 3. Compute all-pairs shortest path distances using BFS
        self.distances = {}
        for start_loc in self.locations:
            self.distances[start_loc] = self._bfs(start_loc)

    def _bfs(self, start_location):
        """
        Performs Breadth-First Search from a start location to find distances
        to all other reachable locations in the road graph.
        Returns a dictionary mapping reachable locations to their distance from start_location.
        """
        distances = {loc: float('inf') for loc in self.locations}
        distances[start_location] = 0
        queue = deque([start_location])

        while queue:
            current_loc = queue.popleft()

            # Check if current_loc exists in the graph (it should if it's in self.locations)
            if current_loc in self.road_graph:
                for neighbor in self.road_graph[current_loc]:
                    if distances[neighbor] == float('inf'):
                        distances[neighbor] = distances[current_loc] + 1
                        queue.append(neighbor)
        return distances

    def __call__(self, node):
        """
        Estimate the minimum cost to transport all remaining packages to their goals.
        """
        state = node.state  # Current world state (frozenset of facts)

        total_cost = 0

        # Iterate through each package and its goal
        for package, goal_location in self.goal_locations.items():
            # Find the package's current status: on the ground or in a vehicle
            package_on_ground_location = None
            package_in_vehicle = None

            # Search the state for the package's location
            for fact in state:
                 parts = get_parts(fact)
                 if len(parts) >= 3: # Ensure enough parts for predicate and args
                     if parts[0] == "at" and parts[1] == package:
                          package_on_ground_location = parts[2]
                          # A package should not be both 'at' a location and 'in' a vehicle simultaneously
                          break # Found package location on ground
                     elif parts[0] == "in" and parts[1] == package:
                          package_in_vehicle = parts[2]
                          # A package should not be both 'at' a location and 'in' a vehicle simultaneously
                          break # Found package in vehicle

            # Check if the package is already at its goal location on the ground
            if package_on_ground_location == goal_location:
                 continue # Package is at goal, cost is 0 for this package

            # If it's not at the goal, calculate its contribution
            package_cost = 0

            if package_on_ground_location is not None: # Package is on the ground (but not at goal)
                current_package_location = package_on_ground_location
                # Needs pick-up, drive, drop
                package_cost += 1 # Pick-up cost

                # Find distance from current location to goal location
                # Check if locations exist in our precomputed distances
                if current_package_location not in self.distances or goal_location not in self.distances.get(current_package_location, {}):
                     # One of the locations wasn't in the road graph, or goal is unreachable
                     return float('inf') # State is likely unsolvable

                dist = self.distances[current_package_location][goal_location]

                if dist == float('inf'):
                     return float('inf') # Goal is unreachable

                package_cost += dist # Drive cost
                package_cost += 1 # Drop cost

            elif package_in_vehicle is not None: # Package is inside a vehicle
                vehicle = package_in_vehicle
                # Needs drive (by vehicle), drop
                # Find the vehicle's current location
                vehicle_current_location = None
                for fact in state:
                     if match(fact, "at", vehicle, "*"):
                          vehicle_current_location = get_parts(fact)[2]
                          break # Found vehicle location

                if vehicle_current_location is None:
                     # Vehicle location unknown - invalid state?
                     return float('inf') # Vehicle carrying package must be at a location

                # Find distance from vehicle's current location to package's goal location
                # Check if locations exist in our precomputed distances
                if vehicle_current_location not in self.distances or goal_location not in self.distances.get(vehicle_current_location, {}):
                     # One of the locations wasn't in the road graph, or goal is unreachable
                     return float('inf') # State is likely unsolvable

                dist = self.distances[vehicle_current_location][goal_location]

                if dist == float('inf'):
                     return float('inf') # Goal is unreachable

                package_cost += dist # Drive cost
                package_cost += 1 # Drop cost
            else:
                 # Package is not 'at' a location and not 'in' a vehicle.
                 # This implies an invalid state representation according to the domain.
                 # Or perhaps the package is not mentioned at all, which shouldn't happen for goal packages.
                 # Return infinity as a safe indicator of a problematic state.
                 return float('inf')

            total_cost += package_cost

        return total_cost
