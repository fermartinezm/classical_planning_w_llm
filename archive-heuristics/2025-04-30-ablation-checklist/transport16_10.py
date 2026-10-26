import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty strings or malformed facts defensively
    if not fact or not isinstance(fact, str) or len(fact) < 2:
        return []
    # Remove outer parentheses and split by spaces
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at package1 location1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure the number of parts matches the number of pattern arguments
    if len(parts) != len(args):
        return False
    return all(fnmatch.fnmatch(part, arg) for part, arg in zip(parts, args))

class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions required to move all packages
    to their goal locations. It sums the estimated cost for each package
    individually, ignoring vehicle capacity and potential synergies from
    carrying multiple packages. The estimated cost for a single package
    includes the minimum actions directly involving the package (pick-up, drop)
    plus the shortest path distance the package needs to travel via a vehicle.

    # Assumptions
    - All packages need to reach a specific goal location.
    - Vehicles can travel between connected locations on the road network.
    - Any vehicle can potentially transport any package (capacity is ignored
      in the heuristic calculation).
    - The cost of driving between two locations is the shortest path distance
      in the road network.
    - The cost of pick-up and drop actions is 1 each.

    # Heuristic Initialization
    - Extracts goal locations for each package from the task goals.
    - Builds a graph of locations based on `road` facts.
    - Computes all-pairs shortest paths between locations using BFS.

    # Step-by-Step Thinking for Computing the Heuristic Value
    For a given state:
    1. Identify the current location of every package. A package can be
       either `at` a location on the ground or `in` a vehicle. If it's
       in a vehicle, its effective location for transport purposes is the
       location of the vehicle.
    2. For each package that has a goal location:
       a. If the package is already at its goal location, the cost for this
          package is 0.
       b. If the package is on the ground at a location different from its goal:
          - It needs to be picked up (1 action).
          - It needs to be transported to the goal location. The estimated cost
            for this transport is the shortest path distance between its current
            location and its goal location.
          - It needs to be dropped at the goal location (1 action).
          - Total estimated cost for this package: 1 (pick-up) + distance + 1 (drop).
       c. If the package is inside a vehicle:
          - It needs to be transported from the vehicle's current location to
            the package's goal location. The estimated cost is the shortest
            path distance between the vehicle's current location and the
            package's goal location.
          - It needs to be dropped at the goal location (1 action).
          - Total estimated cost for this package: distance + 1 (drop).
    3. The total heuristic value for the state is the sum of the estimated
       costs for all packages that are not yet at their goal locations.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, building the
        road network graph, and computing shortest paths.

        Args:
            task: The planning task object containing initial state, goals, etc.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # Store goal locations for each package.
        self.goal_locations = {}
        for goal in self.goals:
            # Goal facts are typically (at package location)
            parts = get_parts(goal)
            if parts and parts[0] == "at" and len(parts) == 3:
                package, location = parts[1], parts[2]
                self.goal_locations[package] = location

        # Build the road network graph.
        self.road_graph = {}
        locations = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "road" and len(parts) == 3:
                l1, l2 = parts[1], parts[2]
                self.road_graph.setdefault(l1, set()).add(l2)
                self.road_graph.setdefault(l2, set()).add(l1) # Assuming roads are bidirectional
                locations.add(l1)
                locations.add(l2)

        # Ensure all locations mentioned in goals are in the graph, even if isolated
        for loc in self.goal_locations.values():
             locations.add(loc)
             self.road_graph.setdefault(loc, set()) # Add location to graph even if no roads

        # Compute all-pairs shortest paths using BFS.
        self.distances = {}
        for start_node in locations:
            self.distances[start_node] = self._bfs(start_node, locations)

    def _bfs(self, start_node, all_locations):
        """
        Performs BFS from a start_node to find shortest distances to all
        reachable locations in the road graph.

        Args:
            start_node: The starting location for BFS.
            all_locations: A set of all known locations in the domain.

        Returns:
            A dictionary mapping reachable locations to their distance from start_node.
            Unreachable locations are implicitly not included or could be considered infinity.
        """
        distances = {loc: float('inf') for loc in all_locations}
        distances[start_node] = 0
        queue = deque([start_node])

        while queue:
            current_node = queue.popleft()
            current_dist = distances[current_node]

            # Get neighbors from the graph, handle nodes with no roads
            neighbors = self.road_graph.get(current_node, set())

            for neighbor in neighbors:
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = current_dist + 1
                    queue.append(neighbor)

        return distances

    def get_distance(self, l1, l2):
        """
        Retrieves the precomputed shortest distance between two locations.

        Args:
            l1: The starting location.
            l2: The ending location.

        Returns:
            The shortest distance (number of drive actions) or float('inf')
            if no path exists or if locations are unknown.
        """
        # Check if locations are known and distance is computed
        if l1 in self.distances and l2 in self.distances[l1]:
             return self.distances[l1][l2]
        # Should not happen in valid problems if all locations are in static/goals
        return float('inf')


    def __call__(self, node):
        """
        Compute the domain-dependent heuristic value for the given state.

        Args:
            node: The search node containing the current state.

        Returns:
            An estimate of the number of actions to reach a goal state.
        """
        state = node.state  # Current world state (frozenset of facts).

        # Track where packages and vehicles are currently located.
        # package_locations: maps package -> location (if on ground) or vehicle (if in vehicle)
        # vehicle_locations: maps vehicle -> location
        package_locations = {}
        vehicle_locations = {}

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == "at" and len(parts) == 3:
                obj, location = parts[1], parts[2]
                # Determine if the object is a package or vehicle based on goals/domain knowledge
                # A more robust way would be to parse types from the domain, but checking if it's a package with a goal is sufficient here.
                if obj in self.goal_locations: # It's a package
                     package_locations[obj] = location
                # Assume anything else with 'at' is a vehicle for heuristic purposes
                # A more robust check would use domain types, but vehicle names often start with 'v'
                elif obj.startswith('v'): # Heuristic assumption: vehicles start with 'v'
                     vehicle_locations[obj] = location

            elif predicate == "in" and len(parts) == 3:
                package, vehicle = parts[1], parts[2]
                if package in self.goal_locations: # It's a package
                    package_locations[package] = vehicle # Store the vehicle name

        total_cost = 0  # Initialize heuristic cost.

        # Iterate through packages that have a goal location
        for package, goal_location in self.goal_locations.items():
            current_status = package_locations.get(package)

            # If package is not found in state facts, something is wrong or it's irrelevant
            if current_status is None:
                 # This package doesn't seem to exist or is not locatable/in a vehicle.
                 # Assuming valid states where packages are always locatable.
                 continue

            # Case 1: Package is at its goal location
            if current_status == goal_location:
                continue # Cost is 0 for this package

            # Case 2: Package is on the ground at a different location
            if current_status in self.road_graph: # Check if current_status is a location
                l_curr = current_status
                # Cost: pick-up (1) + drive (distance) + drop (1)
                distance = self.get_distance(l_curr, goal_location)
                if distance == float('inf'):
                    # If goal is unreachable, return infinity
                    return float('inf')
                total_cost += 1 + distance + 1

            # Case 3: Package is inside a vehicle
            elif current_status in vehicle_locations: # Check if current_status is a vehicle name
                vehicle = current_status
                l_v_curr = vehicle_locations.get(vehicle)

                if l_v_curr is None:
                    # Vehicle location unknown, cannot estimate transport cost
                    # This shouldn't happen in a valid state if the package is in the vehicle
                    # but handle defensively. Treat as unreachable goal for this package.
                     return float('inf')

                # Cost: drive (distance) + drop (1)
                distance = self.get_distance(l_v_curr, goal_location)
                if distance == float('inf'):
                    # If goal is unreachable, return infinity
                    return float('inf')
                total_cost += distance + 1
            else:
                 # current_status is neither a known location nor a known vehicle carrying a package
                 # This indicates an unexpected state structure. Treat as unreachable.
                 return float('inf')


        return total_cost

