from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque, defaultdict


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and has parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
         # Return empty list for malformed facts, or handle as needed
         return []
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Check if the number of parts matches the number of arguments in the pattern
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions required to move each package
    that is not at its goal location to its goal location. It sums the estimated
    costs for each package independently, ignoring vehicle capacity and potential
    for shared trips, but accounting for loading/unloading and driving. The
    driving cost is estimated as the shortest path distance in the road network.

    # Assumptions
    - Packages need to be transported from their current location (on the ground
      or inside a vehicle) to a specific goal location.
    - The road network defined by `(road l1 l2)` facts is static and represents
      possible movements for vehicles.
    - Each pick-up, drop, and drive action costs 1.
    - Vehicle capacity is ignored for heuristic calculation (simplification).
    - Packages are the only objects that need to reach specific goal locations.
    - The road network is connected for all relevant locations, or unreachable
      goals result in an infinite heuristic value.

    # Heuristic Initialization
    - Build the road network graph from `(road l1 l2)` facts in `task.static`.
      The graph is represented as an adjacency list. Roads are assumed bidirectional.
    - Extract the goal location for each package from `task.goals`.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current location or containing vehicle for every package and vehicle.
       Store this information in a dictionary (e.g., `current_states`). This is done
       by iterating through the state facts and looking for `(at obj loc)` and
       `(in pkg veh)` predicates.
    2. Initialize the total heuristic cost to 0.
    3. For each package `p` that has a goal location `goal_loc` (extracted during initialization):
       a. Check if the package is already at its `goal_loc`. This means checking
          if the fact `(at p goal_loc)` is present in the current state. If yes,
          the cost for this package is 0; continue to the next package.
       b. If the package is not at its goal, determine its effective current location
          for transport and the initial cost.
          - Get the package's current state from the `current_states` dictionary.
          - If the package's current state is a location name (i.e., it's on the ground):
            The effective location is this location. Add 1 to the package's cost
            for the required `pick-up` action.
          - If the package's current state is a vehicle name (i.e., it's inside a vehicle):
            Find the location of that vehicle from the `current_states` dictionary.
            The effective location is the vehicle's location. No `pick-up` cost is added
            initially as it's already loaded.
       c. Calculate the shortest path distance (number of `drive` actions) from the
          effective current location to the `goal_loc` using the pre-computed
          road graph and BFS. Let this distance be `drive_cost`.
       d. If `drive_cost` is infinite (goal unreachable), the state is likely
          unsolvable. Return `float('inf')` immediately.
       e. Add `drive_cost` to the package's cost.
       f. Add 1 to the package's cost for the required `drop` action at the goal location.
       g. Add the total cost calculated for this package to the overall `total_cost`.
    4. Return the final `total_cost`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by building the road graph and extracting
        package goal locations.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # Build the road graph (adjacency list).
        self.road_graph = defaultdict(list)
        # Keep track of all locations mentioned in the graph for easy checking
        self.locations = set()
        for fact in static_facts:
            if match(fact, "road", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                self.road_graph[loc1].append(loc2)
                # Assuming roads are bidirectional based on example static facts
                self.road_graph[loc2].append(loc1)
                self.locations.add(loc1)
                self.locations.add(loc2)


        # Store goal locations for each package.
        self.goal_locations = {}
        for goal in self.goals:
            # Goal facts are typically '(at package location)'
            if match(goal, "at", "*", "*"):
                _, package, location = get_parts(goal)
                self.goal_locations[package] = location
            # Add other potential goal types if necessary, but 'at' is most common for packages

    def bfs(self, start_loc, end_loc):
        """
        Performs Breadth-First Search to find the shortest path distance
        (number of drive actions) between two locations in the road graph.

        Returns the distance or float('inf') if the end_loc is unreachable
        from the start_loc.
        """
        if start_loc == end_loc:
            return 0

        # Ensure start_loc is a valid node in the graph
        if start_loc not in self.locations:
             return float('inf')

        queue = deque([(start_loc, 0)]) # (location, distance)
        visited = {start_loc}

        while queue:
            current_loc, dist = queue.popleft()

            # current_loc is guaranteed to be in self.locations if we reached here

            # Check if current_loc is in the graph keys before accessing neighbors
            # It might be a location in self.locations but have no outgoing roads
            if current_loc not in self.road_graph:
                 continue # Cannot move from here

            for neighbor in self.road_graph[current_loc]:
                if neighbor == end_loc:
                    return dist + 1 # Found the shortest path

                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))

        # If the queue is empty and the end_loc was not reached
        return float('inf')

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions
        to get all packages to their goal locations.
        """
        state = node.state  # Current world state.

        # Map locatable objects (packages, vehicles) to their current state
        # State can be a location (if on ground) or a vehicle name (if inside)
        current_states = {}
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == "at":
                # (at ?x - locatable ?v - location)
                if len(parts) == 3:
                    obj, loc = parts[1], parts[2]
                    current_states[obj] = loc
            elif predicate == "in":
                 # (in ?x - package ?v - vehicle)
                 if len(parts) == 3:
                    pkg, veh = parts[1], parts[2]
                    current_states[pkg] = veh # Store the vehicle name

        total_cost = 0  # Initialize action cost counter.

        # Iterate through packages that have a goal location
        for package, goal_location in self.goal_locations.items():
            # Check if the package is already at its goal
            if (f"(at {package} {goal_location})") in state:
                continue # Package is already at goal, cost is 0 for this package

            # Package is not at goal, calculate cost to move it

            package_current_state = current_states.get(package)

            # Should not happen in valid states, but defensive check
            if package_current_state is None:
                 # Package state unknown, treat as unreachable
                 return float('inf')

            # Determine if the package is on the ground or in a vehicle
            # A package is on the ground if its current_state is a location name.
            # We check if the state is a known location from the road graph initialization.
            is_on_ground = package_current_state in self.locations

            if is_on_ground:
                 effective_location = package_current_state
                 # Need to pick up the package
                 package_cost = 1 # pick-up action
            else:
                 # package_current_state is a vehicle name
                 vehicle_name = package_current_state
                 vehicle_location = current_states.get(vehicle_name)

                 # Handle case where vehicle location is not found (invalid state)
                 if vehicle_location is None:
                     return float('inf') # Vehicle carrying package has no location?

                 effective_location = vehicle_location
                 # Package is already in vehicle, no pick-up needed
                 package_cost = 0

            # Calculate drive cost from effective location to goal location
            # effective_location must be a location string.
            if effective_location not in self.locations:
                 # Effective location is not a known location, cannot drive from here
                 return float('inf') # Unreachable

            drive_cost = self.bfs(effective_location, goal_location)

            # If goal is unreachable from the effective location, return infinity
            if drive_cost == float('inf'):
                return float('inf')

            # Add drive cost
            package_cost += drive_cost

            # Add drop cost
            package_cost += 1 # drop action

            # Add this package's cost to the total
            total_cost += package_cost

        return total_cost
