from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at obj loc)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Use zip to handle patterns shorter or longer than fact parts gracefully
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    Estimates the cost to move each misplaced package to its goal location
    by summing the minimum actions required for each package independently.
    The cost for a package includes:
    - 1 pick-up action if it's on the ground.
    - Minimum drive actions based on shortest path distance on the road network.
    - 1 drop action.

    Capacity constraints and vehicle coordination are ignored, treating each
    package's transport as an independent subproblem.

    # Heuristic Initialization
    - Extracts goal locations for packages.
    - Identifies all packages, vehicles, and locations involved in the problem
      from initial state, static facts, and goals.
    - Precomputes all-pairs shortest path distances between locations
      on the road network using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current location of every package and vehicle. A package
       can be on the ground at a location or inside a vehicle.
    2. Initialize total heuristic cost to 0.
    3. For each package that has a goal location:
       a. If the package is already on the ground at its goal location, it
          contributes 0 to the heuristic.
       b. If the package is on the ground at a different location:
          - It needs a pick-up (cost 1).
          - It needs to be transported by a vehicle to the goal location. The
            minimum drive cost is the shortest path distance between its current
            location and the goal location.
          - It needs a drop (cost 1).
          - Total cost for this package: 1 + shortest_distance + 1.
       c. If the package is inside a vehicle:
          - It needs to be transported by the vehicle from the vehicle's current
            location to the package's goal location. The minimum drive cost is
            the shortest path distance between the vehicle's current location
            and the package's goal location.
          - It needs a drop (cost 1).
          - Total cost for this package: shortest_distance + 1.
       d. If any required location or path is not found (indicating an invalid
          state or unsolvable subproblem), return infinity.
    4. The total heuristic value is the sum of costs for all misplaced packages.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting problem details and precomputing
        shortest path distances.
        """
        self.goals = task.goals
        static_facts = task.static
        initial_state = task.initial_state

        # Extract goal locations for packages
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                package, location = args
                self.goal_locations[package] = location
            # Assuming goals only specify package locations like (at package location)

        # Identify all packages, vehicles, and locations in the problem
        # This helps in correctly interpreting 'at' and 'in' facts in states
        self.all_packages = set(self.goal_locations.keys()) # Packages explicitly in goal
        self.all_vehicles = set()
        self.all_locations = set()

        # Collect objects and locations from static facts
        for fact in static_facts:
             parts = get_parts(fact)
             if parts[0] == "road":
                 self.all_locations.add(parts[1])
                 self.all_locations.add(parts[2])
             # capacity-predecessor doesn't mention objects/locations

        # Collect objects and locations from initial state
        for fact in initial_state:
             parts = get_parts(fact)
             if parts[0] == "capacity":
                 self.all_vehicles.add(parts[1])
             elif parts[0] == "at":
                 obj, loc = parts[1:]
                 self.all_locations.add(loc)
                 # Objects in 'at' facts in initial state are locatable.
                 # We'll refine package/vehicle identification below.
             elif parts[0] == "in":
                 p, v = parts[1:]
                 self.all_packages.add(p)
                 self.all_vehicles.add(v)
             # Ignore other predicates like capacity-predecessor in initial state

        # Refine object typing: Assume any object with an 'at' fact in the initial state
        # that is not identified as a vehicle (by capacity or being carried) is a package.
        locatable_objects_in_init = set()
        for fact in initial_state:
            parts = get_parts(fact)
            if parts[0] == "at":
                locatable_objects_in_init.add(parts[1])

        for obj in locatable_objects_in_init:
             if obj not in self.all_vehicles:
                 self.all_packages.add(obj)

        # Ensure all locations mentioned in goals are included, even if not in init/static roads
        for loc in self.goal_locations.values():
             self.all_locations.add(loc)

        # Build road network graph using all identified locations
        adj = {}
        for fact in static_facts:
            if match(fact, "road", "?l1", "?l2"):
                l1, l2 = get_parts(fact)[1:]
                # Ensure locations are in our set (they should be if parsed correctly)
                if l1 in self.all_locations and l2 in self.all_locations:
                    adj.setdefault(l1, []).append(l2)
                    adj.setdefault(l2, []).append(l1) # Assuming roads are bidirectional

        # Compute all-pairs shortest paths using BFS
        self.dist = {}
        for start_loc in self.all_locations:
            self.dist[start_loc] = {}
            queue = deque([(start_loc, 0)])
            visited = {start_loc}
            self.dist[start_loc][start_loc] = 0

            while queue:
                current_loc, d = queue.popleft()

                # Handle locations with no outgoing roads gracefully
                for neighbor in adj.get(current_loc, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        self.dist[start_loc][neighbor] = d + 1
                        queue.append((neighbor, d + 1))

        # Note: If the graph is disconnected, BFS will only find distances within the component.
        # Unreachable locations will not be in self.dist[start_loc]. This is handled in __call__.


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Find current locations of packages and vehicles in the current state
        package_locations = {} # package -> location_or_vehicle (vehicle name if inside)
        vehicle_locations = {} # vehicle -> location

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at":
                obj, loc = parts[1:]
                if obj in self.all_packages:
                    package_locations[obj] = loc
                elif obj in self.all_vehicles:
                    vehicle_locations[obj] = loc
                # Ignore 'at' facts for objects not identified as packages/vehicles
            elif parts[0] == "in":
                p, v = parts[1:]
                if p in self.all_packages and v in self.all_vehicles:
                    package_locations[p] = v # Package p is inside vehicle v
                # Ignore 'in' facts for objects not identified as packages/vehicles
            # Ignore other predicates like capacity in the state for location tracking

        total_cost = 0

        # Iterate through packages that have a goal location
        for package, goal_location in self.goal_locations.items():
            # If a package from the goal is not found in the current state's location facts,
            # it implies an invalid state representation or the package is no longer relevant.
            # Assuming valid states where packages in goals are always locatable.
            if package not in package_locations:
                 # This case should ideally not happen in valid planning states.
                 # Returning infinity indicates a potentially problematic state or problem definition.
                 return float('inf')

            current_loc_or_vehicle = package_locations[package]

            # Check if the package is already at its goal location
            # A package is at its goal only if it's on the ground at the goal location.
            if current_loc_or_vehicle == goal_location:
                continue # Package is already at the goal

            # Package is not at the goal. Calculate cost for this package.
            cost_for_package = 0

            if current_loc_or_vehicle in self.all_vehicles: # It's a vehicle name, package is inside a vehicle
                vehicle = current_loc_or_vehicle
                current_loc = vehicle_locations.get(vehicle) # Get vehicle's location

                if current_loc is None:
                    # Vehicle location not found? Invalid state.
                    return float('inf') # Indicate unsolvable or problematic state

                # Package is in vehicle, needs to be driven and dropped
                # Cost = drives + drop
                drive_cost = self.dist.get(current_loc, {}).get(goal_location)
                if drive_cost is None:
                     # Goal location unreachable from current vehicle location
                     return float('inf') # Indicate unsolvable or problematic state

                cost_for_package = drive_cost + 1 # 1 for drop action

            elif current_loc_or_vehicle in self.all_locations: # It's a location name, package is on the ground
                current_loc = current_loc_or_vehicle

                # Package is on the ground, needs pick-up, drives, and drop
                # Cost = pick-up + drives + drop
                drive_cost = self.dist.get(current_loc, {}).get(goal_location)
                if drive_cost is None:
                     # Goal location unreachable from current package location
                     return float('inf') # Indicate unsolvable or problematic state

                cost_for_package = 1 + drive_cost + 1 # 1 for pick-up, 1 for drop
            else:
                 # current_loc_or_vehicle is neither a known vehicle nor a known location.
                 # Invalid state representation.
                 return float('inf')

            total_cost += cost_for_package

        # The heuristic is 0 iff all packages in goal_locations are at their goal_location.
        # This aligns with the typical goal structure in the transport domain examples.
        return total_cost
