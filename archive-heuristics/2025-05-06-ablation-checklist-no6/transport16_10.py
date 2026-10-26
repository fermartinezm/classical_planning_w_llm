# Add the required import
from fnmatch import fnmatch
# Assume Heuristic base class is available as specified in the problem description
# from heuristics.heuristic_base import Heuristic

# The problem description implies a Heuristic base class exists and this class
# should inherit from it. The base class definition is not provided, so we
# assume its interface (__init__ takes task, __call__ takes node).

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential leading/trailing whitespace or multiple spaces
    return fact.strip()[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# Assuming Heuristic base class is defined elsewhere and imported.
# class Heuristic: ...

class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    Estimates the cost based on the distance each misplaced package needs
    to travel, plus a penalty for packages stranded without a vehicle.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions,
        building the road network graph, computing distances,
        and mapping size predicates to capacity units.
        """
        self.task = task
        self.goals = task.goals
        static_facts = task.static

        # 1. Extract goal locations for each package
        self.goal_locations = {}
        for goal in self.goals:
            # Goal is typically (at package location)
            if match(goal, "at", "*", "*"):
                _, package, location = get_parts(goal)
                self.goal_locations[package] = location

        # 2. Build road network graph and compute distances
        self.graph = {}
        locations = set()

        # Collect all locations from road facts
        for fact in static_facts:
            if match(fact, "road", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                locations.add(loc1)
                locations.add(loc2)
                self.graph.setdefault(loc1, []).append(loc2)
                self.graph.setdefault(loc2, []).append(loc1) # Roads are typically bidirectional

        # Ensure all locations mentioned in goals are in the graph, even if isolated
        for loc in self.goal_locations.values():
             locations.add(loc)
             self.graph.setdefault(loc, []) # Add location even if it has no roads

        self.locations = list(locations) # Store as list

        # Compute all-pairs shortest paths using BFS
        self.distances = {}
        for start_loc in self.locations:
            self.distances[start_loc] = {}
            q = [(start_loc, 0)]
            visited = {start_loc}
            self.distances[start_loc][start_loc] = 0

            head = 0
            while head < len(q):
                current_loc, dist = q[head]
                head += 1

                # Handle locations with no roads (isolated)
                if current_loc not in self.graph:
                    continue

                for neighbor in self.graph[current_loc]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        self.distances[start_loc][neighbor] = dist + 1
                        q.append((neighbor, dist + 1))

        # 3. Map size predicates to capacity units
        self.capacity_units = {}
        # successor_map: s1 -> s2 (size_after_pickup -> size_before)
        # (capacity-predecessor s1 s2) means s1 has 1 less unit capacity than s2
        successor_map = {}
        sizes = set()

        for fact in static_facts:
            if match(fact, "capacity-predecessor", "*", "*"):
                _, s1, s2 = get_parts(fact)
                successor_map[s1] = s2 # s2 is the successor of s1 (s2 has 1 more unit capacity)
                sizes.add(s1)
                sizes.add(s2)
            # Also collect sizes from initial capacity facts in static (if any)
            if match(fact, "capacity", "*", "*"):
                 _, vehicle, size = get_parts(fact)
                 sizes.add(size)

        # Find the size with minimum capacity (0 units) - it is never the first argument (s1)
        predecessor_sources = {get_parts(fact)[1] for fact in static_facts if match(fact, "capacity-predecessor", "*", "*")}
        min_capacity_size = None
        for size in sizes:
            if size not in predecessor_sources:
                 min_capacity_size = size
                 break

        if min_capacity_size is not None:
            self.capacity_units[min_capacity_size] = 0
            current_level_sizes = [min_capacity_size]

            while current_level_sizes:
                next_level_sizes = []
                for current_size in current_level_sizes:
                    current_units = self.capacity_units[current_size]
                    # Find the size that has current_size as its predecessor (s1)
                    # i.e., find s2 such that (capacity-predecessor current_size s2)
                    next_size = successor_map.get(current_size)
                    if next_size and next_size not in self.capacity_units:
                         self.capacity_units[next_size] = current_units + 1
                         next_level_sizes.append(next_size)
                current_level_sizes = next_level_sizes

        # If min_capacity_size is None or capacity_units is still empty,
        # it implies an issue with size definitions in PDDL.
        # Defaulting capacity lookup to 0 might be the safest fallback.


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Track where packages and vehicles are currently located.
        current_locations = {} # obj -> location_or_vehicle
        vehicle_current_capacity = {} # vehicle -> units
        packages_on_ground_needing_pickup_locations = set() # Locations with packages on ground needing pickup.
        vehicles_with_capacity_at_loc = set() # Locations where a vehicle with capacity > 0 is present.

        # Parse state to populate current locations, capacities, and initial sets for penalty
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at":
                _, obj, loc = parts
                current_locations[obj] = loc
            elif parts[0] == "in":
                _, package, vehicle = parts
                current_locations[package] = vehicle # Package is inside a vehicle
            elif parts[0] == "capacity":
                 _, vehicle, size = parts
                 # Use .get(size, 0) in case a size isn't in our map (shouldn't happen in valid PDDL)
                 vehicle_current_capacity[vehicle] = self.capacity_units.get(size, 0)

        total_cost = 0  # Initialize action cost counter.

        # Calculate base cost per package
        for package, goal_location in self.goal_locations.items():
            # Check if package is already at goal
            if f"(at {package} {goal_location})" in state:
                continue # Package is at goal, cost is 0 for this package

            # Get current location/container of the package
            current_loc_or_vehicle = current_locations.get(package)

            if current_loc_or_vehicle is None:
                 # This state is likely invalid if a goal package has no location/container
                 # Treat as infinite cost if a required package is lost.
                 return float('inf')

            # Check if the package is on the ground or in a vehicle
            if current_loc_or_vehicle in self.locations: # It's a location name
                current_loc = current_loc_or_vehicle
                # Package is on the ground, needs pick, drive, drop
                # Cost = 1 (pick) + distance(current_loc, goal_location) + 1 (drop)
                dist = self.distances.get(current_loc, {}).get(goal_location, float('inf'))
                if dist == float('inf'):
                    return float('inf') # Goal is unreachable from here
                total_cost += 2 + dist
                packages_on_ground_needing_pickup_locations.add(current_loc)

            else: # It's a vehicle name
                vehicle = current_loc_or_vehicle
                # Package is in a vehicle, needs drive, drop
                # Find the vehicle's location
                vehicle_loc = current_locations.get(vehicle)
                if vehicle_loc is None or vehicle_loc not in self.locations:
                     # Vehicle location unknown or invalid
                     return float('inf') # Should not happen in valid states

                # Cost = distance(vehicle_loc, goal_location) + 1 (drop)
                dist = self.distances.get(vehicle_loc, {}).get(goal_location, float('inf'))
                if dist == float('inf'):
                    return float('inf') # Goal is unreachable from here
                total_cost += 1 + dist

        # Identify locations where vehicles with capacity > 0 are present
        for vehicle, capacity in vehicle_current_capacity.items():
            if capacity > 0:
                vehicle_loc = current_locations.get(vehicle)
                if vehicle_loc in self.locations: # Ensure vehicle location is valid
                    vehicles_with_capacity_at_loc.add(vehicle_loc)

        # Add penalty for locations needing vehicle pickup
        for loc in packages_on_ground_needing_pickup_locations:
            if loc not in vehicles_with_capacity_at_loc:
                # There's a package needing pickup here, but no vehicle with capacity is present.
                # Add a penalty, e.g., the cost of one drive action (min distance 1).
                # This encourages moving a vehicle to this location.
                total_cost += 1 # Penalty

        # The heuristic should be 0 only for goal states.
        # If total_cost is 0, it means all packages are at their goal locations
        # and there are no stranded packages needing pickup without a vehicle.
        # This seems correct for a goal state.

        return total_cost
