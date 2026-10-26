import fnmatch
import collections
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure the fact is a string and looks like a PDDL fact
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        # Handle unexpected input, maybe log a warning or raise an error
        # For this heuristic, we expect valid fact strings.
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
    return all(fnmatch.fnmatch(part, arg) for part, arg in zip(parts, args))

class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions required to move each package
    to its goal location independently. It sums the minimum actions needed for
    each package, ignoring vehicle capacity and the fact that multiple packages
    can be transported together.

    # Assumptions
    - The goal is to move specific packages to specific locations.
    - The road network is static and provides connections between locations.
    - Vehicles can move between connected locations.
    - Packages can be picked up by a vehicle at their current location (if on the ground)
      and dropped at a location (if in a vehicle).
    - Capacity constraints are ignored.
    - Vehicle availability is ignored (any vehicle can be used for any package).
    - The cost of each action (drive, pick-up, drop) is 1.

    # Heuristic Initialization
    - Extracts the goal location for each package from the task's goal conditions.
    - Builds a graph representing the road network from static `(road l1 l2)` facts.
    - Computes all-pairs shortest path distances between locations using BFS on the road network graph.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current location or containment (in a vehicle) for every package and vehicle.
    2. Initialize the total heuristic cost to 0.
    3. For each package that is not yet at its goal location:
       a. Determine the package's current state: either `(at package location)` or `(in package vehicle)`.
       b. If the package is `(at package current_location)`:
          - The minimum actions required for this package are:
            1 (pick-up) + shortest_distance(current_location, goal_location) (drive) + 1 (drop).
          - Add this cost (2 + distance) to the total heuristic.
       c. If the package is `(in package vehicle)`:
          - Find the current location of the vehicle: `(at vehicle vehicle_location)`.
          - The minimum actions required for this package are:
            shortest_distance(vehicle_location, goal_location) (drive) + 1 (drop).
          - Add this cost (1 + distance) to the total heuristic.
       d. If the goal location is unreachable from the package's current location (or its vehicle's location)
          via the road network, the heuristic for this state is considered infinite.
    4. The total heuristic value is the sum of costs calculated for all packages not at their goal.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal locations and precomputing
        shortest path distances in the road network.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # Store goal locations for each package.
        self.package_goals = {}
        for goal in self.goals:
            # Assuming goals are only (at package location)
            parts = get_parts(goal)
            if len(parts) == 3 and parts[0] == "at":
                package, location = parts[1], parts[2]
                self.package_goals[package] = location

        # Build the road network graph and collect all locations.
        graph = collections.defaultdict(set)
        locations = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if len(parts) == 3 and parts[0] == "road":
                l1, l2 = parts[1], parts[2]
                graph[l1].add(l2)
                graph[l2].add(l1) # Roads are typically bidirectional
                locations.add(l1)
                locations.add(l2)

        # Compute all-pairs shortest paths using BFS.
        self.distances = {}
        for start_loc in locations:
            queue = collections.deque([(start_loc, 0)])
            visited = {start_loc}
            self.distances[(start_loc, start_loc)] = 0 # Distance to self is 0

            while queue:
                current_loc, d = queue.popleft()

                # Store distance from start_loc to current_loc
                # (already stored for start_loc,start_loc, update for others)
                if (start_loc, current_loc) not in self.distances:
                     self.distances[(start_loc, current_loc)] = d

                for neighbor in graph.get(current_loc, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, d + 1))

        # If any location is unreachable from another, its distance won't be in self.distances.
        # We handle this during __call__ by using .get(..., float('inf')).


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Map current locations/containment for packages and vehicles.
        package_states = {} # {package: ('at', location) or ('in', vehicle)}
        vehicle_locations = {} # {vehicle: location}

        for fact in state:
            parts = get_parts(fact)
            if len(parts) == 3:
                predicate, obj1, obj2 = parts
                if predicate == 'at':
                    # Distinguish packages and vehicles based on appearance in 'in' facts
                    # This is a simplification; proper parsing of object types is better.
                    # Assuming objects starting with 'p' are packages and 'v' are vehicles
                    # based on example instances.
                    if obj1.startswith('p'): # Assume package
                         package_states[obj1] = ('at', obj2)
                    elif obj1.startswith('v'): # Assume vehicle
                         vehicle_locations[obj1] = obj2
                elif predicate == 'in':
                    # 'in' facts are always (in package vehicle)
                    package, vehicle = obj1, obj2
                    package_states[package] = ('in', vehicle)

        total_cost = 0  # Initialize action cost counter.

        # Iterate through packages that have a goal location
        for package, goal_location in self.package_goals.items():
            # Check if the package is already at its goal location
            if package in package_states and package_states[package] == ('at', goal_location):
                continue # Package is already at goal, no cost for this package

            # Package is not at goal, calculate cost based on its current state
            if package not in package_states:
                 # This package is not mentioned in 'at' or 'in' facts in the state.
                 # This should ideally not happen in valid states if the package exists.
                 # Assume it's unreachable or an invalid state.
                 return float('inf')

            state_type, obj_or_loc = package_states[package]

            if state_type == 'at': # Package is on the ground at obj_or_loc (current_location)
                current_location = obj_or_loc
                # Cost = 1 (pick-up) + drive_cost + 1 (drop)
                drive_cost = self.distances.get((current_location, goal_location), float('inf'))
                if drive_cost == float('inf'):
                    return float('inf') # Goal is unreachable from current location
                total_cost += 1 + drive_cost + 1

            elif state_type == 'in': # Package is in vehicle obj_or_loc (vehicle)
                vehicle = obj_or_loc
                if vehicle not in vehicle_locations:
                    # Vehicle containing the package is not at any location? Invalid state.
                    return float('inf') # Should not happen in valid states

                vehicle_location = vehicle_locations[vehicle]
                # Cost = drive_cost + 1 (drop)
                drive_cost = self.distances.get((vehicle_location, goal_location), float('inf'))
                if drive_cost == float('inf'):
                    return float('inf') # Goal is unreachable from vehicle's location
                total_cost += drive_cost + 1

        return total_cost

