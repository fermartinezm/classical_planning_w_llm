from fnmatch import fnmatch
from collections import deque

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and has parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        # Handle unexpected input gracefully, though state facts should be strings
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

class transportHeuristic:
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions (pick-up, drive, drop)
    required to move each package from its current location to its goal location.
    It sums the estimated costs for each package independently.

    # Assumptions:
    - The cost of each action (pick-up, drive, drop) is 1.
    - Vehicle capacity constraints are ignored. Any vehicle can pick up any package.
    - Vehicle availability is ignored. A vehicle is assumed to be available
      whenever a package needs to be picked up or transported.
    - Roads are represented by `(road l1 l2)` facts and define possible
      movements for vehicles. Shortest path driving is assumed.
    - Packages are identified by names starting with 'p', and vehicles by names
      starting with 'v'. This is an inference based on common PDDL naming
      conventions and the provided examples, as explicit type information
      for objects is not readily available in the task structure provided.

    # Heuristic Initialization
    - Extracts the goal location for each package from the task's goal conditions.
    - Builds a directed graph representing the road network from static `(road l1 l2)` facts.
    - Computes the shortest path distance between all pairs of locations using Breadth-First Search (BFS).

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current status and location of every package relevant to the goal:
       - Is the package on the ground at a location `l` (`(at package l)`)?
       - Is the package inside a vehicle `v` (`(in package v)`)? If so, find the location of vehicle `v` (`(at v l)`).
    2. For each package `p` with a goal location `goal_l` (extracted during initialization):
       - Determine the package's current physical location (`current_l`). This is either the location where it's on the ground, or the location of the vehicle it's inside. If the vehicle's location is unknown, the state is considered problematic.
       - If `current_l` is the same as `goal_l`:
         - If `p` is on the ground at `goal_l`: The cost for this package is 0.
         - If `p` is inside a vehicle `v` which is currently at `goal_l`: The package needs to be dropped. The cost for this package is 1 (drop).
       - If `current_l` is different from `goal_l`:
         - Find the shortest distance `d` from `current_l` to `goal_l` using the precomputed distances. If no path exists in the road network, the state is considered effectively unreachable for this package via simple movement, and a large cost (infinity) is returned for the total heuristic.
         - If `p` is on the ground at `current_l`: It needs 1 (pick-up) + `d` (drive) + 1 (drop) actions. Add `2 + d` to the total cost.
         - If `p` is inside a vehicle at `current_l`: It needs `d` (drive) + 1 (drop) actions. Add `1 + d` to the total cost.
    3. The total heuristic value is the sum of the estimated costs for all packages. If any package's goal is unreachable, the total heuristic is infinity.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and building
        the location graph for shortest path calculations.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # Store goal locations for each package.
        self.package_goals = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                # Assuming goal facts are always (at package location)
                if len(args) == 2:
                    package, location = args
                    self.package_goals[package] = location

        # Build directed graph from road facts.
        locations = set()
        self.location_graph = {} # {l1: {l2, l3, ...}} where road l1 l2 and road l1 l3 exist

        for fact in static_facts:
            if match(fact, "road", "*", "*"):
                _, l1, l2 = get_parts(fact)
                locations.add(l1)
                locations.add(l2)
                self.location_graph.setdefault(l1, set()).add(l2)

        # Compute all-pairs shortest paths using BFS.
        self.distances = {} # {(l1, l2): dist}
        all_locations = list(locations) # Get a stable list of locations

        for start_loc in all_locations:
            self.distances[(start_loc, start_loc)] = 0
            queue = deque([(start_loc, 0)])
            visited = {start_loc}

            while queue:
                current_loc, dist = queue.popleft()

                # Check if current_loc has outgoing roads
                if current_loc in self.location_graph:
                    for neighbor in self.location_graph[current_loc]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            self.distances[(start_loc, neighbor)] = dist + 1
                            queue.append((neighbor, dist + 1))

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Track where packages and vehicles are currently located/contained.
        package_status = {} # {package: ('at', location) or ('in', vehicle)}
        vehicle_locations = {} # {vehicle: location}

        # Populate package_status and vehicle_locations from the current state
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            args = parts[1:]

            if predicate == "at" and len(args) == 2:
                obj, location = args
                # Simple inference: if obj is a package we care about (in goals), it's a package.
                # Otherwise, assume it's a vehicle if its name starts with 'v'.
                if obj in self.package_goals:
                     package_status[obj] = ('at', location)
                elif obj.startswith('v'): # Infer vehicles by name prefix
                     vehicle_locations[obj] = location

            elif predicate == "in" and len(args) == 2:
                 package, vehicle = args
                 # Only track packages we care about
                 if package in self.package_goals:
                    package_status[package] = ('in', vehicle)

        total_cost = 0

        # Calculate cost for each package not at its goal
        for package, goal_location in self.package_goals.items():
            # If package is not mentioned in the state, it might be an issue,
            # but assuming valid states always include package location/containment.
            if package not in package_status:
                 # This case indicates an unexpected state structure.
                 # For robustness, return infinity or a large value.
                 # Let's assume valid states for solvable problems.
                 continue

            status, current_loc_or_vehicle = package_status[package]

            # Determine the package's current physical location
            current_location = None
            if status == 'at':
                current_location = current_loc_or_vehicle
            elif status == 'in':
                vehicle = current_loc_or_vehicle
                if vehicle in vehicle_locations:
                    current_location = vehicle_locations[vehicle]
                else:
                    # Package is in a vehicle, but vehicle location is unknown.
                    # This state is problematic. Return infinity.
                    return float('inf')

            # If current_location is still None, something went wrong.
            if current_location is None:
                 return float('inf')

            # Check if the package is already at the goal location
            if current_location == goal_location:
                # If it's on the ground at the goal, cost is 0.
                if status == 'at':
                    continue
                # If it's in a vehicle at the goal, it needs 1 drop action.
                elif status == 'in':
                    total_cost += 1
                    continue # Done with this package

            # Package is not at the goal location. Estimate cost to move it.
            # Find distance from current location to goal location
            # Use .get() with a default of infinity to handle disconnected locations
            distance = self.distances.get((current_location, goal_location), float('inf'))

            if distance == float('inf'):
                # Goal location is unreachable from the package's current location.
                # Return a large value indicating this path is blocked or non-existent.
                return float('inf')

            # Estimate actions:
            # If on ground: pick-up (1) + drive (distance) + drop (1)
            # If in vehicle: drive (distance) + drop (1)
            if status == 'at':
                # Needs pick-up, drive, drop
                total_cost += 1 # pick-up
                total_cost += distance # drive
                total_cost += 1 # drop
            elif status == 'in':
                # Needs drive, drop
                total_cost += distance # drive
                total_cost += 1 # drop

        return total_cost
