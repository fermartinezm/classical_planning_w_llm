# Assuming Heuristic base class is available in heuristics.heuristic_base
from heuristics.heuristic_base import Heuristic

from fnmatch import fnmatch
from collections import deque

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty strings or malformed facts gracefully
    if not isinstance(fact, str) or len(fact) < 2 or fact[0] != '(' or fact[-1] != ')':
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
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# --- transportHeuristic implementation ---

class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    Estimates the number of actions needed to move each package to its goal location.
    It sums the estimated costs for each package independently, ignoring vehicle
    capacity and availability constraints.

    The estimated cost for a package is:
    - If on the ground at current_l: 1 (pickup) + distance(current_l, goal_l) + 1 (drop)
    - If in a vehicle at v_current_l: distance(v_current_l, goal_l) + 1 (drop)

    Distances between locations are precomputed using BFS on the road network.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal locations, identifying objects
        by type, building the road graph, and precomputing shortest path distances
        between locations.
        """
        self.task = task
        self.goals = task.goals
        static_facts = task.static

        # 1. Identify objects by type and extract goal locations
        self.goal_locations = {} # {package: goal_location}
        all_objects = set()
        is_package = set()
        is_vehicle = set()
        is_location = set()
        is_size = set()

        def process_fact_for_types(fact):
            parts = get_parts(fact)
            if not parts: return
            predicate = parts[0]
            args = parts[1:]
            all_objects.update(args)

            if predicate == "at" and len(args) == 2:
                # ?obj is locatable, ?loc is location
                is_location.add(args[1])
            elif predicate == "in" and len(args) == 2:
                # ?pkg is package, ?veh is vehicle
                is_package.add(args[0])
                is_vehicle.add(args[1])
            elif predicate == "capacity" and len(args) == 2:
                # ?veh is vehicle, ?s is size
                is_vehicle.add(args[0])
                is_size.add(args[1])
            elif predicate == "capacity-predecessor" and len(args) == 2:
                # ?s1 is size, ?s2 is size
                is_size.add(args[0])
                is_size.add(args[1])
            elif predicate == "road" and len(args) == 2:
                # ?l1 is location, ?l2 is location
                is_location.add(args[0])
                is_location.add(args[1])

        # Process all facts to identify object types
        for fact in task.initial_state: process_fact_for_types(fact)
        for fact in task.static: process_fact_for_types(fact)
        for goal in task.goals:
            process_fact_for_types(goal)
            # Also extract goal locations specifically
            if match(goal, "at", "?p", "?l"):
                package, location = get_parts(goal)[1:]
                self.goal_locations[package] = location
                is_package.add(package) # Ensure goal objects are marked as packages

        # Store sets for quick lookup in __call__
        self._packages = frozenset(is_package)
        self._vehicles = frozenset(is_vehicle)
        self._locations = frozenset(is_location)
        self._sizes = frozenset(is_size) # Not used in heuristic, but collected

        # 2. Build road graph using identified locations
        self.road_graph = {loc: set() for loc in self._locations}
        for fact in static_facts:
            if match(fact, "road", "?l1", "?l2"):
                l1, l2 = get_parts(fact)[1:]
                if l1 in self.road_graph: # Ensure locations are valid
                    self.road_graph[l1].add(l2)

        # 3. Compute all-pairs shortest path distances using BFS
        self.distances = {} # {start_loc: {end_loc: distance}}
        for start_loc in self._locations:
            self.distances[start_loc] = {}
            queue = deque([(start_loc, 0)])
            visited = {start_loc}
            self.distances[start_loc][start_loc] = 0

            while queue:
                current_loc, dist = queue.popleft()

                if current_loc in self.road_graph: # Ensure location exists in graph
                    for neighbor in self.road_graph.get(current_loc, set()): # Use .get for safety
                        if neighbor not in visited:
                            visited.add(neighbor)
                            self.distances[start_loc][neighbor] = dist + 1
                            queue.append((neighbor, dist + 1))

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions
        to move all packages to their goal locations.
        """
        state = node.state

        # Track current status of packages and vehicles
        package_current_status = {} # {package: ('at', loc) or ('in', veh)}
        vehicle_current_locations = {} # {vehicle: loc}

        # Populate status and locations from the current state
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip empty facts if any

            predicate = parts[0]
            if predicate == "at" and len(parts) == 3:
                obj, loc = parts[1:]
                if obj in self._packages:
                    package_current_status[obj] = ('at', loc)
                elif obj in self._vehicles:
                    vehicle_current_locations[obj] = loc
            elif predicate == "in" and len(parts) == 3:
                pkg, veh = parts[1:]
                if pkg in self._packages and veh in self._vehicles:
                     package_current_status[pkg] = ('in', veh)

        total_cost = 0

        # Calculate cost for each package that is not at its goal
        for package, goal_l in self.goal_locations.items():
            # If package is not in current state facts (e.g., not 'at' or 'in'),
            # it's an unexpected state. Return infinity.
            if package not in package_current_status:
                 return float('inf')

            status, current_loc_or_veh = package_current_status[package]

            if status == 'at':
                current_l = current_loc_or_veh
                if current_l != goal_l:
                    # Package is on the ground, not at goal.
                    # Needs: Pick-up (1) + Drive (distance) + Drop (1)
                    # Find distance from package's current location to its goal location
                    dist = self.distances.get(current_l, {}).get(goal_l)
                    if dist is None: # Goal location unreachable from current location
                        return float('inf')
                    total_cost += 1 + dist + 1
                # Else: Package is at goal, cost for this package is 0.

            elif status == 'in':
                veh = current_loc_or_veh
                v_current_l = vehicle_current_locations.get(veh)

                if v_current_l is None:
                    # Vehicle location unknown (e.g., vehicle not 'at' any location)
                    # This indicates an invalid state representation.
                    return float('inf')

                # Package is in a vehicle.
                # Needs: Drive (distance) + Drop (1)
                # The vehicle must drive from its current location to the package's goal location.
                dist = self.distances.get(v_current_l, {}).get(goal_l)
                if dist is None: # Goal location unreachable from vehicle's current location
                    return float('inf')
                total_cost += dist + 1

        return total_cost
