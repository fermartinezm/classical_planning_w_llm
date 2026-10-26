# Assume Heuristic base class is available from the environment
from heuristics.heuristic_base import Heuristic

from fnmatch import fnmatch
from collections import deque
import math

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # The number of parts in the fact must match the number of arguments in the pattern
    if len(parts) != len(args):
        return False
    # Check if each part matches the corresponding argument pattern
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the total number of actions required to move all
    packages to their goal locations. It calculates the minimum actions needed
    for each package independently, ignoring vehicle capacity and potential
    conflicts, and sums these minimums. The minimum actions for a package
    depend on its current state (at a location or in a vehicle) and the
    shortest path distance in the road network.

    # Assumptions
    - Each package needs to be picked up, transported, and dropped at its goal.
    - The cost of transporting a package between two locations is the shortest
      path distance between those locations in the road network, assuming a
      vehicle is available.
    - Vehicle capacity constraints are ignored.
    - Vehicle availability is ignored (assumes a vehicle is available when needed).
    - The heuristic sums the minimum costs for each package independently.
    - Goals are always of the form (at package location).
    - Any locatable object in an 'at' predicate that is not a goal package is treated as a vehicle for location tracking purposes.

    # Heuristic Initialization
    - Extracts the goal location for each package from the task goals.
    - Builds a graph of locations based on the 'road' facts.
    - Computes the shortest path distance between all pairs of locations
      using Breadth-First Search (BFS).

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current location or containment (inside a vehicle) for every package that has a goal.
    2. Identify the current location for every vehicle present in the state (approximated by any locatable in an 'at' fact that isn't a goal package).
    3. Initialize the total heuristic cost to 0.
    4. For each goal fact `(at package goal_location)` in the task goals:
       a. If the goal fact `(at package goal_location)` is already true in the current state, the cost for this goal is 0. Continue to the next goal.
       b. If the goal fact is not true, find the package's current status (location or vehicle containment) from the state information gathered in step 1.
       c. If the package's state (location or containment) cannot be determined from the current state facts, this indicates an invalid state or problem definition. Return infinity.
       d. If the package is currently at `l_current`:
          - The package needs to be picked up (1 action), transported from `l_current` to `goal_location` (shortest path distance), and dropped (1 action).
          - Add `1 + shortest_path(l_current, goal_location) + 1` to the total cost. If the goal location is unreachable from the current location, the path cost is infinity, and the heuristic should return infinity.
       e. If the package is currently inside a vehicle `v_current`:
          - Find the current location `l_v` of the vehicle `v_current` from the state information gathered in step 2.
          - If the vehicle's location cannot be determined, return infinity.
          - If `l_v` is the same as `goal_location`, the package only needs to be dropped (1 action). Add `1` to the total cost.
          - If `l_v` is different from `goal_location`, the vehicle needs to drive from `l_v` to `goal_location` (shortest path distance), and then the package needs to be dropped (1 action). Add `shortest_path(l_v, goal_location) + 1` to the total cost. If the goal location is unreachable from the vehicle's current location, the path cost is infinity, and the heuristic should return infinity.
    5. Return the total accumulated cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and building
        the location graph for shortest path calculations.
        """
        super().__init__(task) # Call base class constructor if inheriting

        # Extract goal locations for packages
        self.package_goal_locations = {}
        # Also identify all packages that are goals
        self.goal_packages = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "at" and len(parts) == 3:
                package, location = parts[1], parts[2]
                # Assuming goals are always (at package location)
                self.package_goal_locations[package] = location
                self.goal_packages.add(package)


        # Build the road graph from static facts
        self.road_graph = {}
        locations = set()
        for fact in self.static: # Use self.static from base class
            parts = get_parts(fact)
            if parts[0] == "road" and len(parts) == 3:
                l1, l2 = parts[1], parts[2]
                if l1 not in self.road_graph:
                    self.road_graph[l1] = []
                self.road_graph[l1].append(l2)
                locations.add(l1)
                locations.add(l2)

        # Ensure all locations mentioned in roads are keys, even if they have no outgoing roads
        for loc in locations:
             if loc not in self.road_graph:
                 self.road_graph[loc] = []

        # Compute all-pairs shortest paths using BFS
        self.shortest_paths = {}
        all_locations = list(self.road_graph.keys()) # Use keys to get all locations mentioned in roads

        for start_node in all_locations:
            distances = {loc: math.inf for loc in all_locations}
            distances[start_node] = 0
            queue = deque([start_node])

            while queue:
                current_node = queue.popleft()

                # If current_node is not in graph keys (shouldn't happen if all_locations comes from keys), skip
                if current_node not in self.road_graph:
                     continue

                for neighbor in self.road_graph[current_node]:
                    if distances[neighbor] == math.inf:
                        distances[neighbor] = distances[current_node] + 1
                        queue.append(neighbor)

            # Store distances from start_node to all reachable nodes
            for end_node in all_locations:
                 self.shortest_paths[(start_node, end_node)] = distances[end_node]


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Track current locations/containment for packages and vehicles
        current_package_state = {} # package -> ('at', loc) or ('in', vehicle)
        current_vehicle_location = {} # vehicle -> loc

        # Populate current state dictionaries by iterating through the state facts
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at" and len(parts) == 3:
                locatable, location = parts[1], parts[2]
                # Check if this locatable is one of the packages we care about (i.e., has a goal)
                if locatable in self.goal_packages:
                     current_package_state[locatable] = ('at', location)
                else:
                     # Assume other locatables are vehicles for tracking their location
                     current_vehicle_location[locatable] = location

            elif parts[0] == "in" and len(parts) == 3:
                package, vehicle = parts[1], parts[2]
                # Check if this is a package we care about
                if package in self.goal_packages:
                    current_package_state[package] = ('in', vehicle)
                # Vehicle location is tracked by 'at' predicate, not 'in'

        total_cost = 0  # Initialize action cost counter.

        # Iterate through packages that have a goal
        for package, goal_location in self.package_goal_locations.items():
             # Check if the goal for this specific package is already met
             goal_fact_for_package = f"(at {package} {goal_location})"
             if goal_fact_for_package in state:
                 continue # Goal already met for this package, cost is 0 for this package

             # Goal is not met. Calculate cost for this package.
             package_status = current_package_state.get(package)

             if package_status is None:
                 # Package state is unknown (not at any location, not in any vehicle).
                 # This indicates an invalid state or problem definition.
                 # Return infinity as this state is likely unreachable or problematic.
                 return math.inf

             status_type, current_loc_or_vehicle = package_status

             if status_type == 'at':
                 l_current = current_loc_or_vehicle
                 # Package is at l_current, needs to go to goal_location
                 # We already know l_current != goal_location because goal_fact is not in state
                 d = self.shortest_paths.get((l_current, goal_location), math.inf)
                 if d == math.inf:
                     # Goal location is unreachable from current location
                     return math.inf
                 total_cost += 1 + d + 1 # pick-up, drive, drop

             elif status_type == 'in':
                 v_current = current_loc_or_vehicle
                 # Package is in v_current, need vehicle location
                 l_v = current_vehicle_location.get(v_current)

                 if l_v is None:
                     # Vehicle location is unknown. Should not happen in a valid state.
                     return math.inf # Vehicle location unknown, cannot proceed

                 # Vehicle is at l_v, needs to go to goal_location to drop package
                 # We already know that if l_v == goal_location, the goal_fact would be in state
                 # (assuming the package is dropped immediately upon arrival).
                 # However, the goal is (at package goal_loc), not (in package vehicle) AND (at vehicle goal_loc).
                 # So, if (in package vehicle) and (at vehicle goal_loc) is true, the package still needs to be dropped.
                 # The goal (at package goal_loc) is only met *after* the drop.
                 # So, if the package is in a vehicle, it always needs a drop action.
                 # If the vehicle is not at the goal location, it also needs to drive.

                 if l_v != goal_location:
                     d = self.shortest_paths.get((l_v, goal_location), math.inf)
                     if d == math.inf:
                         # Goal location is unreachable from vehicle location
                         return math.inf
                     total_cost += d + 1 # drive, drop
                 else: # Vehicle is already at goal_location
                     total_cost += 1 # drop


        # The heuristic should be 0 only for goal states.
        # Our calculation sums costs for unmet goals. If all goals are met, total_cost is 0.
        # If not all goals are met, total_cost > 0 (assuming reachable).
        # If any goal location is unreachable from the package/vehicle location, we return inf.
        # This satisfies the condition that heuristic is 0 only for goal states and finite for solvable states.

        return total_cost
