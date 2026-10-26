# Need to ensure the Heuristic base class is available.
# Assuming it's in a file named heuristic_base.py in a directory named heuristics.
# The import `from heuristics.heuristic_base import Heuristic` is correct based on the example.

from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import math

# Helper functions (adapted from Logistics example)
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
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions needed to move each package
    from its current location to its goal location, summing the costs for all packages.
    It considers pick-up, drop, and drive actions, using shortest path distances
    in the road network for drive costs.

    # Assumptions
    - Capacity constraints are ignored. Any vehicle can theoretically carry any package.
    - Any vehicle can be used to transport any package.
    - The cost of driving between two locations is the shortest path distance in the road network,
      where each road segment costs 1 drive action.
    - The heuristic sums the minimum cost for each package independently, ignoring potential
      synergies (like transporting multiple packages in one trip) or conflicts (like
      multiple packages needing the same vehicle).
    - Goal conditions only involve packages being at specific locations (`(at package location)`).

    # Heuristic Initialization
    - Extract the goal locations for each package from the task goals.
    - Build a graph of locations based on the `road` predicates from static facts.
    - Compute all-pairs shortest path distances between all locations using the road network.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Check if the current state is a goal state using the task's goal conditions. If yes, the heuristic value is 0.
    2. If not a goal state, identify the current location of every package that has a goal.
       A package can be either `at` a location or `in` a vehicle. If it's in a vehicle,
       find the location of that vehicle.
    3. Initialize the total heuristic cost to 0.
    4. For each package that has a goal location:
       a. Determine the package's current status: `at l_current` or `in v` (where `v` is `at l_v`).
       b. Get the package's goal location, `l_goal`.
       c. If the package is currently `at` location `l_current`:
          - If `l_current` is the same as `l_goal`, the cost for this package is 0.
          - If `l_current` is different from `l_goal`, the estimated cost for this package is:
            1 (pick-up action) + shortest_distance(l_current, l_goal) (drive actions) + 1 (drop action).
            If `l_goal` is unreachable from `l_current`, the distance is infinity, and the cost is infinity.
       d. If the package is currently `in` vehicle `v`, and vehicle `v` is `at` location `l_v`:
          - The estimated cost for this package is:
            shortest_distance(l_v, l_goal) (drive actions) + 1 (drop action).
            If `l_goal` is unreachable from `l_v`, the distance is infinity, and the cost is infinity.
       e. Add the estimated cost for this package to the total heuristic cost.
    5. Return the total heuristic cost. If any package's goal is unreachable, the total cost will be infinity.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal locations and precomputing
        shortest path distances between locations.
        """
        self.task = task # Store task to use task.goal_reached
        self.goals = task.goals
        static_facts = task.static

        # Store goal locations for each package.
        self.package_goals = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                # Goal is (at package location)
                package, location = args
                self.package_goals[package] = location
            # Assuming only (at package location) goals based on typical transport problems.

        # Build the location graph from road facts.
        self.locations = set()
        self.road_graph = {} # Adjacency list: location -> set of connected locations

        for fact in static_facts:
            if match(fact, "road", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                self.locations.add(loc1)
                self.locations.add(loc2)
                self.road_graph.setdefault(loc1, set()).add(loc2)
                self.road_graph.setdefault(loc2, set()).add(loc1) # Assuming roads are bidirectional

        self.locations = sorted(list(self.locations)) # Ensure consistent ordering
        self.location_to_index = {loc: i for i, loc in enumerate(self.locations)}
        num_locations = len(self.locations)

        # Compute all-pairs shortest paths using Floyd-Warshall
        self.distances = {} # Store distances as (loc1, loc2) -> distance

        # Initialize distances
        for i in range(num_locations):
            for j in range(num_locations):
                loc_i = self.locations[i]
                loc_j = self.locations[j]
                if i == j:
                    self.distances[(loc_i, loc_j)] = 0
                elif loc_j in self.road_graph.get(loc_i, set()):
                    self.distances[(loc_i, loc_j)] = 1 # Cost of one drive action
                else:
                    self.distances[(loc_i, loc_j)] = math.inf # Not directly connected

        # Floyd-Warshall
        for k_idx in range(num_locations):
            k_loc = self.locations[k_idx]
            for i_idx in range(num_locations):
                i_loc = self.locations[i_idx]
                for j_idx in range(num_locations):
                    j_loc = self.locations[j_idx]
                    current_dist = self.distances[(i_loc, j_loc)]
                    path_through_k = self.distances.get((i_loc, k_loc), math.inf) + self.distances.get((k_loc, j_loc), math.inf)
                    if path_through_k < current_dist:
                        self.distances[(i_loc, j_loc)] = path_through_k


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Check if the state is a goal state
        if self.task.goal_reached(state):
             return 0

        # Track current locations of locatables (packages and vehicles)
        current_locations = {}
        # Track which package is inside which vehicle
        package_in_vehicle = {}

        for fact in state:
            predicate, *args = get_parts(fact)
            if predicate == "at":
                # (at ?x ?l) where ?x is locatable (vehicle or package)
                locatable, location = args
                current_locations[locatable] = location
            elif predicate == "in":
                # (in ?p ?v) where ?p is package, ?v is vehicle
                package, vehicle = args
                package_in_vehicle[package] = vehicle

        total_cost = 0

        # Consider each package that has a goal location
        for package, goal_location in self.package_goals.items():
            cost_for_package = 0

            # Find the package's current state
            if package in package_in_vehicle:
                # Package is inside a vehicle
                vehicle = package_in_vehicle[package]
                # Find the vehicle's location
                vehicle_location = current_locations.get(vehicle)
                if vehicle_location is None:
                     # Vehicle location unknown - implies an issue or unreachable goal
                     cost_for_package = math.inf
                else:
                    # Cost: drive vehicle from its location to package goal location + drop
                    dist = self.distances.get((vehicle_location, goal_location), math.inf)
                    if dist == math.inf:
                        cost_for_package = math.inf # Goal unreachable from vehicle's location
                    else:
                        cost_for_package = dist + 1 # drive + drop

            elif package in current_locations:
                # Package is at a location
                package_location = current_locations[package]

                # If package is already at goal, cost is 0 for this package
                if package_location == goal_location:
                    cost_for_package = 0
                else:
                    # Cost: pick-up + drive from package location to goal location + drop
                    dist = self.distances.get((package_location, goal_location), math.inf)
                    if dist == math.inf:
                         cost_for_package = math.inf # Goal unreachable from package's location
                    else:
                        cost_for_package = 1 + dist + 1 # pick-up + drive + drop
            else:
                 # Package location is unknown (not at, not in) - implies an issue or unreachable goal
                 cost_for_package = math.inf

            # If any package has an infinite cost, the total cost is infinite
            if cost_for_package == math.inf:
                return math.inf

            total_cost += cost_for_package

        return total_cost
