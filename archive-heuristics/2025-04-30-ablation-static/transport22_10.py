# Import necessary modules
from collections import deque
# Assuming a Heuristic base class exists and is imported elsewhere
from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact string."""
    # Remove parentheses and split by whitespace
    return fact[1:-1].split()

# Define the heuristic class inheriting from the assumed base class
class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions required to move each package
    to its goal location, summing the individual costs. It considers the current
    location of the package (on the ground or in a vehicle) and the shortest
    path distance in the road network. Capacity constraints and vehicle availability
    beyond the package's current state are ignored.

    # Assumptions
    - Each package needs to reach a specific goal location specified in the task goals.
    - The cost of driving between locations is the shortest path distance in the
      road network defined by `road` predicates. Each drive action moves a vehicle
      one step along a road.
    - Picking up a package costs 1 action.
    - Dropping a package costs 1 action.
    - Capacity constraints are not explicitly modeled in the heuristic cost calculation.
    - Vehicle availability is not explicitly modeled; it's assumed a vehicle
      can be used when needed for a package's transport segment.
    - All locations involved in package initial positions and goals are reachable
      from each other via the road network, possibly indirectly. Unreachable goals
      result in an infinite heuristic value.

    # Heuristic Initialization
    - Extracts goal locations for each package from the task goals.
    - Builds the road network graph from `road` static facts.
    - Computes shortest path distances between all pairs of locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize total heuristic cost to 0.
    2. Parse the current state to determine the location of each package and vehicle.
       - A package can be on the ground at a location `l_p` (`(at package l_p)`).
       - A package can be inside a vehicle `v` (`(in package v)`).
       - A vehicle is on the ground at a location `l_v` (`(at vehicle l_v)`).
    3. For each package that has a goal location specified in the task:
       a. Check if the package is already at its goal location on the ground (`(at package goal_location)` is in the state).
          If yes, this package contributes 0 to the heuristic. Continue to the next package.
       b. If the package is not at its goal location on the ground:
          i. Determine the package's current status based on the state facts:
             - If `(at package l_p)` is in the state (where `l_p` is not the goal): The package is on the ground at `l_p`.
             - If `(in package v)` is in the state: The package is inside vehicle `v`.
          ii. Calculate the estimated cost for this package:
              - If the package is on the ground at `l_p`:
                  - It needs to be picked up (cost 1).
                  - It needs to be transported from `l_p` to its goal `l_goal`. The minimum drive cost is the shortest path distance between `l_p` and `l_goal`.
                  - It needs to be dropped at `l_goal` (cost 1).
                  - Estimated cost = 1 (pick-up) + distance(l_p, l_goal) (drive) + 1 (drop).
              - If the package is inside a vehicle `v`:
                  - Find the current location `l_v` of vehicle `v` from the state (`(at v l_v)`).
                  - It needs to be transported from `l_v` to its goal `l_goal`. The minimum drive cost is the shortest path distance between `l_v` and `l_goal`.
                  - It needs to be dropped at `l_goal` (cost 1).
                  - Estimated cost = distance(l_v, l_goal) (drive) + 1 (drop).
              - If the package's status cannot be determined (e.g., not `at` any location and not `in` any vehicle), this indicates an invalid state or an unreachable package, assign infinite cost.
          iii. Add the estimated cost for this package to the total heuristic cost.
    4. If the total cost calculated is infinite (meaning at least one package goal is unreachable), return a large number or infinity. Otherwise, return the total cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal locations and computing
        shortest path distances in the road network.
        """
        # The set of facts that must hold in goal states.
        self.goals = task.goals
        # Static facts are not affected by actions.
        static_facts = task.static

        # 1. Extract goal locations for each package from the task goals.
        self.package_goals = {}
        for goal in self.goals:
            parts = get_parts(goal)
            # Goal facts are typically (at package location)
            if parts[0] == "at":
                package, location = parts[1], parts[2]
                self.package_goals[package] = location

        # 2. Build road network graph and compute distances.
        locations, graph = self._build_road_graph(static_facts)
        self.distances = self._compute_distances(locations, graph)
        self.locations = locations # Store locations list for potential use

    def _build_road_graph(self, static_facts):
        """Build adjacency list for locations based on road facts."""
        graph = {}
        locations = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "road":
                l1, l2 = parts[1], parts[2]
                locations.add(l1)
                locations.add(l2)
                # Add edge l1 -> l2
                graph.setdefault(l1, []).append(l2)
                # Assuming roads are bidirectional unless specified otherwise
                # Add edge l2 -> l1
                graph.setdefault(l2, []).append(l1)
        return list(locations), graph

    def _compute_distances(self, locations, graph):
        """Compute shortest path distances between all pairs of locations using BFS."""
        distances = {}
        # Use a large number to represent infinity for unreachable locations
        infinity = float('inf')

        for start_loc in locations:
            q = deque([(start_loc, 0)])
            visited = {start_loc}
            distances[(start_loc, start_loc)] = 0 # Distance from a location to itself is 0

            while q:
                current_loc, dist = q.popleft()

                # Check if current_loc exists as a key in the graph
                if current_loc in graph:
                    for neighbor in graph[current_loc]:
                        # Ensure neighbor is a valid location and not already visited in this BFS run
                        if neighbor in locations and neighbor not in visited:
                            visited.add(neighbor)
                            distances[(start_loc, neighbor)] = dist + 1
                            q.append((neighbor, dist + 1))

        # Ensure all pairs have a distance entry (either computed or infinity)
        for l1 in locations:
            for l2 in locations:
                if (l1, l2) not in distances:
                    distances[(l1, l2)] = infinity

        return distances


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state as a frozenset of strings

        # Track current locations of packages and vehicles by parsing the state
        current_locations = {} # Maps locatable object (package or vehicle) to its location (if on ground)
        package_in_vehicle = {} # Maps package to vehicle it's in

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at":
                obj, location = parts[1], parts[2]
                current_locations[obj] = location
            elif parts[0] == "in":
                package, vehicle = parts[1], parts[2]
                package_in_vehicle[package] = vehicle

        total_cost = 0
        infinity = float('inf')

        # Iterate through packages that have a goal location defined in the task
        for package, goal_location in self.package_goals.items():
            # Check if the package is already at its goal location on the ground
            if f"(at {package} {goal_location})" in state:
                 continue # This package goal is satisfied in the current state

            # Package is not yet at its goal location on the ground.
            # Determine its current status (on ground elsewhere or in a vehicle).

            package_cost = 0
            current_package_status_known = False

            # Case 1: Package is on the ground at some location l_p (which is not the goal)
            if package in current_locations and current_locations[package] != goal_location:
                 current_package_location = current_locations[package]
                 current_package_status_known = True

                 # Estimated cost: pick-up + drive + drop
                 pick_up_cost = 1
                 drop_cost = 1
                 drive_dist = self.distances.get((current_package_location, goal_location), infinity)

                 package_cost = pick_up_cost + drive_dist + drop_cost

            # Case 2: Package is inside a vehicle v
            elif package in package_in_vehicle:
                 current_vehicle = package_in_vehicle[package]
                 current_package_status_known = True

                 # Find the location of the vehicle
                 if current_vehicle in current_locations:
                     current_vehicle_location = current_locations[current_vehicle]

                     # Estimated cost: drive (by vehicle) + drop
                     drop_cost = 1
                     drive_dist = self.distances.get((current_vehicle_location, goal_location), infinity)

                     package_cost = drive_dist + drop_cost
                 else:
                     # Vehicle location is unknown - indicates an issue or unreachable vehicle
                     package_cost = infinity

            # Case 3: Package status is not known (not at any location, not in any vehicle)
            # This should not happen in a valid state, but handle defensively.
            if not current_package_status_known:
                 package_cost = infinity


            # Add the cost for this package to the total
            total_cost += package_cost

            # If any package is unreachable, the total cost becomes infinity
            if total_cost == infinity:
                 return infinity # Return infinity immediately if any part is unreachable

        # Return the total estimated cost.
        # If total_cost is 0, it means all package goals were satisfied.
        return total_cost
