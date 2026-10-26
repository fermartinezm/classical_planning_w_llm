# Need to import necessary modules
from heuristics.heuristic_base import Heuristic
from collections import deque # For BFS
# import logging # Optional: for debugging

# Helper function to parse PDDL fact strings
def parse_fact(fact_string):
    """Parses a PDDL fact string into a tuple."""
    # Remove surrounding brackets and split by spaces
    parts = fact_string.strip("()").split()
    return tuple(parts)

class transportHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Transport domain.

    Summary:
    This heuristic estimates the cost to reach the goal state by summing the
    estimated costs for each package that is not yet at its goal location.
    For a package on the ground at location L1 that needs to reach location L2,
    the estimated cost is 2 (for pick-up and drop actions) plus the shortest
    path distance between L1 and L2 in the road network.
    For a package inside a vehicle at location L1 that needs to reach location L2,
    the estimated cost is 1 (for the drop action) plus the shortest path distance
    between L1 and L2. If L1 is already L2, the cost is just 1 (drop).

    Assumptions:
    - The heuristic ignores vehicle capacity constraints. It assumes any package
      can be picked up by any vehicle.
    - The heuristic ignores the cost of getting an *available* vehicle to a package's location
      if the package is on the ground. It assumes a vehicle is available at the
      package's location when needed for pick-up.
    - The heuristic sums costs for each package independently, ignoring potential
      synergies (e.g., a single drive action moving multiple packages) or conflicts
      (e.g., multiple packages needing the same vehicle).
    - The road network is static and provides directed connections.
    - Goal facts are always of the form (at package location).
    - Vehicles are identified by having a (capacity v s) fact in the initial or static state.
    - Packages are identified by being the first argument of an (at p l) goal fact.

    Heuristic Initialization:
    1. Parses the goal facts to identify the target location for each package.
       Stores this in a dictionary `self.package_goals`.
    2. Parses the static and initial state facts to identify all vehicles
       (based on capacity facts). Stores this in a set `self.vehicles`.
    3. Parses the static facts to build a directed graph representation of the
       road network. Stores locations and road connections. Includes locations
       mentioned in goals and initial state even if they have no road facts.
    4. Computes all-pairs shortest path distances between all locations in the
       road network using Breadth-First Search (BFS). Stores these distances
       in a dictionary `self.distances`. Unreachable locations have infinite distance.

    Step-By-Step Thinking for Computing Heuristic:
    1. Initialize the total heuristic value `h_value` to 0.
    2. Create temporary dictionaries to quickly access package locations,
       packages inside vehicles, and vehicle locations from the current state.
       These are populated by iterating through the current state facts and
       checking if the object is a known package or vehicle.
    3. Iterate through each package `p` for which a goal location `goal_l` is defined
       in `self.package_goals`.
    4. Check if the goal fact `(at p goal_l)` is already present in the current state.
       If yes, the package is already at its goal location on the ground, and its
       contribution to the heuristic is 0. Continue to the next package.
    5. If the goal fact `(at p goal_l)` is not in the state:
       a. Check if the package `p` is on the ground. This is true if `(at p current_l)`
          is in the state for some `current_l`.
          - If found at `current_l`: The estimated cost for this package is 2 (pick-up + drop)
            plus the shortest path distance from `current_l` to `goal_l`. Get this distance
            from the precomputed `self.distances`. If the distance is infinity, the cost
            contribution is infinity.
       b. Check if the package `p` is inside a vehicle `v`. This is true if `(in p v)`
          is in the state for some `v`.
          - If found inside vehicle `v`: Find the current location `vehicle_l` of vehicle `v`
            from the state `(at v vehicle_l)`.
            - If `vehicle_l` is the same as `goal_l`: The estimated cost is 1 (drop).
            - If `vehicle_l` is different from `goal_l`: The estimated cost is 1 (drop)
              plus the shortest path distance from `vehicle_l` to `goal_l`. Get this distance
              from `self.distances`. If the distance is infinity, the cost contribution
              is infinity.
       c. If the package is neither on the ground nor in a vehicle (should not happen in valid states),
          its cost contribution is infinity.
       d. Add the calculated cost contribution for package `p` to `h_value`.
    6. Return the final `h_value`. If any cost contribution was infinity, the total
       heuristic value will be infinity, indicating a potentially unsolvable state
       under the heuristic's assumptions.
    """

    def __init__(self, task):
        super().__init__(task) # Call parent constructor

        self.package_goals = {}
        self.locations = set()
        self.road_graph = {} # Adjacency list {l1: {l2, l3}, ...}
        self.distances = {} # {(l1, l2): dist, ...}
        self.vehicles = set() # Collect vehicle names

        # 1. Parse goal facts
        for goal_fact_str in task.goals:
            parsed = parse_fact(goal_fact_str)
            if parsed[0] == 'at' and len(parsed) == 3:
                # Assuming goal facts are (at package location)
                package, goal_l = parsed[1], parsed[2]
                self.package_goals[package] = goal_l
                self.locations.add(goal_l) # Add goal location to known locations

        # 2. Parse static and initial state facts to identify vehicles and build road graph
        # Static facts
        for static_fact_str in task.static:
            parsed = parse_fact(static_fact_str)
            if parsed[0] == 'road' and len(parsed) == 3:
                l1, l2 = parsed[1], parsed[2]
                self.locations.add(l1)
                self.locations.add(l2)
                self.road_graph.setdefault(l1, set()).add(l2)
            elif parsed[0] == 'capacity' and len(parsed) == 3:
                 vehicle = parsed[1]
                 self.vehicles.add(vehicle)
            # Ignore capacity-predecessor for this heuristic

        # Initial state facts (also check for vehicles and initial locations)
        # We don't need initial locations/in status here for __init__, only in __call__
        for init_fact_str in task.initial_state:
             parsed = parse_fact(init_fact_str)
             if parsed[0] == 'capacity' and len(parsed) == 3:
                 vehicle = parsed[1]
                 self.vehicles.add(vehicle)
             # Add initial locations of packages and vehicles to known locations
             elif parsed[0] == 'at' and len(parsed) == 3:
                 obj, loc = parsed[1], parsed[2]
                 self.locations.add(loc)


        # 3. Compute all-pairs shortest paths using BFS
        for start_l in self.locations:
            q = deque([(start_l, 0)])
            visited = {start_l}
            self.distances[(start_l, start_l)] = 0

            while q:
                curr_l, dist = q.popleft()

                # Get neighbors, handle locations with no outgoing roads
                neighbors = self.road_graph.get(curr_l, set())

                for next_l in neighbors:
                    if next_l not in visited:
                        visited.add(next_l)
                        self.distances[(start_l, next_l)] = dist + 1
                        q.append((next_l, dist + 1))

        # Ensure all pairs have a distance (infinity if not found)
        for l1 in self.locations:
            for l2 in self.locations:
                if (l1, l2) not in self.distances:
                    self.distances[(l1, l2)] = float('inf')


    def __call__(self, node):
        state = node.state
        h_value = 0

        # Temporary dictionaries for quick state lookup
        package_locations = {} # {p: l} if (at p l)
        package_in_vehicle = {} # {p: v} if (in p v)
        vehicle_locations = {} # {v: l} if (at v l)

        for fact_str in state:
            parsed = parse_fact(fact_str)
            if parsed[0] == 'at' and len(parsed) == 3:
                obj, loc = parsed[1], parsed[2]
                if obj in self.package_goals: # It's a package with a goal
                     package_locations[obj] = loc
                elif obj in self.vehicles: # It's a vehicle
                     vehicle_locations[obj] = loc
                # Ignore other 'at' facts if any

            elif parsed[0] == 'in' and len(parsed) == 3:
                p, v = parsed[1], parsed[2]
                package_in_vehicle[p] = v
            # Ignore capacity facts in the state for this heuristic

        # Iterate through packages that have a goal location
        for package, goal_l in self.package_goals.items():
            # Check if the goal is already satisfied: (at package goal_l) is in state
            goal_fact_str = f'(at {package} {goal_l})'
            if goal_fact_str in state:
                continue # Goal for this package is reached

            # Goal is not reached, calculate cost for this package
            package_cost = float('inf') # Default to infinity

            if package in package_locations:
                # Package is on the ground
                current_l = package_locations[package]
                # Cost = pick-up (1) + drive (dist) + drop (1)
                dist = self.distances.get((current_l, goal_l), float('inf'))
                if dist != float('inf'):
                     package_cost = 2 + dist
                else:
                     package_cost = float('inf') # Cannot reach goal location from current ground location

            elif package in package_in_vehicle:
                # Package is inside a vehicle
                vehicle = package_in_vehicle[package]
                vehicle_l = vehicle_locations.get(vehicle) # Get vehicle's location

                if vehicle_l is not None: # Vehicle must have a location
                    # Cost = drive (dist) + drop (1)
                    dist = self.distances.get((vehicle_l, goal_l), float('inf'))
                    if dist != float('inf'):
                        package_cost = 1 + dist
                    else:
                        package_cost = float('inf') # Cannot reach goal location from vehicle's current location
                else:
                    # This case should ideally not happen in a valid state,
                    # but handle defensively. Vehicle location unknown -> cannot move package.
                    # This implies the state is likely invalid or unsolvable.
                    package_cost = float('inf')
            else:
                 # Package is not on the ground and not in a vehicle.
                 # This should not happen in a valid state based on domain predicates.
                 # If it does, the state is likely invalid or unsolvable.
                 package_cost = float('inf')


            # Add package cost to total heuristic value
            # If any package cost is infinity, the total becomes infinity
            h_value += package_cost


        return h_value
