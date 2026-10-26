from fnmatch import fnmatch
from collections import deque
import math

# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic # This import is assumed


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact string or malformed fact
    if not fact or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()


class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the minimum number of actions (pick-up, drop, drive)
    required to move each package from its current location to its goal location.
    It sums the estimated costs for each package independently.

    # Assumptions
    - Each package needs to reach a specific goal location.
    - Vehicles are assumed to be available and have sufficient capacity whenever needed
      to pick up a package or transport it.
    - The cost of moving a package is the sum of:
        - 1 action to pick up (if on the ground).
        - The shortest path distance (number of drive actions) for the vehicle
          carrying the package from its current location to the package's goal location.
        - 1 action to drop the package at the goal location.
    - Road network is static and bidirectional.

    # Heuristic Initialization
    - Extracts the goal location for each package from the task's goal conditions.
    - Builds a graph of locations based on static 'road' facts.
    - Computes the shortest path distance (number of drive actions) between all pairs
      of locations using Breadth-First Search (BFS). These distances are stored
      for quick lookup during heuristic computation.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current location or containing vehicle for every package that has a goal.
       Also identify the current location of every vehicle present in the state.
    2. Initialize the total heuristic cost to 0.
    3. For each package that has a goal location:
        a. Check if the package is already at its goal location. If yes, the cost for this package is 0.
        b. If the package is currently on the ground at a location different from its goal:
            - Estimate the cost as 2 (for pick-up and drop actions) plus the shortest
              distance (number of drive actions) from the package's current location
              to its goal location. Add this to the total cost.
            - If the goal location is unreachable from the current location, the cost
              for this package is infinite (or a very large number), making the state
              undesirable for greedy search.
        c. If the package is currently inside a vehicle:
            - Find the current location of the vehicle.
            - Estimate the cost as 1 (for the drop action) plus the shortest distance
              (number of drive actions) from the vehicle's current location to the
              package's goal location. Add this to the total cost.
            - If the goal location is unreachable from the vehicle's current location,
              the cost for this package is infinite.
    4. The total heuristic value for the state is the sum of the estimated costs
       for all packages not yet at their goal.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal locations and pre-calculating
        shortest path distances between locations.
        """
        self.goals = task.goals
        static_facts = task.static

        # 1. Extract goal locations for each package.
        self.goal_locations = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "at":
                # Goal is (at package location)
                if len(parts) == 3:
                    package, location = parts[1], parts[2]
                    self.goal_locations[package] = location
                # Assuming goals are always (at package location).

        # 2. Build the location graph from road facts.
        self.location_graph = {}
        locations = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "road":
                if len(parts) == 3:
                    loc1, loc2 = parts[1], parts[2]
                    locations.add(loc1)
                    locations.add(loc2)
                    self.location_graph.setdefault(loc1, set()).add(loc2)
                    # Assuming roads are bidirectional based on examples
                    self.location_graph.setdefault(loc2, set()).add(loc1)

        self.locations = list(locations) # Store as list for consistent iteration order

        # 3. Compute all-pairs shortest paths using BFS.
        self.distances = {}
        for start_loc in self.locations:
            self._bfs(start_loc)

    def _bfs(self, start_node):
        """
        Performs BFS from a start node to find distances to all reachable nodes.
        Stores results in self.distances.
        """
        q = deque([start_node])
        # Use a dictionary for distances for sparse graphs and easy lookup
        dist = {node: math.inf for node in self.locations}
        dist[start_node] = 0

        while q:
            u = q.popleft()

            # Store the distance from start_node to u
            self.distances[(start_node, u)] = dist[u]

            # Explore neighbors
            neighbors = self.location_graph.get(u, [])
            for v in neighbors:
                if dist[v] == math.inf: # If not visited
                    dist[v] = dist[u] + 1
                    q.append(v)

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state

        # Build quick lookups for current locations/containment
        package_locations = {} # Maps package to its location or vehicle
        vehicle_locations = {} # Maps vehicle to its location

        # Identify packages and vehicles present in the state
        # This is needed to distinguish objects when parsing 'at' facts
        # and to check if a 'current_pos' is a vehicle name.
        packages_in_goals = set(self.goal_locations.keys())
        vehicles_in_state = set() # Collect vehicle names found in state

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]
            if predicate == "at" and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                if obj in packages_in_goals:
                    package_locations[obj] = loc
                elif loc in self.locations: # Assume anything else 'at' a location is a vehicle
                     vehicle_locations[obj] = loc
                     vehicles_in_state.add(obj) # Add to set of known vehicles

            elif predicate == "in" and len(parts) == 3:
                package, vehicle = parts[1], parts[2]
                # Assuming the first argument of 'in' is always a package
                if package in packages_in_goals:
                     package_locations[package] = vehicle
                # Assuming the second argument of 'in' is always a vehicle
                vehicles_in_state.add(vehicle)


        total_cost = 0

        # Iterate through packages that have a goal
        for package, goal_loc in self.goal_locations.items():
            # Check if package is already at goal
            # Need to check if the fact (at package goal_loc) is explicitly in the state
            if f"(at {package} {goal_loc})" in state:
                 continue # Package is at goal, cost is 0 for this package

            current_pos = package_locations.get(package)

            if current_pos is None:
                 # Package state is unknown. Treat as unreachable.
                 return math.inf

            if current_pos in self.locations: # Package is on the ground at a location
                current_loc = current_pos
                # Cost: pick-up (1) + drive (distance) + drop (1)
                dist = self.distances.get((current_loc, goal_loc), math.inf)
                if dist == math.inf:
                    return math.inf # Goal is unreachable from current location
                total_cost += 2 + dist

            elif current_pos in vehicles_in_state: # Package is inside a vehicle
                vehicle = current_pos
                loc_v = vehicle_locations.get(vehicle)

                if loc_v is None:
                    # Vehicle location is unknown. Invalid state.
                    return math.inf

                # Cost: drive (distance) + drop (1)
                dist = self.distances.get((loc_v, goal_loc), math.inf)
                if dist == math.inf:
                    return math.inf # Goal is unreachable from vehicle's location
                total_cost += 1 + dist
            else:
                 # current_pos is neither a known location nor a known vehicle. Invalid state.
                 return math.inf

        # The heuristic is 0 if and only if total_cost is 0, which happens
        # if and only if all packages in self.goal_locations were found
        # to be at their respective goal locations in the state.
        # This matches the goal condition structure.

        return total_cost
