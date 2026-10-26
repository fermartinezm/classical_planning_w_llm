# Assuming heuristics.heuristic_base.Heuristic is available
# from heuristics.heuristic_base import Heuristic

# If running standalone for testing, define a dummy base class
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    # Define a dummy Heuristic base class if the actual one is not available
    class Heuristic:
        def __init__(self, task):
            self.task = task
            self.goals = task.goals
            self.static = task.static

        def __call__(self, node):
            raise NotImplementedError

from fnmatch import fnmatch
from collections import deque

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and starts/ends with parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
         # Handle potential malformed facts or non-fact strings in state/static
         return []
    return fact[1:-1].split()

# Note: The 'match' function from examples is useful but not strictly necessary
# if we parse facts manually as done in __call__. Let's keep it simple and
# parse directly in __call__ where needed for clarity and to avoid potential
# issues with complex patterns vs simple fact structures in this domain.
# However, let's keep the get_parts function as it's clean.

class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions required to move each package
    to its goal location. It sums the estimated costs for each package
    independently, ignoring vehicle capacity constraints and assuming a vehicle
    is available when needed. The cost for a package includes pick-up (if on ground),
    driving the vehicle, and dropping off the package. Driving cost is estimated
    by the shortest path distance in the road network.

    # Assumptions
    - The goal is to move specific packages to specific locations.
    - Any vehicle can transport any package (ignoring size/capacity).
    - A vehicle is available at the package's location (or the vehicle's location if carried)
      when needed for pick-up or transport.
    - The road network is undirected (if road A-B exists, road B-A exists).
    - All locations relevant to goals and initial state are part of the road network
      or reachable within connected components.
    - In any valid state, a goal package is either at a location on the ground
      or inside a vehicle which is itself at a location.

    # Heuristic Initialization
    - Extracts the goal location for each package from the task goals.
    - Builds a graph representation of the road network from static facts
      to enable shortest path calculations.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current location of every locatable object (packages on ground, vehicles)
       by parsing `(at obj loc)` facts.
    2. Identify which packages are inside which vehicles by parsing `(in package vehicle)` facts.
    3. Initialize the total heuristic cost to 0.
    4. Initialize a cache for storing computed distances between locations within this call.
    5. For each package that has a goal location specified:
       a. Check if the package is already at its goal location in the current state by checking
          for the fact `(at package goal_location)`. If yes, the cost for this package is 0,
          continue to the next package.
       b. If the package is not at its goal:
          i. Determine the package's current physical location. This is either
             the location where it is on the ground (found in step 1 if not in a vehicle)
             or the location of the vehicle it is inside (found in step 2, then look up
             the vehicle's location from step 1).
          ii. If the package's location/vehicle or the vehicle's location cannot be found
              in the state (indicating an unexpected state structure), add a large penalty
              and continue.
          iii. Calculate the shortest distance (number of drive actions) between
              the package's current physical location and its goal location using
              the precomputed road network graph (BFS). Use the distance cache to avoid
              recomputing the same path.
          iv. Estimate the actions needed for this package:
              - If the package is on the ground at its current location:
                Cost = 1 (pick-up) + distance + 1 (drop).
              - If the package is inside a vehicle at the vehicle's current location:
                Cost = distance + 1 (drop). (Pick-up is already done).
          v. Add this estimated cost to the total heuristic cost.
    6. Return the total heuristic cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting:
        - Goal locations for each package.
        - The road network graph from static facts.
        """
        super().__init__(task)

        # Store goal locations for each package.
        self.goal_locations = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "at" and len(parts) == 3:
                # Goal is (at package location)
                package, location = parts[1], parts[2]
                self.goal_locations[package] = location

        # Build the location graph from (road l1 l2) facts.
        self.location_graph = {}
        locations = set()

        # Collect all locations mentioned in roads
        for fact in self.static:
            parts = get_parts(fact)
            if parts and parts[0] == "road" and len(parts) == 3:
                l1, l2 = parts[1], parts[2]
                locations.add(l1)
                locations.add(l2)
                self.location_graph.setdefault(l1, set()).add(l2)
                self.location_graph.setdefault(l2, set()).add(l1) # Roads are bidirectional

        # Also collect locations from initial state and goals to ensure all relevant nodes are in the graph
        for fact in self.task.initial_state:
             parts = get_parts(fact)
             if parts and parts[0] == "at" and len(parts) == 3:
                 locations.add(parts[2]) # location is the 3rd part

        for goal in self.goals:
             parts = get_parts(goal)
             if parts and parts[0] == "at" and len(parts) == 3:
                 locations.add(parts[2]) # location is the 3rd part

        # Ensure all collected locations are keys in the graph dictionary, even if they have no roads
        for loc in locations:
             self.location_graph.setdefault(loc, set())


    def get_distance(self, start_loc, end_loc, cache):
        """
        Computes the shortest path distance (number of drive actions) between
        two locations using BFS on the road network graph. Caches results.
        Returns a large number if locations are in different unconnected components
        or if either location is not part of the road network graph.
        """
        if start_loc == end_loc:
            return 0

        # Check cache
        cache_key = (start_loc, end_loc)
        if cache_key in cache:
            return cache[cache_key]

        # If either location is not in the graph built from road facts, they are unreachable
        # via the road network (unless they are the same location, handled above).
        if start_loc not in self.location_graph or end_loc not in self.location_graph:
             distance = 1000000 # Large penalty for unreachable
        else:
            # Perform BFS
            queue = deque([(start_loc, 0)])
            visited = {start_loc}
            distance = float('inf') # Use inf initially to detect unreachability

            while queue:
                current_loc, dist = queue.popleft()

                if current_loc == end_loc:
                    distance = dist
                    break

                # current_loc is guaranteed to be in self.location_graph keys here
                for neighbor in self.location_graph[current_loc]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, dist + 1))

            # If BFS completes without finding end_loc, distance remains inf.
            if distance == float('inf'):
                 distance = 1000000 # Large penalty for unreachable

        # Store result in cache (for both directions since graph is undirected)
        # Only cache if a path was found (distance is not the large penalty)
        if distance != 1000000:
            cache[cache_key] = distance
            cache[(end_loc, start_loc)] = distance # Assuming bidirectional roads

        return distance

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions
        to move all packages to their goal locations.
        """
        state = node.state  # Current world state.

        # Track locations of all locatable objects (packages on ground, vehicles)
        locatable_locations = {} # {object_name: location_name}
        # Track which packages are inside which vehicles
        package_in_vehicle = {} # {package_name: vehicle_name}

        # Collect current status and locations from the state
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == "at" and len(parts) == 3:
                 obj, loc = parts[1], parts[2]
                 locatable_locations[obj] = loc
            elif predicate == "in" and len(parts) == 3:
                 package, vehicle = parts[1], parts[2]
                 package_in_vehicle[package] = vehicle

        # Cache for distances within this heuristic call
        distance_cache = {}

        total_cost = 0

        # Iterate through packages that have a goal
        for package, goal_l in self.goal_locations.items():
            # Check if package is already at goal
            if f"(at {package} {goal_l})" in state:
                continue # Package is at goal, cost is 0 for this package

            # Determine the package's current physical location
            current_physical_location = None
            is_on_ground = True # Assume on ground unless found in a vehicle

            if package in package_in_vehicle:
                # Package is in a vehicle, find vehicle's location
                vehicle = package_in_vehicle[package]
                current_physical_location = locatable_locations.get(vehicle)
                is_on_ground = False
            else:
                # Package must be on the ground
                current_physical_location = locatable_locations.get(package)
                is_on_ground = True # Redundant, but explicit

            if current_physical_location is None:
                 # Package or its vehicle has no location in the state. Invalid state?
                 # print(f"Warning: Goal package {package} or its vehicle has no location in state.")
                 total_cost += 1000000 # Large penalty
                 continue

            # Calculate the shortest distance to the goal location
            drive_cost = self.get_distance(current_physical_location, goal_l, distance_cache)

            # Estimate actions needed
            if is_on_ground:
                # Needs pick-up, drive, drop
                total_cost += 1 + drive_cost + 1
            else:
                # Needs drive, drop (already picked up)
                total_cost += drive_cost + 1

        return total_cost
