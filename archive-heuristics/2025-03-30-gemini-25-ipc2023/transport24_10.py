import collections
import itertools
from fnmatch import fnmatch
# Assuming the existence of a base class like this:
# from heuristics.heuristic_base import Heuristic
# If the base class is not available in the environment, 
# define a dummy base class or remove the inheritance.
class Heuristic: # Dummy base class if not provided by the environment
    def __init__(self, task): pass
    def __call__(self, node): raise NotImplementedError

def get_parts(fact):
    """
    Extracts predicate and arguments from a PDDL fact string.
    Removes parentheses and splits the string by spaces.
    Example: "(at p1 l1)" -> ["at", "p1", "l1"]
    Handles potential extra spaces within the fact.
    """
    # Remove leading/trailing whitespace and the parentheses
    content = fact.strip()[1:-1]
    # Split by space and filter out empty strings resulting from multiple spaces
    return [part for part in content.split(' ') if part]

class TransportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the PDDL domain 'transport'.

    # Summary
    This heuristic estimates the remaining cost to reach the goal state by summing
    the estimated costs for each package that is not yet at its final destination.
    The cost for a package includes the actions needed to pick it up (if not already
    in a vehicle), drive it to the destination, and drop it off. Driving costs are
    estimated using the shortest path distance in the road network graph. The
    heuristic aims for informativeness to guide a greedy best-first search and is
    not necessarily admissible.

    # Assumptions
    - The primary goal consists of facts of the form `(at package location)`. Other
      goal types are ignored by this heuristic.
    - The cost of driving between two locations is the number of 'drive' actions
      along the shortest path. Each 'drive', 'pick-up', and 'drop' action has a cost of 1.
    - The heuristic approximates the total cost by summing individual package costs.
      This might overcount vehicle movements if one vehicle handles multiple packages
      efficiently, but serves as a reasonable estimate for guiding the search.
    - Vehicle capacity constraints (`capacity`, `capacity-predecessor`) are ignored
      during the heuristic calculation for simplicity and efficiency.
    - If a package's goal location is unreachable from its current location (or the
      location of the vehicle carrying it) via the road network, the heuristic
      returns infinity, indicating a potentially unsolvable state from that point.

    # Heuristic Initialization
    - Parses the goal conditions (`task.goals`) to identify the target location
      for each package involved in the goal. Stores this mapping in `self.goal_locations`.
    - Parses static facts (`task.static`), specifically the 'road' predicates,
      to build an adjacency list representation (`self.adj`) of the location graph.
      It also collects all unique location names found in goals and roads into `self.locations`.
    - Computes all-pairs shortest path distances between all known locations using
      Breadth-First Search (BFS) starting from each location. Stores these distances
      in `self.distances[start_loc][end_loc]`. Unreachable pairs have a value of infinity.

    # Step-By-Step Thinking for Computing Heuristic
    1.  Initialize the total heuristic estimate `h` to 0.
    2.  Parse the current `state` (a set of fact strings) to determine:
        - `current_pkg_loc`: A dictionary mapping packages currently `(at p loc)`
          to their location {package: location}.
        - `package_in_vehicle`: A dictionary mapping packages currently `(in p v)`
          to the vehicle they are in {package: vehicle}.
        - `vehicle_loc`: A dictionary mapping vehicles `(at v loc)` to their current
          location {vehicle: location}.
    3.  Iterate through each package `p` and its corresponding `goal_loc` stored in
        `self.goal_locations`.
    4.  For each package `p`:
        a.  Determine if the package is currently in a vehicle (`is_in_vehicle`).
        b.  Find the package's current effective location `current_loc`.
            - If `is_in_vehicle`, `current_loc` is the location of the vehicle carrying it.
              Look up the vehicle's location in `vehicle_loc`.
            - Otherwise (package is `at` some location), `current_loc` is the location
              where the package is `at`. Look up the package's location in `current_pkg_loc`.
            - If the location cannot be determined (e.g., inconsistent state where a
              package is 'in' a vehicle without a location, or a goal package is neither
              'at' nor 'in'), return infinity as the state seems invalid or unsolvable.
        c.  Check if the goal for this package `p` is already satisfied. The goal is
           `(at p goal_loc)`. This requires the package to be *at* the location,
           not *in* a vehicle at that location. If `not is_in_vehicle` and
           `current_loc == goal_loc`, the goal for `p` is met; add 0 to `h` for this
           package and continue to the next package.
        d.  If the package's goal is not met:
            i.  **Estimate Pickup Cost:** If the package is not currently in a vehicle
                (`not is_in_vehicle`), it will need a `pick-up` action. Add 1 to `h`.
            ii. **Estimate Drop Cost:** The package will eventually need a `drop` action
                at the `goal_loc` to satisfy the `(at p goal_loc)` condition. Add 1 to `h`.
            iii. **Estimate Drive Cost:** If the package's `current_loc` is different
                 from its `goal_loc`, calculate the shortest driving distance `d` using
                 the precomputed `self.distances[current_loc].get(goal_loc, float('inf'))`.
                 - If `d` is infinity, the goal is unreachable for this package; return infinity.
                 - Otherwise, add `d` (representing the number of `drive` actions) to `h`.
    5.  After iterating through all packages defined in the goals, return the total
        accumulated estimate `h`. If `h` is 0, it means all package goals are satisfied.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by parsing goals, building the road graph,
        and precomputing all-pairs shortest paths.
        """
        self.goals = task.goals
        static_facts = task.static
        self.locations = set()

        # 1. Parse goal locations and identify packages/locations involved in goals
        self.goal_locations = {}
        for goal in self.goals:
            try:
                parts = get_parts(goal)
                # Ensure goal is of the expected type (at package location)
                if parts[0] == "at" and len(parts) == 3:
                    # Basic check: assume second arg is package, third is location
                    package, location = parts[1], parts[2]
                    self.goal_locations[package] = location
                    self.locations.add(location) # Collect locations mentioned in goals
            except IndexError:
                # Handle potential malformed goal facts gracefully
                # print(f"Warning: Skipping malformed goal fact: {goal}")
                continue

        # 2. Build road graph (adjacency list) and collect all locations from roads
        self.adj = collections.defaultdict(set)
        for fact in static_facts:
            try:
                parts = get_parts(fact)
                if parts[0] == "road" and len(parts) == 3:
                    loc1, loc2 = parts[1], parts[2]
                    # Add locations from road facts
                    self.locations.add(loc1)
                    self.locations.add(loc2)
                    # Add edge to adjacency list (assuming roads are potentially one-way)
                    self.adj[loc1].add(loc2)
            except IndexError:
                 # Handle potential malformed static facts gracefully
                 # print(f"Warning: Skipping malformed static fact: {fact}")
                 continue

        # 3. Compute all-pairs shortest paths using BFS
        self.distances = collections.defaultdict(lambda: collections.defaultdict(lambda: float('inf')))

        if not self.locations:
            # Handle case with no locations (e.g., trivial problem)
            # print("Warning: No locations found. Heuristic may not function correctly.")
            return # Skip distance calculation if no locations exist

        for start_node in self.locations:
            # Set distance to self as 0
            self.distances[start_node][start_node] = 0

            # Perform BFS from start_node
            queue = collections.deque([(start_node, 0)]) # Store (node, distance)
            # Keep track of visited nodes and their shortest distance found so far *in this specific BFS run*
            visited_bfs = {start_node: 0}

            while queue:
                u, dist = queue.popleft()

                # Access neighbors using .get() for safety, in case u isn't in adj keys
                # (e.g., a location mentioned in goal but with no outgoing roads)
                for v in self.adj.get(u, set()):
                    # If v hasn't been visited in this BFS, or if we found a shorter path
                    # (though BFS naturally finds shortest paths first, this check is redundant here)
                    if v not in visited_bfs:
                        visited_bfs[v] = dist + 1
                        self.distances[start_node][v] = dist + 1
                        queue.append((v, dist + 1))


    def __call__(self, node):
        """
        Calculates the heuristic value for a given state node.
        Returns an estimate of the number of actions required to reach the goal.
        Returns float('inf') if the goal seems unreachable from the current state.
        """
        state = node.state
        h = 0

        # Parse current state efficiently to find locations of packages and vehicles
        current_pkg_loc = {} # package -> location (if 'at')
        package_in_vehicle = {} # package -> vehicle (if 'in')
        vehicle_loc = {} # vehicle -> location

        for fact in state:
            try:
                parts = get_parts(fact)
                predicate = parts[0]
                if predicate == "at" and len(parts) == 3:
                    obj, loc = parts[1], parts[2]
                    # Check if obj is a package with a goal location
                    if obj in self.goal_locations:
                        current_pkg_loc[obj] = loc
                    else:
                        # Assume it's a vehicle if not a package with a goal
                        # This might misclassify other 'locatable' objects if the domain
                        # were extended, but is standard for this transport domain.
                        vehicle_loc[obj] = loc
                elif predicate == "in" and len(parts) == 3:
                    package, vehicle = parts[1], parts[2]
                    # Ensure the package is one we care about (has a goal)
                    if package in self.goal_locations:
                        package_in_vehicle[package] = vehicle
            except IndexError:
                 # Handle potential malformed state facts gracefully
                 # print(f"Warning: Skipping malformed state fact: {fact}")
                 continue


        # Calculate heuristic value based on unsatisfied package goals
        for package, goal_loc in self.goal_locations.items():

            is_in_vehicle = package in package_in_vehicle
            current_loc = None # The effective location of the package for distance calculation

            if is_in_vehicle:
                vehicle = package_in_vehicle[package]
                if vehicle in vehicle_loc:
                    current_loc = vehicle_loc[vehicle]
                else:
                    # State inconsistency: package is 'in' a vehicle whose location is unknown.
                    # This implies the state is invalid or the problem is unsolvable.
                    # print(f"Heuristic Error: Location of vehicle {vehicle} carrying package {package} not found.")
                    return float('inf')
            elif package in current_pkg_loc:
                current_loc = current_pkg_loc[package]
            else:
                # State inconsistency: package with a goal is neither 'at' nor 'in'.
                # This could happen if the initial state doesn't define the location
                # of all goal packages, or if an action incorrectly removed location info.
                # print(f"Heuristic Error: Location of package {package} not found.")
                return float('inf')

            # Check if goal for this package is met: (at package goal_loc)
            # The package must be AT the location, not IN a vehicle AT the location.
            goal_met = (not is_in_vehicle) and (current_loc == goal_loc)

            if goal_met:
                continue # This package's goal is satisfied, move to the next.

            # --- Package goal is not satisfied ---

            # Estimate cost for pickup action (needed only if package is currently 'at' a location)
            if not is_in_vehicle:
                h += 1 # Add 1 cost for the pick-up action

            # Estimate cost for drop action (always needed to reach the 'at' goal state from any other state)
            h += 1 # Add 1 cost for the drop action

            # Estimate cost for driving action(s)
            if current_loc != goal_loc:
                # Look up precomputed shortest path distance
                # Use .get() on the outer dict too, in case current_loc wasn't in self.locations
                # (e.g., if a package starts at a location with no roads)
                distance = self.distances.get(current_loc, {}).get(goal_loc, float('inf'))

                if distance == float('inf'):
                    # Goal location is unreachable from current location via the road network.
                    # print(f"Heuristic Error: Goal location {goal_loc} unreachable for package {package} from {current_loc}.")
                    return float('inf') # Indicate unsolvable state

                h += distance # Add driving cost (number of drive actions)

        return h
