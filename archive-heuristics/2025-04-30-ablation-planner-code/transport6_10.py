from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact string or malformed fact
    if not fact or not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
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

class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions required to move all packages
    to their goal locations. It calculates the minimum number of pick-up, drive,
    and drop actions needed for each package independently, based on shortest
    path distances in the road network.

    # Assumptions
    - Each package needs to reach a specific goal location.
    - The cost of moving a package from location A to location B is estimated
      by the shortest path distance between A and B in the road network, plus
      actions for picking up and dropping the package.
    - Vehicle availability and capacity constraints are *not* explicitly modeled
      in the heuristic calculation. It assumes that a suitable vehicle will
      eventually be available for each package's transport needs.
    - The cost of a drive action is 1. The cost of pick-up and drop actions is 1.
    - The `task.goals` attribute is an iterable where each element is a string
      representing a single goal fact (e.g., `'(at p1 l2)'`). If the goal is
      a conjunction `(and ...)`, it is assumed to be already broken down into
      individual facts.
    - The `task.init` attribute is an iterable of initial fact strings.

    # Heuristic Initialization
    - Extracts the goal location for each package from the task goals.
    - Builds the road network graph from the static facts.
    - Collects all relevant locations from road facts, initial state, and goals.
    - Computes all-pairs shortest path distances between all collected locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Initialize total heuristic cost to 0.
    2. Determine the current location and status (on ground or in vehicle) for all
       locatable objects (packages and vehicles) by parsing `(at ?x ?l)` and
       `(in ?p ?v)` facts in the current state. Store this in a map `current_status_map`.
    3. For each package `p` that has a goal location `goal_l` (extracted during initialization):
       a. Get the package's current status from `current_status_map`. This is either
          its location string (if on the ground) or the vehicle name string (if in a vehicle).
       b. If the package is in a vehicle, look up the vehicle's location from `current_status_map`
          to find the package's effective location `current_l`. If the package is on the ground,
          its status *is* its effective location `current_l`. Determine `is_in_vehicle` status.
       c. If `current_l` or `goal_l` are not valid locations found during initialization,
          or if the vehicle containing a package is not at a valid location, the state
          is likely unsolvable by driving. Return `float('inf')`.
       d. Get the shortest path distance `dist` between `current_l` and `goal_l` from the
          precomputed distances. If `dist` is `float('inf')`, the locations are unreachable.
          Return `float('inf')`.
       e. Calculate the minimum actions needed for this package:
          - If `current_l == goal_l` and not `is_in_vehicle`: Cost is 0 (package is at goal).
          - If `current_l == goal_l` and `is_in_vehicle`: Cost is 1 (needs drop).
          - If `current_l != goal_l` and not `is_in_vehicle`: Cost is `dist + 2` (pick, drive, drop).
          - If `current_l != goal_l` and `is_in_vehicle`: Cost is `dist + 1` (drive, drop).
       f. Add the calculated cost for this package to the total cost.
    4. The total heuristic value is the sum of costs for all packages.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and building the road network."""
        self.goals = task.goals
        static_facts = task.static
        # Assuming task object provides initial facts as task.init
        self.initial_facts = task.init if hasattr(task, 'init') else frozenset()


        # 1. Extract goal locations for each package.
        self.goal_locations = {}
        # Assuming task.goals is an iterable of goal fact strings (e.g., from a parsed PDDL goal)
        for goal_fact_string in self.goals:
            predicate, *args = get_parts(goal_fact_string)
            if predicate == "at":
                # We assume goals are of the form (at package location)
                # Check if args has enough elements to avoid index error
                if len(args) >= 2:
                    package, location = args[:2] # Take first two args just in case
                    self.goal_locations[package] = location
            # Ignore other potential goal types if any

        # 2. Build the road network graph and collect all locations.
        self.road_graph = {} # Adjacency list: location -> [neighbor1, neighbor2, ...]
        all_locations = set()

        for fact in static_facts:
            predicate, *args = get_parts(fact)
            if predicate == "road":
                # Check if args has enough elements
                if len(args) >= 2:
                    l1, l2 = args[:2]
                    self.road_graph.setdefault(l1, []).append(l2)
                    self.road_graph.setdefault(l2, []).append(l1) # Roads are bidirectional
                    all_locations.add(l1)
                    all_locations.add(l2)

        # Add locations mentioned in the initial state and goals to ensure they are included
        # in the distance calculation, even if they are isolated (no roads).
        for fact in self.initial_facts:
             predicate, *args = get_parts(fact)
             if predicate == "at":
                 # (at ?x ?l) where ?x is locatable, ?l is location
                 # Check if args has enough elements
                 if len(args) >= 2:
                    loc = args[1]
                    all_locations.add(loc)

        all_locations.update(self.goal_locations.values())

        self.locations = sorted(list(all_locations)) # Keep a sorted list of all known locations

        # 3. Compute all-pairs shortest path distances.
        self.distances = {}
        for start_loc in self.locations:
            self.distances[start_loc] = self._bfs(start_loc)

    def _bfs(self, start_location):
        """Perform BFS from a start location to find distances to all reachable locations."""
        distances = {loc: float('inf') for loc in self.locations}

        # Check if start_location is a known location
        if start_location not in self.locations:
             # This case should ideally not happen if self.locations includes all locations
             # from init, goals, and road facts. But as a safeguard:
             return distances # All distances remain infinity

        distances[start_location] = 0
        queue = deque([start_location])

        while queue:
            current_loc = queue.popleft()

            # Only process neighbors if the current location is part of the road network graph
            if current_loc in self.road_graph:
                for neighbor in self.road_graph[current_loc]:
                    # Ensure neighbor is also a known location before updating distance
                    if neighbor in self.locations and distances[neighbor] == float('inf'):
                        distances[neighbor] = distances[current_loc] + 1
                        queue.append(neighbor)

        return distances


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state (frozenset of fact strings).

        # Track current location/status for all locatable objects (packages and vehicles)
        # This map stores obj -> location (if at ground) or obj -> vehicle (if in vehicle)
        current_status_map = {}

        for fact in state:
            predicate, *args = get_parts(fact)
            if predicate == "at":
                # Check if args has enough elements
                if len(args) >= 2:
                    obj, loc = args[:2]
                    current_status_map[obj] = loc
            elif predicate == "in":
                 # Check if args has enough elements
                 if len(args) >= 2:
                    package, vehicle = args[:2]
                    current_status_map[package] = vehicle # Package is inside this vehicle

        total_cost = 0

        for package, goal_location in self.goal_locations.items():
            # Find the package's current effective location and status
            current_status = current_status_map.get(package)

            if current_status is None:
                 # Package with a goal is not found in the state. This might indicate
                 # an invalid state or an issue with state representation.
                 # Assuming valid states where all goal packages are present.
                 # Treat as unsolvable.
                 return float('inf')

            # Determine if the package is in a vehicle and its effective location
            is_in_vehicle = current_status not in self.locations # If status is not a known location, it's a vehicle name
            current_location = None

            if is_in_vehicle:
                vehicle_name = current_status
                # Find the location of this vehicle
                current_location = current_status_map.get(vehicle_name)
                # Check if vehicle location is valid
                if current_location is None or current_location not in self.locations:
                     # Vehicle containing the package is not at a known location.
                     # Invalid state or unsolvable.
                     return float('inf')
            else:
                current_location = current_status # Package is on the ground at this location
                # Check if package location is valid
                if current_location not in self.locations:
                     # Package is at an unknown location. Invalid state or unsolvable.
                     return float('inf')


            # Check if goal_location is a known location
            if goal_location not in self.locations:
                 # Goal location is not in the set of locations found during init.
                 # This is unexpected if init/goals/road facts were parsed correctly.
                 # Treat as unsolvable.
                 return float('inf')

            # Get the shortest distance between current effective location and goal location
            # Use .get() with a default inf value for robustness, although check above should cover this.
            dist = self.distances.get(current_location, {}).get(goal_location, float('inf'))

            if dist == float('inf'):
                 # Cannot reach goal location from current location via road network.
                 # This package cannot reach its goal by driving.
                 return float('inf')

            # Calculate cost based on package status and location relative to goal
            if current_location == goal_location:
                if is_in_vehicle:
                    # Package is at goal location but inside a vehicle. Needs drop.
                    total_cost += 1
                else:
                    # Package is at goal location on the ground. Goal achieved for this package.
                    pass # Cost is 0
            else: # current_location != goal_location
                if is_in_vehicle:
                    # Package is in a vehicle, not at goal location. Needs drive and drop.
                    total_cost += dist + 1 # drive + drop
                else:
                    # Package is on the ground, not at goal location. Needs pick-up, drive, and drop.
                    total_cost += 1 + dist + 1 # pick + drive + drop

        # The heuristic should be 0 only for goal states.
        # The calculated total_cost is 0 if and only if for every package with a goal,
        # its current effective location is its goal location AND it is not in a vehicle.
        # This exactly matches the structure of the typical goal condition in this domain
        # (a conjunction of (at p l) facts for all goal packages).
        # So, total_cost == 0 implies the state is a goal state.

        return total_cost
