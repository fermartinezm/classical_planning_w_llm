from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential leading/trailing whitespace or multiple spaces
    return fact.strip()[1:-1].split()

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

class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the required number of actions to move all packages
    to their goal locations. It sums the estimated cost for each package
    independently, ignoring vehicle capacity and availability constraints.
    The estimated cost for a package is the sum of:
    - 1 action for pick-up (if on the ground).
    - Shortest path distance (number of drive actions) for the vehicle carrying
      the package (or that will pick it up) from its current location to the
      package's goal location.
    - 1 action for drop-off.

    # Assumptions
    - The road network is static and defined by `(road l1 l2)` facts.
    - All locations relevant to package goals and initial positions are part of
      the road network.
    - Vehicle capacity is ignored.
    - Vehicle availability and location are considered only for the package
      currently being evaluated; no coordination or shared trips are assumed.
    - Packages needing to reach a goal location are initially either at a
      location or inside a vehicle.
    - Roads are bidirectional if the static facts contain pairs like `(road l1 l2)` and `(road l2 l1)`. The heuristic assumes bidirectionality if either direction is specified in the static facts.

    # Heuristic Initialization
    - Identify all locations from static facts and initial/goal states.
    - Build the road network graph based on `(road l1 l2)` static facts, assuming bidirectionality.
    - Precompute shortest path distances between all pairs of locations using BFS.
    - Identify packages and vehicles by inspecting predicates in the initial
      state and goals (`at`, `in`, `capacity`).
    - Store the goal location for each package that needs to be moved.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Parse the state to determine the current location of each package
       (either a location or inside a vehicle) and the current location of
       each vehicle.
    2. Initialize the total heuristic cost to 0.
    3. For each package that has a goal location defined and is not yet at its goal location:
       a. Determine the package's current status: on the ground at a location,
          or inside a vehicle.
       b. If the package is on the ground at `current_l`:
          - Estimate the cost as 1 (pick-up) + shortest_distance(`current_l`, `goal_l`)
            (drive) + 1 (drop). Add this to the total cost.
       c. If the package is inside a vehicle `v`, and the vehicle is at
          `vehicle_l`:
          - Estimate the cost as shortest_distance(`vehicle_l`, `goal_l`) (drive)
            + 1 (drop). Add this to the total cost.
       d. If the package's current location (or its vehicle's location) cannot
          reach the goal location in the road network, the heuristic for this
          package is infinite, and thus the total heuristic is infinite.
       e. If the package's current location or vehicle is not found in the parsed state,
          treat the goal as unreachable and return infinity.
    4. The total heuristic value is the sum of the estimated costs for all
       packages not at their goal. If the state is a goal state, this sum is 0.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information and goal facts.
        """
        self.goals = task.goals

        # 1. Identify object types and locations
        self.packages = set()
        self.vehicles = set()
        self.locations = set()
        self.road_graph = {}

        # Collect vehicles from capacity and in predicates
        for fact in task.initial_state | task.goals:
             if match(fact, 'capacity', '*', '*'):
                 self.vehicles.add(get_parts(fact)[1])
             if match(fact, 'in', '*', '*'):
                 self.vehicles.add(get_parts(fact)[2])
                 self.packages.add(get_parts(fact)[1]) # Also collect packages from 'in'

        # Collect locations and build initial road graph structure
        all_relevant_facts = task.initial_state | task.goals | task.static
        for fact in all_relevant_facts:
            parts = get_parts(fact)
            if parts[0] == 'at':
                # parts[1] is locatable, parts[2] is location
                self.locations.add(parts[2])
            elif parts[0] == 'road':
                # parts[1] and parts[2] are locations
                l1, l2 = parts[1:3]
                self.locations.add(l1)
                self.locations.add(l2)
                # Assume bidirectionality based on common domain patterns
                self.road_graph.setdefault(l1, []).append(l2)
                self.road_graph.setdefault(l2, []).append(l1)

        # Ensure all identified locations are in the road graph dictionary keys
        for loc in self.locations:
             self.road_graph.setdefault(loc, [])

        # Add any objects in 'at' facts that are not vehicles to packages
        for fact in task.initial_state | task.goals:
            if match(fact, 'at', '*', '*'):
                obj = get_parts(fact)[1]
                if obj not in self.vehicles:
                    self.packages.add(obj)

        # 2. Precompute shortest path distances between all pairs of locations
        self.distances = {}
        for start_loc in self.locations:
            q = deque([start_loc])
            dist = {loc: float('inf') for loc in self.locations}
            dist[start_loc] = 0

            while q:
                curr = q.popleft()
                current_dist = dist[curr]

                # Ensure curr is a valid key in road_graph
                if curr in self.road_graph:
                    for neighbor in self.road_graph[curr]:
                        if dist[neighbor] == float('inf'):
                            dist[neighbor] = current_dist + 1
                            q.append(neighbor)

            # Store distances from start_loc to all reachable locations
            for end_loc, d in dist.items():
                 if d != float('inf'):
                    self.distances[(start_loc, end_loc)] = d

        # 3. Store goal locations for packages
        self.goal_locations = {}
        for goal in self.goals:
            if match(goal, "at", "*", "*"):
                package, location = get_parts(goal)[1:3]
                # Only consider packages we identified and that have a goal location
                if package in self.packages:
                    self.goal_locations[package] = location


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Check if goal is reached
        if self.goals <= state:
            return 0

        # Parse current state to find locations of packages and vehicles
        package_locations = {} # Map package -> location or vehicle
        vehicle_locations = {} # Map vehicle -> location

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at':
                obj, loc = parts[1:3]
                if obj in self.packages:
                    package_locations[obj] = loc
                elif obj in self.vehicles:
                    vehicle_locations[obj] = loc
            elif parts[0] == 'in':
                p, v = parts[1:3]
                if p in self.packages and v in self.vehicles:
                     package_locations[p] = v

        total_cost = 0

        # Estimate cost for each package not at its goal
        for package, goal_location in self.goal_locations.items():
            # If package is already at goal, cost is 0 for this package
            if (f"(at {package} {goal_location})") in state:
                 continue

            current_loc_or_vehicle = package_locations.get(package)

            # If package is not found in the state, it's an issue or unreachable goal
            if current_loc_or_vehicle is None:
                 # This package is not accounted for. Assume unreachable or problem definition issue.
                 # Return infinity as heuristic indicates unsolvable from here.
                 return float('inf')

            # Case 1: Package is inside a vehicle
            if current_loc_or_vehicle in self.vehicles:
                vehicle = current_loc_or_vehicle
                vehicle_loc = vehicle_locations.get(vehicle)

                # If vehicle location is unknown, cannot estimate path
                if vehicle_loc is None:
                    return float('inf')

                # Cost = Drive vehicle from its current location to package goal + Drop package
                dist = self.distances.get((vehicle_loc, goal_location), float('inf'))

                if dist == float('inf'):
                    # Goal location unreachable from vehicle's current location
                    return float('inf')

                total_cost += dist + 1 # drive + drop

            # Case 2: Package is on the ground at a location
            elif current_loc_or_vehicle in self.locations:
                current_l = current_loc_or_vehicle

                # Cost = Pick up package + Drive from package location to goal + Drop package
                dist = self.distances.get((current_l, goal_location), float('inf'))

                if dist == float('inf'):
                    # Goal location unreachable from package's current location
                    return float('inf')

                total_cost += 1 + dist + 1 # pick + drive + drop

            else:
                 # Object is neither a known vehicle nor a known location.
                 # This indicates an unexpected state fact structure.
                 return float('inf')


        return total_cost
