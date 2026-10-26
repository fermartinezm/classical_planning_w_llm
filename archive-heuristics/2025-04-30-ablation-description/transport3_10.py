# Ensure the Heuristic base class is available or define a dummy one
# Assuming heuristics.heuristic_base exists and provides a Heuristic base class
from heuristics.heuristic_base import Heuristic

from collections import deque
from fnmatch import fnmatch

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and starts/ends with parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
         # Handle potential errors or unexpected fact formats gracefully
         return [] # Return empty list for malformed facts
    return fact[1:-1].split()


# Note: The 'match' function is not strictly needed for this specific heuristic
# implementation but is kept as a useful utility function for PDDL fact parsing.
# def match(fact, *args):
#     """
#     Check if a PDDL fact matches a given pattern.
#
#     - `fact`: The complete fact as a string, e.g., "(at package1 location1)".
#     - `args`: The expected pattern (wildcards `*` allowed).
#     - Returns `True` if the fact matches the pattern, else `False`.
#     """
#     parts = get_parts(fact)
#     # Check if the number of parts matches the number of arguments in the pattern
#     if len(parts) != len(args):
#         return False
#     return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing, for each package not at its goal location,
    the shortest path distance from its current location (or its vehicle's location) to its goal location,
    plus a fixed cost for necessary pick-up and drop actions.

    # Assumptions
    - The road network is static and bidirectional.
    - Shortest path distances represent the minimum number of drive actions.
    - Capacity constraints are ignored (any vehicle can pick up any package if at the same location).
    - Vehicle availability is ignored (a vehicle is assumed to be available when needed).
    - Each pick-up and drop action costs 1.
    - The goal for a package is always to be on the ground at a specific location.

    # Heuristic Initialization
    - Extracts goal locations for each package from the task goals.
    - Builds a graph representing the road network from static facts.
    - Computes all-pairs shortest path distances between locations using BFS.
    - Identifies vehicles based on capacity facts in the initial state or static facts.
    - Identifies package objects based on their appearance in the goals.

    # Step-By-Step Thinking for Computing Heuristic
    For each package that is not yet at its goal location:
    1. Determine the package's current status: Is it on the ground at some location, or is it inside a vehicle?
    2. If the package is on the ground at `l_current`:
       - Estimate the cost as the shortest path distance from `l_current` to the package's goal location `l_goal`, plus 2 actions (1 for pick-up, 1 for drop).
       - This assumes a vehicle can reach `l_current`, pick up the package, drive to `l_goal`, and drop it. The cost of getting a vehicle to `l_current` is ignored.
    3. If the package is inside a vehicle `v`, and the vehicle is at `l_vehicle`:
       - If `l_vehicle` is the same as the package's goal location `l_goal`:
         - Estimate the cost as 1 action (for dropping the package).
       - If `l_vehicle` is different from `l_goal`:
         - Estimate the cost as the shortest path distance from `l_vehicle` to `l_goal`, plus 1 action (for dropping the package).
         - This assumes the vehicle can drive directly to `l_goal` and drop the package.
    4. The total heuristic value is the sum of these estimated costs for all packages not at their goal.
    5. If all packages are at their goal locations, the heuristic is 0. If any required location is unreachable via roads, the heuristic is infinity.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, building the
        road network graph, computing all-pairs shortest paths, and identifying vehicles.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.
        initial_state = task.initial_state # Initial state facts

        # 1. Extract goal locations for each package and identify package objects.
        self.goal_locations = {}
        self.package_objects = set()
        for goal in self.goals:
            # Goal is typically (at package location)
            predicate, *args = get_parts(goal)
            if predicate == "at" and len(args) == 2:
                package, location = args
                self.goal_locations[package] = location
                self.package_objects.add(package)
            # Add handling for other potential goal types if necessary,
            # but (at ?p ?l) is the primary goal type in transport.

        # 2. Build the road network graph.
        self.locations = set()
        self.roads = {} # Adjacency list: {location1: {location2, location3}, ...}
        for fact in static_facts:
            predicate, *args = get_parts(fact)
            if predicate == "road" and len(args) == 2:
                l1, l2 = args
                self.locations.add(l1)
                self.locations.add(l2)
                self.roads.setdefault(l1, set()).add(l2)
                self.roads.setdefault(l2, set()).add(l1) # Assuming roads are bidirectional

        # 3. Compute all-pairs shortest path distances using BFS.
        self.dist = {} # Store distances as {(start_loc, end_loc): distance}

        for start_loc in self.locations:
            q = deque([(start_loc, 0)])
            visited = {start_loc}
            self.dist[(start_loc, start_loc)] = 0 # Distance to self is 0

            while q:
                curr_loc, d = q.popleft()

                # Explore neighbors
                for neighbor in self.roads.get(curr_loc, set()):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        self.dist[(start_loc, neighbor)] = d + 1
                        q.append((neighbor, d + 1))

        # 4. Identify vehicles from initial state or static facts (capacity)
        self.vehicles = set()
        for fact in initial_state | static_facts:
             parts = get_parts(fact)
             if not parts: continue
             if parts[0] == "capacity" and len(parts) >= 2: # capacity fact has at least 2 parts: predicate and vehicle name
                 self.vehicles.add(parts[1])


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions
        to move all packages to their goal locations.
        """
        state = node.state  # Current world state.

        # Map current locations/statuses of packages and vehicles.
        pkg_locations = {} # {package: location or vehicle}
        vehicle_locations = {} # {vehicle: location}

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]
            if predicate == "at" and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                if obj in self.vehicles:
                    vehicle_locations[obj] = loc
                elif obj in self.package_objects:
                    pkg_locations[obj] = loc
                # else: it's some other locatable object we don't care about for this heuristic

            elif predicate == "in" and len(parts) == 3:
                 p, v = parts[1], parts[2]
                 if p in self.package_objects and v in self.vehicles:
                     pkg_locations[p] = v # Package p is inside vehicle v


        total_cost = 0  # Initialize action cost counter.

        # Iterate through each package and its goal location.
        for package, goal_location in self.goal_locations.items():
            # Check if the package is already at its goal location on the ground.
            # We check the exact fact string for efficiency and correctness based on state representation.
            if f"(at {package} {goal_location})" in state:
                 continue # Goal for this package is satisfied.

            # Package is not at its goal location on the ground.
            current_status = pkg_locations.get(package)

            if current_status is None:
                 # Package status is unknown - this shouldn't happen in a valid state
                 # but handle defensively. Assume unreachable goal.
                 return float('inf')

            if current_status in self.locations: # Package is on the ground at current_status
                l_current_p = current_status
                # Cost: drive from current_loc to goal_loc + pick-up + drop
                dist_to_goal = self.dist.get((l_current_p, goal_location), float('inf'))
                if dist_to_goal == float('inf'):
                    return float('inf') # Goal is unreachable
                total_cost += dist_to_goal + 2 # 1 for pick-up, 1 for drop

            elif current_status in self.vehicles: # Package is inside vehicle current_status
                v = current_status
                l_vehicle = vehicle_locations.get(v)

                if l_vehicle is None:
                    # Vehicle location unknown - shouldn't happen
                    return float('inf')

                if l_vehicle == goal_location:
                    # Vehicle is at the goal location, package needs to be dropped.
                    total_cost += 1 # 1 for drop
                else:
                    # Vehicle needs to drive to goal_location, then package needs to be dropped.
                    dist_to_goal = self.dist.get((l_vehicle, goal_location), float('inf'))
                    if dist_to_goal == float('inf'):
                        return float('inf') # Goal is unreachable
                    total_cost += dist_to_goal + 1 # 1 for drop

            else:
                 # current_status is neither a location nor a vehicle - shouldn't happen
                 return float('inf')


        return total_cost
