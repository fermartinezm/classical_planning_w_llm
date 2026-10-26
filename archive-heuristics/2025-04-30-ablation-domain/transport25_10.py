from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic

# Utility functions to parse PDDL facts
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
    # Ensure the number of parts matches the number of arguments in the pattern
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions required to move each package
    to its goal location. It calculates this by summing the estimated minimum
    actions for each package independently, ignoring vehicle capacity and
    availability constraints. The estimated actions for a single package
    include loading, driving, and unloading steps, where driving cost is
    estimated by the shortest path distance in the road network.

    # Assumptions
    - The road network is static and provides connections between locations.
    - Any package can be transported by any vehicle (capacity is ignored).
    - Vehicles are always available when needed (their location and state are
      only considered to determine the package's effective current location).
    - The cost of loading, unloading, and driving one step is 1.
    - The heuristic sums the costs for each package independently (relaxation).
    - Goal conditions only involve packages being at specific locations on the ground.

    # Heuristic Initialization
    - Extracts the goal location for each package from the task goals.
    - Builds a graph representation of the road network from static facts.
    - Computes all-pairs shortest paths between all locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state `s`:
    1. Initialize total heuristic cost `h(s) = 0`.
    2. For each package `p` that has a goal location `loc_goal(p)`:
       a. Determine the current effective location of package `p`.
          - If `p` is on the ground at `loc` (fact `(at p loc)` in `s`), its effective location is `loc`.
          - If `p` is inside a vehicle `v` (fact `(in p v)` in `s`), find the location of vehicle `v` (fact `(at v loc_v)` in `s`). The package's effective location is `loc_v`.
          - If the package's location cannot be determined (e.g., not 'at' or 'in'), assume unreachable (infinity cost).
       b. If the package's effective current location `loc_p_current` is the same as its goal location `loc_goal(p)`:
          - If the package is on the ground at the goal, the cost for this package is 0.
          - If the package is inside a vehicle at the goal, it needs to be unloaded. Cost for this package is 1.
       c. If the package's effective current location `loc_p_current` is different from its goal location `loc_goal(p)`:
          - Calculate the shortest path distance `dist` between `loc_p_current` and `loc_goal(p)` using the precomputed distances. If unreachable, the cost is infinity.
          - If the package is on the ground at `loc_p_current`, it needs to be loaded, the vehicle needs to drive, and it needs to be unloaded. Estimated cost: 1 (load) + `dist` (drive) + 1 (unload) = `dist + 2`.
          - If the package is inside a vehicle at `loc_p_current`, the vehicle needs to drive, and it needs to be unloaded. Estimated cost: `dist` (drive) + 1 (unload) = `dist + 1`.
       d. Add the estimated cost for package `p` to the total heuristic cost `h(s)`.
    3. Return the total heuristic cost `h(s)`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal locations and building
        the road network graph to compute shortest paths.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # Store goal locations for each package.
        self.goal_locations = {}
        for goal in self.goals:
            # Goals are typically (at package location)
            if match(goal, "at", "*", "*"):
                package, location = get_parts(goal)[1:]
                self.goal_locations[package] = location
            # Ignore other potential goal types for this heuristic

        # Build the road network graph and compute shortest paths.
        self.road_graph = {}
        locations = set()

        for fact in static_facts:
            if match(fact, "road", "*", "*"):
                loc1, loc2 = get_parts(fact)[1:]
                locations.add(loc1)
                locations.add(loc2)
                if loc1 not in self.road_graph:
                    self.road_graph[loc1] = set()
                self.road_graph[loc1].add(loc2)
                # Assuming roads are bidirectional if (road l1 l2) and (road l2 l1) exist
                # The example shows bidirectional roads, so we build an undirected graph
                if loc2 not in self.road_graph:
                    self.road_graph[loc2] = set()
                self.road_graph[loc2].add(loc1)

        # Compute all-pairs shortest paths using BFS
        self.distances = {}
        all_locations = list(locations) # Get a list of all unique locations

        for start_node in all_locations:
            self.distances[start_node] = {}
            queue = deque([(start_node, 0)])
            visited = {start_node}

            while queue:
                current_loc, dist = queue.popleft()
                self.distances[start_node][current_loc] = dist

                if current_loc in self.road_graph:
                    for neighbor in self.road_graph[current_loc]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append((neighbor, dist + 1))

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        total_cost = 0  # Initialize action cost counter.

        # Create a quick lookup for package/vehicle locations in the current state
        # This avoids iterating through the state multiple times per package
        at_facts = {} # Maps object (package or vehicle) to its location if on ground
        in_facts = {} # Maps package to vehicle if inside

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at":
                obj, loc = parts[1:]
                at_facts[obj] = loc
            elif parts[0] == "in":
                package, vehicle = parts[1:]
                in_facts[package] = vehicle

        for package, goal_location in self.goal_locations.items():
            current_location = None
            is_in_vehicle = False

            if package in at_facts:
                current_location = at_facts[package]
                is_in_vehicle = False
            elif package in in_facts:
                vehicle = in_facts[package]
                if vehicle in at_facts:
                    current_location = at_facts[vehicle]
                    is_in_vehicle = True
                else:
                    # Vehicle containing the package is not located? Invalid state?
                    # Treat as unreachable for safety.
                    return float('inf')
            else:
                 # Package is not 'at' a location and not 'in' a vehicle. Invalid state?
                 # Treat as unreachable.
                 return float('inf')

            # If package is already at the goal location on the ground, cost is 0.
            if current_location == goal_location and not is_in_vehicle:
                 continue # Cost is 0 for this package

            # Calculate cost for this package
            h_p = 0

            if current_location == goal_location and is_in_vehicle:
                # Package is at the goal location but inside a vehicle, needs unload
                h_p = 1
            elif current_location != goal_location:
                # Package is not at the goal location
                if current_location not in self.distances or goal_location not in self.distances[current_location]:
                    # Goal is unreachable from current location
                    return float('inf')

                dist = self.distances[current_location][goal_location]

                if is_in_vehicle:
                    # Package is in a vehicle, needs drive and unload
                    h_p = dist + 1
                else:
                    # Package is on the ground, needs load, drive, and unload
                    h_p = 1 + dist + 1 # load + drive + unload
            # else: current_location == goal_location and not is_in_vehicle, h_p is already 0

            total_cost += h_p

        # The heuristic is 0 if and only if all packages are at their goal locations
        # and on the ground, assuming the goal only contains (at p loc) facts.
        # If total_cost is 0, it implies all packages in self.goal_locations
        # satisfy the (at p goal_loc) condition.
        # This matches the requirement that h=0 only at goal states for this domain.

        return total_cost
