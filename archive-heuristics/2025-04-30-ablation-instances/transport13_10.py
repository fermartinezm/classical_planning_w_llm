from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    if not fact or fact[0] != '(' or fact[-1] != ')':
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
    This heuristic estimates the number of actions needed to move each package
    from its current location to its goal location, summing the costs for all
    packages. It considers pick-up, drop, and driving actions. Driving cost
    is estimated by the shortest path distance in the road network.

    # Assumptions
    - The goal is solely defined by the target locations for specific packages.
    - All packages mentioned in the goal must reach their specified location.
    - Vehicle capacity constraints are ignored.
    - Vehicle availability at pick-up locations is ignored.
    - Multiple packages can be transported by a single vehicle, but the heuristic
      calculates costs for each package independently and sums them. This might
      overestimate the cost but provides a measure of remaining work.
    - Roads are bidirectional.

    # Heuristic Initialization
    - Extracts the goal location for each package from the task's goal conditions.
    - Builds a graph representing the road network from static facts.
    - Precomputes shortest path distances between all pairs of locations in the
      road network using Breadth-First Search (BFS).

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic is computed as follows:

    1.  Identify the current status of every package that needs to reach a goal
        location (as defined in the task's goals). A package can be:
        -   At a specific location on the ground (`(at package location)`).
        -   Inside a vehicle (`(in package vehicle)`). If inside a vehicle,
            find the vehicle's current location (`(at vehicle location)`).
        -   Already at its goal location on the ground.

    2.  Initialize the total heuristic cost to 0.

    3.  For each package that is *not* yet at its goal location:
        a.  Determine the package's current physical location (either directly
            if on the ground, or the location of the vehicle it is in).
        b.  Determine the package's goal location.
        c.  Calculate the estimated cost to move this single package to its goal:
            -   If the package is on the ground at `current_loc` and needs to go
                to `goal_loc`: The cost is 1 (pick-up) + `distance(current_loc, goal_loc)`
                (drive actions) + 1 (drop).
            -   If the package is inside a vehicle located at `current_loc` and
                needs to go to `goal_loc`: The cost is `distance(current_loc, goal_loc)`
                (drive actions) + 1 (drop).
            -   The `distance(loc1, loc2)` is the precomputed shortest path
                distance in the road network graph. If no path exists, the
                distance is considered infinite, making the state highly undesirable.
        d.  Add this estimated package cost to the total heuristic cost.

    4.  The final heuristic value is the sum of the estimated costs for all
        packages not yet at their goal location. If any package's goal is
        unreachable, the total heuristic is infinite.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        # Build the road network graph
        self.road_graph = {}
        locations = set()
        for fact in self.static:
            if match(fact, "road", "*", "*"):
                _, l1, l2 = get_parts(fact)
                locations.add(l1)
                locations.add(l2)
                self.road_graph.setdefault(l1, set()).add(l2)
                self.road_graph.setdefault(l2, set()).add(l1) # Roads are bidirectional

        # Precompute all-pairs shortest paths
        self.distances = {}
        all_locations = list(locations) # Use a list for consistent iteration order if needed, set is fine too
        for start_loc in all_locations:
            self.distances[start_loc] = {}
            # Run BFS from start_loc
            queue = deque([(start_loc, 0)])
            visited = {start_loc}
            self.distances[start_loc][start_loc] = 0

            while queue:
                current_loc, dist = queue.popleft()

                # Check if current_loc is a valid key in the graph before accessing neighbors
                if current_loc in self.road_graph:
                    for neighbor in self.road_graph[current_loc]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            self.distances[start_loc][neighbor] = dist + 1
                            queue.append((neighbor, dist + 1))

        # Store goal locations for each package
        self.goal_locations = {}
        for goal in self.goals:
            if match(goal, "at", "*", "*"):
                _, package, location = get_parts(goal)
                self.goal_locations[package] = location

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Track where packages and vehicles are currently located or contained
        locatables_at_locations = {} # Maps obj -> location (for packages on ground or vehicles)
        packages_in_vehicles = {}    # Maps package -> vehicle

        for fact in state:
            if match(fact, "at", "*", "*"):
                _, obj, loc = get_parts(fact)
                locatables_at_locations[obj] = loc
            elif match(fact, "in", "*", "*"):
                _, package, vehicle = get_parts(fact)
                packages_in_vehicles[package] = vehicle

        total_cost = 0

        # Iterate through packages that need to reach a goal location
        for package, goal_loc in self.goal_locations.items():
            package_cost = 0

            # Find the current status of the package
            current_loc = None
            current_vehicle = None

            if package in locatables_at_locations:
                current_loc = locatables_at_locations[package]
            elif package in packages_in_vehicles:
                current_vehicle = packages_in_vehicles[package]
                # Find the location of the vehicle
                if current_vehicle in locatables_at_locations:
                    current_loc = locatables_at_locations[current_vehicle]
                else:
                    # Vehicle location is unknown - implies an invalid state or unreachable goal
                    total_cost += float('inf')
                    continue # Cannot estimate cost for this package

            # If package is already at its goal location on the ground, cost is 0 for this package
            if current_loc == goal_loc and current_vehicle is None:
                 continue

            # If package is not at goal, calculate cost
            if current_loc is not None:
                # Get the shortest distance from the current location to the goal location
                # Use .get() with default float('inf') to handle cases where goal_loc
                # is not reachable from current_loc or locations are not in the graph.
                dist = self.distances.get(current_loc, {}).get(goal_loc, float('inf'))

                if dist == float('inf'):
                    total_cost += float('inf') # Goal is unreachable for this package
                    continue

                if current_vehicle is None: # Package is on the ground
                    # Cost: pick-up (1) + drive (dist) + drop (1)
                    package_cost = 1 + dist + 1
                else: # Package is inside a vehicle
                    # Cost: drive (dist) + drop (1)
                    package_cost = dist + 1
            else:
                 # Package is not at a location and not in a known vehicle at a location.
                 # This shouldn't happen in a valid state representation where all locatables
                 # are either at a location or in a container that is at a location.
                 # Treat as unreachable.
                 total_cost += float('inf')
                 continue

            total_cost += package_cost

        return total_cost
