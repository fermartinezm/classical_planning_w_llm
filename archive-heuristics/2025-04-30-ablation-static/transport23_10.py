from fnmatch import fnmatch
from collections import deque
# Assuming Heuristic base class is available in heuristics.heuristic_base
from heuristics.heuristic_base import Heuristic

# Helper functions from Logistics example
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure we don't try to match more args than parts
    if len(args) > len(parts):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions required to move each package
    to its goal location. It sums the estimated costs for each package
    independently, ignoring vehicle capacity constraints and vehicle availability.
    The cost for a package is estimated based on its current location (on the
    ground or inside a vehicle) and the shortest path distance to its goal location.

    # Assumptions
    - Any package can be picked up by any vehicle at its location, regardless of capacity.
    - A suitable vehicle is always available at the required location for pick-up or drop-off.
    - The cost of driving is the shortest path distance between locations.
    - Each pick-up and drop action costs 1.
    - The heuristic is non-admissible and designed for greedy best-first search.
    - All locations mentioned in initial state, goals, and road facts are part of the connected road network (or reachable from it). Unreachable goals incur a large penalty.

    # Heuristic Initialization
    - Parses the static facts to build a graph of locations connected by roads.
    - Computes all-pairs shortest paths between locations using BFS from each node.
    - Extracts the goal location for each package from the task's goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Initialize total heuristic cost to 0.
    2. Check if the state is a goal state by verifying if all goal facts are present in the state. If yes, return 0.
    3. Track the current location of all locatable objects (packages and vehicles) and which packages are inside which vehicles by iterating through the state facts.
    4. For each package `P` that has a goal location `L_goal` according to the task goals:
       a. Determine the current location `L_curr` relevant for transport for package `P`. This is either the location where `P` is on the ground (`(at P L_curr)`) or the location of the vehicle `V` that `P` is inside (`(in P V)` and `(at V L_curr)`).
       b. If the package is already at its goal location on the ground (`(at P L_goal)` is in the state), the cost for this package is 0. Continue to the next package.
       c. If the package is not at its goal:
          - Get the shortest path distance `distance` from the package's current relevant location (`L_curr`) to its goal location (`L_goal`) using the precomputed distances.
          - If `distance` is None (meaning no path exists or locations are not in the graph), add a large penalty to `total_cost` for this package and continue.
          - If `distance` is not None:
             - If the package is currently inside a vehicle:
                - The estimated cost for this package is the drive distance (`distance`) plus the cost of dropping it (1 action).
             - If the package is currently on the ground:
                - The estimated cost for this package is the cost of picking it up (1 action), plus the drive distance (`distance`), plus the cost of dropping it (1 action).
    5. Return the total sum `total_cost`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal locations and road network,
        and precomputing shortest paths.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # Build the road network graph
        self.road_graph = {}
        all_locations_set = set() # Collect all locations mentioned in road facts
        for fact in static_facts:
            if match(fact, "road", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                all_locations_set.add(loc1)
                all_locations_set.add(loc2)
                if loc1 not in self.road_graph:
                    self.road_graph[loc1] = []
                if loc2 not in self.road_graph:
                    self.road_graph[loc2] = []
                self.road_graph[loc1].append(loc2)
                # Assuming roads are bidirectional unless specified otherwise
                # The example instance shows bidirectional roads, so let's assume that.
                self.road_graph[loc2].append(loc1)

        # Add locations from goals to ensure they are nodes in our potential graph,
        # even if they are isolated in the road network definition.
        for goal in self.goals:
             if match(goal, "at", "*", "*"):
                 _, _, loc = get_parts(goal)
                 all_locations_set.add(loc)

        all_locations = list(all_locations_set)

        # Compute all-pairs shortest paths using BFS from each location
        self.distances = {}
        for start_loc in all_locations:
            q = deque([(start_loc, 0)])
            visited = {start_loc}
            self.distances[(start_loc, start_loc)] = 0

            # Only explore from start_loc if it's part of the road network graph
            # Otherwise, it's an isolated node, and distances to others will remain None
            if start_loc in self.road_graph:
                while q:
                    current_loc, dist = q.popleft()

                    # Check neighbors only if current_loc is in the road graph keys
                    if current_loc in self.road_graph:
                        for neighbor in self.road_graph[current_loc]:
                            if neighbor not in visited:
                                visited.add(neighbor)
                                self.distances[(start_loc, neighbor)] = dist + 1
                                q.append((neighbor, dist + 1))


        # Store goal locations for each package.
        self.package_goals = {}
        for goal in self.goals:
            # Goals are typically (at package location)
            if match(goal, "at", "*", "*"):
                _, package, location = get_parts(goal)
                self.package_goals[package] = location
            # Ignore other potential goal predicates if any (like capacity)

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # If the goal is already reached, heuristic is 0.
        if self.goals <= state:
             return 0

        # Track where packages and vehicles are currently located or contained.
        current_locations = {} # Maps locatable object (package or vehicle) to its location
        package_in_vehicle = {} # Maps package to the vehicle it's in (if any)

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at":
                # Fact is (at obj loc)
                if len(parts) == 3: # Basic check for expected format
                    _, obj, loc = parts
                    current_locations[obj] = loc
            elif parts[0] == "in":
                 # Fact is (in package vehicle)
                 if len(parts) == 3: # Basic check for expected format
                    _, package, vehicle = parts
                    package_in_vehicle[package] = vehicle

        total_cost = 0  # Initialize action cost counter.
        LARGE_PENALTY = 1000 # Penalty for unreachable goals or unknown locations

        # Iterate through packages that have a goal location
        for package, goal_location in self.package_goals.items():
            # Check if the package is already at its goal location on the ground
            if (f"(at {package} {goal_location})" in state):
                 # Package is already at goal, no cost for this package
                 continue

            # Package is not at goal. Find its current status.
            current_package_location = None
            is_in_vehicle = False

            if package in package_in_vehicle:
                # Package is inside a vehicle
                vehicle = package_in_vehicle[package]
                # Find the vehicle's location
                if vehicle in current_locations:
                    current_package_location = current_locations[vehicle]
                    is_in_vehicle = True
                else:
                    # Package is in a vehicle, but vehicle location is unknown.
                    # This state is problematic. Assign penalty.
                    total_cost += LARGE_PENALTY
                    continue # Cannot estimate cost for this package

            elif package in current_locations:
                 # Package is on the ground
                 current_package_location = current_locations[package]
                 is_in_vehicle = False
            else:
                 # Package is not 'at' a location and not 'in' a vehicle.
                 # This state is problematic. Assign penalty.
                 total_cost += LARGE_PENALTY
                 continue # Cannot estimate cost for this package


            # Calculate cost based on current status and goal
            # Get the shortest path distance.
            # Check if both current and goal locations are known nodes in our distance map
            # and if a path was found (distance is not None after BFS).
            distance = self.distances.get((current_package_location, goal_location))


            if distance is None:
                # This happens if current_package_location or goal_location is not in the graph
                # (i.e., not part of any road fact or goal processed in __init__)
                # or if there is no path between them during BFS.
                # Assign a large penalty as this package cannot reach its goal via the known road network.
                total_cost += LARGE_PENALTY
            else:
                if is_in_vehicle:
                    # Package is in a vehicle at current_package_location
                    # Cost = drive from current_package_location to goal_location + drop
                    total_cost += distance + 1
                else:
                    # Package is on the ground at current_package_location
                    # Cost = pick-up + drive from current_package_location to goal_location + drop
                    total_cost += 1 + distance + 1 # = 2 + distance

        return total_cost
