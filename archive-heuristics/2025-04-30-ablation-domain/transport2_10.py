from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque

# Helper functions to parse PDDL facts
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
    # Ensure we don't go out of bounds if fact has fewer parts than args
    if len(parts) < len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions needed to transport each package
    to its goal location, summing the individual estimates. It considers load, unload,
    and drive actions. It simplifies the problem by ignoring vehicle capacity and
    availability, focusing solely on the movement requirements for each package
    based on shortest paths in the road network.

    # Assumptions
    - Vehicle capacity is ignored. Any package can be loaded into any vehicle.
    - Vehicle availability is ignored. A vehicle is assumed to be available at a location if needed.
    - A vehicle carrying a package is assumed to travel towards that package's goal location without detours for other packages.
    - The cost of each action (load, unload, drive) is 1.
    - Road network is static and bidirectional (inferred from examples).
    - Goal conditions only involve packages being at specific locations (inferred from examples).

    # Heuristic Initialization
    - Parses the goal facts to map each package to its destination location.
    - Collects all relevant locations mentioned in the initial state, goals, and road facts.
    - Parses the static 'road' facts to build a graph of locations.
    - Computes the shortest path distance (number of drive actions) between all pairs of relevant locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic is calculated as follows:
    1. Initialize total heuristic cost to 0.
    2. For each package that has a goal location specified in the task:
        a. Determine the package's goal location (`loc_goal`).
        b. Check if the package is already at `loc_goal` on the ground (i.e., `(at package loc_goal)` is in the state). If yes, the cost for this package is 0, continue to the next package.
        c. If the package is not at `loc_goal`, find its current status: Is it on the ground at `loc_current`, or is it inside a vehicle `v` which is at `loc_v`?
        d. If the package is on the ground at `loc_current`:
            - The package needs to be loaded, the vehicle needs to drive from `loc_current` to `loc_goal`, and the package needs to be unloaded.
            - The estimated cost is 1 (load) + shortest_distance(`loc_current`, `loc_goal`) + 1 (unload).
            - If `loc_goal` is unreachable from `loc_current` via roads, this state is likely on an unsolvable path, and the heuristic should reflect this (e.g., return infinity).
        e. If the package is inside a vehicle `v` which is at `loc_v`:
            - The vehicle needs to drive from `loc_v` to `loc_goal`, and the package needs to be unloaded.
            - The estimated cost is shortest_distance(`loc_v`, `loc_goal`) + 1 (unload).
            - If `loc_goal` is unreachable from `loc_v` via roads, return infinity.
        f. Add the estimated cost for this package to the total heuristic cost.
    3. The total heuristic value is the sum of the estimated costs for all packages that are not yet at their goal location. If any required distance was infinite for any package's movement, the total heuristic is infinite.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal locations and precomputing
        shortest path distances between locations.
        """
        # The base class Heuristic likely has task as an attribute, but we store
        # relevant parts explicitly for clarity and direct access.
        # super().__init__(task) # If inheriting and base class needs task

        self.goals = task.goals
        static_facts = task.static
        initial_state = task.initial_state

        # Store goal locations for each package.
        self.goal_locations = {}
        for goal in self.goals:
            # Assuming goals are always (at ?p ?l) for packages
            if match(goal, "at", "*", "*"):
                parts = get_parts(goal)
                if len(parts) == 3: # Ensure it's a binary predicate like (at obj loc)
                    package, location = parts[1:]
                    self.goal_locations[package] = location
            # Ignore other potential goal types for this heuristic

        # Collect all relevant locations from initial state, goals, and road facts
        all_locations = set()
        # Locations from initial state (where objects are at)
        for fact in initial_state:
             if match(fact, "at", "*", "*"):
                 parts = get_parts(fact)
                 if len(parts) == 3:
                     # The second argument of 'at' is the location
                     all_locations.add(parts[2])
        # Locations from goals (where packages need to be at)
        for goal in self.goals:
             if match(goal, "at", "*", "*"):
                 parts = get_parts(goal)
                 if len(parts) == 3:
                     all_locations.add(parts[2])
        # Locations from road facts
        for fact in static_facts:
            if match(fact, "road", "*", "*"):
                parts = get_parts(fact)
                if len(parts) == 3:
                    _, loc1, loc2 = parts
                    all_locations.add(loc1)
                    all_locations.add(loc2)

        # Build the road graph and compute shortest paths.
        self.distances = self._compute_shortest_paths(static_facts, all_locations)

    def _compute_shortest_paths(self, static_facts, all_locations):
        """
        Builds the location graph from 'road' facts and computes all-pairs
        shortest paths using BFS for all relevant locations.
        Returns a dictionary distances[start_loc][end_loc] = shortest_distance.
        If end_loc is unreachable from start_loc, distances[start_loc][end_loc]
        will not exist.
        """
        graph = {}
        for fact in static_facts:
            if match(fact, "road", "*", "*"):
                parts = get_parts(fact)
                if len(parts) == 3:
                    _, loc1, loc2 = parts
                    graph.setdefault(loc1, []).append(loc2)
                    # Assuming roads are bidirectional based on examples
                    graph.setdefault(loc2, []).append(loc1)

        distances = {}
        for start_loc in all_locations:
            distances[start_loc] = {}
            queue = deque([(start_loc, 0)])
            visited = {start_loc}
            distances[start_loc][start_loc] = 0

            while queue:
                current_loc, dist = queue.popleft()

                # Locations might be in all_locations but not in graph if they have no roads
                # or are isolated.
                if current_loc in graph:
                    for neighbor in graph[current_loc]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            distances[start_loc][neighbor] = dist + 1
                            queue.append((neighbor, dist + 1))
                # If current_loc is not in graph, BFS stops for this branch.
                # Unreachable locations will not be added to distances[start_loc].

        return distances

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        total_cost = 0

        # Find current locations/status of all objects (packages and vehicles)
        current_object_status = {} # Maps object name to its location or the vehicle it's in
        for fact in state:
            if match(fact, "at", "*", "*"):
                parts = get_parts(fact)
                if len(parts) == 3:
                    obj, loc = parts[1:]
                    current_object_status[obj] = loc
            elif match(fact, "in", "*", "*"):
                parts = get_parts(fact)
                if len(parts) == 3:
                    obj, vehicle = parts[1:]
                    current_object_status[obj] = vehicle # Store vehicle name if inside

        # Iterate through packages that have a goal
        for package, goal_location in self.goal_locations.items():
            # Check if the package is already at its goal location (on the ground)
            # The goal is (at package goal_location).
            if f"(at {package} {goal_location})" in state:
                 continue # Package is already at goal

            # Find package's current status
            current_status = current_object_status.get(package)

            # This case indicates an issue with state representation or problem definition
            if current_status is None:
                 # If a package isn't 'at' a location or 'in' a vehicle, something is wrong.
                 # This state is likely invalid or on an unsolvable path.
                 return float('inf')

            # Case: Package is on the ground at a location
            # Check if current_status is a location by seeing if it's a key in our distance map
            # (locations are the start nodes for BFS in self.distances)
            if current_status in self.distances:
                loc_current = current_status
                # Package needs load, drive, unload
                # Cost = 1 (load) + distance(loc_current, goal_location) + 1 (unload)
                dist = self.distances.get(loc_current, {}).get(goal_location)

                if dist is None:
                    # Goal location is unreachable from current location
                    # This state is likely on an unsolvable path.
                    return float('inf')
                else:
                    total_cost += 2 + dist

            # Case: Package is inside a vehicle
            # If current_status is not a location key in self.distances, assume it's a vehicle name
            # (relying on PDDL conventions and state representation)
            else: # current_status is assumed to be a vehicle name
                 vehicle = current_status
                 loc_v = current_object_status.get(vehicle) # Get vehicle's location

                 if loc_v is None:
                     # Vehicle location not found? Invalid state representation.
                     return float('inf')

                 # Vehicle needs to drive to goal_location, then package needs unload
                 # Cost = distance(loc_v, goal_location) + 1 (unload)
                 dist = self.distances.get(loc_v, {}).get(goal_location)

                 if dist is None:
                     # Goal location is unreachable from vehicle's current location
                     return float('inf')
                 else:
                      total_cost += 1 + dist

        # If we reached here without returning inf, total_cost is the sum of finite costs.
        # If total_cost is 0, it means all packages were already at their goal (checked by 'continue').
        # So, h=0 iff goal state (for package locations).
        return total_cost
