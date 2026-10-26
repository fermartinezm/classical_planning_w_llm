import collections
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

# Helper function to match PDDL facts with patterns
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
    This heuristic estimates the number of actions required to move all packages
    to their goal locations. It calculates the cost for each package independently,
    ignoring vehicle capacity and potential synergies (like one vehicle moving
    multiple packages or serving multiple pick/drop points on a single trip).
    The cost for a package is estimated based on its current status (at a location
    or in a vehicle) and the shortest path distance on the road network.

    # Assumptions
    - All packages specified in the goal must reach their target location.
    - Roads are bidirectional (if road l1 l2 exists, road l2 l1 also exists).
    - Any vehicle can carry any package (capacity is ignored in the cost calculation,
      though it's a precondition for actions).
    - The cost of pick-up, drop, and driving between adjacent locations is 1.

    # Heuristic Initialization
    - Extract goal locations for each package.
    - Build the road network graph from static `road` facts.
    - Compute all-pairs shortest paths between all locations using BFS.

    # Step-by-Step Thinking for Computing the Heuristic Value
    For each package `p` that needs to be at a goal location `l_goal`:
    1.  Find the current status of package `p`:
        - Is it at a location `l_current`? (`(at p l_current)`)
        - Is it inside a vehicle `v`? (`(in p v)`)
    2.  If package `p` is already at its goal location `l_goal` (i.e., `(at p l_goal)` is true), the cost for this package is 0.
    3.  If package `p` is at a location `l_current` and `l_current != l_goal`:
        - It needs to be picked up (1 action).
        - The vehicle carrying it needs to drive from `l_current` to `l_goal`. The estimated cost is the shortest path distance `dist(l_current, l_goal)`.
        - It needs to be dropped at `l_goal` (1 action).
        - Total estimated cost for this package: 1 + `dist(l_current, l_goal)` + 1 = 2 + `dist(l_current, l_goal)`.
    4.  If package `p` is inside a vehicle `v`:
        - Find the current location `l_v` of vehicle `v` (`(at v l_v)`).
        - The vehicle needs to drive from `l_v` to `l_goal`. The estimated cost is `dist(l_v, l_goal)`.
        - The package needs to be dropped at `l_goal` (1 action).
        - Total estimated cost for this package: `dist(l_v, l_goal)` + 1.
    5.  The total heuristic value is the sum of the estimated costs for all packages
        that are not yet at their goal location.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal locations and precomputing
        shortest paths between all locations.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.
        initial_state = task.initial_state # Initial state to find all objects/locations

        # Store goal locations for each package.
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                package, location = args
                self.goal_locations[package] = location

        # Identify all locations and build the road network graph.
        self.locations = set()
        self.road_graph = collections.defaultdict(set)

        # Extract locations from road facts
        for fact in static_facts:
            if match(fact, "road", "*", "*"):
                _, l1, l2 = get_parts(fact)
                self.locations.add(l1)
                self.locations.add(l2)
                self.road_graph[l1].add(l2)
                self.road_graph[l2].add(l1) # Assuming roads are bidirectional

        # Extract locations from initial state and goals to ensure all relevant locations are included
        for fact in initial_state:
             if match(fact, "at", "*", "*"):
                 _, obj, loc = get_parts(fact)
                 self.locations.add(loc)
        for goal in self.goals:
             if match(goal, "at", "*", "*"):
                 _, obj, loc = get_parts(goal)
                 self.locations.add(loc)


        # Compute all-pairs shortest paths using BFS.
        self.distances = {}
        for start_loc in self.locations:
            self.distances[start_loc] = self._bfs(start_loc)

    def _bfs(self, start_node):
        """
        Performs Breadth-First Search starting from start_node to find shortest
        distances to all other reachable nodes in the road graph.
        """
        distances = {node: float('inf') for node in self.locations}
        distances[start_node] = 0
        queue = collections.deque([start_node])

        while queue:
            current_node = queue.popleft()
            current_dist = distances[current_node]

            if current_node in self.road_graph: # Check if node has neighbors
                for neighbor in self.road_graph[current_node]:
                    if distances[neighbor] == float('inf'):
                        distances[neighbor] = current_dist + 1
                        queue.append(neighbor)
        return distances

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Map objects (packages and vehicles) to their current location or container.
        current_status = {}
        # Map vehicles to their current location
        vehicle_locations = {}

        for fact in state:
            if match(fact, "at", "*", "*"):
                _, obj, loc = get_parts(fact)
                current_status[obj] = loc
                # Assuming anything 'at' a location that is not a package is a vehicle
                # This is a simplification; a more robust way would be to know object types
                # from the task definition, but it's not provided in the Task class.
                # We can infer vehicles by checking if the object is NOT a package in the goals.
                # Or, more simply, check if the object is a 'vehicle' type if that info was available.
                # Given the domain structure, objects at locations are either packages or vehicles.
                # Let's assume objects starting with 'v' are vehicles based on examples.
                if obj.startswith('v'): # Simple check based on example naming convention
                     vehicle_locations[obj] = loc
                elif obj in self.goal_locations: # If it's a package with a goal
                     pass # Already added to current_status
                else: # Could be other locatables? Let's stick to packages/vehicles.
                     pass

            elif match(fact, "in", "*", "*"):
                _, package, vehicle = get_parts(fact)
                current_status[package] = vehicle # Package is inside a vehicle

        total_cost = 0  # Initialize action cost counter.

        # Iterate through packages that have a goal location
        for package, goal_location in self.goal_locations.items():
            # Check if the package is already at its goal location on the ground
            if (f"(at {package} {goal_location})" in state):
                 continue # Package is already at its goal location, cost is 0 for this package

            # Package is not at its goal location on the ground. Find its current status.
            if package not in current_status:
                 # This case should ideally not happen in a valid state representation
                 # where every locatable object has an 'at' or 'in' predicate.
                 # For robustness, we could assign a large penalty or skip.
                 # Let's assume valid states for now.
                 continue

            current_loc_or_vehicle = current_status[package]

            # Case 1: Package is at some location l_current (not the goal)
            if current_loc_or_vehicle in self.locations:
                l_current = current_loc_or_vehicle
                # Needs pick-up, drive, drop
                # Cost = 1 (pick) + dist(l_current, l_goal) (drive) + 1 (drop)
                drive_cost = self.distances[l_current].get(goal_location, float('inf'))
                if drive_cost == float('inf'):
                    # Goal location is unreachable from current location
                    # This indicates an unsolvable problem or a very high cost
                    # For a non-admissible heuristic, a large number is fine.
                    total_cost += 1000 # Penalty for unreachable goal
                else:
                    total_cost += 1 + drive_cost + 1

            # Case 2: Package is inside a vehicle v
            elif current_loc_or_vehicle.startswith('v'): # Simple check for vehicle name
                vehicle = current_loc_or_vehicle
                # Find the vehicle's location
                if vehicle not in vehicle_locations:
                    # Vehicle location is unknown - indicates invalid state or issue
                    # Assign a large penalty
                    total_cost += 1000
                    continue

                l_v = vehicle_locations[vehicle]

                # Needs drive (by vehicle v), drop
                # Cost = dist(l_v, l_goal) (drive) + 1 (drop)
                drive_cost = self.distances[l_v].get(goal_location, float('inf'))
                if drive_cost == float('inf'):
                     total_cost += 1000 # Penalty for unreachable goal
                else:
                    total_cost += drive_cost + 1

            # Else: Status is something unexpected (e.g., inside another package?)
            # Assign a large penalty
            else:
                 total_cost += 1000


        return total_cost

