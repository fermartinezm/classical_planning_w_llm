from collections import deque
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper functions (can be outside the class)
def get_parts(fact):
    """Extract the components of a PDDL fact."""
    # Handle potential empty facts or malformed strings defensively
    if not fact or not isinstance(fact, str) or len(fact) < 2 or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    `fact`: The complete fact string, e.g., "(at package1 locationA)".
    `args`: The expected pattern (wildcards `*` allowed).
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions required to move each package
    from its current location to its goal location, independently. It sums the
    estimated costs for all packages that are not yet at their destination.
    The cost for a single package is estimated based on whether it needs to be
    picked up, the shortest path distance its carrier vehicle needs to travel,
    and the final drop action.

    # Assumptions:
    - The road network is static and provides bidirectional connections.
    - Vehicle capacity constraints are ignored. Any vehicle is assumed capable
      of carrying any package if it is available at the package's location.
    - Vehicle availability is ignored. It is assumed a vehicle is available
      to pick up a package when needed (though the vehicle's current location
      is considered if the package is already inside one).
    - Multiple packages can be transported by the same vehicle, but the
      heuristic sums costs independently per package, potentially overestimating
      or underestimating due to shared vehicle trips.
    - The cost of each action (drive, pick-up, drop) is 1.

    # Heuristic Initialization
    - The heuristic precomputes the shortest path distances between all pairs
      of relevant locations (locations mentioned in static road facts or goal
      facts) using Breadth-First Search (BFS) on the road network defined by
      the static facts.
    - It extracts the goal location for each package from the task's goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current location of every package and every vehicle.
       Also, identify which packages are currently inside which vehicles.
    2. Initialize the total heuristic cost to 0.
    3. For each package that has a specified goal location:
       a. Check if the package is already at its goal location according to the current state. If yes, the cost for this package is 0, continue to the next package.
       b. If the package is not at its goal:
          i. Determine the package's current status: Is it on the ground at some location, or is it inside a vehicle?
          ii. If the package is on the ground at location `l_p`:
              - It needs a pick-up action (cost 1).
              - It needs to be transported from `l_p` to its goal location `l_goal`. The minimum number of drive actions required for the vehicle carrying it is the shortest path distance between `l_p` and `l_goal`.
              - It needs a drop action at `l_goal` (cost 1).
              - The estimated cost for this package is 1 (pickup) + distance(`l_p`, `l_goal`) + 1 (drop).
          iii. If the package is inside a vehicle `v`:
              - Find the current location `l_v` of vehicle `v`.
              - It needs to be transported from `l_v` to its goal location `l_goal`. The minimum number of drive actions required is the shortest path distance between `l_v` and `l_goal`.
              - It needs a drop action at `l_goal` (cost 1).
              - The estimated cost for this package is distance(`l_v`, `l_goal`) + 1 (drop).
       c. If the goal location is unreachable from the package's or vehicle's current location (distance is infinity), the heuristic for this state is infinity.
       d. Add the estimated cost for this package to the total heuristic cost.
    4. Return the total heuristic cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by precomputing distances and storing goal locations.
        """
        super().__init__(task) # Call base class constructor

        self.goal_locations = {}
        locations = set()
        graph = {}

        # Parse static facts to build the road network graph and collect locations
        for fact in self.task.static:
            parts = get_parts(fact)
            if match(fact, "road", "*", "*"):
                l1, l2 = parts[1], parts[2]
                locations.add(l1)
                locations.add(l2)
                graph.setdefault(l1, []).append(l2)
                graph.setdefault(l2, []).append(l1) # Roads are bidirectional

        # Parse goal facts to get package goal locations and collect goal locations
        # Assuming task.goals is an iterable of goal facts (strings)
        for goal in self.task.goals:
             parts = get_parts(goal)
             if match(goal, "at", "*", "*"):
                 package, location = parts[1], parts[2]
                 self.goal_locations[package] = location
                 locations.add(location) # Add goal locations to the set of relevant locations

        # Ensure all collected locations are in the graph dictionary keys, even if isolated
        # This is important for BFS to explore all known locations. Isolated locations
        # will have empty adjacency lists.
        for loc in locations:
             graph.setdefault(loc, [])

        # Compute all-pairs shortest paths using BFS
        self.distances = {}
        # Compute BFS from every location that is part of the graph (either in road or goal)
        for start_node in graph:
            dist_from_start = self._bfs(graph, start_node)
            for end_node, dist in dist_from_start.items():
                self.distances[(start_node, end_node)] = dist

    def _bfs(self, graph, start_node):
        """
        Perform Breadth-First Search to find shortest distances from start_node.
        """
        # Initialize distances for all nodes known in the graph
        distances = {node: float('inf') for node in graph}

        # Start BFS only if the start_node is a valid key in the graph
        if start_node in graph:
            distances[start_node] = 0
            queue = deque([start_node])

            while queue:
                current = queue.popleft()

                # If current node is unreachable (shouldn't happen if start_node had dist 0), skip
                if distances[current] == float('inf'):
                    continue

                # Check neighbors only if current is a valid key in the graph
                for neighbor in graph.get(current, []):
                    if distances[neighbor] == float('inf'):
                        distances[neighbor] = distances[current] + 1
                        queue.append(neighbor)
        return distances


    def __call__(self, node):
        """
        Compute the heuristic estimate for the given state.
        """
        state = node.state

        # Map current locations of locatables (packages and vehicles)
        current_locations = {}
        # Map packages to the vehicles they are inside
        package_in_vehicle = {}

        for fact in state:
            parts = get_parts(fact)
            if not parts: # Skip malformed facts
                continue
            predicate = parts[0]

            if predicate == "at" and len(parts) == 3:
                 # (at ?x - locatable ?v - location)
                 locatable_obj, location = parts[1], parts[2]
                 current_locations[locatable_obj] = location
            elif predicate == "in" and len(parts) == 3:
                 # (in ?x - package ?v - vehicle)
                 package, vehicle = parts[1], parts[2]
                 package_in_vehicle[package] = vehicle

        total_cost = 0

        # Iterate through packages that have a goal location
        for package, goal_location in self.goal_locations.items():
            # Check if the package is already at its goal location
            # This check needs to be against the state facts directly
            is_at_goal = f"(at {package} {goal_location})" in state

            if is_at_goal:
                continue # Package is already at its goal, cost is 0 for this package

            # Package is not at its goal, calculate estimated cost
            package_cost = 0

            if package in package_in_vehicle:
                # Package is inside a vehicle
                vehicle = package_in_vehicle[package]
                vehicle_location = current_locations.get(vehicle)

                if vehicle_location is None:
                    # Vehicle location unknown - indicates an invalid state or parsing issue
                    # Return infinity as the goal is likely unreachable
                    return float('inf')

                # Cost is distance from vehicle's current location to goal + drop action
                # Use .get() with default float('inf') to handle cases where goal_location
                # or vehicle_location was not part of the initial graph construction
                dist = self.distances.get((vehicle_location, goal_location), float('inf'))

                if dist == float('inf'):
                    # Goal location is unreachable from the vehicle's current location
                    return float('inf')

                package_cost = dist + 1 # drive(s) + drop

            else:
                # Package is on the ground
                package_location = current_locations.get(package)

                if package_location is None:
                    # Package location unknown - indicates it's not 'at' a location and not 'in' a vehicle
                    # This shouldn't happen in a valid state, but handle defensively
                    # Return infinity as the package is effectively lost
                    return float('inf')

                # Cost is pickup action + distance from package's location to goal + drop action
                # Use .get() with default float('inf') to handle unreachable goals
                dist = self.distances.get((package_location, goal_location), float('inf'))

                if dist == float('inf'):
                    # Goal location is unreachable from the package's current location
                    return float('inf')

                package_cost = 1 + dist + 1 # pickup + drive(s) + drop

            total_cost += package_cost

        return total_cost
