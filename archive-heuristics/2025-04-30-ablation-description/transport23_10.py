from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic

# Utility functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at obj loc)".
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
    This heuristic estimates the cost to reach the goal state by summing
    up the estimated minimum actions required for each package to reach
    its goal location independently. The estimate for a single package
    includes the cost of pick-up, drop, and the shortest path distance
    the package (or the vehicle carrying it) needs to travel.

    # Assumptions
    - The cost of each action (drive, pick-up, drop) is 1.
    - Vehicle availability and capacity constraints are ignored when
      estimating the cost for individual packages. It assumes a suitable
      vehicle is always available when needed.
    - The road network is static and bidirectional (if road l1 l2 exists,
      road l2 l1 also exists, or travel cost is the same). The heuristic
      computes shortest paths based on the provided road facts.

    # Heuristic Initialization
    - Parses the goal conditions to identify the target location for each package.
    - Builds a graph representation of the road network from static facts.
    - Computes all-pairs shortest path distances between all locations
      using Breadth-First Search (BFS).

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current location of every package. A package can be
       either on the ground at a location `l` (`(at p l)`) or inside a
       vehicle `v` (`(in p v)`), in which case its effective location
       is the location of the vehicle `(at v l)`.
    2. For each package `p` that has a goal location `goal_l`:
       a. If the package is already at its goal location (`current_l == goal_l`),
          the estimated cost for this package is 0.
       b. If the package is not at its goal location (`current_l != goal_l`):
          i. Calculate the shortest path distance between the package's
             current effective location (`current_l`) and its goal location (`goal_l`)
             using the precomputed distances. If the goal is unreachable,
             the cost for this package is considered infinite.
          ii. Estimate the number of pick-up/drop actions needed:
              - If the package is currently on the ground (`(at p current_l)`),
                it needs a pick-up action (1) and a drop action (1). Add 2.
              - If the package is currently inside a vehicle (`(in p v)`),
                it only needs a drop action (1) at the destination. Add 1.
          iii. The estimated cost for this package is the shortest path distance
               plus the estimated pick-up/drop actions.
    3. The total heuristic value for the state is the sum of the estimated
       costs for all packages that are not yet at their goal locations.
       If any package's goal is unreachable, the total heuristic is infinite.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, building
        the road network graph, and computing shortest paths.
        """
        super().__init__(task)
        self.goals = task.goals
        static_facts = task.static

        # 1. Store goal locations for each package.
        self.goal_locations = {}
        for goal in self.goals:
            if match(goal, "at", "*", "*"):
                predicate, package, location = get_parts(goal)
                self.goal_locations[package] = location

        # 2. Build the road network graph and collect all locations.
        self.road_graph = {}
        locations = set()

        for fact in static_facts:
            if match(fact, "road", "*", "*"):
                predicate, l1, l2 = get_parts(fact)
                locations.add(l1)
                locations.add(l2)
                self.road_graph.setdefault(l1, []).append(l2)
                self.road_graph.setdefault(l2, []).append(l1) # Assuming bidirectional roads

        # Collect all locations from initial state (at facts) and goals (at facts)
        # to ensure all relevant locations are in our graph keys, even if isolated.
        for fact in task.initial_state:
             if match(fact, "at", "*", "*"):
                 predicate, obj, loc = get_parts(fact)
                 locations.add(loc)
        for loc in self.goal_locations.values():
             locations.add(loc)

        # Ensure all found locations are keys in the graph, even if isolated.
        # This is important so BFS can be started from any location.
        for loc in locations:
             self.road_graph.setdefault(loc, [])

        # 3. Compute all-pairs shortest path distances using BFS.
        self.distances = {}
        all_locations = list(self.road_graph.keys()) # Get all unique locations found
        for start_node in all_locations:
            self._bfs(start_node) # BFS from each location

    def _bfs(self, start_node):
        """
        Performs BFS starting from start_node to find distances to all
        reachable nodes in the road graph. Stores results in self.distances.
        Distances to unreachable nodes from start_node are not stored,
        and will be treated as infinity by self.distances.get() in __call__.
        """
        q = deque([(start_node, 0)])
        visited = {start_node}
        self.distances[(start_node, start_node)] = 0 # Distance to self is 0

        while q:
            current_node, dist = q.popleft()

            # Explore neighbors
            # Check if current_node exists in graph keys (should always if built correctly)
            if current_node in self.road_graph:
                for neighbor in self.road_graph[current_node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        # Store distance when first visited (shortest path)
                        self.distances[(start_node, neighbor)] = dist + 1
                        q.append((neighbor, dist + 1))


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions
        to reach the goal state.
        """
        state = node.state  # Current world state.

        # Track current locations of all locatables (packages and vehicles)
        current_locations = {}
        # Track which package is inside which vehicle
        package_in_vehicle = {}

        for fact in state:
            # (at ?x ?l) where ?x is locatable (vehicle or package)
            if match(fact, "at", "*", "*"):
                predicate, obj, loc = get_parts(fact)
                current_locations[obj] = loc
            # (in ?p ?v) where ?p is package and ?v is vehicle
            elif match(fact, "in", "*", "*"):
                predicate, package, vehicle = get_parts(fact)
                package_in_vehicle[package] = vehicle

        total_cost = 0

        # Iterate through packages that have a goal location
        for package, goal_location in self.goal_locations.items():
            current_location = None
            is_in_vehicle = False

            # Find the package's current effective location
            if package in package_in_vehicle:
                vehicle = package_in_vehicle[package]
                if vehicle in current_locations:
                     current_location = current_locations[vehicle]
                     is_in_vehicle = True
                # else: Vehicle location unknown - treat as unreachable?

            if current_location is None and package in current_locations:
                current_location = current_locations[package]
                is_in_vehicle = False

            # If package location is not found, assume unreachable goal for this package
            if current_location is None:
                 # This indicates a problem with the state representation or problem definition
                 # where a goal package doesn't have a location or isn't in a vehicle.
                 # For heuristic purposes, this state is likely not on a valid path to the goal.
                 return float('inf')


            # If package is already at its goal, cost is 0 for this package
            if current_location == goal_location:
                continue

            # Estimate cost if not at goal
            # Cost = distance + pick-up/drop actions

            # Get shortest path distance. Handle unreachable locations.
            # Use .get() with default float('inf') if distance is not found
            # (meaning goal_location is unreachable from current_location).
            distance = self.distances.get((current_location, goal_location), float('inf'))

            if distance == float('inf'):
                 # Goal location is unreachable from current location for this package.
                 return float('inf')

            # Add cost for pick-up and drop actions
            action_cost = 0
            if is_in_vehicle:
                action_cost = 1 # Drop action needed at destination
            else:
                action_cost = 2 # Pick-up action at current_location + Drop action at destination

            # Total cost for this package
            total_cost += distance + action_cost

        return total_cost
