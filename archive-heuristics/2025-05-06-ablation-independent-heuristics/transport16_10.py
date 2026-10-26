# Required imports
from heuristics.heuristic_base import Heuristic
from task import Task
from collections import deque
import math

# Assume Heuristic and Task classes are available from the planning framework

class transportHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Transport domain.

    Summary:
    Estimates the cost to reach the goal by summing the minimum actions
    required for each package to reach its goal location. The minimum actions
    for a package are calculated based on the shortest path distance in the
    road network, plus the necessary pick-up and drop actions. This heuristic
    ignores vehicle capacity constraints and vehicle availability beyond the
    package's current state (whether it's at a location or in a vehicle).

    Assumptions:
    - The road network defined by `(road ?l1 ?l2)` facts is static.
    - Goal facts are always of the form `(at package location)`.
    - Valid states ensure that all locatable objects (packages and vehicles)
      are either `at` a specific location or, in the case of packages, `in`
      a vehicle which is itself `at` a specific location.
    - The road network might not be fully connected; unreachable goals are
      handled by returning infinity.

    Heuristic Initialization:
    The constructor performs the following steps once when the heuristic is
    created:
    1.  Parses the goal facts from the task to identify which packages need
        to be at which target locations. This information is stored in
        `self.package_goals`.
    2.  Parses the static `(road ?l1 ?l2)` facts to build a graph representation
        of the road network. This graph is stored as an adjacency list.
    3.  Computes all-pairs shortest paths on the road network graph using
        Breadth-First Search (BFS) starting from every location. The distances
        (number of drive actions) are stored in `self.shortest_distances`.
    4.  Static capacity information (`capacity-predecessor`) is identified but
        ignored in this version of the heuristic for simplicity and efficiency.

    Step-By-Step Thinking for Computing Heuristic:
    The `__call__` method computes the heuristic value for a given state:
    1.  Initialize the total heuristic value `h` to 0.
    2.  Iterate through the current state facts to build dictionaries that map
        locatable objects to their current location (`current_locations`) and
        packages to the vehicle they are in (`packages_in_vehicles`).
    3.  Iterate through each package that has a defined goal location
        (obtained during initialization from `self.package_goals`).
    4.  For the current package, determine its current location. This is done
        by checking if the package is directly `at` a location or if it is
        `in` a vehicle whose location is known.
    5.  If the package's current location cannot be determined from the state
        (e.g., the package is `in` a vehicle, but the vehicle's location is
        not specified by an `(at vehicle location)` fact), the state is
        considered problematic or unsolvable for this package, and the
        heuristic immediately returns `math.inf`.
    6.  If the package is already at its goal location, add 0 to the total
        heuristic for this package and proceed to the next package.
    7.  If the package is not at its goal location, calculate the minimum
        cost to move it:
        -   Retrieve the shortest distance (number of drive actions) between
            the package's current location and its goal location from the
            precomputed `self.shortest_distances`. If the goal location is
            unreachable from the current location, the distance will be
            `math.inf`.
        -   If the distance is `math.inf`, the goal is unreachable for this
            package, and the heuristic immediately returns `math.inf`.
        -   Otherwise, add this finite distance to the package's cost.
        -   If the package is currently `at` a location (meaning it's not
            already in a vehicle), add 1 to the package's cost for the
            necessary `pick-up` action.
        -   Add 1 to the package's cost for the necessary `drop` action at the
            goal location.
    8.  Add the calculated minimum cost for the current package to the total
        heuristic `h`.
    9.  After processing all packages with goals, return the accumulated total
        heuristic value `h`.
    """

    def __init__(self, task: Task):
        """
        Initializes the transport heuristic.

        Args:
            task: The planning task instance.
        """
        super().__init__()
        self.task = task

        # 1. Store goal locations for packages.
        self.package_goals = {} # {package_name: goal_location}
        for goal_fact_str in task.goals:
            pred, args = self._parse_fact(goal_fact_str)
            if pred == 'at' and len(args) == 2:
                item, loc = args
                # Assuming items in 'at' goals are packages
                self.package_goals[item] = loc
            # Ignore other types of goal facts if any

        # 2. Build road network graph and compute all-pairs shortest paths.
        self.road_graph, self.locations = self._build_road_graph(task.static)
        self.shortest_distances = self._compute_shortest_paths(self.road_graph, self.locations)

        # Capacity information is ignored in this simple heuristic.
        # self.capacity_predecessors = {}
        # for fact_str in task.static:
        #     pred, args = self._parse_fact(fact_str)
        #     if pred == 'capacity-predecessor':
        #         s1, s2 = args
        #         self.capacity_predecessors[s1] = s2 # s1 < s2


    def _parse_fact(self, fact_str):
        """
        Helper method to parse a PDDL fact string.

        Args:
            fact_str: A string representing a PDDL fact (e.g., '(at p1 l1)').

        Returns:
            A tuple containing the predicate name (string) and a list of
            arguments (strings). Returns (None, []) for empty or invalid strings.
        """
        # Remove surrounding parentheses and split by space
        # Handle potential empty string or just '()'
        if not fact_str or fact_str.strip() == '()':
             return None, []
        
        # Ensure fact_str starts with '(' and ends with ')'
        if not (fact_str.startswith('(') and fact_str.endswith(')')):
             # Or raise an error, depending on desired robustness
             return None, []

        parts = fact_str[1:-1].split()
        if not parts: # Handle fact like '()' after stripping
             return None, []
             
        return parts[0], parts[1:] # predicate, arguments

    def _build_road_graph(self, static_facts):
        """
        Helper method to build the road network graph from static facts.

        Args:
            static_facts: A frozenset of static fact strings.

        Returns:
            A tuple containing:
            - graph: An adjacency list representation {location: [neighbor_location, ...]}
            - locations: A list of all unique location names.
        """
        graph = {}
        locations = set()
        for fact_str in static_facts:
            pred, args = self._parse_fact(fact_str)
            if pred == 'road' and len(args) == 2:
                l1, l2 = args
                locations.add(l1)
                locations.add(l2)
                if l1 not in graph:
                    graph[l1] = []
                # Ensure all locations are keys in the graph dictionary, even if they have no outgoing roads
                if l2 not in graph:
                    graph[l2] = []
                graph[l1].append(l2)
        return graph, list(locations)

    def _compute_shortest_paths(self, graph, locations):
        """
        Helper method to compute all-pairs shortest paths using BFS.

        Args:
            graph: The road network graph (adjacency list).
            locations: A list of all unique location names.

        Returns:
            A dictionary mapping (start_location, end_location) tuples to
            their shortest distance (number of drive actions). Unreachable
            pairs are not included.
        """
        distances = {}
        for start_loc in locations:
            q = deque([(start_loc, 0)])
            visited = {start_loc}
            distances[(start_loc, start_loc)] = 0 # Distance to self is 0

            while q:
                current_loc, dist = q.popleft()

                # Check if current_loc exists in graph keys (handles isolated locations)
                if current_loc in graph:
                    for neighbor in graph[current_loc]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            distances[(start_loc, neighbor)] = dist + 1
                            q.append((neighbor, dist + 1))
        return distances

    def __call__(self, node):
        """
        Computes the heuristic value for the given state.

        Args:
            node: The current search node, containing the state.

        Returns:
            The estimated number of actions to reach a goal state, or infinity
            if the goal is unreachable from this state.
        """
        state = node.state
        h = 0

        # Build location maps from the current state
        current_locations = {} # {item_name: location_name} for items that are 'at' a location
        packages_in_vehicles = {} # {package_name: vehicle_name} for packages that are 'in' a vehicle

        for fact_str in state:
            pred, args = self._parse_fact(fact_str)
            if pred == 'at' and len(args) == 2:
                item, loc = args
                current_locations[item] = loc
            elif pred == 'in' and len(args) == 2:
                package, vehicle = args
                packages_in_vehicles[package] = vehicle

        # Calculate heuristic for each package that has a goal
        for package, goal_loc in self.package_goals.items():
            package_h = 0
            current_loc = None
            is_in_vehicle = False

            if package in current_locations:
                # Package is at a location
                current_loc = current_locations[package]
                is_in_vehicle = False
            elif package in packages_in_vehicles:
                # Package is in a vehicle, find vehicle's location
                vehicle = packages_in_vehicles[package]
                if vehicle in current_locations:
                    current_loc = current_locations[vehicle]
                    is_in_vehicle = True
                else:
                    # Vehicle exists but its location is unknown. This state is problematic.
                    # Assuming valid states always have vehicles 'at' a location.
                    # If not, this branch implies an unsolvable situation for this package.
                    # Return infinity as the goal is unreachable for this package.
                    return math.inf
            else:
                 # Package is neither 'at' a location nor 'in' a vehicle. Problematic state.
                 # Return infinity as the goal is unreachable for this package.
                 return math.inf

            # If package is already at goal, cost is 0 for this package
            if current_loc == goal_loc:
                continue

            # Get distance from current location to goal location
            # If current_loc or goal_loc is not in locations (shouldn't happen if graph built correctly)
            # or if goal_loc is unreachable from current_loc, distance will not be in shortest_distances
            # Use math.inf if the distance is not found (unreachable)
            dist = self.shortest_distances.get((current_loc, goal_loc), math.inf)

            if dist == math.inf:
                # Goal is unreachable for this package from its current location/vehicle location
                return math.inf

            # Calculate cost for this package
            # Cost = Drive actions + Pick-up action (if needed) + Drop action
            package_h = dist # Minimum drive actions
            if not is_in_vehicle:
                package_h += 1 # Pick-up action needed if package is at a location
            package_h += 1 # Drop action needed

            h += package_h

        return h
