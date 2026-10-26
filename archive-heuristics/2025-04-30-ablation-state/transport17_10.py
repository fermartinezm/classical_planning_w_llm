# Imports needed
from fnmatch import fnmatch
from collections import deque
# Assuming Heuristic base class is available in heuristics.heuristic_base
from heuristics.heuristic_base import Heuristic

# Helper functions from example
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact strings or malformed facts gracefully
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

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

# Define the heuristic class
class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions required to move all packages
    to their goal locations. It sums the estimated cost for each package
    independently.

    # Assumptions
    - Any vehicle can pick up any package if they are at the same location.
      Capacity constraints are ignored.
    - A suitable vehicle is always available at the required location when
      a package needs to be picked up or dropped. The cost of moving a vehicle
      to a package's location is not explicitly accounted for if the vehicle
      is not already there.
    - Road network is static and bidirectional.
    - Action costs are uniform (1 per action).

    # Heuristic Initialization
    - The goal location for each package is extracted from the task goals.
    - The road network graph is built from static facts.
    - All-pairs shortest path distances between locations are precomputed
      using Breadth-First Search (BFS) on the road network graph.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic value is calculated as follows:
    1. Initialize the total heuristic cost to 0.
    2. Determine the current location of every locatable object (packages and vehicles)
       and which packages are inside which vehicles by parsing the state facts.
    3. For each package that has a specified goal location:
       a. Check if the package is already at its goal location on the ground. If yes,
          this package contributes 0 to the heuristic.
       b. If the package is not at its goal location:
          i. If the package is currently inside a vehicle:
             - Find the current location of that vehicle.
             - If the vehicle is at the package's goal location, the estimated cost
               for this package is 1 (for the 'drop' action).
             - If the vehicle is not at the package's goal location, the estimated cost
               is the shortest distance from the vehicle's current location to the
               package's goal location (for 'drive' actions) plus 1 (for the 'drop' action).
          ii. If the package is currently on the ground at some location (not the goal):
              - The estimated cost for this package is 1 (for 'pick-up') plus the
                shortest distance from the package's current location to its goal
                location (for 'drive' actions) plus 1 (for 'drop'). This simplifies
                to 2 + distance.
    4. The total heuristic value is the sum of the estimated costs for all packages
       that are not yet at their goal locations on the ground.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal locations and precomputing
        shortest path distances in the road network.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # Build the road network graph
        roads = {}  # Adjacency list: location -> list of connected locations
        locations_in_roads = set()
        for fact in static_facts:
            predicate, *args = get_parts(fact)
            if predicate == "road":
                if len(args) == 2:
                    l1, l2 = args
                    roads.setdefault(l1, []).append(l2)
                    roads.setdefault(l2, []).append(l1)  # Assuming roads are bidirectional
                    locations_in_roads.add(l1)
                    locations_in_roads.add(l2)

        # Store goal locations for each package
        self.package_goals = {}
        goal_locations_set = set()
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                if len(args) == 2:
                    package, location = args
                    self.package_goals[package] = location
                    goal_locations_set.add(location)

        # Combine locations from roads and goals to ensure all relevant locations are considered
        all_relevant_locations = locations_in_roads.union(goal_locations_set)

        # Precompute all-pairs shortest path distances using BFS
        self.dist = {}
        for start_loc in all_relevant_locations:
            distances_from_start = self._bfs(roads, start_loc)
            for end_loc, d in distances_from_start.items():
                 # Store distance only if both start and end are relevant locations
                 if end_loc in all_relevant_locations:
                    self.dist[(start_loc, end_loc)] = d
            # Ensure distance from start_loc to itself is 0, even if isolated
            if (start_loc, start_loc) not in self.dist:
                 self.dist[(start_loc, start_loc)] = 0


    def _bfs(self, graph, start_node):
        """
        Performs Breadth-First Search to find shortest distances from a start node.
        Assumes uniform edge weights (distance 1 per step).
        Returns a dictionary mapping reachable nodes to their distance from start_node.
        """
        # If start_node is not in the graph keys (e.g., goal location not in roads)
        # it's an isolated node in the road network graph.
        if start_node not in graph:
             return {start_node: 0} # Distance to itself is 0, nothing else is reachable

        q = deque([start_node])
        visited = {start_node}
        dist = {start_node: 0}

        while q:
            u = q.popleft()
            # u is guaranteed to be in graph keys here because we checked start_node
            # and only add neighbors from graph[u] to the queue.
            for v in graph[u]:
                if v not in visited:
                    visited.add(v)
                    dist[v] = dist[u] + 1
                    q.append(v)
        return dist


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state (frozenset of facts)

        # Track where packages and vehicles are currently located.
        current_locations = {} # Maps locatable object (package/vehicle) -> location object
        package_in_vehicle = {} # Maps package object -> vehicle object

        for fact in state:
            predicate, *args = get_parts(fact)
            if predicate == "at":
                if len(args) == 2:
                    obj, loc = args
                    current_locations[obj] = loc
            elif predicate == "in":
                if len(args) == 2:
                    package, vehicle = args
                    package_in_vehicle[package] = vehicle

        total_cost = 0  # Initialize action cost counter.

        # Iterate through packages that have a goal location specified
        for package, goal_location in self.package_goals.items():
            # Check if the package is already at its goal location on the ground
            if package in current_locations and current_locations[package] == goal_location:
                # Package is already at the goal location on the ground, no cost needed for this package
                continue

            # Package is not at the goal location on the ground. Estimate cost to move it.
            if package in package_in_vehicle:
                # Package is currently inside a vehicle
                vehicle = package_in_vehicle[package]
                # Find the current location of the vehicle
                vehicle_location = current_locations.get(vehicle) # Use .get for safety

                if vehicle_location is None:
                     # Vehicle location unknown, cannot estimate cost. This indicates an invalid state.
                     # Return infinity as it's likely unsolvable from here.
                     return float('inf')

                # If vehicle_location is not in the precomputed distances (e.g., isolated location),
                # distance lookup will return inf.
                distance = self.dist.get((vehicle_location, goal_location), float('inf'))

                if distance == float('inf'):
                     # Cannot reach goal location from vehicle's current location
                     return float('inf') # Problem is likely unsolvable

                if vehicle_location == goal_location:
                    # Vehicle is at the goal location, just need to drop the package
                    total_cost += 1 # Cost of 'drop' action
                else:
                    # Vehicle needs to drive to the goal location and then drop the package
                    # Add distance cost (drive actions) + drop action cost
                    total_cost += distance + 1 # Distance + 'drop' action

            elif package in current_locations:
                # Package is currently on the ground, not at the goal location
                current_package_location = current_locations[package]

                # If current_package_location is not in the precomputed distances (e.g., isolated location),
                # distance lookup will return inf.
                distance = self.dist.get((current_package_location, goal_location), float('inf'))

                if distance == float('inf'):
                     # Cannot reach goal location from package's current location
                     return float('inf') # Problem is likely unsolvable

                # Need to pick up the package, drive it to the goal, and drop it.
                # Cost is 1 (pick-up) + distance (drive) + 1 (drop)
                total_cost += 1 + distance + 1 # 'pick-up' + distance + 'drop'

            # If package is not in package_in_vehicle and not in current_locations,
            # it's not mentioned in the state facts as being anywhere. This is an
            # invalid state according to PDDL semantics for locatable objects.
            # We assume valid states are provided, so this case should not be reached.
            # If it were reached, returning inf might be appropriate.

        return total_cost
