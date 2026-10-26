from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact strings or malformed facts gracefully,
    # although in a planning state, facts are expected to be well-formed.
    if not fact or not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
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

def bfs(graph, start_node, all_nodes):
    """
    Performs Breadth-First Search to find shortest distances from start_node
    to all reachable nodes in the graph.

    Args:
        graph: Adjacency list representation {node: [neighbor1, ...]}
        start_node: The starting node for BFS.
        all_nodes: A set of all possible nodes in the graph.

    Returns:
        A dictionary {node: distance} for all nodes reachable from start_node.
        Unreachable nodes will have distance float('inf').
    """
    distances = {node: float('inf') for node in all_nodes}
    if start_node not in all_nodes:
         # Start node is not in the graph, cannot reach anything
         return distances

    distances[start_node] = 0
    queue = deque([start_node])

    while queue:
        u = queue.popleft()
        # Ensure u is a key in the graph, even if it has no neighbors
        for v in graph.get(u, []):
            if distances[v] == float('inf'):
                distances[v] = distances[u] + 1
                queue.append(v)
    return distances


class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions required to move all packages
    to their goal locations. It sums the estimated cost for each package
    independently, ignoring vehicle capacity and assuming vehicles are available
    when needed. The cost for a package is estimated based on its current
    location (on the ground or in a vehicle) and the shortest path distance
    on the road network to its goal location.

    # Assumptions
    - Each package needs to reach a specific goal location.
    - Vehicles can move between locations connected by roads.
    - The cost of moving a vehicle between adjacent locations is 1 action (drive).
    - Loading a package into a vehicle costs 1 action (load).
    - Unloading a package from a vehicle costs 1 action (unload).
    - Vehicle capacity constraints are ignored.
    - Vehicle availability is ignored (assumes a vehicle is available where needed).
    - Road network is static and defined by 'road' predicates.
    - Packages and vehicles are always located somewhere ('at' or 'in').

    # Heuristic Initialization
    - Extracts the goal location for each package from the task's goal conditions.
    - Builds the road network graph from 'road' static facts.
    - Computes the shortest path distance (number of drive actions) between all
      pairs of locations using Breadth-First Search (BFS). This is stored
      for quick lookup during heuristic computation.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic is computed as follows:
    1. Identify the current location of every package and vehicle, and which
       packages are currently inside which vehicles by parsing the state facts.
    2. Initialize the total heuristic cost to 0.
    3. For each package that has a goal location specified in the task's goals:
        a. Check if the package is already at its goal location in the current state.
           If yes, the cost for this package is 0, continue to the next package.
        b. If the package is not at its goal:
            i. Determine the package's current status: Is it on the ground at some
               location `l_curr`, or is it inside a vehicle `v` which is at location `l_curr`?
            ii. If the package is on the ground at `l_curr` (and `l_curr` is not the goal):
                - It needs to be loaded into a vehicle (1 action).
                - The vehicle needs to drive from `l_curr` to the goal location `l_goal`.
                  The minimum number of drive actions is the shortest path distance
                  `dist(l_curr, l_goal)` on the road network.
                - The package needs to be unloaded at `l_goal` (1 action).
                - The estimated cost for this package is `1 + dist(l_curr, l_goal) + 1`.
            iii. If the package is inside a vehicle `v` which is at `l_curr`:
                - If `l_curr` is the goal location `l_goal`:
                    - The package only needs to be unloaded (1 action).
                    - The estimated cost for this package is `1`.
                - If `l_curr` is not the goal location `l_goal`:
                    - The vehicle needs to drive from `l_curr` to `l_goal`.
                      The minimum number of drive actions is `dist(l_curr, l_goal)`.
                    - The package needs to be unloaded at `l_goal` (1 action).
                    - The estimated cost for this package is `dist(l_curr, l_goal) + 1`.
            iv. If the goal location is unreachable from the package's current location
                (or vehicle's location) via the road network, assign a very large cost
                to indicate this path is likely not part of a solution.
        c. Add the estimated cost for this package to the total heuristic cost.
    4. Return the total heuristic cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and building
        the road network graph to precompute shortest path distances.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # Store goal locations for each package.
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                if len(args) == 2: # Expects (at package location)
                    package, location = args
                    self.goal_locations[package] = location

        # Build the road network graph from static facts.
        self.road_graph = {}
        all_locations = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "road":
                if len(parts) == 3: # Expects (road l1 l2)
                    l1, l2 = parts[1], parts[2]
                    self.road_graph.setdefault(l1, []).append(l2)
                    all_locations.add(l1)
                    all_locations.add(l2)

        # Ensure all locations mentioned in road facts are in the graph keys
        # even if they have no outgoing roads (though unlikely in transport).
        for loc in all_locations:
             self.road_graph.setdefault(loc, [])

        self.all_locations = list(all_locations) # Store as list for consistent iteration if needed

        # Compute all-pairs shortest paths using BFS.
        self.distances = {}
        # Use a large integer for heuristic infinity, as heuristic values are typically integers
        self.infinity_cost = 1000000

        for start_loc in self.all_locations:
            distances_from_start = bfs(self.road_graph, start_loc, set(self.all_locations)) # Pass set for BFS
            for end_loc, dist in distances_from_start.items():
                 if dist != float('inf'):
                     self.distances[(start_loc, end_loc)] = dist
                 else:
                     # Store unreachable as the large integer cost
                     self.distances[(start_loc, end_loc)] = self.infinity_cost


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Track where packages and vehicles are currently located.
        current_locations = {}  # object -> location (for packages and vehicles)
        package_in_vehicle = {} # package -> vehicle

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts if any
            predicate = parts[0]
            if predicate == "at":
                if len(parts) == 3: # Expects (at obj loc)
                    obj, loc = parts[1], parts[2]
                    current_locations[obj] = loc
            elif predicate == "in":
                 if len(parts) == 3: # Expects (in pkg veh)
                    pkg, veh = parts[1], parts[2]
                    package_in_vehicle[pkg] = veh

        total_cost = 0

        for package, goal_location in self.goal_locations.items():
            # Check if package is already at goal
            if (f"(at {package} {goal_location})") in state:
                continue  # Package is at goal, cost is 0 for this package

            # Package is not at goal. Find its current location/status.
            cost_for_package = 0

            if package in package_in_vehicle:
                # Package is in a vehicle
                vehicle = package_in_vehicle[package]
                if vehicle in current_locations:
                    current_location = current_locations[vehicle]
                    # Package is in vehicle at current_location, needs to reach goal_location
                    if current_location != goal_location:
                        # Needs drive + unload
                        drive_cost = self.distances.get((current_location, goal_location), self.infinity_cost)
                        if drive_cost == self.infinity_cost:
                            # Unreachable goal location for this package
                            return self.infinity_cost # Return large value if goal is unreachable
                        cost_for_package += drive_cost
                    # Needs unload at goal_location (either vehicle was already there or drove there)
                    cost_for_package += 1  # unload action
                else:
                    # Vehicle location unknown? Should not happen in valid state.
                    # Treat as unreachable.
                    return self.infinity_cost

            elif package in current_locations:
                # Package is on the ground at current_location
                current_location = current_locations[package]
                # Package is at current_location, needs to reach goal_location
                # Needs load + drive + unload
                drive_cost = self.distances.get((current_location, goal_location), self.infinity_cost)
                if drive_cost == self.infinity_cost:
                     # Unreachable goal location for this package
                     return self.infinity_cost # Return large value if goal is unreachable
                cost_for_package += 1  # load action
                cost_for_package += drive_cost  # drive action(s)
                cost_for_package += 1  # unload action

            else:
                # Package location unknown? Should not happen in valid state.
                # Treat as unreachable.
                return self.infinity_cost

            total_cost += cost_for_package

        # If total_cost somehow accumulated infinity_cost from a package, return infinity_cost
        if total_cost >= self.infinity_cost:
             return self.infinity_cost

        return total_cost
