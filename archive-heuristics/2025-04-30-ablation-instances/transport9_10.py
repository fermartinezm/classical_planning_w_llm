from collections import deque
from fnmatch import fnmatch
# Assuming Heuristic base class is available in the environment
# from heuristics.heuristic_base import Heuristic

# Helper functions for parsing PDDL facts
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
    # Basic check for number of parts
    if len(parts) < len(args):
        return False
    # Check if each part matches the corresponding argument pattern
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# BFS implementation for shortest path on unweighted graph
def bfs_shortest_path(graph, start_node):
    """
    Computes shortest path distances from a start_node to all other nodes
    in an unweighted graph using BFS.
    Returns a dictionary {node: distance}.
    """
    distances = {node: float('inf') for node in graph}
    distances[start_node] = 0
    queue = deque([start_node])

    while queue:
        current = queue.popleft()

        # If current node has no outgoing edges, continue
        if current not in graph:
             continue

        for neighbor in graph[current]:
            if distances[neighbor] == float('inf'):
                distances[neighbor] = distances[current] + 1
                queue.append(neighbor)

    return distances

# The heuristic class
class transportHeuristic: # Inherit from Heuristic if base class is provided
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the total number of actions required to move all
    packages to their goal locations. It calculates the minimum actions needed
    for each package independently, considering its current location (on ground
    or in a vehicle) and the shortest path distance to its goal location.
    It ignores vehicle capacity constraints and the possibility of optimizing
    trips by carrying multiple packages.

    # Assumptions
    - Each package needs to reach a specific goal location on the ground.
    - Vehicles can move between locations connected by roads.
    - Shortest path distance between locations represents the minimum number of drive actions.
    - Vehicle capacity is ignored.
    - A vehicle is assumed to be available whenever needed to pick up or transport a package.
    - All actions (pick-up, drop, drive) have a cost of 1.
    - Road facts define bidirectional connections unless explicitly stated otherwise (standard assumption).
    - Packages are objects starting with 'p', vehicles with 'v', locations with 'l'.

    # Heuristic Initialization
    - Build a graph of locations based on the static `road` facts.
    - Compute all-pairs shortest path distances between all locations using BFS.
    - Extract the goal location for each package from the task's goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Check if the state is a goal state. If yes, the heuristic is 0.
    2. Identify the current location and status (on ground or in vehicle) for every package mentioned in the state. Also, identify the location of every vehicle. Store this information in dictionaries for quick lookup.
    3. Initialize total estimated cost to 0.
    4. For each goal fact `(at p l_goal)` specified in the task goals:
       a. If the fact `(at p l_goal)` is already true in the current state, this package is done; continue to the next goal fact.
       b. Otherwise, package `p` is not yet at its goal location `l_goal` on the ground.
       c. Determine package `p`'s effective current location (`l_current`) and whether it is inside a vehicle using the information gathered in step 2.
          - If package `p` is not found in the collected status (implying an invalid state or missing package), return infinity.
       d. Calculate the minimum actions needed for package `p` to reach `l_goal` on the ground from its effective current location `l_current`:
          i. If `l_current` is the same as `l_goal`: The package must be inside a vehicle (since `(at p l_goal)` is not true). It needs 1 'drop' action. Cost for this package = 1.
          ii. If `l_current` is different from `l_goal`:
              - If the package is on the ground at `l_current`, it needs 1 'pick-up' action. Add 1 to cost.
              - It needs to be transported from `l_current` to `l_goal`. Find the shortest path distance between `l_current` and `l_goal` using the precomputed distances. This distance is the minimum number of 'drive' actions required. Add this distance to cost. If no path exists (distance is infinity), return infinity for the total heuristic.
              - It needs 1 'drop' action at `l_goal`. Add 1 to cost.
              - Total cost for this package = (1 if on ground else 0) + distance(`l_current`, `l_goal`) + 1.
       e. Add the calculated cost for package `p` to the total estimated cost.
    5. Return the total estimated cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by building the location graph, computing
        distances, and extracting package goal locations.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # Build the location graph from road facts
        self.graph = {}
        locations = set()
        for fact in static_facts:
            if match(fact, "road", "?l1", "?l2"):
                l1, l2 = get_parts(fact)[1:]
                locations.add(l1)
                locations.add(l2)
                self.graph.setdefault(l1, []).append(l2)
                # Assuming roads are bidirectional unless specified otherwise
                self.graph.setdefault(l2, []).append(l1)

        # Add any locations mentioned in goals or initial state to the graph nodes,
        # even if they have no roads connected (though this might imply unsolvability).
        # Iterate through initial state facts to find locations.
        # Note: Accessing initial_state requires the Task object structure.
        # Assuming task object has initial_state attribute.
        # For robustness, we could iterate through all objects of type location
        # if that information were easily available, but parsing facts is simpler.
        # Let's just add locations from goals and roads for now.
        for goal in self.goals:
             if match(goal, "at", "?p", "?l"):
                 loc = get_parts(goal)[2]
                 locations.add(loc)

        # Add any locations found to the graph structure if not already present
        for loc in locations:
             self.graph.setdefault(loc, [])

        # Compute all-pairs shortest paths
        self.distance = {}
        for start_loc in self.graph:
            self.distance[start_loc] = bfs_shortest_path(self.graph, start_loc)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Check if the goal is reached (heuristic is 0)
        if self.goals <= state:
            return 0

        # Track current locations and status of packages and vehicles
        package_current_status = {} # {package: {'location': effective_loc, 'in_vehicle': vehicle_obj or None}}
        vehicle_location = {} # {vehicle: location}

        # First, find vehicle locations
        for fact in state:
            if match(fact, "at", "?v", "?l"):
                 obj, loc = get_parts(fact)[1:]
                 # Assuming objects starting with 'v' are vehicles
                 if obj.startswith('v'):
                     vehicle_location[obj] = loc

        # Then, find package locations/status
        for fact in state:
            if match(fact, "at", "?p", "?l"):
                obj, loc = get_parts(fact)[1:]
                # Assuming objects starting with 'p' are packages
                if obj.startswith('p'):
                    package_current_status[obj] = {'location': loc, 'in_vehicle': None}
            elif match(fact, "in", "?p", "?v"):
                pkg, veh = get_parts(fact)[1:]
                # Assuming objects starting with 'p' are packages and 'v' are vehicles
                if pkg.startswith('p') and veh.startswith('v'):
                    # Get the location of the vehicle the package is in
                    veh_loc = vehicle_location.get(veh)
                    if veh_loc is None:
                        # This implies an invalid state where a package is in a vehicle
                        # but the vehicle's location is unknown.
                        # Return infinity to indicate an issue or unsolvable path from here.
                        return float('inf')
                    package_current_status[pkg] = {'location': veh_loc, 'in_vehicle': veh}


        total_cost = 0  # Initialize action cost counter.

        # Iterate over the goal facts to find packages that need moving
        for goal_fact in self.goals:
            # Only consider (at package location) goals for this heuristic
            if not match(goal_fact, "at", "?p", "?l"):
                continue

            p, loc_goal = get_parts(goal_fact)[1:]

            # If the goal fact is already true, this package is done
            if goal_fact in state:
                continue

            # Package p is not yet at its goal location on the ground (loc_goal)

            current_status = package_current_status.get(p)

            if current_status is None:
                 # Package not found in 'at' or 'in' facts. Invalid state?
                 # Or maybe package doesn't exist? Assume valid states where
                 # packages relevant to goals are always 'at' or 'in' something.
                 # Return infinity to signal an issue.
                 return float('inf')

            current_loc_p = current_status['location']
            is_in_vehicle = current_status['in_vehicle'] is not None

            # Calculate cost for this package p
            cost_p = 0

            if current_loc_p == loc_goal:
                # Package is at the goal location, but inside a vehicle (since goal_fact not in state)
                if is_in_vehicle:
                    cost_p = 1 # Needs drop action
                else:
                    # This case implies current_loc_p == loc_goal, package is on ground (not in vehicle),
                    # but (at p loc_goal) is NOT in state. This is inconsistent with goal_fact not being in state.
                    # Return infinity.
                    return float('inf')
            else: # current_loc_p != loc_goal
                # Package needs to be moved from current_loc_p to loc_goal

                # If on ground, needs pick-up
                if not is_in_vehicle:
                    cost_p += 1 # pick-up action

                # Needs vehicle movement from current_loc_p to loc_goal
                # The distance is the number of drive actions
                if current_loc_p not in self.distance or loc_goal not in self.distance[current_loc_p]:
                     # No path exists between locations. Unsolvable problem or invalid state.
                     return float('inf')

                dist = self.distance[current_loc_p][loc_goal]
                if dist == float('inf'):
                     # No path exists
                     return float('inf')

                cost_p += dist # drive actions

                # Needs drop action at the goal location
                cost_p += 1 # drop action

            total_cost += cost_p

        # The heuristic is the sum of costs for all packages not yet at their goal location on the ground.
        return total_cost
