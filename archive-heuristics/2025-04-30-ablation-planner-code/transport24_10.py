# Assuming this import works in the target environment
from heuristics.heuristic_base import Heuristic

from fnmatch import fnmatch
from collections import deque

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact strings or malformed facts defensively
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
    This heuristic estimates the number of actions required to move each package
    to its goal location independently. It sums the estimated costs for each
    package that is not yet at its destination. The cost for a package includes
    pick-up (if on the ground), driving the vehicle carrying it along the shortest
    path, and dropping it at the goal location.

    # Assumptions
    - Roads are bidirectional.
    - Vehicle capacity is sufficient for any package (capacity constraints are ignored in the heuristic calculation).
    - A suitable vehicle is always available when a package needs to be picked up or transported.
    - The cost of each action (drive, pick-up, drop) is 1.
    - The goal state is defined by the final locations of specific packages.

    # Heuristic Initialization
    - Extract the goal location for each package from the task's goal conditions.
    - Build a graph representing the road network from the static `road` facts.
    - Compute the shortest path distance between all pairs of locations using Breadth-First Search (BFS). These distances are stored for quick lookup during heuristic computation.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Extract the current location of every package and vehicle. A package can be on the ground at a location `L` (`(at package L)`) or inside a vehicle `V` (`(in package V)`), in which case its effective location is the location of vehicle `V` (`(at V L_v)`).
    2. Initialize the total heuristic cost to 0.
    3. For each package that has a specified goal location:
       a. Check if the package is already on the ground at its goal location (`(at package goal_loc)`). If yes, this package contributes 0 to the heuristic, and we move to the next package.
       b. If the package is not yet at its goal location, determine its current status:
          - If the package is on the ground at `L_curr` (`(at package L_curr)` where `L_curr != goal_loc`):
            - It needs to be picked up (1 action).
            - It needs to be transported from `L_curr` to `goal_loc`. The minimum number of drive actions required is the shortest path distance between `L_curr` and `goal_loc`.
            - It needs to be dropped at `goal_loc` (1 action).
            - Add `1 + distance(L_curr, goal_loc) + 1` to the total cost.
          - If the package is inside a vehicle `V` (`(in package V)`):
            - Find the current location `L_v` of vehicle `V` (`(at V L_v)`).
            - If `L_v == goal_loc`: The package is in the vehicle at the goal location. It only needs to be dropped (1 action). Add 1 to the total cost.
            - If `L_v != goal_loc`: The package needs to be transported from `L_v` to `goal_loc`. The minimum number of drive actions is the shortest path distance between `L_v` and `goal_loc`. It also needs to be dropped at `goal_loc` (1 action). Add `distance(L_v, goal_loc) + 1` to the total cost.
          - If the package's status is unknown (neither on ground nor in vehicle, or vehicle location unknown), the state might be unreachable or invalid; return infinity.
    4. If any required shortest path distance is infinite (goal location unreachable from the package's current effective location), the total heuristic cost is infinite.
    5. The final total cost is the heuristic value for the state.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # Heuristic Initialization

        # 1. Extract goal locations for packages
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at" and len(args) == 2:
                package, location = args
                self.goal_locations[package] = location
            # Assuming goals are only (at package location)

        # 2. Build road graph and compute shortest path distances
        self.road_graph = {}
        locations = set()
        for fact in static_facts:
            if match(fact, "road", "*", "*"):
                parts = get_parts(fact)
                if len(parts) == 3:
                    _, l1, l2 = parts
                    locations.add(l1)
                    locations.add(l2)
                    self.road_graph.setdefault(l1, set()).add(l2)
                    self.road_graph.setdefault(l2, set()).add(l1) # Assuming roads are bidirectional

        self.distances = {}
        for start_node in locations:
            q = deque([(start_node, 0)])
            visited = {start_node}
            self.distances[(start_node, start_node)] = 0

            while q:
                curr, dist = q.popleft()
                for neighbor in self.road_graph.get(curr, set()):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        self.distances[(start_node, neighbor)] = dist + 1
                        q.append((neighbor, dist + 1))

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Step-By-Step Thinking for Computing Heuristic

        # 1. Extract current locations of packages and vehicles from the state
        at_map = {} # Maps locatable (package or vehicle) to location
        in_map = {} # Maps package to vehicle
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == "at" and len(parts) == 3:
                _, obj, loc = parts
                at_map[obj] = loc
            elif predicate == "in" and len(parts) == 3:
                _, pkg, veh = parts
                in_map[pkg] = veh

        total_cost = 0

        # 3. For each package that has a specified goal location
        for package, goal_loc in self.goal_locations.items():
            # 3a. Check if the package is already on the ground at its goal location
            if at_map.get(package) == goal_loc:
                continue # Goal satisfied for this package, cost 0

            # Find current status of the package
            pkg_on_ground_loc = at_map.get(package) # Location if on ground
            pkg_in_vehicle = in_map.get(package) # Vehicle if in vehicle

            if pkg_on_ground_loc is not None: # Package is on the ground
                p_curr_loc = pkg_on_ground_loc
                # Needs pick-up, drive, drop
                dist = self.distances.get((p_curr_loc, goal_loc), float('inf'))
                if dist == float('inf'):
                    return float('inf') # Goal unreachable
                total_cost += 1 # pick-up
                total_cost += dist # drive
                total_cost += 1 # drop

            elif pkg_in_vehicle is not None: # Package is in a vehicle
                vehicle = pkg_in_vehicle
                v_curr_loc = at_map.get(vehicle) # Get the location of the vehicle

                if v_curr_loc is None:
                    # Vehicle location unknown - indicates an issue
                    return float('inf')

                # Package is in vehicle, needs to reach goal_loc and be dropped
                dist = self.distances.get((v_curr_loc, goal_loc), float('inf'))
                if dist == float('inf'):
                    return float('inf') # Goal unreachable

                # If vehicle is already at goal_loc, only drop is needed
                if v_curr_loc == goal_loc:
                     total_cost += 1 # drop
                else:
                    # Needs drive, drop
                    total_cost += dist # drive
                    total_cost += 1 # drop
            else:
                 # Package status unknown (not on ground, not in vehicle)
                 # This case should ideally not be reached in a valid state representation
                 return float('inf')

        # 5. The final total cost is the heuristic value for the state.
        return total_cost
