from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(road l1 l2)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class TransportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions needed to transport all packages
    to their goal locations. It considers:
    - The distance packages need to be moved (using a simple count of required road segments)
    - Whether packages need to be picked up or dropped
    - Vehicle capacity constraints

    # Assumptions:
    - Each package transport requires at least one pick-up and one drop action.
    - Vehicles may need to move between locations to pick up or deliver packages.
    - The road network is bidirectional (if (road l1 l2) exists, movement is possible both ways).
    - Vehicle capacity is handled by tracking available capacity levels.

    # Heuristic Initialization
    - Extract goal locations for each package.
    - Build a road graph from static facts for pathfinding.
    - Extract capacity predecessor relationships.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each package not at its goal location:
        a) If package is not in a vehicle:
            - Find the closest vehicle that can carry it (considering capacity)
            - Estimate moves needed for vehicle to reach package
            - Add pick-up action
        b) If package is in a vehicle:
            - Check if vehicle is at goal location (just needs drop)
            - If not, estimate moves needed to reach goal location
            - Add drop action
    2. Sum all estimated actions:
        - Each road segment traversal counts as 1 action
        - Each pick-up/drop counts as 1 action
    3. For vehicles carrying multiple packages:
        - Prioritize dropping packages at closer locations first
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        # Build road graph (bidirectional)
        self.road_graph = {}
        for fact in self.static:
            if match(fact, "road", "*", "*"):
                _, l1, l2 = get_parts(fact)
                self.road_graph.setdefault(l1, set()).add(l2)
                self.road_graph.setdefault(l2, set()).add(l1)

        # Extract capacity predecessor relationships
        self.capacity_pred = {}
        for fact in self.static:
            if match(fact, "capacity-predecessor", "*", "*"):
                _, s1, s2 = get_parts(fact)
                self.capacity_pred[s2] = s1

        # Store goal locations for each package
        self.goal_locations = {}
        for goal in self.goals:
            if match(goal, "at", "*", "*"):
                _, package, location = get_parts(goal)
                self.goal_locations[package] = location

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        total_cost = 0

        # Track package locations and vehicle states
        package_locations = {}
        vehicle_locations = {}
        vehicle_capacities = {}
        packages_in_vehicles = {}

        for fact in state:
            parts = get_parts(fact)
            if match(fact, "at", "*", "*"):
                obj, loc = parts[1], parts[2]
                if obj.startswith("p"):  # Package
                    package_locations[obj] = loc
                else:  # Vehicle
                    vehicle_locations[obj] = loc
            elif match(fact, "in", "*", "*"):
                pkg, vehicle = parts[1], parts[2]
                packages_in_vehicles.setdefault(vehicle, set()).add(pkg)
            elif match(fact, "capacity", "*", "*"):
                vehicle, size = parts[1], parts[2]
                vehicle_capacities[vehicle] = size

        def estimate_distance(start, end):
            """Simple BFS to estimate distance between two locations."""
            if start == end:
                return 0
            visited = {start}
            queue = [(start, 0)]
            while queue:
                loc, dist = queue.pop(0)
                for neighbor in self.road_graph.get(loc, []):
                    if neighbor == end:
                        return dist + 1
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, dist + 1))
            return float('inf')  # No path found

        for package, goal_loc in self.goal_locations.items():
            current_loc = package_locations.get(package, None)
            
            # Package is already at goal
            if current_loc == goal_loc:
                continue
                
            # Package is in a vehicle
            if package in packages_in_vehicles.get(vehicle, set()):
                vehicle = [v for v, pkgs in packages_in_vehicles.items() 
                          if package in pkgs][0]
                vehicle_loc = vehicle_locations[vehicle]
                
                if vehicle_loc == goal_loc:
                    # Just need to drop
                    total_cost += 1
                else:
                    # Need to move and drop
                    dist = estimate_distance(vehicle_loc, goal_loc)
                    if dist != float('inf'):
                        total_cost += dist + 1  # move actions + drop
            else:
                # Package is not in a vehicle - need to pick up
                # Find closest vehicle that can carry it
                min_cost = float('inf')
                for vehicle, capacity in vehicle_capacities.items():
                    if capacity != "c0":  # Vehicle has some capacity
                        vehicle_loc = vehicle_locations[vehicle]
                        dist = estimate_distance(vehicle_loc, current_loc)
                        if dist + 1 < min_cost:  # +1 for pick-up
                            min_cost = dist + 1
                
                if min_cost != float('inf'):
                    total_cost += min_cost + 1  # pick-up + move to goal + drop
                else:
                    # No vehicle can reach package - use fallback
                    total_cost += 2  # minimum pick-up and drop

        return total_cost
