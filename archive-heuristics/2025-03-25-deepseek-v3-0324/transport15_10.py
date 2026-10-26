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
    - The distance packages need to be transported (using road network distances)
    - Whether packages need to be picked up or dropped
    - Vehicle capacity constraints

    # Assumptions:
    - Packages can be transported by any vehicle with sufficient capacity
    - The road network is bidirectional (if (road l1 l2) exists, we can travel both ways)
    - Vehicles can carry multiple packages if their capacity allows
    - The heuristic doesn't need to be admissible (can overestimate)

    # Heuristic Initialization
    - Extract goal locations for each package
    - Build a graph representation of the road network
    - Extract capacity information for vehicles
    - Store capacity predecessor relationships

    # Step-By-Step Thinking for Computing Heuristic
    1. For each package not at its goal location:
        a) If the package is in a vehicle:
            - Add cost for dropping it at current vehicle location
            - Treat it as being at the vehicle's location
        b) Find the shortest path from current location to goal location
        c) Estimate drive actions needed (path length)
        d) Add pick-up and drop-off actions (2 actions per package)
    2. For vehicles:
        a) If a vehicle needs to transport multiple packages, only count drive actions once
        b) Consider capacity constraints when estimating pick-up/drop-off sequences
    3. Sum all estimated actions:
        - Drive actions (based on road distances)
        - Pick-up actions (1 per package)
        - Drop actions (1 per package)
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static
        
        # Extract goal locations for packages
        self.goal_locations = {}
        for goal in self.goals:
            if match(goal, "at", "*", "*"):
                _, package, location = get_parts(goal)
                self.goal_locations[package] = location
        
        # Build road network graph
        self.road_graph = {}
        for fact in self.static:
            if match(fact, "road", "*", "*"):
                _, l1, l2 = get_parts(fact)
                self.road_graph.setdefault(l1, set()).add(l2)
                self.road_graph.setdefault(l2, set()).add(l1)  # roads are bidirectional
        
        # Extract capacity predecessor relationships
        self.capacity_pred = {}
        for fact in self.static:
            if match(fact, "capacity-predecessor", "*", "*"):
                _, s1, s2 = get_parts(fact)
                self.capacity_pred[s2] = s1

    def _shortest_path(self, start, goal):
        """Compute shortest path between two locations using BFS."""
        if start == goal:
            return 0
        
        visited = set()
        queue = [(start, 0)]
        
        while queue:
            current, distance = queue.pop(0)
            if current == goal:
                return distance
            if current in visited:
                continue
            visited.add(current)
            
            for neighbor in self.road_graph.get(current, []):
                if neighbor not in visited:
                    queue.append((neighbor, distance + 1))
        
        # If no path found, return a large number (problem might be unsolvable)
        return float('inf')

    def __call__(self, node):
        """Compute heuristic estimate for the given state."""
        state = node.state
        
        # If all goals are satisfied, return 0
        if all(goal in state for goal in self.goals):
            return 0
        
        # Track package locations and vehicle capacities
        package_locations = {}
        vehicle_locations = {}
        vehicle_capacities = {}
        packages_in_vehicles = {}
        
        for fact in state:
            parts = get_parts(fact)
            if match(fact, "at", "*", "*"):
                obj, loc = parts[1], parts[2]
                if obj.startswith("p"):  # package
                    package_locations[obj] = loc
                else:  # vehicle
                    vehicle_locations[obj] = loc
            elif match(fact, "in", "*", "*"):
                pkg, vehicle = parts[1], parts[2]
                packages_in_vehicles.setdefault(vehicle, set()).add(pkg)
            elif match(fact, "capacity", "*", "*"):
                vehicle, size = parts[1], parts[2]
                vehicle_capacities[vehicle] = size
        
        total_cost = 0
        
        # Process each package that's not at its goal
        for package, goal_loc in self.goal_locations.items():
            # Get current location
            if package in packages_in_vehicles:
                # Package is in a vehicle - need to drop it first
                for vehicle, packages in packages_in_vehicles.items():
                    if package in packages:
                        current_loc = vehicle_locations[vehicle]
                        total_cost += 1  # drop action
                        break
            else:
                current_loc = package_locations.get(package, None)
                if not current_loc:
                    continue  # package location unknown
            
            if current_loc == goal_loc:
                continue  # already at goal
                
            # Estimate drive distance
            distance = self._shortest_path(current_loc, goal_loc)
            if distance == float('inf'):
                return float('inf')  # unreachable goal
            
            total_cost += distance  # drive actions
            total_cost += 1  # pick-up action
            total_cost += 1  # drop action
        
        return total_cost
