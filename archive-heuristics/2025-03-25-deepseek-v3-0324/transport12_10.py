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
    - The distance packages need to be moved (using road connections)
    - Whether packages need to be picked up or dropped
    - Vehicle capacity constraints

    # Assumptions:
    - All packages must be transported to their goal locations.
    - Vehicles can carry multiple packages if they have sufficient capacity.
    - Road connections are bidirectional (though this is handled via static facts).
    - The heuristic doesn't need to be admissible (can overestimate).

    # Heuristic Initialization
    - Extract goal locations for each package from task.goals
    - Extract road connections from static facts
    - Extract capacity information from static facts

    # Step-By-Step Thinking for Computing Heuristic
    1. For each package not at its goal location:
        a. If package is in a vehicle:
            - Add cost for dropping it at current location (if needed)
            - Consider vehicle's current location as package's location
        b. Find shortest path from current location to goal location using roads
        c. Add driving actions needed (distance * 1 per step)
        d. Add pick-up and drop actions if needed (2 actions per package)
    2. For vehicles:
        a. If vehicle needs to move to pick up a package, add driving distance
        b. Consider capacity constraints when multiple packages need transport
    3. Sum all actions to get total heuristic estimate
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static
        
        # Extract goal locations for packages
        self.goal_locations = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "at" and parts[1].startswith("p"):
                self.goal_locations[parts[1]] = parts[2]
        
        # Build road graph (bidirectional)
        self.road_graph = {}
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "road":
                l1, l2 = parts[1], parts[2]
                self.road_graph.setdefault(l1, set()).add(l2)
                self.road_graph.setdefault(l2, set()).add(l1)
        
        # Extract capacity predecessor relationships
        self.capacity_pred = {}
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "capacity-predecessor":
                self.capacity_pred[parts[2]] = parts[1]

    def _bfs_distance(self, start, goal):
        """Calculate shortest path distance between two locations using BFS."""
        if start == goal:
            return 0
            
        visited = set()
        queue = [(start, 0)]
        
        while queue:
            current, dist = queue.pop(0)
            if current == goal:
                return dist
            if current in visited:
                continue
            visited.add(current)
            
            for neighbor in self.road_graph.get(current, []):
                queue.append((neighbor, dist + 1))
        
        # If no path found (shouldn't happen in valid problems)
        return float('inf')

    def __call__(self, node):
        """Compute heuristic estimate for given state."""
        state = node.state
        
        # Track package locations and vehicle capacities
        package_locations = {}
        vehicle_locations = {}
        vehicle_capacities = {}
        packages_in_vehicles = {}
        
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at":
                obj, loc = parts[1], parts[2]
                if obj.startswith("p"):
                    package_locations[obj] = loc
                elif obj.startswith("v"):
                    vehicle_locations[obj] = loc
            elif parts[0] == "in":
                pkg, veh = parts[1], parts[2]
                packages_in_vehicles.setdefault(veh, set()).add(pkg)
            elif parts[0] == "capacity":
                veh, cap = parts[1], parts[2]
                vehicle_capacities[veh] = cap
        
        total_cost = 0
        
        # Process each package not at its goal
        for pkg, goal_loc in self.goal_locations.items():
            current_loc = package_locations.get(pkg, None)
            
            # If package is in a vehicle, we need to drop it first
            in_vehicle = False
            for veh, pkgs in packages_in_vehicles.items():
                if pkg in pkgs:
                    in_vehicle = True
                    current_loc = vehicle_locations[veh]
                    # Add cost for dropping (1 action)
                    total_cost += 1
                    break
            
            if current_loc != goal_loc:
                # Add driving distance (1 per road segment)
                distance = self._bfs_distance(current_loc, goal_loc)
                total_cost += distance
                
                # Add pick-up action (unless already in vehicle)
                if not in_vehicle:
                    total_cost += 1
        
        # Estimate vehicle movements for packages not yet picked up
        for pkg in package_locations:
            if pkg not in self.goal_locations:
                continue  # Package doesn't need to be moved
                
            if pkg in packages_in_vehicles.get(any, []):
                continue  # Already in a vehicle
                
            # Find closest vehicle that can carry this package
            pkg_loc = package_locations[pkg]
            min_dist = float('inf')
            
            for veh, veh_loc in vehicle_locations.items():
                # Check if vehicle has capacity
                current_cap = vehicle_capacities.get(veh, None)
                if current_cap is None or current_cap == "c0":
                    continue  # No capacity
                    
                # Calculate distance from vehicle to package
                dist = self._bfs_distance(veh_loc, pkg_loc)
                if dist < min_dist:
                    min_dist = dist
            
            if min_dist != float('inf'):
                total_cost += min_dist
        
        return total_cost
