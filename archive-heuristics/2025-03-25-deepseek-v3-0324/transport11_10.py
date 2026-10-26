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
    - The distance packages need to be moved (using a simple graph distance)
    - Whether vehicles need to pick up/drop off packages
    - Vehicle capacity constraints

    # Assumptions:
    - The road network is bidirectional (if (road l1 l2) exists, we can travel both ways)
    - Each package needs exactly one vehicle to transport it
    - Vehicles can carry multiple packages if they have capacity
    - The heuristic doesn't need to be admissible (can overestimate)

    # Heuristic Initialization
    - Extract goal locations for each package
    - Build a graph representation of the road network
    - Extract capacity information for vehicles

    # Step-By-Step Thinking for Computing Heuristic
    1. For each package not at its goal:
        a) If package is not in a vehicle:
            - Find nearest vehicle that can carry it (considering capacity)
            - Add cost to pick up package (1 action)
        b) Calculate distance from current location to goal location
        c) Add driving cost (distance * 1 action per step)
        d) Add cost to drop package (1 action)
    2. For vehicles:
        - If vehicle needs to move to pick up packages, add that movement cost
        - If vehicle is carrying packages, add cost to drop them at goals
    3. Sum all these costs to get the total heuristic estimate
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
        
        # Store goal locations for packages
        self.goal_locations = {}
        for goal in self.goals:
            if match(goal, "at", "*", "*"):
                _, package, location = get_parts(goal)
                self.goal_locations[package] = location
        
        # Extract capacity predecessor relationships
        self.capacity_pred = {}
        for fact in self.static:
            if match(fact, "capacity-predecessor", "*", "*"):
                _, s1, s2 = get_parts(fact)
                self.capacity_pred[s2] = s1

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
        
        # If no path found, return large number (problem might be unsolvable)
        return float('inf')

    def __call__(self, node):
        """Compute heuristic estimate for given state."""
        state = node.state
        
        # Check if goal is already reached
        if all(goal in state for goal in self.goals):
            return 0
            
        # Track package and vehicle information
        package_locations = {}
        vehicle_locations = {}
        vehicle_capacities = {}
        packages_in_vehicles = {}
        
        for fact in state:
            parts = get_parts(fact)
            if match(fact, "at", "*", "*"):
                obj, loc = parts[1], parts[2]
                if obj.startswith('p'):  # Package
                    package_locations[obj] = loc
                else:  # Vehicle
                    vehicle_locations[obj] = loc
            elif match(fact, "in", "*", "*"):
                pkg, veh = parts[1], parts[2]
                packages_in_vehicles.setdefault(veh, set()).add(pkg)
            elif match(fact, "capacity", "*", "*"):
                veh, cap = parts[1], parts[2]
                vehicle_capacities[veh] = cap
        
        total_cost = 0
        
        # Process each package not at its goal
        for pkg, goal_loc in self.goal_locations.items():
            current_loc = package_locations.get(pkg, None)
            
            # If package is already at goal, skip
            if current_loc == goal_loc:
                continue
                
            # If package is in a vehicle
            in_vehicle = False
            carrying_vehicle = None
            for veh, pkgs in packages_in_vehicles.items():
                if pkg in pkgs:
                    in_vehicle = True
                    carrying_vehicle = veh
                    break
            
            if in_vehicle:
                # Vehicle needs to drive to goal location
                veh_loc = vehicle_locations[carrying_vehicle]
                dist = self._bfs_distance(veh_loc, goal_loc)
                total_cost += dist + 1  # drive + drop
            else:
                # Package needs to be picked up first
                # Find nearest vehicle that can carry it
                min_pickup_cost = float('inf')
                
                for veh, veh_loc in vehicle_locations.items():
                    # Check if vehicle has capacity
                    current_cap = vehicle_capacities.get(veh, None)
                    if current_cap is None:
                        continue  # vehicle has no capacity (shouldn't happen)
                        
                    # Check if vehicle can carry another package
                    # (current capacity is not c0)
                    if current_cap != 'c0':
                        dist = self._bfs_distance(veh_loc, current_loc)
                        if dist < min_pickup_cost:
                            min_pickup_cost = dist
                
                if min_pickup_cost == float('inf'):
                    # No vehicle can pick up this package
                    return float('inf')
                    
                # Add cost to pick up and drive to goal
                dist_to_goal = self._bfs_distance(current_loc, goal_loc)
                total_cost += min_pickup_cost + 1 + dist_to_goal + 1  # drive to package + pick up + drive to goal + drop
        
        return total_cost
