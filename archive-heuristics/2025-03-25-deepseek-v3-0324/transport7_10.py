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
    - Whether vehicles need to pick up/drop packages
    - Vehicle capacity constraints

    # Assumptions:
    - The road network is bidirectional (if (road l1 l2) exists, we can travel both ways)
    - Each package needs exactly one vehicle to transport it
    - Vehicles can carry multiple packages if capacity allows
    - The heuristic doesn't need to be admissible (can overestimate)

    # Heuristic Initialization
    - Extract goal locations for each package
    - Build a graph representation of the road network
    - Extract capacity information for vehicles
    - Store capacity predecessor relationships

    # Step-By-Step Thinking for Computing Heuristic
    1. For each package not at its goal:
        a) Find its current location (either directly or via the vehicle it's in)
        b) Calculate the shortest path distance from current to goal location
        c) Add the distance to the total cost (each step is a drive action)
    2. For each package not at its goal:
        a) If not in a vehicle, add 1 for pick-up action
        b) Add 1 for eventual drop action
    3. For vehicles:
        a) If a vehicle needs to move to pick up a package, add the distance
        b) Consider capacity constraints when multiple packages need transport
    4. Return the sum of all these costs
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
        
        # Precompute shortest paths between all locations
        self.shortest_paths = {}
        locations = set(self.road_graph.keys())
        for loc in locations:
            self.shortest_paths[loc] = self._bfs(loc)
    
    def _bfs(self, start):
        """BFS to compute shortest paths from start location to all others."""
        visited = {start: 0}
        queue = [start]
        
        while queue:
            current = queue.pop(0)
            for neighbor in self.road_graph.get(current, []):
                if neighbor not in visited:
                    visited[neighbor] = visited[current] + 1
                    queue.append(neighbor)
        return visited
    
    def _get_distance(self, loc1, loc2):
        """Get precomputed shortest path distance between two locations."""
        return self.shortest_paths.get(loc1, {}).get(loc2, float('inf'))
    
    def __call__(self, node):
        """Compute heuristic estimate for the given state."""
        state = node.state
        
        # If it's a goal state, return 0
        if self.goals <= state:
            return 0
        
        # Track package locations and vehicle capacities
        package_locations = {}
        vehicle_locations = {}
        vehicle_capacities = {}
        in_vehicle = {}  # package -> vehicle
        
        for fact in state:
            parts = get_parts(fact)
            if match(fact, "at", "*", "*"):
                obj, loc = parts[1], parts[2]
                if obj.startswith('p'):  # package
                    package_locations[obj] = loc
                else:  # vehicle
                    vehicle_locations[obj] = loc
            elif match(fact, "in", "*", "*"):
                package, vehicle = parts[1], parts[2]
                in_vehicle[package] = vehicle
            elif match(fact, "capacity", "*", "*"):
                vehicle, size = parts[1], parts[2]
                vehicle_capacities[vehicle] = size
        
        total_cost = 0
        
        # Process each package not at its goal
        for package, goal_loc in self.goal_locations.items():
            if package not in package_locations and package not in in_vehicle:
                continue  # package not in state (shouldn't happen)
                
            # Get current location (either directly or via vehicle)
            if package in in_vehicle:
                vehicle = in_vehicle[package]
                current_loc = vehicle_locations[vehicle]
                # Add cost for drop action
                total_cost += 1
            else:
                current_loc = package_locations[package]
                # Add cost for pick-up action (assuming we'll need to pick it up)
                total_cost += 1
            
            # Add driving distance from current to goal location
            distance = self._get_distance(current_loc, goal_loc)
            if distance == float('inf'):
                return float('inf')  # unreachable
            total_cost += distance
        
        # Estimate vehicle movement costs for picking up packages
        # This is simplified - we assume each vehicle might need to move to pick up packages
        for vehicle, current_loc in vehicle_locations.items():
            # Find packages that might need this vehicle
            # This is a rough estimate - we assume the vehicle might need to move
            # to pick up at least one package
            total_cost += 1  # small cost for potential vehicle movement
        
        return total_cost
