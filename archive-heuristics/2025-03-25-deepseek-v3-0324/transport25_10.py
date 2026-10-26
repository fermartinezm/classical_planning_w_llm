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
    - The road network is bidirectional (if (road l1 l2) exists, movement is possible both ways)
    - Packages can only be transported by vehicles (no direct movement)
    - Each vehicle can carry only one package at a time (based on capacity constraints)
    - The heuristic doesn't need to be admissible (can overestimate)

    # Heuristic Initialization
    - Extract goal locations for each package
    - Build a graph representation of the road network
    - Extract capacity information for vehicles

    # Step-By-Step Thinking for Computing Heuristic
    1. For each package not at its goal location:
        a) If package is not in a vehicle:
            - Find nearest vehicle that can carry it (based on capacity)
            - Estimate distance for vehicle to reach package
            - Add pick-up action
        b) If package is in a vehicle:
            - Check if vehicle is at package's goal location
            - If not, estimate distance to goal location
            - Add drop-off action
    2. For each vehicle needed to transport packages:
        - Estimate driving distance to pick up and deliver packages
    3. Sum all estimated actions (pick-ups, drop-offs, and driving)
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

    def _bfs_distance(self, start, goal):
        """Compute shortest path distance between two locations using BFS."""
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
        """Compute heuristic estimate for the given state."""
        state = node.state

        # Check if goal is already reached
        if all(goal in state for goal in self.goals):
            return 0

        total_cost = 0
        package_info = {}
        vehicle_info = {}

        # Extract current state information
        for fact in state:
            parts = get_parts(fact)
            if match(fact, "at", "*", "*"):
                obj, loc = parts[1], parts[2]
                if obj.startswith('p'):  # Package
                    package_info[obj] = {'location': loc, 'in_vehicle': False}
                else:  # Vehicle
                    vehicle_info[obj] = {'location': loc, 'capacity': None}
            elif match(fact, "in", "*", "*"):
                package, vehicle = parts[1], parts[2]
                package_info[package] = {'location': vehicle, 'in_vehicle': True}
            elif match(fact, "capacity", "*", "*"):
                vehicle, capacity = parts[1], parts[2]
                if vehicle in vehicle_info:
                    vehicle_info[vehicle]['capacity'] = capacity

        # Process each package not at its goal
        for package, info in package_info.items():
            if package not in self.goal_locations:
                continue  # Package has no goal (shouldn't happen in valid tasks)
                
            current_loc = info['location']
            goal_loc = self.goal_locations[package]
            
            if current_loc == goal_loc:
                continue  # Package already at goal
                
            if info['in_vehicle']:
                # Package is in a vehicle - need to drop it at goal
                vehicle = current_loc
                vehicle_loc = vehicle_info[vehicle]['location']
                
                # Add cost for driving to goal (if not already there)
                if vehicle_loc != goal_loc:
                    dist = self._bfs_distance(vehicle_loc, goal_loc)
                    total_cost += dist
                
                # Add drop action cost
                total_cost += 1
            else:
                # Package is not in vehicle - need to pick it up
                package_loc = current_loc
                
                # Find nearest vehicle that can carry it
                min_dist = float('inf')
                best_vehicle = None
                
                for vehicle, v_info in vehicle_info.items():
                    # Check if vehicle has capacity to carry package
                    if v_info['capacity'] is not None and v_info['capacity'] != 'c0':
                        # Compute distance from vehicle to package
                        dist = self._bfs_distance(v_info['location'], package_loc)
                        if dist < min_dist:
                            min_dist = dist
                            best_vehicle = vehicle
                
                if best_vehicle is not None:
                    # Add driving cost to package
                    total_cost += min_dist
                    
                    # Add pick-up action cost
                    total_cost += 1
                    
                    # Add driving cost from package to goal
                    goal_dist = self._bfs_distance(package_loc, goal_loc)
                    total_cost += goal_dist
                    
                    # Add drop action cost
                    total_cost += 1
                else:
                    # No available vehicle - problem might be unsolvable
                    return float('inf')

        return total_cost
