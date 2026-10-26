from collections import deque
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact strings or malformed facts gracefully
    if not fact or not isinstance(fact, str) or len(fact) < 2 or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()

class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    Estimates the number of actions needed to move packages to their goal locations.
    This heuristic ignores vehicle capacity and assumes any package can be picked
    up and transported by an available vehicle. It sums the minimum actions
    (pick-up, drive segments, drop) required for each package independently.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and building
        the road network graph to compute shortest path distances.
        """
        self.goals = task.goals
        static_facts = task.static

        # 1. Extract goal locations for each package
        self.goal_locations = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == 'at':
                # Assuming goal is always (at package location)
                if len(parts) == 3:
                    package, location = parts[1], parts[2]
                    self.goal_locations[package] = location
                # else: malformed goal fact, ignore or handle error

        # 2. Build road network graph and compute all-pairs shortest paths
        self.road_graph = {}
        locations = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == 'road':
                if len(parts) == 3:
                    l1, l2 = parts[1], parts[2]
                    self.road_graph.setdefault(l1, []).append(l2)
                    self.road_graph.setdefault(l2, []).append(l1) # Assume roads are bidirectional
                    locations.add(l1)
                    locations.add(l2)
                # else: malformed road fact, ignore

        # Add locations from goals to ensure they are included in distance map calculation
        for loc in self.goal_locations.values():
             locations.add(loc)

        self.distance_map = {}
        for start_loc in locations:
            self.distance_map[start_loc] = {}
            q = deque([(start_loc, 0)])
            visited = {start_loc}
            self.distance_map[start_loc][start_loc] = 0

            while q:
                curr_loc, d = q.popleft()

                # Handle locations that might be in 'locations' set but not in road_graph (isolated)
                if curr_loc not in self.road_graph:
                    continue

                for neighbor in self.road_graph[curr_loc]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        self.distance_map[start_loc][neighbor] = d + 1
                        q.append((neighbor, d + 1))

    def get_distance(self, l1, l2):
        """
        Retrieves the precomputed shortest distance between two locations.
        Returns float('inf') if locations are not in the map or unreachable.
        """
        # Check if l1 is a known location and l2 is reachable from l1
        if l1 in self.distance_map and l2 in self.distance_map[l1]:
            return self.distance_map[l1][l2]
        else:
            # If l1 or l2 are not in the set of locations encountered in static facts/goals,
            # or if l2 is simply unreachable from l1 within the road network.
            # This implies the problem might be unsolvable or the state is invalid
            # w.r.t. the road network. Return infinity.
            return float('inf')


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions
        to reach the goal state from the current state.
        """
        state = node.state

        # Check if goal is already reached
        if self.goals <= state:
             return 0

        # Track current status of packages and vehicles
        current_package_status = {} # package -> location or vehicle
        current_vehicle_location = {} # vehicle -> location

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == 'at' and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                # Assuming object names starting with 'p' are packages and 'v' are vehicles
                # This is a domain-specific assumption based on examples.
                if obj.startswith('p'):
                    current_package_status[obj] = loc
                elif obj.startswith('v'):
                    current_vehicle_location[obj] = loc
                # Ignore other 'at' facts if any (e.g., at-robby in gripper)
            elif predicate == 'in' and len(parts) == 3:
                package, vehicle = parts[1], parts[2]
                # Assuming first object is package, second is vehicle
                # This is a domain-specific assumption based on examples.
                if package.startswith('p') and vehicle.startswith('v'):
                    current_package_status[package] = vehicle # Store the vehicle name
                # Ignore other 'in' facts if any

        total_cost = 0

        # Calculate cost for each package not at its goal
        for package, goal_location in self.goal_locations.items():
            # If package is not mentioned in the current state facts,
            # its location is unknown. This should not happen in a well-formed state.
            # Treat as unsolvable from this state.
            if package not in current_package_status:
                 # print(f"Warning: Goal package {package} not found in state facts.")
                 return float('inf')

            # If the goal fact (at package goal_location) is already true, cost is 0 for this package.
            if f'(at {package} {goal_location})' in state:
                 continue # This package goal is met, cost is 0 for this package.

            # Package is not at its goal location. Calculate cost.
            current_status = current_package_status[package]

            if current_status.startswith('p'): # Package is at a location (on the ground)
                loc_p_current = current_status
                # Cost = 1 (pick) + dist(loc_p_current, goal_location) (drive) + 1 (drop)
                drive_cost = self.get_distance(loc_p_current, goal_location)
                if drive_cost == float('inf'):
                    return float('inf') # Goal location unreachable
                total_cost += 1 + drive_cost + 1

            elif current_status.startswith('v'): # Package is in a vehicle
                vehicle = current_status
                if vehicle not in current_vehicle_location:
                    # Vehicle location unknown - cannot transport package
                    # print(f"Warning: Package {package} is in vehicle {vehicle}, but vehicle location is unknown.")
                    return float('inf')

                loc_v = current_vehicle_location[vehicle]
                # Cost = dist(loc_v, goal_location) (drive) + 1 (drop)
                drive_cost = self.get_distance(loc_v, goal_location)
                if drive_cost == float('inf'):
                    return float('inf') # Goal location unreachable
                total_cost += drive_cost + 1
            else:
                 # Unknown status format? Malformed state?
                 # print(f"Warning: Package {package} has unknown status format: {current_status}")
                 return float('inf')

        return total_cost
