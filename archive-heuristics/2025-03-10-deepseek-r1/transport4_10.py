from fnmatch import fnmatch
from collections import defaultdict, deque
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    return fact[1:-1].split()

def match(fact, *args):
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class transport4Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions required to move all packages to their goal locations. It considers the drive actions needed for vehicles to reach packages and their destinations, as well as the pick-up and drop actions.

    # Assumptions
    - Each package can be transported by any available vehicle.
    - Vehicles can adjust their capacity as needed (ignores capacity constraints for simplicity).
    - Roads are directed, and shortest paths are precomputed for drive actions.

    # Heuristic Initialization
    - Extract goal locations for each package from the task's goals.
    - Build a road graph from static road predicates.
    - Precompute shortest paths between all pairs of locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each package:
        a. If the package is already at its goal, cost is 0.
        b. If the package is in a vehicle:
            i. Add drive steps from the vehicle's current location to the goal.
            ii. Add 1 action for dropping the package.
        c. If the package is not in a vehicle:
            i. Find the closest vehicle (minimal drive steps to the package's location).
            ii. Add drive steps for the vehicle to reach the package.
            iii. Add 1 action for picking up the package.
            iv. Add drive steps from the package's location to the goal.
            v. Add 1 action for dropping the package.
    2. Sum the costs for all packages to get the total heuristic estimate.
    """

    def __init__(self, task):
        """Initialize the heuristic with goals, road graph, and precomputed distances."""
        self.goal_locations = {}
        # Extract goal locations for each package
        for goal in task.goals:
            parts = get_parts(goal)
            if parts[0] == 'at' and len(parts) == 3:
                package = parts[1]
                location = parts[2]
                self.goal_locations[package] = location

        # Build road graph from static facts
        self.road_map = defaultdict(list)
        for fact in task.static:
            if match(fact, 'road', '*', '*'):
                parts = get_parts(fact)
                l1, l2 = parts[1], parts[2]
                self.road_map[l1].append(l2)

        # Precompute shortest paths between all locations
        self.distances = {}
        locations = set()
        for l1 in self.road_map:
            locations.add(l1)
            for l2 in self.road_map[l1]:
                locations.add(l2)
        locations = list(locations)

        for start in locations:
            queue = deque([(start, 0)])
            visited = {start: 0}
            while queue:
                current, dist = queue.popleft()
                for neighbor in self.road_map.get(current, []):
                    if neighbor not in visited or visited[neighbor] > dist + 1:
                        visited[neighbor] = dist + 1
                        queue.append((neighbor, dist + 1))
            for end in locations:
                self.distances[(start, end)] = visited.get(end, float('inf'))

    def __call__(self, node):
        """Compute the heuristic estimate for the given state."""
        state = node.state
        total_cost = 0

        # Collect all vehicles and their locations
        vehicles = set()
        for fact in state:
            if match(fact, 'capacity', '*', '*'):
                parts = get_parts(fact)
                vehicles.add(parts[1])
        vehicle_locs = {}
        for fact in state:
            if match(fact, 'at', '*', '*'):
                parts = get_parts(fact)
                obj, loc = parts[1], parts[2]
                if obj in vehicles:
                    vehicle_locs[obj] = loc

        for package, goal_loc in self.goal_locations.items():
            current_loc = None
            in_vehicle = False

            # Check if package is in a vehicle
            for fact in state:
                if match(fact, 'in', package, '*'):
                    vehicle = get_parts(fact)[2]
                    current_loc = vehicle_locs.get(vehicle, None)
                    in_vehicle = True
                    break
            if not in_vehicle:
                # Find package's current location
                for fact in state:
                    if match(fact, 'at', package, '*'):
                        current_loc = get_parts(fact)[2]
                        break

            if current_loc == goal_loc:
                continue

            if in_vehicle:
                # Drive from current_loc to goal and drop
                drive_steps = self.distances.get((current_loc, goal_loc), float('inf'))
                total_cost += drive_steps + 1 if drive_steps != float('inf') else 1000
            else:
                # Find closest vehicle to package's location
                min_drive = float('inf')
                for v_loc in vehicle_locs.values():
                    drive = self.distances.get((v_loc, current_loc), float('inf'))
                    if drive < min_drive:
                        min_drive = drive
                if min_drive == float('inf'):
                    min_drive = 1000  # Penalize unreachable

                # Drive from package's location to goal
                drive_goal = self.distances.get((current_loc, goal_loc), float('inf'))
                if drive_goal == float('inf'):
                    drive_goal = 1000

                total_cost += min_drive + 1 + drive_goal + 1

        return total_cost
