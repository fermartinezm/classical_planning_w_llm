from fnmatch import fnmatch
from collections import deque, defaultdict
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    return fact[1:-1].split()

def match(fact, *args):
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class transport12Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions required to move all packages to their goal locations. It considers the distance vehicles need to drive to pick up and deliver each package, as well as the necessary pick-up and drop actions. The heuristic also accounts for vehicle capacities to determine if a pick-up is possible.

    # Assumptions
    - Each package is transported by a single vehicle.
    - The shortest path between locations is used for drive actions.
    - Vehicles can only pick up packages if their current capacity allows it.
    - Static road and capacity-predecessor facts define the movement and capacity constraints.

    # Heuristic Initialization
    - Extract road network to build a graph for shortest path calculations.
    - Identify capacity-predecessor relationships to determine valid pick-ups.
    - Determine vehicles and packages from the initial state.
    - Record goal locations for each package.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each package in the goal:
        a. If the package is already at its goal, contribute 0.
        b. If the package is in a vehicle:
            i. Calculate the drive distance from the vehicle's current location to the goal.
            ii. Add 1 action for dropping the package.
        c. If the package is not in a vehicle:
            i. Find the closest vehicle that can pick it up (considering capacity and drive distance).
            ii. Calculate drive distance from the vehicle's location to the package's location.
            iii. Calculate drive distance from the package's location to the goal.
            iv. Add 2 actions for pick-up and drop.
    2. Sum the calculated actions for all packages.
    """

    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static

        # Build road graph
        self.road_map = defaultdict(list)
        for fact in self.static:
            if match(fact, 'road', '*', '*'):
                parts = get_parts(fact)
                l1, l2 = parts[1], parts[2]
                self.road_map[l1].append(l2)
                self.road_map[l2].append(l1)

        # Extract capacity-predecessor relationships
        self.capacity_pred = {}
        for fact in self.static:
            if match(fact, 'capacity-predecessor', '*', '*'):
                parts = get_parts(fact)
                s1, s2 = parts[1], parts[2]
                self.capacity_pred[s2] = s1

        # Identify vehicles and packages from initial state
        self.vehicles = set()
        self.packages = set()
        self.goal_locations = {}

        for fact in task.initial_state:
            parts = get_parts(fact)
            if parts[0] == 'capacity':
                self.vehicles.add(parts[1])
            elif parts[0] == 'at' and len(parts) == 3:
                obj = parts[1]
                if obj not in self.vehicles:
                    self.packages.add(obj)
            elif parts[0] == 'in':
                self.packages.add(parts[1])

        # Extract goal locations
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == 'at' and parts[1] in self.packages:
                self.goal_locations[parts[1]] = parts[2]

    def shortest_path_length(self, start, end):
        if start == end:
            return 0
        visited = set()
        queue = deque([(start, 0)])
        while queue:
            current, dist = queue.popleft()
            if current == end:
                return dist
            if current in visited:
                continue
            visited.add(current)
            for neighbor in self.road_map.get(current, []):
                if neighbor not in visited:
                    queue.append((neighbor, dist + 1))
        return float('inf')

    def __call__(self, node):
        state = node.state
        total_cost = 0

        for package, goal_loc in self.goal_locations.items():
            # Check if package is in a vehicle
            in_vehicle = None
            for fact in state:
                parts = get_parts(fact)
                if parts[0] == 'in' and parts[1] == package:
                    in_vehicle = parts[2]
                    break

            if in_vehicle:
                # Find vehicle's current location
                vehicle_loc = None
                for fact in state:
                    parts = get_parts(fact)
                    if parts[0] == 'at' and parts[1] == in_vehicle:
                        vehicle_loc = parts[2]
                        break
                if not vehicle_loc:
                    continue

                # Check if already at goal
                if vehicle_loc == goal_loc:
                    continue

                # Compute distance from vehicle to goal
                distance = self.shortest_path_length(vehicle_loc, goal_loc)
                total_cost += distance + 1  # drop action
                continue

            # Package is not in a vehicle
            current_package_loc = None
            for fact in state:
                parts = get_parts(fact)
                if parts[0] == 'at' and parts[1] == package:
                    current_package_loc = parts[2]
                    break
            if not current_package_loc or current_package_loc == goal_loc:
                continue

            min_cost = float('inf')
            for vehicle in self.vehicles:
                # Get vehicle's current location
                vehicle_loc = None
                for fact in state:
                    parts = get_parts(fact)
                    if parts[0] == 'at' and parts[1] == vehicle:
                        vehicle_loc = parts[2]
                        break
                if not vehicle_loc:
                    continue

                # Check vehicle's capacity
                vehicle_capacity = None
                for fact in state:
                    parts = get_parts(fact)
                    if parts[0] == 'capacity' and parts[1] == vehicle:
                        vehicle_capacity = parts[2]
                        break
                if not vehicle_capacity:
                    continue

                # Check if vehicle can pick up (has capacity predecessor)
                if vehicle_capacity not in self.capacity_pred:
                    continue

                # Compute distance from vehicle to package and package to goal
                distance_a = self.shortest_path_length(vehicle_loc, current_package_loc)
                distance_b = self.shortest_path_length(current_package_loc, goal_loc)
                if distance_a == float('inf') or distance_b == float('inf'):
                    continue

                total_steps = distance_a + distance_b + 2  # pick-up and drop
                if total_steps < min_cost:
                    min_cost = total_steps

            if min_cost != float('inf'):
                total_cost += min_cost
            else:
                # Penalize if no vehicle can pick up
                total_cost += 1000

        return total_cost
