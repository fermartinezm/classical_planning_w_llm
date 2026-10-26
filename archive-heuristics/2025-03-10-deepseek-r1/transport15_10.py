from fnmatch import fnmatch
from collections import defaultdict, deque
from heuristics.heuristic_base import Heuristic

class transport15Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions required to move all packages to their goal locations. It considers the drive actions needed for vehicles to reach packages and their destinations, as well as the pick-up and drop actions, taking into account vehicle capacities.

    # Assumptions
    - Roads are directed, but the static facts include both directions where applicable.
    - Each pick-up and drop action requires adjusting the vehicle's capacity based on capacity-predecessor relationships.
    - Vehicles can carry multiple packages if their capacity allows, but the heuristic assumes each package is handled individually for simplicity.

    # Heuristic Initialization
    - Precompute shortest paths between all locations using BFS based on road connections.
    - Extract capacity-predecessor relationships to determine valid pick-up capacities.
    - Identify goal locations for each package from the task's goals.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each package in the goal:
        a. If already at the goal, cost is 0.
        b. If in a vehicle:
            i. Calculate drive distance from vehicle's current location to the goal.
            ii. Add drive actions and a drop action.
        c. If not in a vehicle:
            i. Find vehicles that can pick up the package (current capacity allows).
            ii. For each valid vehicle, compute drive distance to package, pick-up, drive to goal, and drop.
            iii. Take the minimal cost from all valid vehicles.
            iv. If no valid vehicle, use a high estimate based on maximum possible drive steps.
    2. Sum the costs for all packages to get the heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic with static information from the task."""
        self.goal_locations = {}
        for goal in task.goals:
            parts = self.get_parts(goal)
            if parts[0] == 'at' and len(parts) == 3:
                package = parts[1]
                location = parts[2]
                self.goal_locations[package] = location

        # Build road graph and compute shortest paths
        self.road_graph = defaultdict(list)
        self.capacity_predecessors = {}
        self.locations = set()

        for fact in task.static:
            parts = self.get_parts(fact)
            if parts[0] == 'road':
                l1, l2 = parts[1], parts[2]
                self.road_graph[l1].append(l2)
                self.locations.update([l1, l2])
            elif parts[0] == 'capacity-predecessor':
                s1, s2 = parts[1], parts[2]
                self.capacity_predecessors[s2] = s1

        # Precompute shortest paths between all locations
        self.shortest_paths = {}
        for location in self.locations:
            self.shortest_paths[location] = self.bfs(location, self.road_graph)

        # Compute max drive distance for unreachable fallback
        self.max_drive = 0
        for source in self.shortest_paths:
            for dest, dist in self.shortest_paths[source].items():
                if dist != float('inf') and dist > self.max_drive:
                    self.max_drive = dist

    def bfs(self, start, graph):
        """Compute shortest paths from start using BFS."""
        distances = {loc: float('inf') for loc in self.locations}
        distances[start] = 0
        queue = deque([start])

        while queue:
            current = queue.popleft()
            for neighbor in graph.get(current, []):
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = distances[current] + 1
                    queue.append(neighbor)
        return distances

    def get_parts(self, fact):
        """Split a PDDL fact into its components."""
        return fact[1:-1].split()

    def match(self, fact, *args):
        """Check if a fact matches a pattern with wildcards."""
        parts = self.get_parts(fact)
        return all(fnmatch(part, arg) for part, arg in zip(parts, args))

    def __call__(self, node):
        """Compute the heuristic value for the given state."""
        state = node.state
        cost = 0

        # Extract current state information
        package_locations = {}  # {package: location or vehicle}
        vehicle_locations = {}  # {vehicle: location}
        vehicle_capacities = {}  # {vehicle: capacity}

        for fact in state:
            parts = self.get_parts(fact)
            if parts[0] == 'at':
                obj = parts[1]
                loc = parts[2]
                if obj in self.goal_locations:
                    package_locations[obj] = loc
                else:
                    vehicle_locations[obj] = loc
            elif parts[0] == 'in':
                package = parts[1]
                vehicle = parts[2]
                package_locations[package] = vehicle
            elif parts[0] == 'capacity':
                vehicle = parts[1]
                capacity = parts[2]
                vehicle_capacities[vehicle] = capacity

        for package, goal_loc in self.goal_locations.items():
            current = package_locations.get(package, None)
            if current is None:
                continue

            if current == goal_loc:
                continue  # Already at goal

            # Check if package is in a vehicle
            if current in vehicle_locations:
                vehicle = current
                vehicle_loc = vehicle_locations.get(vehicle, None)
                if vehicle_loc is None:
                    continue

                # Drive from vehicle's current location to goal
                drive_steps = self.shortest_paths.get(vehicle_loc, {}).get(goal_loc, float('inf'))
                if drive_steps == float('inf'):
                    drive_steps = self.max_drive
                cost += drive_steps + 1  # drive steps + drop action
            else:
                # Package is at a location, need to be picked up
                package_loc = current
                min_cost = float('inf')
                for vehicle, vehicle_loc in vehicle_locations.items():
                    # Check if vehicle can pick up (has capacity predecessor)
                    capacity = vehicle_capacities.get(vehicle, None)
                    if capacity not in self.capacity_predecessors:
                        continue

                    # Drive to package's location
                    drive_to_package = self.shortest_paths.get(vehicle_loc, {}).get(package_loc, float('inf'))
                    if drive_to_package == float('inf'):
                        continue

                    # Drive to goal from package's location
                    drive_to_goal = self.shortest_paths.get(package_loc, {}).get(goal_loc, float('inf'))
                    if drive_to_goal == float('inf'):
                        continue

                    total = drive_to_package + drive_to_goal + 2  # pick-up and drop
                    if total < min_cost:
                        min_cost = total

                if min_cost == float('inf'):
                    # No valid vehicle found, use fallback cost
                    cost += 2 * self.max_drive + 2
                else:
                    cost += min_cost

        return cost
