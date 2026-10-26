from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at package1 location1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class transport4Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the transport domain.

    # Summary
    This heuristic estimates the number of actions required to move all packages to their goal locations.
    It considers the number of pick-up, drop, and drive actions needed.

    # Assumptions
    - Each package needs to be picked up, transported, and dropped off at its destination.
    - The heuristic assumes that vehicles are always available at the package's initial location.
    - The heuristic only considers the shortest path between locations.

    # Heuristic Initialization
    - Extract the goal locations for each package.
    - Extract the road network information to calculate shortest paths between locations.

    # Step-By-Step Thinking for Computing Heuristic
    1. Extract Goal Information:
       - Determine the goal location for each package from the task's goal conditions.

    2. Extract Road Network:
       - Build a road network represented as a dictionary where keys are locations and values are lists of adjacent locations.

    3. Calculate Package Costs:
       - For each package, determine its current location and its goal location.
       - If the package is already at its goal location, the cost is 0.
       - Otherwise, estimate the cost as follows:
         - 1 pick-up action.
         - Estimate the number of drive actions using a simple shortest path calculation (number of roads to traverse).
         - 1 drop action.

    4. Sum Package Costs:
       - The total heuristic value is the sum of the estimated costs for all packages.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting:
        - Goal locations for each package.
        - Road network information.
        """
        self.goals = task.goals
        static_facts = task.static

        # Extract goal locations for each package.
        self.goal_locations = {}
        for goal in self.goals:
            if match(goal, "at", "*", "*"):
                parts = get_parts(goal)
                package = parts[1]
                location = parts[2]
                self.goal_locations[package] = location

        # Build road network.
        self.road_network = {}
        for fact in static_facts:
            if match(fact, "road", "*", "*"):
                parts = get_parts(fact)
                l1 = parts[1]
                l2 = parts[2]
                if l1 not in self.road_network:
                    self.road_network[l1] = []
                if l2 not in self.road_network:
                    self.road_network[l2] = []
                self.road_network[l1].append(l2)
                self.road_network[l2].append(l1)

    def __call__(self, node):
        """Compute an estimate of the number of actions required to reach the goal."""
        state = node.state
        total_cost = 0

        for package, goal_location in self.goal_locations.items():
            current_location = None
            for fact in state:
                if match(fact, "at", package, "*"):
                    current_location = get_parts(fact)[2]
                    break
                elif match(fact, "in", package, "*"):
                    # Package is in a vehicle, find the vehicle's location
                    vehicle = get_parts(fact)[2]
                    for vehicle_fact in state:
                        if match(vehicle_fact, "at", vehicle, "*"):
                            current_location = get_parts(vehicle_fact)[2]
                            break
                    break

            if current_location is None:
                # Package is not at any location or in any vehicle, something is wrong
                continue

            if current_location == goal_location:
                continue  # Package is already at the goal

            # Estimate cost: 1 pick-up + drive actions + 1 drop
            cost = 2  # pick-up and drop

            # Estimate drive actions (shortest path)
            if current_location in self.road_network and goal_location in self.road_network:
                # Simple shortest path calculation (number of roads to traverse)
                queue = [(current_location, 0)]
                visited = {current_location}
                shortest_path = float('inf')

                while queue:
                    location, distance = queue.pop(0)
                    if location == goal_location:
                        shortest_path = distance
                        break

                    for neighbor in self.road_network[location]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append((neighbor, distance + 1))

                if shortest_path != float('inf'):
                    cost += shortest_path

            total_cost += cost

        return total_cost
