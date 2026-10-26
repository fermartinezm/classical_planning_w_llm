from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    fact = fact.strip()
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        # Handle potential malformed facts gracefully
        return []
    # Split by whitespace, handle multiple spaces
    return fact[1:-1].split()


class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    Estimates the number of actions needed to move each package to its goal location
    independently, using shortest path distances for driving.
    Ignores vehicle capacity and multi-package interactions.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, identifying objects,
        and precomputing shortest path distances between locations.
        """
        self.task = task

        # 1. Extract goal locations for each package.
        self.package_goals = {}
        for goal in self.task.goals:
            parts = get_parts(goal)
            if parts and parts[0] == 'at':
                package, location = parts[1], parts[2]
                self.package_goals[package] = location

        # 2. Identify packages and vehicles from initial state and goals
        # Assume packages are those with goals. Assume other objects in at/in facts are vehicles.
        self.packages = set(self.package_goals.keys())
        self.vehicles = set()
        locations = set() # Collect all locations

        # Collect all objects and locations mentioned in initial state and goals
        all_mentioned_objects = set()
        for fact in self.task.initial_state | self.task.goals:
            parts = get_parts(fact)
            if not parts: continue
            predicate = parts[0]
            if predicate == 'at':
                obj, loc = parts[1], parts[2]
                all_mentioned_objects.add(obj)
                locations.add(loc)
            elif predicate == 'in':
                 p, v = parts[1], parts[2]
                 all_mentioned_objects.add(p)
                 all_mentioned_objects.add(v)

        # Assume any mentioned object that is not a package (from goals) is a vehicle
        self.vehicles = {obj for obj in all_mentioned_objects if obj not in self.packages}


        # 3. Build road network graph and collect locations from static facts
        self.graph = {}
        for fact in self.task.static:
            parts = get_parts(fact)
            if parts and parts[0] == 'road':
                l1, l2 = parts[1], parts[2]
                if l1 not in self.graph: self.graph[l1] = []
                if l2 not in self.graph: self.graph[l2] = []
                self.graph[l1].append(l2)
                self.graph[l2].append(l1) # Assuming bidirectional roads
                locations.add(l1)
                locations.add(l2)

        # Ensure all collected locations are nodes in the graph
        for loc in locations:
             if loc not in self.graph:
                 self.graph[loc] = []

        # 4. Compute all-pairs shortest paths using BFS
        self.shortest_paths = {}
        all_locations = list(self.graph.keys())

        for start_node in all_locations:
            queue = [(start_node, 0)]
            visited = {start_node}

            while queue:
                (current_loc, dist) = queue.pop(0)

                self.shortest_paths[(start_node, current_loc)] = dist

                for neighbor in self.graph.get(current_loc, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, dist + 1))


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        Sum of costs for each package not at its goal.
        Cost for a package = (1 if not in vehicle) + shortest_drive_cost + 1 (drop)
        """
        state = node.state

        # Build current package status and vehicle locations from the state
        package_status = {} # {package: ('at', loc) or ('in', vehicle)}
        vehicle_locations = {} # {vehicle: loc}

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue
            predicate = parts[0]
            if predicate == 'at':
                obj, loc = parts[1], parts[2]
                if obj in self.packages:
                    package_status[obj] = ('at', loc)
                elif obj in self.vehicles:
                    vehicle_locations[obj] = loc
            elif predicate == 'in':
                 p, v = parts[1], parts[2]
                 if p in self.packages and v in self.vehicles:
                     package_status[p] = ('in', v)


        total_cost = 0

        # Iterate through packages that have a goal location
        for package, goal_location in self.package_goals.items():

            # Check if the package is already at its goal location
            # The goal fact is (at package goal_location)
            if f"(at {package} {goal_location})" in state:
                continue # Package is at goal, cost is 0 for this package.

            # Package is not at goal, estimate cost to move it.
            current_status = package_status.get(package)

            if current_status is None:
                 # This package is not found in any 'at' or 'in' fact in the state.
                 # This implies an invalid state representation for a package not at its goal.
                 # Return infinity as it's likely unsolvable from this state.
                 return float('inf')


            status_type, loc_or_vehicle = current_status

            if status_type == 'at':
                # Package is on the ground at loc_or_vehicle
                current_l = loc_or_vehicle
                # Need to pick up, drive, and drop
                # Cost = pick-up (1) + drive (shortest_path) + drop (1)
                drive_cost = self.shortest_paths.get((current_l, goal_location), float('inf'))
                if drive_cost == float('inf'):
                    # Cannot reach goal location from current location
                    return float('inf')
                total_cost += 1 + drive_cost + 1

            elif status_type == 'in':
                # Package is inside vehicle loc_or_vehicle
                vehicle = loc_or_vehicle
                # Need to drive vehicle to goal location and drop
                # Cost = drive (shortest_path) + drop (1)
                vehicle_l = vehicle_locations.get(vehicle)
                if vehicle_l is None:
                    # Vehicle location unknown (invalid state?)
                    return float('inf')

                drive_cost = self.shortest_paths.get((vehicle_l, goal_location), float('inf'))
                if drive_cost == float('inf'):
                    # Cannot reach goal location from vehicle's current location
                    return float('inf')
                total_cost += drive_cost + 1

        return total_cost
