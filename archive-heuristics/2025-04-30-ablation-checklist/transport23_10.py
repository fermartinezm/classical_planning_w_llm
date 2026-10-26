import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty strings or malformed facts gracefully
    if not fact or not isinstance(fact, str) or len(fact) < 2:
        return []
    # Remove outer parentheses and split by spaces
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at package1 location1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Check if the number of parts matches the number of arguments in the pattern
    if len(parts) != len(args):
        return False
    # Check if each part matches the corresponding pattern argument
    return all(fnmatch.fnmatch(part, arg) for part, arg in zip(parts, args))

class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions required to move all packages
    to their goal locations. It considers the current location of packages
    (on the ground or in a vehicle) and the shortest path distance on the
    road network. It is relaxed by ignoring vehicle capacity and assuming
    a vehicle is available for pickup if needed.

    # Heuristic Initialization
    - Parses the static facts to build the road network graph.
    - Computes all-pairs shortest paths between locations using BFS.
    - Extracts the goal location for each package.
    - Identifies all locations, packages, and vehicles from the task definition.

    # Step-By-Step Thinking for Computing Heuristic
    For each package that is not yet at its goal location:
    1. Determine the package's current status: is it on the ground at a location, or is it inside a vehicle?
    2. If the package is on the ground at location L_p_current:
       - It needs to be picked up (1 action).
       - It needs to be transported by a vehicle from L_p_current to its goal location L_p_goal. The minimum number of drive actions is the shortest path distance between L_p_current and L_p_goal.
       - It needs to be dropped at L_p_goal (1 action).
       - Total estimated cost for this package: 1 (pick-up) + distance(L_p_current, L_p_goal) (drive) + 1 (drop) = 2 + distance(L_p_current, L_p_goal).
       - This step *relaxes* the need for a vehicle to be present at L_p_current and assumes one can reach it instantly for the purpose of the drive distance calculation, but we add the pick-up cost. A more refined version could add the cost for a vehicle to reach L_p_current, but this adds complexity (which vehicle? where is it?). The current approach is simpler and often effective.
    3. If the package is inside a vehicle V, and vehicle V is at location L_v_current:
       - It needs to be transported by vehicle V from L_v_current to its goal location L_p_goal. The minimum number of drive actions is the shortest path distance between L_v_current and L_p_goal.
       - It needs to be dropped at L_p_goal (1 action).
       - Total estimated cost for this package: distance(L_v_current, L_p_goal) (drive) + 1 (drop) = 1 + distance(L_v_current, L_p_goal).
    4. Sum the estimated costs for all packages not at their goal.

    This heuristic is non-admissible because it sums costs per package independently, ignoring shared vehicle trips and capacity constraints. However, it captures the essential costs of movement and handling (pick/drop) for each package.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, static facts,
        building the road network, and computing shortest path distances.
        """
        super().__init__(task)
        self.goals = task.goals
        self.static_facts = task.static
        self.initial_state = task.initial_state

        # Identify objects and goal locations
        self.locations = set()
        self.packages = set()
        self.vehicles = set()
        self.package_goals = {}

        # Parse static facts to find locations and road network
        self.road_graph = {} # Adjacency list for road network
        for fact in self.static_facts:
            if match(fact, "road", "*", "*"):
                l1, l2 = get_parts(fact)[1:]
                self.locations.add(l1)
                self.locations.add(l2)
                self.road_graph.setdefault(l1, set()).add(l2)
                self.road_graph.setdefault(l2, set()).add(l1) # Assuming roads are bidirectional

            # Identify vehicles from capacity facts (they are static in init)
            if match(fact, "capacity", "?v", "*"):
                 self.vehicles.add(get_parts(fact)[1])

        # Parse initial state to find initial locations and identify object types
        for fact in self.initial_state:
             if match(fact, "at", "?x", "?l"):
                 x, l = get_parts(fact)[1:]
                 self.locations.add(l) # Ensure all initial locations are included
                 # Infer type based on presence in vehicles set (from capacity)
                 if x in self.vehicles:
                     pass # Already identified as vehicle
                 else:
                     # Assume anything else 'at' a location initially is a package
                     self.packages.add(x)
             elif match(fact, "in", "?p", "?v"):
                 p, v = get_parts(fact)[1:]
                 self.packages.add(p)
                 self.vehicles.add(v) # Ensure vehicle is added if only appears in 'in'

        # Parse goals to find package goals and identify packages/locations
        for goal in self.goals:
            if match(goal, "at", "?p", "?l"):
                p, l = get_parts(goal)[1:]
                self.packages.add(p) # Ensure all goal packages are included
                self.locations.add(l) # Ensure all goal locations are included
                self.package_goals[p] = l

        # Ensure all locations mentioned in road_graph are in self.locations
        # (redundant if parsing road facts correctly, but safe)
        self.locations.update(self.road_graph.keys())
        for neighbors in self.road_graph.values():
             self.locations.update(neighbors)


        # Compute all-pairs shortest paths using BFS
        self.distances = {}
        for start_loc in self.locations:
            q = deque([(start_loc, 0)])
            visited = {start_loc}
            self.distances[(start_loc, start_loc)] = 0

            while q:
                current_loc, dist = q.popleft()

                # Add distance to self.distances
                self.distances[(start_loc, current_loc)] = dist

                # Explore neighbors
                if current_loc in self.road_graph:
                    for neighbor in self.road_graph[current_loc]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            q.append((neighbor, dist + 1))

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions
        to reach the goal state from the current state.
        """
        state = node.state

        # Track current locations of all locatables (packages and vehicles)
        # A package might be at a location or inside a vehicle.
        # A vehicle is always at a location.
        current_locations = {} # {object_name: location_or_vehicle_name}

        for fact in state:
            if match(fact, "at", "?x", "?l"):
                x, l = get_parts(fact)[1:]
                current_locations[x] = l
            elif match(fact, "in", "?p", "?v"):
                p, v = get_parts(fact)[1:]
                current_locations[p] = v # Store the vehicle name if package is inside

        total_cost = 0

        # Iterate through each package that has a goal location
        for package, goal_location in self.package_goals.items():
            # If package is not in the current state, something is wrong, skip or handle
            if package not in current_locations:
                 # This shouldn't happen in valid states, but as a safeguard:
                 # If a package is missing, it likely means it's not at its goal.
                 # We can't calculate a meaningful heuristic for it without its location.
                 # For a non-admissible heuristic, we could add a large penalty,
                 # but let's assume valid states for now.
                 continue

            package_current_status = current_locations[package]

            # If the package is already at its goal location, no cost for this package
            if package_current_status == goal_location:
                continue

            # Case 1: Package is on the ground at a location
            if package_current_status in self.locations:
                l_p_current = package_current_status
                # Cost = pick-up (1) + drive (distance) + drop (1)
                # We need the distance from the package's current location to its goal location
                drive_distance = self.distances.get((l_p_current, goal_location), float('inf'))

                # If goal is unreachable from current location, return infinity
                if drive_distance == float('inf'):
                    return float('inf')

                # Add cost for this package: pick-up + drive + drop
                total_cost += 1 + drive_distance + 1

            # Case 2: Package is inside a vehicle
            elif package_current_status in self.vehicles:
                vehicle_name = package_current_status
                # Find the location of the vehicle
                if vehicle_name not in current_locations:
                    # Vehicle is inside something else? Invalid state for this domain.
                    # Or vehicle location is not specified (shouldn't happen).
                    # Return infinity as goal is likely unreachable.
                    return float('inf')

                l_v_current = current_locations[vehicle_name]

                # If vehicle is not at a known location, return infinity
                if l_v_current not in self.locations:
                     return float('inf')

                # Cost = drive (distance) + drop (1)
                # We need the distance from the vehicle's current location to the package's goal location
                drive_distance = self.distances.get((l_v_current, goal_location), float('inf'))

                # If goal is unreachable from vehicle's current location, return infinity
                if drive_distance == float('inf'):
                    return float('inf')

                # Add cost for this package: drive + drop
                total_cost += drive_distance + 1

            # If package_current_status is neither a location nor a vehicle, it's an invalid state
            else:
                 # This should not happen in a valid state representation
                 return float('inf') # Indicate unreachable goal due to invalid state

        return total_cost

