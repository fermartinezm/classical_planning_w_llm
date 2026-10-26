# Dummy Heuristic base class (replace with actual if available in the planning framework)
# This is included to make the code block self-contained and runnable structurally.
class Heuristic:
    def __init__(self, task):
        self.task = task
    def __call__(self, node):
        raise NotImplementedError

from fnmatch import fnmatch
from collections import deque

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact strings or malformed facts gracefully
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at package1 location1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the minimum number of actions required to move
    each package from its current location to its goal location, independently.
    It sums the estimated costs for each package that is not yet at its goal.
    The cost for a package includes:
    - 1 action for pickup (if on the ground).
    - The shortest path distance (number of drive actions) for the vehicle
      carrying the package (or that will pick it up) from its current location
      to the package's goal location.
    - 1 action for drop (when the vehicle reaches the goal location).

    # Assumptions
    - The road network is static and bidirectional.
    - Any vehicle can theoretically transport any package (vehicle capacity is ignored
      in the cost calculation, only location and state - 'at' or 'in' - matter).
    - Shortest path distances between locations represent the minimum number
      of drive actions.
    - The heuristic assumes a package needs a dedicated trip, ignoring the
      possibility of a vehicle carrying multiple packages simultaneously.
      This means it might overestimate the total number of drive actions
      in scenarios where vehicle sharing is optimal, but it provides a
      lower bound on pickup/drop actions per package.
    - All locations and packages mentioned in the goal and static facts are relevant.
    - State representation includes facts for all relevant objects (packages and vehicles).
    - Object names starting with 'p' are packages, 'v' are vehicles, 'l' are locations, 'c' are sizes.
      This is a heuristic assumption based on common PDDL conventions.

    # Heuristic Initialization
    The heuristic precomputes static information from the task:
    - Extracts goal locations for each package from the task goals.
    - Builds the road network graph from `road` facts in static information.
    - Collects all relevant locations from goals and road facts.
    - Computes all-pairs shortest path distances between all relevant locations using BFS.
    - (Optional: Extracts capacity hierarchy from `capacity-predecessor` facts,
      though not used in the current cost calculation).

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Initialize the total heuristic cost to 0.
    2. Parse the current state to determine the location of each package
       and each vehicle. A package is either `at` a location or `in` a vehicle.
       If a package is `in` a vehicle, its effective current location for
       transport planning is the location of that vehicle.
    3. Iterate through each package that has a specified goal location
       (extracted during initialization from the task goals).
    4. For the current package, check if the fact `(at package goal_location)`
       is present in the current state. If yes, add 0 to the total cost for
       this package and proceed to the next package.
    5. If the package is not at its goal:
       a. Determine the package's current "effective" location:
          - Look for `(at p l_current)` in the state. If found, the effective
            location is `l_current`.
          - If not found, look for `(in p v)` in the state. If found, find the
            location of vehicle `v` by looking for `(at v l_v)` in the state.
            The effective location is `l_v`.
          - If the package's location cannot be determined from the state,
            the state is considered unreachable or malformed, and the heuristic
            returns infinity.
       b. Calculate the minimum actions needed for this package:
          - If the package was found `(at p l_current)`: It needs a pickup action (1),
            transport from `l_current` to its goal `l_goal` (distance(l_current, l_goal)
            drive actions), and a drop action (1). Total: 1 + distance + 1.
          - If the package was found `(in p v)` at `l_v`: It needs transport from `l_v`
            to its goal `l_goal` (distance(l_v, l_goal) drive actions), and a
            drop action (1). Total: distance + 1.
       c. Add this calculated cost to the total heuristic cost. If the goal location
          is unreachable from the current effective location via the road network,
          the cost is infinite, indicating an unsolvable state.
    6. Return the total calculated cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, static facts,
        building the road graph, and computing shortest path distances.
        """
        # Assuming task object has attributes: goals, static
        self.goals = task.goals
        self.static_facts = task.static

        self.goal_locations = {}
        self.locations = set()
        self.packages = set()
        self.capacity_predecessors = {} # s1 -> s2 mapping
        self.capacity_successors = {} # s2 -> s1 mapping
        self.road_graph = {} # l1 -> set of l2

        # Parse goals to find package goals and identify packages/locations
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "at":
                # Goal is (at package location)
                package, location = parts[1], parts[2]
                self.goal_locations[package] = location
                self.packages.add(package)
                self.locations.add(location)

        # Parse static facts to build road graph and capacity hierarchy
        for fact in self.static_facts:
            parts = get_parts(fact)
            if not parts: continue # Skip empty or malformed facts

            if parts[0] == "road":
                # Fact is (road l1 l2)
                l1, l2 = parts[1], parts[2]
                self.road_graph.setdefault(l1, set()).add(l2)
                self.road_graph.setdefault(l2, set()).add(l1) # Assuming bidirectional roads
                self.locations.add(l1)
                self.locations.add(l2)
            elif parts[0] == "capacity-predecessor":
                # Fact is (capacity-predecessor s1 s2)
                s1, s2 = parts[1], parts[2]
                self.capacity_predecessors[s1] = s2
                self.capacity_successors[s2] = s1

        # Compute all-pairs shortest path distances
        self.distances = self._compute_all_pairs_shortest_paths()

        # Compute capacity holds (optional for this simple heuristic, but good practice)
        # self.capacity_holds = self._compute_capacity_holds()


    def _compute_all_pairs_shortest_paths(self):
        """
        Computes shortest path distances between all pairs of locations
        using BFS on the road graph.
        Returns a dictionary distances[l1][l2] = shortest_path_length.
        Returns float('inf') for unreachable locations.
        """
        distances = {l: {l2: float('inf') for l2 in self.locations} for l in self.locations}

        for start_node in self.locations:
            distances[start_node][start_node] = 0
            q = deque([(start_node, 0)])
            visited = {start_node}

            while q:
                current_loc, dist = q.popleft()

                # Ensure current_loc is a valid key in the graph (should be if from self.locations)
                # if current_loc not in self.road_graph:
                #      continue # Should not happen if locations are from road facts

                for neighbor in self.road_graph.get(current_loc, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        distances[start_node][neighbor] = dist + 1
                        q.append((neighbor, dist + 1))

        return distances

    def _compute_capacity_holds(self):
        """
        Computes how many packages a vehicle can hold based on its capacity level.
        Returns a dictionary capacity_holds[s] = num_packages.
        Assumes a single minimum capacity level (c0).
        This method is included for completeness but not used in the current heuristic cost calculation.
        """
        all_sizes = set(self.capacity_predecessors.keys()) | set(self.capacity_predecessors.values())
        sizes_that_are_successors = set(self.capacity_predecessors.values())

        # Find c0: the size that is a predecessor but is not a successor to any other size
        # Or simply, find s such that no (capacity-predecessor s_prev s) exists.
        c0 = None
        c0_candidates = all_sizes - sizes_that_are_successors
        if c0_candidates:
             c0 = next(iter(c0_candidates)) # Pick one if multiple candidates exist (should be unique)

        if c0 is None:
             # No capacity facts or no clear c0
             return {} # Return empty map

        capacity_holds = {c0: 0}
        q = deque([c0])

        # Build a map from s1 to list of s2 where (capacity-predecessor s1 s2)
        # We want to find s_next where (capacity-predecessor current_s s_next)
        # This means current_s is s1 and s_next is s2 in the predicate.
        # We want to find s2 for a given s1. This is self.capacity_predecessors.
        # We start with c0, Holds[c0]=0. Find s2 where (capacity-predecessor c0 s2).
        # Holds[s2] = Holds[c0] + 1. Add s2 to queue.
        # Then find s3 where (capacity-predecessor s2 s3). Holds[s3] = Holds[s2] + 1. Add s3.

        while q:
            current_s = q.popleft()

            # Find s_next such that (capacity-predecessor current_s s_next)
            s_next = self.capacity_predecessors.get(current_s) # This map is s1 -> s2

            if s_next is not None and s_next not in capacity_holds:
                 capacity_holds[s_next] = capacity_holds[current_s] + 1
                 q.append(s_next)

        return capacity_holds


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions
        to move all packages to their goal locations.
        """
        state = node.state
        total_cost = 0

        # Dictionaries to store current locations/containers
        package_loc = {} # package -> location or vehicle
        vehicle_loc = {} # vehicle -> location
        # vehicle_capacity = {} # vehicle -> size (not used in this heuristic cost)

        # Parse the current state to populate location/container maps
        # Identify object types based on appearance in relevant predicates
        # Relying on naming convention ('p' for package, 'v' for vehicle) for simplicity
        # as a more robust parser would require PDDL type information not directly in Task object.

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]
            if predicate == "at":
                obj, loc = parts[1], parts[2]
                if obj.startswith('p'): # Assume 'p' prefix means package
                    package_loc[obj] = loc
                elif obj.startswith('v'): # Assume 'v' prefix means vehicle
                     vehicle_loc[obj] = loc

            elif predicate == "in":
                p, v = parts[1], parts[2]
                if p.startswith('p'): # Assume 'p' prefix means package
                    package_loc[p] = v
                # Assume 'v' prefix means vehicle for the container

            # elif predicate == "capacity":
            #     v, s = parts[1], parts[2]
            #     if v.startswith('v'): # Assume 'v' prefix means vehicle
            #          vehicle_capacity[v] = s


        # Iterate through all packages that have a goal location (from task goals)
        for package, goal_l in self.goal_locations.items():
            # Check if the package is already at its goal location in the current state
            goal_fact = f"(at {package} {goal_l})"
            if goal_fact in state:
                continue # Package is already at its goal, cost is 0 for this package

            # If package is not at goal, it must be somewhere else or in a vehicle.
            # Its location/container must be in package_loc if the state is valid.
            current_p_container = package_loc.get(package)

            if current_p_container is None:
                 # Package location is unknown in the state. This indicates a malformed state.
                 # Return infinity as we can't estimate cost.
                 # print(f"Error: Location of package {package} (goal: {goal_l}) is unknown in the state.")
                 return float('inf')


            # Determine the effective current location for transport
            if current_p_container in self.locations: # It's a location string (e.g., 'l1')
                l_current = current_p_container
                # Package is on the ground at l_current
                # Needs pickup (1) + drive (dist) + drop (1)
                dist = self.distances.get(l_current, {}).get(goal_l, float('inf'))
                if dist == float('inf'):
                    # print(f"Warning: Goal location {goal_l} unreachable from {l_current} for package {package}.")
                    return float('inf') # Goal is unreachable

                total_cost += 1 # pickup
                total_cost += dist # drive
                total_cost += 1 # drop

            elif current_p_container.startswith('v'): # It's a vehicle string (e.g., 'v1')
                vehicle = current_p_container
                # Package is inside a vehicle. Find vehicle's location.
                l_v = vehicle_loc.get(vehicle)

                if l_v is None:
                    # Vehicle location is unknown - this is a problem.
                    # print(f"Error: Location of vehicle {vehicle} (carrying {package}) is unknown in the state.")
                    return float('inf') # Vehicle location unknown

                # Needs drive (dist) + drop (1)
                dist = self.distances.get(l_v, {}).get(goal_l, float('inf'))
                if dist == float('inf'):
                    # print(f"Warning: Goal location {goal_l} unreachable from vehicle location {l_v} for package {package}.")
                    return float('inf') # Goal is unreachable

                total_cost += dist # drive
                total_cost += 1 # drop
            else:
                 # current_p_container is neither a known location nor a vehicle? Malformed state.
                 # print(f"Error: Invalid container type for package {package}: {current_p_container}")
                 return float('inf')


        return total_cost
