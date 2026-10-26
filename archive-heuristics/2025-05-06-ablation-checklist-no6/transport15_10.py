import collections
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure the input is a string and looks like a PDDL fact
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        # Return empty list for invalid format, match function will handle length mismatch
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Pattern must match the number of parts
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    Estimates the number of actions (pick-up, drop, drive) required
    to move each package to its goal location, ignoring vehicle capacity
    and availability constraints.

    The heuristic sums the minimum actions needed for each package independently:
    - If a package is on the ground at location L and needs to go to L_goal:
      1 (pick-up) + distance(L, L_goal) + 1 (drop)
    - If a package is in a vehicle at location L_v and needs to go to L_goal:
      distance(L_v, L_goal) + 1 (drop)
    - If a package is already at its goal location (on the ground):
      0
    - If a package is in a vehicle at its goal location:
      1 (drop)

    Distance is the shortest path distance in the road network.
    Returns a large value if a required path segment is unreachable in the static graph.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static facts and computing
        shortest path distances.
        """
        super().__init__(task)

        # Extract goal locations for each package
        self.goal_locations = {}
        for goal in self.goals:
            # Goal facts are typically (at package location)
            if match(goal, "at", "*", "*"):
                parts = get_parts(goal)
                if len(parts) == 3: # Ensure it's (at obj loc)
                    _, package, location = parts
                    self.goal_locations[package] = location

        # Build the road network graph and collect all locations
        self.roads = collections.defaultdict(list)
        locations = set()
        for fact in self.static:
            if match(fact, "road", "*", "*"):
                parts = get_parts(fact)
                if len(parts) == 3: # Ensure it's (road loc1 loc2)
                    _, l1, l2 = parts
                    self.roads[l1].append(l2)
                    locations.add(l1)
                    locations.add(l2)

        # Add locations from goals and initial state to ensure BFS covers all relevant locations
        for loc in self.goal_locations.values():
             locations.add(loc)
        for fact in task.initial_state:
             if match(fact, "at", "*", "*"):
                  parts = get_parts(fact)
                  if len(parts) == 3:
                       _, obj, loc = parts
                       locations.add(loc)

        self.locations = list(locations) # Store locations for BFS

        # Compute all-pairs shortest paths using BFS
        self.dist = {}
        for start_loc in self.locations:
            self._bfs(start_loc)

    def _bfs(self, start_node):
        """
        Perform BFS from a start node to compute shortest distances
        to all reachable locations within the static road network.
        """
        q = collections.deque([(start_node, 0)])
        visited = {start_node: 0}
        self.dist[(start_node, start_node)] = 0

        while q:
            current_loc, current_dist = q.popleft()

            # Only explore if the current location is in the road graph keys
            # (i.e., there are roads leading *from* this location)
            if current_loc in self.roads:
                for neighbor in self.roads[current_loc]:
                    if neighbor not in visited:
                        visited[neighbor] = current_dist + 1
                        self.dist[(start_node, neighbor)] = current_dist + 1
                        q.append((neighbor, current_dist + 1))

        # Distances to unreachable locations from start_node are not stored in self.dist.
        # Lookup will return None, which we handle in __call__.


    def __call__(self, node):
        """
        Compute the heuristic value for the given state.
        """
        state = node.state

        # Check if the goal is reached (h=0 only at goal)
        if self.task.goal_reached(state):
             return 0

        # Track current locations of packages and vehicles
        package_current_status = {} # package -> location string or vehicle name string
        vehicle_location = {} # vehicle -> location string

        # Populate vehicle_location first, as package status might depend on it
        for fact in state:
             if match(fact, "at", "*", "*"):
                  parts = get_parts(fact)
                  if len(parts) == 3:
                       obj, loc = parts[1], parts[2]
                       # Simple heuristic assumption: objects starting with 'v' are vehicles
                       if obj.startswith('v'):
                            vehicle_location[obj] = loc

        # Now populate package_current_status for packages with goals
        for fact in state:
            if match(fact, "at", "*", "*"):
                parts = get_parts(fact)
                if len(parts) == 3:
                    obj, loc = parts[1], parts[2]
                    # If this object is a package with a goal
                    if obj in self.goal_locations:
                         package_current_status[obj] = loc # Package is on the ground

            elif match(fact, "in", "*", "*"):
                parts = get_parts(fact)
                if len(parts) == 3:
                    package, vehicle = parts[1], parts[2]
                    # If this package has a goal
                    if package in self.goal_locations:
                         package_current_status[package] = vehicle # Package is inside a vehicle


        total_cost = 0
        UNREACHABLE_PENALTY = 1000000 # Large value for unreachable segments

        # Iterate through packages that have a goal location
        for package, goal_location in self.goal_locations.items():
            # The goal for this package is (at package goal_location)
            # Check if this specific goal fact is in the state
            if f"(at {package} {goal_location})" in state:
                 continue # This package is already at its goal location on the ground

            # Package is not at its goal location on the ground.
            # Find its current status.
            current_status = package_current_status.get(package)

            if current_status is None:
                 # This package is not found in 'at' or 'in' facts in the state.
                 # This indicates an invalid state representation for a package
                 # that is part of the goal. Treat as unreachable.
                 # This should ideally not happen in valid states reachable from init.
                 return UNREACHABLE_PENALTY # Indicate bad state/unreachable goal for this package

            # Case A: Package is on the ground at l_current, needs to go to l_goal
            # current_status is a location string like 'l1', 'l2', etc.
            # Check if current_status is a known location (present in self.locations)
            if current_status in self.locations:
                l_current = current_status
                # We already know l_current != goal_location because we checked f"(at {package} {goal_location})" above.
                # Package needs pick-up, drive, and drop
                cost_for_package = 1 # pick-up

                # Add drive cost from current location to goal location
                drive_dist = self.dist.get((l_current, goal_location))
                if drive_dist is None:
                    # Goal location is unreachable from current location in the static graph.
                    # This state is likely not on a solvable path.
                    return UNREACHABLE_PENALTY # Indicate bad state/unreachable goal

                cost_for_package += drive_dist
                cost_for_package += 1 # drop
                total_cost += cost_for_package

            # Case B: Package is inside a vehicle v, needs to go to l_goal
            # current_status is a vehicle name string like 'v1', 'v2', etc.
            # Assuming objects starting with 'v' are vehicles based on examples
            elif current_status.startswith('v'):
                 vehicle = current_status
                 l_v = vehicle_location.get(vehicle) # Get vehicle's physical location

                 if l_v is None:
                      # Vehicle location unknown? Should not happen in valid state.
                      return UNREACHABLE_PENALTY # Indicate bad state

                 # Package needs drive (by the vehicle) and drop
                 # Add drive cost from vehicle's location to package's goal location
                 drive_dist = self.dist.get((l_v, goal_location))
                 if drive_dist is None:
                      return UNREACHABLE_PENALTY # Indicate bad state

                 cost_for_package = drive_dist
                 cost_for_package += 1 # drop
                 total_cost += cost_for_package

            # Case C: Package is in some other state (e.g., 'in' something not a vehicle?)
            # Based on domain, packages are either 'at' a location or 'in' a vehicle.
            # This case should not be reached in a valid state.
            else:
                 # Unknown state for package status, treat as highly undesirable
                 return UNREACHABLE_PENALTY


        return total_cost
