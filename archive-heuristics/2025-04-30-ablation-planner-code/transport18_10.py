from collections import deque
# from heuristics.heuristic_base import Heuristic # Assuming this base class exists

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential leading/trailing whitespace or malformed facts gracefully
    fact = fact.strip()
    if not fact.startswith('(') or not fact.endswith(')'):
        # Return empty list for invalid format
        return []
    return fact[1:-1].split()

# BFS function to compute shortest paths
def bfs(start_loc, road_graph, all_locations):
    """
    Computes shortest path distances from start_loc to all other locations
    in the road network using BFS.
    """
    distances = {loc: float('inf') for loc in all_locations}
    distances[start_loc] = 0
    queue = deque([start_loc])

    # Use a set for visited nodes for efficiency
    visited = {start_loc}

    while queue:
        current_loc = queue.popleft()

        # Check if current_loc is a valid key in the graph
        if current_loc in road_graph:
            for neighbor in road_graph[current_loc]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    distances[neighbor] = distances[current_loc] + 1
                    queue.append(neighbor)
    return distances

# Define the heuristic class
# class transportHeuristic(Heuristic): # Use this line in the actual planning system
class transportHeuristic: # Use this line for standalone code output
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the required number of actions (pick-up, drive, drop)
    to move all packages that are not currently at their goal location. It sums
    the estimated costs for each misplaced package independently, ignoring
    vehicle capacity constraints and potential conflicts or synergies when
    multiple packages need the same vehicle or route.

    # Assumptions
    - Actions have unit cost.
    - The road network is static and bidirectional (as per example instances).
    - Package size is implicitly handled by vehicle capacity predicates, but the
      heuristic simplifies by assuming any vehicle with non-zero capacity can
      pick up any package. Capacity constraints are otherwise ignored.
    - The heuristic calculates the shortest path distance for driving using BFS.
    - Unreachable goals result in an infinite heuristic value.
    - The goal only involves packages being at specific locations.

    # Heuristic Initialization
    - Parses static facts to build the road network graph, assuming bidirectionality.
    - Computes all-pairs shortest paths between locations using BFS.
    - Extracts goal locations for each package from the task's goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic is computed as follows:
    1. Identify the current location or container (vehicle) for every package
       and vehicle present in the state facts.
    2. Initialize the total heuristic cost to 0.
    3. For each package that has a specified goal location:
        a. Determine the package's current status (on the ground at a location,
           or inside a vehicle). If the package is not mentioned in the state,
           it is assumed to be delivered if it had a goal, or irrelevant otherwise.
           We only consider packages with goals.
        b. If the package's current status indicates it is already at its goal
           location on the ground, it requires 0 further actions for this package.
           Continue to the next package.
        c. If the package is on the ground at a location `l_current` which is
           not its goal location `l_goal`:
            - It needs to be picked up (1 action).
            - It needs to be transported by a vehicle from `l_current` to `l_goal`.
              The estimated cost for driving is the shortest path distance
              `dist(l_current, l_goal)`.
            - It needs to be dropped at `l_goal` (1 action).
            - Add `1 + dist(l_current, l_goal) + 1` to the total cost.
            - If `l_goal` is unreachable from `l_current`, the heuristic is
              infinite, indicating an unsolvable state.
        d. If the package is inside a vehicle `v`, and the vehicle is at location `l_v`:
            - It needs to be transported by the vehicle from `l_v` to `l_goal`.
              The estimated cost for driving is the shortest path distance
              `dist(l_v, l_goal)`.
            - It needs to be dropped at `l_goal` (1 action).
            - Add `dist(l_v, l_goal) + 1` to the total cost.
            - If `l_goal` is unreachable from `l_v`, the heuristic is infinite.
    4. Return the total computed cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, static facts,
        building the road graph, and computing shortest path distances.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # Build the road network graph
        self.road_graph = {}
        locations_set = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "road":
                # Ensure fact has enough parts
                if len(parts) >= 3:
                    loc1, loc2 = parts[1], parts[2]
                    if loc1 not in self.road_graph:
                        self.road_graph[loc1] = []
                    if loc2 not in self.road_graph:
                        self.road_graph[loc2] = []
                    # Add road in both directions assuming bidirectionality
                    self.road_graph[loc1].append(loc2)
                    self.road_graph[loc2].append(loc1)
                    locations_set.add(loc1)
                    locations_set.add(loc2)
                # else: Ignore malformed road fact

        self.all_locations = list(locations_set)

        # Compute all-pairs shortest paths
        self.dist = {}
        for start_loc in self.all_locations:
            self.dist[start_loc] = bfs(start_loc, self.road_graph, self.all_locations)

        # Store goal locations for each package
        self.goal_locations = {}
        for goal in self.goals:
            parts = get_parts(goal)
            # Assuming goals are only (at package location)
            if parts and parts[0] == "at":
                 # Ensure fact has enough parts
                if len(parts) >= 3:
                    package, location = parts[1], parts[2]
                    self.goal_locations[package] = location
                # else: Ignore malformed goal fact or non-'at' goals

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Track where packages and vehicles are currently located or contained.
        current_locations = {} # Maps locatable -> location or vehicle
        for fact in state:
            parts = get_parts(fact)
            if not parts: # Skip malformed facts
                continue
            if parts[0] == "at":
                 if len(parts) >= 3:
                    locatable, location = parts[1], parts[2]
                    current_locations[locatable] = location
            elif parts[0] == "in":
                 if len(parts) >= 3:
                    package, vehicle = parts[1], parts[2]
                    current_locations[package] = vehicle # Package is inside vehicle
            # Ignore other predicates in the state

        total_cost = 0  # Initialize action cost counter.

        # Consider only packages that have a goal location specified
        for package, goal_location in self.goal_locations.items():
            current_status = current_locations.get(package)

            # If package is not in the state facts, it cannot be moved.
            # This might indicate an unreachable goal if the package needs moving.
            # However, standard PDDL problems list all objects in init.
            # Let's assume packages with goals are always in the state.
            if current_status is None:
                 # This shouldn't happen in valid problem instances.
                 # Treat as unreachable goal for this package.
                 return float('inf')

            # Check if the package is already at its goal location on the ground
            # A package is at its goal if its current status is the goal location string.
            if current_status == goal_location:
                 # This package is already delivered.
                 continue

            # Package is not at its goal. Estimate cost to get it there.
            # Case 1: Package is on the ground at l_current
            # We check if the current_status is a location by seeing if it's in our known locations set.
            # This assumes vehicle names and location names are distinct.
            if current_status in self.all_locations:
                l_current = current_status
                l_goal = goal_location

                # Cost: pick-up (1) + drive (dist) + drop (1)
                # Need distance from l_current to l_goal
                drive_cost = self.dist.get(l_current, {}).get(l_goal, float('inf'))

                if drive_cost == float('inf'):
                    # Goal is unreachable for this package from its current location
                    return float('inf')

                total_cost += 1  # pick-up action
                total_cost += drive_cost # drive actions
                total_cost += 1  # drop action

            # Case 2: Package is inside a vehicle v
            else: # current_status must be a vehicle name
                vehicle = current_status
                # Find the location of the vehicle
                l_v = current_locations.get(vehicle)
                if l_v is None:
                    # Vehicle location not found, state is inconsistent?
                    # Treat as unreachable goal for this package.
                    return float('inf')

                l_goal = goal_location

                # Cost: drive (dist) + drop (1)
                # Need distance from vehicle's location l_v to l_goal
                drive_cost = self.dist.get(l_v, {}).get(l_goal, float('inf'))

                if drive_cost == float('inf'):
                     # Goal is unreachable for this package via its current vehicle
                     return float('inf')

                total_cost += drive_cost # drive actions
                total_cost += 1  # drop action

        # The total_cost represents the sum of estimated actions for all
        # packages that are not yet at their goal location.
        # If total_cost is 0, it means all packages with goals are at their goals.
        # Assuming the goal is solely about package locations, this implies
        # the state is the goal state, and the heuristic is correctly 0.
        # If the state is not the goal state but total_cost is 0 (because
        # all package goals are met, but other goal conditions exist),
        # this heuristic would incorrectly return 0. However, based on the
        # provided domain and instances, goals are only package locations.

        return total_cost
