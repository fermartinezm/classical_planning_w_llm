import sys
from collections import deque
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL fact strings
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions required to move all packages
    to their goal locations. It calculates the cost for each package independently
    based on its current state (on the ground or in a vehicle) and location
    relative to its goal location. It includes costs for pick-up, drop, and
    driving, using shortest path distances for driving costs.

    # Assumptions
    - The primary goal is to get packages to their specified locations.
    - Vehicle capacity and availability are relaxed; we assume a suitable vehicle
      will eventually be available for each package's transport needs.
    - The cost of driving between two locations is the shortest path distance
      in the road network.

    # Heuristic Initialization
    - Extracts goal locations for each package.
    - Builds the road network graph from static facts.
    - Computes all-pairs shortest paths between locations using BFS.

    # Step-by-Step Thinking for Computing the Heuristic Value
    For each package that is not yet at its goal location:

    1. Determine the package's current state:
       - Is it on the ground at some location `l_current`? (`(at p l_current)`)
       - Is it inside a vehicle `v`? (`(in p v)`) If inside a vehicle, find the vehicle's location `l_current` (`(at v l_current)`).

    2. Compare the package's current location `l_current` with its goal location `l_goal`.

    3. Calculate the estimated cost for this package:
       - If the package is `(at p l_current)` and `l_current != l_goal`:
         It needs to be picked up (1 action), the vehicle needs to drive from `l_current` to `l_goal` (shortest path distance actions), and it needs to be dropped (1 action).
         Estimated cost: `1 (pick-up) + dist(l_current, l_goal) (drive) + 1 (drop) = 2 + dist(l_current, l_goal)`.
       - If the package is `(in p v)` and the vehicle `v` is `(at v l_current)`:
         - If `l_current != l_goal`: The vehicle needs to drive from `l_current` to `l_goal` (shortest path distance actions), and the package needs to be dropped (1 action).
           Estimated cost: `dist(l_current, l_goal) (drive) + 1 (drop) = 1 + dist(l_current, l_goal)`.
         - If `l_current == l_goal`: The package is in the vehicle at the correct destination location. It only needs to be dropped (1 action) to satisfy the `(at p l_goal)` goal predicate.
           Estimated cost: `1 (drop)`.

    4. Sum the estimated costs for all packages that are not yet at their goal locations.
    5. If any package's current location is unreachable from its goal location (or vice-versa) in the road network, the state is likely a dead end or very far, so return infinity.

    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, static facts,
        building the road network, and computing shortest paths.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # Store goal locations for each package.
        self.goal_locations = {}
        for goal in self.goals:
            # Goals are typically (at package location)
            parts = get_parts(goal)
            if parts[0] == "at" and len(parts) == 3:
                package, location = parts[1], parts[2]
                self.goal_locations[package] = location

        # Build the road network graph and compute shortest paths.
        self.distances = self._compute_shortest_paths(static_facts)

    def _compute_shortest_paths(self, static_facts):
        """
        Builds the road network graph from static facts and computes
        all-pairs shortest paths using BFS.
        """
        locations = set()
        graph = {}

        # Extract locations and build initial graph structure
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "road" and len(parts) == 3:
                l1, l2 = parts[1], parts[2]
                locations.add(l1)
                locations.add(l2)
                if l1 not in graph:
                    graph[l1] = []
                graph[l1].append(l2)

        # Ensure all locations are in the graph dictionary, even if they have no roads
        for loc in locations:
            if loc not in graph:
                graph[loc] = []

        distances = {}

        # Run BFS from each location
        for start_node in locations:
            q = deque([(start_node, 0)])
            visited = {start_node}
            distances[(start_node, start_node)] = 0

            while q:
                current_loc, dist = q.popleft()

                # Get neighbors from the graph (handle locations with no outgoing roads)
                neighbors = graph.get(current_loc, [])

                for neighbor in neighbors:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        distances[(start_loc, neighbor)] = dist + 1
                        q.append((neighbor, dist + 1))

            # For any location not reached from start_node, distance is infinity
            for loc in locations:
                if (start_node, loc) not in distances:
                    distances[(start_node, loc)] = float('inf')

        return distances

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state (frozenset of fact strings).

        # Map locatables (packages, vehicles) to their current state/location
        current_state_map = {} # {object_name: location_or_container_name}
        vehicle_locations = {} # {vehicle_name: location_name}

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at" and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                current_state_map[obj] = loc
                # Check if the object is a vehicle (simple heuristic: starts with 'v')
                # A more robust way would be to parse types from the domain, but this is faster
                if obj.startswith('v'):
                     vehicle_locations[obj] = loc
            elif parts[0] == "in" and len(parts) == 3:
                 package, vehicle = parts[1], parts[2]
                 current_state_map[package] = vehicle # Package is inside this vehicle

        total_cost = 0  # Initialize action cost counter.

        # Iterate through packages that have a goal location
        for package, goal_location in self.goal_locations.items():
            # If package is not in the current state map, it might be an object
            # that doesn't move or isn't relevant to the goal (e.g., sizes). Skip.
            if package not in current_state_map:
                 continue

            current_status = current_state_map[package]

            # Check if the package is already at its goal location on the ground
            if current_status == goal_location:
                # We need to check if it's AT the goal location, not IN a vehicle at the goal location
                # The goal is (at package location), not (in package vehicle)
                # So, if current_status is a location and matches goal_location, cost is 0 for this package.
                # If current_status is a vehicle, it's not AT the goal location yet.
                is_at_goal_on_ground = False
                for fact in state:
                    parts = get_parts(fact)
                    if parts[0] == "at" and parts[1] == package and parts[2] == goal_location:
                        is_at_goal_on_ground = True
                        break # Found the goal fact for this package

                if is_at_goal_on_ground:
                    continue # Package is already at goal, cost is 0 for this package

            # Package is not at its goal location on the ground. Calculate cost.

            # Case 1: Package is on the ground at current_location
            if current_status in self.distances: # Check if current_status is a location
                current_location = current_status
                # Needs pick-up (1), drive (dist), drop (1)
                dist = self.distances.get((current_location, goal_location), float('inf'))
                if dist == float('inf'):
                    return float('inf') # Unreachable goal
                total_cost += 2 + dist

            # Case 2: Package is inside a vehicle (current_status is vehicle name)
            elif current_status.startswith('v'): # Simple check for vehicle name
                vehicle = current_status
                # Find the vehicle's location
                if vehicle not in vehicle_locations:
                     # This shouldn't happen in a valid state generated by domain ops,
                     # but as a fallback, treat as unreachable or high cost.
                     # Let's assume valid states for now.
                     return float('inf') # Vehicle location unknown

                current_location = vehicle_locations[vehicle]

                # Needs drive (dist), drop (1)
                dist = self.distances.get((current_location, goal_location), float('inf'))
                if dist == float('inf'):
                    return float('inf') # Unreachable goal
                total_cost += 1 + dist
            else:
                 # current_status is neither a known location nor a vehicle.
                 # This state might be invalid or represent something unexpected.
                 # Treat as unreachable.
                 return float('inf')


        return total_cost

