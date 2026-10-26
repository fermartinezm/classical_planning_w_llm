from collections import deque
from fnmatch import fnmatch # Although not strictly needed with get_parts, included for consistency with examples
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the total number of actions required to move all
    packages to their goal locations. It calculates the minimum actions needed
    for each package independently, ignoring vehicle capacity constraints and
    potential conflicts when multiple packages need the same vehicle or path.
    The cost for a package includes pick-up (if on the ground), driving the
    vehicle carrying it, and dropping it off at the destination. Driving cost
    is estimated using shortest path distances on the road network.

    # Assumptions
    - Roads are bidirectional.
    - All locations relevant to the problem (initial package/vehicle locations,
      goal locations) are part of a connected component in the road network.
    - The cost of pick-up, drop, and driving across one road segment is 1.
    - Vehicle capacity and availability are ignored; it's assumed a vehicle
      is available with sufficient capacity when needed by a package.

    # Heuristic Initialization
    - Extracts the goal location for each package from the task goals.
    - Builds a graph representing the road network from static facts.
    - Computes all-pairs shortest path distances between locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Check if the state is a goal state. If yes, the heuristic value is 0.
    2. Identify the current location of every package (either on the ground or inside a vehicle)
       and the current location of every vehicle.
    3. Initialize the total heuristic cost to 0.
    4. For each package that has a specified goal location:
       a. Determine the package's current status:
          - Is it already at its goal location? If yes, it contributes 0 to the cost.
          - Is it inside a vehicle?
          - Is it on the ground at a location other than its goal?
       b. If the package is inside a vehicle:
          - Find the current location of that vehicle.
          - The estimated cost for this package is the shortest distance from the vehicle's
            current location to the package's goal location, plus 1 action for dropping the package.
       c. If the package is on the ground at a location other than its goal:
          - The estimated cost for this package is 1 action for picking it up, plus the
            shortest distance from its current location to its goal location, plus 1 action
            for dropping it at the goal. (This assumes a vehicle will become available).
    5. The total heuristic value is the sum of the estimated costs for all packages
       that are not yet at their goal locations.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and static facts
        to build the road network graph and compute distances.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # Store goal locations for each package.
        self.goal_locations = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "at":
                package, location = parts[1], parts[2]
                self.goal_locations[package] = location

        # Build the road network graph and collect all locations.
        self.road_graph = {}
        self.locations = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'road':
                l1, l2 = parts[1], parts[2]
                self.road_graph.setdefault(l1, set()).add(l2)
                self.road_graph.setdefault(l2, set()).add(l1) # Roads are bidirectional
                self.locations.add(l1)
                self.locations.add(l2)

        # Add goal locations to the set of locations if they are not already present
        # in the road graph (unlikely in valid problems, but defensive).
        for goal_loc in self.goal_locations.values():
             if goal_loc not in self.locations:
                 self.locations.add(goal_loc)
                 self.road_graph.setdefault(goal_loc, set()) # Ensure it exists in graph dict

        # Compute all-pairs shortest paths using BFS.
        self.distances = {}
        for start_node in self.locations:
            q = deque([(start_node, 0)])
            visited = {start_node}
            self.distances[(start_node, start_node)] = 0 # Distance to self is 0

            while q:
                (current_loc, dist) = q.popleft()

                # Add distance for all visited nodes relative to this start_node
                # (This is already handled by the loop structure, but explicitly storing)
                # self.distances[(start_node, current_loc)] = dist # Already stored when adding to visited/queue

                for neighbor in self.road_graph.get(current_loc, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        self.distances[(start_node, neighbor)] = dist + 1
                        q.append((neighbor, dist + 1))

        # Note: Unreachable locations will not have entries in self.distances.
        # We handle this during lookup in __call__ by returning infinity.

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # If the goal is already reached, the heuristic is 0.
        if self.goals <= state:
            return 0

        # Track where packages and vehicles are currently located.
        package_state = {}  # package -> location or vehicle
        vehicle_locations = {} # vehicle -> location
        vehicles = set() # Collect all vehicle names

        # First pass to identify vehicles and packages based on predicates
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip invalid facts

            if parts[0] == 'in':
                package, vehicle = parts[1], parts[2]
                package_state[package] = vehicle
                vehicles.add(vehicle)
            elif parts[0] == 'capacity':
                vehicle = parts[1]
                vehicles.add(vehicle)

        # Second pass to get locations
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip invalid facts

            if parts[0] == 'at':
                obj, loc = parts[1], parts[2]
                if obj in vehicles:
                    vehicle_locations[obj] = loc
                else:
                    # Assume anything else 'at' a location is a package
                    package_state[obj] = loc

        total_cost = 0  # Initialize action cost counter.

        # Calculate cost for each package not at its goal
        for package, goal_location in self.goal_locations.items():
            current_state = package_state.get(package)

            # If package is not in the state or already at goal, continue
            if current_state is None or current_state == goal_location:
                continue

            # Check if the package is in a vehicle
            if current_state in vehicles:
                vehicle = current_state
                vehicle_loc = vehicle_locations.get(vehicle)
                if vehicle_loc is None:
                     # Vehicle location unknown - problem state might be malformed or unreachable
                     # For a greedy search heuristic, returning inf is appropriate if progress is blocked
                     return float('inf')

                # Package is in a vehicle, needs vehicle to drive and then drop
                dist = self.distances.get((vehicle_loc, goal_location), float('inf'))
                if dist == float('inf'):
                    # Goal location unreachable from vehicle location
                    return float('inf')
                total_cost += dist + 1 # drive + drop

            # Package is on the ground at a location other than the goal
            elif current_state in self.locations: # Check if it's a known location
                current_loc = current_state
                # Package needs pick-up, drive, drop
                dist = self.distances.get((current_loc, goal_location), float('inf'))
                if dist == float('inf'):
                    # Goal location unreachable from package location
                    return float('inf')
                total_cost += 1 + dist + 1 # pick-up + drive + drop
            else:
                 # Package is in an unknown state/location type
                 # This indicates a potentially malformed state
                 return float('inf')


        return total_cost

