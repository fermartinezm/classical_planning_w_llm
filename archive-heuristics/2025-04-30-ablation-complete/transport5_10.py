# Required imports
from collections import deque, defaultdict
# Assuming Heuristic base class is available in heuristics.heuristic_base
from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and starts/ends with parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        # Handle unexpected input, maybe log a warning or return empty
        return [] # Return empty list for safety

    return fact[1:-1].split()


class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions required to move all
    misplaced packages to their goal locations. It considers the cost of
    picking up, dropping, and driving. It uses precomputed shortest path
    distances between locations. It adds a small penalty if a package on
    the ground cannot be immediately picked up by any vehicle at its location
    due to capacity constraints.

    # Assumptions
    - Packages need to be picked up by a vehicle, driven to the destination,
      and dropped.
    - Vehicle movement cost is the shortest path distance in the road network.
    - All packages consume one unit of capacity. Capacity predicates c_i
      correspond to i available slots, where (capacity-predecessor c_i c_{i+1})
      implies c_i has one less slot than c_{i+1}.
    - The heuristic sums the estimated costs for each package that is not
      currently satisfying its goal condition (being on the ground at the goal location).

    # Heuristic Initialization
    - Extracts goal locations for each package from task.goals.
    - Builds the road network graph from (road l1 l2) static facts.
    - Computes all-pairs shortest path distances between locations using BFS.
    - Builds a mapping from capacity predicates (c0, c1, ...) to integer
      available slots using (capacity-predecessor s1 s2) static facts.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current location of every package (on ground or in a vehicle).
    2. Identify the current location and capacity predicate of every vehicle.
    3. Initialize total heuristic cost to 0.
    4. For each package P with a goal location L_goal:
       a. Check if the goal fact `(at P L_goal)` is true in the current state. If yes, this package contributes 0 to the heuristic.
       b. If the goal fact is not true:
          i. Find the package's current location/status (on ground at L_curr or in vehicle V).
          ii. If P is on the ground at L_curr (and L_curr != L_goal):
              - Calculate base cost = 1 (pick-up) + distance(L_curr, L_goal) (drive) + 1 (drop).
              - Check if there is any vehicle V currently at L_curr with available capacity (> 0 slots).
              - If no such vehicle exists, add a penalty of 1 to the base cost (representing the need to bring a vehicle or free capacity).
              - Add the calculated cost (base + penalty) to the total cost.
          iii. If P is inside a vehicle V:
              - Find the current location L_v of vehicle V.
              - Calculate cost = distance(L_v, L_goal) (drive) + 1 (drop).
              - Add this cost to the total cost.
          iv. If package location/status is unknown or invalid, return infinity.
    5. Return the total heuristic cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, static facts,
        building the road graph, computing distances, and mapping capacities.
        """
        self.goals = task.goals
        static_facts = task.static

        # 1. Extract goal locations for packages
        self.goal_locations = {}
        for goal in self.goals:
            parts = get_parts(goal)
            # Assuming goals are always (at package location)
            if parts and parts[0] == "at" and len(parts) == 3:
                package, location = parts[1:]
                self.goal_locations[package] = location

        # 2. Build road network graph and collect all locations
        self.road_graph = defaultdict(list)
        locations = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "road" and len(parts) == 3:
                l1, l2 = parts[1:]
                self.road_graph[l1].append(l2)
                locations.add(l1)
                locations.add(l2)
        self.locations = list(locations) # Store list of all locations

        # 3. Compute all-pairs shortest path distances
        self.distances = {}
        for start_node in self.locations:
            q = deque([(start_node, 0)])
            visited = {start_node}
            self.distances[(start_node, start_node)] = 0

            while q:
                current_loc, dist = q.popleft()

                for neighbor in self.road_graph.get(current_loc, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        self.distances[(start_node, neighbor)] = dist + 1
                        q.append((neighbor, dist + 1))

        # 4. Map capacity predicates to integer slots
        self.capacity_map = {}
        capacity_predecessors_map = {} # s2 -> s1 mapping
        capacity_successors_map = {} # s1 -> s2 mapping
        all_capacity_predicates = set()

        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "capacity-predecessor" and len(parts) == 3:
                s1, s2 = parts[1:]
                capacity_predecessors_map[s2] = s1
                capacity_successors_map[s1] = s2
                all_capacity_predicates.add(s1)
                all_capacity_predicates.add(s2)

        # Find the predicate corresponding to 0 available slots (most full)
        # This is the one that is a predecessor but never a successor (i.e., only appears on the left of predecessor facts)
        c_zero_slots = None
        # Find predicates that are *not* the second argument (s2) of any capacity-predecessor fact
        is_s2_in_predecessor = set(capacity_predecessors_map.keys())
        for s in all_capacity_predicates:
             if s not in is_s2_in_predecessor:
                 c_zero_slots = s
                 break

        if c_zero_slots:
            self.capacity_map[c_zero_slots] = 0
            q = deque([c_zero_slots])
            visited = {c_zero_slots}

            while q:
                s1 = q.popleft()
                s2 = capacity_successors_map.get(s1) # Find the predicate s2 where (capacity-predecessor s1 s2)
                if s2 and s2 not in visited:
                    visited.add(s2)
                    self.capacity_map[s2] = self.capacity_map[s1] + 1
                    q.append(s2)

        # If no capacity predicates or no predecessor chain found, capacity check is effectively disabled
        # because get(capacity_pred, 0) will always return 0, and the penalty check will always fail.
        # This is acceptable; the heuristic degrades gracefully.


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify current locations and capacities
        current_package_locations = {} # package -> location or vehicle
        current_vehicle_locations = {} # vehicle -> location
        current_vehicle_capacities = {} # vehicle -> capacity_predicate

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip invalid facts

            if parts[0] == "at":
                if len(parts) == 3:
                    obj, loc = parts[1:]
                    if obj.startswith('p'): # is a package
                        current_package_locations[obj] = loc
                    elif obj.startswith('v'): # is a vehicle
                        current_vehicle_locations[obj] = loc
            elif parts[0] == "in":
                 if len(parts) == 3:
                    package, vehicle = parts[1:]
                    current_package_locations[package] = vehicle # package is in vehicle
            elif parts[0] == "capacity":
                 if len(parts) == 3:
                    vehicle, capacity_pred = parts[1:]
                    current_vehicle_capacities[vehicle] = capacity_pred

        total_cost = 0

        # 2. Calculate cost for each misplaced package
        for package, goal_location in self.goal_locations.items():
            # Check if the package is already at its goal location on the ground
            goal_fact = f"(at {package} {goal_location})"
            if goal_fact in state:
                continue # Package is already where it needs to be

            # Package is not at the goal location on the ground.
            # Find its current state: on ground or in vehicle.
            current_location_info = current_package_locations.get(package)

            # Package location is unknown (should not happen in valid states)
            if current_location_info is None:
                 # This indicates an issue with state representation or parsing
                 # Return infinity or a large constant
                 return float('inf') # Or a large constant

            # Package is on the ground at L_curr (which is not L_goal)
            if current_location_info in self.locations: # Check if it's a known location string
                L_curr = current_location_info
                dist_to_goal = self.distances.get((L_curr, goal_location), float('inf'))

                # If goal is unreachable, this path is likely invalid or very long
                if dist_to_goal == float('inf'):
                    return float('inf') # Or a large constant

                # Base cost: pick + drive + drop
                cost = 1 + dist_to_goal + 1

                # Check for immediate pickup feasibility (capacity)
                # Only apply penalty if capacity mapping was successfully built
                if self.capacity_map:
                    can_be_picked_up_immediately = False
                    # Check if any vehicle is at L_curr with capacity > 0
                    for vehicle, v_loc in current_vehicle_locations.items():
                        if v_loc == L_curr: # Vehicle is at the same location
                            capacity_pred = current_vehicle_capacities.get(vehicle)
                            # Get available slots, default to 0 if predicate not mapped (e.g., c0)
                            available_slots = self.capacity_map.get(capacity_pred, 0)
                            if available_slots > 0:
                                can_be_picked_up_immediately = True
                                break # Found a vehicle that can pick it up

                    # Add penalty if no vehicle can pick it up immediately
                    if not can_be_picked_up_immediately:
                        cost += 1 # Penalty for needing to move a vehicle or free capacity

                total_cost += cost

            # Package is in a vehicle V
            elif current_location_info.startswith('v'): # Check if it's a vehicle name string
                vehicle = current_location_info
                L_v = current_vehicle_locations.get(vehicle)

                # Vehicle location is unknown (should not happen)
                if L_v is None:
                    return float('inf') # Or a large constant

                # Package is in vehicle V, V is at L_v. Needs to get to L_goal on the ground.
                # Cost: drive V from L_v to L_goal + drop package at L_goal.
                dist_to_goal = self.distances.get((L_v, goal_location), float('inf'))

                # If goal is unreachable for the vehicle, this path is likely invalid
                if dist_to_goal == float('inf'):
                     return float('inf') # Or a large constant

                # Cost: drive + drop
                cost = dist_to_goal + 1
                total_cost += cost
            else:
                 # Unknown package location type
                 return float('inf') # Or a large constant

        # Heuristic must be 0 for goal states.
        # The loop above calculates cost only for packages NOT satisfying the goal fact (at P L_goal).
        # If all packages satisfy the goal fact, the loop is skipped and total_cost remains 0.
        # This is correct.

        return total_cost
