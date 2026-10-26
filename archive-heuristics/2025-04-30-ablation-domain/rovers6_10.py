from fnmatch import fnmatch
from collections import deque, defaultdict
# Assume Heuristic base class is available in the environment
# from heuristics.heuristic_base import Heuristic

# Define a dummy Heuristic base class if it's not provided in the execution environment
# This allows the code to be syntactically correct for testing outside the planner framework.
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    class Heuristic:
        def __init__(self, task):
            pass
        def __call__(self, node):
            pass


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure the fact is a string and starts/ends with parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        # Handle potential errors or unexpected fact formats
        # print(f"Warning: Unexpected fact format: {fact}") # Debugging
        return []
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at rover1 waypoint1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # The number of parts must match the number of arguments in the pattern
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve all goal conditions.
    It sums the estimated minimum cost for each unachieved goal independently.
    The cost for a goal is estimated based on the current state of progress towards that goal
    (e.g., sample taken, analysis done, image taken) and includes estimated movement costs
    using precomputed shortest paths between waypoints for each rover.

    # Assumptions
    - All goal conditions are of the form `(communicated_soil_data ?w)`, `(communicated_rock_data ?w)`,
      or `(communicated_image_data ?o ?m)`.
    - The cost of each relevant action (move, take_sample, analyze, calibrate, take_image, communicate) is 1.
    - Store constraints for samples are simplified: assume an equipped rover can use its store if it is currently empty.
    - Calibration state is simplified: assume a suitable camera can be calibrated if needed, adding the calibration cost.
    - Unreachable goals contribute a large cost (infinity).

    # Heuristic Initialization
    - Parse static facts to identify:
        - Lander location.
        - Rover capabilities (imaging, soil, rock).
        - Camera information (on-board rover, supported modes, calibration target).
        - Objective visibility from waypoints.
        - Sample locations (soil, rock).
        - Rover store mapping.
        - Waypoint connectivity for each rover (`can_traverse`).
    - Collect all relevant waypoints mentioned in static facts or goals.
    - Build a graph for each rover based on `can_traverse` facts using the collected waypoints.
    - Compute all-pairs shortest paths between waypoints for each rover using BFS on its graph.
    - Store goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify all goal facts that are not yet true in the current state.
    2. Initialize total heuristic cost to 0.
    3. Pre-process the current state into a quick-lookup structure (e.g., dictionary of predicates to argument tuples).
    4. Get the current location of each rover and the lander location.
    5. For each unachieved goal fact:
        a. Initialize minimum cost for this goal to infinity.
        b. If the goal is `(communicated_soil_data ?waypoint)`:
            i. Check if `(have_soil_analysis ?rover ?waypoint)` is true for any rover in the current state. If yes, the cost is `dist(?rover, current_rover_loc, lander_wp) + 1` (communicate). Update minimum cost.
            ii. Else, check if `(have_soil_sample ?rover ?waypoint)` is true for any rover. If yes, the cost is `dist(?rover, current_rover_loc, ?waypoint) + 1` (analyze) `+ dist(?rover, ?waypoint, lander_wp) + 1` (communicate). Update minimum cost.
            iii. Else (need to take sample): Check if `(at_soil_sample ?waypoint)` is a static fact. If yes, find an equipped rover (`equipped_for_soil_analysis`) whose store is currently empty. The cost is `dist(?rover, current_rover_loc, ?waypoint) + 1` (take sample) `+ 1` (analyze) `+ dist(?rover, ?waypoint, lander_wp) + 1` (communicate). Update minimum cost over all suitable rovers.
            iv. If after checking all cases, the minimum cost is still infinity, the goal is unreachable.
        c. If the goal is `(communicated_rock_data ?waypoint)`: Follow a similar logic as for soil data, using rock-specific predicates and capabilities.
        d. If the goal is `(communicated_image_data ?objective ?mode)`:
            i. Check if `(have_image ?rover ?objective ?mode)` is true for any rover. If yes, the cost is `dist(?rover, current_rover_loc, lander_wp) + 1` (communicate). Update minimum cost.
            ii. Else (need to take image): Find a rover equipped for imaging with a camera on board that supports the mode and is a calibration target for the objective. Find a waypoint visible from the objective (`visible_from`). The cost is `dist(?rover, current_rover_loc, visible_waypoint) + 1` (calibrate) `+ 1` (take image) `+ dist(?rover, visible_waypoint, lander_wp) + 1` (communicate). Update minimum cost over all suitable rover/camera/waypoint combinations.
            iii. If after checking all cases, the minimum cost is still infinity, the goal is unreachable.
        e. Add the minimum cost for this goal to the total heuristic cost. If any goal is unreachable, the total cost is infinity.
    6. Return the total heuristic cost.
    """

    def __init__(self, task):
        """Initialize the heuristic with static task information."""
        self.goals = task.goals
        self.static_facts = task.static

        # --- Extract Static Information ---
        self.lander_waypoint = None
        self.rover_capabilities = defaultdict(set) # {rover: {cap1, cap2}}
        self.camera_info = {} # {camera: {'on_board': rover, 'supports': {mode1}, 'calibration_target': objective}}
        self.objective_visibility = defaultdict(set) # {objective: {waypoint1}}
        self.sample_locations = {'soil': set(), 'rock': set()} # {type: {waypoint}}
        self.rover_stores = {} # {rover: store}
        self.waypoint_graph = defaultdict(lambda: defaultdict(set)) # {rover: {wp1: {wp2, wp3}}}
        self.rovers = set()
        self.waypoints = set()

        for fact in self.static_facts:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]

            if predicate == "at_lander":
                if len(parts) == 3: self.lander_waypoint = parts[2]
            elif predicate.startswith("equipped_for_"):
                if len(parts) == 2:
                    capability_type = predicate[len("equipped_for_"):]
                    rover = parts[1]
                    self.rover_capabilities[rover].add(capability_type)
                    self.rovers.add(rover)
            elif predicate == "on_board":
                if len(parts) == 3:
                    camera, rover = parts[1], parts[2]
                    if camera not in self.camera_info:
                        self.camera_info[camera] = {'on_board': None, 'supports': set(), 'calibration_target': None}
                    self.camera_info[camera]['on_board'] = rover
                    self.rovers.add(rover)
            elif predicate == "supports":
                 if len(parts) == 3:
                    camera, mode = parts[1], parts[2]
                    if camera not in self.camera_info:
                         self.camera_info[camera] = {'on_board': None, 'supports': set(), 'calibration_target': None}
                    self.camera_info[camera]['supports'].add(mode)
            elif predicate == "calibration_target":
                 if len(parts) == 3:
                    camera, objective = parts[1], parts[2]
                    if camera not in self.camera_info:
                         self.camera_info[camera] = {'on_board': None, 'supports': set(), 'calibration_target': None}
                    self.camera_info[camera]['calibration_target'] = objective
            elif predicate == "visible_from":
                 if len(parts) == 3:
                    objective, waypoint = parts[1], parts[2]
                    self.objective_visibility[objective].add(waypoint)
                    self.waypoints.add(waypoint)
            elif predicate == "at_soil_sample":
                 if len(parts) == 2:
                    self.sample_locations['soil'].add(parts[1])
                    self.waypoints.add(parts[1])
            elif predicate == "at_rock_sample":
                 if len(parts) == 2:
                    self.sample_locations['rock'].add(parts[1])
                    self.waypoints.add(parts[1])
            elif predicate == "store_of":
                 if len(parts) == 3:
                    store, rover = parts[1], parts[2]
                    self.rover_stores[rover] = store
                    self.rovers.add(rover)
            elif predicate == "can_traverse":
                 if len(parts) == 4:
                    rover, wp1, wp2 = parts[1], parts[2], parts[3]
                    self.waypoint_graph[rover][wp1].add(wp2)
                    self.rovers.add(rover)
                    self.waypoints.add(wp1)
                    self.waypoints.add(wp2)
            # Ignore dynamic predicates or those not needed for static analysis

        # Ensure all waypoints mentioned in goals or samples are in the graph nodes
        for goal in self.goals:
             parts = get_parts(goal)
             if not parts: continue
             predicate = parts[0]
             if predicate in ("communicated_soil_data", "communicated_rock_data"):
                 if len(parts) > 1: self.waypoints.add(parts[1])
             elif predicate == "communicated_image_data":
                 if len(parts) > 1:
                     objective = parts[1]
                     if objective in self.objective_visibility:
                         self.waypoints.update(self.objective_visibility[objective])

        # Add lander waypoint if not already in waypoints
        if self.lander_waypoint:
             self.waypoints.add(self.lander_waypoint)

        # --- Compute Shortest Paths for each rover ---
        self.distances = {} # {rover: {from_wp: {to_wp: dist}}}
        for rover in self.rovers:
            self.distances[rover] = self._compute_all_pairs_shortest_paths(rover)

    def _compute_all_pairs_shortest_paths(self, rover):
        """Compute shortest paths for a single rover using BFS."""
        distances = {wp: {other_wp: float('inf') for other_wp in self.waypoints} for wp in self.waypoints}

        for start_wp in self.waypoints:
            distances[start_wp][start_wp] = 0
            queue = deque([(start_wp, 0)])

            visited = {start_wp}

            while queue:
                current_wp, dist = queue.popleft()

                # Check if the rover can traverse from current_wp
                if current_wp in self.waypoint_graph[rover]:
                    for neighbor_wp in self.waypoint_graph[rover][current_wp]:
                        if neighbor_wp in self.waypoints and neighbor_wp not in visited: # Ensure neighbor is a known waypoint
                            visited.add(neighbor_wp)
                            distances[start_wp][neighbor_wp] = dist + 1
                            queue.append((neighbor_wp, dist + 1))

        return distances

    def get_distance(self, rover, from_wp, to_wp):
        """Lookup precomputed distance."""
        if rover not in self.distances or from_wp not in self.distances[rover] or to_wp not in self.distances[rover][from_wp]:
             # This can happen if from_wp or to_wp are not in the set of waypoints
             # used during BFS computation (e.g., malformed state or goal).
             # Treat as unreachable.
             return float('inf')
        return self.distances[rover][from_wp][to_wp]

    def get_rover_location(self, state, rover):
        """Find the current waypoint of a rover in the state."""
        for fact in state:
            if match(fact, "at", rover, "*"):
                parts = get_parts(fact)
                if len(parts) == 3: return parts[2]
        return None # Rover location not found (should not happen in valid states)

    def is_store_empty(self, state, rover):
        """Check if a rover's store is empty in the state."""
        if rover in self.rover_stores:
            store = self.rover_stores[rover]
            return f"(empty {store})" in state
        return False # Rover has no store or store info missing

    def __call__(self, node):
        """Compute the heuristic value for the given state."""
        state = node.state

        # If goal is reached, heuristic is 0
        if self.goals <= state:
            return 0

        total_heuristic_cost = 0

        # Pre-process state for quick lookups
        state_predicates = defaultdict(set)
        for fact in state:
             parts = get_parts(fact)
             if parts:
                 state_predicates[parts[0]].add(tuple(parts[1:]))

        # Get current rover locations
        current_rover_locations = {}
        for rover in self.rovers:
             loc = self.get_rover_location(state, rover)
             if loc:
                 current_rover_locations[rover] = loc

        # Get lander location
        lander_wp = self.lander_waypoint
        if not lander_wp:
             # Lander location is static and should be found in __init__.
             # If not found, communication is impossible.
             # print("Error: Lander location not found in static facts.") # Debugging
             return float('inf')


        # Iterate through each goal fact
        for goal in self.goals:
            if goal in state:
                continue # Goal already achieved

            parts = get_parts(goal)
            if not parts:
                 # print(f"Warning: Skipping malformed goal fact: {goal}") # Debugging
                 continue # Skip malformed goal

            predicate = parts[0]
            min_goal_cost = float('inf')

            if predicate == "communicated_soil_data":
                if len(parts) != 2:
                     # print(f"Warning: Skipping malformed soil goal: {goal}") # Debugging
                     continue
                wp = parts[1]

                # Case 1: have_soil_analysis
                for rover in self.rovers:
                    if ("have_soil_analysis", rover, wp) in state_predicates["have_soil_analysis"]:
                        current_rover_wp = current_rover_locations.get(rover)
                        if current_rover_wp:
                            move_cost = self.get_distance(rover, current_rover_wp, lander_wp)
                            if move_cost != float('inf'):
                                min_goal_cost = min(min_goal_cost, move_cost + 1) # +1 communicate

                # Case 2: have_soil_sample (only if analysis not done)
                if min_goal_cost == float('inf'):
                    for rover in self.rovers:
                        if ("have_soil_sample", rover, wp) in state_predicates["have_soil_sample"]:
                            current_rover_wp = current_rover_locations.get(rover)
                            if current_rover_wp:
                                move_cost1 = self.get_distance(rover, current_rover_wp, wp)
                                move_cost2 = self.get_distance(rover, wp, lander_wp)
                                if move_cost1 != float('inf') and move_cost2 != float('inf'):
                                    min_goal_cost = min(min_goal_cost, move_cost1 + 1 + move_cost2 + 1) # +1 analyze, +1 communicate

                # Case 3: Need to take sample (only if sample not taken)
                if min_goal_cost == float('inf'):
                    if wp in self.sample_locations['soil']: # Check if sample exists statically
                        for rover in self.rovers:
                            if 'soil_analysis' in self.rover_capabilities[rover]:
                                # Check if rover has an empty store
                                if self.is_store_empty(state, rover):
                                     current_rover_wp = current_rover_locations.get(rover)
                                     if current_rover_wp:
                                         move_cost1 = self.get_distance(rover, current_rover_wp, wp)
                                         move_cost2 = self.get_distance(rover, wp, lander_wp)
                                         if move_cost1 != float('inf') and move_cost2 != float('inf'):
                                             # move to wp + take sample + analyze + move to lander + communicate
                                             min_goal_cost = min(min_goal_cost, move_cost1 + 1 + 1 + move_cost2 + 1)


            elif predicate == "communicated_rock_data":
                if len(parts) != 2:
                     # print(f"Warning: Skipping malformed rock goal: {goal}") # Debugging
                     continue
                wp = parts[1]

                # Case 1: have_rock_analysis
                for rover in self.rovers:
                    if ("have_rock_analysis", rover, wp) in state_predicates["have_rock_analysis"]:
                        current_rover_wp = current_rover_locations.get(rover)
                        if current_rover_wp:
                            move_cost = self.get_distance(rover, current_rover_wp, lander_wp)
                            if move_cost != float('inf'):
                                min_goal_cost = min(min_goal_cost, move_cost + 1) # +1 communicate

                # Case 2: have_rock_sample (only if analysis not done)
                if min_goal_cost == float('inf'):
                    for rover in self.rovers:
                        if ("have_rock_sample", rover, wp) in state_predicates["have_rock_sample"]:
                            current_rover_wp = current_rover_locations.get(rover)
                            if current_rover_wp:
                                move_cost1 = self.get_distance(rover, current_rover_wp, wp)
                                move_cost2 = self.get_distance(rover, wp, lander_wp)
                                if move_cost1 != float('inf') and move_cost2 != float('inf'):
                                    min_goal_cost = min(min_goal_cost, move_cost1 + 1 + move_cost2 + 1) # +1 analyze, +1 communicate

                # Case 3: Need to take sample (only if sample not taken)
                if min_goal_cost == float('inf'):
                    if wp in self.sample_locations['rock']: # Check if sample exists statically
                        for rover in self.rovers:
                            if 'rock_analysis' in self.rover_capabilities[rover]:
                                # Check if rover has an empty store
                                if self.is_store_empty(state, rover):
                                     current_rover_wp = current_rover_locations.get(rover)
                                     if current_rover_wp:
                                         move_cost1 = self.get_distance(rover, current_rover_wp, wp)
                                         move_cost2 = self.get_distance(rover, wp, lander_wp)
                                         if move_cost1 != float('inf') and move_cost2 != float('inf'):
                                             # move to wp + take sample + analyze + move to lander + communicate
                                             min_goal_cost = min(min_goal_cost, move_cost1 + 1 + 1 + move_cost2 + 1)

            elif predicate == "communicated_image_data":
                if len(parts) != 3:
                     # print(f"Warning: Skipping malformed image goal: {goal}") # Debugging
                     continue
                obj, mode = parts[1], parts[2]

                # Case 1: have_image
                for rover in self.rovers:
                    if ("have_image", rover, obj, mode) in state_predicates["have_image"]:
                        current_rover_wp = current_rover_locations.get(rover)
                        if current_rover_wp:
                            move_cost = self.get_distance(rover, current_rover_wp, lander_wp)
                            if move_cost != float('inf'):
                                min_goal_cost = min(min_goal_cost, move_cost + 1) # +1 communicate

                # Case 2: Need to take image (only if image not taken)
                if min_goal_cost == float('inf'):
                    # Find suitable rover/camera/waypoint combinations
                    suitable_combinations = [] # List of (rover, camera, image_wp)
                    for rover in self.rovers:
                         if 'imaging' in self.rover_capabilities[rover]:
                             # Find cameras on this rover
                             for camera, cam_info in self.camera_info.items():
                                 if cam_info['on_board'] == rover and \
                                    mode in cam_info['supports'] and \
                                    cam_info['calibration_target'] == obj:
                                     # Find visible waypoints for objective
                                     if obj in self.objective_visibility:
                                         for image_wp in self.objective_visibility[obj]:
                                             current_rover_wp = current_rover_locations.get(rover)
                                             if current_rover_wp:
                                                 suitable_combinations.append((rover, camera, image_wp, current_rover_wp))

                    for rover, camera, image_wp, current_rover_wp in suitable_combinations:
                         # Cost: move to image_wp + calibrate + take image + move to lander + communicate
                         move_cost1 = self.get_distance(rover, current_rover_wp, image_wp)
                         move_cost2 = self.get_distance(rover, image_wp, lander_wp)
                         if move_cost1 != float('inf') and move_cost2 != float('inf'):
                             min_goal_cost = min(min_goal_cost, move_cost1 + 1 + 1 + move_cost2 + 1)

            else:
                 # Ignore goal types not handled by this heuristic
                 # print(f"Warning: Unhandled goal predicate: {goal}") # Debugging
                 # If we encounter an unhandled goal, we cannot estimate its cost.
                 # To be safe, assume it's unreachable if it's not already achieved.
                 # If it were achievable, a different heuristic might handle it.
                 # For this domain-specific heuristic, we assume goals are one of the three types.
                 # If a goal is not one of these types and not in the state, it's effectively unreachable by this heuristic's logic.
                 min_goal_cost = float('inf')


            # Add cost for this goal to total
            if min_goal_cost == float('inf'):
                # If any goal is unreachable, the whole state is likely on an unreachable path
                return float('inf') # Use infinity for unreachable

            total_heuristic_cost += min_goal_cost

        return total_heuristic_cost
