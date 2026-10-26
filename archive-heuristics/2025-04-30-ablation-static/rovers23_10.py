# Assume this is available from the planning framework
# from heuristics.heuristic_base import Heuristic

from fnmatch import fnmatch
from collections import deque

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact string or malformed fact
    if not fact or fact[0] != '(' or fact[-1] != ')':
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
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# Assume Heuristic base class is available from heuristics.heuristic_base
# If running standalone for testing, define a dummy base class:
# class Heuristic:
#     def __init__(self, task):
#         self.goals = task.goals
#         self.static = task.static
#     def __call__(self, node):
#         raise NotImplementedError


class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve
    each uncommunicated goal fact (soil data, rock data, image data).
    It sums up the estimated costs for all unachieved goals.
    The cost for each goal is estimated based on whether the required
    data has been collected and the navigation steps needed to collect
    and communicate the data.

    # Assumptions
    - All rovers can traverse any edge marked as visible between waypoints.
      (This simplifies navigation cost calculation).
    - The cost of navigation between waypoints is the shortest path distance
      in the waypoint graph.
    - The heuristic does not track individual rover assignments to goals
      or complex resource constraints (like multiple samples requiring
      multiple store uses or drops). It estimates the minimum cost
      assuming *a* suitable rover can perform the task.
    - Camera calibration state is considered; re-calibration after taking
      an image is implicitly handled by requiring calibration if the image
      is not yet taken.
    - Store capacity is considered by adding a potential `drop` action cost
      if any suitable rover's store is full when sampling is needed.
    - Navigation costs for collecting image data (calibration + taking image)
      are estimated as the sum of minimum distances from the current location
      to a calibration waypoint and from the current location to an image waypoint.
      This is a simplification of the actual path (curr -> calib -> image).

    # Heuristic Initialization
    - Parse static facts to build the navigation graph, identify lander
      location, sample locations, objective visibility, camera properties,
      rover capabilities, and store ownership.
    - Precompute all-pairs shortest paths between waypoints using BFS.
    - Identify communication waypoints (visible from the lander).
    - Store goal facts for quick lookup (implicitly done by inheriting from Heuristic).

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Initialize total heuristic cost to 0.
    2. Extract current state information: rover locations, collected data (`have_...`),
       store status (`full`), camera calibration status (`calibrated`).
    3. For each goal fact `g` in `self.goals`:
       - If `g` is already in the current state, continue to the next goal.
       - Initialize `goal_cost` for this specific unachieved goal.
       - Add base cost for the final communication action (1).
       - Check if the required data (`have_...`) for `g` is present in the state for *any* rover.
       - If the data is *not* present:
         - Add base cost for the action to collect the data (sample/take_image).
         - If sampling (soil/rock):
           - Add potential cost for `drop` (1) if any suitable rover's store is full.
           - Estimate navigation cost to the sample location: Find the minimum distance
             from *any* suitable rover's current location to the sample waypoint. Add this.
         - If imaging:
           - Check if calibration is needed (i.e., no suitable camera is calibrated).
           - If calibration is needed: Add base cost for `calibrate` (1).
           - Estimate navigation cost for calibration: Find the minimum distance
             from *any* suitable imaging-equipped rover with a suitable camera
             from its current location to *any* valid calibration waypoint. Add this.
           - Estimate navigation cost for taking the image: Find the minimum distance
             from *any* suitable imaging-equipped rover with a suitable camera
             from its current location to *any* valid image waypoint for the objective. Add this.
       - Estimate navigation cost for communication: Find the minimum distance
         from the current location of *any* rover that *has* (or *could obtain*)
         the required data to *any* communication waypoint. Add this.
       - If any necessary static condition (e.g., equipped rover, reachable waypoint)
         is missing for this goal, add a large penalty to `total_cost` and skip
         further calculation for this goal.
       - Add `goal_cost` to `total_cost`.
    4. Return the total heuristic cost.

    Note: The heuristic sums costs for *each* unachieved goal independently,
    potentially overcounting shared subgoals (like navigation to a common
    waypoint or using the same rover/store/camera). This is acceptable for a
    non-admissible heuristic aiming for greedy search performance.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        # self.goals = task.goals # Inherited from Heuristic
        # self.static = task.static # Inherited from Heuristic

        # --- Precompute Navigation Graph and Shortest Paths ---
        self.waypoints = set()
        self.navigation_graph = {} # waypoint -> set of connected waypoints
        self.lander_location = None
        self.communication_waypoints = set() # waypoints visible from lander

        # Assuming all rovers can traverse any visible edge for simplicity
        # If can_traverse is rover-specific and matters, need graph per rover
        # For this heuristic, let's stick to a single graph based on visible
        # and assume any rover can use it.

        for fact in task.static:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            if match(fact, "visible", "*", "*"):
                w1, w2 = parts[1], parts[2]
                self.waypoints.add(w1)
                self.waypoints.add(w2)
                self.navigation_graph.setdefault(w1, set()).add(w2)
                self.navigation_graph.setdefault(w2, set()).add(w1) # Assuming visible is symmetric

            elif match(fact, "at_lander", "*", "*"):
                self.lander_location = parts[2]

        # Identify communication waypoints
        if self.lander_location:
             for fact in task.static:
                 parts = get_parts(fact)
                 if match(fact, "visible", "*", self.lander_location):
                     self.communication_waypoints.add(parts[1])
                 elif match(fact, "visible", self.lander_location, "*"):
                     self.communication_waypoints.add(parts[2])

        # Ensure lander location is in waypoints if it exists
        if self.lander_location:
             self.waypoints.add(self.lander_location)


        # Compute all-pairs shortest paths using BFS
        self.shortest_paths = {} # (w1, w2) -> distance
        for start_node in list(self.waypoints): # Use list to iterate over a copy
            q = deque([(start_node, 0)])
            visited = {start_node}
            self.shortest_paths[(start_node, start_node)] = 0

            while q:
                current_node, dist = q.popleft()

                if current_node in self.navigation_graph:
                    for neighbor in self.navigation_graph[current_node]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            self.shortest_paths[(start_node, neighbor)] = dist + 1
                            q.append((neighbor, dist + 1))

        # --- Extract other static information ---
        self.soil_sample_locations = set()
        self.rock_sample_locations = set()
        self.objective_visible_from = {} # objective -> set of waypoints
        self.camera_calibration_target = {} # camera -> objective
        self.camera_on_board = {} # camera -> rover
        self.camera_supports_mode = {} # camera -> set of modes
        self.rover_equipped_soil = set()
        self.rover_equipped_rock = set()
        self.rover_equipped_imaging = set()
        self.store_of_rover = {} # store -> rover

        for fact in task.static:
            parts = get_parts(fact)
            if not parts: continue

            if match(fact, "at_soil_sample", "*"):
                self.soil_sample_locations.add(parts[1])
            elif match(fact, "at_rock_sample", "*"):
                self.rock_sample_locations.add(parts[1])
            elif match(fact, "visible_from", "*", "*"):
                obj, wp = parts[1], parts[2]
                self.objective_visible_from.setdefault(obj, set()).add(wp)
            elif match(fact, "calibration_target", "*", "*"):
                camera, obj = parts[1], parts[2]
                self.camera_calibration_target[camera] = obj
            elif match(fact, "on_board", "*", "*"):
                camera, rover = parts[1], parts[2]
                self.camera_on_board[camera] = rover
            elif match(fact, "supports", "*", "*"):
                camera, mode = parts[1], parts[2]
                self.camera_supports_mode.setdefault(camera, set()).add(mode)
            elif match(fact, "equipped_for_soil_analysis", "*"):
                self.rover_equipped_soil.add(parts[1])
            elif match(fact, "equipped_for_rock_analysis", "*"):
                self.rover_equipped_rock.add(parts[1])
            elif match(fact, "equipped_for_imaging", "*"):
                self.rover_equipped_imaging.add(parts[1])
            elif match(fact, "store_of", "*", "*"):
                store, rover = parts[1], parts[2]
                self.store_of_rover[store] = rover

        # Map rover to its store (assuming one store per rover)
        self.rover_store = {v: k for k, v in self.store_of_rover.items()}


    def get_distance(self, w1, w2):
        """Get shortest path distance between two waypoints."""
        # Return infinity if no path exists or waypoints are unknown
        return self.shortest_paths.get((w1, w2), float('inf'))

    def get_min_dist_to_set(self, start_wp, target_wps):
        """Get minimum distance from start_wp to any waypoint in target_wps."""
        if not target_wps or start_wp not in self.waypoints:
            return float('inf') # No valid targets or start waypoint is invalid

        min_dist = float('inf')
        for target_wp in target_wps:
            dist = self.get_distance(start_wp, target_wp)
            min_dist = min(min_dist, dist)
        return min_dist


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # If goal is reached, heuristic is 0
        if self.goals <= state:
            return 0

        total_cost = 0
        IMPOSSIBLE_PENALTY = 1000 # Penalty for goals that seem impossible based on static/state info

        # Extract current state information
        rover_locations = {} # rover -> waypoint
        rover_has_soil = set() # {(rover, waypoint), ...}
        rover_has_rock = set() # {(rover, waypoint), ...}
        rover_has_image = set() # {(rover, objective, mode), ...}
        store_is_full = set() # {store, ...}
        camera_calibrated = set() # {(camera, rover), ...}

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            if match(fact, "at", "*", "*"):
                # Check if the first part is a rover object name (more robust check needed in real parser)
                # For this heuristic, assume 'rover' prefix implies a rover object
                if parts[1].startswith('rover'):
                     rover, wp = parts[1], parts[2]
                     rover_locations[rover] = wp
            elif match(fact, "have_soil_analysis", "*", "*"):
                rover, wp = parts[1], parts[2]
                rover_has_soil.add((rover, wp))
            elif match(fact, "have_rock_analysis", "*", "*"):
                rover, wp = parts[1], parts[2]
                rover_has_rock.add((rover, wp))
            elif match(fact, "have_image", "*", "*", "*"):
                rover, obj, mode = parts[1], parts[2], parts[3]
                rover_has_image.add((rover, obj, mode))
            elif match(fact, "full", "*"):
                 store_is_full.add(parts[1])
            elif match(fact, "calibrated", "*", "*"):
                camera, rover = parts[1], parts[2]
                camera_calibrated.add((camera, rover))

        # Estimate cost for each unachieved goal
        for goal in self.goals:
            if goal in state:
                continue # Goal already achieved

            parts = get_parts(goal)
            if not parts: continue

            predicate = parts[0]

            if predicate == "communicated_soil_data":
                wp = parts[1]
                goal_cost = 0 # Cost for this specific goal

                # Cost for communication action
                goal_cost += 1

                # Check if soil data is collected by any rover
                soil_collected = any((r, wp) in rover_has_soil for r in self.rover_equipped_soil)

                if not soil_collected:
                    # Cost for sample action
                    goal_cost += 1
                    # Navigation cost to sample location
                    min_nav_to_sample = float('inf')
                    needs_drop = False
                    suitable_rovers = list(self.rover_equipped_soil)

                    if not suitable_rovers:
                         total_cost += IMPOSSIBLE_PENALTY # Cannot sample if no rover is equipped
                         continue # Cannot sample/communicate this goal

                    if wp not in self.soil_sample_locations:
                         total_cost += IMPOSSIBLE_PENALTY # Cannot sample if no sample at waypoint
                         continue # Cannot sample/communicate this goal

                    for rover in suitable_rovers:
                        current_wp = rover_locations.get(rover)
                        if current_wp:
                            dist = self.get_distance(current_wp, wp)
                            min_nav_to_sample = min(min_nav_to_sample, dist)
                            # Check if this rover's store is full
                            store = self.rover_store.get(rover)
                            if store and store in store_is_full:
                                needs_drop = True # At least one suitable rover needs a drop

                    if min_nav_to_sample != float('inf'):
                         goal_cost += min_nav_to_sample
                    else:
                         total_cost += IMPOSSIBLE_PENALTY # Cannot reach sample location
                         continue # Cannot sample/communicate this goal

                    if needs_drop:
                         goal_cost += 1 # Add cost for drop if any suitable rover needs it

                # Navigation cost to communication waypoint
                min_nav_to_comm = float('inf')
                communicating_rovers = set()
                rovers_with_data = {r for r, w in rover_has_soil if w == wp}

                if rovers_with_data:
                     # If data exists, only rovers with the data can communicate it
                     communicating_rovers = rovers_with_data
                else:
                     # If data doesn't exist, any equipped rover *could* get it and communicate it
                     communicating_rovers = self.rover_equipped_soil

                if not communicating_rovers:
                     total_cost += IMPOSSIBLE_PENALTY # Should be caught earlier if sampling is impossible
                     continue # Cannot communicate

                if not self.communication_waypoints:
                     total_cost += IMPOSSIBLE_PENALTY # No communication waypoints
                     continue # Cannot communicate

                for rover in communicating_rovers:
                     current_wp = rover_locations.get(rover)
                     if current_wp:
                         dist = self.get_min_dist_to_set(current_wp, self.communication_waypoints)
                         min_nav_to_comm = min(min_nav_to_comm, dist)

                if min_nav_to_comm != float('inf'):
                     goal_cost += min_nav_to_comm
                else:
                     total_cost += IMPOSSIBLE_PENALTY # Cannot reach communication waypoint
                     continue # Cannot communicate


            elif predicate == "communicated_rock_data":
                wp = parts[1]
                goal_cost = 0 # Cost for this specific goal

                # Cost for communication action
                goal_cost += 1

                # Check if rock data is collected by any rover
                rock_collected = any((r, wp) in rover_has_rock for r in self.rover_equipped_rock)

                if not rock_collected:
                    # Cost for sample action
                    goal_cost += 1
                    # Navigation cost to sample location
                    min_nav_to_sample = float('inf')
                    needs_drop = False
                    suitable_rovers = list(self.rover_equipped_rock)

                    if not suitable_rovers:
                         total_cost += IMPOSSIBLE_PENALTY # Cannot sample if no rover is equipped
                         continue # Cannot sample/communicate this goal

                    if wp not in self.rock_sample_locations:
                         total_cost += IMPOSSIBLE_PENALTY # Cannot sample if no sample at waypoint
                         continue # Cannot sample/communicate this goal

                    for rover in suitable_rovers:
                        current_wp = rover_locations.get(rover)
                        if current_wp:
                            dist = self.get_distance(current_wp, wp)
                            min_nav_to_sample = min(min_nav_to_sample, dist)
                            # Check if this rover's store is full
                            store = self.rover_store.get(rover)
                            if store and store in store_is_full:
                                needs_drop = True

                    if min_nav_to_sample != float('inf'):
                         goal_cost += min_nav_to_sample
                    else:
                         total_cost += IMPOSSIBLE_PENALTY # Cannot reach sample location
                         continue # Cannot sample/communicate this goal

                    if needs_drop:
                         goal_cost += 1

                # Navigation cost to communication waypoint
                min_nav_to_comm = float('inf')
                communicating_rovers = set()
                rovers_with_data = {r for r, w in rover_has_rock if w == wp}

                if rovers_with_data:
                     communicating_rovers = rovers_with_data
                else:
                     communicating_rovers = self.rover_equipped_rock

                if not communicating_rovers:
                     total_cost += IMPOSSIBLE_PENALTY # Should be caught earlier
                     continue # Cannot communicate

                if not self.communication_waypoints:
                     total_cost += IMPOSSIBLE_PENALTY # No communication waypoints
                     continue # Cannot communicate

                for rover in communicating_rovers:
                     current_wp = rover_locations.get(rover)
                     if current_wp:
                         dist = self.get_min_dist_to_set(current_wp, self.communication_waypoints)
                         min_nav_to_comm = min(min_nav_to_comm, dist)

                if min_nav_to_comm != float('inf'):
                     goal_cost += min_nav_to_comm
                else:
                     total_cost += IMPOSSIBLE_PENALTY # Cannot reach communication waypoint
                     continue # Cannot communicate


            elif predicate == "communicated_image_data":
                obj, mode = parts[1], parts[2]
                goal_cost = 0 # Cost for this specific goal

                # Cost for communication action
                goal_cost += 1

                # Check if image data is collected by any rover
                image_collected = any((r, obj, mode) in rover_has_image for r in self.rover_equipped_imaging)

                if not image_collected:
                    # Cost for take_image action
                    goal_cost += 1

                    # Find suitable cameras/rovers for this objective/mode
                    suitable_camera_rovers = [] # List of (camera, rover) tuples
                    for camera, supported_modes in self.camera_supports_mode.items():
                        if mode in supported_modes:
                            rover = self.camera_on_board.get(camera)
                            if rover and rover in self.rover_equipped_imaging:
                                suitable_camera_rovers.append((camera, rover))

                    if not suitable_camera_rovers:
                         total_cost += IMPOSSIBLE_PENALTY # Cannot take image if no suitable camera/rover
                         continue # Cannot take image/communicate this goal

                    # Check if any suitable camera is calibrated on its rover
                    calibrated_suitable_camera_exists = any(
                        (camera, rover) in camera_calibrated
                        for camera, rover in suitable_camera_rovers
                    )

                    if not calibrated_suitable_camera_exists:
                        # Cost for calibrate action
                        goal_cost += 1

                        # Navigation cost to calibration waypoint
                        min_nav_to_calib = float('inf')
                        has_calib_waypoint = False
                        for camera, rover in suitable_camera_rovers:
                            current_wp = rover_locations.get(rover)
                            calib_target = self.camera_calibration_target.get(camera)
                            if current_wp and calib_target:
                                calib_wps = self.objective_visible_from.get(calib_target, set())
                                if calib_wps: # Ensure there are valid calibration waypoints
                                    has_calib_waypoint = True
                                    dist = self.get_min_dist_to_set(current_wp, calib_wps)
                                    min_nav_to_calib = min(min_nav_to_calib, dist)

                        if not has_calib_waypoint:
                             total_cost += IMPOSSIBLE_PENALTY # Cannot calibrate if no calib waypoint exists for any suitable camera
                             continue # Cannot calibrate/take image/communicate

                        if min_nav_to_calib != float('inf'):
                             goal_cost += min_nav_to_calib
                        else:
                             total_cost += IMPOSSIBLE_PENALTY # Cannot reach calibration waypoint
                             continue # Cannot calibrate/take image/communicate


                    # Navigation cost to image waypoint
                    min_nav_to_image = float('inf')
                    has_image_waypoint = False
                    image_wps = self.objective_visible_from.get(obj, set())

                    if not image_wps:
                         total_cost += IMPOSSIBLE_PENALTY # Cannot take image if no image waypoint exists for the objective
                         continue # Cannot take image/communicate this goal

                    for camera, rover in suitable_camera_rovers:
                         current_wp = rover_locations.get(rover)
                         if current_wp:
                             has_image_waypoint = True
                             dist = self.get_min_dist_to_set(current_wp, image_wps)
                             min_nav_to_image = min(min_nav_to_image, dist)

                    if not has_image_waypoint:
                         total_cost += IMPOSSIBLE_PENALTY # Cannot take image if no suitable rover can reach an image waypoint
                         continue # Cannot take image/communicate this goal

                    if min_nav_to_image != float('inf'):
                         goal_cost += min_nav_to_image
                    else:
                         total_cost += IMPOSSIBLE_PENALTY # Cannot reach image waypoint
                         continue # Cannot take image/communicate this goal


                # Navigation cost to communication waypoint
                min_nav_to_comm = float('inf')
                communicating_rovers = set()
                rovers_with_data = {r for r, o, m in rover_has_image if o == obj and m == mode}

                if rovers_with_data:
                     communicating_rovers = rovers_with_data
                else:
                     # If data doesn't exist, any imaging-equipped rover with a suitable camera could get it
                     suitable_camera_rovers = []
                     for camera, supported_modes in self.camera_supports_mode.items():
                         if mode in supported_modes:
                             rover = self.camera_on_board.get(camera)
                             if rover and rover in self.rover_equipped_imaging:
                                 suitable_camera_rovers.append((camera, rover))
                     communicating_rovers = {r for c, r in suitable_camera_rovers}

                if not communicating_rovers:
                     total_cost += IMPOSSIBLE_PENALTY # Should be caught earlier
                     continue # Cannot communicate

                if not self.communication_waypoints:
                     total_cost += IMPOSSIBLE_PENALTY # No communication waypoints
                     continue # Cannot communicate

                for rover in communicating_rovers:
                     current_wp = rover_locations.get(rover)
                     if current_wp:
                         dist = self.get_min_dist_to_set(current_wp, self.communication_waypoints)
                         min_nav_to_comm = min(min_nav_to_comm, dist)

                if min_nav_to_comm != float('inf'):
                     goal_cost += min_nav_to_comm
                else:
                     total_cost += IMPOSSIBLE_PENALTY # Cannot reach communication waypoint
                     continue # Cannot communicate


            else:
                 # Ignore other potential goal types not related to communication
                 # (the domain only defines communicated_... goals)
                 continue

            total_cost += goal_cost

        return total_cost
