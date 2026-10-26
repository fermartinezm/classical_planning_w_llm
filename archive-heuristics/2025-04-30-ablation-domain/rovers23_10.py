import collections
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and starts/ends with parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        # Handle cases like frozenset elements which are already strings
        # If it's a string but doesn't look like a fact, return as is or handle error
        return [fact] # Or raise an error, depending on expected input
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Check if the number of parts matches the number of arguments in the pattern
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to satisfy all
    goal conditions. It focuses on the main goal types: communicating soil,
    rock, mineral, and image data. For each unsatisfied goal, it estimates
    the minimum cost to achieve it by considering the current state of relevant
    objects (rovers, samples, images, calibration) and the minimum movement
    cost between necessary locations (sample/image/calibration location,
    communication location). The total heuristic is the sum of the estimated
    costs for all unsatisfied goals.

    # Assumptions
    - Actions have a cost of 1. Movement cost is estimated by shortest path
      distance on the waypoint graph.
    - Sampling (soil, rock, mineral) requires being at the sample location
      with an equipped rover and an empty store (store capacity is ignored
      for simplicity in cost calculation, only the 'empty' state is considered
      if present, but the heuristic doesn't strictly enforce it).
    - Analysis (soil, rock, mineral) requires having the sample and being
      at the sample location (simplified to rover's current location if sample
      is on board).
    - Taking an image requires being at a waypoint visible from the objective,
      having an equipped rover with a camera supporting the mode, and the
      camera being calibrated.
    - Calibration requires being at a waypoint visible from the calibration
      target objective, having the camera on board, and the camera having
      a calibration target defined.
    - Communication requires having the data (analysis or image) and being
      at a waypoint visible from the lander's location.
    - The lander's location is static.
    - The heuristic sums the costs for each unsatisfied goal independently,
      ignoring potential synergies (e.g., one move serving multiple goals).
    - If a required location is unreachable by any suitable rover, a large
      penalty is added for that goal.

    # Heuristic Initialization
    The heuristic pre-computes and stores static information from the task:
    - Lander's location.
    - Waypoint visibility graph.
    - Rover-specific traversal graphs.
    - All-pairs shortest path distances for each rover on its traversal graph.
    - Rover capabilities (soil, rock, mineral, imaging).
    - Mapping of rovers to stores and cameras.
    - Mapping of cameras to supported modes and calibration targets.
    - Mapping of objectives to waypoints they are visible from.
    - Locations of soil, rock, and mineral samples.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Parse the current state to identify:
       - Rover locations.
       - Empty stores.
       - Calibrated cameras (on which rover).
       - Samples held by rovers.
       - Images held by rovers.
       - Analyses performed.
       - Goals already communicated.
    2. Initialize total heuristic cost `h = 0`.
    3. Identify the lander's location and the set of waypoints visible from it (communication waypoints).
    4. For each goal fact in the task's goals:
       a. If the goal fact is already true in the current state, add 0 to `h`.
       b. If the goal fact is not true, estimate the minimum cost to achieve it:
          i.  Determine the type of goal (soil, rock, mineral, image).
          ii. Find all rovers/cameras capable of achieving this goal type.
          iii.Calculate the minimum cost across all suitable rovers/cameras by considering the stages required (e.g., sample -> analyse -> communicate for soil) and the minimum movement cost between the locations needed for each stage (current location -> sample/cal/image location -> communication location). Use pre-computed shortest path distances for movement costs.
          iv. If the necessary samples, analyses, or images already exist on a rover, the initial stages are skipped, and the cost calculation starts from the earliest uncompleted stage.
          v. If any required location is unreachable by all suitable rovers, assign a large penalty cost for this goal.
          vi. Add the minimum estimated cost for this goal to `h`.
    5. Return the total heuristic cost `h`.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals

        # --- Pre-compute Static Information ---
        self.lander_location = None
        self.waypoint_visibility = collections.defaultdict(set)
        self.rover_can_traverse_graph = collections.defaultdict(lambda: collections.defaultdict(set))
        self.all_waypoints = set() # Collect all waypoints mentioned in static facts

        self.rover_capabilities = collections.defaultdict(set) # e.g., {'rover1': {'soil', 'imaging'}}
        self.rover_cameras = collections.defaultdict(set)      # e.g., {'rover1': {'camera1'}}
        self.camera_supports = collections.defaultdict(set)    # e.g., {'camera1': {'high_res', 'low_res'}}
        self.camera_calibration_target = {}                  # e.g., {'camera1': 'objective1'}
        self.rover_stores = {}                               # e.g., {'rover1': 'rover1store'}

        self.soil_sample_locations = set()
        self.rock_sample_locations = set()
        self.mineral_sample_locations = set()
        self.objective_visible_from = collections.defaultdict(set) # e.g., {'objective1': {'waypoint1', 'waypoint2'}}

        for fact in task.static:
            parts = get_parts(fact)
            if not parts: continue # Skip empty or malformed facts

            predicate = parts[0]
            args = parts[1:]

            if predicate == "at_lander" and len(args) == 2:
                self.lander_location = args[1]
            elif predicate == "visible" and len(args) == 2:
                wp1, wp2 = args
                self.waypoint_visibility[wp1].add(wp2)
                self.waypoint_visibility[wp2].add(wp1) # Visible is symmetric
                self.all_waypoints.add(wp1)
                self.all_waypoints.add(wp2)
            elif predicate == "can_traverse" and len(args) == 3:
                rover, from_wp, to_wp = args
                self.rover_can_traverse_graph[rover][from_wp].add(to_wp)
                self.all_waypoints.add(from_wp)
                self.all_waypoints.add(to_wp)
            elif predicate.startswith("equipped_for_") and len(args) == 1:
                capability = predicate.split("_")[2] # soil, rock, imaging, mineral (if present)
                self.rover_capabilities[args[0]].add(capability)
            elif predicate == "store_of" and len(args) == 2:
                self.rover_stores[args[1]] = args[0] # rover -> store
            elif predicate == "on_board" and len(args) == 2:
                self.rover_cameras[args[1]].add(args[0]) # rover -> camera
            elif predicate == "supports" and len(args) == 2:
                self.camera_supports[args[0]].add(args[1]) # camera -> mode
            elif predicate == "calibration_target" and len(args) == 2:
                self.camera_calibration_target[args[0]] = args[1] # camera -> objective
            elif predicate == "visible_from" and len(args) == 2:
                self.objective_visible_from[args[0]].add(args[1]) # objective -> waypoint
            elif predicate == "at_soil_sample" and len(args) == 1:
                self.soil_sample_locations.add(args[0])
            elif predicate == "at_rock_sample" and len(args) == 1:
                self.rock_sample_locations.add(args[0])
            elif predicate == "at_mineral_sample" and len(args) == 1:
                 self.mineral_sample_locations.add(args[0])

        # Ensure all waypoints from visibility are also in the set
        for wp_set in self.waypoint_visibility.values():
             self.all_waypoints.update(wp_set)

        # Pre-compute distances for each rover from each waypoint
        self.rover_waypoint_distances = collections.defaultdict(dict)
        # Iterate over rovers that have traversal facts
        for rover in self.rover_can_traverse_graph:
            # Compute distances from ALL known waypoints
            for start_wp in self.all_waypoints:
                 self.rover_waypoint_distances[rover][start_wp] = self._rover_bfs(rover, start_wp)

    def _rover_bfs(self, rover, start_node):
        """Perform BFS for a specific rover to find distances from start_node."""
        graph = self.rover_can_traverse_graph.get(rover, {}) # Get graph for this rover
        distances = {wp: float('inf') for wp in self.all_waypoints} # Initialize for all waypoints

        if start_node not in distances:
             # Start node is not a known waypoint, impossible to start BFS
             return distances

        distances[start_node] = 0
        queue = collections.deque([start_node])

        while queue:
            current_node = queue.popleft()

            # Check if current_node has outgoing edges for this rover
            if current_node in graph:
                for neighbor in graph[current_node]:
                    if distances[neighbor] == float('inf'):
                        distances[neighbor] = distances[current_node] + 1
                        queue.append(neighbor)
        return distances

    def get_distance(self, rover, from_wp, to_wp):
        """Get the pre-computed shortest distance for a rover between two waypoints."""
        if rover not in self.rover_waypoint_distances: return float('inf')
        if from_wp not in self.rover_waypoint_distances[rover]: return float('inf')
        return self.rover_waypoint_distances[rover][from_wp].get(to_wp, float('inf'))

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        total_cost = 0
        PENALTY = 1000 # Large cost for seemingly impossible goals

        # --- Parse Current State Information ---
        current_rover_locations = {}
        current_empty_stores = set()
        current_calibrated_cameras = set() # Store as (camera, rover) tuple
        current_have_soil_sample = set() # Store as (rover, waypoint) tuple
        current_have_rock_sample = set() # Store as (rover, waypoint) tuple
        current_have_mineral_sample = set() # Store as (rover, waypoint) tuple
        current_have_image = set() # Store as (rover, objective, mode) tuple
        current_soil_analysis = set() # Store as (waypoint, rover) tuple
        current_rock_analysis = set() # Store as (waypoint, rover) tuple
        current_mineral_analysis = set() # Store as (waypoint, rover) tuple
        current_communicated_goals = set() # Store as tuple of predicate and args

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue
            predicate = parts[0]
            args = parts[1:]

            if predicate == "at" and len(args) == 2 and args[0].startswith("rover"):
                current_rover_locations[args[0]] = args[1]
            elif predicate == "empty" and len(args) == 1 and args[0].endswith("store"):
                current_empty_stores.add(args[0])
            elif predicate == "calibrated" and len(args) == 2 and args[0].startswith("camera") and args[1].startswith("rover"):
                current_calibrated_cameras.add(tuple(args))
            elif predicate == "have_soil_sample" and len(args) == 2 and args[0].startswith("rover") and args[1].startswith("waypoint"):
                current_have_soil_sample.add(tuple(args))
            elif predicate == "have_rock_sample" and len(args) == 2 and args[0].startswith("rover") and args[1].startswith("waypoint"):
                current_have_rock_sample.add(tuple(args))
            elif predicate == "have_mineral_sample" and len(args) == 2 and args[0].startswith("rover") and args[1].startswith("waypoint"):
                current_have_mineral_sample.add(tuple(args))
            elif predicate == "have_image" and len(args) == 3 and args[0].startswith("rover") and args[1].startswith("objective") and args[2] in self.camera_supports.get(args[0], {args[2]}): # Check if mode is valid (approx)
                 current_have_image.add(tuple(args))
            elif predicate == "soil_analysis" and len(args) == 2 and args[0].startswith("waypoint") and args[1].startswith("rover"):
                current_soil_analysis.add(tuple(args))
            elif predicate == "rock_analysis" and len(args) == 2 and args[0].startswith("waypoint") and args[1].startswith("rover"):
                current_rock_analysis.add(tuple(args))
            elif predicate == "mineral_analysis" and len(args) == 2 and args[0].startswith("waypoint") and args[1].startswith("rover"):
                current_mineral_analysis.add(tuple(args))
            elif predicate.startswith("communicated_"):
                 current_communicated_goals.add(tuple(parts))

        # --- Calculate Cost for Unsatisfied Goals ---
        lander_wp = self.lander_location
        comm_wps = self.waypoint_visibility.get(lander_wp, set()) if lander_wp else set()

        for goal_fact_str in self.goals:
            goal_tuple = get_parts(goal_fact_str)
            if goal_tuple in current_communicated_goals:
                continue # Goal already met

            predicate = goal_tuple[0]
            args = goal_tuple[1:]
            min_cost_for_goal = float('inf')

            if predicate == 'communicated_soil_data' and len(args) == 1:
                waypoint = args[0]
                min_cost_for_goal = self._cost_for_sample_goal(
                    state, waypoint, 'soil', current_rover_locations,
                    current_have_soil_sample, current_soil_analysis,
                    self.rover_capabilities, self.soil_sample_locations,
                    comm_wps
                )
            elif predicate == 'communicated_rock_data' and len(args) == 1:
                 waypoint = args[0]
                 min_cost_for_goal = self._cost_for_sample_goal(
                     state, waypoint, 'rock', current_rover_locations,
                     current_have_rock_sample, current_rock_analysis,
                     self.rover_capabilities, self.rock_sample_locations,
                     comm_wps
                 )
            elif predicate == 'communicated_mineral_data' and len(args) == 1:
                 waypoint = args[0]
                 min_cost_for_goal = self._cost_for_sample_goal(
                     state, waypoint, 'mineral', current_rover_locations,
                     current_have_mineral_sample, current_mineral_analysis,
                     self.rover_capabilities, self.mineral_sample_locations,
                     comm_wps
                 )
            elif predicate == 'communicated_image_data' and len(args) == 2:
                objective, mode = args
                min_cost_for_goal = self._cost_for_image_goal(
                    state, objective, mode, current_rover_locations,
                    current_have_image, current_calibrated_cameras,
                    self.rover_capabilities, self.rover_cameras,
                    self.camera_supports, self.camera_calibration_target,
                    self.objective_visible_from, comm_wps
                )
            # Add other goal types if necessary

            total_cost += min_cost_for_goal if min_cost_for_goal != float('inf') else PENALTY

        return total_cost

    def _cost_for_sample_goal(self, state, waypoint, sample_type, current_rover_locations,
                               current_have_sample, current_analysis, rover_capabilities,
                               sample_locations, comm_wps):
        """Estimate cost for a soil/rock/mineral communication goal."""
        min_cost = float('inf')
        equipped_rovers = {r for r, caps in rover_capabilities.items() if sample_type in caps}

        if not equipped_rovers:
            return PENALTY # No rover can perform this task

        if not comm_wps:
             return PENALTY # No communication location

        # Check if analysis exists for any equipped rover
        analysis_exists = any((waypoint, r) in current_analysis for r in equipped_rovers)

        if analysis_exists:
            # Find rover with analysis
            rover_with_analysis = next(r for r in equipped_rovers if (waypoint, r) in current_analysis)
            rover_wp = current_rover_locations.get(rover_with_analysis)
            if not rover_wp: return PENALTY # Rover location unknown

            # Cost = move to comm + communicate
            min_dist_to_comm = float('inf')
            for comm_wp in comm_wps:
                min_dist_to_comm = min(min_dist_to_comm, self.get_distance(rover_with_analysis, rover_wp, comm_wp))

            if min_dist_to_comm != float('inf'):
                min_cost = min(min_cost, min_dist_to_comm + 1) # move + communicate

        else: # Analysis does not exist
            # Check if sample exists for any equipped rover
            sample_exists = any((r, waypoint) in current_have_sample for r in equipped_rovers)

            if sample_exists:
                # Find rover with sample
                rover_with_sample = next(r for r in equipped_rovers if (r, waypoint) in current_have_sample)
                rover_wp = current_rover_locations.get(rover_with_sample)
                if not rover_wp: return PENALTY # Rover location unknown

                # Cost = analyse + move to comm + communicate
                # Assume analysis happens at rover's current location
                cost_this_path = 1 # analyse
                min_dist_to_comm = float('inf')
                for comm_wp in comm_wps:
                    min_dist_to_comm = min(min_dist_to_comm, self.get_distance(rover_with_sample, rover_wp, comm_wp))

                if min_dist_to_comm != float('inf'):
                    cost_this_path += min_dist_to_comm + 1 # move + communicate
                    min_cost = min(min_cost, cost_this_path)
                else:
                    min_cost = min(min_cost, PENALTY) # Cannot reach comm location

            else: # Sample does not exist
                # Need to sample, analyse, communicate
                # Cost = move to sample location + sample + analyse + move to comm + communicate
                sample_location = waypoint # Sample is at this waypoint (static fact)
                if sample_location not in sample_locations:
                     return PENALTY # No sample at this location

                # Find the best rover to do the job (closest equipped rover to sample location)
                best_rover = None
                min_dist_to_sample = float('inf')
                for rover in equipped_rovers:
                    rover_wp = current_rover_locations.get(rover)
                    if not rover_wp: continue
                    dist = self.get_distance(rover, rover_wp, sample_location)
                    if dist < min_dist_to_sample:
                        min_dist_to_sample = dist
                        best_rover = rover

                if best_rover and min_dist_to_sample != float('inf'):
                    cost_this_path = min_dist_to_sample + 1 # move to sample + sample
                    # Rover is now at sample_location after sampling
                    current_wp_after_sample = sample_location
                    cost_this_path += 1 # analyse (at sample location)
                    # Rover is still at sample_location after analysis
                    current_wp_after_analyse = sample_location

                    min_dist_from_analyse_to_comm = float('inf')
                    for comm_wp in comm_wps:
                        min_dist_from_analyse_to_comm = min(min_dist_from_analyse_to_comm, self.get_distance(best_rover, current_wp_after_analyse, comm_wp))

                    if min_dist_from_analyse_to_comm != float('inf'):
                        cost_this_path += min_dist_from_analyse_to_comm + 1 # move to comm + communicate
                        min_cost = min(min_cost, cost_this_path)
                    else:
                        min_cost = min(min_cost, PENALTY) # Cannot reach comm location
                else:
                    min_cost = min(min_cost, PENALTY) # No reachable equipped rover

        return min_cost

    def _cost_for_image_goal(self, state, objective, mode, current_rover_locations,
                             current_have_image, current_calibrated_cameras,
                             rover_capabilities, rover_cameras, camera_supports,
                             camera_calibration_target, objective_visible_from, comm_wps):
        """Estimate cost for an image communication goal."""
        min_cost = float('inf')

        # Find equipped imaging rovers with cameras supporting the mode
        suitable_rover_camera_combos = []
        for rover, caps in rover_capabilities.items():
            if 'imaging' in caps:
                for camera in rover_cameras.get(rover, set()):
                    if mode in camera_supports.get(camera, set()):
                        suitable_rover_camera_combos.append((rover, camera))

        if not suitable_rover_camera_combos:
            return PENALTY # No rover/camera can perform this task

        if not comm_wps:
             return PENALTY # No communication location

        image_wps = objective_visible_from.get(objective, set())
        if not image_wps:
             return PENALTY # Cannot take image if objective not visible from anywhere

        # Check if image exists for any suitable combo
        image_exists = any((r, objective, mode) in current_have_image for r, c in suitable_rover_camera_combos)

        if image_exists:
            # Find a combo that has the image
            rover_with_image = next((r for r, c in suitable_rover_camera_combos if (r, objective, mode) in current_have_image), None)
            if not rover_with_image: return PENALTY # Should not happen

            rover_wp = current_rover_locations.get(rover_with_image)
            if not rover_wp: return PENALTY # Rover location unknown

            # Cost = move to comm + communicate
            min_dist_to_comm = float('inf')
            for comm_wp in comm_wps:
                min_dist_to_comm = min(min_dist_to_comm, self.get_distance(rover_with_image, rover_wp, comm_wp))

            if min_dist_to_comm != float('inf'):
                min_cost = min(min_cost, min_dist_to_comm + 1) # move + communicate
            else:
                min_cost = min(min_cost, PENALTY) # Cannot reach comm location

        else: # Image does not exist
            # Need to calibrate (if needed), take image, communicate
            # Find the best combo and path
            min_cost_full_image_path = float('inf')

            for rover, camera in suitable_rover_camera_combos:
                rover_wp = current_rover_locations.get(rover)
                if not rover_wp: continue

                cost_this_path = 0
                current_wp = rover_wp

                # Calibration stage (if needed)
                if (camera, rover) not in current_calibrated_cameras:
                    cost_this_path += 1 # calibrate action
                    cal_target = camera_calibration_target.get(camera)
                    if not cal_target: continue # Cannot calibrate this camera
                    cal_wps = objective_visible_from.get(cal_target, set())
                    if not cal_wps: continue # No waypoint to calibrate from

                    min_dist_to_cal = float('inf')
                    target_cal_wp = None
                    for cal_wp in cal_wps:
                        dist = self.get_distance(rover, current_wp, cal_wp)
                        if dist < min_dist_to_cal:
                            min_dist_to_cal = dist
                            target_cal_wp = cal_wp

                    if min_dist_to_cal == float('inf'): continue # Cannot reach any cal location
                    cost_this_path += min_dist_to_cal # move to cal
                    current_wp = target_cal_wp # Rover is now at calibration waypoint

                # Take image stage
                cost_this_path += 1 # take_image action
                min_dist_to_image = float('inf')
                target_image_wp = None
                for image_wp in image_wps:
                    dist = self.get_distance(rover, current_wp, image_wp)
                    if dist < min_dist_to_image:
                        min_dist_to_image = dist
                        target_image_wp = image_wp

                if min_dist_to_image == float('inf'): continue # Cannot reach any image location
                cost_this_path += min_dist_to_image # move to image
                current_wp = target_image_wp # Rover is now at image waypoint

                # Communication stage
                cost_this_path += 1 # communicate action
                min_dist_to_comm = float('inf')
                for comm_wp in comm_wps:
                    min_dist_to_comm = min(min_dist_to_comm, self.get_distance(rover, current_wp, comm_wp))

                if min_dist_to_comm == float('inf'): continue # Cannot reach any comm location
                cost_this_path += min_dist_to_comm # move to comm
                # current_wp = comm_wp # Rover is now at communication waypoint

                min_cost_full_image_path = min(min_cost_full_image_path, cost_this_path)

            min_cost = min(min_cost, min_cost_full_image_path)
            if min_cost == float('inf'): min_cost = PENALTY # If no path found for any combo

        return min_cost

