from fnmatch import fnmatch
from collections import deque

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact strings or malformed facts defensively
    if not fact or not isinstance(fact, str) or len(fact) < 2 or fact[0] != '(' or fact[-1] != ')':
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
    # Ensure fact has at least as many parts as args
    if len(parts) < len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def build_graph_and_waypoints(facts, rover_name):
    """Build navigation graph and list of waypoints involved for a specific rover."""
    graph = {}
    waypoints = set()
    for fact in facts:
        parts = get_parts(fact)
        if match(fact, "can_traverse", rover_name, "*", "*"):
            _, r, wp1, wp2 = parts
            waypoints.add(wp1)
            waypoints.add(wp2)
            graph.setdefault(wp1, set()).add(wp2)
    for wp in waypoints:
        graph.setdefault(wp, set())
    return graph, list(waypoints)

def bfs_distances(graph, waypoints):
    """Calculate all-pairs shortest path distances using BFS on a graph."""
    distances = {wp: {other_wp: float('inf') for other_wp in waypoints} for wp in waypoints}
    for start_wp in waypoints:
        distances[start_wp][start_wp] = 0
        queue = deque([start_wp])
        while queue:
            current_wp = queue.popleft()
            dist = distances[start_wp][current_wp]
            if current_wp in graph:
                for neighbor_wp in graph[current_wp]:
                    if neighbor_wp in distances[start_wp] and distances[start_wp][neighbor_wp] == float('inf'):
                        distances[start_wp][neighbor_wp] = dist + 1
                        queue.append(neighbor_wp)
    return distances

# Assume Heuristic base class is provided externally and does not need to be imported/defined here.
# class Heuristic: ...

class roversHeuristic: # Inherit from Heuristic if available in the environment
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve all goal
    conditions. It sums the estimated cost for each unachieved goal fact,
    considering the necessary steps (sampling/imaging, calibration, navigation,
    communication) and minimum navigation distances between relevant waypoints.

    # Assumptions
    - The cost of each action (navigate, sample, drop, calibrate, take_image,
      communicate) is 1.
    - The heuristic calculates the minimum navigation cost using pre-computed
      shortest paths (BFS) for each rover.
    - The heuristic simplifies resource constraints (like rover availability,
      store capacity, camera calibration state) by summing costs per goal
      fact independently, potentially underestimating or overestimating the true cost
      in scenarios with resource contention or shared prerequisites.
    - If a sample (soil/rock) is required for a goal but is no longer at its
      initial location and the analysis is not held by any rover, the goal is
      considered unachievable from this state (infinite cost).
    - The heuristic assumes that if a goal is specified, the necessary static
      conditions (like equipped rovers, cameras, visible_from facts) exist to
      make it potentially achievable.

    # Heuristic Initialization
    - Parses static facts from the task definition.
    - Identifies rovers and their capabilities (soil, rock, imaging).
    - Maps stores to rovers.
    - Maps cameras to rovers, their supported modes, and calibration targets.
    - Identifies lander locations and communication waypoints visible from them.
    - Identifies imaging waypoints for each objective.
    - Identifies calibration waypoints for each camera/target pair.
    - Builds navigation graphs for each rover based on `can_traverse` facts.
    - Pre-computes all-pairs shortest path distances for each rover's navigation
      graph using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Check if the state is a goal state. If yes, return 0.
    2. Initialize total heuristic cost to 0.
    3. Determine the current location of each rover.
    4. Determine which soil/rock samples are still at their initial locations.
    5. Determine which soil/rock analyses and images are currently held by each rover.
    6. Determine which cameras are currently calibrated on each rover.
    7. Iterate through each goal fact specified in the task:
        a. If the goal fact is already true in the current state, continue to the next goal.
        b. If the goal is `(communicated_soil_data ?w)`:
            i. If any equipped rover `?r` holds `(have_soil_analysis ?r ?w)`:
               - Cost = 1 (communicate) + min navigation cost from `?r`'s current location to any communication waypoint.
            ii. If no rover holds the analysis:
               - If `(at_soil_sample ?w)` is in the state:
                 - Find the minimum cost over all equipped rovers `?r`:
                   - Cost = navigation cost from `?r`'s current location to `?w` + 1 (sample) + navigation cost from `?w` to any communication waypoint + 1 (communicate). (Store constraint is ignored).
               - If `(at_soil_sample ?w)` is NOT in the state, this goal is likely unachievable from this state; assign infinite cost for this goal.
            iii. Add the minimum calculated cost for this goal to the total cost.
        c. If the goal is `(communicated_rock_data ?w)`: Similar logic as soil data, using rock-specific predicates and capabilities.
        d. If the goal is `(communicated_image_data ?o ?m)`:
            i. If any equipped rover `?r` holds `(have_image ?r ?o ?m)`:
               - Cost = 1 (communicate) + min navigation cost from `?r`'s current location to any communication waypoint.
            ii. If no rover holds the image:
               - Find the minimum cost over all equipped rovers `?r` and suitable cameras `?i` on board `?r` supporting `?m`:
                 - Find calibration target `?t` for `?i`.
                 - Find calibration waypoints `W_calib` for `(?i, ?t)` and imaging waypoints `W_image` for `?o`. If either set is empty, this rover/camera path is impossible.
                 - Calculate cost assuming calibration is needed:
                   - Nav cost to calib = min nav cost from `?r`'s current location to any `w_calib` in `W_calib`.
                   - Action cost 1 = 1 (calibrate).
                   - Nav cost calib to image = min nav cost from any `w_calib` in `W_calib` to any `w_image` in `W_image`.
                   - Action cost 2 = 1 (take_image).
                   - Nav cost image to comm = min nav cost from any `w_image` in `W_image` to any communication waypoint.
                   - Action cost 3 = 1 (communicate).
                   - Total path cost = Nav cost to calib + Action cost 1 + Nav cost calib to image + Action cost 2 + Nav cost image to comm + Action cost 3.
                 - If `(calibrated ?i ?r)` is true in the state:
                   - Calculate cost assuming existing calibration is used:
                     - Nav cost to image = min nav cost from `?r`'s current location to any `w_image` in `W_image`.
                     - Action cost = 1 (take_image) + 1 (communicate).
                     - Nav cost image to comm = min nav cost from any `w_image` in `W_image` to any communication waypoint.
                     - Total path cost = Nav cost to image + Action cost + Nav cost image to comm.
                   - Take the minimum of the two path costs (with or without existing calibration).
                 - If `(calibrated ?i ?r)` is NOT true, only the 'calibration needed' path is possible.
               - If no possible rover/camera path found (due to missing capabilities, waypoints, or unreachable locations), assign infinite cost for this goal.
            iii. Add the minimum calculated cost for this goal to the total cost.
    8. Return the total calculated cost. If any goal had infinite cost, the total cost will be infinity.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static
        # Include initial state facts that represent static or semi-static info
        # like initial sample locations. These don't change except by being consumed.
        all_relevant_facts = task.initial_state | static_facts

        # --- Extract Static Information ---

        # Collect all unique object names that are waypoints from relevant facts
        self.waypoints = set()
        for fact in all_relevant_facts:
            parts = get_parts(fact)
            if len(parts) > 1:
                predicate = parts[0]
                if predicate in ["at", "at_lander", "at_soil_sample", "at_rock_sample"]:
                    if len(parts) > 2: self.waypoints.add(parts[2]) # (at rover wp), (at_lander lander wp)
                    else: self.waypoints.add(parts[1]) # (at_soil_sample wp), (at_rock_sample wp)
                elif predicate in ["visible", "can_traverse"]:
                     if len(parts) > 2: self.waypoints.add(parts[1])
                     if len(parts) > 3: self.waypoints.add(parts[2])
                     if len(parts) > 4: self.waypoints.add(parts[3])
                elif predicate == "visible_from":
                     if len(parts) > 2: self.waypoints.add(parts[2])

        self.rovers = {get_parts(fact)[1] for fact in all_relevant_facts if match(fact, "at", "*", "*") and get_parts(fact)[1].startswith("rover")}
        self.landers = {get_parts(fact)[1] for fact in all_relevant_facts if match(fact, "at_lander", "*", "*")}
        self.cameras = {get_parts(fact)[1] for fact in all_relevant_facts if match(fact, "on_board", "*", "*")}
        self.objectives = {get_parts(fact)[1] for fact in all_relevant_facts if match(fact, "visible_from", "*", "*")} | \
                          {get_parts(fact)[2] for fact in all_relevant_facts if match(fact, "calibration_target", "*", "*")}


        self.rover_capabilities = {}
        self.rover_stores = {} # {rover: store}
        self.camera_info = {} # {camera: {rover: r, modes: {m}, cal_target: t}}
        self.imaging_wps = {} # {objective: {wp}}
        self.calibration_wps = {} # {(camera, target): {wp}}
        self.lander_locations = {} # {lander: wp}
        self.comm_waypoint_groups = {} # {lander_wp: {comm_wp}}
        self.rover_nav_graphs = {}
        self.rover_distances = {}

        # Populate rover capabilities and stores
        for rover in self.rovers:
            self.rover_capabilities[rover] = {
                'soil': f"(equipped_for_soil_analysis {rover})" in static_facts,
                'rock': f"(equipped_for_rock_analysis {rover})" in static_facts,
                'imaging': f"(equipped_for_imaging {rover})" in static_facts,
            }
            for fact in static_facts:
                 if match(fact, "store_of", "*", rover):
                     self.rover_stores[rover] = get_parts(fact)[1]

        # Populate camera info (on_board, supports, calibration_target)
        for camera in self.cameras:
             self.camera_info[camera] = {}
             for fact in static_facts:
                 if match(fact, "on_board", camera, "*"):
                     self.camera_info[camera]['rover'] = get_parts(fact)[2]
                 elif match(fact, "supports", camera, "*"):
                     self.camera_info[camera].setdefault('modes', set()).add(get_parts(fact)[2])
                 elif match(fact, "calibration_target", camera, "*"):
                     self.camera_info[camera]['cal_target'] = get_parts(fact)[2]

        # Populate lander locations
        for fact in static_facts:
            if match(fact, "at_lander", "*", "*"):
                lander, wp = get_parts(fact)[1:3]
                self.lander_locations[lander] = wp

        # Populate communication waypoints
        for fact in static_facts:
            if match(fact, "visible", "*", "*"):
                wp1, wp2 = get_parts(fact)[1:3]
                # Check if wp2 is a lander location
                for lander_wp in self.lander_locations.values():
                    if wp2 == lander_wp:
                        self.comm_waypoint_groups.setdefault(lander_wp, set()).add(wp1)
                # Check if wp1 is a lander location (visibility is symmetric)
                for lander_wp in self.lander_locations.values():
                    if wp1 == lander_wp:
                        self.comm_waypoint_groups.setdefault(lander_wp, set()).add(wp2)


        # Populate imaging and calibration waypoints
        for fact in static_facts:
            if match(fact, "visible_from", "*", "*"):
                obj_or_target, wp = get_parts(fact)[1:3]
                # Add as an imaging waypoint for the objective itself
                self.imaging_wps.setdefault(obj_or_target, set()).add(wp)

                # Check if this objective is a calibration target for any camera
                for camera, info in self.camera_info.items():
                    if info.get('cal_target') == obj_or_target:
                        # This waypoint is visible from a calibration target objective
                        self.calibration_wps.setdefault((camera, obj_or_target), set()).add(wp)


        # Build navigation graphs and calculate distances for each rover
        all_waypoints_list = list(self.waypoints) # Use a consistent list of all waypoints
        for rover in self.rovers:
            graph, waypoints_in_graph = build_graph_and_waypoints(static_facts, rover)
            self.rover_nav_graphs[rover] = graph
            # Calculate distances only between waypoints present in the graph
            self.rover_distances[rover] = bfs_distances(graph, waypoints_in_graph)
            # Add unreachable waypoints with infinite distance
            for wp1 in all_waypoints_list:
                 self.rover_distances[rover].setdefault(wp1, {})
                 for wp2 in all_waypoints_list:
                      self.rover_distances[rover][wp1].setdefault(wp2, float('inf'))


        # Identify initial sample locations (needed to check if sampling is possible)
        self.initial_soil_samples = {get_parts(fact)[1] for fact in task.initial_state if match(fact, "at_soil_sample", "*")}
        self.initial_rock_samples = {get_parts(fact)[1] for fact in task.initial_state if match(fact, "at_rock_sample", "*")}


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Check if goal is reached
        if self.goals <= state:
            return 0

        # --- Extract State Information ---
        rover_locations = {} # {rover: wp}
        rover_soil_samples = {} # {rover: {wp}}
        rover_rock_samples = {} # {rover: {wp}}
        rover_images = {} # {rover: {(obj, mode)}}
        rover_calibrated_cameras = {} # {rover: {camera}}
        current_soil_sample_locs = set() # {wp}
        current_rock_sample_locs = set() # {wp}

        # Initialize state info structures for all rovers
        for rover in self.rovers:
             rover_locations[rover] = None # Initialize location as None
             rover_soil_samples[rover] = set()
             rover_rock_samples[rover] = set()
             rover_images[rover] = set()
             rover_calibrated_cameras[rover] = set()


        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == "at" and len(parts) > 2 and parts[1].startswith("rover"):
                rover, wp = parts[1:3]
                if rover in self.rovers: # Ensure rover is known
                    rover_locations[rover] = wp
            elif predicate == "have_soil_analysis" and len(parts) > 2:
                rover, wp = parts[1:3]
                if rover in rover_soil_samples: # Ensure rover is known
                    rover_soil_samples[rover].add(wp)
            elif predicate == "have_rock_analysis" and len(parts) > 2:
                rover, wp = parts[1:3]
                if rover in rover_rock_samples: # Ensure rover is known
                    rover_rock_samples[rover].add(wp)
            elif predicate == "have_image" and len(parts) > 3:
                rover, obj, mode = parts[1:4]
                if rover in rover_images: # Ensure rover is known
                    rover_images[rover].add((obj, mode))
            elif predicate == "calibrated" and len(parts) > 2:
                camera, rover = parts[1:3]
                if rover in rover_calibrated_cameras: # Ensure rover is known
                    rover_calibrated_cameras[rover].add(camera)
            elif predicate == "at_soil_sample" and len(parts) > 1:
                 current_soil_sample_locs.add(parts[1])
            elif predicate == "at_rock_sample" and len(parts) > 1:
                 current_rock_sample_locs.add(parts[1])


        total_cost = 0

        # Find all possible communication waypoints across all landers
        all_comm_wps = set()
        for lander_wp in self.comm_waypoint_groups.keys():
             all_comm_wps.update(self.comm_waypoint_groups[lander_wp])

        # Iterate through goals
        for goal in self.goals:
            if goal in state:
                continue # Goal already achieved

            parts = get_parts(goal)
            if not parts: continue # Skip malformed goals

            predicate = parts[0]

            if predicate == "communicated_soil_data" and len(parts) > 1:
                wp = parts[1]
                goal_achieved_via_have = False
                min_comm_nav_cost = float('inf')

                # Check if any equipped rover already has the analysis
                for rover in self.rovers:
                    if self.rover_capabilities.get(rover, {}).get('soil', False) and wp in rover_soil_samples.get(rover, set()):
                        # Rover has the analysis, just needs to navigate and communicate
                        current_rover_loc = rover_locations.get(rover)
                        if current_rover_loc: # Ensure rover location is known
                            nav_cost = self._min_nav_cost(rover, current_rover_loc, all_comm_wps)
                            min_comm_nav_cost = min(min_comm_nav_cost, nav_cost)
                            goal_achieved_via_have = True

                if goal_achieved_via_have:
                    if min_comm_nav_cost != float('inf'):
                         total_cost += 1 + min_comm_nav_cost # 1 for communicate
                    else:
                         # Cannot reach any communication point from current location
                         total_cost += float('inf') # Unreachable
                else:
                    # Need to sample and communicate
                    if wp not in current_soil_sample_locs:
                        # Sample is gone and no rover has the analysis - likely unachievable
                        total_cost += float('inf')
                        continue # Skip to next goal

                    min_total_goal_cost = float('inf')

                    for rover in self.rovers:
                        if self.rover_capabilities.get(rover, {}).get('soil', False):
                            current_rover_loc = rover_locations.get(rover)
                            if not current_rover_loc: continue # Rover location unknown

                            # Cost to sample
                            nav_cost_to_sample = self._min_nav_cost(rover, current_rover_loc, {wp})
                            if nav_cost_to_sample == float('inf'): continue # Cannot reach sample location

                            sample_cost = 1 # sample_soil action
                            # Ignore store constraint for simplicity

                            # Cost to communicate after sampling (assume rover is at wp after sampling)
                            nav_cost_sample_to_comm = self._min_nav_cost(rover, wp, all_comm_wps)
                            if nav_cost_sample_to_comm == float('inf'): continue # Cannot reach communication point

                            comm_cost = 1 # communicate_soil_data action

                            total_goal_cost = nav_cost_to_sample + sample_cost + nav_cost_sample_to_comm + comm_cost
                            min_total_goal_cost = min(min_total_goal_cost, total_goal_cost)

                    if min_total_goal_cost != float('inf'):
                        total_cost += min_total_goal_cost
                    else:
                        total_cost += float('inf') # Unreachable


            elif predicate == "communicated_rock_data" and len(parts) > 1:
                wp = parts[1]
                goal_achieved_via_have = False
                min_comm_nav_cost = float('inf')

                # Check if any equipped rover already has the analysis
                for rover in self.rovers:
                    if self.rover_capabilities.get(rover, {}).get('rock', False) and wp in rover_rock_samples.get(rover, set()):
                        # Rover has the analysis, just needs to navigate and communicate
                        current_rover_loc = rover_locations.get(rover)
                        if current_rover_loc:
                            nav_cost = self._min_nav_cost(rover, current_rover_loc, all_comm_wps)
                            min_comm_nav_cost = min(min_comm_nav_cost, nav_cost)
                            goal_achieved_via_have = True

                if goal_achieved_via_have:
                    if min_comm_nav_cost != float('inf'):
                         total_cost += 1 + min_comm_nav_cost # 1 for communicate
                    else:
                         total_cost += float('inf') # Unreachable
                else:
                    # Need to sample and communicate
                    if wp not in current_rock_sample_locs:
                        # Sample is gone and no rover has the analysis - likely unachievable
                        total_cost += float('inf')
                        continue # Skip to next goal

                    min_total_goal_cost = float('inf')

                    for rover in self.rovers:
                        if self.rover_capabilities.get(rover, {}).get('rock', False):
                            current_rover_loc = rover_locations.get(rover)
                            if not current_rover_loc: continue # Rover location unknown

                            # Cost to sample
                            nav_cost_to_sample = self._min_nav_cost(rover, current_rover_loc, {wp})
                            if nav_cost_to_sample == float('inf'): continue # Cannot reach sample location

                            sample_cost = 1 # sample_rock action
                            # Ignore store constraint

                            # Cost to communicate after sampling (assume rover is at wp after sampling)
                            nav_cost_sample_to_comm = self._min_nav_cost(rover, wp, all_comm_wps)
                            if nav_cost_sample_to_comm == float('inf'): continue # Cannot reach communication point

                            comm_cost = 1 # communicate_rock_data action

                            total_goal_cost = nav_cost_to_sample + sample_cost + nav_cost_sample_to_comm + comm_cost
                            min_total_goal_cost = min(min_total_goal_cost, total_goal_cost)

                    if min_total_goal_cost != float('inf'):
                        total_cost += min_total_goal_cost
                    else:
                        total_cost += float('inf') # Unreachable


            elif predicate == "communicated_image_data" and len(parts) > 2:
                obj, mode = parts[1:3]
                goal_achieved_via_have = False
                min_comm_nav_cost = float('inf')

                # Check if any equipped rover already has the image
                for rover in self.rovers:
                    if self.rover_capabilities.get(rover, {}).get('imaging', False) and (obj, mode) in rover_images.get(rover, set()):
                        # Rover has the image, just needs to navigate and communicate
                        current_rover_loc = rover_locations.get(rover)
                        if current_rover_loc:
                            nav_cost = self._min_nav_cost(rover, current_rover_loc, all_comm_wps)
                            min_comm_nav_cost = min(min_comm_nav_cost, nav_cost)
                            goal_achieved_via_have = True

                if goal_achieved_via_have:
                    if min_comm_nav_cost != float('inf'):
                         total_cost += 1 + min_comm_nav_cost # 1 for communicate
                    else:
                         total_cost += float('inf') # Unreachable
                else:
                    # Need to take image and communicate
                    min_total_goal_cost = float('inf')

                    # Find suitable imaging waypoints for this objective
                    imaging_wps = self.imaging_wps.get(obj, set())
                    if not imaging_wps:
                         total_cost += float('inf') # Cannot image this objective
                         continue # Skip to next goal

                    for rover in self.rovers:
                        if self.rover_capabilities.get(rover, {}).get('imaging', False):
                            current_rover_loc = rover_locations.get(rover)
                            if not current_rover_loc: continue # Rover location unknown

                            # Find cameras on this rover that support the mode
                            suitable_cameras = {
                                cam for cam, info in self.camera_info.items()
                                if info.get('rover') == rover and mode in info.get('modes', set())
                            }

                            for camera in suitable_cameras:
                                cal_target = self.camera_info.get(camera, {}).get('cal_target')

                                # Find calibration waypoints for this camera/target
                                calib_wps = set()
                                if cal_target:
                                     calib_wps = self.calibration_wps.get((camera, cal_target), set())

                                # --- Calculate cost path: Calibrate -> Image -> Communicate ---
                                path_cost_calib_image_comm = float('inf')
                                # This path is only possible if the camera has a calibration target and there are calibration waypoints
                                if cal_target and calib_wps:
                                    nav_cost_to_calib = self._min_nav_cost(rover, current_rover_loc, calib_wps)
                                    if nav_cost_to_calib != float('inf'):
                                        calib_cost = 1 # calibrate action
                                        nav_cost_calib_to_image = self._min_nav_cost_between_sets(rover, calib_wps, imaging_wps)
                                        if nav_cost_calib_to_image != float('inf'):
                                            image_cost = 1 # take_image action
                                            nav_cost_image_to_comm = self._min_nav_cost_between_sets(rover, imaging_wps, all_comm_wps)
                                            if nav_cost_image_to_comm != float('inf'):
                                                comm_cost = 1 # communicate_image_data action
                                                path_cost_calib_image_comm = nav_cost_to_calib + calib_cost + nav_cost_calib_to_image + image_cost + nav_cost_image_to_comm + comm_cost


                                # --- Calculate cost path: Image -> Communicate (if already calibrated) ---
                                path_cost_image_comm = float('inf')
                                if camera in rover_calibrated_cameras.get(rover, set()):
                                     # Already calibrated, skip calibrate step and nav to calib
                                     nav_cost_to_image = self._min_nav_cost(rover, current_rover_loc, imaging_wps)
                                     if nav_cost_to_image != float('inf'):
                                         image_cost = 1 # take_image action
                                         nav_cost_image_to_comm = self._min_nav_cost_between_sets(rover, imaging_wps, all_comm_wps)
                                         if nav_cost_image_to_comm != float('inf'):
                                             comm_cost = 1 # communicate_image_data action
                                             path_cost_image_comm = nav_cost_to_image + image_cost + nav_cost_image_to_comm + comm_cost


                                # Take the minimum cost path for this rover/camera combination
                                # If camera cannot be calibrated (no target or no calib wps), path_cost_calib_image_comm will be inf.
                                # If camera is not calibrated, path_cost_image_comm will be inf.
                                min_path_cost_for_rc = min(path_cost_calib_image_comm, path_cost_image_comm)
                                min_total_goal_cost = min(min_total_goal_cost, min_path_cost_for_rc)


                    if min_total_goal_cost != float('inf'):
                        total_cost += min_total_goal_cost
                    else:
                        total_cost += float('inf') # Unreachable


        return total_cost

    def _min_nav_cost(self, rover, start_wp, target_wps):
        """Helper to find min navigation cost from start_wp to any waypoint in target_wps."""
        if not target_wps: return float('inf')
        min_cost = float('inf')
        # Ensure rover and start_wp are valid in the pre-calculated distances
        if rover not in self.rover_distances or start_wp not in self.rover_distances[rover]:
             # This case should ideally not happen if start_wp is a known waypoint from init
             # and rover exists. But defensive check is good.
             return float('inf')

        # Get the distance map for the starting waypoint for this rover
        distances_from_start = self.rover_distances[rover].get(start_wp, {})

        for target_wp in target_wps:
             # Ensure target_wp is a valid key in the distance map from start_wp
             if target_wp in distances_from_start:
                 min_cost = min(min_cost, distances_from_start[target_wp])
        return min_cost

    def _min_nav_cost_between_sets(self, rover, start_wps, target_wps):
        """Helper to find min navigation cost from any waypoint in start_wps to any waypoint in target_wps."""
        if not start_wps or not target_wps: return float('inf')
        min_cost = float('inf')
        for start_wp in start_wps:
            min_cost = min(min_cost, self._min_nav_cost(rover, start_wp, target_wps))
        return min_cost
