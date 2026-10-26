import collections
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper functions to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
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

class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve all
    goal conditions, which involve communicating soil, rock, and image data.
    It sums the estimated costs for each unachieved goal fact.

    # Heuristic Initialization
    - Parses static facts to build the navigation graph, identify landers,
      communication points, imaging points, calibration targets, rover
      capabilities, and camera details.
    - Precomputes all-pairs shortest paths between waypoints using BFS.

    # Step-by-Step Thinking for Computing Heuristic
    For each unachieved goal fact (communicated_soil_data, communicated_rock_data,
    communicated_image_data), estimate the cost to achieve it independently
    and sum these costs.

    1.  **Identify Unachieved Goals:** Determine which `communicated_X_data` facts
        from the task goals are not present in the current state.

    2.  **Estimate Cost for Soil/Rock Data Goal `(communicated_X_data ?w)`:**
        - If already communicated: Cost is 0.
        - If not communicated:
            - Cost += 1 (for the `communicate_X_data` action).
            - Check if `(have_X_analysis ?r ?w)` exists for any rover `?r`.
            - If data is *not* held by any rover:
                - Cost += 1 (for the `sample_X` action).
                - Find an equipped rover `r_s` for this task.
                - Cost += minimum navigation cost from `r_s`'s current location to waypoint `?w`.
                - If any store on `r_s` is full, Cost += 1 (for a `drop` action before sampling - a simplification).
                - Cost += minimum navigation cost from waypoint `?w` (where sampling occurs) to any communication waypoint.
            - If data *is* held by some rover `r_h`:
                - Cost += minimum navigation cost from `r_h`'s current location to any communication waypoint.

    3.  **Estimate Cost for Image Data Goal `(communicated_image_data ?o ?m)`:**
        - If already communicated: Cost is 0.
        - If not communicated:
            - Cost += 1 (for the `communicate_image_data` action).
            - Check if `(have_image ?r ?o ?m)` exists for any rover `?r`.
            - If image is *not* held by any rover:
                - Cost += 1 (for the `take_image` action).
                - Find a suitable rover `r_i` (equipped for imaging, has camera supporting mode `?m`).
                - Find camera `c` on `r_i` supporting `?m`.
                - Check if `(calibrated c r_i)` is true.
                - If *not* calibrated:
                    - Cost += 1 (for the `calibrate` action).
                    - Find calibration target `t` for camera `c`.
                    - Find calibration waypoints `w_calibs` visible from `t`.
                    - Cost += minimum navigation cost from `r_i`'s current location to any `w_calib`.
                    - Cost += minimum navigation cost from a `w_calib` to any imaging waypoint `w_image` visible from objective `?o`.
                - If *is* calibrated:
                    - Cost += minimum navigation cost from `r_i`'s current location to any imaging waypoint `w_image` visible from objective `?o`.
                - Cost += minimum navigation cost from an imaging waypoint `w_image` (where image is taken) to any communication waypoint.
            - If image *is* held by some rover `r_h`:
                - Cost += minimum navigation cost from `r_h`'s current location to any communication waypoint.

    4.  **Navigation Costs:** Use the precomputed shortest path distances between waypoints. When multiple options exist (multiple communication waypoints, imaging waypoints, calibration waypoints, or rovers), choose the option that minimizes the navigation cost for that specific step.

    5.  **Summation:** The total heuristic value is the sum of the estimated costs for all unachieved goal facts.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and static facts,
        and precomputing navigation distances.
        """
        self.goals = task.goals

        # --- Parse Static Facts ---
        self.rover_equipment = collections.defaultdict(set)
        self.rover_stores = collections.defaultdict(set)
        self.camera_details = {} # camera -> {rover, modes: set(), calib_target}
        self.objective_imaging_wps = collections.defaultdict(set)
        self.calib_target_calib_wps = collections.defaultdict(set)
        self.lander_locations = set()
        self.comm_waypoints = set() # Waypoints visible from any lander
        self.waypoints = set()
        self.nav_graph = collections.defaultdict(set) # waypoint -> set(neighbor_waypoint)

        # Collect all waypoints first
        for fact in task.initial_state | task.static:
             if match(fact, "at", "*", "*"):
                 parts = get_parts(fact)
                 if parts[1].startswith("rover"): # Assuming rover names start with 'rover'
                     self.waypoints.add(parts[2])
             elif match(fact, "at_lander", "*", "*"):
                 self.waypoints.add(get_parts(fact)[2])
             elif match(fact, "visible", "*", "*"):
                 self.waypoints.add(get_parts(fact)[1])
                 self.waypoints.add(get_parts(fact)[2])
             elif match(fact, "can_traverse", "*", "*", "*"):
                 self.waypoints.add(get_parts(fact)[2])
                 self.waypoints.add(get_parts(fact)[3])
             elif match(fact, "at_soil_sample", "*"):
                 self.waypoints.add(get_parts(fact)[1])
             elif match(fact, "at_rock_sample", "*"):
                 self.waypoints.add(get_parts(fact)[1])
             elif match(fact, "visible_from", "*", "*"):
                 self.waypoints.add(get_parts(fact)[2])


        for fact in task.static:
            if match(fact, "equipped_for_soil_analysis", "*"):
                self.rover_equipment[get_parts(fact)[1]].add("soil")
            elif match(fact, "equipped_for_rock_analysis", "*"):
                self.rover_equipment[get_parts(fact)[1]].add("rock")
            elif match(fact, "equipped_for_imaging", "*"):
                self.rover_equipment[get_parts(fact)[1]].add("imaging")
            elif match(fact, "store_of", "*", "*"):
                self.rover_stores[get_parts(fact)[2]].add(get_parts(fact)[1])
            elif match(fact, "on_board", "*", "*"):
                camera, rover = get_parts(fact)[1:3]
                if camera not in self.camera_details:
                    self.camera_details[camera] = {'rover': rover, 'modes': set(), 'calib_target': None}
                self.camera_details[camera]['rover'] = rover
            elif match(fact, "supports", "*", "*"):
                camera, mode = get_parts(fact)[1:3]
                if camera not in self.camera_details:
                     # Handle cases where supports is listed before on_board
                     self.camera_details[camera] = {'rover': None, 'modes': set(), 'calib_target': None}
                self.camera_details[camera]['modes'].add(mode)
            elif match(fact, "calibration_target", "*", "*"):
                camera, target = get_parts(fact)[1:3]
                if camera not in self.camera_details:
                     # Handle cases where calibration_target is listed before on_board
                     self.camera_details[camera] = {'rover': None, 'modes': set(), 'calib_target': None}
                self.camera_details[camera]['calib_target'] = target
            elif match(fact, "visible_from", "*", "*"):
                objective, waypoint = get_parts(fact)[1:3]
                self.objective_imaging_wps[objective].add(waypoint)
                # If this waypoint is visible from a calibration target, it's a calibration waypoint
                for cam, details in self.camera_details.items():
                    if details['calib_target'] == objective:
                         self.calib_target_calib_wps[objective].add(waypoint)

            elif match(fact, "at_lander", "*", "*"):
                self.lander_locations.add(get_parts(fact)[2])
            elif match(fact, "visible", "*", "*"):
                 wp1, wp2 = get_parts(fact)[1:3]
                 # Build graph based on visible, assuming traversable if visible
                 # A more accurate graph would use can_traverse per rover, but this is a heuristic simplification
                 self.nav_graph[wp1].add(wp2)
                 self.nav_graph[wp2].add(wp1) # Assuming visible is symmetric

        # Identify communication waypoints (visible from any lander location)
        for lander_loc in self.lander_locations:
             if lander_loc in self.nav_graph: # Lander location must be a known waypoint
                 self.comm_waypoints.update(self.nav_graph[lander_loc])
        # Also, the lander location itself might be a communication point if a rover can be there
        self.comm_waypoints.update(self.lander_locations)


        # --- Precompute Shortest Paths (APSP) ---
        self.dist = self._compute_shortest_paths()

    def _compute_shortest_paths(self):
        """
        Computes all-pairs shortest paths using BFS.
        Returns a dictionary dist[start_wp][end_wp] = distance.
        Unreachable pairs have distance infinity.
        """
        dist = {}
        for start_node in self.waypoints:
            dist[start_node] = {}
            q = collections.deque([(start_node, 0)])
            visited = {start_node}
            dist[start_node][start_node] = 0

            while q:
                current_node, current_dist = q.popleft()

                if current_node in self.nav_graph:
                    for neighbor in self.nav_graph[current_node]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            dist[start_node][neighbor] = current_dist + 1
                            q.append((neighbor, current_dist + 1))

            # Mark unreachable waypoints with infinity
            for end_node in self.waypoints:
                 if end_node not in dist[start_node]:
                     dist[start_node][end_node] = float('inf')

        return dist

    def get_distance(self, wp1, wp2):
        """Helper to get distance, handling potential missing waypoints gracefully."""
        return self.dist.get(wp1, {}).get(wp2, float('inf'))

    def get_min_dist_to_set(self, start_wp, target_wps):
        """Helper to get minimum distance from start_wp to any waypoint in target_wps."""
        if not target_wps:
            return float('inf')
        return min(self.get_distance(start_wp, target_wp) for target_wp in target_wps)

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        h = 0

        # --- Extract Dynamic State ---
        rover_locations = {} # rover -> waypoint
        rover_soil_samples = collections.defaultdict(set) # rover -> set(waypoint)
        rover_rock_samples = collections.defaultdict(set) # rover -> set(waypoint)
        rover_images = collections.defaultdict(set) # rover -> set((objective, mode))
        store_full = set() # set of full store names
        rover_calibrated_cameras = collections.defaultdict(set) # rover -> set(camera)

        for fact in state:
            if match(fact, "at", "rover*", "*"):
                rover, wp = get_parts(fact)[1:3]
                rover_locations[rover] = wp
            elif match(fact, "have_soil_analysis", "rover*", "*"):
                rover, wp = get_parts(fact)[1:3]
                rover_soil_samples[rover].add(wp)
            elif match(fact, "have_rock_analysis", "rover*", "*"):
                rover, wp = get_parts(fact)[1:3]
                rover_rock_samples[rover].add(wp)
            elif match(fact, "have_image", "rover*", "*", "*"):
                rover, obj, mode = get_parts(fact)[1:4]
                rover_images[rover].add((obj, mode))
            elif match(fact, "full", "rover*store"): # Assuming store names follow this pattern
                store_full.add(get_parts(fact)[1])
            elif match(fact, "calibrated", "camera*", "rover*"):
                camera, rover = get_parts(fact)[1:3]
                rover_calibrated_cameras[rover].add(camera)

        # Determine which rovers have at least one full store
        rover_with_full_store = {rover for rover, stores in self.rover_stores.items()
                                 if any(store in store_full for store in stores)}

        # --- Estimate Cost for Each Goal ---
        for goal in self.goals:
            if goal in state:
                continue # Goal already achieved

            parts = get_parts(goal)
            predicate = parts[0]

            if predicate == "communicated_soil_data":
                waypoint = parts[1]
                h += 1 # Cost for communicate action

                # Check if sample is held by any rover
                has_sample = any(waypoint in samples for samples in rover_soil_samples.values())

                if not has_sample:
                    h += 1 # Cost for sample action
                    min_nav_cost_to_sample = float('inf')
                    min_nav_cost_sample_to_comm = float('inf')
                    needs_drop_penalty = False

                    # Find best equipped rover for sampling
                    equipped_rovers = [r for r, equip in self.rover_equipment.items() if "soil" in equip]
                    if not equipped_rovers: # Should not happen in solvable problems, but handle defensively
                         min_nav_cost_to_sample = float('inf')
                    else:
                        for rover in equipped_rovers:
                            if rover in rover_locations:
                                nav_to_sample = self.get_distance(rover_locations[rover], waypoint)
                                min_nav_cost_to_sample = min(min_nav_cost_to_sample, nav_to_sample)
                                if rover in rover_with_full_store:
                                     needs_drop_penalty = True # Any equipped rover needing to sample has a full store

                        min_nav_cost_sample_to_comm = self.get_min_dist_to_set(waypoint, self.comm_waypoints)

                    h += min_nav_cost_to_sample
                    h += min_nav_cost_sample_to_comm
                    if needs_drop_penalty:
                         h += 1 # Penalty for needing to drop a sample

                else: # Has sample, just need to communicate
                    min_nav_cost_to_comm = float('inf')
                    # Find best rover that has the sample
                    rovers_with_sample = [r for r, samples in rover_soil_samples.items() if waypoint in samples]
                    if not rovers_with_sample: # Should not happen
                         min_nav_cost_to_comm = float('inf')
                    else:
                        for rover in rovers_with_sample:
                            if rover in rover_locations:
                                nav_to_comm = self.get_min_dist_to_set(rover_locations[rover], self.comm_waypoints)
                                min_nav_cost_to_comm = min(min_nav_cost_to_comm, nav_to_comm)
                    h += min_nav_cost_to_comm

            elif predicate == "communicated_rock_data":
                waypoint = parts[1]
                h += 1 # Cost for communicate action

                # Check if sample is held by any rover
                has_sample = any(waypoint in samples for samples in rover_rock_samples.values())

                if not has_sample:
                    h += 1 # Cost for sample action
                    min_nav_cost_to_sample = float('inf')
                    min_nav_cost_sample_to_comm = float('inf')
                    needs_drop_penalty = False

                    # Find best equipped rover for sampling
                    equipped_rovers = [r for r, equip in self.rover_equipment.items() if "rock" in equip]
                    if not equipped_rovers:
                         min_nav_cost_to_sample = float('inf')
                    else:
                        for rover in equipped_rovers:
                            if rover in rover_locations:
                                nav_to_sample = self.get_distance(rover_locations[rover], waypoint)
                                min_nav_cost_to_sample = min(min_nav_cost_to_sample, nav_to_sample)
                                if rover in rover_with_full_store:
                                     needs_drop_penalty = True # Any equipped rover needing to sample has a full store

                        min_nav_cost_sample_to_comm = self.get_min_dist_to_set(waypoint, self.comm_waypoints)

                    h += min_nav_cost_to_sample
                    h += min_nav_cost_sample_to_comm
                    if needs_drop_penalty:
                         h += 1 # Penalty for needing to drop a sample

                else: # Has sample, just need to communicate
                    min_nav_cost_to_comm = float('inf')
                    # Find best rover that has the sample
                    rovers_with_sample = [r for r, samples in rover_rock_samples.items() if waypoint in samples]
                    if not rovers_with_sample:
                         min_nav_cost_to_comm = float('inf')
                    else:
                        for rover in rovers_with_sample:
                            if rover in rover_locations:
                                nav_to_comm = self.get_min_dist_to_set(rover_locations[rover], self.comm_waypoints)
                                min_nav_cost_to_comm = min(min_nav_cost_to_comm, nav_to_comm)
                    h += min_nav_cost_to_comm

            elif predicate == "communicated_image_data":
                objective, mode = parts[1:3]
                h += 1 # Cost for communicate action

                # Check if image is held by any rover
                has_image = any((objective, mode) in images for images in rover_images.values())

                if not has_image:
                    h += 1 # Cost for take_image action

                    min_total_nav_cost = float('inf')

                    # Find best suitable rover/camera combination
                    suitable_rovers = [r for r, equip in self.rover_equipment.items() if "imaging" in equip]

                    if not suitable_rovers:
                         min_total_nav_cost = float('inf')
                    else:
                        for rover in suitable_rovers:
                            if rover not in rover_locations: continue # Rover must be placed

                            for camera, details in self.camera_details.items():
                                if details['rover'] == rover and mode in details['modes']:
                                    calib_target = details['calib_target']
                                    calib_wps = self.calib_target_calib_wps.get(calib_target, set())
                                    image_wps = self.objective_imaging_wps.get(objective, set())
                                    comm_wps = self.comm_waypoints

                                    if not image_wps or not comm_wps: continue # Cannot achieve this goal

                                    is_calibrated = camera in rover_calibrated_cameras.get(rover, set())

                                    if not is_calibrated:
                                        if not calib_wps: continue # Cannot calibrate

                                        h_calib = 1 # Cost for calibrate action
                                        # Nav: current_loc -> calib_wp -> image_wp -> comm_wp
                                        nav_cost_to_calib = self.get_min_dist_to_set(rover_locations[rover], calib_wps)
                                        nav_cost_calib_to_image = min(self.get_min_dist_to_set(cw, image_wps) for cw in calib_wps) if calib_wps else float('inf')
                                        nav_cost_image_to_comm = min(self.get_min_dist_to_set(iw, comm_wps) for iw in image_wps) if image_wps else float('inf')

                                        total_nav = nav_cost_to_calib + nav_cost_calib_to_image + nav_cost_image_to_comm
                                        min_total_nav_cost = min(min_total_nav_cost, h_calib + total_nav)

                                    else: # Is calibrated
                                        # Nav: current_loc -> image_wp -> comm_wp
                                        nav_cost_to_image = self.get_min_dist_to_set(rover_locations[rover], image_wps)
                                        nav_cost_image_to_comm = min(self.get_min_dist_to_set(iw, comm_wps) for iw in image_wps) if image_wps else float('inf')

                                        total_nav = nav_cost_to_image + nav_cost_image_to_comm
                                        min_total_nav_cost = min(min_total_nav_cost, total_nav)

                    h += min_total_nav_cost


                else: # Has image, just need to communicate
                    min_nav_cost_to_comm = float('inf')
                    # Find best rover that has the image
                    rovers_with_image = [r for r, images in rover_images.items() if (objective, mode) in images]
                    if not rovers_with_image: # Should not happen
                         min_nav_cost_to_comm = float('inf')
                    else:
                        for rover in rovers_with_image:
                            if rover in rover_locations:
                                nav_to_comm = self.get_min_dist_to_set(rover_locations[rover], self.comm_waypoints)
                                min_nav_cost_to_comm = min(min_nav_cost_to_comm, nav_to_comm)
                    h += min_nav_cost_to_comm

        return h

