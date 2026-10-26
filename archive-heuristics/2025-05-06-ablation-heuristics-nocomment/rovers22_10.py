import collections
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Helper function to split a PDDL fact string into its parts."""
    # Removes leading/trailing parentheses and splits by space
    return fact[1:-1].split()

def match(fact, *args):
    """Helper function to check if a fact matches a pattern using fnmatch."""
    parts = get_parts(fact)
    # Check if the number of parts matches the number of args, then check each part
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class roversHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Rovers domain.

    Estimates the cost to reach a goal state by summing the estimated costs
    for each unachieved communication goal. The heuristic is non-admissible
    and aims to guide a greedy best-first search efficiently.

    The heuristic breaks down each unachieved communication goal into
    necessary steps (collecting data, communicating data) and estimates
    the minimum cost for each step based on rover capabilities, locations,
    and precomputed navigation distances.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by precomputing static information from the task.

        Args:
            task: The planning task object containing initial state, goals, and static facts.
        """
        self.goals = task.goals
        self.static_facts = task.static
        self.initial_state = task.initial_state # Needed to check initial sample locations

        # --- Precompute Static Information ---

        # Lander location
        self.lander_location = None
        for fact in self.static_facts:
            if match(fact, "at_lander", "*", "*"):
                self.lander_location = get_parts(fact)[2]
                break # Assuming only one lander

        # Visibility graph between waypoints
        self.visible_waypoints = collections.defaultdict(set)
        for fact in self.static_facts:
            if match(fact, "visible", "*", "*"):
                wp1, wp2 = get_parts(fact)[1:]
                self.visible_waypoints[wp1].add(wp2)
                self.visible_waypoints[wp2].add(wp1) # Visibility is symmetric

        # Waypoints visible from the lander (communication points)
        self.lander_comm_wps = self.visible_waypoints.get(self.lander_location, set())

        # Traversability graph for each rover
        self.can_traverse = collections.defaultdict(lambda: collections.defaultdict(set))
        rovers = set()
        for fact in self.static_facts:
            if match(fact, "can_traverse", "*", "*", "*"):
                rover, wp1, wp2 = get_parts(fact)[1:]
                self.can_traverse[rover][wp1].add(wp2)
                rovers.add(rover)

        # Precompute shortest paths between waypoints for each rover using BFS
        self.rover_waypoint_distances = {}
        for rover in rovers:
            self.rover_waypoint_distances[rover] = self._compute_all_pairs_shortest_paths(rover)

        # Rover capabilities
        self.rover_capabilities = collections.defaultdict(set)
        for fact in self.static_facts:
            if match(fact, "equipped_for_soil_analysis", "*"):
                self.rover_capabilities[get_parts(fact)[1]].add("soil")
            elif match(fact, "equipped_for_rock_analysis", "*"):
                self.rover_capabilities[get_parts(fact)[1]].add("rock")
            elif match(fact, "equipped_for_imaging", "*"):
                self.rover_capabilities[get_parts(fact)[1]].add("imaging")

        # Store mapping (rover -> store)
        self.rover_stores = {}
        for fact in self.static_facts:
             if match(fact, "store_of", "*", "*"):
                 store, rover = get_parts(fact)[1:]
                 self.rover_stores[rover] = store # Assuming one store per rover

        # Camera information (on_board rover, supports modes, calibration_target)
        self.camera_info = collections.defaultdict(lambda: {"on_board": None, "supports": set(), "calibration_target": None})
        for fact in self.static_facts:
            if match(fact, "on_board", "*", "*"):
                camera, rover = get_parts(fact)[1:]
                self.camera_info[camera]["on_board"] = rover
            elif match(fact, "supports", "*", "*"):
                camera, mode = get_parts(fact)[1:]
                self.camera_info[camera]["supports"].add(mode)
            elif match(fact, "calibration_target", "*", "*"):
                camera, target = get_parts(fact)[1:]
                self.camera_info[camera]["calibration_target"] = target

        # Objective visibility (objective -> set of visible waypoints)
        self.objective_visibility = collections.defaultdict(set)
        for fact in self.static_facts:
            if match(fact, "visible_from", "*", "*"):
                objective, waypoint = get_parts(fact)[1:]
                self.objective_visibility[objective].add(waypoint)

        # Initial sample locations (used to check if a sample ever existed)
        self.initial_soil_samples = {get_parts(fact)[1] for fact in task.initial_state if match(fact, "at_soil_sample", "*")}
        self.initial_rock_samples = {get_parts(fact)[1] for fact in task.initial_state if match(fact, "at_rock_sample", "*")}


    def _compute_all_pairs_shortest_paths(self, rover):
        """
        Computes shortest path distances between all reachable waypoints
        for a given rover using BFS.

        Args:
            rover: The name of the rover.

        Returns:
            A dictionary mapping (start_wp, end_wp) to distance, or float('inf')
            if unreachable.
        """
        graph = self.can_traverse.get(rover, {})
        waypoints = set(graph.keys())
        for wps in graph.values():
            waypoints.update(wps)

        dist = {wp: {wp2: float('inf') for wp2 in waypoints} for wp in waypoints}
        for wp in waypoints:
            dist[wp][wp] = 0

        # Perform BFS starting from each waypoint
        for start_wp in waypoints:
             q = collections.deque([(start_wp, 0)])
             visited = {start_wp}
             while q:
                 curr_wp, d = q.popleft()
                 dist[start_wp][curr_wp] = d

                 if curr_wp in graph:
                     for next_wp in graph[curr_wp]:
                         if next_wp not in visited:
                             visited.add(next_wp)
                             q.append((next_wp, d + 1))
        return dist

    def get_distance(self, rover, start_wp, end_wp):
        """
        Returns the shortest distance (number of navigate actions) for a rover
        between two waypoints.

        Args:
            rover: The name of the rover.
            start_wp: The starting waypoint.
            end_wp: The destination waypoint.

        Returns:
            The shortest distance, or float('inf') if unreachable or rover/waypoints invalid.
        """
        if rover in self.rover_waypoint_distances and start_wp in self.rover_waypoint_distances[rover] and end_wp in self.rover_waypoint_distances[rover][start_wp]:
             return self.rover_waypoint_distances[rover][start_wp][end_wp]
        return float('inf') # Not reachable

    def __call__(self, node):
        """
        Computes the heuristic value for a given state.

        Args:
            node: The search node containing the state.

        Returns:
            An integer heuristic value estimating the remaining cost to the goal.
        """
        state = node.state
        h = 0
        IMPOSSIBLE_PENALTY = 1000 # Penalty for goals deemed impossible

        # --- Extract Dynamic State Information ---
        rover_locations = {}
        rover_have_soil = collections.defaultdict(set)
        rover_have_rock = collections.defaultdict(set)
        rover_have_image = collections.defaultdict(lambda: collections.defaultdict(set)) # rover -> objective -> modes
        rover_calibrated_cameras = collections.defaultdict(set) # rover -> set of calibrated cameras
        rover_stores_full = set() # set of rovers with full stores
        soil_samples_at = set() # set of waypoints with soil samples
        rock_samples_at = set() # set of waypoints with rock samples

        for fact in state:
            if match(fact, "at", "rover*", "*"):
                rover, wp = get_parts(fact)[1:]
                rover_locations[rover] = wp
            elif match(fact, "have_soil_analysis", "rover*", "*"):
                rover, wp = get_parts(fact)[1:]
                rover_have_soil[rover].add(wp)
            elif match(fact, "have_rock_analysis", "rover*", "*"):
                rover, wp = get_parts(fact)[1:]
                rover_have_rock[rover].add(wp)
            elif match(fact, "have_image", "rover*", "*", "*"):
                rover, obj, mode = get_parts(fact)[1:]
                rover_have_image[rover][obj].add(mode)
            elif match(fact, "calibrated", "camera*", "rover*"):
                camera, rover = get_parts(fact)[1:]
                rover_calibrated_cameras[rover].add(camera)
            elif match(fact, "full", "rover*store"):
                 store = get_parts(fact)[1]
                 # Find which rover this store belongs to
                 for r, s in self.rover_stores.items():
                     if s == store:
                         rover_stores_full.add(r)
                         break
            elif match(fact, "at_soil_sample", "*"):
                 soil_samples_at.add(get_parts(fact)[1])
            elif match(fact, "at_rock_sample", "*"):
                 rock_samples_at.add(get_parts(fact)[1])

        # --- Identify Unachieved Goals ---
        unachieved_goals = self.goals - state

        # Group goals by type
        needed_soil_data = set() # waypoint
        needed_rock_data = set() # waypoint
        needed_image_data = set() # (objective, mode)

        for goal in unachieved_goals:
            if match(goal, "communicated_soil_data", "*"):
                needed_soil_data.add(get_parts(goal)[1])
            elif match(goal, "communicated_rock_data", "*"):
                needed_rock_data.add(get_parts(goal)[1])
            elif match(goal, "communicated_image_data", "*", "*"):
                obj, mode = get_parts(goal)[1:]
                needed_image_data.add((obj, mode))

        # If no unachieved goals, heuristic is 0
        if not needed_soil_data and not needed_rock_data and not needed_image_data:
            return 0

        # --- Estimate Cost for Each Unachieved Goal ---

        # Estimate cost for needed soil data
        for soil_wp in needed_soil_data:
            # Check if data is already collected by any rover
            collecting_rover = None
            for rover, have_set in rover_have_soil.items():
                if soil_wp in have_set:
                    collecting_rover = rover
                    break # Found a rover holding the data

            if collecting_rover:
                # Data is collected, need to communicate
                min_comm_cost = float('inf')
                current_wp = rover_locations.get(collecting_rover)
                if current_wp:
                    for comm_wp in self.lander_comm_wps:
                         dist = self.get_distance(collecting_rover, current_wp, comm_wp)
                         min_comm_cost = min(min_comm_cost, dist)
                if min_comm_cost != float('inf'):
                    h += min_comm_cost + 1 # navigate + communicate
                else:
                    h += IMPOSSIBLE_PENALTY # Cannot reach lander from current location

            else:
                # Data is not collected, need to sample and communicate
                # Check if sample exists at the waypoint (must be in current state)
                if soil_wp not in soil_samples_at:
                    # Sample is gone from waypoint. Check if it was ever in the initial state.
                    if soil_wp in self.initial_soil_samples:
                         # Sample existed initially but is gone and no rover has it. Assume impossible (e.g., dropped).
                         h += IMPOSSIBLE_PENALTY
                         continue # Skip this goal
                    else:
                         # Sample never existed at this waypoint. Goal impossible.
                         h += IMPOSSIBLE_PENALTY
                         continue # Skip this goal

                # Sample is available at soil_wp. Find best rover to sample and communicate.
                min_total_cost = float('inf')
                candidate_rovers = [r for r, caps in self.rover_capabilities.items() if "soil" in caps]

                if not candidate_rovers:
                     h += IMPOSSIBLE_PENALTY # No rover can sample soil
                     continue

                for rover in candidate_rovers:
                    current_wp = rover_locations.get(rover)
                    if not current_wp: continue # Rover location unknown

                    store_is_full = rover in rover_stores_full

                    # Cost to get to sample location
                    dist_to_sample = self.get_distance(rover, current_wp, soil_wp)
                    if dist_to_sample == float('inf'): continue # Rover cannot reach sample

                    # Cost for sampling (includes potential drop if store is full)
                    # Note: Dropping requires being at *any* waypoint. We assume the rover is at a waypoint.
                    # We add 1 for the sample action itself.
                    sample_action_cost = 1
                    drop_action_cost = 1 if store_is_full else 0
                    sample_phase_cost = dist_to_sample + drop_action_cost + sample_action_cost

                    # Cost to get to communication point from sample location
                    min_dist_to_comm = float('inf')
                    for comm_wp in self.lander_comm_wps:
                         dist = self.get_distance(rover, soil_wp, comm_wp) # From sample location
                         min_dist_to_comm = min(min_dist_to_comm, dist)

                    if min_dist_to_comm != float('inf'):
                        comm_phase_cost = min_dist_to_comm + 1 # navigate + communicate
                        total_cost = sample_phase_cost + comm_phase_cost
                        min_total_cost = min(min_total_cost, total_cost)

                if min_total_cost != float('inf'):
                    h += min_total_cost
                else:
                    h += IMPOSSIBLE_PENALTY # No suitable rover can reach sample and lander communication point

        # Estimate cost for needed rock data (similar logic to soil)
        for rock_wp in needed_rock_data:
            collecting_rover = None
            for rover, have_set in rover_have_rock.items():
                if rock_wp in have_set:
                    collecting_rover = rover
                    break

            if collecting_rover:
                min_comm_cost = float('inf')
                current_wp = rover_locations.get(collecting_rover)
                if current_wp:
                    for comm_wp in self.lander_comm_wps:
                         dist = self.get_distance(collecting_rover, current_wp, comm_wp)
                         min_comm_cost = min(min_comm_cost, dist)
                if min_comm_cost != float('inf'):
                    h += min_comm_cost + 1
                else:
                    h += IMPOSSIBLE_PENALTY

            else:
                if rock_wp not in rock_samples_at:
                    if rock_wp in self.initial_rock_samples:
                         h += IMPOSSIBLE_PENALTY
                         continue
                    else:
                         h += IMPOSSIBLE_PENALTY
                         continue

                min_total_cost = float('inf')
                candidate_rovers = [r for r, caps in self.rover_capabilities.items() if "rock" in caps]

                if not candidate_rovers:
                     h += IMPOSSIBLE_PENALTY
                     continue

                for rover in candidate_rovers:
                    current_wp = rover_locations.get(rover)
                    if not current_wp: continue

                    store_is_full = rover in rover_stores_full

                    dist_to_sample = self.get_distance(rover, current_wp, rock_wp)
                    if dist_to_sample == float('inf'): continue

                    sample_action_cost = 1
                    drop_action_cost = 1 if store_is_full else 0
                    sample_phase_cost = dist_to_sample + drop_action_cost + sample_action_cost

                    min_dist_to_comm = float('inf')
                    for comm_wp in self.lander_comm_wps:
                         dist = self.get_distance(rover, rock_wp, comm_wp)
                         min_dist_to_comm = min(min_dist_to_comm, dist)

                    if min_dist_to_comm != float('inf'):
                        comm_phase_cost = min_dist_to_comm + 1
                        total_cost = sample_phase_cost + comm_phase_cost
                        min_total_cost = min(min_total_cost, total_cost)

                if min_total_cost != float('inf'):
                    h += min_total_cost
                else:
                    h += IMPOSSIBLE_PENALTY

        # Estimate cost for needed image data
        for obj, mode in needed_image_data:
            # Check if data is already collected by any rover
            collecting_rover = None
            for rover, have_modes in rover_have_image.items():
                if mode in have_modes.get(obj, set()):
                    collecting_rover = rover
                    break

            if collecting_rover:
                # Data is collected, need to communicate
                min_comm_cost = float('inf')
                current_wp = rover_locations.get(collecting_rover)
                if current_wp:
                    for comm_wp in self.lander_comm_wps:
                         dist = self.get_distance(collecting_rover, current_wp, comm_wp)
                         min_comm_cost = min(min_comm_cost, dist)
                if min_comm_cost != float('inf'):
                    h += min_comm_cost + 1
                else:
                    h += IMPOSSIBLE_PENALTY

            else:
                # Data is not collected, need to take image and communicate
                min_total_cost = float('inf')
                candidate_rover_cameras = []
                for camera, info in self.camera_info.items():
                    rover = info["on_board"]
                    # Check if rover exists, is equipped for imaging, and camera supports the mode
                    if rover and "imaging" in self.rover_capabilities.get(rover, set()) and mode in info["supports"]:
                        candidate_rover_cameras.append((rover, camera))

                if not candidate_rover_cameras:
                     h += IMPOSSIBLE_PENALTY # No rover/camera can take this image
                     continue

                # Find waypoints visible from the objective
                image_wps = self.objective_visibility.get(obj, set())
                if not image_wps:
                     h += IMPOSSIBLE_PENALTY # Objective not visible from any waypoint
                     continue

                for rover, camera in candidate_rover_cameras:
                    current_wp = rover_locations.get(rover)
                    if not current_wp: continue # Rover location unknown

                    cal_target = self.camera_info[camera]["calibration_target"]
                    if not cal_target: continue # Camera has no calibration target

                    cal_wps = self.objective_visibility.get(cal_target, set())
                    if not cal_wps: continue # Calibration target not visible from any waypoint

                    # Cost to get to calibration point, calibrate, get to image point, take image
                    min_image_prep_cost_details = (float('inf'), None) # (cost, final_image_wp)

                    # Find the best sequence of (navigate to cal_wp, calibrate, navigate to image_wp, take_image)
                    for cal_wp in cal_wps:
                        dist_to_cal = self.get_distance(rover, current_wp, cal_wp)
                        if dist_to_cal == float('inf'): continue # Cannot reach calibration point

                        for img_wp in image_wps:
                            dist_cal_to_img = self.get_distance(rover, cal_wp, img_wp)
                            if dist_cal_to_img == float('inf'): continue # Cannot reach image point from cal point

                            # Cost includes navigation to cal, calibrate, navigation to image, take image
                            prep_cost = dist_to_cal + 1 + dist_cal_to_img + 1
                            if prep_cost < min_image_prep_cost_details[0]:
                                min_image_prep_cost_details = (prep_cost, img_wp)

                    if min_image_prep_cost_details[0] == float('inf'):
                         continue # Cannot reach calibration or image point with this rover/camera

                    final_image_wp = min_image_prep_cost_details[1]
                    image_prep_cost = min_image_prep_cost_details[0]

                    # Cost to get to communication point from the image location
                    min_dist_to_comm = float('inf')
                    for comm_wp in self.lander_comm_wps:
                         dist = self.get_distance(rover, final_image_wp, comm_wp)
                         min_dist_to_comm = min(min_dist_to_comm, dist)

                    if min_dist_to_comm != float('inf'):
                        comm_phase_cost = min_dist_to_comm + 1 # navigate + communicate
                        total_cost = image_prep_cost + comm_phase_cost
                        min_total_cost = min(min_total_cost, total_cost)

                if min_total_cost != float('inf'):
                    h += min_total_cost
                else:
                    h += IMPOSSIBLE_PENALTY # No suitable rover/camera can complete the sequence

        # Return 0 if all goals are met, otherwise return the sum of costs,
        # capped at IMPOSSIBLE_PENALTY if any goal was deemed impossible.
        return h if h < IMPOSSIBLE_PENALTY else IMPOSSIBLE_PENALTY

