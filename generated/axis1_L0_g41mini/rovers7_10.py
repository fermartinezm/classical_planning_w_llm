from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(have_image rover1 objective1 colour)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    Summary
    -------
    Estimates the number of actions required for rovers to achieve the goal conditions by 
    considering movement, sampling, imaging, calibration, and communication tasks.

    Assumptions
    -----------
    - Each goal fact of types communicated_soil_data, communicated_rock_data, and communicated_image_data 
      must become true.
    - Rover stores hold samples and must be emptied (by dropping) before sampling again.
    - Cameras must be calibrated before imaging and become uncalibrated after taking an image.
    - Roaming between waypoints requires traverse steps (with assumption of shortest path).
    - Communication of data requires the rover to be at a visible location of the lander.
    - Sampling (soil or rock) can be done only when the rover is equipped, at a sample location, and the store is empty.
    - Imaging requires a calibrated camera on board supporting the mode, rover at a visible location from the objective.
    - The heuristic ignores exact ordering of actions but counts minimal necessary actions per goal fact.
    - The heuristic is not admissible but designed to guide greedy best-first search efficiently.

    Heuristic Initialization
    ------------------------
    Extracts and organizes from static facts:
    - Waypoint graph for navigation (traversable waypoint pairs per rover).
    - Camera support modes per camera.
    - Calibration targets for cameras.
    - Mapping from stores to rovers.
    - Samples locations (soil and rock).
    - Which rovers are equipped for soil analysis, rock analysis, and imaging.
    - Lander waypoint and the list of landers.
    - Visibility and visible_from relations for use in navigation and imaging.
    
    Precomputes shortest paths per rover between waypoints using static can_traverse relations to estimate minimal navigation steps.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given state:

    1. If the goal is already reached, return 0.

    2. Extract rovers' current locations from the state.

    3. For each goal fact:
       - For communicated_soil_data(wp):
         * If not yet communicated, estimate actions for a rover to:
           - Have soil analysis from that waypoint or get the soil sample and analyze it (sampling requires empty store).
           - Navigate to lander location visible to that rover.
           - Communicate the soil data.
         * If the soil analysis is not held or no soil sample known, count accordingly.

       - For communicated_rock_data(wp):
         * Same logic as soil data but for rock samples and equipment.

       - For communicated_image_data(objective, mode):
         * If not yet communicated:
           - Find a rover with a camera supporting that mode.
           - If not calibrated, include calibrate actions (requires being at calibration target visible location).
           - Include navigating to a visible waypoint for taking the image.
           - Taking the image (un-calibrates the camera).
           - Navigating to the lander.
           - Communicating the image.
    
    4. For soil/rock/image data already obtained but not communicated, count navigation + communication actions.

    5. For sampling, calibration, imaging, communication and navigation needed, sum minimal required actions assuming shortest paths.

    6. For multiple goals, sum minimal estimated actions independently (may overestimate but guides search well).

    Navigation distance estimates use precomputed shortest paths.

    Overall, the heuristic adds costs of:
    - Navigations to necessary locations.
    - Sampling (1 action).
    - Calibration (1 action).
    - Taking images (1 action).
    - Dropping samples when needed (1 action).
    - Communication (1 action).

    It tries to assign goals to rovers minimizing travel distance heuristically.
    """

    def __init__(self, task):
        # Store goals and static facts
        self.goals = task.goals
        static = task.static

        # --- Extract all rovers ---
        self.rovers = set()
        for fact in static:
            if match(fact, "equipped_for_imaging", "*"):
                self.rovers.add(get_parts(fact)[1])
            elif match(fact, "equipped_for_soil_analysis", "*"):
                self.rovers.add(get_parts(fact)[1])
            elif match(fact, "equipped_for_rock_analysis", "*"):
                self.rovers.add(get_parts(fact)[1])
            elif match(fact, "on_board", "*", "*"):
                self.rovers.add(get_parts(fact)[2])
            elif match(fact, "store_of", "*", "*"):
                self.rovers.add(get_parts(fact)[2])

        # --- Extract stores per rover ---
        self.store_of = {}  # store -> rover
        for fact in static:
            if match(fact, "store_of", "*", "*"):
                store, rover = get_parts(fact)[1:]
                self.store_of[store] = rover

        # --- Extract which rovers are equipped for each analysis ---
        self.rovers_soil = set(get_parts(f)[1] for f in static if match(f, "equipped_for_soil_analysis", "*"))
        self.rovers_rock = set(get_parts(f)[1] for f in static if match(f, "equipped_for_rock_analysis", "*"))
        self.rovers_imaging = set(get_parts(f)[1] for f in static if match(f, "equipped_for_imaging", "*"))

        # --- Extract cameras per rover (on_board) and their supported modes ---
        self.cameras_on_rover = {}  # rover -> set of camera
        self.camera_supports = {}  # camera -> set of mode
        self.calibration_targets = {}  # camera -> objective
        for fact in static:
            if match(fact, "on_board", "*", "*"):
                camera, rover = get_parts(fact)[1:]
                self.cameras_on_rover.setdefault(rover, set()).add(camera)
            elif match(fact, "supports", "*", "*"):
                camera, mode = get_parts(fact)[1:]
                self.camera_supports.setdefault(camera, set()).add(mode)
            elif match(fact, "calibration_target", "*", "*"):
                camera, objective = get_parts(fact)[1:]
                self.calibration_targets[camera] = objective

        # --- Extract visibility from and visible relations ---
        # visible_from(objective, waypoint) means objective visible from waypoint
        self.visible_from_objective = {}  # objective -> set(waypoint)
        for fact in static:
            if match(fact, "visible_from", "*", "*"):
                objective, wp = get_parts(fact)[1:]
                self.visible_from_objective.setdefault(objective, set()).add(wp)

        # visible(waypoint1, waypoint2)
        self.visible_graph = {}  # waypoint -> set(visible waypoints)
        for fact in static:
            if match(fact, "visible", "*", "*"):
                wp1, wp2 = get_parts(fact)[1:]
                self.visible_graph.setdefault(wp1, set()).add(wp2)

        # at_lander(lander, waypoint)
        self.lander_at = {}  # lander -> waypoint
        for fact in static:
            if match(fact, "at_lander", "*", "*"):
                lander, wp = get_parts(fact)[1:]
                self.lander_at[lander] = wp

        # can_traverse(rover, wp1, wp2)
        # Build per rover traversable edges for navigation graph
        self.can_traverse = {}  # rover -> dict wp -> set(neighbors)
        for fact in static:
            if match(fact, "can_traverse", "*", "*", "*"):
                rover, wp1, wp2 = get_parts(fact)[1:]
                self.can_traverse.setdefault(rover, {}).setdefault(wp1, set()).add(wp2)

        # --- Extract soil and rock sample waypoints ---
        self.soil_samples = set(get_parts(f)[1] for f in static if match(f, "at_soil_sample", "*"))
        self.rock_samples = set(get_parts(f)[1] for f in static if match(f, "at_rock_sample", "*"))

        # --- Precompute shortest paths per rover between waypoints using BFS ---
        # This enables quick navigation cost estimates
        self._shortest_paths_cache = {}
        for rover in self.rovers:
            self._shortest_paths_cache[rover] = self._compute_shortest_paths(rover)

    def _compute_shortest_paths(self, rover):
        """Compute shortest distances between all reachable waypoint pairs for a rover.

        Returns a dict: waypoint -> dict(waypoint -> int(distance))
        """
        from collections import deque, defaultdict

        graph = self.can_traverse.get(rover, {})
        distances = defaultdict(dict)

        # BFS from each node
        for start in graph:
            dist = {start: 0}
            queue = deque([start])
            while queue:
                current = queue.popleft()
                for nb in graph.get(current, []):
                    if nb not in dist:
                        dist[nb] = dist[current] + 1
                        queue.append(nb)
            distances[start] = dist
        return distances

    def _nav_cost(self, rover, wp_from, wp_to):
        """Estimate navigation cost (steps) from wp_from to wp_to for given rover."""
        if wp_from == wp_to:
            return 0
        dist_map = self._shortest_paths_cache.get(rover)
        if dist_map is None:
            return 10**6  # large penalty if no info
        dist_from = dist_map.get(wp_from)
        if dist_from is None:
            return 10**6
        dist = dist_from.get(wp_to)
        return dist if dist is not None else 10**6

    def __call__(self, node):
        state = node.state

        # --- Goal check ---
        if self.goals <= state:
            return 0

        # --- Extract rover locations ---
        rover_locations = {}
        for fact in state:
            if match(fact, "at", "*", "*"):
                rover, wp = get_parts(fact)[1:]
                if rover in self.rovers:
                    rover_locations[rover] = wp

        # --- Extract store states (empty/full) ---
        store_status = {}
        for fact in state:
            if match(fact, "empty", "*"):
                store = get_parts(fact)[1]
                store_status[store] = "empty"
            elif match(fact, "full", "*"):
                store = get_parts(fact)[1]
                store_status[store] = "full"

        # --- Extract what analysis samples rovers hold ---
        have_soil = set()  # (rover, wp)
        have_rock = set()  # (rover, wp)
        for fact in state:
            if match(fact, "have_soil_analysis", "*", "*"):
                rover, wp = get_parts(fact)[1:]
                have_soil.add((rover, wp))
            elif match(fact, "have_rock_analysis", "*", "*"):
                rover, wp = get_parts(fact)[1:]
                have_rock.add((rover, wp))

        # --- Extract calibrated cameras per rover ---
        calibrated = set()  # (camera, rover)
        for fact in state:
            if match(fact, "calibrated", "*", "*"):
                camera, rover = get_parts(fact)[1:]
                calibrated.add((camera, rover))

        # --- Extract images rovers hold ---
        have_image = set()  # (rover, objective, mode)
        for fact in state:
            if match(fact, "have_image", "*", "*", "*"):
                rover, obj, mode = get_parts(fact)[1:]
                have_image.add((rover, obj, mode))

        # --- Extract communicated goals completed ---
        communicated_soil = set(get_parts(f)[1] for f in state if match(f, "communicated_soil_data", "*"))
        communicated_rock = set(get_parts(f)[1] for f in state if match(f, "communicated_rock_data", "*"))
        communicated_image = set((get_parts(f)[1], get_parts(f)[2]) for f in state if match(f, "communicated_image_data", "*", "*"))

        # --- Extract camera modes per camera on rovers ---
        cameras_calibrated_on_rover = {}
        for (cam, rov) in calibrated:
            cameras_calibrated_on_rover.setdefault(rov, set()).add(cam)
        cameras_on_rover = self.cameras_on_rover

        # --- Extract which rovers have which store empty/full ---
        rover_store_empty = {}
        rover_store_full = {}
        for store, rover in self.store_of.items():
            status = store_status.get(store, "empty")
            if status == "empty":
                rover_store_empty[rover] = rover_store_empty.get(rover, 0) + 1
            else:
                rover_store_full[rover] = rover_store_full.get(rover, 0) + 1

        # --- Identify landers and their waypoints ---
        # Usually one lander, but could be more.
        landers = list(self.lander_at.keys())

        # --- Helper to find shortest nav cost for one rover to one target ---
        def shortest_nav(rover, src_wp, tgt_wps):
            # Returns minimal cost to any wp in tgt_wps from src_wp
            costs = [self._nav_cost(rover, src_wp, wp) for wp in tgt_wps]
            return min(costs) if costs else 10**6

        # --- Heuristic cost accumulator ---
        total_cost = 0

        # --- Assign each goal fact a minimal cost by considering best rover that can fulfill it ---

        # 1) communicated_soil_data waypoints not yet communicated
        soil_goals = [get_parts(g)[1] for g in self.goals if match(g, "communicated_soil_data", "*")]
        for wp in soil_goals:
            if wp in communicated_soil:
                continue  # already done

            # We must ensure some rover has soil analysis of wp and it is communicated.

            # Possible ways:
            # a) Some rover already has soil analysis for wp and can navigate to a lander waypoint visible to communicate.
            # b) Otherwise, some rover must sample soil at wp (which requires empty store) and navigate to lander, then communicate.

            # Find rovers equipped for soil analysis
            candidate_rovers = list(self.rovers_soil)

            best_cost = 10**6
            for r in candidate_rovers:
                # rover location
                loc = rover_locations.get(r)
                if loc is None:
                    continue

                # check if rover already has soil analysis for wp
                already_have = (r, wp) in have_soil

                actions = 0
                # If already_have, just navigate to lander and communicate
                if already_have:
                    # Navigation to any lander waypoint visible from rover position?
                    # Choose lander waypoint visible from rover loc (visible relation indicates visibility)
                    # Also the lander waypoint must be visible from rover loc
                    # We assume landers are communicating at their waypoint, need rover adjacent visible waypoint.

                    # We consider "visible" relation from rover loc to lander wp
                    lander_wps_visible = []
                    for lander in landers:
                        lander_wp = self.lander_at[lander]
                        # rover loc must be visible to lander waypoint or vice versa (PDDL visible relation is uni-directional)
                        if loc == lander_wp or loc in self.visible_graph.get(lander_wp, set()) or lander_wp in self.visible_graph.get(loc, set()):
                            lander_wps_visible.append(lander_wp)
                    if not lander_wps_visible:
                        # no visible lander waypoint from rover loc, try to navigate closer
                        # So calculate min distance to waypoint visible to lander
                        visible_to_lander = set()
                        for lander in landers:
                            visible_to_lander.update(self.visible_graph.get(self.lander_at[lander], set()))
                        nav_cost = shortest_nav(r, loc, list(visible_to_lander))
                        actions += nav_cost
                    else:
                        # The rover can communicate from current loc (cost 0 movement)
                        actions += 0

                    # Communication action
                    actions += 1  # communicate_soil_data action

                    if actions < best_cost:
                        best_cost = actions
                    continue

                # Otherwise:
                # Steps:
                # - Navigate to soil sample wp (if not there)
                # - Sample soil (requires empty store, otherwise drop first)
                # - Navigate to lander (visible waypoint)
                # - Communicate soil data

                # Check if wp is a soil sample location (or if it already disappeared? We can check state)
                sample_exists = False
                for f in state:
                    if match(f, "at_soil_sample", wp):
                        sample_exists = True
                        break

                if not sample_exists:
                    # The soil sample is already taken but no analyzed data? This is abnormal, treat as need to communicate only.
                    # So just try communicate with rover holding soil analysis for wp (none), so high cost.
                    continue

                actions = 0
                # Navigation to sample site
                nav_to_sample = self._nav_cost(r, loc, [wp])
                actions += nav_to_sample

                # Check if rover store is empty - if not, add a drop
                if rover_store_empty.get(r, 0) == 0:
                    actions += 1  # drop action

                actions += 1  # sample_soil

                # Navigation to lander visible location
                loc_after_sample = wp
                lander_wps_visible = []
                for lander in landers:
                    lander_wp = self.lander_at[lander]
                    # check visible from sample location or closeby
                    if loc_after_sample == lander_wp or loc_after_sample in self.visible_graph.get(lander_wp, set()) or lander_wp in self.visible_graph.get(loc_after_sample, set()):
                        lander_wps_visible.append(lander_wp)
                if not lander_wps_visible:
                    visible_to_lander = set()
                    for lander in landers:
                        visible_to_lander.update(self.visible_graph.get(self.lander_at[lander], set()))
                    nav_to_lander = self._nav_cost(r, loc_after_sample, list(visible_to_lander))
                else:
                    nav_to_lander = 0
                actions += nav_to_lander

                actions += 1  # communicate_soil_data

                if actions < best_cost:
                    best_cost = actions

            total_cost += best_cost

        # 2) communicated_rock_data waypoints not yet communicated
        rock_goals = [get_parts(g)[1] for g in self.goals if match(g, "communicated_rock_data", "*")]
        for wp in rock_goals:
            if wp in communicated_rock:
                continue

            candidate_rovers = list(self.rovers_rock)
            best_cost = 10**6
            for r in candidate_rovers:
                loc = rover_locations.get(r)
                if loc is None:
                    continue

                already_have = (r, wp) in have_rock

                if already_have:
                    # Navigate to lander visible waypoint and communicate
                    lander_wps_visible = []
                    for lander in landers:
                        lander_wp = self.lander_at[lander]
                        if loc == lander_wp or loc in self.visible_graph.get(lander_wp, set()) or lander_wp in self.visible_graph.get(loc, set()):
                            lander_wps_visible.append(lander_wp)
                    if not lander_wps_visible:
                        visible_to_lander = set()
                        for lander in landers:
                            visible_to_lander.update(self.visible_graph.get(self.lander_at[lander], set()))
                        nav_cost = shortest_nav(r, loc, list(visible_to_lander))
                    else:
                        nav_cost = 0
                    cost = nav_cost + 1  # communicate_rock_data
                    if cost < best_cost:
                        best_cost = cost
                    continue

                sample_exists = False
                for f in state:
                    if match(f, "at_rock_sample", wp):
                        sample_exists = True
                        break
                if not sample_exists:
                    continue

                cost = 0

                nav_to_sample = self._nav_cost(r, loc, [wp])
                cost += nav_to_sample

                if rover_store_empty.get(r, 0) == 0:
                    cost += 1  # drop

                cost += 1  # sample_rock

                loc_after_sample = wp
                lander_wps_visible = []
                for lander in landers:
                    lander_wp = self.lander_at[lander]
                    if loc_after_sample == lander_wp or loc_after_sample in self.visible_graph.get(lander_wp, set()) or lander_wp in self.visible_graph.get(loc_after_sample, set()):
                        lander_wps_visible.append(lander_wp)
                if not lander_wps_visible:
                    visible_to_lander = set()
                    for lander in landers:
                        visible_to_lander.update(self.visible_graph.get(self.lander_at[lander], set()))
                    nav_to_lander = self._nav_cost(r, loc_after_sample, list(visible_to_lander))
                else:
                    nav_to_lander = 0
                cost += nav_to_lander

                cost += 1  # communicate_rock_data

                if cost < best_cost:
                    best_cost = cost

            total_cost += best_cost

        # 3) communicated_image_data not yet communicated
        image_goals = [(get_parts(g)[1], get_parts(g)[2]) for g in self.goals if match(g, "communicated_image_data", "*", "*")]
        for (obj, mode) in image_goals:
            if (obj, mode) in communicated_image:
                continue

            # For each goal, try to find best rover-camera combo to minimize cost:
            best_cost = 10**6

            for r in self.rovers_imaging:
                loc = rover_locations.get(r)
                if loc is None:
                    continue

                cameras = cameras_on_rover.get(r, set())
                # Cameras supporting mode
                cams_supporting = [cam for cam in cameras if mode in self.camera_supports.get(cam, set())]
                if not cams_supporting:
                    continue  # no suitable camera on rover for this mode

                for cam in cams_supporting:
                    cost = 0

                    # Find calibration target objective and its visible waypoints
                    calib_obj = self.calibration_targets.get(cam)
                    # Waypoints from where calib_obj is visible
                    calib_targets_wps = self.visible_from_objective.get(calib_obj, set())

                    # Waypoints from where obj is visible (for imaging)
                    image_targets_wps = self.visible_from_objective.get(obj, set())

                    # If camera is calibrated on rover, no calibrate action needed, else add 1 action and navigate to calib spot
                    if (cam, r) not in calibrated:
                        # Navigate to some waypoint from which calib_obj is visible
                        nav_to_calib = self._nav_cost(r, loc, list(calib_targets_wps)) if calib_targets_wps else 10**6
                        cost += nav_to_calib + 1  # calibrate action
                        loc_after_calib = None
                        # Assume rover at calibration waypoint after calibrate
                        if calib_targets_wps:
                            # pick the closest one for navigation step
                            cost -= self._nav_cost(r, loc, list(calib_targets_wps))  # to avoid double counting, we just add 1 calibration and navigation once
                            loc_after_calib = min(calib_targets_wps, key=lambda wp: self._nav_cost(r, loc, [wp]))
                        else:
                            loc_after_calib = loc
                    else:
                        loc_after_calib = loc

                    # Navigate to some waypoint to take image (visible from obj)
                    nav_to_image_spots = self._nav_cost(r, loc_after_calib, list(image_targets_wps)) if image_targets_wps else 10**6
                    cost += nav_to_image_spots + 1  # take_image action

                    # After take_image camera becomes uncalibrated for that camera, so assume next tasks need recalibrate

                    # Navigate to lander visible spot for communication
                    lander_wps_visible = []
                    for lander in landers:
                        lander_wp = self.lander_at[lander]
                        # rover loc after image position might be one of image_targets_wps; assume for heuristic it stays there
                        # Check visible between loc_after_image and lander
                        if loc_after_calib in self.visible_graph.get(lander_wp, set()) or lander_wp in self.visible_graph.get(loc_after_calib, set()) or loc_after_calib == lander_wp:
                            lander_wps_visible.append(lander_wp)
                    if not lander_wps_visible:
                        visible_to_lander = set()
                        for lander in landers:
                            visible_to_lander.update(self.visible_graph.get(self.lander_at[lander], set()))
                        nav_to_lander = self._nav_cost(r, loc_after_calib, list(visible_to_lander))
                    else:
                        nav_to_lander = 0
                    cost += nav_to_lander + 1  # communicate_image_data

                    if cost < best_cost:
                        best_cost = cost

            total_cost += best_cost

        return total_cost
