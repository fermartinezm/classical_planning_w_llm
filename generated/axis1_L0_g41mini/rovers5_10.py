from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """Check if a PDDL fact matches a given pattern, supporting wildcards."""
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    Summary
    -------
    This heuristic estimates the number of actions required to reach the goal state from the current state.
    The plan consists of making required observations (soil analysis, rock analysis, imaging) at objectives or samples,
    calibrating cameras if needed, and communicating the collected data back to the lander.

    Assumptions
    -----------
    - Each rover can hold exactly one sample at a time in its store.
    - Rovers have specific sensor capabilities: soil analysis, rock analysis, imaging.
    - Communication must be done from a waypoint visible to the lander.
    - Calibration is required before taking images and always invalidates after image capture.
    - Samples must be taken where soil or rock samples exist.
    - Images must be taken from waypoints visible from the objectives, with cameras requiring calibration.
    - Navigation actions move rovers between waypoints connected by can_traverse, and visible predicates limit these traversals.
    - Dropping a sample store empties it, allowing a next sample.
    - Communication actions happen once per analyzed sample or image.

    Heuristic Initialization
    ------------------------
    The constructor extracts and indexes static facts and goal predicates:
    - It computes:
      * All waypoints, rovers, stores, cameras, landers, objectives.
      * Mapping of store -> rover.
      * The lander(s) and their waypoint location(s).
      * The visibility graph between waypoints (using visible predicates).
      * The can_traverse edges per rover.
      * Calibration targets per camera.
      * Camera modes support per camera.
      * For each rover, which cameras are on board and what analysis they support.
      * Which rover can perform soil analysis, rock analysis, and imaging.
    - Parses the goals:
      * Which samples (soil or rock) need to be communicated (waypoints).
      * Which images (objective+mode) need to be communicated.
    - Prepares for quick heuristic calculations:
      * Distances between waypoints for navigation cost estimation, by rover using the can_traverse + visible predicate graph.
      * Visible_from relations for images.
    
    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. If the state is a goal state, heuristic value is 0.

    2. For each goal communication not yet achieved:
        a) For soil and rock analysis goals:
            - Check if rover(s) have already analyzed samples at the goal waypoints.
            - Otherwise, estimate the cost to navigate to the sample waypoint, sample it,
              possibly drop a full store first, and communicate it to a lander nearby.

        b) For image communication goals:
            - Check if rover(s) have the image taken.
            - Otherwise, estimate the cost to
              i) calibrate the camera on some rover (navigate to calibration waypoint)
              ii) navigate to a suitable waypoint visible from the objective,
              iii) take the image,
              iv) navigate to lander waypoint,
              v) communicate the image.
            - Account for the fact that calibration is lost after taking the image.

    3. When estimating distances, use shortest paths considering rover-specific traversability.

    4. Assume one rover does each required goal subtask; pick the minimal action cost over rovers.

    5. Approximate costs as sums of:
        - Navigation steps (1 action per navigate).
        - Sampling takes 1 action.
        - Dropping a sample store takes 1 action.
        - Calibrating takes 1 action.
        - Taking an image takes 1 action.
        - Communicating takes 1 action.
       This heuristic ignores some order dependencies and concurrent achievements for efficiency.

    6. The total heuristic is the sum over all unresolved goals of their minimal achievable costs.

    This heuristic is not admissible but aims to guide a greedy best-first search by closely
    estimating required actions with domain-specific reasoning and distance estimation.
    """

    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static

        # Extract typed objects from static facts and goals
        self.rovers = set()
        self.waypoints = set()
        self.stores = set()
        self.cameras = set()
        self.landers = set()
        self.objectives = set()
        self.modes = set()

        # Mapping store -> rover
        self.store_of = {}

        # Mapping (lander) -> waypoint
        self.lander_positions = {}

        # Visible relations: waypoint -> set of visible waypoints
        self.visible_graph = {}

        # For each rover: adjacency graph of waypoints it can traverse (considering visible predicates)
        self.rover_graphs = {}

        # Equipped capabilities per rover (soil, rock, imaging)
        self.equipped_for_soil_analysis = set()
        self.equipped_for_rock_analysis = set()
        self.equipped_for_imaging = set()

        # Cameras on board per rover
        self.on_board_camera = {}  # camera -> rover

        # Calibration targets per camera
        self.calibration_target = {}  # camera -> objective

        # Camera modes support
        self.camera_supports = {}  # camera -> set of modes

        # visible_from relations: objective -> set of waypoints from which objective is visible
        self.visible_from = {}

        # Parse static facts
        for fact in self.static:
            parts = get_parts(fact)
            pred = parts[0]

            if pred == "at_lander":
                lander, wp = parts[1], parts[2]
                self.landers.add(lander)
                self.waypoints.add(wp)
                self.lander_positions[lander] = wp

            elif pred == "store_of":
                store, rover = parts[1], parts[2]
                self.stores.add(store)
                self.rovers.add(rover)
                self.store_of[store] = rover

            elif pred == "can_traverse":
                rover, wp_from, wp_to = parts[1], parts[2], parts[3]
                self.rovers.add(rover)
                self.waypoints.add(wp_from)
                self.waypoints.add(wp_to)
                if rover not in self.rover_graphs:
                    self.rover_graphs[rover] = {}
                self.rover_graphs[rover].setdefault(wp_from, set()).add(wp_to)

            elif pred == "visible":
                wp1, wp2 = parts[1], parts[2]
                self.waypoints.add(wp1)
                self.waypoints.add(wp2)
                self.visible_graph.setdefault(wp1, set()).add(wp2)
                self.visible_graph.setdefault(wp2, set())  # ensure key exists

            elif pred == "equipped_for_soil_analysis":
                self.equipped_for_soil_analysis.add(parts[1])
            elif pred == "equipped_for_rock_analysis":
                self.equipped_for_rock_analysis.add(parts[1])
            elif pred == "equipped_for_imaging":
                self.equipped_for_imaging.add(parts[1])

            elif pred == "on_board":
                camera, rover = parts[1], parts[2]
                self.cameras.add(camera)
                self.rovers.add(rover)
                self.on_board_camera[camera] = rover

            elif pred == "calibration_target":
                camera, objective = parts[1], parts[2]
                self.cameras.add(camera)
                self.objectives.add(objective)
                self.calibration_target[camera] = objective

            elif pred == "supports":
                camera, mode = parts[1], parts[2]
                self.cameras.add(camera)
                self.modes.add(mode)
                self.camera_supports.setdefault(camera, set()).add(mode)

            elif pred == "visible_from":
                objective, wp = parts[1], parts[2]
                self.objectives.add(objective)
                self.waypoints.add(wp)
                self.visible_from.setdefault(objective, set()).add(wp)

            elif pred == "rover":
                self.rovers.add(parts[1])

            elif pred == "waypoint":
                self.waypoints.add(parts[1])

            elif pred == "store":
                self.stores.add(parts[1])

            elif pred == "camera":
                self.cameras.add(parts[1])

            elif pred == "lander":
                self.landers.add(parts[1])

            elif pred == "objective":
                self.objectives.add(parts[1])

            elif pred == "mode":
                self.modes.add(parts[1])

        # Ensure all waypoints exist in visible graph and rover graphs (even if empty)
        for wp in self.waypoints:
            self.visible_graph.setdefault(wp, set())
            for rover in self.rovers:
                self.rover_graphs.setdefault(rover, {})
                self.rover_graphs[rover].setdefault(wp, set())

        # Build full rover navigation graphs by filtering on visible predicates:
        # Edges are only allowed if waypoint visibility is mutual (visible from current to next)
        # and can_traverse indicates rover can move from current to next
        # We filter edges where "visible from current wp" to next wp is true, consistent with domain action preconditions.
        
        # The domain requires "visible" predicate from y to z for navigate(y->z).
        # So edges in rover_graphs must confirm visible(y,z).
        for rover in self.rover_graphs:
            edges = {}
            for wp_from, neighbors in self.rover_graphs[rover].items():
                filtered = set()
                for wp_to in neighbors:
                    if wp_to in self.visible_graph.get(wp_from, []):
                        filtered.add(wp_to)
                edges[wp_from] = filtered
            self.rover_graphs[rover] = edges

        # For distance computation cache per rover
        self._dist_cache = {}

        # Parse goal requirements and store:
        # Track soil and rock sample analysis waypoints needing to be communicated,
        # image communication needs with objective and mode.
        self.goals_soil_comm = set()  # waypoints for communicated_soil_data(?w)
        self.goals_rock_comm = set()  # waypoints for communicated_rock_data(?w)
        self.goals_image_comm = set()  # (objective, mode) pairs for communicated_image_data(?o ?m)

        for goal in self.goals:
            parts = get_parts(goal)
            pred = parts[0]
            if pred == "communicated_soil_data":
                self.goals_soil_comm.add(parts[1])
            elif pred == "communicated_rock_data":
                self.goals_rock_comm.add(parts[1])
            elif pred == "communicated_image_data":
                self.goals_image_comm.add((parts[1], parts[2]))

        # For quick inverse queries: camera -> rover
        # Already in self.on_board_camera

        # For each rover, which cameras belong
        self.cameras_on_rover = {}
        for cam, rover in self.on_board_camera.items():
            self.cameras_on_rover.setdefault(rover, set()).add(cam)

    def _shortest_path_length(self, rover, start, goal):
        """Compute shortest number of navigation steps for rover to move from start to goal.
        Return a finite integer or a large number if no path exists.
        Uses BFS caching results per rover."""
        if (rover, start, goal) in self._dist_cache:
            return self._dist_cache[(rover, start, goal)]

        graph = self.rover_graphs[rover]
        if start == goal:
            self._dist_cache[(rover, start, goal)] = 0
            return 0

        from collections import deque
        queue = deque([(start, 0)])
        visited = {start}

        while queue:
            node, dist = queue.popleft()
            for neighbor in graph.get(node, []):
                if neighbor == goal:
                    self._dist_cache[(rover, start, goal)] = dist + 1
                    return dist + 1
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))

        # No path found: assign large cost but finite (say 1000)
        self._dist_cache[(rover, start, goal)] = 1000
        return 1000

    def _get_rover_position(self, state, rover):
        """Find the current waypoint of the rover in the state."""
        # Look for (at rover waypoint)
        for fact in state:
            if match(fact, "at", rover, "*"):
                return get_parts(fact)[2]
        return None  # Should not happen in normal state

    def _store_empty(self, state, store):
        """Check if store is empty in the state."""
        return f"(empty {store})" in state

    def _store_full(self, state, store):
        """Check if store is full in the state."""
        return f"(full {store})" in state

    def _have_soil_analysis(self, state, rover, wp):
        """Check if rover has soil analysis from waypoint wp."""
        return f"(have_soil_analysis {rover} {wp})" in state

    def _have_rock_analysis(self, state, rover, wp):
        """Check if rover has rock analysis from waypoint wp."""
        return f"(have_rock_analysis {rover} {wp})" in state

    def _calibrated(self, state, camera, rover):
        """Check if camera is calibrated on rover."""
        return f"(calibrated {camera} {rover})" in state

    def _have_image(self, state, rover, objective, mode):
        """Check if rover has an image of objective in mode."""
        return f"(have_image {rover} {objective} {mode})" in state

    def _communicated_soil_data(self, state, wp):
        """Check if soil data for waypoint communicated."""
        return f"(communicated_soil_data {wp})" in state

    def _communicated_rock_data(self, state, wp):
        """Check if rock data for waypoint communicated."""
        return f"(communicated_rock_data {wp})" in state

    def _communicated_image_data(self, state, objective, mode):
        """Check if image data communicated for objective and mode."""
        return f"(communicated_image_data {objective} {mode})" in state

    def _at_soil_sample(self, state, wp):
        """Check if there is a soil sample at waypoint wp."""
        return f"(at_soil_sample {wp})" in state

    def _at_rock_sample(self, state, wp):
        """Check if there is a rock sample at waypoint wp."""
        return f"(at_rock_sample {wp})" in state

    def _find_rovers_with_capability(self, capability_set):
        """Return list of rovers that have a given capability."""
        # capability_set is a set of rover names
        return list(capability_set)

    def __call__(self, node):
        state = node.state
        if self.goals <= state:
            return 0

        total_cost = 0

        # Precompute lander waypoints for communication
        lander_wps = set(self.lander_positions.values())

        # Compute rover positions once
        rover_positions = {r: self._get_rover_position(state, r) for r in self.rovers}

        # -------------------------------------
        # 1) Soil data communication goals
        # Need: analyze soil sample (if not done), then communicate
        for wp in self.goals_soil_comm:
            if self._communicated_soil_data(state, wp):
                continue

            # First find rovers with soil analysis ability
            candidate_rovers = [r for r in self.rovers if r in self.equipped_for_soil_analysis]

            min_cost = 10000
            for r in candidate_rovers:
                cost = 0
                pos = rover_positions[r]
                # Check if rover already has soil analysis at wp:
                if self._have_soil_analysis(state, r, wp):
                    # Only communication needed: navigate to lander and communicate
                    # Find nearest lander waypoint reachable by rover
                    min_nav = min(
                        self._shortest_path_length(r, pos, lwp) for lwp in lander_wps
                    )
                    cost += min_nav + 1  # 1 to communicate
                    min_cost = min(min_cost, cost)
                    continue

                # Otherwise, cost includes:
                # Navigate to soil sample , sample, possibly drop, navigate to lander, communicate

                # Check that there is still soil sample at wp (if not, sample is assumed already taken):
                soil_sample = self._at_soil_sample(state, wp)
                # Sample action requires store empty, so if full, we must drop first.

                store = None
                # Find store of this rover:
                for s, ro in self.store_of.items():
                    if ro == r:
                        store = s
                        break
                # Check if store full in state:
                drop_cost = 1 if (store is not None and self._store_full(state, store)) else 0

                # Navigation to sample
                nav_to_sample = self._shortest_path_length(r, pos, wp)

                # Sample action costs 1

                # Navigation to lander waypoint
                nav_to_lander = min(
                    self._shortest_path_length(r, wp, lwp) for lwp in lander_wps
                )

                # Communication action cost 1

                cost = drop_cost + nav_to_sample + 1 + nav_to_lander + 1

                min_cost = min(min_cost, cost)

            total_cost += min_cost

        # -------------------------------------
        # 2) Rock data communication goals
        for wp in self.goals_rock_comm:
            if self._communicated_rock_data(state, wp):
                continue

            candidate_rovers = [r for r in self.rovers if r in self.equipped_for_rock_analysis]

            min_cost = 10000
            for r in candidate_rovers:
                cost = 0
                pos = rover_positions[r]

                if self._have_rock_analysis(state, r, wp):
                    min_nav = min(
                        self._shortest_path_length(r, pos, lwp) for lwp in lander_wps
                    )
                    cost += min_nav + 1
                    min_cost = min(min_cost, cost)
                    continue

                rock_sample = self._at_rock_sample(state, wp)

                store = None
                for s, ro in self.store_of.items():
                    if ro == r:
                        store = s
                        break
                drop_cost = 1 if (store is not None and self._store_full(state, store)) else 0

                nav_to_sample = self._shortest_path_length(r, pos, wp)

                nav_to_lander = min(
                    self._shortest_path_length(r, wp, lwp) for lwp in lander_wps
                )

                cost = drop_cost + nav_to_sample + 1 + nav_to_lander + 1

                min_cost = min(min_cost, cost)

            total_cost += min_cost

        # -------------------------------------
        # 3) Image data communication goals: these are (objective, mode) pairs.
        # Need to check if image taken and communicated; else plan calibrate + navigate + image + navigate + communicate

        for objective, mode in self.goals_image_comm:
            if self._communicated_image_data(state, objective, mode):
                continue

            # Find rovers with imaging equipped
            candidate_rovers = [r for r in self.rovers if r in self.equipped_for_imaging]

            min_cost = 100000
            for r in candidate_rovers:
                cost = 0
                pos = rover_positions[r]

                # Check if rover has the image (but not communicated)
                if self._have_image(state, r, objective, mode):
                    nav_to_lander = min(
                        self._shortest_path_length(r, pos, lwp) for lwp in lander_wps
                    )
                    cost += nav_to_lander + 1  # communicate
                    min_cost = min(min_cost, cost)
                    continue

                # Else need to:
                # 1) calibrate a camera capable for this mode on this rover
                # 2) navigate to waypoint visible from objective to take image
                # 3) take image
                # 4) navigate to lander
                # 5) communicate image

                cameras = self.cameras_on_rover.get(r, set())

                # Cameras on this rover that support desired mode and calibration target matches objective
                suitable_cameras = set()
                for cam in cameras:
                    if (
                        mode in self.camera_supports.get(cam, set())
                        and self.calibration_target.get(cam, None) == objective
                    ):
                        suitable_cameras.add(cam)
                if not suitable_cameras:
                    continue  # no suitable camera in this rover

                # Cost components to minimize over cameras:

                cam_costs = []

                for cam in suitable_cameras:
                    camera_calibrated = self._calibrated(state, cam, r)
                    # If not calibrated, must calibrate first

                    # Calibration requires rover at waypoint visible from calibration target objective
                    calib_obj = self.calibration_target[cam]
                    # Waypoints from which calibration target is visible:
                    calib_visible_wps = self.visible_from.get(calib_obj, set())
                    if not calib_visible_wps:
                        # No place to calibrate, skip this camera
                        continue

                    # Waypoints from which objective is visible (required for imaging)
                    image_visible_wps = self.visible_from.get(objective, set())
                    if not image_visible_wps:
                        # No image spots, skip
                        continue

                    # Step 1: calibrate if needed -> navigate to any calibration-visible waypoint:
                    if camera_calibrated:
                        nav_to_calib_wp = 0
                    else:
                        nav_to_calib_wp = min(
                            self._shortest_path_length(r, pos, wp) for wp in calib_visible_wps
                        )
                    cost_calibrate = nav_to_calib_wp + (0 if camera_calibrated else 1)

                    # After calibration, rover must navigate to image-visible waypoint to take image
                    # After calibrate, rover is at calib waypoint (pos replaced to calib waypoint if cal needed)
                    pos_after_cal = (
                        pos
                        if camera_calibrated
                        else min(
                            calib_visible_wps,
                            key=lambda wp: self._shortest_path_length(r, pos, wp),
                        )
                    )

                    nav_to_image_wp = min(
                        self._shortest_path_length(r, pos_after_cal, wp) for wp in image_visible_wps
                    )

                    # Take image cost 1
                    cost_take_image = 1

                    # After taking image, calibration lost - no penalty here since we only care about this image

                    # Navigate to lander to communicate
                    nav_to_lander = min(
                        self._shortest_path_length(
                            r,  
                            next(
                                (wp for wp in image_visible_wps if self._shortest_path_length(r, pos_after_cal, wp) == nav_to_image_wp),
                                pos_after_cal,
                            ),
                            lwp,
                        )
                        for lwp in lander_wps
                    )

                    # Communicate cost 1
                    cost_communicate = 1

                    total_cam_cost = (
                        cost_calibrate + nav_to_image_wp + cost_take_image + nav_to_lander + cost_communicate
                    )

                    cam_costs.append(total_cam_cost)

                if cam_costs:
                    cam_min = min(cam_costs)
                    min_cost = min(min_cost, cam_min)

            total_cost += min_cost

        return total_cost
