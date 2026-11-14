from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


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
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    Summary
    -------
    This heuristic estimates the number of actions needed to achieve the goal from a given state by
    evaluating the remaining required key steps:
    - Which images, soil analyses, and rock analyses still need to be communicated.
    - Which calibration steps remain for cameras on rovers that must take images.
    - The estimated minimal travel steps for rovers to reach required waypoints to acquire data and communicate it.

    Assumptions
    -----------
    - The rover can only carry one soil or rock sample at a time. An empty store is required to sample.
    - Each store full of sample needs a drop action to empty it.
    - Calibrations become uncalibrated after taking an image and must be redone for subsequent images by that camera on the same rover.
    - Communication of data requires the rover to be at a visible waypoint from the lander.
    - Navigation between waypoints costs 1 per step; visible and can_traverse relations define connectivity.
    - Taking an image requires the corresponding calibration and camera support for the mode.
    - The heuristic ignores exact optimal routing but approximates minimal needed navigation steps by BFS shortest paths on the static can_traverse graph.
    - The heuristic is not admissible but attempts to minimize nodes expanded by focusing on realistic cost components.

    Heuristic Initialization
    ------------------------
    - Extract from static facts:
      - The directed graph of waypoints with edges where the rover can traverse and where visible relation holds (for navigation).
      - The waypoints of landers.
      - Mapping of cameras to their rovers and the modes they support.
      - Calibration targets for each camera and associated objectives.
    - Extract equipped capabilities for each rover (imaging, soil analysis, rock analysis).
    - Extract store associations to rovers.
    - Extract visibility from objectives to waypoints.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. Identify which goals have not yet been achieved:
       - For all communicated_image_data, communicated_soil_data, and communicated_rock_data goals,
         determine which are missing in the current state.
    2. For each missing communicated fact, determine:
       - The rover(s) capable of obtaining that data (equipment and camera support).
       - The waypoint(s) from which the data can be acquired or communicated.
    3. Estimate minimal navigation steps for relevant rovers:
       - From rover current position to objectives/waypoints needed for sampling or imaging.
       - From rover positions to a visible waypoint from the lander for communication.
    4. For each missing piece of data:
       - Add cost for the required actions:
         - Sampling soil/rock (requires rover at sample location, store empty).
         - Calibrating camera for image (if required).
         - Taking image (resets calibration).
         - Drop store if full before next sample.
         - Communication from visible waypoint of lander.
       - Account for travel between these locations using shortest path lengths.
    5. Sum all costs as a heuristic estimate.

    The heuristic returns 0 if the goal is reached (no missing communicated data).
    """

    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static
        # Extract waypoints and build a graph of navigable edges
        # For each rover, edges where can_traverse and visible holds define navigation possibilities
        self.waypoints = set()
        self.landers = set()
        self.rovers = set()
        self.stores = dict()  # store -> rover
        self.rover_equipped = {
            "imaging": set(),
            "soil": set(),
            "rock": set()
        }
        self.cameras_on_rover = dict()  # camera -> rover
        self.camera_modes = dict()  # camera -> set of modes
        self.calibration_target = dict()  # camera -> objective
        self.objective_visible_from = dict()  # objective -> set of waypoints
        self.lander_locations = set()  # waypoints with landers
        self.can_traverse_edges = dict()  # for each rover: waypoint -> set of reachable waypoints
        self.visible_map = dict()  # waypoint -> set of waypoints visible from it

        # A set of all waypoints for quick references
        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == "visible":
                w1, w2 = parts[1], parts[2]
                self.visible_map.setdefault(w1, set()).add(w2)
                # visible is likely symmetric but not guaranteed; only as stated in static facts
            elif parts[0] == "at_lander":
                # (at_lander lander waypoint)
                _, lander, waypoint = parts
                self.landers.add(lander)
                self.lander_locations.add(waypoint)
                self.waypoints.add(waypoint)
            elif parts[0] == "can_traverse":
                # (can_traverse rover waypoint1 waypoint2)
                _, rover, w_from, w_to = parts
                if rover not in self.can_traverse_edges:
                    self.can_traverse_edges[rover] = dict()
                self.can_traverse_edges[rover].setdefault(w_from, set()).add(w_to)
                self.waypoints.add(w_from)
                self.waypoints.add(w_to)
                self.rovers.add(rover)
            elif parts[0] == "equipped_for_imaging":
                self.rover_equipped["imaging"].add(parts[1])
            elif parts[0] == "equipped_for_soil_analysis":
                self.rover_equipped["soil"].add(parts[1])
            elif parts[0] == "equipped_for_rock_analysis":
                self.rover_equipped["rock"].add(parts[1])
            elif parts[0] == "store_of":
                # (store_of store rover)
                _, store, rover = parts
                self.stores[store] = rover
                self.rovers.add(rover)
            elif parts[0] == "on_board":
                # (on_board camera rover)
                _, camera, rover = parts
                self.cameras_on_rover[camera] = rover
                self.rovers.add(rover)
            elif parts[0] == "supports":
                # (supports camera mode)
                _, camera, mode = parts
                self.camera_modes.setdefault(camera, set()).add(mode)
            elif parts[0] == "calibration_target":
                # (calibration_target camera objective)
                _, camera, objective = parts
                self.calibration_target[camera] = objective
            elif parts[0] == "visible_from":
                # (visible_from objective waypoint)
                _, objective, waypoint = parts
                self.objective_visible_from.setdefault(objective, set()).add(waypoint)
                self.waypoints.add(waypoint)
            # We add any rovers found to the rovers set also
            elif parts[0] == "equipped_for_soil_analysis":
                self.rovers.add(parts[1])
            elif parts[0] == "equipped_for_rock_analysis":
                self.rovers.add(parts[1])
            elif parts[0] == "equipped_for_imaging":
                self.rovers.add(parts[1])

        # Precompute shortest paths for all rovers between waypoints, using BFS
        # to estimate navigation costs later.

        # For each rover, build graph with visible edges filtered by can_traverse
        self.rover_graphs = {}
        for rover in self.rovers:
            graph = {}
            visible = self.visible_map
            can_traverse = self.can_traverse_edges.get(rover, {})
            for w_from in can_traverse:
                # A rover can move from w_from to w_to only if visible(w_from, w_to) and can_traverse(rover, w_from, w_to)
                for w_to in can_traverse[w_from]:
                    # Check if visible relation holds (it must, as per domain precondition)
                    if w_to in visible.get(w_from, set()):
                        graph.setdefault(w_from, set()).add(w_to)
            self.rover_graphs[rover] = graph

    def shortest_path_length(self, graph, start, goals):
        """
        Return the length of the shortest path from start to any waypoint in goals.
        If start is in goals, returns 0.
        Returns a large constant if no path found.
        """
        if start in goals:
            return 0
        from collections import deque

        visited = set()
        queue = deque([(start, 0)])
        while queue:
            current, dist = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            if current in goals:
                return dist
            for neighbor in graph.get(current, []):
                if neighbor not in visited:
                    queue.append((neighbor, dist + 1))
        return 9999  # No path found, large penalty

    def __call__(self, node):
        state = node.state

        # If goal reached, heuristic is zero.
        if self.goals <= state:
            return 0

        # Helper to find facts quickly
        def facts_with_prefix(prefix):
            # Returns list of facts in state starting with prefix string (including '(')
            return [fact for fact in state if fact.startswith(prefix)]

        # Extract current rover positions
        rover_positions = {}
        for fact in facts_with_prefix("(at "):
            parts = get_parts(fact)
            if len(parts) == 3:  # predicate rover waypoint
                pred, rover, waypoint = parts
                if rover in self.rovers:
                    rover_positions[rover] = waypoint

        # Extract which stores are empty or full
        store_empty = set()
        store_full = set()
        for fact in state:
            if match(fact, "empty", "*"):
                store_empty.add(get_parts(fact)[1])
            elif match(fact, "full", "*"):
                store_full.add(get_parts(fact)[1])

        # Extract which have calibrations
        calibrated = set()
        for fact in state:
            if match(fact, "calibrated", "*", "*"):
                _, camera, rover = get_parts(fact)
                calibrated.add((camera, rover))

        # Extract which have soil analysis and rock analysis
        have_soil_analysis = set()
        have_rock_analysis = set()
        for fact in state:
            if match(fact, "have_soil_analysis", "*", "*"):
                _, rover, waypoint = get_parts(fact)
                have_soil_analysis.add((rover, waypoint))
            elif match(fact, "have_rock_analysis", "*", "*"):
                _, rover, waypoint = get_parts(fact)
                have_rock_analysis.add((rover, waypoint))

        # Extract which have images (not necessarily communicated yet)
        have_image = set()
        for fact in state:
            if match(fact, "have_image", "*", "*", "*"):
                _, rover, objective, mode = get_parts(fact)
                have_image.add((rover, objective, mode))

        # Extract communicated data sets to compare with goals
        communicated_soil = set()
        communicated_rock = set()
        communicated_image = set()
        for fact in state:
            if match(fact, "communicated_soil_data", "*"):
                _, waypoint = get_parts(fact)
                communicated_soil.add(waypoint)
            elif match(fact, "communicated_rock_data", "*"):
                _, waypoint = get_parts(fact)
                communicated_rock.add(waypoint)
            elif match(fact, "communicated_image_data", "*", "*"):
                _, objective, mode = get_parts(fact)
                communicated_image.add((objective, mode))

        # Determine goal communication needs from self.goals
        need_communicated_soil = set()
        need_communicated_rock = set()
        need_communicated_image = set()
        for goal in self.goals:
            if match(goal, "communicated_soil_data", "*"):
                _, waypoint = get_parts(goal)
                if waypoint not in communicated_soil:
                    need_communicated_soil.add(waypoint)
            elif match(goal, "communicated_rock_data", "*"):
                _, waypoint = get_parts(goal)
                if waypoint not in communicated_rock:
                    need_communicated_rock.add(waypoint)
            elif match(goal, "communicated_image_data", "*", "*"):
                _, objective, mode = get_parts(goal)
                if (objective, mode) not in communicated_image:
                    need_communicated_image.add((objective, mode))

        # Heuristic score accumulator
        heuristic_cost = 0

        # COST PARAMETERS (estimated average cost per action)
        NAV_COST = 1          # navigating between waypoints per step
        SAMPLE_COST = 1       # cost for sample_soil or sample_rock action
        DROP_COST = 1         # cost for drop action
        CALIBRATE_COST = 1    # cost for calibrate action
        IMAGE_COST = 1        # cost for take_image action
        COMM_COST = 1         # cost for communicate action

        # Approach:
        # For each needed communication (soil, rock, image), compute the minimal cost to:
        # 1) Have the rover acquire the data (sampling or imaging).
        # 2) Have rover travel to a lander-visible waypoint to communicate it.

        # Helper to get rovers that can perform a capability:
        def capable_rovers(cap):
            return self.rover_equipped.get(cap, set())

        # Helper to get stores associated with a rover
        rover_stores = {}
        for store, rover in self.stores.items():
            rover_stores.setdefault(rover, set()).add(store)

        # For communicating data: rovers must be at visible waypoint of lander
        # So collect all lander waypoints for communication goals
        lander_waypoints = self.lander_locations

        # Mapping cameras to their modes and rovers
        cameras_by_rover = {}
        for camera, rover in self.cameras_on_rover.items():
            cameras_by_rover.setdefault(rover, set()).add(camera)

        # Preprocessing: 
        # For each rover, know which cameras it has and what modes they support
        rover_camera_modes = {}
        for camera, rover in self.cameras_on_rover.items():
            modes = self.camera_modes.get(camera, set())
            rover_camera_modes.setdefault(rover, {})[camera] = modes

        # For quick lookup: map cameras to calibration target objective
        camera_to_objective = self.calibration_target

        # HELPER function to estimate cost for a rover to communicate soil/rock data at a waypoint
        def cost_soil_rock(rover, wp, sample_type):
            """
            Estimate cost for rover to acquire (soil/rock) data at wp and communicate it.
            Considers navigation, sampling, drop if needed, communication.
            """
            graph = self.rover_graphs.get(rover, {})
            start_wp = rover_positions.get(rover)
            if start_wp is None:
                return 9999  # No info about rover position --> penalize high

            # Check store states for rover
            stores = rover_stores.get(rover, set())
            empty_stores = [s for s in stores if s in store_empty]
            full_stores = [s for s in stores if s in store_full]

            cost = 0

            # If rover already has sample at wp (have_soil_analysis or have_rock_analysis), no sampling needed
            if sample_type == "soil":
                if (rover, wp) in have_soil_analysis:
                    # Only need communication
                    # Navigate to lander visible waypoint
                    dist_comm = self.shortest_path_length(graph, start_wp, lander_waypoints)
                    cost += dist_comm * NAV_COST
                    cost += COMM_COST
                    return cost
            else:  # rock
                if (rover, wp) in have_rock_analysis:
                    # Only need communication
                    dist_comm = self.shortest_path_length(graph, start_wp, lander_waypoints)
                    cost += dist_comm * NAV_COST
                    cost += COMM_COST
                    return cost

            # Otherwise, sequence:
            # Navigate to sample location if not there
            if start_wp != wp:
                dist_nav_sample = self.shortest_path_length(graph, start_wp, {wp})
                cost += dist_nav_sample * NAV_COST

            # Need empty store to sample: if all stores full, must drop first
            if len(empty_stores) == 0 and len(full_stores) > 0:
                # Navigate to drop location (assume drop at current rover position, so if not at drop location, add cost)
                # Drop action costs DROP_COST
                # Dropping store is possible anywhere (not clearly specified), assume it can be done at current location, so no nav cost
                cost += DROP_COST
                # After drop, store empty, so sampling possible

            # Sample action cost
            cost += SAMPLE_COST

            # Navigate to communication location (visible from lander)
            # May need to navigate from sample wp to lander waypoint
            dist_comm = self.shortest_path_length(graph, wp, lander_waypoints)
            cost += dist_comm * NAV_COST

            # Communicate cost
            cost += COMM_COST

            return cost

        # HELPER function to estimate cost for rover to take and communicate an image of (objective, mode)
        def cost_image(rover, objective, mode):
            """
            Estimate cost for rover to achieve image (objective, mode) and communicate it.
            Considers navigation, calibrations, images, and communication.
            """

            graph = self.rover_graphs.get(rover, {})
            start_wp = rover_positions.get(rover)
            if start_wp is None:
                return 9999

            # Find cameras on this rover compatible with mode and linked to the objective (via calibration_target)
            candidate_cameras = []
            for camera, modes in rover_camera_modes.get(rover, {}).items():
                if mode in modes:
                    # Camera must have calibration target equal to the objective to be used for imaging it
                    if camera_to_objective.get(camera) == objective:
                        candidate_cameras.append(camera)

            if not candidate_cameras:
                return 9999  # No suitable camera

            # If rover already has the image (have_image), no cost
            if (rover, objective, mode) in have_image:
                # Only need communication
                dist_comm = self.shortest_path_length(graph, start_wp, lander_waypoints)
                return dist_comm * NAV_COST + COMM_COST

            # For cost estimation choose the camera with minimal estimated cost
            min_cost = 9999
            # The waypoints from which the objective is visible need to be used for imaging
            visible_waypoints = self.objective_visible_from.get(objective, set())
            if not visible_waypoints:
                return 9999  # Can't take image if no visible waypoints

            # For each candidate camera, calculate cost to:
            # - go to a visible waypoint for imaging
            # - calibrate camera there (if not calibrated)
            # - take image (resets calibration)
            # - navigate to lander visible waypoint
            # - communicate image

            # Find calibrated status of each candidate camera
            calibrated_cams = {cam for cam, r in calibrated if r == rover}

            for camera in candidate_cameras:
                # Try all waypoints visible from objective to take image
                for image_wp in visible_waypoints:
                    cost = 0

                    # Navigation from current to image waypoint
                    if start_wp != image_wp:
                        dist_nav_image = self.shortest_path_length(graph, start_wp, {image_wp})
                        cost += dist_nav_image * NAV_COST

                    # Calibrate if not calibrated now
                    if (camera, rover) not in calibrated_cams:
                        cost += CALIBRATE_COST

                    # Take image action cost
                    cost += IMAGE_COST

                    # Navigation to lander visible waypoint
                    dist_comm = self.shortest_path_length(graph, image_wp, lander_waypoints)
                    cost += dist_comm * NAV_COST

                    # Communicate image cost
                    cost += COMM_COST

                    if cost < min_cost:
                        min_cost = cost

            return min_cost

        # Estimate costs for soil data communications
        for wp in need_communicated_soil:
            best_cost = 9999
            # Choose rover capable for soil analysis to do the job with minimal cost
            for rover in capable_rovers("soil"):
                c = cost_soil_rock(rover, wp, "soil")
                if c < best_cost:
                    best_cost = c
            heuristic_cost += best_cost

        # Estimate costs for rock data communications
        for wp in need_communicated_rock:
            best_cost = 9999
            for rover in capable_rovers("rock"):
                c = cost_soil_rock(rover, wp, "rock")
                if c < best_cost:
                    best_cost = c
            heuristic_cost += best_cost

        # Estimate costs for images communications
        for objective, mode in need_communicated_image:
            best_cost = 9999
            for rover in capable_rovers("imaging"):
                c = cost_image(rover, objective, mode)
                if c < best_cost:
                    best_cost = c
            heuristic_cost += best_cost

        # return heuristic estimate (finite and zero only if goal is reached)
        return heuristic_cost if heuristic_cost < 9999 else 9999
