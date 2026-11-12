from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """
    Extract components of a PDDL fact by removing parentheses and splitting the string.
    """
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - fact: The complete fact as a string, e.g., "(at rover1 waypoint2)".
    - args: Pattern elements, with wildcards '*' allowed.
    Return True if the fact matches the pattern, False otherwise.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    Summary
    -------
    Estimates the number of actions required to reach the goal state from a given state.
    The heuristic counts how many goal conditions are still unmet (soil/rock analyses communicated,
    images communicated) and estimates the minimal moves and actions needed by rovers to fulfill them.
    It accounts for:
    - Missing communicated data for soil samples, rock samples, and images per objective and mode.
    - The need to be at or move to sample or objective waypoints.
    - The need to calibrate cameras before imaging.
    - The necessity of dropping full stores before new sampling.
    - The visibility and traversal restrictions based on static facts.
    The heuristic is not admissible but aims to guide a greedy best-first search efficiently.

    Assumptions
    -----------
    - Each required communicated datum (soil, rock, image) requires sampling/imaging, possible camera calibration, possibly dropping before sampling, navigating to samples or objectives, and communicating at a lander position.
    - Each rover can perform one store drop at a time before resampling if needed.
    - Movement cost is approximated by the minimal expected number of navigation actions (1 per waypoint step).
    - Visibility and traversal are respected, but precise pathfinding is not performed; instead, adjacency is used as approximate cost.
    - Communication requires the rover to be at a waypoint visible from the lander.
    - Calibration is required before every image taken with a camera.
    - Multiple modes per objective need individual images communicated.
    - We ignore exact scheduling of rovers but sum requirements over all rovers and goals to estimate the cost.

    Heuristic Initialization
    ------------------------
    From static facts the heuristic extracts:
    - Landers and their waypoints.
    - Which waypoints are lander positions.
    - Which rovers are equipped for soil, rock, and imaging.
    - Stores and which rover owns which store.
    - Camera support modes, calibration targets, and which cameras are onboard which rovers.
    - Visibility relations between waypoints.
    - Waypoints that are soil or rock sample locations.
    - Objectives and from which waypoints they are visible.
    - Connectivity graph of waypoints for each rover built from can_traverse relations.
    This information is organized in dictionaries and sets to support fast heuristic computations.

    Step-By-Step Thinking for Computing Heuristic
    ----------------------------------------------
    For a given state, the heuristic performs the following main steps:

    1. Identify which goal conditions (communicated_soil_data, communicated_rock_data, communicated_image_data)
       are not yet satisfied.

    2. For each required communicated soil or rock data at some waypoint:
       - If not communicated yet:
         * Check if a rover equipped for that analysis exists.
         * Check if a rover's store is empty or full (if full, a drop action is required).
         * Estimate navigation cost from current rover location to sample (soil/rock) waypoint.
         * Include sampling action.
         * Estimate navigation cost to lander waypoint visible from rover's current or sample location.
         * Include communication action.

    3. For each required communicated image data for an objective and mode:
       - If not communicated yet:
         * Check which rover(s) have appropriate cameras which support that mode.
         * For each capable rover:
             - Estimate navigation cost from current location to calibration point (a waypoint visible from the calibration target objective).
             - Include calibrate action.
             - Estimate navigation cost from calibration point to some visible waypoint of the objective.
             - Include take_image action.
             - Estimate navigation cost to lander waypoint visible from rover.
             - Include communicate_image_data action.
         * Use minimal cost over all rovers with suitable cameras.

    4. Sum all estimated costs for all undone goals.

    5. If the goal state is already reached, return 0.

    The heuristic thus combines structural domain knowledge with reachability approximations to estimate remaining effort in a way that balances computation time with search guidance quality.
    """

    def __init__(self, task):
        """
        Extract static knowledge and build data structures to support heuristic computation:
        - Landers and their waypoints.
        - Rover equips and stores.
        - Cameras, modes, calibration targets.
        - Visibility.
        - Sample locations.
        - Objectives visibility.
        - Waypoint traversal graphs per rover.
        """
        self.goals = task.goals
        self.static = task.static

        # Identify landers and their waypoint positions
        self.landers = set()
        self.lander_waypoints = set()
        for fact in self.static:
            if match(fact, "at_lander", "*", "*"):
                parts = get_parts(fact)
                lander = parts[1]
                wp = parts[2]
                self.landers.add(lander)
                self.lander_waypoints.add(wp)

        # Keep sets of waypoints that are soil or rock sample locations
        self.soil_sample_wps = set()
        self.rock_sample_wps = set()
        for fact in self.static:
            if match(fact, "at_soil_sample", "*"):
                self.soil_sample_wps.add(get_parts(fact)[1])
            elif match(fact, "at_rock_sample", "*"):
                self.rock_sample_wps.add(get_parts(fact)[1])

        # Visibility: which waypoints are visible from others
        self.visible_from = dict()  # objective -> set of waypoints visible from it, from static (visible_from objective WP)
        self.visible = dict()  # waypoint -> set of visible waypoints
        for fact in self.static:
            if match(fact, "visible_from", "*", "*"):
                parts = get_parts(fact)
                obj = parts[1]
                wp = parts[2]
                self.visible_from.setdefault(obj, set()).add(wp)
            elif match(fact, "visible", "*", "*"):
                parts = get_parts(fact)
                wp1, wp2 = parts[1], parts[2]
                self.visible.setdefault(wp1, set()).add(wp2)

        # Calibration targets: camera -> objective
        self.calibration_targets = dict()
        for fact in self.static:
            if match(fact, "calibration_target", "*", "*"):
                parts = get_parts(fact)
                cam = parts[1]
                obj = parts[2]
                self.calibration_targets[cam] = obj

        # Camera onboard which rover
        self.camera_onboard = dict()
        for fact in self.static:
            if match(fact, "on_board", "*", "*"):
                parts = get_parts(fact)
                cam = parts[1]
                rover = parts[2]
                self.camera_onboard.setdefault(rover, set()).add(cam)

        # Camera supports which mode
        self.camera_supports = dict()
        for fact in self.static:
            if match(fact, "supports", "*", "*"):
                parts = get_parts(fact)
                cam = parts[1]
                mode = parts[2]
                self.camera_supports.setdefault(cam, set()).add(mode)

        # Which rovers are equipped for which tasks
        self.equipped_for_soil = set()
        self.equipped_for_rock = set()
        self.equipped_for_imaging = set()
        for fact in self.static:
            if match(fact, "equipped_for_soil_analysis", "*"):
                self.equipped_for_soil.add(get_parts(fact)[1])
            elif match(fact, "equipped_for_rock_analysis", "*"):
                self.equipped_for_rock.add(get_parts(fact)[1])
            elif match(fact, "equipped_for_imaging", "*"):
                self.equipped_for_imaging.add(get_parts(fact)[1])

        # Store of which store belongs to which rover
        self.store_of = dict()
        for fact in self.static:
            if match(fact, "store_of", "*", "*"):
                parts = get_parts(fact)
                store = parts[1]
                rover = parts[2]
                self.store_of[store] = rover

        # Build per-rover waypoint traversal graphs from can_traverse
        # used to estimate navigation steps (simple BFS)
        self.can_traverse = dict()  # rover -> adjacency dict waypoint: set(waypoints)
        for fact in self.static:
            if match(fact, "can_traverse", "*", "*", "*"):
                parts = get_parts(fact)
                rover, wp_from, wp_to = parts[1], parts[2], parts[3]
                self.can_traverse.setdefault(rover, dict())
                self.can_traverse[rover].setdefault(wp_from, set()).add(wp_to)

        # Map Landers to their waypoint(s) for communication
        # From at_lander facts we have landers and their waypoint already; 
        # will assume only one waypoint per lander (common in domain)
        # For heuristic, store set of lander waypoints

        # Precompute objective-modes required by goal communicated_image_data facts
        self.goal_soil_comm = set()   # waypoints needing communicated_soil_data
        self.goal_rock_comm = set()   # waypoints needing communicated_rock_data
        self.goal_img_comm = set()    # (objective, mode) pairs needing communicated_image_data

        for goal in self.goals:
            if match(goal, "communicated_soil_data", "*"):
                wp = get_parts(goal)[1]
                self.goal_soil_comm.add(wp)
            elif match(goal, "communicated_rock_data", "*"):
                wp = get_parts(goal)[1]
                self.goal_rock_comm.add(wp)
            elif match(goal, "communicated_image_data", "*", "*"):
                parts = get_parts(goal)
                obj = parts[1]
                mode = parts[2]
                self.goal_img_comm.add((obj, mode))

    def _bfs_distance(self, graph, start, goals):
        """
        Compute the shortest distance in graph from start to any of the goals.
        graph: dict waypoint -> set of neighbor waypoints
        start: start waypoint
        goals: set of goal waypoints
        Return the shortest number of edges (steps).
        If no path found, return a large number (e.g. 1000).
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
        # No path found
        return 1000

    def __call__(self, node):
        """
        Compute heuristic estimate from state node to goal.

        Detailed Steps:
        - If goal state, return 0.
        - Extract rovers' current locations.
        - Extract store states (empty/full).
        - Extract images, soil analyses, rock analyses already have/have communicated.
        - For each remaining goal condition (soil, rock, image):
          * Estimate minimal cost to achieve it including:
            - Navigation (rover to sample/objective/calibration/lander)
            - Sampling/imaging
            - Calibration (for imaging)
            - Dropping full stores if needed before sampling
            - Communicating at lander visible location
        - Sum estimated costs over all unmet goals.
        """

        state = node.state

        # If goal reached, heuristic is zero
        if self.goals <= state:
            return 0

        # Parse rover locations
        rover_locations = dict()  # rover -> waypoint
        for fact in state:
            if match(fact, "at", "*", "*"):
                parts = get_parts(fact)
                rover, wp = parts[1], parts[2]
                rover_locations[rover] = wp

        # Parse store states (empty/full)
        store_empty = set()
        store_full = set()
        for fact in state:
            if match(fact, "empty", "*"):
                store_empty.add(get_parts(fact)[1])
            if match(fact, "full", "*"):
                store_full.add(get_parts(fact)[1])

        # Parse facts indicating which analyses/images are already obtained or communicated
        # have_soil_analysis(rover, wp), have_rock_analysis(rover, wp), have_image(rover, obj, mode)
        # communicated_soil_data(wp), communicated_rock_data(wp), communicated_image_data(obj, mode)

        have_soil = set()    # set of waypoints with soil analysis obtained (per rover - aggregated)
        have_rock = set()
        have_image = set()   # (obj, mode)
        communicated_soil = set()
        communicated_rock = set()
        communicated_image = set()

        # Collect harvested data (have_*)
        for fact in state:
            if match(fact, "have_soil_analysis", "*", "*"):
                parts = get_parts(fact)
                wp = parts[2]
                have_soil.add(wp)
            elif match(fact, "have_rock_analysis", "*", "*"):
                parts = get_parts(fact)
                wp = parts[2]
                have_rock.add(wp)
            elif match(fact, "have_image", "*", "*", "*"):
                parts = get_parts(fact)
                obj = parts[2]
                mode = parts[3]
                have_image.add((obj, mode))

        # Collect communicated data
        for fact in state:
            if match(fact, "communicated_soil_data", "*"):
                wp = get_parts(fact)[1]
                communicated_soil.add(wp)
            elif match(fact, "communicated_rock_data", "*"):
                wp = get_parts(fact)[1]
                communicated_rock.add(wp)
            elif match(fact, "communicated_image_data", "*", "*"):
                parts = get_parts(fact)
                obj = parts[1]
                mode = parts[2]
                communicated_image.add((obj, mode))

        # For each rover, identify which cameras it has onboard and their calibration status
        calibrated_cameras = set()  # (camera, rover)
        for fact in state:
            if match(fact, "calibrated", "*", "*"):
                parts = get_parts(fact)
                cam = parts[1]
                rover = parts[2]
                calibrated_cameras.add((cam, rover))

        # Actions costs: assume uniform cost per action = 1
        ACTION_COST = 1

        total_cost = 0

        # --- Helper function to estimate navigation cost ---
        def nav_cost(rover, start_wp, target_wps):
            """
            Estimate navigation cost for rover from start_wp to any of target_wps.
            Uses BFS on can_traverse graph.
            If rover has no traversal info, use large default cost.
            """
            graph = self.can_traverse.get(rover, {})
            dist = self._bfs_distance(graph, start_wp, target_wps)
            return dist * ACTION_COST if dist < 1000 else 10  # fallback large cost

        # --- Helper function: find landing waypoints visible from rover location for communication ---
        def communication_waypoints(rover_wp):
            """
            Return lander waypoints visible from rover_wp for communication.
            If none, fallback to all lander waypoints.
            """
            visible_landers = set()
            for lw in self.lander_waypoints:
                if rover_wp in self.visible.get(lw, set()) or lw in self.visible.get(rover_wp, set()):
                    visible_landers.add(lw)
            if visible_landers:
                return visible_landers
            else:
                return self.lander_waypoints

        # -- Handle soil data goals --
        for wp in self.goal_soil_comm:
            if wp in communicated_soil:
                # already done
                continue

            # Need to perform soil analysis and communication for wp.
            # Check if soil sample is still present (if not present, can skip sampling)
            sample_still_present = wp in self.soil_sample_wps
            analysis_done = wp in have_soil

            # Find rovers equipped for soil analysis and with stores
            capable_rovers = [r for r in self.equipped_for_soil if r in rover_locations]

            if not capable_rovers:
                # No rover can do soil analysis => return a large cost
                total_cost += 50
                continue

            # For heuristic: pick rover with minimal estimated navigation cost from current location to sample (if sample exists)
            # plus sample + possible drop + communication costs.

            min_cost_for_soil = 1000
            for rover in capable_rovers:
                loc = rover_locations[rover]

                # Check rover's store status
                stores = [s for s, r in self.store_of.items() if r == rover]
                if not stores:
                    continue
                store = stores[0]
                droppable = (store in store_full)
                store_empty_now = (store in store_empty)

                # Navigation cost to sample waypoint if sample still present and analysis not done
                nav_to_sample = 0
                if sample_still_present and not analysis_done:
                    nav_to_sample = nav_cost(rover, loc, {wp})

                # Sampling action cost (if needed)
                sample_cost = ACTION_COST if (sample_still_present and not analysis_done) else 0

                # Drop action cost if store full (if sample needed)
                drop_cost = ACTION_COST if (droppable and sample_still_present and not analysis_done) else 0

                # After sampling, rover must go to a lander waypoint visible from its current location or sample location
                # Approximate rover position after sampling as sample wp (or current loc if no sampling)
                pos_after_sampling = wp if (sample_still_present and not analysis_done) else loc
                comm_wps = communication_waypoints(pos_after_sampling)
                nav_to_lander = nav_cost(rover, pos_after_sampling, comm_wps)

                # One communication action cost
                comm_cost = ACTION_COST

                total_rover_cost = drop_cost + nav_to_sample + sample_cost + nav_to_lander + comm_cost
                if total_rover_cost < min_cost_for_soil:
                    min_cost_for_soil = total_rover_cost

            total_cost += min_cost_for_soil

        # -- Handle rock data goals (similar to soil) --
        for wp in self.goal_rock_comm:
            if wp in communicated_rock:
                continue

            sample_still_present = wp in self.rock_sample_wps
            analysis_done = wp in have_rock

            capable_rovers = [r for r in self.equipped_for_rock if r in rover_locations]

            if not capable_rovers:
                total_cost += 50
                continue

            min_cost_for_rock = 1000
            for rover in capable_rovers:
                loc = rover_locations[rover]

                stores = [s for s, r in self.store_of.items() if r == rover]
                if not stores:
                    continue
                store = stores[0]
                droppable = (store in store_full)
                store_empty_now = (store in store_empty)

                nav_to_sample = 0
                if sample_still_present and not analysis_done:
                    nav_to_sample = nav_cost(rover, loc, {wp})

                sample_cost = ACTION_COST if (sample_still_present and not analysis_done) else 0
                drop_cost = ACTION_COST if (droppable and sample_still_present and not analysis_done) else 0

                pos_after_sampling = wp if (sample_still_present and not analysis_done) else loc
                comm_wps = communication_waypoints(pos_after_sampling)
                nav_to_lander = nav_cost(rover, pos_after_sampling, comm_wps)
                comm_cost = ACTION_COST

                total_rover_cost = drop_cost + nav_to_sample + sample_cost + nav_to_lander + comm_cost
                if total_rover_cost < min_cost_for_rock:
                    min_cost_for_rock = total_rover_cost

            total_cost += min_cost_for_rock

        # -- Handle image data goals --
        for (obj, mode) in self.goal_img_comm:
            if (obj, mode) in communicated_image:
                continue

            # Find rovers equipped for imaging
            capable_rovers = [r for r in self.equipped_for_imaging if r in rover_locations]

            if not capable_rovers:
                total_cost += 50
                continue

            min_cost_img = 1000
            for rover in capable_rovers:
                loc = rover_locations[rover]

                # Find cameras onboard this rover that support the mode and have a calibration target
                cams = [cam for cam in self.camera_onboard.get(rover, set())
                        if mode in self.camera_supports.get(cam, set())]

                # Filter cameras with calibration target
                cams = [cam for cam in cams if cam in self.calibration_targets]

                if not cams:
                    continue

                # For each camera pick strategy: calibrate, move to objective-visible waypoint, take image, move to lander, communicate

                # Calibrate at waypoint visible from calibration target objective
                # Calibration target objective for this camera is self.calibration_targets[cam]

                # However we need to pick the camera that yields minimal cost - try all cameras

                for cam in cams:
                    cam_obj = self.calibration_targets[cam]
                    # Calibration target objective could be different from target obj, but must be visible from somewhere
                    calib_obj = cam_obj

                    # Waypoints visible from calibration target
                    calib_wps = self.visible_from.get(calib_obj, set())
                    if not calib_wps:
                        continue  # no visible calibration wp found, skip cam

                    # Waypoints visible from image objective
                    img_wps = self.visible_from.get(obj, set())
                    if not img_wps:
                        continue  # no visible image wp found, skip cam

                    # Navigation to calibration waypoint
                    cost_nav_to_calib = nav_cost(rover, loc, calib_wps)

                    # Calibration action
                    cost_calibrate = ACTION_COST

                    # Navigation from calibration waypoint to image waypoint
                    cost_nav_to_img = self._bfs_distance_on_graph(self.can_traverse.get(rover, {}),
                                                                 calib_wps, img_wps) * ACTION_COST
                    # If no path found, fallback to summed distances
                    if cost_nav_to_img == -1:
                        cost_nav_to_img = 10

                    # Take image action
                    cost_take_image = ACTION_COST

                    # Navigation from image waypoint to lander visible waypoints
                    # Approximate rover position after imaging at one of img_wps
                    comm_wps = communication_waypoints(next(iter(img_wps)))
                    cost_nav_to_lander = nav_cost(rover, next(iter(img_wps)), comm_wps)

                    # Communicate image data action
                    cost_comm_img = ACTION_COST

                    # If camera calibrated already by the rover -> we can skip calibration (calibrate fact)
                    if (cam, rover) in calibrated_cameras:
                        cost_calibrate = 0

                    total_cam_cost = cost_nav_to_calib + cost_calibrate + cost_nav_to_img + cost_take_image + cost_nav_to_lander + cost_comm_img

                    if total_cam_cost < min_cost_img:
                        min_cost_img = total_cam_cost

            total_cost += min_cost_img

        return total_cost

    def _bfs_distance_on_graph(self, graph, starts, goals):
        """
        BFS shortest distance from any start in 'starts' to any goal in 'goals' on graph.
        graph: dict waypoint -> set neighbors
        starts: iterable of start waypoints
        goals: iterable of goal waypoints
        Returns shortest distance or -1 if no path.
        """
        from collections import deque

        visited = set()
        queue = deque()

        for start in starts:
            queue.append((start, 0))
            visited.add(start)

        goal_set = set(goals)
        while queue:
            current, dist = queue.popleft()
            if current in goal_set:
                return dist
            for nbr in graph.get(current, []):
                if nbr not in visited:
                    visited.add(nbr)
                    queue.append((nbr, dist + 1))
        return -1
