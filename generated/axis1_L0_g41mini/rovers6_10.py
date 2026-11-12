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
    Domain-dependent heuristic for the rovers domain.

    Summary
    -------
    This heuristic estimates the number of actions required to reach a goal state by analyzing
    the difference between the current state and the goal conditions related to soil, rock, 
    and image data communication. It accounts for rover movements, sampling, image capturing, 
    calibration, and communication actions. It aims to balance accuracy and efficiency for 
    guiding greedy best-first search, without guaranteeing admissibility.

    Assumptions
    -----------
    - Each rover can only hold one sample at a time in its store (store is either empty or full).
    - Sampling actions remove the sample location once done.
    - Imaging requires calibrated cameras on rovers and specific modes.
    - Cameras become uncalibrated after a single image is taken (needs recalibration for subsequent images).
    - Communication must be done at the lander waypoint visible from the rover's waypoint.
    - Rovers can navigate only between waypoints connected by `can_traverse` and `visible`.
    - Calibration targets and camera ownership are static and known.
    - The heuristic ignores obstacles like full stores preventing sampling by assuming
      rovers drop samples before further sampling.
    - Communication actions are counted once per data item to be communicated.
    - The heuristic does not consider complex pathfinding costs but counts minimal moves needed between waypoints.

    Heuristic Initialization
    ------------------------
    During initialization, the heuristic extracts:
    - The set of goal facts related to communicated soil, rock, and image data.
    - Static knowledge about:
      * Waypoint connectivity (can_traverse edges and visibility).
      * Landers and their waypoints.
      * Calibration targets per (camera, objective).
      * Cameras onboard rovers and modes supported by cameras.
      * Which rovers are equipped for soil analysis, rock analysis, and imaging.
      * Stores and their association to rovers.

    Additionally, it builds internal graphs to estimate minimal navigation steps between waypoints.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. Identify what data still needs to be communicated to satisfy goal conditions:
       - Soil data at waypoints
       - Rock data at waypoints
       - Image data of (objective, mode)
    2. For each kind of data missing communication, check if the rover already "has" the data (sampled or images taken).
       If not, it must perform sampling, imaging, calibration or both.
    3. Estimate minimum navigation costs:
       - For sampling: rover must navigate to sample waypoint.
       - For imaging: rover must navigate to a waypoint visible from the objective.
       - For calibration: rover must navigate to a waypoint visible from the calibration target objective.
       - For communication: rover must navigate to the lander waypoint.
    4. For rovers holding samples/images, estimate whether they need to drop (via communication) before new sampling/imaging.
       The heuristic assumes rovers drop samples immediately after communication to enable reuse.
    5. Approximate the costs by counting:
       - Navigation actions to move rovers between relevant waypoints.
       - The sampling (soil/rock) or image taking actions.
       - Calibration actions if needed before imaging.
       - Communication actions for each missing transmitted data.
       - Drop actions if needed for store freeing (assumed 1 per full store before next sample).
    6. Sum the minimal costs over all missing goals, assuming optimal assignment of rovers to tasks and ignoring detailed contention.
    7. The heuristic returns 0 if all goals are already achieved.

    The heuristic is designed as a fast, informative estimator for guiding search rather than guaranteeing admissibility.
    """

    def __init__(self, task):
        """
        Extract goals, static connectivity, rover capabilities, cameras, and stores.
        Build graphs for navigation distance heuristics.
        """
        self.goals = task.goals
        self.static = task.static

        # --- Extract landers and their waypoints ---
        self.lander_waypoints = set()
        for fact in self.static:
            if match(fact, "at_lander", "*", "*"):
                _, lander, waypoint = get_parts(fact)
                self.lander_waypoints.add(waypoint)

        # --- Extract visibility graph between waypoints ---
        # visible(w1, w2)
        self.visible = {}
        for fact in self.static:
            if match(fact, "visible", "*", "*"):
                _, w1, w2 = get_parts(fact)
                self.visible.setdefault(w1, set()).add(w2)

        # --- Extract can_traverse graph for each rover ---
        # can_traverse(r, w1, w2)
        self.can_traverse = {}
        for fact in self.static:
            if match(fact, "can_traverse", "*", "*", "*"):
                _, r, w1, w2 = get_parts(fact)
                self.can_traverse.setdefault(r, {}).setdefault(w1, set()).add(w2)

        # --- Extract rovers and their capabilities ---
        self.rovers = set()
        self.equipped_soil = set()
        self.equipped_rock = set()
        self.equipped_imaging = set()
        for fact in self.static:
            if match(fact, "equipped_for_soil_analysis", "*"):
                _, r = get_parts(fact)
                self.equipped_soil.add(r)
                self.rovers.add(r)
            elif match(fact, "equipped_for_rock_analysis", "*"):
                _, r = get_parts(fact)
                self.equipped_rock.add(r)
                self.rovers.add(r)
            elif match(fact, "equipped_for_imaging", "*"):
                _, r = get_parts(fact)
                self.equipped_imaging.add(r)
                self.rovers.add(r)

        # --- Extract stores and map to rovers ---
        self.store_to_rover = {}
        self.rover_to_store = {}
        for fact in self.static:
            if match(fact, "store_of", "*", "*"):
                _, store, rover = get_parts(fact)
                self.store_to_rover[store] = rover
                self.rover_to_store[rover] = store

        # --- Extract cameras onboard rovers ---
        self.camera_onboard = {}  # camera -> rover
        for fact in self.static:
            if match(fact, "on_board", "*", "*"):
                _, camera, rover = get_parts(fact)
                self.camera_onboard[camera] = rover

        # --- Extract calibration targets: camera -> objective ---
        self.calibration_targets = {}
        for fact in self.static:
            if match(fact, "calibration_target", "*", "*"):
                _, camera, objective = get_parts(fact)
                self.calibration_targets[camera] = objective

        # --- Extract modes supported by cameras ---
        self.camera_modes = {}
        for fact in self.static:
            if match(fact, "supports", "*", "*"):
                _, camera, mode = get_parts(fact)
                self.camera_modes.setdefault(camera, set()).add(mode)

        # --- For quick lookup of objects ---
        self.waypoints = {get_parts(fact)[-1] for fact in self.static if fact.startswith("(visible")}
        # Potentially more additive during heuristic computation.

    def _bfs_min_distance(self, rover, start, goals):
        """
        Approximate minimum navigation steps from start to any of goal waypoints for given rover,
        considering can_traverse and visible predicates.
        Returns integer steps or a large number if no path found.

        BFS on can_traverse + visible graph for the rover.
        """
        if start in goals:
            return 0
        visited = set()
        queue = [(start, 0)]
        while queue:
            current, dist = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            if current in goals:
                return dist
            # From current, explore waypoints reachable by rover:
            # must have can_traverse(rover,current, neigh) and visible(current, neigh)
            neighbors = self.can_traverse.get(rover, {}).get(current, set())
            for neigh in neighbors:
                if neigh in visited:
                    continue
                # visible(current, neigh) must hold:
                if neigh in self.visible.get(current, set()):
                    queue.append((neigh, dist + 1))
        # Large cost if no path found:
        return 1000

    def __call__(self, node):
        """
        Return an estimate of the number of actions required to reach goal from the given state.
        0 if in goal state, otherwise heuristic cost estimation.
        """
        state = node.state

        if self.goals <= state:
            return 0  # Already in goal state

        # --- Helper functions to find facts quickly ---

        def facts_for(pattern):
            # Returns list of facts matching given predicate pattern with wildcards '*'
            # Example pattern: ("have_image", "*", "*", "*")
            pred = pattern[0]
            length = len(pattern)
            result = []
            for f in state:
                if not f.startswith("(" + pred):
                    continue
                parts = get_parts(f)
                if len(parts) != length:
                    continue
                if all(p == "*" or p == parts[i] for i, p in enumerate(pattern)):
                    result.append(parts)
            return result

        # --- Rovers current locations ---
        rover_locations = {}
        for fact in state:
            if match(fact, "at", "*", "*"):
                _, r, w = get_parts(fact)
                rover_locations[r] = w
        # For rovers not present, assume unknown or no location

        # --- Stores state (empty/full) ---
        store_status = {}
        for fact in state:
            if match(fact, "empty", "*"):
                _, s = get_parts(fact)
                store_status[s] = "empty"
            elif match(fact, "full", "*"):
                _, s = get_parts(fact)
                store_status[s] = "full"

        # --- Calibrated cameras ---
        calibrated_cameras = set()
        for fact in state:
            if match(fact, "calibrated", "*", "*"):
                _, camera, rover = get_parts(fact)
                calibrated_cameras.add((camera, rover))

        # --- Have soil analysis (r,p) and rock analysis (r,p) ---
        have_soil = set()
        for fact in state:
            if match(fact, "have_soil_analysis", "*", "*"):
                _, r, p = get_parts(fact)
                have_soil.add((r, p))

        have_rock = set()
        for fact in state:
            if match(fact, "have_rock_analysis", "*", "*"):
                _, r, p = get_parts(fact)
                have_rock.add((r, p))

        # --- Have images (r,o,m) ---
        have_images = set()
        for fact in state:
            if match(fact, "have_image", "*", "*", "*"):
                _, r, o, m = get_parts(fact)
                have_images.add((r, o, m))

        # --- Communicated soil/rock/image data (goal predicates) ---
        communicated_soil = set()
        for fact in state:
            if match(fact, "communicated_soil_data", "*"):
                _, p = get_parts(fact)
                communicated_soil.add(p)

        communicated_rock = set()
        for fact in state:
            if match(fact, "communicated_rock_data", "*"):
                _, p = get_parts(fact)
                communicated_rock.add(p)

        communicated_image = set()
        for fact in state:
            if match(fact, "communicated_image_data", "*", "*"):
                _, o, m = get_parts(fact)
                communicated_image.add((o, m))

        # --- Remaining goals to achieve ---
        soil_goals = set()
        rock_goals = set()
        image_goals = set()
        for goal in self.goals:
            pred, *args = get_parts(goal)
            if pred == "communicated_soil_data":
                soil_goals.add(args[0])
            elif pred == "communicated_rock_data":
                rock_goals.add(args[0])
            elif pred == "communicated_image_data":
                image_goals.add((args[0], args[1]))

        # --- Samples still needing communication ---
        soil_to_communicate = soil_goals - communicated_soil
        rock_to_communicate = rock_goals - communicated_rock
        image_to_communicate = image_goals - communicated_image

        # --- Locations of soil and rock samples remaining in the world ---
        soil_sample_locations = set()
        rock_sample_locations = set()
        for fact in state:
            if match(fact, "at_soil_sample", "*"):
                _, w = get_parts(fact)
                soil_sample_locations.add(w)
            if match(fact, "at_rock_sample", "*"):
                _, w = get_parts(fact)
                rock_sample_locations.add(w)

        # --- All objectives whose images still need communication ---
        image_objectives_modes = list(image_to_communicate)

        # --- Lander waypoints --- for communication
        lander_wps = self.lander_waypoints

        # --- Heuristic cost accumulators ---
        total_cost = 0

        # For each soil data to communicate:
        # Need to either have soil analysis from waypoint or sample it first.
        # Then communicate it at lander waypoint.
        for waypoint in soil_to_communicate:
            # Check if any rover already has soil analysis for this waypoint
            having_rover = None
            for r in self.equipped_soil:
                if (r, waypoint) in have_soil:
                    having_rover = r
                    break

            if having_rover is None:
                # Need to sample soil at that waypoint first
                # Find a rover that can do it
                # We pick the rover minimizing navigation cost to waypoint + sampling + communication

                min_cost = 10000
                for r in self.equipped_soil:
                    if r not in rover_locations:
                        continue
                    rover_wp = rover_locations[r]
                    # Estimate navigation cost: rover_wp -> sample waypoint
                    nav_to_sample = self._bfs_min_distance(r, rover_wp, {waypoint})
                    # Sampling cost = 1 action: sample_soil
                    sample_cost = 1
                    # Communication requires rover go to some lander waypoint visible from current rover waypoint (visible?)
                    # Mobile rover locations have to navigate from sample waypoint to lander waypoint visible from sample or wherever
                    nav_to_communicate = 1000
                    for lander_wp in lander_wps:
                        # Must be visible(sample waypoint,pos) to communicate at lander
                        # According to action precondition, (have_soil_analysis) with rover at waypoint x, at lander at y and visible x y must hold
                        # We consider that rover must navigate from sample waypoint to a waypoint visible to lander_wp
                        # So estimate nav from sample_wp to some wp visible from lander_wp that rover can reach
                        # For heuristic, assume rover must navigate sample_wp->waypoint visible from lander_wp OR navigate to lander_wp if visible(sample_wp, lander_wp)
                        # We look if sample_wp visible from lander_wp - if so rover can directly communicate
                        if sample_wp in self.visible.get(lander_wp, set()) or lander_wp in self.visible.get(sample_wp, set()):
                            # Check if rover can navigate sample_wp->lander_wp directly
                            dist = self._bfs_min_distance(r, waypoint, {lander_wp})
                            if dist < nav_to_communicate:
                                nav_to_communicate = dist
                        else:
                            # More complex, approximate by BFS to waypoint visible from lander_wp
                            # We'll try waypoints visible from lander_wp as goals
                            for vis_wp in self.visible.get(lander_wp, set()):
                                dist = self._bfs_min_distance(r, waypoint, {vis_wp})
                                # And then assume 1 step from vis_wp to lander_wp (visible)
                                dist += 1
                                if dist < nav_to_communicate:
                                    nav_to_communicate = dist

                    communication_cost = 1  # communicate_soil_data action
                    cost = nav_to_sample + sample_cost + nav_to_communicate + communication_cost
                    if cost < min_cost:
                        min_cost = cost
                total_cost += min_cost
            else:
                # Rover already has soil analysis; must communicate it
                r = having_rover
                rover_wp = rover_locations.get(r)
                if rover_wp is None:
                    continue  # no info, skip heuristic for this one

                # Navigation cost to lander waypoint for communication
                nav_to_communicate = 1000
                for lander_wp in lander_wps:
                    # check visibility
                    if rover_wp in self.visible.get(lander_wp, set()) or lander_wp in self.visible.get(rover_wp, set()):
                        dist = self._bfs_min_distance(r, rover_wp, {lander_wp})
                        if dist < nav_to_communicate:
                            nav_to_communicate = dist
                    else:
                        # try to get to visible waypoint to lander
                        for vis_wp in self.visible.get(lander_wp, set()):
                            dist = self._bfs_min_distance(r, rover_wp, {vis_wp})
                            dist += 1
                            if dist < nav_to_communicate:
                                nav_to_communicate = dist

                communication_cost = 1
                total_cost += nav_to_communicate + communication_cost

        # --- Repeat similar logic for rock data ---
        for waypoint in rock_to_communicate:
            having_rover = None
            for r in self.equipped_rock:
                if (r, waypoint) in have_rock:
                    having_rover = r
                    break

            if having_rover is None:
                min_cost = 10000
                for r in self.equipped_rock:
                    if r not in rover_locations:
                        continue
                    rover_wp = rover_locations[r]
                    nav_to_sample = self._bfs_min_distance(r, rover_wp, {waypoint})
                    sample_cost = 1
                    nav_to_communicate = 1000
                    for lander_wp in lander_wps:
                        if waypoint in self.visible.get(lander_wp, set()) or lander_wp in self.visible.get(waypoint, set()):
                            dist = self._bfs_min_distance(r, waypoint, {lander_wp})
                            if dist < nav_to_communicate:
                                nav_to_communicate = dist
                        else:
                            for vis_wp in self.visible.get(lander_wp, set()):
                                dist = self._bfs_min_distance(r, waypoint, {vis_wp})
                                dist += 1
                                if dist < nav_to_communicate:
                                    nav_to_communicate = dist
                    communication_cost = 1
                    cost = nav_to_sample + sample_cost + nav_to_communicate + communication_cost
                    if cost < min_cost:
                        min_cost = cost
                total_cost += min_cost
            else:
                r = having_rover
                rover_wp = rover_locations.get(r)
                if rover_wp is None:
                    continue

                nav_to_communicate = 1000
                for lander_wp in lander_wps:
                    if rover_wp in self.visible.get(lander_wp, set()) or lander_wp in self.visible.get(rover_wp, set()):
                        dist = self._bfs_min_distance(r, rover_wp, {lander_wp})
                        if dist < nav_to_communicate:
                            nav_to_communicate = dist
                    else:
                        for vis_wp in self.visible.get(lander_wp, set()):
                            dist = self._bfs_min_distance(r, rover_wp, {vis_wp})
                            dist += 1
                            if dist < nav_to_communicate:
                                nav_to_communicate = dist
                communication_cost = 1
                total_cost += nav_to_communicate + communication_cost

        # --- For each image to communicate ---
        # Check if any rover already has the image (r,o,m)
        # If not: rover must first:
        #   - calibrate camera (if not calibrated)
        #   - take image
        #   - communicate image data at lander waypoint
        #   - calibration becomes invalid after image taken (but heuristic ignores repeated images on same camera here for simplicity)
        for (objective, mode) in image_to_communicate:
            having_rover_camera = None
            for (r, o, m) in have_images:
                if o == objective and m == mode:
                    having_rover_camera = (r, o, m)
                    break

            if having_rover_camera is None:
                # Need to produce image first
                # Find a rover and camera capable (equipped for imaging, camera onboard, supports mode, camera cal target known)
                min_cost = 10000
                for camera, rover in self.camera_onboard.items():
                    if rover not in self.equipped_imaging:
                        continue
                    if mode not in self.camera_modes.get(camera, set()):
                        continue
                    # Find camera calibration target objective
                    cal_target_obj = self.calibration_targets.get(camera)
                    if cal_target_obj is None:
                        continue

                    if rover not in rover_locations:
                        continue
                    rover_wp = rover_locations[rover]

                    # Image must be taken at a waypoint visible from objective
                    # Find waypoints from visible_from(objective, wp)
                    visible_from_objective = set()
                    for fact in self.static:
                        if match(fact, "visible_from", objective, "*"):
                            _, o_, w_ = get_parts(fact)
                            visible_from_objective.add(w_)

                    if not visible_from_objective:
                        continue

                    # Calibration must be done at a waypoint visible_from calibration target
                    visible_from_cal_target = set()
                    for fact in self.static:
                        if match(fact, "visible_from", cal_target_obj, "*"):
                            _, _, w_ = get_parts(fact)
                            visible_from_cal_target.add(w_)

                    if not visible_from_cal_target:
                        continue

                    # Heuristic paths:
                    # 1) Navigate rover from current location to a calibration waypoint
                    dist_to_cal_wp = self._bfs_min_distance(rover, rover_wp, visible_from_cal_target)
                    calibration_cost = 1  # calibrate action

                    # 2) Navigate to imaging waypoint visible from objective
                    # Calibration action does not change rover location, so imaging starts from calibration waypoint,
                    # but heuristic approximates worst case by doing separately:
                    # Dist from calibration waypoint to imaging waypoint:
                    dist_cal_to_imag_wp = 1000
                    for cal_wp in visible_from_cal_target:
                        for imag_wp in visible_from_objective:
                            d = self._bfs_min_distance(rover, cal_wp, {imag_wp})
                            if d < dist_cal_to_imag_wp:
                                dist_cal_to_imag_wp = d

                    take_image_cost = 1  # take_image action

                    # 3) Navigate to lander waypoint for communication
                    # We assume imaging location is rover final location before communication
                    dist_imag_to_lander = 1000
                    for imag_wp in visible_from_objective:
                        for lander_wp in lander_wps:
                            # Need visible(imag_wp, lander_wp)
                            if imag_wp in self.visible.get(lander_wp, set()) or lander_wp in self.visible.get(imag_wp, set()):
                                d = self._bfs_min_distance(rover, imag_wp, {lander_wp})
                                if d < dist_imag_to_lander:
                                    dist_imag_to_lander = d
                            else:
                                for vis_wp in self.visible.get(lander_wp, set()):
                                    d = self._bfs_min_distance(rover, imag_wp, {vis_wp}) + 1
                                    if d < dist_imag_to_lander:
                                        dist_imag_to_lander = d

                    communication_cost = 1  # communicate_image_data

                    cost = (
                        dist_to_cal_wp
                        + calibration_cost
                        + dist_cal_to_imag_wp
                        + take_image_cost
                        + dist_imag_to_lander
                        + communication_cost
                    )
                    if cost < min_cost:
                        min_cost = cost

                total_cost += min_cost
            else:
                # Already have image; must communicate it
                r = having_rover_camera[0]
                rover_wp = rover_locations.get(r)
                if rover_wp is None:
                    continue

                nav_to_communicate = 1000
                for lander_wp in lander_wps:
                    if rover_wp in self.visible.get(lander_wp, set()) or lander_wp in self.visible.get(rover_wp, set()):
                        dist = self._bfs_min_distance(r, rover_wp, {lander_wp})
                        if dist < nav_to_communicate:
                            nav_to_communicate = dist
                    else:
                        for vis_wp in self.visible.get(lander_wp, set()):
                            dist = self._bfs_min_distance(r, rover_wp, {vis_wp})
                            dist += 1
                            if dist < nav_to_communicate:
                                nav_to_communicate = dist
                communication_cost = 1
                total_cost += nav_to_communicate + communication_cost

        return total_cost
