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
    The heuristic estimates the number of actions required to achieve all goal facts from the current state
    in the rovers domain. It focuses on the main types of goals encountered: communicated data (soil, rock, image),
    and accounts for the rovers' need to navigate to relevant waypoints, collect samples or images, calibrate cameras,
    and communicate data back to the lander.

    Assumptions
    -----------
    - Communication actions are final steps for each type of data; partial data must be collected before communication.
    - Rovers can navigate only along waypoints connected via `can_traverse` and visibility constraints.
    - Cameras require calibration before taking images, and calibration targets and visibility are prerequisites.
    - Each rover has a fixed store to hold samples and must be emptied via drop action before collecting a new sample.
    - The heuristic does not consider concurrent actions or exact path lengths but approximates actions based on minimum distances and required subtasks.
    - The heuristic may overestimate or underestimate but aims to guide greedy best-first search efficiently.

    Heuristic Initialization
    ------------------------
    On initialization, the heuristic extracts from static facts:
    - The set of waypoints and their visibility graph;
    - Connectivity graphs (`can_traverse`) for each rover;
    - Location of the lander(s) and associated waypoints;
    - Mapping of stores to rovers;
    - Cameras on board each rover, their calibration targets and supported modes;
    - Which rovers are equipped for which analyses and imaging;
    - Visibility relations from objectives to waypoints.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. Extract current facts of the state:
       - Locations of rovers.
       - Facts about which data (soil, rock, image) has been obtained but not yet communicated.
       - Calibration states of cameras per rover.
       - Contents of stores (empty/full).
       - Which targets have been communicated.

    2. Identify remaining goals per goal type: communicated soil data, rock data, and image data for objectives and modes.

    3. For each required communicated fact not yet satisfied:
       a) For soil or rock data at a waypoint:
          - If not sampled yet, add estimated cost to navigate rover(s) to waypoint, take sample (including checking if store empty, possible drop needed).
          - Add cost estimated to navigate rover to lander and communicate data.
       b) For image data at an objective and mode:
          - Identify a rover with an appropriate camera that supports the mode.
          - If camera not calibrated, add cost for navigating to calibration target and calibrating.
          - Add cost for navigating to a waypoint visible from the objective, taking image.
          - Add cost to navigate to lander and communicate data.

    4. Estimating navigation costs:
       - Use shortest-path distances (in number of moves) over the rover's visible and can_traverse graph.
       - Minimal distance for rover to reach required waypoints for sampling, calibration, imaging, communication.

    5. Handle store constraints for soil and rock samples:
       - Sampling requires rover store to be empty.
       - If store full, add cost for drop action before sampling.

    6. Sum estimated action counts over all unsatisfied goals across all rovers, minimizing over rovers capable of doing the task to get a lower bound.

    7. Return zero if current state satisfies goals.

    This heuristic efficiently estimates the needed actions by ignoring some interleavings and concurrency but provides guidance focused on navigation, sampling, calibration, imaging, and communication steps required by the rovers domain.
    """

    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static

        # Extract lander waypoints
        self.landers = set()
        self.lander_waypoints = set()
        for fact in self.static:
            if match(fact, "at_lander", "*", "*"):
                _, lander, wp = get_parts(fact)
                self.landers.add(lander)
                self.lander_waypoints.add(wp)

        # Visibility graph (waypoint -> visible waypoints)
        self.visible = {}
        for fact in self.static:
            if match(fact, "visible", "*", "*"):
                _, wp_from, wp_to = get_parts(fact)
                self.visible.setdefault(wp_from, set()).add(wp_to)

        # can_traverse per rover: dict rover -> waypoint -> waypoint neighbors
        self.can_traverse = {}
        for fact in self.static:
            if match(fact, "can_traverse", "*", "*", "*"):
                _, rover, wp_from, wp_to = get_parts(fact)
                if rover not in self.can_traverse:
                    self.can_traverse[rover] = {}
                self.can_traverse[rover].setdefault(wp_from, set()).add(wp_to)

        # Store to rover map
        self.store_of = {}
        for fact in self.static:
            if match(fact, "store_of", "*", "*"):
                _, store, rover = get_parts(fact)
                self.store_of[store] = rover

        # Equipped capabilities
        self.equipped_soil = set()
        self.equipped_rock = set()
        self.equipped_imaging = set()
        for fact in self.static:
            if match(fact, "equipped_for_soil_analysis", "*"):
                _, rover = get_parts(fact)
                self.equipped_soil.add(rover)
            elif match(fact, "equipped_for_rock_analysis", "*"):
                _, rover = get_parts(fact)
                self.equipped_rock.add(rover)
            elif match(fact, "equipped_for_imaging", "*"):
                _, rover = get_parts(fact)
                self.equipped_imaging.add(rover)

        # Cameras on board per rover
        self.cameras_on_board = {}
        for fact in self.static:
            if match(fact, "on_board", "*", "*"):
                _, camera, rover = get_parts(fact)
                self.cameras_on_board.setdefault(rover, set()).add(camera)

        # Which cameras support which mode
        self.camera_supports = {}
        for fact in self.static:
            if match(fact, "supports", "*", "*"):
                _, camera, mode = get_parts(fact)
                self.camera_supports.setdefault(camera, set()).add(mode)

        # Calibration targets for cameras (camera -> objective)
        self.calibration_targets = {}
        for fact in self.static:
            if match(fact, "calibration_target", "*", "*"):
                _, camera, objective = get_parts(fact)
                self.calibration_targets[camera] = objective

        # Visibility from objectives to waypoints: objective -> set of visible waypoints
        self.visible_from = {}
        for fact in self.static:
            if match(fact, "visible_from", "*", "*"):
                _, objective, wp = get_parts(fact)
                self.visible_from.setdefault(objective, set()).add(wp)

        # For navigation cost calculation: precompute shortest paths per rover using BFS over can_traverse graph filtered by visibility
        self._precompute_shortest_paths()

    def _precompute_shortest_paths(self):
        # For each rover, compute shortest path distances between all waypoints
        # where movement is allowed if can_traverse[cur][n] exists and visible[cur] contains n

        self.shortest_paths = {}  # rover -> dict waypoint -> dict waypoint -> distance(int)

        for rover, adj in self.can_traverse.items():
            distances = {}
            for start in adj.keys():
                distances[start] = self._bfs(rover, start)
            self.shortest_paths[rover] = distances

    def _bfs(self, rover, start):
        """Breadth-first search for shortest distances from start for rover."""
        dist = {start: 0}
        queue = [start]
        adj = self.can_traverse[rover]
        visible = self.visible
        while queue:
            current = queue.pop(0)
            if current not in adj:
                continue
            for neighbor in adj[current]:
                # Only move if visible from current to neighbor
                if current in visible and neighbor in visible[current]:
                    if neighbor not in dist:
                        dist[neighbor] = dist[current] + 1
                        queue.append(neighbor)
        return dist

    def _navigation_cost(self, rover, start_wp, goal_wp):
        """Estimate navigation cost (number of navigate actions) for rover from start_wp to goal_wp."""
        # If either waypoint missing, or no path found: return large cost
        if rover not in self.shortest_paths:
            return 1000  # no info about rover's connectivity => large cost
        dists_from_start = self.shortest_paths[rover].get(start_wp, {})
        return dists_from_start.get(goal_wp, 1000)

    def __call__(self, node):
        state = node.state

        # If goal reached, heuristic is zero
        if self.goals <= state:
            return 0

        # Extract rover positions
        rover_positions = {}
        for fact in state:
            if match(fact, "at", "*", "*"):
                _, rover, wp = get_parts(fact)
                rover_positions[rover] = wp

        # Extract stores state: empty or full
        empty_stores = set()
        full_stores = set()
        for fact in state:
            if match(fact, "empty", "*"):
                _, store = get_parts(fact)
                empty_stores.add(store)
            elif match(fact, "full", "*"):
                _, store = get_parts(fact)
                full_stores.add(store)

        # Extract calibrated cameras per rover
        calibrated = set()
        for fact in state:
            if match(fact, "calibrated", "*", "*"):
                _, camera, rover = get_parts(fact)
                calibrated.add((camera, rover))

        # Extract which soil and rock analyses have been done (have_soil_analysis, have_rock_analysis)
        have_soil_analysis = set()
        have_rock_analysis = set()
        for fact in state:
            if match(fact, "have_soil_analysis", "*", "*"):
                _, rover, wp = get_parts(fact)
                have_soil_analysis.add((rover, wp))
            elif match(fact, "have_rock_analysis", "*", "*"):
                _, rover, wp = get_parts(fact)
                have_rock_analysis.add((rover, wp))

        # Extract which images have been taken but possibly not communicated (have_image)
        have_images = set()  # (rover, objective, mode)
        for fact in state:
            if match(fact, "have_image", "*", "*", "*"):
                _, rover, objective, mode = get_parts(fact)
                have_images.add((rover, objective, mode))

        # Extract which data is communicated already
        communicated_soil = set()
        communicated_rock = set()
        communicated_image = set()
        for fact in state:
            if match(fact, "communicated_soil_data", "*"):
                _, wp = get_parts(fact)
                communicated_soil.add(wp)
            elif match(fact, "communicated_rock_data", "*"):
                _, wp = get_parts(fact)
                communicated_rock.add(wp)
            elif match(fact, "communicated_image_data", "*", "*"):
                _, objective, mode = get_parts(fact)
                communicated_image.add((objective, mode))

        # Collect soil sample waypoints that appear in goals:
        # Goals are explicitly fact strings; extract relevant goals 
        # Goal predicates to handle: communicated_soil_data, communicated_rock_data, communicated_image_data

        # Needed soil waypoints to communicate soil data: those goals of type communicated_soil_data not yet communicated
        needed_soil_waypoints = set()
        needed_rock_waypoints = set()
        needed_images = set()  # tuples (objective, mode)

        for goal in self.goals:
            if goal in state:
                continue  # already true
            if match(goal, "communicated_soil_data", "*"):
                _, wp = get_parts(goal)
                needed_soil_waypoints.add(wp)
            elif match(goal, "communicated_rock_data", "*"):
                _, wp = get_parts(goal)
                needed_rock_waypoints.add(wp)
            elif match(goal, "communicated_image_data", "*", "*"):
                _, objective, mode = get_parts(goal)
                needed_images.add((objective, mode))

        # To fulfill communicated_soil_data(wp), soil must be analyzed at wp by a rover and then communicated.
        # Similar for rock data.
        # For images, must have an image taken by calibrated camera supporting mode and communicated.

        # Helper: find store for a rover
        def store_of_rover(r):
            for store, rover in self.store_of.items():
                if rover == r:
                    return store
            return None

        # Helper: possible rovers for tasks
        soil_rovers = self.equipped_soil.copy()
        rock_rovers = self.equipped_rock.copy()
        image_rovers = self.equipped_imaging.copy()

        # Minimal heuristic sum across tasks
        cost = 0

        # Precompute lander waypoints for communication (use any lander waypoint)
        if not self.lander_waypoints:
            # If no lander waypoints known, assign large cost to communication steps
            communication_waypoints = {"UNKNOWN_LANDER_WP"}
        else:
            communication_waypoints = self.lander_waypoints

        # Helper function: minimal navigation cost for a rover to any of targets from start_wp
        def min_navigation_to(rover, start_wp, targets):
            best = 1000
            for tgt in targets:
                dist = self._navigation_cost(rover, start_wp, tgt)
                if dist < best:
                    best = dist
            return best

        # --- HANDLE Soil Data Communication ---
        for wp in needed_soil_waypoints:
            # If soil analysis already done and communicated: skip
            if any((r, wp) in have_soil_analysis for r in soil_rovers) and wp in communicated_soil:
                continue

            # For each rover capable, estimate cost to get soil analysis at wp and communicate it
            best_rover_cost = 1000
            for r in soil_rovers:
                if r not in rover_positions:
                    continue
                pos = rover_positions[r]
                # Navigation to wp
                nav_to_sample = self._navigation_cost(r, pos, wp)

                # Sampling requires at_soil_sample(wp) fact (assumed in instance)
                # Must have store empty or drop (estimate drop cost if needed)
                store = store_of_rover(r)
                need_drop = 0
                if store in full_stores:
                    need_drop = 1  # One drop action

                # Other actions: sample_soil at wp: 1 action, then navigate to lander and communicate: ?
                nav_to_lander = min_navigation_to(r, wp, communication_waypoints)
                communicate = 1  # 1 action

                subtotal = nav_to_sample + need_drop + 1 + nav_to_lander + communicate
                if subtotal < best_rover_cost:
                    best_rover_cost = subtotal
            cost += best_rover_cost

        # --- HANDLE Rock Data Communication ---
        for wp in needed_rock_waypoints:
            if any((r, wp) in have_rock_analysis for r in rock_rovers) and wp in communicated_rock:
                continue

            best_rover_cost = 1000
            for r in rock_rovers:
                if r not in rover_positions:
                    continue
                pos = rover_positions[r]

                nav_to_sample = self._navigation_cost(r, pos, wp)
                store = store_of_rover(r)
                need_drop = 0
                if store in full_stores:
                    need_drop = 1
                nav_to_lander = min_navigation_to(r, wp, communication_waypoints)
                communicate = 1

                subtotal = nav_to_sample + need_drop + 1 + nav_to_lander + communicate
                if subtotal < best_rover_cost:
                    best_rover_cost = subtotal
            cost += best_rover_cost

        # --- HANDLE Image Data Communication ---
        for (objective, mode) in needed_images:
            # If already communicated, continue
            if (objective, mode) in communicated_image:
                continue

            best_rover_cost = 1000

            # We need:
            # - a rover equipped for imaging and having a camera supporting mode,
            # - the camera calibrated on its calibration target objective,
            # - a waypoint from which objective is visible,
            # - navigate rover to calibration waypoint (if needed) and calibrate,
            # - navigate rover to visible waypoint and take image,
            # - navigate rover to lander and communicate.

            # Consider all such rovers and their cameras
            for r in image_rovers:
                if r not in rover_positions:
                    continue
                pos = rover_positions[r]

                # Find all cameras on board of r
                cams = self.cameras_on_board.get(r, set())
                # Keep only cameras supporting the mode
                cams = {c for c in cams if mode in self.camera_supports.get(c, set())}
                if not cams:
                    continue  # no suitable camera

                rover_best_cam_cost = 1000

                for c in cams:
                    # Check calibration status
                    cal_ok = (c, r) in calibrated

                    # Calibration target objective for this camera
                    cal_obj = self.calibration_targets.get(c)
                    if cal_obj is None:
                        # No calibration target? Assume calibration not required or done
                        cal_ok = True

                    # Need to navigate rover to calibration position: waypoint visible from cal_obj
                    cal_nav_cost = 0
                    if not cal_ok and cal_obj:
                        cal_pos_candidates = self.visible_from.get(cal_obj, set())
                        if not cal_pos_candidates:
                            # No calibration viewpoints known => assume high cost
                            cal_nav_cost = 20
                        else:
                            cal_nav_cost = min(self._navigation_cost(r, pos, cal_pos_candidates), 20)

                    # Navigate rover to waypoint visible from objective for taking image
                    image_pos_candidates = self.visible_from.get(objective, set())
                    if not image_pos_candidates:
                        image_nav_cost = 20  # no visibility info, penalty
                    else:
                        # Start position for image capture: is it pos or after calibration? 
                        # If calibration needed, rover moves from pos->calibration_wp, then to image_wp
                        nav_start = pos
                        if not cal_ok and cal_pos_candidates:
                            # Approximate cost: navigate to calibration + calibration + navigate to image viewpoint
                            calibrate_action_cost = 1
                            # Add navigation cost from calibration waypoint to image waypoint as minimal path between any pair (best possible)
                            cal_to_img_cost = 20
                            # Approximate by pick best pair from cal_pos_candidates x image_pos_candidates
                            for cav in cal_pos_candidates:
                                for iv in image_pos_candidates:
                                    dists = self.shortest_paths[r]
                                    dist1 = dists.get(cav, 1000)
                                    dist2 = dist1.get(iv, 1000) if dist1 != 1000 else 1000
                                    if dist2 < cal_to_img_cost:
                                        cal_to_img_cost = dist2
                            image_nav_cost = calibrate_action_cost + cal_to_img_cost
                        else:
                            image_nav_cost = min(self._navigation_cost(r, pos, image_pos_candidates), 20)

                    # Image taking costs 1 action
                    take_image_cost = 1

                    # Navigate to lander to communicate
                    nav_to_lander = min_navigation_to(r, pos, communication_waypoints)

                    communicate = 1

                    if not cal_ok and cal_obj:
                        total_cam_cost = cal_nav_cost + image_nav_cost + take_image_cost + nav_to_lander + communicate
                    else:
                        # Camera calibrated, just navigate to image position, take image, navigate communicate
                        # Assume from current pos go to image viewpoint (if different), else 0
                        total_cam_cost = image_nav_cost + take_image_cost + nav_to_lander + communicate

                    if total_cam_cost < rover_best_cam_cost:
                        rover_best_cam_cost = total_cam_cost

                if rover_best_cam_cost < best_rover_cost:
                    best_rover_cost = rover_best_cam_cost

            cost += best_rover_cost

        # Return the heuristic value as an estimated minimal number of actions
        return cost
