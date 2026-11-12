from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(have_image rover1 objective1 colour)".
    - `args`: The expected pattern (wildcards '*' allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the rovers domain.

    Summary
    -------
    Estimates the number of actions needed to reach the goal from a given state by counting:

    - The required image data communications not yet done.
    - The required soil and rock data communications not yet done.
    - The calibration, imaging, sample collection, and communication steps for each rover as needed.
    - The navigation steps roughly estimated by shortest traversals between waypoints, based on static facts.

    Assumptions
    -----------
    - Each goal fact must be satisfied independently.
    - Navigation steps are roughly estimated by counting visible and can_traverse edges,
      using shortest paths between waypoints.
    - Communication requires the rover to be at a visible location from the lander.
    - Rovers have only one store each for samples, requiring drop actions before re-sampling.
    - Calibrations are lost after taking an image and must be redone if another image is needed with the same camera.
    - Sampling a soil or rock sample removes it from the map; samples can only be taken once.
    - Calibration targets and camera modes influence imaging possibility.
    - The heuristic is not guaranteed to be admissible but aims to be informative and efficient.

    Heuristic Initialization
    ------------------------
    - Extract static facts about waypoints, visibility, traversability, rovers, cameras, objectives,
      calibration targets, stores, and their relations.
    - Build adjacency maps for waypoints based on 'can_traverse'.
    - Identify which rovers are equipped for soil analysis, rock analysis, and imaging.
    - Identify camera support modes and calibration targets.
    - Precompute shortest paths between all waypoints using a BFS for navigation estimates.
    - Extract lander's waypoint for communication checks.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. Extract the current facts from the state: positions of rovers, camera calibrations,
       which samples have been taken, which data/images have been communicated, stores status.

    2. Identify which goal facts are still unsatisfied (e.g., communicated image data, soil data, rock data).

    3. For each unsatisfied goal fact:
       - Estimate the minimal cost to fulfill it by computing:
         a) The actions to get the required data/sample (sampling if needed).
         b) For images:
            - Calibrate the correct camera if needed.
            - Take image action.
         c) For soil/rock analysis:
            - Sample soil/rock if not obtained.
         d) Drop full stores if needed before sampling.
         e) Communication:
            - Rovers must be moved so that they are at a waypoint visible to the lander.
            - Count navigation steps needed to get rover to that waypoint.
            - Count communication action.

    4. For sample collection goals:
       - Ensure rover has sample locally or can sample it (requires navigation to sample location).
       - Account for store empty/full status and drop if necessary.

    5. Sum up estimated actions (navigation, sampling, calibration, imaging, dropping, communication)
       for all remaining goal facts, aggregating costs efficiently by rover and avoiding double-counting.

    6. Return the total estimate as heuristic value; zero if goal already reached.

    This heuristic aims to balance accuracy and computational efficiency heuristically guiding search.
    """

    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static

        # Parse static facts into structures for quick access
        self.waypoints = set()
        self.landers = {}
        self.rovers = set()
        self.cameras = set()
        self.modes = set()
        self.objectives = set()
        self.stores = {}  # store -> rover
        self.equipped_soil = set()
        self.equipped_rock = set()
        self.equipped_imaging = set()
        self.visibles = {}  # waypoint -> set of visible waypoints
        self.can_traverse = {}  # rover -> dict from waypoint to set of waypoints reachable
        self.calibration_targets = {}  # camera -> objective
        self.on_board_camera = {}  # camera -> rover
        self.supports = {}  # camera -> set of modes
        self.visible_from = {}  # objective -> set of waypoints visible from
        self.at_lander = {}  # lander -> waypoint

        # Temporary sets to help build above
        waypoint_candidates = set()
        rover_candidates = set()
        store_candidates = set()
        camera_candidates = set()
        mode_candidates = set()
        objective_candidates = set()
        lander_candidates = set()

        # First extract typed objects from static facts by pattern matching
        # The domain does not provide explicit type facts, so we infer from predicates.
        # Will use facts patterns:
        # (at_lander ?lander ?waypoint)
        # (equipped_for_soil_analysis ?rover)
        # (equipped_for_rock_analysis ?rover)
        # (equipped_for_imaging ?rover)
        # (store_of ?store ?rover)
        # (visible ?w1 ?w2)
        # (can_traverse ?rover ?w1 ?w2)
        # (calibration_target ?camera ?objective)
        # (on_board ?camera ?rover)
        # (supports ?camera ?mode)
        # (visible_from ?objective ?waypoint)

        # Scan static facts and build data structures
        for fact in self.static:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            args = parts[1:]
            if pred == "at_lander" and len(args) == 2:
                lander, waypoint = args
                self.landers[lander] = waypoint
                lander_candidates.add(lander)
                waypoint_candidates.add(waypoint)
            elif pred == "equipped_for_soil_analysis" and len(args) == 1:
                self.equipped_soil.add(args[0])
                rover_candidates.add(args[0])
            elif pred == "equipped_for_rock_analysis" and len(args) == 1:
                self.equipped_rock.add(args[0])
                rover_candidates.add(args[0])
            elif pred == "equipped_for_imaging" and len(args) == 1:
                self.equipped_imaging.add(args[0])
                rover_candidates.add(args[0])
            elif pred == "store_of" and len(args) == 2:
                store, rover = args
                self.stores[store] = rover
                store_candidates.add(store)
                rover_candidates.add(rover)
            elif pred == "visible" and len(args) == 2:
                w1, w2 = args
                waypoint_candidates.add(w1)
                waypoint_candidates.add(w2)
                self.visibles.setdefault(w1, set()).add(w2)
            elif pred == "can_traverse" and len(args) == 3:
                rover, w1, w2 = args
                rover_candidates.add(rover)
                waypoint_candidates.add(w1)
                waypoint_candidates.add(w2)
                self.can_traverse.setdefault(rover, {}).setdefault(w1, set()).add(w2)
            elif pred == "calibration_target" and len(args) == 2:
                camera, objective = args
                self.calibration_targets[camera] = objective
                camera_candidates.add(camera)
                objective_candidates.add(objective)
            elif pred == "on_board" and len(args) == 2:
                camera, rover = args
                self.on_board_camera[camera] = rover
                camera_candidates.add(camera)
                rover_candidates.add(rover)
            elif pred == "supports" and len(args) == 2:
                camera, mode = args
                self.supports.setdefault(camera, set()).add(mode)
                camera_candidates.add(camera)
                mode_candidates.add(mode)
            elif pred == "visible_from" and len(args) == 2:
                objective, waypoint = args
                objective_candidates.add(objective)
                waypoint_candidates.add(waypoint)
                self.visible_from.setdefault(objective, set()).add(waypoint)

        self.waypoints = waypoint_candidates
        self.rovers = rover_candidates
        self.cameras = camera_candidates
        self.modes = mode_candidates
        self.objectives = objective_candidates
        self.landers = lander_candidates.union(set(self.landers.keys()))  # lander names encountered

        # Build shortest path distance maps for each rover over its waypoint graph (can_traverse edges).
        # We will precompute and store shortest-path lengths for each rover between all pairs of waypoints.
        # If a rover has no can_traverse info, shortest path defaults to None.

        # For efficiency, we store a dict: rover -> dict (src waypoint -> dict dst waypoint -> length)
        from collections import deque

        self.shortest_paths = {}

        for rover in self.rovers:
            adj = self.can_traverse.get(rover, {})
            # BFS for each waypoint of this rover
            sp = {}
            for src in self.waypoints:
                if src not in adj:
                    continue
                dist = {src: 0}
                queue = deque([src])
                while queue:
                    current = queue.popleft()
                    current_dist = dist[current]
                    for neigh in adj.get(current, []):
                        if neigh not in dist:
                            dist[neigh] = current_dist + 1
                            queue.append(neigh)
                sp[src] = dist
            self.shortest_paths[rover] = sp

        # Identify lander's waypoint for communication purposes.
        # If multiple landers exist, we consider the first arbitrarily.
        self.lander_waypoint = None
        if self.landers:
            # Pick one lander and its waypoint
            _, wpt = next(iter(self.landers.items()))
            self.lander_waypoint = wpt

    def __call__(self, node):
        state = node.state

        # Quick check if goal reached: heuristic is 0 if goal satisfied
        if self.goals <= state:
            return 0

        # Extract relevant information from the state

        # Current rover locations: rover -> waypoint
        rover_positions = {}
        for fact in state:
            if fact.startswith("(at "):
                parts = get_parts(fact)
                if len(parts) == 3 and parts[1] in self.rovers and parts[2] in self.waypoints:
                    rover_positions[parts[1]] = parts[2]

        # Stores empty/full states: store -> "empty"/"full"
        store_empty = set()
        store_full = set()
        for fact in state:
            if fact.startswith("(empty "):
                st = get_parts(fact)[1]
                store_empty.add(st)
            elif fact.startswith("(full "):
                st = get_parts(fact)[1]
                store_full.add(st)

        # Camera calibrations: (calibrated ?camera ?rover)
        calibrated = set()  # tuples (camera, rover)
        for fact in state:
            if fact.startswith("(calibrated "):
                parts = get_parts(fact)
                if len(parts) == 3:
                    calibrated.add((parts[1], parts[2]))

        # Analysis and image facts:
        # soil samples collected: (have_soil_analysis ?rover ?waypoint)
        soil_analysis = set()
        for fact in state:
            if fact.startswith("(have_soil_analysis "):
                parts = get_parts(fact)
                if len(parts) == 3:
                    soil_analysis.add((parts[1], parts[2]))

        # rock samples collected: (have_rock_analysis ?rover ?waypoint)
        rock_analysis = set()
        for fact in state:
            if fact.startswith("(have_rock_analysis "):
                parts = get_parts(fact)
                if len(parts) == 3:
                    rock_analysis.add((parts[1], parts[2]))

        # images taken but not necessarily communicated: (have_image ?rover ?objective ?mode)
        images_taken = set()
        for fact in state:
            if fact.startswith("(have_image "):
                parts = get_parts(fact)
                if len(parts) == 4:
                    images_taken.add((parts[1], parts[2], parts[3]))

        # communicated soil data: (communicated_soil_data ?waypoint)
        communicated_soil = set()
        for fact in state:
            if fact.startswith("(communicated_soil_data "):
                parts = get_parts(fact)
                if len(parts) == 2:
                    communicated_soil.add(parts[1])

        # communicated rock data: (communicated_rock_data ?waypoint)
        communicated_rock = set()
        for fact in state:
            if fact.startswith("(communicated_rock_data "):
                parts = get_parts(fact)
                if len(parts) == 2:
                    communicated_rock.add(parts[1])

        # communicated image data: (communicated_image_data ?objective ?mode)
        communicated_images = set()
        for fact in state:
            if fact.startswith("(communicated_image_data "):
                parts = get_parts(fact)
                if len(parts) == 3:
                    communicated_images.add((parts[1], parts[2]))

        # Current locations of soil/rock samples (if still present, i.e., if (at_soil_sample ?w) or (at_rock_sample ?w))
        soil_samples = set()
        rock_samples = set()
        for fact in state:
            if fact.startswith("(at_soil_sample "):
                wpt = get_parts(fact)[1]
                soil_samples.add(wpt)
            elif fact.startswith("(at_rock_sample "):
                wpt = get_parts(fact)[1]
                rock_samples.add(wpt)

        # For navigation estimates, utility function for shortest path length between two waypoints for a rover
        def nav_cost(rover, start, goal):
            if start == goal:
                return 0
            sp = self.shortest_paths.get(rover)
            if sp is None:
                return 1  # Fallback if no data
            dist_map = sp.get(start)
            if dist_map is None:
                return 1
            return dist_map.get(goal, 1)  # fallback 1 if no path found

        # Find which rovers have stores full (cannot sample new until drop)
        # We'll map rover -> stores and their empty/full state
        rover_stores = {}
        for store, rover in self.stores.items():
            rover_stores.setdefault(rover, []).append(store)

        rover_store_status = {}
        for rover in self.rovers:
            empty_stores = [s for s in rover_stores.get(rover, []) if s in store_empty]
            full_stores = [s for s in rover_stores.get(rover, []) if s in store_full]
            rover_store_status[rover] = (empty_stores, full_stores)

        # Identify goals that are not yet satisfied from self.goals but missing in current state

        # Separate goal facts by type for easier handling
        goal_comm_soil = set()
        goal_comm_rock = set()
        goal_comm_image = set()
        for goal_fact in self.goals:
            if goal_fact in state:
                continue
            parts = get_parts(goal_fact)
            if not parts:
                continue
            pred = parts[0]
            args = parts[1:]
            if pred == "communicated_soil_data" and len(args) == 1:
                goal_comm_soil.add(args[0])
            elif pred == "communicated_rock_data" and len(args) == 1:
                goal_comm_rock.add(args[0])
            elif pred == "communicated_image_data" and len(args) == 2:
                goal_comm_image.add((args[0], args[1]))

        # Heuristic sum initialization
        total_cost = 0

        # Helper: minimal nav cost for rover to reach waypoint visible from lander for communication
        def min_comm_nav_cost(rover, from_wp):
            # To communicate, rover must be at waypoint visible to lander waypoint
            # We find a waypoint visible from lander waypoint and pick shortest route rover -> those waypoints.
            lander_wp = self.lander_waypoint
            if lander_wp is None:
                # No known lander waypoint, cost minimal 0 or fallback 1
                return 1
            candidate_wps = self.visibles.get(lander_wp, set())
            if not candidate_wps:
                # If none visible from lander waypoint, maybe rover can communicate directly there
                candidate_wps = {lander_wp}
            min_dist = None
            for wpt in candidate_wps:
                dist = nav_cost(rover, from_wp, wpt)
                if min_dist is None or dist < min_dist:
                    min_dist = dist
            return min_dist if min_dist is not None else 1

        # -------------- Handle Soil Data Goals ----------------
        # For each soil communication goal waypoint:
        for soil_wp in goal_comm_soil:
            # We must have communicated soil data at soil_wp

            # If soil data not sampled yet at soil_wp, we need a rover to sample it first

            # Find rovers that are equipped for soil analysis
            # We'll estimate cost for the cheapest rover to fulfill this goal:
            best_cost = None
            for rover in self.equipped_soil:
                # Conditions to sample soil:
                # Rover must be at soil_wp or navigate there.
                # Rover's store must be empty (else drop needed)
                # The soil sample must still be at soil_wp (i.e. not sampled yet)
                # Sample_soil action takes 1 action.

                empty_stores, full_stores = rover_store_status.get(rover, ([], []))
                # We'll favor rover with empty store ideally

                # Navigation cost from rover current position to soil_wp
                rpos = rover_positions.get(rover)
                if rpos is None:
                    continue

                navto_soil = nav_cost(rover, rpos, soil_wp)

                # Drop cost if store full
                drop_cost_action = 0
                if full_stores:
                    drop_cost_action = 1  # one drop action

                # Sampling soil cost is 1 action
                sample_cost = 1

                # After sampling, rover must navigate to communication point and communicate

                comm_nav = min_comm_nav_cost(rover, soil_wp)
                comm_action = 1

                # Total cost for this rover:
                cost = drop_cost_action + navto_soil + sample_cost + comm_nav + comm_action

                # If soil sample already collected by some rover (in have_soil_analysis), omit sampling cost
                sampled = any(soil_wp == s for r, s in soil_analysis)
                if sampled:
                    # only navigation+communication costs needed (rover must have data to communicate)
                    cost = comm_nav + comm_action

                if best_cost is None or cost < best_cost:
                    best_cost = cost

            if best_cost is not None:
                total_cost += best_cost
            else:
                # No rover available to sample soil => set arbitrary positive cost
                total_cost += 5

        # ------------ Handle Rock Data Goals ----------------
        for rock_wp in goal_comm_rock:
            best_cost = None
            for rover in self.equipped_rock:
                rpos = rover_positions.get(rover)
                if rpos is None:
                    continue
                empty_stores, full_stores = rover_store_status.get(rover, ([], []))
                navto_rock = nav_cost(rover, rpos, rock_wp)
                drop_cost_action = 1 if full_stores else 0
                sample_cost = 1
                comm_nav = min_comm_nav_cost(rover, rock_wp)
                comm_action = 1

                cost = drop_cost_action + navto_rock + sample_cost + comm_nav + comm_action

                sampled = any(rock_wp == s for r, s in rock_analysis)
                if sampled:
                    # if already sampled, only nav + communication
                    cost = comm_nav + comm_action

                if best_cost is None or cost < best_cost:
                    best_cost = cost
            if best_cost is not None:
                total_cost += best_cost
            else:
                total_cost += 5

        # -------------- Handle Image Communication Goals ---------------
        # For each communicated_image_data ?objective ?mode goal not satisfied:
        for objective, mode in goal_comm_image:
            best_cost = None

            # Find rovers equipped for imaging and cameras supporting this mode and calibrated
            for rover in self.equipped_imaging:
                rpos = rover_positions.get(rover)
                if rpos is None:
                    continue
                # Find cameras on board rover that support mode and calibration target is the objective
                cams = [c for c, r in self.on_board_camera.items() if r == rover]
                relevant_cameras = []
                for c in cams:
                    if mode in self.supports.get(c, set()) and self.calibration_targets.get(c) == objective:
                        relevant_cameras.append(c)
                if not relevant_cameras:
                    # no suitable camera on this rover for this mode & objective
                    continue

                # For each relevant camera, estimate cost and take minimal
                for camera in relevant_cameras:
                    # Before taking image, camera must be calibrated at waypoint visible from objective (visible_from)
                    visible_wps = self.visible_from.get(objective, set())
                    if not visible_wps:
                        # no visible waypoints for objective => minimal cost assume 1
                        visible_wps = self.waypoints
                    # For calibration, rover must be at visible waypoint from objective (different from image location)
                    # For taking image, rover must be at a waypoint visible from objective too

                    # Get current calibrated state of this camera-rover pair
                    calibrated_now = (camera, rover) in calibrated

                    # Navigation for calibration (if needed)
                    # Calibration waypoint: must be visible_from objective
                    # Among visible_wps pick the closest to rover
                    best_calib_nav = None
                    for wpt in visible_wps:
                        navc = nav_cost(rover, rpos, wpt)
                        if best_calib_nav is None or navc < best_calib_nav:
                            best_calib_nav = navc

                    # Navigation for taking image
                    # The take_image precondition requires rover at waypoint visible_from objective as well (could be same)
                    best_img_nav = None
                    for wpt in visible_wps:
                        # If calibrated and calibrated camera not lost already, we can use calibration without new calibration
                        # Assume calibration must be redone if take_image previously done for this camera (calibrated fact removed after image)
                        navt = nav_cost(rover, rpos, wpt)
                        if best_img_nav is None or navt < best_img_nav:
                            best_img_nav = navt

                    # Cost components:
                    # If not calibrated: calibration action (1) + nav to calib location + nav to image location + image action (1)
                    # If calibrated: nav to image location + image action (1)
                    # Note calibration effect is lost after taking image => if multiple images needed with same camera, multiple calibrations.
                    # For heuristic we consider only this image.

                    # For simplicity estimate navigation cost as max(nav to calib, nav to image):
                    # But in real, calibration and image at potentially different locations with navigation cost between,
                    # so sum nav to calib + nav from calib to image if different.

                    # We approximate:
                    nav_move_cost = best_calib_nav + 1 + best_img_nav if not calibrated_now else best_img_nav
                    calibrate_cost = 1 if not calibrated_now else 0
                    image_cost = 1

                    # Communication must happen at a waypoint visible from lander, rover must navigate there
                    # After take_image, rover must communicate the data

                    # Estimate nav cost from image location (closest visible_from objective) to communication location (visible from lander waypoint)
                    comm_nav = min_comm_nav_cost(rover, rpos)
                    comm_action = 1

                    total_estimated_cost = calibrate_cost + nav_move_cost + image_cost + comm_nav + comm_action

                    # If rover already have_image for objective-mode, only communication needed
                    if (rover, objective, mode) in images_taken:
                        total_estimated_cost = comm_nav + comm_action

                    if best_cost is None or total_estimated_cost < best_cost:
                        best_cost = total_estimated_cost

            if best_cost is not None:
                total_cost += best_cost
            else:
                # No rover/camera combo found for this image goal
                total_cost += 10

        return total_cost
