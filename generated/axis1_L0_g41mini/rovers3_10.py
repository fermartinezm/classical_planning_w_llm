from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract components of a PDDL fact by removing parentheses and splitting."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern with wildcards (*).

    :param fact: The PDDL fact string, e.g. "(at rover1 waypoint2)"
    :param args: The pattern parts to match against, e.g. ("at", "*", "waypoint2")
    :return: True if fact matches pattern, else False
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    Summary:
    ----------
    This heuristic estimates the number of actions required to achieve all goal facts in a given rovers state.
    It accounts for movement (navigation) costs of rovers, calibration requirements of cameras before imaging,
    sampling soil and rock data (if needed), taking images with appropriate modes, and communicating data
    to the lander. The heuristic is designed to be efficiently computable and to guide greedy best-first search effectively,
    while not necessarily being admissible.

    Assumptions:
    ------------
    - Rovers can only navigate between waypoints they can traverse and that are visible.
    - For imaging goals, cameras must be calibrated beforehand at a visible from location of calibration target.
    - Sampling requires the rover to be on a waypoint with a soil or rock sample and have an empty store belonging to it.
    - Communication requires presence at the lander's location or a waypoint visible from it.
    - Each rover is equipped for certain data collection tasks; only applicable goals are considered.
    - Calibration may be lost after imaging (per domain actions), so re-calibration might be needed for multiple images.
    - The heuristic does not perfectly estimate exact minimal paths but uses relaxed shortest-path estimates and counts
      sampling/calibration/imaging/communication steps heuristically.

    Heuristic Initialization:
    -------------------------
    - Extract all goal facts that need to hold for the problem to be solved.
    - Extract static facts to build:
      * Mapping of rovers to their stores.
      * Visibility graph between waypoints (for navigation cost estimation).
      * Camera capabilities (which rover they're on, which modes they support, calibration targets).
      * Locations of landers and which waypoint they are at.
      * Which waypoints have soil or rock samples.
      * Which waypoints are visible from which objectives.
      * Which rovers are equipped for soil analysis, rock analysis, and imaging.
    - Precompute shortest-path distances (number of navigation steps) between waypoints using static `can_traverse` and `visible` facts combined (since navigation precondition requires both).

    Step-By-Step Thinking for Computing Heuristic:
    -----------------------------------------------
    1. If the state already satisfies all goal facts, return 0.

    2. For each type of goal fact (communicated_soil_data, communicated_rock_data, communicated_image_data):
       - Extract the required waypoints/objective-mode pairs from the goals that are still not satisfied.

    3. Estimate movement costs:
       - For each rover, note its current location.
       - For each data collection (soil/rock sampling, imaging), estimate:
         a) Minimal distance to reach the sample/objective's visible waypoint.
         b) Additional actions needed for sampling or imaging (e.g., sample action, calibrate and take_image).
       - For communicating the data:
         a) Estimate minimal distance to the lander's waypoint from the rover's final waypoint of data collection.
         b) The communicate action(s) per piece of data.

    4. For image goals, factor in calibration:
       - Cameras may need calibrating at a calibration target objective visible from the rover's current or some waypoint.

    5. Sum these partial costs for all rovers and all required goals.

    6. Return the minimum aggregate heuristic estimated cost for all rovers to collectively fulfill the outstanding goals.

    This heuristic uses shortest-path distances and relaxed assumptions (e.g., ignoring full store/drop logic) to stay efficient.

    """

    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static

        # Helper: parse static facts
        self.rovers = set()
        self.stores_of_rover = dict()  # store -> rover
        self.store_empty_facts = set()
        self.rover_equipped_soil = set()
        self.rover_equipped_rock = set()
        self.rover_equipped_imaging = set()

        self.at_lander = dict()  # lander -> waypoint
        self.landers = set()

        self.can_traverse = dict()  # (rover, from_wp) -> set(to_wp)
        self.visible = dict()  # from_wp -> set(to_wp)
        self.visible_from = dict()  # objective -> set(waypoint)
        self.calibration_targets = dict()  # camera -> objective

        self.on_board_camera_of_rover = dict()  # rover -> set of cameras
        self.camera_supports = dict()  # camera -> set(mode)
        self.cameras_of_rover = dict()  # rover -> set of cameras
        self.calibrated_cameras = dict()  # rover -> set of calibrated cameras (static info doesn't have calibrated info, state may)

        self.at_rock_sample = set()  # waypoints
        self.at_soil_sample = set()  # waypoints

        # Parse static facts
        for fact in self.static:
            parts = get_parts(fact)
            pred = parts[0]

            if pred == "store_of":
                store, rover = parts[1], parts[2]
                self.stores_of_rover[store] = rover
            elif pred == "equipped_for_soil_analysis":
                self.rover_equipped_soil.add(parts[1])
            elif pred == "equipped_for_rock_analysis":
                self.rover_equipped_rock.add(parts[1])
            elif pred == "equipped_for_imaging":
                self.rover_equipped_imaging.add(parts[1])
            elif pred == "at_lander":
                self.landers.add(parts[1])
                self.at_lander[parts[1]] = parts[2]
            elif pred == "can_traverse":
                rover, from_wp, to_wp = parts[1], parts[2], parts[3]
                self.rovers.add(rover)
                self.can_traverse.setdefault((rover, from_wp), set()).add(to_wp)
            elif pred == "visible":
                from_wp, to_wp = parts[1], parts[2]
                self.visible.setdefault(from_wp, set()).add(to_wp)
            elif pred == "visible_from":
                objective, wp = parts[1], parts[2]
                self.visible_from.setdefault(objective, set()).add(wp)
            elif pred == "calibration_target":
                camera, obj = parts[1], parts[2]
                self.calibration_targets[camera] = obj
            elif pred == "on_board":
                # camera on rover
                if parts[0] == "on_board" and fact.count("camera") > 0:
                    camera, rover = parts[1], parts[2]
                    self.on_board_camera_of_rover.setdefault(rover, set()).add(camera)
                    self.cameras_of_rover.setdefault(rover, set()).add(camera)
            elif pred == "supports":
                camera, mode = parts[1], parts[2]
                self.camera_supports.setdefault(camera, set()).add(mode)
            elif pred == "at_rock_sample":
                self.at_rock_sample.add(parts[1])
            elif pred == "at_soil_sample":
                self.at_soil_sample.add(parts[1])

        # Build waypoint connectivity graph for navigation for each rover (intersection of can_traverse and visible)
        # For heuristic we only consider the rover with most goals or just assume symmetrical graph ignoring rovers?
        # For simplicity: build one generic navigable graph of waypoints with edges where any rover can traverse & visible

        # Collect all waypoints
        waypoints = set()
        for (rover, wp), destinations in self.can_traverse.items():
            waypoints.add(wp)
            waypoints.update(destinations)
        for wp_from, wp_tos in self.visible.items():
            waypoints.add(wp_from)
            waypoints.update(wp_tos)
        self.waypoints = waypoints

        # Build an undirected navigation graph of waypoints where navigation is possible for some rover:
        # navigation precondition: can_traverse(rover, y, z) and visible(y, z)
        # We take the union over all rovers here to get a global graph to estimate shortest paths
        self.nav_graph = {wp: set() for wp in waypoints}

        # Construct edges: for any (wp1->wp2) if (can_traverse * wp1 wp2) and visible(wp1 wp2)
        # Since we want to approximate distances ignoring the specific rover (heuristic),
        # we collect all pairs that are navigable by at least one rover
        # There could be a lot of pairs but this is static information.

        # Build a map can_traverse_any[(from, to)] = True if any rover can traverse
        can_traverse_any = set()
        for (rover, from_wp), to_wps in self.can_traverse.items():
            for to_wp in to_wps:
                can_traverse_any.add((from_wp, to_wp))

        for from_wp in waypoints:
            if from_wp in self.visible:
                for to_wp in self.visible[from_wp]:
                    if (from_wp, to_wp) in can_traverse_any:
                        self.nav_graph[from_wp].add(to_wp)
                        self.nav_graph[to_wp].add(from_wp)  # Assume bidirectional for heuristic

        # Precompute all-pairs shortest path distances by Floyd-Warshall or BFS from each node
        # Because waypoints are limited, BFS for each waypoint is efficient
        self.dist = {}  # (wp1, wp2) -> dist (int)

        for start in waypoints:
            self.dist[start] = self._bfs_dist(start)

        # Extract goal requirements by type for fast filtering in heuristic
        self.goal_communicated_soil = set()
        self.goal_communicated_rock = set()
        self.goal_communicated_image = set()  # tuples of (objective, mode)

        for goal_fact in self.goals:
            parts = get_parts(goal_fact)
            pred = parts[0]
            if pred == "communicated_soil_data":
                waypoint = parts[1]
                self.goal_communicated_soil.add(waypoint)
            elif pred == "communicated_rock_data":
                waypoint = parts[1]
                self.goal_communicated_rock.add(waypoint)
            elif pred == "communicated_image_data":
                objective, mode = parts[1], parts[2]
                self.goal_communicated_image.add((objective, mode))

    def _bfs_dist(self, start):
        """Compute shortest distances from start to all other waypoints using self.nav_graph."""
        dist = {wp: float('inf') for wp in self.waypoints}
        dist[start] = 0
        queue = [start]
        head = 0
        while head < len(queue):
            current = queue[head]
            head += 1
            current_dist = dist[current]
            neighbors = self.nav_graph.get(current, [])
            for nbr in neighbors:
                if dist[nbr] == float('inf'):
                    dist[nbr] = current_dist + 1
                    queue.append(nbr)
        return dist

    def __call__(self, node):
        state = node.state

        # If goal already reached, heuristic=0
        if self.goals <= state:
            return 0

        # Extract rover locations, calibration state of cameras, stores states, samples held, images held and communicated

        rover_location = {}
        calibrated_cameras = {}  # rover -> set(camera)
        store_empty = {}  # store -> bool
        have_soil_analysis = set()  # (rover, waypoint)
        have_rock_analysis = set()  # (rover, waypoint)
        have_image = set()  # (rover, objective, mode)
        communicated_soil = set()  # waypoint
        communicated_rock = set()  # waypoint
        communicated_image = set()  # (objective, mode)

        # Extract rover locations and calibrated cameras
        for fact in state:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "at":
                rover, wp = parts[1], parts[2]
                if rover in self.rovers:
                    rover_location[rover] = wp
            elif pred == "calibrated":
                camera, rover = parts[1], parts[2]
                calibrated_cameras.setdefault(rover, set()).add(camera)
            elif pred == "empty":
                store = parts[1]
                store_empty[store] = True
            elif pred == "full":
                store = parts[1]
                store_empty[store] = False
            elif pred == "have_soil_analysis":
                rover, wp = parts[1], parts[2]
                have_soil_analysis.add((rover, wp))
            elif pred == "have_rock_analysis":
                rover, wp = parts[1], parts[2]
                have_rock_analysis.add((rover, wp))
            elif pred == "have_image":
                rover, obj, mode = parts[1], parts[2], parts[3]
                have_image.add((rover, obj, mode))
            elif pred == "communicated_soil_data":
                wp = parts[1]
                communicated_soil.add(wp)
            elif pred == "communicated_rock_data":
                wp = parts[1]
                communicated_rock.add(wp)
            elif pred == "communicated_image_data":
                obj, mode = parts[1], parts[2]
                communicated_image.add((obj, mode))

        # Determine goals still unmet
        remaining_soil = self.goal_communicated_soil - communicated_soil
        remaining_rock = self.goal_communicated_rock - communicated_rock
        remaining_image = self.goal_communicated_image - communicated_image

        # For each rover, identify their store and cameras
        rover_store = {}
        for store, rover in self.stores_of_rover.items():
            rover_store[rover] = store

        # For efficient heuristic, group data collection requirements by rover capability
        # We assign remaining goals to capable rovers heuristically by minimal navigation cost

        # Establish minimal cost helper: distance between waypoints (or waypoint -> objective visible waypoint)

        def navigation_cost(rover, from_wp, to_wp):
            # Cost is shortest path distance between waypoints (each navigation=1 action)
            # If unknown, return large number
            if from_wp not in self.dist or to_wp not in self.dist[from_wp]:
                return 50
            return self.dist[from_wp][to_wp]

        # For imaging, distance is to any waypoint from which objective is visible
        # For calibration: rover must be at waypoint visible from calibration target objective

        # Compose total estimated actions for all remaining goals minimized over rovers

        total_cost = 0

        # --- Handle Soil Data ---
        for wp in remaining_soil:
            # Find rovers equipped for soil and estimate minimal cost to fulfill:
            best_cost = float('inf')
            for rover in self.rovers:
                if rover not in rover_location or rover not in self.rover_equipped_soil:
                    continue
                # If rover already has have_soil_analysis at wp and not communicated, only communication needed
                if (rover, wp) in have_soil_analysis and wp not in communicated_soil:
                    # Cost to navigate to lander's waypoint + communicate soil data + possibly drop full store
                    lander_wp = None
                    # use first lander position:
                    if self.landers:
                        lander = next(iter(self.landers))
                        lander_wp = self.at_lander[lander]
                    cost_comm = 0
                    if lander_wp is not None:
                        nav_cost = navigation_cost(rover, rover_location[rover], lander_wp)
                        cost_comm = nav_cost + 1  # communicate action cost
                    best_cost = min(best_cost, cost_comm)
                    continue

                # Otherwise, we estimate:
                # navigation to soil sample wp
                nav_to_sample = navigation_cost(rover, rover_location[rover], wp)
                # one sample_soil action
                sample_cost = 1
                # navigation to lander waypoint
                lander_wp = None
                if self.landers:
                    lander = next(iter(self.landers))
                    lander_wp = self.at_lander[lander]
                nav_to_lander = navigation_cost(rover, wp, lander_wp) if lander_wp else 0
                # communication action
                comm_cost = 1
                # total: nav_to_sample + sample + nav_to_lander + communicate
                cost = nav_to_sample + sample_cost + nav_to_lander + comm_cost
                best_cost = min(best_cost, cost)
            if best_cost == float('inf'):
                best_cost = 20  # fallback large cost
            total_cost += best_cost

        # --- Handle Rock Data ---
        for wp in remaining_rock:
            best_cost = float('inf')
            for rover in self.rovers:
                if rover not in rover_location or rover not in self.rover_equipped_rock:
                    continue
                if (rover, wp) in have_rock_analysis and wp not in communicated_rock:
                    lander_wp = None
                    if self.landers:
                        lander = next(iter(self.landers))
                        lander_wp = self.at_lander[lander]
                    cost_comm = 0
                    if lander_wp is not None:
                        nav_cost = navigation_cost(rover, rover_location[rover], lander_wp)
                        cost_comm = nav_cost + 1
                    best_cost = min(best_cost, cost_comm)
                    continue

                nav_to_sample = navigation_cost(rover, rover_location[rover], wp)
                sample_cost = 1
                lander_wp = None
                if self.landers:
                    lander = next(iter(self.landers))
                    lander_wp = self.at_lander[lander]
                nav_to_lander = navigation_cost(rover, wp, lander_wp) if lander_wp else 0
                comm_cost = 1
                cost = nav_to_sample + sample_cost + nav_to_lander + comm_cost
                best_cost = min(best_cost, cost)
            if best_cost == float('inf'):
                best_cost = 20
            total_cost += best_cost

        # --- Handle Image Data ---
        # More complicated: need to calibrate camera, take images (consumes calibration), communicate images
        for (obj, mode) in remaining_image:
            best_cost = float('inf')
            # For each rover equipped for imaging that has camera supporting mode
            for rover in self.rovers:
                if rover not in rover_location or rover not in self.rover_equipped_imaging:
                    continue
                cameras = self.cameras_of_rover.get(rover, set())
                # Cameras supporting the mode
                cameras_with_mode = {c for c in cameras if mode in self.camera_supports.get(c, set())}
                if not cameras_with_mode:
                    continue

                # Find waypoints from which object is visible
                visible_wps = self.visible_from.get(obj, set())
                if not visible_wps:
                    continue  # cannot do imaging without visible waypoint

                rover_wp = rover_location[rover]

                lander_wp = None
                if self.landers:
                    lander = next(iter(self.landers))
                    lander_wp = self.at_lander[lander]

                # For each camera supporting mode, estimate cost to do:
                # navigation to waypoint visible from obj (for taking image)
                # calibration if needed (requires calibration target of camera visible from waypoint)
                # take_image action
                # navigate to lander_wp
                # communicate image data action

                for camera in cameras_with_mode:
                    # Is camera calibrated?
                    calibrated = False
                    if rover in calibrated_cameras and camera in calibrated_cameras[rover]:
                        calibrated = True

                    # Calibration target objective for this camera
                    calib_obj = self.calibration_targets.get(camera)
                    if not calib_obj:
                        continue  # no calibration target => ignore

                    # Waypoints from which calib_obj is visible
                    calib_visible_wps = self.visible_from.get(calib_obj, set())
                    if not calib_visible_wps:
                        continue

                    # Minimal navigation cost to get calibrated camera and take image at visible waypoint

                    # 1. If calibrated, only need nav_to_image_wp + take_image + nav_to_lander + communicate
                    # 2. Else nav_to_calib_wp + calibrate + nav_to_image_wp + take_image + nav_to_lander + communicate

                    # Find minimal combined cost over all calibration visible waypoints and image visible waypoints

                    min_cost = float('inf')
                    for calib_wp in calib_visible_wps:
                        nav_to_calib = navigation_cost(rover, rover_wp, calib_wp)
                        for image_wp in visible_wps:
                            nav_calib_to_image = navigation_cost(rover, calib_wp, image_wp)
                            nav_image_to_lander = navigation_cost(rover, image_wp, lander_wp) if lander_wp else 0

                            if calibrated:
                                cost = (navigation_cost(rover, rover_wp, image_wp) +  # direct from rover to image_wp
                                        1 +  # take_image action
                                        nav_image_to_lander +
                                        1)  # communicate_image_data action
                            else:
                                cost = (
                                    nav_to_calib +  # rover to calib_wp
                                    1 +  # calibrate action
                                    nav_calib_to_image +  # calib_wp to image_wp
                                    1 +  # take_image action
                                    nav_image_to_lander +  # image_wp to lander_wp
                                    1   # communicate action
                                )
                            if cost < min_cost:
                                min_cost = cost
                    if min_cost < best_cost:
                        best_cost = min_cost
            if best_cost == float('inf'):
                best_cost = 30  # fallback large cost
            total_cost += best_cost

        # Return heuristic cost estimate (minimum number of actions to achieve remaining goals).
        return int(total_cost)
