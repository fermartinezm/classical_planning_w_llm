from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """
    Extract the components of a PDDL fact by removing surrounding parentheses and splitting.

    Args:
        fact (str): A fact string like "(predicate arg1 arg2)".

    Returns:
        list[str]: List of components, e.g., ["predicate", "arg1", "arg2"].
    """
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern with possible wildcards.

    Args:
        fact (str): The fact string, e.g. "(at rover1 waypoint2)".
        *args (str): One pattern per fact component, e.g., ("at", "rover*", "*").

    Returns:
        bool: True if the fact matches the pattern, False otherwise.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    Summary:
    ----------
    Estimates the number of actions needed to satisfy all remaining goal conditions,
    focusing especially on communication goals (soil, rock, image data), since these require
    rovers to have samples/images and communicate them. The heuristic considers:
    - Which data (soil, rock, image modes) still need to be communicated.
    - Which rovers are equipped to acquire needed data.
    - Rover positions and reachable waypoints via "can_traverse".
    - The minimal number of navigation steps (using a BFS on static connectivity) to reach required locations.
    - Necessary sampling, calibration, imaging, drop, and communication actions.
    
    Assumptions:
    ------------
    - The heuristic is not guaranteed admissible but aims for informative and fast estimates.
    - Communication requires the rover to be at a waypoint visible to the lander.
    - Samples must be taken only if not yet acquired.
    - Cameras must be calibrated before imaging.
    - Stores must be emptied (dropped) before taking new samples.
    - Visibility and camera capabilities affect imaging.
    - Navigation costs between waypoints are estimated via shortest path ignoring dynamic conditions.
    - Heuristic accounts for multiple rovers operating independently and sums a rough minimal estimate.

    Heuristic Initialization:
    --------------------------
    - Extract sets of rovers, waypoints, landers, cameras, objectives, modes, stores from static.
    - Build adjacency graph for waypoints per rover (can_traverse) for shortest path computations.
    - Identify which rovers have which equipments (soil, rock, imaging).
    - Map stores to rovers.
    - Extract calibration targets per camera and modes supported by cameras.
    - Extract waypoints where landers are located.
    - Extract which waypoints are visible from which objectives.
    - Cache visibility relations between waypoints for quick checks.

    Step-By-Step Thinking for Computing Heuristic:
    -----------------------------------------------
    1. Parse the current state:
       - Positions of rovers.
       - What data (soil, rock, images by mode) each rover currently has.
       - What data has already been communicated.
       - Calibration status of cameras on rovers.
       - Store fullness states.
    2. Extract unmet goal communication facts:
       - Which soil/rock/image data still need to be communicated.
    3. For each required data piece not communicated:
       - Identify rovers that can do the work (e.g. equipped for soil analysis).
       - Estimate minimal steps:
         a. Navigation from rover current location to sampling/imaging location (for images: a waypoint visible from the objective).
         b. Actions needed: navigate(s) + sample or calibrate + take image (if applicable) + navigation to lander + communicate.
         c. Account for store dropping if store is full.
    4. For image goals, must consider calibration if camera is not calibrated.
    5. Use BFS on static graph for navigation distances.
    6. Sum the cost estimates for all required communications.
    7. Return 0 if all goals met.
    """

    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static

        # Extract sets of objects by type from static facts for quick membership tests.
        self.rovers = set()
        self.waypoints = set()
        self.landers = set()
        self.stores = set()
        self.cameras = set()
        self.objectives = set()
        self.modes = set()

        # calibr_targets: camera -> objective
        self.calibration_targets = {}
        # supports: camera -> set of modes
        self.camera_supports = {}

        # For quick lookup: store_of_store -> rover, rover -> stores
        self.store_of = {}
        self.stores_of_rover = {}

        # Equipped mappings
        self.equipped_for_soil = set()
        self.equipped_for_rock = set()
        self.equipped_for_imaging = set()

        # can_traverse graph: rover -> {waypoint: set(connected_waypoints)}
        self.can_traverse = {}

        # visibility: waypoint -> set(visible_waypoints)
        self.visible = {}

        # lander at waypoint: lander -> waypoint
        self.lander_at = {}

        # objectives visible from waypoints: objective -> set(waypoints)
        self.visible_from = {}

        # Build sets of objects by their naming patterns from facts
        # NOTE: The code follows implicit assumptions about names since no explicit typing is given in static:
        # We'll base on predicates referring to objects.
        for fact in self.static:
            parts = get_parts(fact)
            pred = parts[0]
            args = parts[1:]
            if pred == "equipped_for_soil_analysis" and len(args) == 1:
                self.equipped_for_soil.add(args[0])
            elif pred == "equipped_for_rock_analysis" and len(args) == 1:
                self.equipped_for_rock.add(args[0])
            elif pred == "equipped_for_imaging" and len(args) == 1:
                self.equipped_for_imaging.add(args[0])
            elif pred == "store_of" and len(args) == 2:
                store, rover = args
                self.store_of[store] = rover
                self.stores_of_rover.setdefault(rover, set()).add(store)
                self.stores.add(store)
                self.rovers.add(rover)
            elif pred == "can_traverse" and len(args) == 3:
                rover, from_wp, to_wp = args
                self.rovers.add(rover)
                self.waypoints.add(from_wp)
                self.waypoints.add(to_wp)
                self.can_traverse.setdefault(rover, {}).setdefault(from_wp, set()).add(to_wp)
            elif pred == "visible" and len(args) == 2:
                from_wp, to_wp = args
                self.waypoints.add(from_wp)
                self.waypoints.add(to_wp)
                self.visible.setdefault(from_wp, set()).add(to_wp)
            elif pred == "at_lander" and len(args) == 2:
                lander, wp = args
                self.landers.add(lander)
                self.waypoints.add(wp)
                self.lander_at[lander] = wp
            elif pred == "calibration_target" and len(args) == 2:
                camera, objective = args
                self.cameras.add(camera)
                self.objectives.add(objective)
                self.calibration_targets[camera] = objective
            elif pred == "supports" and len(args) == 2:
                camera, mode = args
                self.cameras.add(camera)
                self.modes.add(mode)
                self.camera_supports.setdefault(camera, set()).add(mode)
            elif pred == "on_board" and len(args) == 2:
                camera, rover = args
                # We get cameras and rovers here as well.
                self.cameras.add(camera)
                self.rovers.add(rover)
            elif pred == "visible_from" and len(args) == 2:
                objective, wp = args
                self.objectives.add(objective)
                self.waypoints.add(wp)
                self.visible_from.setdefault(objective, set()).add(wp)
            else:
                # If a fact looks like (at rover waypoint) or (at objective waypoint)
                if pred == "at" and len(args) == 2:
                    if args[0].startswith("rover"):
                        self.rovers.add(args[0])
                        self.waypoints.add(args[1])
                    else:
                        # Could be objective or others
                        self.waypoints.add(args[1])
                        if args[0].startswith("objective"):
                            self.objectives.add(args[0])
                # Sometimes types may be deduced from naming patterns:
                for arg in args:
                    if arg.startswith("rover"):
                        self.rovers.add(arg)
                    elif arg.startswith("waypoint"):
                        self.waypoints.add(arg)
                    elif arg.startswith("objective"):
                        self.objectives.add(arg)
                    elif arg.startswith("lander"):
                        self.landers.add(arg)
                    elif arg.startswith("camera"):
                        self.cameras.add(arg)
                    elif arg.startswith("mode"):
                        self.modes.add(arg)
                    elif arg.startswith("store"):
                        self.stores.add(arg)

        # For navigation distances: build undirected graphs for each rover over waypoints reachable via can_traverse.
        # We assume can_traverse facts are directional; make the graph symmetric assuming navigation is possible back and forth
        # if both directions are present. We'll treat only edges present as undirected edges.
        self.navigation_graph = {}
        for rover in self.rovers:
            graph = {}
            if rover in self.can_traverse:
                for from_wp, neighbors in self.can_traverse[rover].items():
                    graph.setdefault(from_wp, set()).update(neighbors)
                # Add reverse edges only if explicitly specified (unidirectional)
                for from_wp in list(graph):
                    for to_wp in list(graph[from_wp]):
                        if to_wp in graph and from_wp in graph[to_wp]:
                            # Both directions present, treat undirected
                            continue
                        else:
                            # Edge is one-directional
                            pass
            self.navigation_graph[rover] = graph

        # We cache some useful lander waypoints (there may be multiple landers)
        self.lander_waypoints = set(self.lander_at.values())

    def _bfs_shortest_path(self, graph, start, goals):
        """
        Compute the shortest distance from start to closest goal using BFS on a graph.

        Args:
            graph (dict[str:set[str]]): adjacency list representing waypoint connectivity.
            start (str): start waypoint.
            goals (set[str]): set of goal waypoints.

        Returns:
            int: minimal number of navigation steps (edges) to reach any of the goal waypoints.
                 Returns a large number if unreachable.
        """
        if start in goals:
            return 0
        visited = {start}
        frontier = [(start, 0)]
        while frontier:
            node, dist = frontier.pop(0)
            neighbors = graph.get(node, set())
            for neighbor in neighbors:
                if neighbor not in visited:
                    if neighbor in goals:
                        return dist + 1
                    visited.add(neighbor)
                    frontier.append((neighbor, dist + 1))
        return 1000  # Large number representing unreachable

    def __call__(self, node):
        """
        Evaluate heuristic estimation of remaining actions to satisfy the goal(s).

        Args:
            node: Search node with attribute `state` (frozenset of facts).

        Returns:
            int: Estimated number of required actions to reach a goal state from current state.
        """
        state = node.state

        # Quick check if goal is reached
        if self.goals <= state:
            return 0

        # Utility matcher to find facts in the current state quickly
        def fact_match(*args):
            return {fact for fact in state if match(fact, *args)}

        # Extract rover positions
        rover_positions = {}
        for fact in fact_match("at", "*", "*"):
            parts = get_parts(fact)
            rover, wp = parts[1], parts[2]
            if rover in self.rovers:  # only rover positions
                rover_positions[rover] = wp

        # Extract store status for each store
        store_empty = set()
        store_full = set()
        for fact in fact_match("empty", "*"):
            store_empty.add(get_parts(fact)[1])
        for fact in fact_match("full", "*"):
            store_full.add(get_parts(fact)[1])

        # Extract samples and images currently held by rovers
        have_soil = set()  # (rover, waypoint)
        have_rock = set()  # (rover, waypoint)
        have_image = set()  # (rover, objective, mode)
        for fact in fact_match("have_soil_analysis", "*", "*"):
            parts = get_parts(fact)
            have_soil.add((parts[1], parts[2]))
        for fact in fact_match("have_rock_analysis", "*", "*"):
            parts = get_parts(fact)
            have_rock.add((parts[1], parts[2]))
        for fact in fact_match("have_image", "*", "*", "*"):
            parts = get_parts(fact)
            have_image.add((parts[1], parts[2], parts[3]))

        # Extract communicated data (soil, rock, image)
        communicated_soil = set()
        communicated_rock = set()
        communicated_image = set()
        for fact in fact_match("communicated_soil_data", "*"):
            communicated_soil.add(get_parts(fact)[1])
        for fact in fact_match("communicated_rock_data", "*"):
            communicated_rock.add(get_parts(fact)[1])
        for fact in fact_match("communicated_image_data", "*", "*"):
            parts = get_parts(fact)
            communicated_image.add((parts[1], parts[2]))

        # Extract calibration status of cameras on rovers
        calibrated_cameras = set()
        for fact in fact_match("calibrated", "*", "*"):
            parts = get_parts(fact)
            calibrated_cameras.add((parts[1], parts[2]))  # (camera, rover)

        # For estimation: We'll consider three categories of goals to accomplish:
        #  1) Communicate soil data at waypoints.
        #  2) Communicate rock data at waypoints.
        #  3) Communicate image data for objectives in specific modes.

        # Determine unmet communication goals from overall goals:
        soil_goals = set()
        rock_goals = set()
        image_goals = set()
        for goal in self.goals:
            parts = get_parts(goal)
            pred = parts[0]
            if pred == "communicated_soil_data":
                soil_goals.add(parts[1])
            elif pred == "communicated_rock_data":
                rock_goals.add(parts[1])
            elif pred == "communicated_image_data":
                image_goals.add((parts[1], parts[2]))

        soil_to_do = soil_goals - communicated_soil
        rock_to_do = rock_goals - communicated_rock
        image_to_do = image_goals - communicated_image

        total_cost = 0

        # For visibility to lander: Identify waypoints where landers are
        # and check visibility from them to other waypoints (for communication)
        lander_waypoints = self.lander_waypoints

        # Cache all visible waypoints from each waypoint (including to itself)
        # We'll build visible sets for quick queries: waypoint -> set(visible waypoints)
        visible_map = {wp: self.visible.get(wp, set()) | {wp} for wp in self.waypoints}

        # Helper to find minimal navigation steps for a rover from position to set of goal waypoints
        def nav_dist(rover, start, goals):
            if rover in self.navigation_graph:
                graph = self.navigation_graph[rover]
                return self._bfs_shortest_path(graph, start, goals)
            return 1000

        # Compute cost estimate for soil data communication per sample waypoint in soil_to_do
        # For each soil communication goal:
        for sample_wp in soil_to_do:
            # We must find a rover that:
            # - is equipped for soil analysis
            # - has empty store or can empty store by drop actions
            # - can navigate to sample_wp
            # - can navigate to a waypoint visible from lander (to communicate)
            candidate_costs = []
            for rover in self.rovers:
                if rover not in self.equipped_for_soil:
                    continue
                rover_wp = rover_positions.get(rover, None)
                if rover_wp is None:
                    continue  # rover position unknown, skip

                # Find rover's store states
                rover_stores = self.stores_of_rover.get(rover, set())

                # We'll check store availability: if no store is empty, must drop first (1 action)
                empty_store = any(s in store_empty for s in rover_stores)
                full_store = any(s in store_full for s in rover_stores)
                drop_cost = 0 if empty_store else 1

                # Check if sample_wp still has soil sample fact (if it's gone, sample cannot be taken)
                # The domain model deletes (at_soil_sample) on sampling, so if present means sample available.
                # But in state we do not have at_soil_sample directly, so we cannot check dynamically.
                # For heuristic, assume sample available if not already "have_soil_analysis" by any rover at that wp.
                already_sampled = any(p == sample_wp for (r, p) in have_soil)
                if already_sampled:
                    # This means the sample was taken by someone; no need to sample again.
                    # Only communicate is left, if not communicated.
                    # Communication requires at waypoint visible to lander.
                    # For communication, rover must have (have_soil_analysis rover wp)
                    # Let's check if rover already has soil analysis at sample_wp
                    has_sample = (rover, sample_wp) in have_soil
                    if has_sample:
                        # Just navigation to communication spot + communicate action
                        # Find waypoints visible to any lander waypoint reachable by rover navigation
                        comm_goals = {wp for wp in self.waypoints if any(wp in visible_map[lwp] for lwp in lander_waypoints)}
                        dist_to_comm = nav_dist(rover, rover_wp, comm_goals)
                        # Actions: navigate steps + communicate_soil_data (1)
                        cost = dist_to_comm + 1
                        candidate_costs.append(cost)
                    else:
                        # Rover does not have sample, might need sample + communicate by some other rover
                        # We'll count no cost here but heuristic conservative: try next rover.
                        continue
                else:
                    # Need to navigate to sample location: navigation cost from rover_wp to sample_wp
                    dist_to_sample = nav_dist(rover, rover_wp, {sample_wp})
                    if dist_to_sample >= 1000:
                        continue  # cannot reach sample waypoint, skip rover

                    # After sampling: take 1 action (sample_soil)
                    # Then navigate to communication waypoint visible from lander
                    comm_goals = {wp for wp in self.waypoints if any(wp in visible_map[lwp] for lwp in lander_waypoints)}
                    dist_to_comm = self._bfs_shortest_path(self.navigation_graph.get(rover, {}), sample_wp, comm_goals)
                    if dist_to_comm >= 1000:
                        continue  # cannot reach communication waypoint, skip rover

                    # Finally, communicate action (1)
                    cost = drop_cost + dist_to_sample + 1 + dist_to_comm + 1
                    candidate_costs.append(cost)

            # If no rover can do this task, add large penalty, else add minimal cost among candidates
            if candidate_costs:
                total_cost += min(candidate_costs)
            else:
                total_cost += 1000  # large penalty (unreachable)

        # Repeat similar procedure for rock data communication goals
        for sample_wp in rock_to_do:
            candidate_costs = []
            for rover in self.rovers:
                if rover not in self.equipped_for_rock:
                    continue
                rover_wp = rover_positions.get(rover, None)
                if rover_wp is None:
                    continue

                rover_stores = self.stores_of_rover.get(rover, set())
                empty_store = any(s in store_empty for s in rover_stores)
                full_store = any(s in store_full for s in rover_stores)
                drop_cost = 0 if empty_store else 1

                already_sampled = any(p == sample_wp for (r, p) in have_rock)
                if already_sampled:
                    has_sample = (rover, sample_wp) in have_rock
                    if has_sample:
                        comm_goals = {wp for wp in self.waypoints if any(wp in visible_map[lwp] for lwp in lander_waypoints)}
                        dist_to_comm = nav_dist(rover, rover_wp, comm_goals)
                        cost = dist_to_comm + 1
                        candidate_costs.append(cost)
                    else:
                        continue
                else:
                    dist_to_sample = nav_dist(rover, rover_wp, {sample_wp})
                    if dist_to_sample >= 1000:
                        continue

                    comm_goals = {wp for wp in self.waypoints if any(wp in visible_map[lwp] for lwp in lander_waypoints)}
                    dist_to_comm = self._bfs_shortest_path(self.navigation_graph.get(rover, {}), sample_wp, comm_goals)
                    if dist_to_comm >= 1000:
                        continue

                    cost = drop_cost + dist_to_sample + 1 + dist_to_comm + 1
                    candidate_costs.append(cost)

            if candidate_costs:
                total_cost += min(candidate_costs)
            else:
                total_cost += 1000

        # For images: consider each (objective, mode) pair that must be communicated
        # Steps: 
        # - rover with imaging equipment
        # - has a camera that supports mode on board
        # - camera must be calibrated before imaging
        # - must be at a waypoint visible from objective for imaging
        # - must have calibration target and use it to calibrate camera before imaging
        # - after imaging, the camera gets uncalibrated
        # - then navigate to waypoint visible to lander and communicate

        for (objective, mode) in image_to_do:
            candidate_costs = []
            waypoints_visible_from_obj = self.visible_from.get(objective, set())
            if not waypoints_visible_from_obj:
                # No visible waypoints from this objective => cannot image, penalize highly
                total_cost += 1000
                continue

            for rover in self.rovers:
                if rover not in self.equipped_for_imaging:
                    continue
                rover_wp = rover_positions.get(rover, None)
                if rover_wp is None:
                    continue

                # Identify cameras on board rover supporting this mode
                cameras_on_rover_supporting_mode = []
                for camera in self.cameras:
                    # on_board(camera, rover) must hold in static or state
                    on_board_fact_static = ("(on_board {} {})".format(camera, rover) in self.static)
                    on_board_fact_state = ("(on_board {} {})".format(camera, rover) in state)
                    if not (on_board_fact_static or on_board_fact_state):
                        continue
                    if mode not in self.camera_supports.get(camera, set()):
                        continue
                    cameras_on_rover_supporting_mode.append(camera)

                if not cameras_on_rover_supporting_mode:
                    continue

                for camera in cameras_on_rover_supporting_mode:
                    # Is camera calibrated with rover?
                    calibrated = (camera, rover) in calibrated_cameras

                    # Get calibration target objective for camera
                    calib_target_obj = self.calibration_targets.get(camera, None)
                    # To calibrate, rover must be at a waypoint visible from calib_target_obj
                    # Find waypoints visible from calib_target_obj
                    calib_visible_wps = self.visible_from.get(calib_target_obj, set())
                    if not calib_visible_wps:
                        continue

                    # Rover must navigate: 
                    # current_wp -> calibration waypoint (if not calibrated)
                    # calibration action (1)
                    # then navigate to imaging waypoint (visible from objective) 
                    # imaging action (1)
                    # then navigate to communication waypoint (visible from lander) 
                    # communication action (1)

                    navigation_cost = 0

                    # Navigation graph for the rover
                    graph = self.navigation_graph.get(rover, {})

                    # Dist to calibration waypoint if not calibrated
                    if not calibrated:
                        dist_to_calib_wp = self._bfs_shortest_path(graph, rover_wp, calib_visible_wps)
                        if dist_to_calib_wp >= 1000:
                            continue
                        navigation_cost += dist_to_calib_wp
                        navigation_cost += 1  # calibrate action
                        position_after_calib = None
                        # After calibration, rover is at calib waypoint
                        # Use calib waypoint position for next navigation
                        # Simplify to use calibration waypoint as starting point to imaging
                        position_after_calib = None
                        # Find the minimal dist to imaging waypoint from calibration waypoint(s)
                        # Choose minimal over all calib_visible_wps
                        minimal_imaging_dist = 1000
                        minimal_imaging_pos = None
                        for calib_wp in calib_visible_wps:
                            dist_imaging = self._bfs_shortest_path(graph, calib_wp, waypoints_visible_from_obj)
                            if dist_imaging < minimal_imaging_dist:
                                minimal_imaging_dist = dist_imaging
                                minimal_imaging_pos = calib_wp
                        if minimal_imaging_dist >= 1000:
                            continue
                        navigation_cost += minimal_imaging_dist
                        navigation_cost += 1  # take_image action
                        position_after_imaging = None
                        if minimal_imaging_pos is None:
                            position_after_imaging = rover_wp  # fallback
                        else:
                            # We assume rover ends at imaging waypoint
                            # Actually take_image requires rover at that waypoint: minimal_imaging_pos
                            position_after_imaging = minimal_imaging_pos
                    else:
                        # Already calibrated
                        # rover_wp -> imaging waypoint
                        dist_imaging = self._bfs_shortest_path(graph, rover_wp, waypoints_visible_from_obj)
                        if dist_imaging >= 1000:
                            continue
                        navigation_cost += dist_imaging
                        navigation_cost += 1  # take_image action
                        position_after_imaging = None
                        # Assume final position is imaging waypoint (one of waypoints_visible_from_obj)
                        # Let's pick minimal distance waypoint for position_after_imaging:
                        min_dist = 1000
                        pos_wp = None
                        for wp in waypoints_visible_from_obj:
                            d = self._bfs_shortest_path(graph, rover_wp, {wp})
                            if d < min_dist:
                                min_dist = d
                                pos_wp = wp
                        position_after_imaging = pos_wp or rover_wp

                    # After imaging, camera is uncalibrated and rover must go communicate
                    # Navigate to communication waypoint (visible from lander) from position_after_imaging

                    comm_waypoints = {wp for wp in self.waypoints if any(wp in visible_map[lwp] for lwp in lander_waypoints)}
                    dist_to_comm = self._bfs_shortest_path(graph, position_after_imaging, comm_waypoints)
                    if dist_to_comm >= 1000:
                        continue
                    navigation_cost += dist_to_comm
                    navigation_cost += 1  # communicate_image_data action

                    candidate_costs.append(navigation_cost)

            if candidate_costs:
                total_cost += min(candidate_costs)
            else:
                total_cost += 1000

        # Return final heuristic estimate (ensure finite and zero only if goal reached)
        # If no unmet goals or no costs accumulated, return 0
        return int(total_cost if total_cost > 0 else 0)
