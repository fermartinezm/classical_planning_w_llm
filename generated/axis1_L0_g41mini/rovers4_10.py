from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    Wildcards '*' allowed.

    - `fact`: The complete fact as a string, e.g., "(at rover1 waypoint1)".
    - `args`: The expected pattern (wildcards allowed).
    Returns True if the fact matches the pattern, else False.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the rovers domain.

    Summary:
    Estimates the number of actions required to achieve the goal by tracking
    all required data collections (soil analysis, rock analysis, imaging)
    and the subsequent communications, as well as necessary calibration and movement costs.
    It takes account of rover capabilities, camera calibrations, sample availability,
    store capacity, and communication constraints.

    Assumptions:
    - Each rover's store capacity is one sample at a time (must drop before new sampling).
    - To obtain data for a particular sample or image, the corresponding rover must be equipped appropriately.
    - Calibration is required before taking images with a camera, and calibration is lost after taking an image.
    - Samples (soil, rock) and images must be communicated from waypoints or landers respectively.
    - Communication requires the rover to be at a waypoint visible to the lander.
    - Movement between waypoints is restricted by can_traverse and visibility.
    - All goals involve communicated data predicates in the final state.

    Heuristic Initialization:
    - Extract from goal facts which soil, rock, and image data (objective-mode combos) need to be communicated.
    - Extract static information on waypoints, rovers, landers, cameras, modes, and objectives.
    - Build relation maps:
      - What sample or image data is required.
      - Which rovers are equipped for what analyses.
      - Camera calibrations and modes supported.
      - Visibility between waypoints and visibility_from objectives.
      - Traverse connections between waypoints.
      - Stores of rovers and whether they start empty/full (assumed empty unless in state).
      - Locations of landers.
      - Which cameras are on which rovers.
      - Calibration target objectives for cameras.

    Step-By-Step Thinking for Computing Heuristic:
    1. Determine data items required to be communicated that are NOT yet communicated.
    2. For soil and rock samples: 
       - Check which rovers can perform sampling.
       - Estimate cost for rover to reach the sample waypoint, sample (requires store empty),
         move to lander waypoint visible from sample waypoint, and communicate.
       - Include actions for dropping full stores if needed (one drop per full store if any).
    3. For imaging objectives:
       - For each image data (objective, mode) not yet communicated:
         - Identify rovers with a camera supporting that mode.
         - Estimate cost for rover to reach waypoint visible_from objective,
           possibly calibrate camera (if not calibrated),
           take_image,
           move to lander,
           and communicate image data.
         - Calibration is lost after image, so count calibration cost as needed.
    4. Use shortest path estimates (based on can_traverse edges) for rover movement.
       - Movement cost is estimated by minimal number of navigate actions.
    5. Sum actions:
       - Navigation steps for reaching sample or imaging waypoint.
       - Sampling/take_image actions.
       - Calibration actions as required.
       - Communicate actions.
       - Drop actions if stores are full.
    6. Take the minimum cost over all available rovers for each data item.
    7. The heuristic is the sum of these minimal costs for all required data items.
    8. Heuristic value is 0 for goal states (all required data communicated).
    9. The heuristic is efficient by using lightweight shortest-path approximations and ignoring intricate dynamic details.

    This heuristic is not guaranteed admissible but aims for accuracy and efficiency to guide greedy best-first search.
    """

    def __init__(self, task):
        self.goals = task.goals
        static = task.static

        # Data structures for static info
        self.rovers = set()
        self.waypoints = set()
        self.landers = set()
        self.cameras = set()
        self.modes = set()
        self.objectives = set()
        self.stores = set()

        # Relations extracted from static facts
        self.can_traverse = {}  # rover -> wp1 -> set(wp2)
        self.visible = {}       # wp -> set(visible wps)
        self.visible_from = {}  # objective -> set(waypoints)
        self.at_lander = {}     # lander -> waypoint
        self.store_of = {}      # store -> rover
        self.equipped_for_soil_analysis = set()
        self.equipped_for_rock_analysis = set()
        self.equipped_for_imaging = set()
        self.on_board = {}       # camera -> rover
        self.calibration_target = {}  # camera -> objective
        self.supports = {}       # camera -> set(modes)

        # Goal extraction - data to communicate
        self.goal_communicated_soil = set()   # waypoints soil samples to have communicated
        self.goal_communicated_rock = set()   # waypoints rock samples to have communicated
        self.goal_communicated_image = set()  # (objective, mode) tuples

        # We'll also record waypoints that currently hold soil or rock samples,
        # but these will need to be derived dynamically from states for accurate heuristic.
        # For heuristic, assume if not in goal communicated means need processing.

        # Parse static facts
        # We identify objects by appearance in static facts and build sets.
        for fact in static:
            p, *args = get_parts(fact)
            if p == "can_traverse":
                rover, wp1, wp2 = args
                self.rovers.add(rover)
                self.waypoints.add(wp1)
                self.waypoints.add(wp2)
                if rover not in self.can_traverse:
                    self.can_traverse[rover] = {}
                self.can_traverse[rover].setdefault(wp1, set()).add(wp2)
            elif p == "visible":
                wp1, wp2 = args
                self.waypoints.add(wp1)
                self.waypoints.add(wp2)
                self.visible.setdefault(wp1, set()).add(wp2)
            elif p == "visible_from":
                obj, wp = args
                self.objectives.add(obj)
                self.waypoints.add(wp)
                self.visible_from.setdefault(obj, set()).add(wp)
            elif p == "at_lander":
                lander, wp = args
                self.landers.add(lander)
                self.waypoints.add(wp)
                self.at_lander[lander] = wp
            elif p == "store_of":
                store, rover = args
                self.stores.add(store)
                self.rovers.add(rover)
                self.store_of[store] = rover
            elif p == "equipped_for_soil_analysis":
                self.rovers.add(args[0])
                self.equipped_for_soil_analysis.add(args[0])
            elif p == "equipped_for_rock_analysis":
                self.rovers.add(args[0])
                self.equipped_for_rock_analysis.add(args[0])
            elif p == "equipped_for_imaging":
                self.rovers.add(args[0])
                self.equipped_for_imaging.add(args[0])
            elif p == "on_board":
                camera, rover = args
                self.cameras.add(camera)
                self.rovers.add(rover)
                self.on_board[camera] = rover
            elif p == "calibration_target":
                camera, obj = args
                self.cameras.add(camera)
                self.objectives.add(obj)
                self.calibration_target[camera] = obj
            elif p == "supports":
                camera, mode = args
                self.cameras.add(camera)
                self.modes.add(mode)
                if camera not in self.supports:
                    self.supports[camera] = set()
                self.supports[camera].add(mode)
            else:
                # Collect object types from args conservatively (rover, waypoint, lander, camera, mode, objective, store)
                for arg in args:
                    if arg.startswith("rover"):
                        self.rovers.add(arg)
                    elif arg.startswith("waypoint"):
                        self.waypoints.add(arg)
                    elif arg.startswith("lander"):
                        self.landers.add(arg)
                    elif arg.startswith("camera"):
                        self.cameras.add(arg)
                    elif arg.startswith("mode"):
                        self.modes.add(arg)
                    elif arg.startswith("objective"):
                        self.objectives.add(arg)
                    elif arg.startswith("store"):
                        self.stores.add(arg)

        # Parse goal facts to extract which data to communicate
        for goal in self.goals:
            p, *args = get_parts(goal)
            if p == "communicated_soil_data":
                # arg is waypoint
                self.goal_communicated_soil.add(args[0])
            elif p == "communicated_rock_data":
                self.goal_communicated_rock.add(args[0])
            elif p == "communicated_image_data":
                obj, mode = args
                self.goal_communicated_image.add((obj, mode))

        # Precompute lander waypoint list for communication (only one lander assumed is common)
        self.lander_waypoints = set(self.at_lander.values())

    def __call__(self, node):
        state = node.state

        # If goal achieved, heuristic is zero
        if self.goals <= state:
            return 0

        # Helper to extract facts matching a predicate name with wildcard args
        def facts_with(predicate, *args):
            for fact in state:
                if match(fact, predicate, *args):
                    yield fact

        # Extract dynamic info from state:

        # Current rover positions: rover -> waypoint
        rover_positions = {}
        for fact in state:
            if match(fact, "at", "*", "*"):
                _, r, wp = get_parts(fact)
                rover_positions[r] = wp

        # Store status: store -> "empty" or "full"
        store_status = {}
        for store in self.stores:
            if f"(empty {store})" in state:
                store_status[store] = "empty"
            elif f"(full {store})" in state:
                store_status[store] = "full"
            else:
                # assume empty if unknown
                store_status[store] = "empty"

        # Camera calibrations: set of (camera, rover)
        calibrated_cameras = set()
        for fact in state:
            if match(fact, "calibrated", "*", "*"):
                _, camera, rover = get_parts(fact)
                calibrated_cameras.add((camera, rover))

        # Have soil analysis: (have_soil_analysis rover waypoint)
        have_soil_analysis = set()
        for fact in state:
            if match(fact, "have_soil_analysis", "*", "*"):
                _, r, wp = get_parts(fact)
                have_soil_analysis.add((r, wp))

        # Have rock analysis: (have_rock_analysis rover waypoint)
        have_rock_analysis = set()
        for fact in state:
            if match(fact, "have_rock_analysis", "*", "*"):
                _, r, wp = get_parts(fact)
                have_rock_analysis.add((r, wp))

        # Have image: (have_image rover objective mode)
        have_image = set()
        for fact in state:
            if match(fact, "have_image", "*", "*", "*"):
                _, r, obj, mode = get_parts(fact)
                have_image.add((r, obj, mode))

        # Communicated soil data waypoints
        communicated_soil = {
            get_parts(fact)[1]
            for fact in state
            if fact.startswith("(communicated_soil_data ")
        }
        # Communicated rock data waypoints
        communicated_rock = {
            get_parts(fact)[1]
            for fact in state
            if fact.startswith("(communicated_rock_data ")
        }
        # Communicated image data (objective, mode)
        communicated_image = {
            (get_parts(fact)[1], get_parts(fact)[2])
            for fact in state
            if fact.startswith("(communicated_image_data ")
        }

        # Remaining data to communicate for soil, rock, images
        soil_to_comm = self.goal_communicated_soil - communicated_soil
        rock_to_comm = self.goal_communicated_rock - communicated_rock
        images_to_comm = self.goal_communicated_image - communicated_image

        # Prepare for shortest path computations:
        # We'll build a graph per rover of waypoint connectivity ignoring visibility direction for simplicity,
        # because navigation requires can_traverse + visible. We'll approximate connectivity.

        # Build graph for each rover: undirected graph of waypoints where rover can navigate
        graphs = {}
        for r in self.rovers:
            graphs[r] = {}
            # For simplification, create undirected graph considering can_traverse and visible edges
            ct = self.can_traverse.get(r, {})
            for src in ct:
                for dst in ct[src]:
                    if src not in graphs[r]:
                        graphs[r][src] = set()
                    if dst not in graphs[r]:
                        graphs[r][dst] = set()
                    # Only add edge if visible both ways
                    if dst in self.visible.get(src, set()) and src in self.visible.get(dst, set()):
                        graphs[r][src].add(dst)
                        graphs[r][dst].add(src)

        # Helper: shortest path length between two waypoints for a rover using BFS
        def shortest_path_length(rover, start, goal):
            if start == goal:
                return 0
            graph = graphs.get(rover)
            if not graph or start not in graph or goal not in graph:
                # No path known, pessimistically large cost
                return 50
            from collections import deque
            visited = {start}
            queue = deque([(start, 0)])
            while queue:
                current, dist = queue.popleft()
                for neighbor in graph.get(current, []):
                    if neighbor == goal:
                        return dist + 1
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, dist + 1))
            # No path found
            return 50

        # Helper to find landers and their waypoints (there is usually one lander)
        lander_wps = list(self.at_lander.values())

        # Heuristic sum of costs to fulfill all required communications
        total_cost = 0

        # 1) Soil samples to communicate:
        for wp in soil_to_comm:
            # The soil sample must be taken first if not already have_soil_analysis by any rover
            # Check if some rover already has the soil analysis at that waypoint
            have = False
            for r in self.rovers:
                if (r, wp) in have_soil_analysis:
                    have = True
                    break
            if not have:
                # Need to sample
                # Find best rover that can sample soil (equipped and store empty or can drop)
                best_cost = 1000
                for r in self.equipped_for_soil_analysis:
                    # Store for this rover
                    store = None
                    for s, rc in self.store_of.items():
                        if rc == r:
                            store = s
                            break
                    if store is None:
                        continue
                    # Rover position
                    rp = rover_positions.get(r)
                    if rp is None:
                        continue

                    nav_to_sample = shortest_path_length(r, rp, wp)
                    # Must have empty store: if full, need to drop (cost 1)
                    drop_cost = 0 if store_status[store] == "empty" else 1
                    # sample_soil action cost =1
                    # after sampling, need to move to lander waypoint visible from rover position + communicate
                    # Use first lander waypoint reachable as target
                    lander_wp = None
                    min_land_nav = 1000
                    for lwp in lander_wps:
                        # Check if sample waypoint visible from lander waypoint or can be communicated by rover at lander_wp
                        # Communication requires rover at some wp visible to lander wp
                        # We assume rover must move to lander wp where can communicate - so rover nav path
                        nav_sample_to_lander = shortest_path_length(r, wp, lwp)
                        # Must be visible for communication: visible wp to lander wp
                        if wp in self.visible and lwp in self.visible[wp]:
                            visible_for_comm = True
                        else:
                            visible_for_comm = False
                        # Conservative: require rover is at lander wp to communicate
                        # Heuristic: only consider lander wp if rover can travel there
                        if nav_sample_to_lander < min_land_nav:
                            min_land_nav = nav_sample_to_lander
                            lander_wp = lwp
                    if lander_wp is None:
                        # no known lander waypoint reachable
                        continue

                    nav_to_lander = min_land_nav
                    # communicate_soil_data action cost =1
                    cost = nav_to_sample + drop_cost + 1 + nav_to_lander + 1
                    if cost < best_cost:
                        best_cost = cost
                total_cost += best_cost if best_cost < 1000 else 50

        # 2) Rock samples to communicate:
        for wp in rock_to_comm:
            have = False
            for r in self.rovers:
                if (r, wp) in have_rock_analysis:
                    have = True
                    break
            if not have:
                best_cost = 1000
                for r in self.equipped_for_rock_analysis:
                    store = None
                    for s, rc in self.store_of.items():
                        if rc == r:
                            store = s
                            break
                    if store is None:
                        continue
                    rp = rover_positions.get(r)
                    if rp is None:
                        continue

                    nav_to_sample = shortest_path_length(r, rp, wp)
                    drop_cost = 0 if store_status[store] == "empty" else 1
                    lander_wp = None
                    min_land_nav = 1000
                    for lwp in lander_wps:
                        nav_sample_to_lander = shortest_path_length(r, wp, lwp)
                        # Check communication visibility
                        if wp in self.visible and lwp in self.visible[wp]:
                            visible_for_comm = True
                        else:
                            visible_for_comm = False
                        if nav_sample_to_lander < min_land_nav:
                            min_land_nav = nav_sample_to_lander
                            lander_wp = lwp
                    if lander_wp is None:
                        continue
                    nav_to_lander = min_land_nav
                    cost = nav_to_sample + drop_cost + 1 + nav_to_lander + 1
                    if cost < best_cost:
                        best_cost = cost
                total_cost += best_cost if best_cost < 1000 else 50

        # 3) Image data to communicate:
        for (obj, mode) in images_to_comm:
            best_cost = 1000
            # Find cameras supporting this mode
            candidate_cameras = [
                c for c in self.cameras if mode in self.supports.get(c, set())
            ]
            if not candidate_cameras:
                # no camera supports this mode: high cost
                total_cost += 50
                continue

            # Find waypoints visible_from objective (where images can be taken)
            visible_wps = self.visible_from.get(obj, set())
            if not visible_wps:
                # no visible waypoint for objective: high cost
                total_cost += 50
                continue

            for camera in candidate_cameras:
                rover = self.on_board.get(camera)
                if rover is None:
                    continue
                rp = rover_positions.get(rover)
                if rp is None:
                    continue

                # Check if camera is calibrated on rover
                calibrated = (camera, rover) in calibrated_cameras

                # Best waypoint to take image at: pick the closest visible waypoint
                best_wp_cost = 1000
                best_wp = None
                for wp in visible_wps:
                    nav_to_wp = shortest_path_length(rover, rp, wp)
                    if nav_to_wp < best_wp_cost:
                        best_wp_cost = nav_to_wp
                        best_wp = wp
                if best_wp is None:
                    continue

                # After taking image, calibration is lost,
                # so calibrate again for next image (for this heuristic just consider calibration cost once here)

                # Calibration action cost=1 (if needed)
                calibration_cost = 0 if calibrated else 1

                # take_image action cost=1
                take_image_cost = 1

                # Find lander waypoint to communicate from
                lander_wp = None
                min_land_nav = 1000
                for lwp in lander_wps:
                    nav_wp_to_lander = shortest_path_length(rover, best_wp, lwp)
                    # Communication requires rover at visible waypoint from lander waypoint;
                    # We simplify: assume rover must be at lander waypoint to communicate.
                    if best_wp in self.visible and lwp in self.visible[best_wp]:
                        visible_to_comm = True
                    else:
                        visible_to_comm = False
                    if nav_wp_to_lander < min_land_nav:
                        min_land_nav = nav_wp_to_lander
                        lander_wp = lwp
                if lander_wp is None:
                    continue

                nav_to_lander = min_land_nav
                # communicate_image_data action cost=1

                cost = (
                    best_wp_cost
                    + calibration_cost
                    + take_image_cost
                    + nav_to_lander
                    + 1
                )
                if cost < best_cost:
                    best_cost = cost

            total_cost += best_cost if best_cost < 1000 else 50

        # 4) Add drop cost for stores that are full but carry no samples to communicate anymore.
        # This encourages dropping full stores blocking new samples.
        for store, status in store_status.items():
            if status == "full":
                # Check if the rover's samples are already communicated:
                # If so, dropping is likely needed to sample others.
                rover = self.store_of.get(store)
                # We'll check if rover has any soil/rock analysis not yet communicated;
                # if none left and store full, adding drop cost to free.
                soil_pending = any(
                    (r, wp) not in have_soil_analysis or wp in soil_to_comm
                    for r, wp in have_soil_analysis
                    if r == rover
                )
                rock_pending = any(
                    (r, wp) not in have_rock_analysis or wp in rock_to_comm
                    for r, wp in have_rock_analysis
                    if r == rover
                )
                if not soil_pending and not rock_pending:
                    total_cost += 1  # cost of one drop

        # Cap heuristic to avoid overestimations
        return min(total_cost, 500)
