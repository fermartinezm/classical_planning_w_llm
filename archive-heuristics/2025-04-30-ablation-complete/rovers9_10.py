from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic # Assuming Heuristic base class is available

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact strings or malformed facts gracefully
    if not fact or not isinstance(fact, str) or len(fact) < 2 or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at rover1 waypoint1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def build_graph(waypoints, visible_facts):
    """Builds an adjacency list representation of the waypoint graph based on visible facts."""
    graph = {wp: set() for wp in waypoints}
    for fact in visible_facts:
        parts = get_parts(fact)
        if len(parts) == 3 and parts[0] == 'visible':
            wp1, wp2 = parts[1], parts[2]
            if wp1 in graph and wp2 in graph:
                graph[wp1].add(wp2)
                graph[wp2].add(wp1) # Assuming visibility is symmetric
    return graph

def bfs(graph, start_node):
    """Computes shortest path distances from a start node to all other nodes in the graph."""
    distances = {node: float('inf') for node in graph}
    if start_node not in graph:
        return distances # Start node not in graph

    distances[start_node] = 0
    queue = deque([start_node])

    while queue:
        u = queue.popleft()
        if u in graph: # Ensure u is a valid key
            for v in graph[u]:
                if distances[v] == float('inf'):
                    distances[v] = distances[u] + 1
                    queue.append(v)
    return distances

class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve all goal conditions.
    It sums up the estimated costs for each unachieved goal fact. The cost for each goal
    is estimated based on the minimum actions needed to collect the required data (sample
    or image) and then navigate to a communication point to transmit the data. Navigation
    costs are estimated using shortest path distances on the waypoint graph.

    # Assumptions
    - Navigation is possible between any two waypoints connected by a `visible` fact (assuming `can_traverse` holds for relevant rovers).
    - Visibility between waypoints is symmetric.
    - Rovers can drop samples if their store is full (costed as 1 action).
    - If a sample is no longer at its original waypoint but not yet communicated, it is assumed to have been collected by *some* equipped rover, and the cost to "get" the sample becomes 0, leaving only the communication cost. If the sample is gone and no rover has `have_soil_analysis` or `have_rock_analysis` for that waypoint, the goal is considered unachievable.
    - If an image is required but not yet taken, it is assumed the best-suited imaging rover (equipped, with supporting camera) will take it.
    - Calibration is required before taking an image and uncalibrates the camera. The heuristic adds the cost if the relevant camera is not calibrated.
    - Communication is possible from any waypoint visible from the lander's location.

    # Heuristic Initialization
    - Parses static facts to identify:
        - Lander location.
        - Rovers equipped for soil, rock, and imaging.
        - Store ownership for each rover.
        - Camera details (on which rover, supported modes, calibration target).
        - Waypoints from which objectives and calibration targets are visible.
        - The waypoint graph based on `visible` facts.
    - Precomputes all-pairs shortest path distances between waypoints using BFS.
    - Identifies communication waypoints (visible from the lander).

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Initialize total heuristic cost to 0.
    2. Parse the current state to get rover positions, store statuses, collected data (`have_...`), calibrated cameras, and remaining samples at waypoints.
    3. Identify communication waypoints visible from the lander. If none exist and there are communication goals, the problem is unsolvable from this state (return infinity).
    4. For each goal fact in the task's goals:
        a. If the goal fact is already true in the current state, add 0 cost for this goal.
        b. If the goal is `(communicated_soil_data ?w)`:
            i. Add 1 for the `communicate_soil_data` action.
            ii. Estimate navigation cost to a communication point: Find the minimum distance from *any* rover's current position to *any* communication waypoint. Add this minimum distance.
            iii. Check if `(have_soil_analysis R ?w)` is true for *any* rover R in the state.
            iv. If not true:
                - Add 1 for the `sample_soil` action.
                - Check if `(at_soil_sample ?w)` is still true in the state.
                - If true: Estimate navigation cost to the sample: Find the minimum distance from an *equipped soil rover* with an *empty store* to waypoint `?w`. Add this minimum distance. If the best rover's store is full, add 1 for a `drop` action. If no equipped soil rover with an empty store exists, find the best equipped soil rover and add 1 for drop if its store is full, plus navigation. If no equipped soil rover exists or is reachable, the goal is unachievable (return infinity).
                - If false (sample is gone): Assume the sample was collected, so the cost to get it is 0. (If sample is gone and no rover has `have_soil_analysis`, this state is unachievable, handled by step 4.b.iii check).
            v. Add the calculated cost for this soil goal to the total cost.
        c. If the goal is `(communicated_rock_data ?w)`: Follow a similar logic as for soil data, using rock-specific predicates and equipped rovers.
        d. If the goal is `(communicated_image_data ?o ?m)`:
            i. Add 1 for the `communicate_image_data` action.
            ii. Estimate navigation cost to a communication point: Find the minimum distance from *any* rover's current position to *any* communication waypoint. Add this minimum distance.
            iii. Check if `(have_image R ?o ?m)` is true for *any* rover R in the state.
            iv. If not true:
                - Add 1 for the `take_image` action.
                - Estimate navigation cost to an imaging location: Find the minimum distance from an *equipped imaging rover* with a *camera supporting mode ?m* to *any waypoint ?p* where `(visible_from ?o ?p)` is true. Add this minimum distance. If no such waypoint exists or is reachable, the goal is unachievable (return infinity).
                - Check if the camera used (or the best camera for this rover/mode) is calibrated. If not:
                    - Add 1 for the `calibrate` action.
                    - Estimate navigation cost to a calibration location: Find the minimum distance from the same rover/camera to *any waypoint ?w* where the camera's calibration target is visible (`(visible_from target ?w)`). Add this minimum distance. If no such waypoint exists or is reachable, or the camera has no calibration target, the goal is unachievable (return infinity).
            v. Add the calculated cost for this image goal to the total cost.
    5. Return the total calculated cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # --- Parse Static Facts ---
        self.lander_wp = None
        self.equipped_soil_rovers = set()
        self.equipped_rock_rovers = set()
        self.equipped_imaging_rovers = set()
        self.rover_to_store = {}
        self.rover_to_camera = {} # rover -> list of cameras
        self.camera_to_modes = {} # camera -> list of modes
        self.camera_to_caltarget = {} # camera -> target
        self.objective_to_wps = {} # objective -> list of wps
        self.target_to_wps = {} # target -> list of wps
        self.waypoints = set()
        visible_facts = set()

        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue

            pred = parts[0]
            if pred == 'at_lander' and len(parts) == 3:
                self.lander_wp = parts[2]
            elif pred == 'equipped_for_soil_analysis' and len(parts) == 2:
                self.equipped_soil_rovers.add(parts[1])
            elif pred == 'equipped_for_rock_analysis' and len(parts) == 2:
                self.equipped_rock_rovers.add(parts[1])
            elif pred == 'equipped_for_imaging' and len(parts) == 2:
                self.equipped_imaging_rovers.add(parts[1])
            elif pred == 'store_of' and len(parts) == 3:
                store, rover = parts[1], parts[2]
                self.rover_to_store[rover] = store
            elif pred == 'on_board' and len(parts) == 3:
                camera, rover = parts[1], parts[2]
                self.rover_to_camera.setdefault(rover, []).append(camera)
            elif pred == 'supports' and len(parts) == 3:
                camera, mode = parts[1], parts[2]
                self.camera_to_modes.setdefault(camera, []).append(mode)
            elif pred == 'calibration_target' and len(parts) == 3:
                camera, target = parts[1], parts[2]
                self.camera_to_caltarget[camera] = target
            elif pred == 'visible_from' and len(parts) == 3:
                obj_or_target, wp = parts[1], parts[2]
                # Need to distinguish objectives from targets. PDDL doesn't explicitly type in facts.
                # Assume anything in calibration_target is a target, others are objectives.
                is_target = False
                for static_fact in static_facts:
                    if match(static_fact, "calibration_target", "*", obj_or_target):
                         is_target = True
                         break

                if is_target:
                    self.target_to_wps.setdefault(obj_or_target, []).append(wp)
                else:
                    self.objective_to_wps.setdefault(obj_or_target, []).append(wp)
            elif pred in ['at', 'at_lander', 'can_traverse', 'visible', 'at_soil_sample', 'at_rock_sample'] and len(parts) >= 3:
                 # Collect all waypoints mentioned in relevant facts
                 for part in parts[1:]:
                     # Simple check if it looks like a waypoint
                     if any(wp_pred in part for wp_pred in ['waypoint', 'lander']):
                          self.waypoints.add(part)

            if pred == 'visible' and len(parts) == 3:
                 visible_facts.add(fact)


        # Ensure lander waypoint is in waypoints if it exists
        if self.lander_wp:
             self.waypoints.add(self.lander_wp)

        # Build waypoint graph and compute distances
        self.graph = build_graph(self.waypoints, visible_facts)
        self.dist = {}
        for wp in self.waypoints:
            self.dist[wp] = bfs(self.graph, wp)

        # Identify communication waypoints
        self.comm_wps = set()
        if self.lander_wp and self.lander_wp in self.graph:
             # Communication is possible from any waypoint visible from the lander's waypoint
             self.comm_wps = self.graph.get(self.lander_wp, set())
             # Also include the lander waypoint itself if it's a valid waypoint
             if self.lander_wp in self.waypoints:
                 self.comm_wps.add(self.lander_wp)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # --- Parse State Facts ---
        rover_pos = {}
        store_full = set()
        have_soil = set() # Store as (rover, waypoint) tuples
        have_rock = set() # Store as (rover, waypoint) tuples
        have_image = set() # Store as (rover, objective, mode) tuples
        calibrated_cams = set() # Store as (camera, rover) tuples
        soil_samples_at_wp = set() # Store as waypoint strings
        rock_samples_at_wp = set() # Store as waypoint strings

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            pred = parts[0]
            if pred == 'at' and len(parts) == 3:
                obj, wp = parts[1], parts[2]
                # Assuming only rovers are 'at' waypoints in the state facts we care about for pos
                if obj.startswith('rover'):
                    rover_pos[obj] = wp
            elif pred == 'full' and len(parts) == 2:
                store_full.add(parts[1])
            elif pred == 'have_soil_analysis' and len(parts) == 3:
                rover, wp = parts[1], parts[2]
                have_soil.add((rover, wp))
            elif pred == 'have_rock_analysis' and len(parts) == 3:
                rover, wp = parts[1], parts[2]
                have_rock.add((rover, wp))
            elif pred == 'have_image' and len(parts) == 4:
                rover, obj, mode = parts[1], parts[2], parts[3]
                have_image.add((rover, obj, mode))
            elif pred == 'calibrated' and len(parts) == 3:
                camera, rover = parts[1], parts[2]
                calibrated_cams.add((camera, rover))
            elif pred == 'at_soil_sample' and len(parts) == 2:
                soil_samples_at_wp.add(parts[1])
            elif pred == 'at_rock_sample' and len(parts) == 2:
                rock_samples_at_wp.add(parts[1])

        total_cost = 0

        # Check if communication is possible at all if needed
        needs_communication_goal = False
        for goal in self.goals:
             if match(goal, "communicated_soil_data", "*") or \
                match(goal, "communicated_rock_data", "*") or \
                match(goal, "communicated_image_data", "*", "*"):
                 if goal not in state:
                     needs_communication_goal = True
                     break

        if needs_communication_goal and not self.comm_wps:
             return float('inf') # Cannot communicate if no comm waypoints

        # --- Estimate Cost for Each Goal ---
        for goal in self.goals:
            if goal in state:
                continue # Goal already achieved

            goal_parts = get_parts(goal)
            if not goal_parts: continue

            pred = goal_parts[0]

            if pred == 'communicated_soil_data' and len(goal_parts) == 2:
                w = goal_parts[1]
                cost_g = 0

                # Cost to communicate
                cost_g += 1 # communicate action
                min_nav_to_comm = float('inf')
                for rover_name, current_wp in rover_pos.items():
                    min_nav_rover_comm = float('inf')
                    if current_wp in self.dist:
                        for comm_wp in self.comm_wps:
                            if comm_wp in self.dist[current_wp]:
                                min_nav_rover_comm = min(min_nav_rover_comm, self.dist[current_wp][comm_wp])
                    min_nav_to_comm = min(min_nav_to_comm, min_nav_rover_comm)

                if min_nav_to_comm == float('inf'): return float('inf') # Cannot reach comm point
                cost_g += min_nav_to_comm

                # Cost to get sample if not already held
                found_have_soil = any((r, w) in have_soil for r in self.equipped_soil_rovers)

                if not found_have_soil:
                    cost_g += 1 # sample action
                    if w in soil_samples_at_wp:
                        min_sample_nav_cost = float('inf')
                        for rover_name in self.equipped_soil_rovers:
                            if rover_name in rover_pos:
                                rover_wp = rover_pos[rover_name]
                                if rover_wp in self.dist and w in self.dist[rover_wp]:
                                    nav_cost = self.dist[rover_wp][w]
                                    sample_nav_cost = nav_cost
                                    store_name = self.rover_to_store.get(rover_name)
                                    if store_name and store_name in store_full:
                                        sample_nav_cost += 1 # Cost to drop
                                    min_sample_nav_cost = min(min_sample_nav_cost, sample_nav_cost)
                        if min_sample_nav_cost == float('inf'): return float('inf') # Cannot reach sample
                        cost_g += min_sample_nav_cost
                    else:
                        # Sample is gone, but not communicated, and no rover has it -> Unachievable
                        return float('inf')

                total_cost += cost_g

            elif pred == 'communicated_rock_data' and len(goal_parts) == 2:
                w = goal_parts[1]
                cost_g = 0

                # Cost to communicate
                cost_g += 1 # communicate action
                min_nav_to_comm = float('inf')
                for rover_name, current_wp in rover_pos.items():
                    min_nav_rover_comm = float('inf')
                    if current_wp in self.dist:
                        for comm_wp in self.comm_wps:
                            if comm_wp in self.dist[current_wp]:
                                min_nav_rover_comm = min(min_nav_rover_comm, self.dist[current_wp][comm_wp])
                    min_nav_to_comm = min(min_nav_to_comm, min_nav_rover_comm)

                if min_nav_to_comm == float('inf'): return float('inf') # Cannot reach comm point
                cost_g += min_nav_to_comm

                # Cost to get sample if not already held
                found_have_rock = any((r, w) in have_rock for r in self.equipped_rock_rovers)

                if not found_have_rock:
                    cost_g += 1 # sample action
                    if w in rock_samples_at_wp:
                        min_sample_nav_cost = float('inf')
                        for rover_name in self.equipped_rock_rovers:
                            if rover_name in rover_pos:
                                rover_wp = rover_pos[rover_name]
                                if rover_wp in self.dist and w in self.dist[rover_wp]:
                                    nav_cost = self.dist[rover_wp][w]
                                    sample_nav_cost = nav_cost
                                    store_name = self.rover_to_store.get(rover_name)
                                    if store_name and store_name in store_full:
                                        sample_nav_cost += 1 # Cost to drop
                                    min_sample_nav_cost = min(min_sample_nav_cost, sample_nav_cost)
                        if min_sample_nav_cost == float('inf'): return float('inf') # Cannot reach sample
                        cost_g += min_sample_nav_cost
                    else:
                        # Sample is gone, but not communicated, and no rover has it -> Unachievable
                        return float('inf')

                total_cost += cost_g

            elif pred == 'communicated_image_data' and len(goal_parts) == 3:
                o, m = goal_parts[1], goal_parts[2]
                cost_g = 0

                # Cost to communicate
                cost_g += 1 # communicate action
                min_nav_to_comm = float('inf')
                for rover_name, current_wp in rover_pos.items():
                    min_nav_rover_comm = float('inf')
                    if current_wp in self.dist:
                        for comm_wp in self.comm_wps:
                            if comm_wp in self.dist[current_wp]:
                                min_nav_rover_comm = min(min_nav_rover_comm, self.dist[current_wp][comm_wp])
                    min_nav_to_comm = min(min_nav_to_comm, min_nav_rover_comm)

                if min_nav_to_comm == float('inf'): return float('inf') # Cannot reach comm point
                cost_g += min_nav_to_comm

                # Cost to get image if not already held
                found_have_image = any((r, o, m) in have_image for r in self.equipped_imaging_rovers)

                if not found_have_image:
                    cost_g += 1 # take_image action
                    min_image_nav_cost = float('inf')

                    img_wps = self.objective_to_wps.get(o, [])
                    if not img_wps: return float('inf') # Cannot image this objective from anywhere

                    for rover_name in self.equipped_imaging_rovers:
                        if rover_name in rover_pos:
                            rover_wp = rover_pos[rover_name]
                            if rover_wp not in self.dist: continue # Rover at unknown waypoint

                            for camera_name in self.rover_to_camera.get(rover_name, []):
                                if m in self.camera_to_modes.get(camera_name, []):
                                    # This rover/camera can take the image in this mode
                                    current_image_nav_cost = float('inf')

                                    # Nav to image location
                                    nav_to_img = float('inf')
                                    for img_wp in img_wps:
                                        if img_wp in self.dist[rover_wp]:
                                            nav_to_img = min(nav_to_img, self.dist[rover_wp][img_wp])
                                    if nav_to_img == float('inf'): continue # Cannot reach any imaging waypoint

                                    current_image_nav_cost = nav_to_img

                                    # Calibration cost if needed
                                    if (camera_name, rover_name) not in calibrated_cams:
                                        current_image_nav_cost += 1 # calibrate action
                                        cal_target = self.camera_to_caltarget.get(camera_name)
                                        if not cal_target: continue # Camera has no calibration target

                                        cal_wps = self.target_to_wps.get(cal_target, [])
                                        if not cal_wps: continue # Calibration target not visible from anywhere

                                        nav_to_cal = float('inf')
                                        for cal_wp in cal_wps:
                                            if cal_wp in self.dist[rover_wp]:
                                                nav_to_cal = min(nav_to_cal, self.dist[rover_wp][cal_wp])
                                        if nav_to_cal == float('inf'): continue # Cannot reach any calibration waypoint

                                        current_image_nav_cost += nav_to_cal

                                    min_image_nav_cost = min(min_image_nav_cost, current_image_nav_cost)

                    if min_image_nav_cost == float('inf'): return float('inf') # Cannot take image
                    cost_g += min_image_nav_cost

                total_cost += cost_g

            # Add other goal types if necessary (though rovers domain goals are typically just communicated_*)

        return total_cost

