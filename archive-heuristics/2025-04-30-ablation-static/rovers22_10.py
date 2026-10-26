from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import collections

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return len(parts) == len(args) and all(fnmatch(part, arg) for part, arg in zip(parts, args))

def bfs(graph, start_node):
    """Computes shortest path distances from start_node to all reachable nodes."""
    distances = {node: float('inf') for node in graph}
    if start_node not in graph:
         # Start node might exist but have no outgoing edges in the graph dict
         # Add it to distances if it's a known node
         if start_node in distances:
             distances[start_node] = 0
         else:
             # If start_node is not even in the graph keys, it might be a waypoint
             # only mentioned in 'at' facts etc. We still need its distance to itself.
             distances[start_node] = 0
             # We cannot reach other nodes from here if it's not in the graph structure
             return distances # Return distances including start_node=0

    distances[start_node] = 0
    queue = collections.deque([start_node])

    while queue:
        current_node = queue.popleft()

        if current_node not in graph:
             continue # Should not happen if graph is built correctly with all nodes

        for neighbor in graph.get(current_node, []):
            if distances[neighbor] == float('inf'):
                distances[neighbor] = distances[current_node] + 1
                queue.append(neighbor)
    return distances

def compute_all_pairs_shortest_paths(graph, nodes):
    """Computes shortest path distances between all pairs of nodes."""
    all_paths = {}
    for start_node in nodes:
        all_paths[start_node] = bfs(graph, start_node)
    return all_paths


class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the minimum number of actions required to achieve
    all goal conditions. It sums the estimated costs for each unachieved goal
    independently. The cost for each goal is estimated by finding the minimum
    cost sequence of actions (navigation, sampling/imaging, communication)
    required to achieve that specific goal fact, considering rover capabilities
    and locations. Navigation costs are estimated using precomputed shortest
    paths on the traversal graph for each rover.

    # Assumptions:
    - Action costs are uniform (cost 1 per action).
    - The heuristic estimates the cost for each goal independently and sums them.
      This means it might overestimate if actions contribute to multiple goals
      (e.g., navigating to a waypoint allows sampling/imaging for multiple goals).
    - Rover traversal graphs are static and defined by `can_traverse` facts.
    - Lander location and waypoint visibility (`visible`, `visible_from`) are static.
    - Rover capabilities (`equipped_for_*`) and camera properties (`on_board`,
      `supports`, `calibration_target`) are static.
    - A rover is always located at exactly one waypoint (`at ?r ?w`).
    - If a sample or image is required for a goal, it exists at a known location
      (`at_soil_sample`, `at_rock_sample`, `visible_from`).

    # Heuristic Initialization
    - Parses static facts from the task definition.
    - Identifies all waypoints, rovers, cameras, objectives, landers, stores, modes.
    - Builds traversal graphs for each rover based on `can_traverse` facts.
    - Computes all-pairs shortest paths for each rover's traversal graph using BFS.
    - Stores static information like lander location, waypoint visibility,
      rover capabilities, store ownership, sample locations, objective visibility,
      and camera properties (on-board rover, supported modes, calibration target,
      and waypoints visible from the calibration target).

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Check if the state is a goal state. If yes, return 0.
    2. Parse the current state to extract dynamic information (rover locations,
       store status, samples/images held by rovers, camera calibration status,
       and communicated data).
    3. Initialize total heuristic cost to 0.
    4. Iterate through each goal fact defined in the task:
       a. If the goal fact is already true in the current state, add 0 to the total cost.
       b. If the goal fact is `(communicated_soil_data ?w)` and not true:
          - Estimate the minimum cost to achieve this goal across all capable rovers.
          - For each rover equipped for soil analysis:
            - If the rover already has the sample `(have_soil_analysis ?r ?w)`:
              - Cost is the estimated cost to communicate from the rover's current location.
            - If the rover does not have the sample:
              - Estimate the cost to navigate to waypoint `?w`, sample the soil (including potential drop action if store is full), and then communicate from `?w`.
            - The minimum of these costs over all capable rovers is the contribution for this goal. Add this minimum to the total cost.
       c. If the goal fact is `(communicated_rock_data ?w)` and not true:
          - Similar logic to soil data, using rock-specific predicates and actions.
       d. If the goal fact is `(communicated_image_data ?o ?m)` and not true:
          - Estimate the minimum cost to achieve this goal across all capable rovers, cameras, and suitable image waypoints.
          - For each rover equipped for imaging:
            - For each camera on board the rover supporting mode `?m`:
              - For each waypoint `?p` from which objective `?o` is visible:
                - If the rover already has the image `(have_image ?r ?o ?m)`:
                  - Cost is the estimated cost to communicate from the rover's current location.
                - If the rover does not have the image:
                  - Estimate the cost to navigate to a calibration waypoint (if camera not calibrated), calibrate, navigate to `?p`, take the image, and then communicate from `?p`.
                - The minimum of these costs over all combinations of rovers, cameras, and image waypoints is the contribution for this goal. Add this minimum to the total cost.
    5. Return the total heuristic cost.

    Navigation costs (`nav_cost`) are looked up in the precomputed shortest path tables.
    Communication costs (`comm_cost`) are estimated as the minimum navigation cost from the rover's location to any waypoint visible from the lander, plus 1 action for communication.
    Sampling costs (`sample_cost`) are estimated as navigation to the sample waypoint, plus 1 for sampling, plus 1 if a drop action is needed first.
    Image costs (`image_cost`) are estimated as navigation to the image waypoint, plus 1 for taking the image. If calibration is needed, it adds navigation to a calibration waypoint and 1 for calibration before navigating to the image waypoint.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.task = task # Store task for goal_reached check

        # Static information storage
        self.rover_capabilities = collections.defaultdict(set) # rover -> {soil, rock, imaging}
        self.rover_stores = {} # rover -> store
        self.lander_location = None # lander_wp
        self.waypoint_visibility = collections.defaultdict(set) # wp1 -> {wp2, ...} (visible wp1 wp2)
        self.rover_traversal_graphs = collections.defaultdict(set) # rover -> {wp1: {wp2, ...}} (can_traverse rover wp1 wp2)
        self.soil_sample_locations = set() # {wp, ...}
        self.rock_sample_locations = set() # {wp, ...}
        self.objective_visibility = collections.defaultdict(set) # objective -> {wp, ...} (visible_from objective wp)
        self.camera_info = collections.defaultdict(dict) # camera -> {on_board_rover: r, supports_modes: {m, ...}, calibration_target: obj}

        # Collect all waypoints mentioned in static facts and initial state
        all_waypoints = set()
        all_rovers = set()
        all_cameras = set()
        all_objectives = set()
        all_modes = set()
        all_stores = set()
        all_landers = set()

        def add_to_set(obj_set, obj_name):
             # Simple check if it looks like an object name (starts with letter)
             if isinstance(obj_name, str) and len(obj_name) > 0 and obj_name[0].isalpha():
                 obj_set.add(obj_name)

        for fact in task.static:
            parts = get_parts(fact)
            predicate = parts[0]
            args = parts[1:]

            if predicate == "can_traverse":
                r, wp1, wp2 = args
                self.rover_traversal_graphs[r].add((wp1, wp2))
                add_to_set(all_rovers, r)
                add_to_set(all_waypoints, wp1)
                add_to_set(all_waypoints, wp2)
            elif predicate == "at_lander":
                lander, wp = args
                self.lander_location = wp
                add_to_set(all_landers, lander)
                add_to_set(all_waypoints, wp)
            elif predicate == "visible":
                wp1, wp2 = args
                self.waypoint_visibility[wp1].add(wp2)
                add_to_set(all_waypoints, wp1)
                add_to_set(all_waypoints, wp2)
            elif predicate == "equipped_for_soil_analysis":
                r = args[0]
                self.rover_capabilities[r].add('soil')
                add_to_set(all_rovers, r)
            elif predicate == "equipped_for_rock_analysis":
                r = args[0]
                self.rover_capabilities[r].add('rock')
                add_to_set(all_rovers, r)
            elif predicate == "equipped_for_imaging":
                r = args[0]
                self.rover_capabilities[r].add('imaging')
                add_to_set(all_rovers, r)
            elif predicate == "store_of":
                s, r = args
                self.rover_stores[r] = s
                add_to_set(all_stores, s)
                add_to_set(all_rovers, r)
            elif predicate == "at_soil_sample":
                wp = args[0]
                self.soil_sample_locations.add(wp)
                add_to_set(all_waypoints, wp)
            elif predicate == "at_rock_sample":
                wp = args[0]
                self.rock_sample_locations.add(wp)
                add_to_set(all_waypoints, wp)
            elif predicate == "visible_from":
                obj, wp = args
                self.objective_visibility[obj].add(wp)
                add_to_set(all_objectives, obj)
                add_to_set(all_waypoints, wp)
            elif predicate == "calibration_target":
                cam, obj = args
                self.camera_info[cam]['calibration_target'] = obj
                add_to_set(all_cameras, cam)
                add_to_set(all_objectives, obj)
            elif predicate == "on_board":
                cam, r = args
                self.camera_info[cam]['on_board_rover'] = r
                add_to_set(all_cameras, cam)
                add_to_set(all_rovers, r)
            elif predicate == "supports":
                cam, mode = args
                if 'supports_modes' not in self.camera_info[cam]:
                    self.camera_info[cam]['supports_modes'] = set()
                self.camera_info[cam]['supports_modes'].add(mode)
                add_to_set(all_cameras, cam)
                add_to_set(all_modes, mode)

        # Also collect waypoints from initial state 'at' facts
        for fact in task.initial_state:
             parts = get_parts(fact)
             if parts[0] == "at":
                 obj, wp = parts[1:]
                 # Assuming objects at waypoints are rovers
                 add_to_set(all_rovers, obj)
                 add_to_set(all_waypoints, wp)
             elif parts[0] == "at_lander":
                 lander, wp = parts[1:]
                 add_to_set(all_landers, lander)
                 add_to_set(all_waypoints, wp)
             elif parts[0] == "store_of":
                 s, r = parts[1:]
                 add_to_set(all_stores, s)
                 add_to_set(all_rovers, r)
             elif parts[0] == "on_board":
                 cam, r = parts[1:]
                 add_to_set(all_cameras, cam)
                 add_to_set(all_rovers, r)


        self.all_waypoints = list(all_waypoints) # Use a list for consistent ordering if needed, though dict lookup is primary
        self.all_rovers = list(all_rovers)
        self.all_cameras = list(all_cameras)
        self.all_objectives = list(all_objectives)
        self.all_modes = list(all_modes)
        self.all_stores = list(all_stores)
        self.all_landers = list(all_landers)


        # Build adjacency list graph for each rover
        self.rover_graphs_adj = collections.defaultdict(lambda: collections.defaultdict(list))
        for rover, edges in self.rover_traversal_graphs.items():
            for wp1, wp2 in edges:
                self.rover_graphs_adj[rover][wp1].append(wp2)
                # Assuming traversal is bidirectional if not specified otherwise?
                # PDDL says can_traverse ?r ?x ?y. If (can_traverse r w1 w2) and (can_traverse r w2 w1) exist, it's bidirectional.
                # Let's stick strictly to defined can_traverse facts.
                # self.rover_graphs_adj[rover][wp2].append(wp1) # Only if bidirectional is implied or explicitly stated

        # Ensure all waypoints are keys in the graph adjacency list, even if they have no outgoing edges
        for rover in self.all_rovers:
             for wp in self.all_waypoints:
                 if wp not in self.rover_graphs_adj[rover]:
                     self.rover_graphs_adj[rover][wp] = []


        # Compute all-pairs shortest paths for each rover
        self.rover_shortest_paths = {}
        for rover in self.all_rovers:
            self.rover_shortest_paths[rover] = compute_all_pairs_shortest_paths(
                self.rover_graphs_adj[rover], self.all_waypoints
            )

        # Compute waypoints visible from the lander location
        self.visible_from_lander_wps = set()
        if self.lander_location and self.lander_location in self.waypoint_visibility:
             # Waypoints FROM which lander location is visible
             # We need waypoints TO which rover can navigate and communicate FROM
             # This means waypoints X such that (visible X lander_wp) is true
             # We need to reverse the visible graph or iterate all pairs
             for wp1, visible_wps in self.waypoint_visibility.items():
                 if self.lander_location in visible_wps:
                     self.visible_from_lander_wps.add(wp1)


        # Compute waypoints visible from each camera's calibration target
        self.calibration_waypoint_visibility = collections.defaultdict(set) # camera -> {wp, ...}
        for camera, info in self.camera_info.items():
            cal_target = info.get('calibration_target')
            if cal_target and cal_target in self.objective_visibility:
                self.calibration_waypoint_visibility[camera] = self.objective_visibility[cal_target]


    def nav_cost(self, rover, start_wp, end_wp):
        """Returns shortest path distance for rover."""
        if rover not in self.rover_shortest_paths or \
           start_wp not in self.rover_shortest_paths[rover] or \
           end_wp not in self.rover_shortest_paths[rover][start_wp]:
             # Should not happen if all waypoints are included in precomputation
             return float('inf')
        return self.rover_shortest_paths[rover][start_wp][end_wp]

    def comm_cost(self, rover, data_location_wp):
        """Estimates cost to communicate data from data_location_wp."""
        if not self.lander_location or not self.visible_from_lander_wps:
            return float('inf') # Cannot communicate if no lander or no visible points

        curr_wp = data_location_wp
        comm_wps = self.visible_from_lander_wps

        if curr_wp in comm_wps:
            return 1 # Communicate action

        min_nav = float('inf')
        for comm_wp in comm_wps:
            min_nav = min(min_nav, self.nav_cost(rover, curr_wp, comm_wp))

        if min_nav == float('inf'):
            return float('inf') # Cannot reach any communication waypoint
        else:
            return min_nav + 1 # Navigate + Communicate

    def sample_cost(self, rover, waypoint, state_info):
        """Estimates cost to sample at waypoint."""
        curr_wp = state_info['rover_locations'].get(rover)
        if curr_wp is None: return float('inf'), None # Rover location unknown

        store = self.rover_stores.get(rover)
        if store is None: return float('inf'), None # Rover has no store

        store_full = (state_info['store_status'].get(store) == 'full')

        nav = self.nav_cost(rover, curr_wp, waypoint)
        if nav == float('inf'): return float('inf'), None

        cost = nav + (1 if store_full else 0) + 1 # nav + (drop) + sample
        return cost, waypoint # Return cost and location after sampling

    def image_cost(self, rover, camera, objective, mode, image_wp, state_info):
        """Estimates cost to take image at image_wp."""
        curr_wp = state_info['rover_locations'].get(rover)
        if curr_wp is None: return float('inf'), None # Rover location unknown

        is_calibrated = state_info['camera_calibration'].get(camera, False)

        if is_calibrated:
            nav_to_image_wp = self.nav_cost(rover, curr_wp, image_wp)
            if nav_to_image_wp == float('inf'): return float('inf'), None
            cost_to_image = nav_to_image_wp + 1 # nav + take_image
            return cost_to_image, image_wp
        else: # not calibrated
            cal_target = self.camera_info.get(camera, {}).get('calibration_target')
            if not cal_target: return float('inf'), None # Camera has no calibration target

            cal_wps = self.calibration_waypoint_visibility.get(camera, set())
            if not cal_wps: return float('inf'), None # No waypoints visible from calibration target

            min_nav_to_cal = float('inf')
            best_cal_wp = None
            for cal_wp in cal_wps:
                nav = self.nav_cost(rover, curr_wp, cal_wp)
                if nav < min_nav_to_cal:
                    min_nav_to_cal = nav
                    best_cal_wp = cal_wp

            if min_nav_to_cal == float('inf'): return float('inf'), None # Cannot reach any cal waypoint

            nav_cal_to_image = self.nav_cost(rover, best_cal_wp, image_wp)
            if nav_cal_to_image == float('inf'): return float('inf'), None # Cannot reach image wp from cal wp

            cost_to_image = min_nav_to_cal + 1 + nav_cal_to_image + 1 # nav_to_cal + calibrate + nav_cal_to_image + take_image
            return cost_to_image, image_wp

    def parse_state(self, state):
        """Parses state facts into usable dictionaries/sets."""
        state_info = {
            'rover_locations': {},
            'store_status': {}, # store -> 'empty' or 'full'
            'rover_samples': collections.defaultdict(lambda: {'soil': set(), 'rock': set()}), # rover -> {'soil': {wp}, 'rock': {wp}}
            'rover_images': collections.defaultdict(set), # rover -> {(obj, mode), ...}
            'camera_calibration': {}, # camera -> True/False
            'communicated_soil': set(), # {wp, ...}
            'communicated_rock': set(), # {wp, ...}
            'communicated_images': set(), # {(obj, mode), ...}
        }

        # Initialize all stores as full or empty if mentioned in state
        for store in self.all_stores:
             state_info['store_status'][store] = 'unknown' # Default

        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]
            args = parts[1:]

            if predicate == "at":
                obj, loc = args
                if obj in self.all_rovers: # Check if it's a known rover
                     state_info['rover_locations'][obj] = loc
            elif predicate == "empty":
                store = args[0]
                state_info['store_status'][store] = 'empty'
            elif predicate == "full":
                store = args[0]
                state_info['store_status'][store] = 'full'
            elif predicate == "have_soil_analysis":
                rover, wp = args
                state_info['rover_samples'][rover]['soil'].add(wp)
            elif predicate == "have_rock_analysis":
                rover, wp = args
                state_info['rover_samples'][rover]['rock'].add(wp)
            elif predicate == "have_image":
                rover, obj, mode = args
                state_info['rover_images'][rover].add((obj, mode))
            elif predicate == "calibrated":
                camera, rover = args
                state_info['camera_calibration'][camera] = True
            elif predicate == "communicated_soil_data":
                wp = args[0]
                state_info['communicated_soil'].add(wp)
            elif predicate == "communicated_rock_data":
                wp = args[0]
                state_info['communicated_rock'].add(wp)
            elif predicate == "communicated_image_data":
                obj, mode = args
                state_info['communicated_images'].add((obj, mode))

        return state_info


    def estimate_soil_goal_cost(self, state_info, waypoint):
        """Estimates cost for a single unachieved soil communication goal."""
        min_total_cost = float('inf')

        # Find rovers equipped for soil analysis
        capable_rovers = [r for r in self.all_rovers if 'soil' in self.rover_capabilities.get(r, set())]

        if not capable_rovers:
            return float('inf') # No rover can perform this task

        # Check if soil sample exists at the waypoint
        if waypoint not in self.soil_sample_locations:
             # This goal is unreachable if the sample doesn't exist initially
             # (assuming samples don't reappear)
             # However, the problem might start with samples already taken.
             # We need to check if the sample was *originally* there.
             # This requires parsing initial state or problem definition fully.
             # For simplicity, let's assume if the goal exists, the sample was initially there.
             # A more robust heuristic would check task.initial_state for (at_soil_sample waypoint).
             pass # Assume sample was/is there if goal exists

        for rover in capable_rovers:
            # Cost to get the sample
            if waypoint in state_info['rover_samples'].get(rover, {}).get('soil', set()):
                # Rover already has the sample
                sample_c = 0
                sample_loc = state_info['rover_locations'].get(rover) # Communicate from current location
                if sample_loc is None: continue # Cannot estimate if rover location unknown
            else:
                # Need to sample
                sample_cost_result = self.sample_cost(rover, waypoint, state_info)
                sample_c, sample_loc = sample_cost_result
                if sample_c == float('inf'): continue # Cannot get sample

            # Cost to communicate
            comm_c = self.comm_cost(rover, sample_loc) # Communicate from sample location
            if comm_c == float('inf'): continue # Cannot communicate

            total_c = sample_c + comm_c
            min_total_cost = min(min_total_cost, total_c)

        return min_total_cost

    def estimate_rock_goal_cost(self, state_info, waypoint):
        """Estimates cost for a single unachieved rock communication goal."""
        min_total_cost = float('inf')

        # Find rovers equipped for rock analysis
        capable_rovers = [r for r in self.all_rovers if 'rock' in self.rover_capabilities.get(r, set())]

        if not capable_rovers:
            return float('inf') # No rover can perform this task

        # Check if rock sample exists at the waypoint (similar assumption as soil)
        if waypoint not in self.rock_sample_locations:
             pass # Assume sample was/is there if goal exists

        for rover in capable_rovers:
            # Cost to get the sample
            if waypoint in state_info['rover_samples'].get(rover, {}).get('rock', set()):
                # Rover already has the sample
                sample_c = 0
                sample_loc = state_info['rover_locations'].get(rover) # Communicate from current location
                if sample_loc is None: continue # Cannot estimate if rover location unknown
            else:
                # Need to sample
                # sample_cost uses soil_sample_locations, need a rock version or make it generic
                # Let's make sample_cost generic
                sample_cost_result = self.sample_cost(rover, waypoint, state_info) # sample_cost is generic enough
                sample_c, sample_loc = sample_cost_result
                if sample_c == float('inf'): continue # Cannot get sample

            # Cost to communicate
            comm_c = self.comm_cost(rover, sample_loc) # Communicate from sample location
            if comm_c == float('inf'): continue # Cannot communicate

            total_c = sample_c + comm_c
            min_total_cost = min(min_total_cost, total_c)

        return min_total_cost


    def estimate_image_goal_cost(self, state_info, objective, mode):
        """Estimates cost for a single unachieved image communication goal."""
        min_total_cost = float('inf')

        # Find rovers equipped for imaging
        capable_rovers = [r for r in self.all_rovers if 'imaging' in self.rover_capabilities.get(r, set())]

        if not capable_rovers:
            return float('inf') # No rover can perform this task

        # Find waypoints visible from the objective
        image_wps = self.objective_visibility.get(objective, set())
        if not image_wps:
            return float('inf') # Objective not visible from anywhere

        for rover in capable_rovers:
            # Find cameras on this rover supporting this mode
            suitable_cameras = [
                cam for cam in self.all_cameras
                if self.camera_info.get(cam, {}).get('on_board_rover') == rover
                and mode in self.camera_info.get(cam, {}).get('supports_modes', set())
            ]
            if not suitable_cameras: continue # Rover has no suitable camera

            for camera in suitable_cameras:
                for image_wp in image_wps:
                    # Cost to get the image
                    if (objective, mode) in state_info['rover_images'].get(rover, set()):
                        # Rover already has the image
                        image_c = 0
                        image_loc = state_info['rover_locations'].get(rover) # Communicate from current location
                        if image_loc is None: continue # Cannot estimate if rover location unknown
                    else:
                        # Need to take image
                        image_cost_result = self.image_cost(rover, camera, objective, mode, image_wp, state_info)
                        image_c, image_loc = image_cost_result
                        if image_c == float('inf'): continue # Cannot take image

                    # Cost to communicate
                    comm_c = self.comm_cost(rover, image_loc) # Communicate from image location
                    if comm_c == float('inf'): continue # Cannot communicate

                    total_c = image_c + comm_c
                    min_total_cost = min(min_total_cost, total_c)

        return min_total_cost


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Check if goal reached
        if self.task.goal_reached(state):
            return 0

        # Parse state facts into usable structures
        state_info = self.parse_state(state)

        total_cost = 0

        # Iterate through goals
        for goal in self.task.goals:
            predicate, *args = get_parts(goal)

            if predicate == "communicated_soil_data":
                wp = args[0]
                if wp not in state_info['communicated_soil']:
                    cost = self.estimate_soil_goal_cost(state_info, wp)
                    total_cost += cost

            elif predicate == "communicated_rock_data":
                wp = args[0]
                if wp not in state_info['communicated_rock']:
                    cost = self.estimate_rock_goal_cost(state_info, wp)
                    total_cost += cost

            elif predicate == "communicated_image_data":
                objective, mode = args
                if (objective, mode) not in state_info['communicated_images']:
                    cost = self.estimate_image_goal_cost(state_info, objective, mode)
                    total_cost += cost

        # If any required goal is unreachable, the total cost will be infinity
        # We might want to return a large finite number instead of infinity
        # depending on the planner's requirements, but float('inf') is standard.
        return total_cost

