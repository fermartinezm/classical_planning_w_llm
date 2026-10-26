import collections

def parse_fact(fact_string):
    """
    Parses a PDDL fact string into a predicate and its arguments.
    e.g., '(at rover1 waypoint1)' -> ('at', ['rover1', 'waypoint1'])
    """
    # Removes parentheses and splits into parts
    parts = fact_string.strip("()").split()
    predicate = parts[0]
    args = parts[1:]
    return predicate, args

def bfs(start_node, graph):
    """
    Performs BFS to find shortest distances from start_node to all other nodes.
    Graph is represented as a dictionary: node -> set of neighbor_nodes.
    Returns a dictionary: node -> distance.
    Returns float('inf') for unreachable nodes.
    """
    distances = {start_node: 0}
    queue = collections.deque([start_node])
    visited = {start_node}
    while queue:
        current_node = queue.popleft()
        current_dist = distances[current_node]
        if current_node in graph: # Handle nodes with no outgoing edges
            for neighbor in graph[current_node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    distances[neighbor] = current_dist + 1
                    queue.append(neighbor)

    # Ensure all nodes in the graph have a distance entry (inf if unreachable)
    all_nodes = set(graph.keys())
    for neighbors in graph.values():
        all_nodes.update(neighbors)

    for node in all_nodes:
        if node not in distances:
            distances[node] = float('inf')

    return distances


class roversHeuristic:
    """
    Domain-dependent heuristic for the Rovers domain.

    Summary:
    The heuristic estimates the cost to reach the goal by summing up the
    estimated costs for each unachieved goal fact. The cost for each goal
    fact is estimated based on the sequence of actions required to achieve it
    (e.g., navigate, sample/image, communicate), considering the current state
    and precomputed navigation distances. It aims to be efficiently computable
    and informative for greedy best-first search.

    Assumptions:
    - Action costs are uniform (1).
    - The heuristic is not admissible; it may overestimate costs by summing
      independent goal costs and simplifying action sequences.
    - The heuristic assumes a simplified sequence of actions for each goal type.
    - For sampling goals, it assumes the sample is either at the original
      location or held by a rover. If the sample is gone from the original
      location and no rover has it, the goal is considered unreachable (high cost).
    - For imaging goals, it estimates the cost of navigating to calibration,
      imaging, and communication waypoints sequentially. It uses the minimum
      cost path through *any* suitable waypoints of each type. It does not
      account for recalibration needs after taking an image if the camera
      becomes uncalibrated mid-plan.
    - The heuristic uses precomputed shortest path distances between all pairs
      of waypoints for each rover, assuming the `can_traverse` predicates
      define the navigable graph.

    Heuristic Initialization:
    The constructor precomputes and stores static information from the task,
    including:
    - Lander location.
    - Rover equipment capabilities (soil, rock, imaging).
    - Store ownership (which store belongs to which rover).
    - Camera information (which rover it's on, supported modes, calibration target).
    - Waypoints visible from objectives (used for imaging and calibration targets).
    - Waypoint visibility graph (used to find communication points visible from the lander).
    - For each rover, the shortest path distances between all pairs of waypoints
      using BFS on the `can_traverse` graph specific to that rover.

    Step-By-Step Thinking for Computing Heuristic:
    1.  Initialize the total heuristic value `h` to 0.
    2.  Identify the set of goal facts that are not true in the current state.
    3.  Extract relevant dynamic information from the current state:
        - Current location of each rover.
        - Samples (soil/rock) and images currently held by rovers.
        - Remaining soil/rock sample locations on waypoints.
        - Store fullness status for each rover.
        - Calibration status of cameras.
    4.  Determine the set of waypoints visible from the lander's location (`CommWps`).
    5.  For each unachieved goal fact `g`:
        a.  If `g` is `(communicated_soil_data ?w)`:
            i.  Check if any rover `r` has `(have_soil_analysis ?r ?w)`.
            ii. If yes: Calculate the minimum cost for such a rover `r` to reach any waypoint in `CommWps` and communicate (1 action). Add this minimum cost to `h`.
            iii. If no:
                - Check if `(at_soil_sample ?w)` is true in the current state.
                - If yes: Calculate the minimum cost for an equipped rover `r` to reach `w`, sample (1 action, potentially 1 action for drop if store is full), reach a waypoint in `CommWps` (using the same rover), and communicate (1 action). Add this minimum cost to `h`.
                - If no: Add a large penalty (e.g., 1000) to `h`, indicating the sample is likely lost or the goal is unreachable from this state.
        b.  If `g` is `(communicated_rock_data ?w)`: Similar logic as soil data, using rock predicates and equipment.
        c.  If `g` is `(communicated_image_data ?o ?m)`:
            i.  Check if any rover `r` has `(have_image ?r ?o ?m)`.
            ii. If yes: Calculate the minimum cost for such a rover `r` to reach any waypoint in `CommWps` and communicate (1 action). Add this minimum cost to `h`.
            iii. If no:
                - Find suitable rovers `r` (equipped for imaging) and cameras `i` (on `r`, supporting `m`).
                - Find the set of calibration waypoints `CalWps(i)` for each suitable camera `i`.
                - Find the set of imaging waypoints `ImgWps(o)` for objective `o`.
                - If any required set is empty or no suitable `r, i` exist, add a large penalty (e.g., 1000) to `h`.
                - Otherwise: Calculate the minimum cost over all suitable `r, i`, all `w` in `CalWps(i)`, all `p` in `ImgWps(o)`, and all `x` in `CommWps` for the sequential path: navigate `at(r)` to `w`, calibrate (1), navigate `w` to `p`, take image (1), navigate `p` to `x`, communicate (1). The cost is `dist[r][at(r)][w] + 1 + dist[r][w][p] + 1 + dist[r][p][x] + 1`. Add this minimum cost to `h`.
    6.  Return the total heuristic value `h`.
    """

    UNREACHABLE_PENALTY = 1000

    def __init__(self, task):
        self.task = task
        self._precompute_static_info()
        self._precompute_distances()

    def _precompute_static_info(self):
        self.lander_wp = None
        self.rover_equipment = collections.defaultdict(set)
        self.rover_store = {}
        self.camera_info = {} # camera -> {'rover': r, 'modes': {m}, 'cal_target': o}
        self.objective_visible_from = collections.defaultdict(set)
        self.rover_can_traverse = collections.defaultdict(set)
        self.waypoint_visibility = collections.defaultdict(set)
        self.rovers = set()
        self.waypoints = set()
        self.objectives = set()
        self.cameras = set()
        self.modes = set()
        self.stores = set()
        self.landers = set()

        # Extract objects and static predicates
        # Consider facts from initial state as well, as they might contain objects/waypoints
        # not explicitly in static facts (though less common in STRIPS).
        # Also consider goal facts for objects/waypoints mentioned there.
        all_facts = self.task.initial_state | self.task.static | self.task.goals

        for fact_string in all_facts:
             pred, args = parse_fact(fact_string)
             if pred == 'at_lander':
                 self.landers.add(args[0])
                 self.waypoints.add(args[1])
                 self.lander_wp = args[1]
             elif pred == 'at':
                 self.rovers.add(args[0])
                 self.waypoints.add(args[1])
             elif pred == 'can_traverse':
                 self.rovers.add(args[0])
                 self.waypoints.add(args[1])
                 self.waypoints.add(args[2])
                 self.rover_can_traverse[args[0]].add((args[1], args[2]))
             elif pred == 'equipped_for_soil_analysis':
                 self.rovers.add(args[0])
                 self.rover_equipment[args[0]].add('soil')
             elif pred == 'equipped_for_rock_analysis':
                 self.rovers.add(args[0])
                 self.rover_equipment[args[0]].add('rock')
             elif pred == 'equipped_for_imaging':
                 self.rovers.add(args[0])
                 self.rover_equipment[args[0]].add('imaging')
             elif pred == 'store_of':
                 self.stores.add(args[0])
                 self.rovers.add(args[1])
                 self.rover_store[args[1]] = args[0] # Assuming one store per rover
             elif pred == 'supports':
                 self.cameras.add(args[0])
                 self.modes.add(args[1])
                 if args[0] not in self.camera_info:
                     self.camera_info[args[0]] = {'modes': set(), 'cal_target': None, 'rover': None}
                 self.camera_info[args[0]]['modes'].add(args[1])
             elif pred == 'visible':
                 self.waypoints.add(args[0])
                 self.waypoints.add(args[1])
                 self.waypoint_visibility[args[0]].add(args[1])
             elif pred == 'visible_from':
                 self.objectives.add(args[0])
                 self.waypoints.add(args[1])
                 self.objective_visible_from[args[0]].add(args[1])
             elif pred == 'calibration_target':
                 self.cameras.add(args[0])
                 self.objectives.add(args[1])
                 if args[0] not in self.camera_info:
                     self.camera_info[args[0]] = {'modes': set(), 'cal_target': None, 'rover': None}
                 self.camera_info[args[0]]['cal_target'] = args[1]
             elif pred == 'on_board':
                 self.cameras.add(args[0])
                 self.rovers.add(args[1])
                 if args[0] not in self.camera_info:
                     self.camera_info[args[0]] = {'modes': set(), 'cal_target': None, 'rover': None}
                 self.camera_info[args[0]]['rover'] = args[1]
             elif pred in ('empty', 'full'):
                 self.stores.add(args[0])
             elif pred in ('have_rock_analysis', 'have_soil_analysis'):
                 self.rovers.add(args[0])
                 self.waypoints.add(args[1])
             elif pred in ('have_image'):
                 self.rovers.add(args[0])
                 self.objectives.add(args[1])
                 self.modes.add(args[2])
             elif pred in ('communicated_soil_data', 'at_soil_sample', 'at_rock_sample', 'communicated_rock_data'):
                 self.waypoints.add(args[0])
             elif pred in ('communicated_image_data'):
                 self.objectives.add(args[0])
                 self.modes.add(args[1])

        # Build navigation graph for each rover
        self.rover_nav_graph = collections.defaultdict(lambda: collections.defaultdict(set))
        for rover, traverses in self.rover_can_traverse.items():
            for wp1, wp2 in traverses:
                self.rover_nav_graph[rover][wp1].add(wp2)

        # Determine communication waypoints
        self.comm_wps = set()
        if self.lander_wp:
             # Waypoints visible *from* the lander, or waypoints *from which* the lander is visible
             # The communicate action requires (visible ?x ?y) where rover is at ?x and lander at ?y
             # So we need waypoints ?x such that (visible ?x lander_wp) is true.
             for wp1, wp2_set in self.waypoint_visibility.items():
                 if self.lander_wp in wp2_set:
                     self.comm_wps.add(wp1)


    def _precompute_distances(self):
        self.rover_dist = collections.defaultdict(dict)
        # Ensure all waypoints are considered as potential start/end points for BFS
        all_relevant_waypoints = set(self.waypoints)
        for rover in self.rovers:
            graph = self.rover_nav_graph[rover]
            # Add any waypoints from the graph that weren't in self.waypoints initially
            for wp in graph:
                all_relevant_waypoints.add(wp)
            for wp_set in graph.values():
                 all_relevant_waypoints.update(wp_set)

            for start_wp in all_relevant_waypoints:
                self.rover_dist[rover][start_wp] = bfs(start_wp, graph)

    def get_distance(self, rover, wp1, wp2):
        """Returns shortest distance or infinity if unreachable for a specific rover."""
        if rover not in self.rover_dist or wp1 not in self.rover_dist[rover] or wp2 not in self.rover_dist[rover][wp1]:
            # If wp1 or wp2 wasn't in the graph used for BFS, distance is inf
            return float('inf')
        return self.rover_dist[rover][wp1][wp2]

    def get_min_dist_rover_to_set(self, rover, wp_set):
        """Returns min distance from rover's current location to any wp in set."""
        rover_loc = self.current_rover_locations.get(rover)
        if not rover_loc:
             return float('inf') # Rover location unknown or rover doesn't exist
        min_d = float('inf')
        for wp in wp_set:
            min_d = min(min_d, self.get_distance(rover, rover_loc, wp))
        return min_d

    def get_min_dist_set_to_set(self, rover, wp_set1, wp_set2):
        """Returns min distance from any wp in set1 to any wp in set2 for a rover."""
        min_d = float('inf')
        for wp1 in wp_set1:
            for wp2 in wp_set2:
                 min_d = min(min_d, self.get_distance(rover, wp1, wp2))
        return min_d


    def __call__(self, state):
        h = 0
        unachieved_goals = self.task.goals - state

        if not unachieved_goals:
            return 0 # Goal reached

        # Extract dynamic info from state
        self.current_rover_locations = {}
        self.current_have_soil = collections.defaultdict(set) # rover -> {wp}
        self.current_have_rock = collections.defaultdict(set) # rover -> {wp}
        self.current_have_image = collections.defaultdict(set) # rover -> {(obj, mode)}
        self.current_at_soil_sample = set() # {wp}
        self.current_at_rock_sample = set() # {wp}
        self.current_store_status = {} # rover -> 'empty' or 'full'
        # self.current_calibrated_cameras = collections.defaultdict(set) # rover -> {camera} # Not directly used in this heuristic logic

        for fact_string in state:
            pred, args = parse_fact(fact_string)
            if pred == 'at':
                self.current_rover_locations[args[0]] = args[1]
            elif pred == 'have_soil_analysis':
                self.current_have_soil[args[0]].add(args[1])
            elif pred == 'have_rock_analysis':
                self.current_have_rock[args[0]].add(args[1])
            elif pred == 'have_image':
                self.current_have_image[args[0]].add((args[1], args[2]))
            elif pred == 'at_soil_sample':
                self.current_at_soil_sample.add(args[0])
            elif pred == 'at_rock_sample':
                self.current_at_rock_sample.add(args[0])
            elif pred == 'empty':
                 store_name = args[0]
                 # Find the rover for this store
                 rover_name = None
                 for r, s in self.rover_store.items():
                     if s == store_name:
                         rover_name = r
                         break
                 if rover_name:
                     self.current_store_status[rover_name] = 'empty'
            elif pred == 'full':
                 store_name = args[0]
                 # Find the rover for this store
                 rover_name = None
                 for r, s in self.rover_store.items():
                     if s == store_name:
                         rover_name = r
                         break
                 if rover_name:
                     self.current_store_status[rover_name] = 'full'
            # elif pred == 'calibrated': # Not directly used in heuristic calculation
            #      self.current_calibrated_cameras[args[1]].add(args[0])


        # Calculate cost for each unachieved goal
        for goal_fact_string in unachieved_goals:
            pred, args = parse_fact(goal_fact_string)

            if pred == 'communicated_soil_data':
                wp = args[0]
                # Check if any rover has the sample
                rover_with_sample = None
                for r, wps in self.current_have_soil.items():
                    if wp in wps:
                        rover_with_sample = r
                        break

                if rover_with_sample:
                    # Sample exists, need to communicate it
                    min_comm_cost = self.get_min_dist_rover_to_set(rover_with_sample, self.comm_wps)
                    if min_comm_cost == float('inf'):
                         h += self.UNREACHABLE_PENALTY # Cannot reach communication point
                    else:
                         h += min_comm_cost + 1 # Navigate + Communicate
                else:
                    # Sample does not exist, need to sample and communicate
                    if wp in self.current_at_soil_sample:
                        # Sample is at the waypoint, need to collect it
                        min_collect_comm_cost = float('inf')
                        for rover in self.rovers:
                            if 'soil' in self.rover_equipment.get(rover, set()):
                                # Rover is equipped for soil analysis
                                rover_loc = self.current_rover_locations.get(rover)
                                if not rover_loc: continue # Rover location unknown

                                dist_to_sample = self.get_distance(rover, rover_loc, wp)
                                if dist_to_sample == float('inf'):
                                     continue # Cannot reach sample location

                                store_is_full = (self.current_store_status.get(rover) == 'full')
                                cost_to_collect = dist_to_sample + (1 if store_is_full else 0) + 1 # Navigate + (Drop) + Sample

                                # After sampling, need to communicate from wp using the same rover
                                min_dist_sample_to_comm = self.get_min_dist_set_to_set(rover, {wp}, self.comm_wps)
                                if min_dist_sample_to_comm == float('inf'):
                                     continue # Cannot reach communication point from sample location

                                cost_to_communicate = min_dist_sample_to_comm + 1 # Navigate + Communicate

                                min_collect_comm_cost = min(min_collect_comm_cost, cost_to_collect + cost_to_communicate)

                        if min_collect_comm_cost == float('inf'):
                             h += self.UNREACHABLE_PENALTY # Cannot collect and communicate
                        else:
                             h += min_collect_comm_cost
                    else:
                        # Sample is gone from waypoint and no rover has it
                        h += self.UNREACHABLE_PENALTY # Sample likely lost, goal unreachable

            elif pred == 'communicated_rock_data':
                wp = args[0]
                # Check if any rover has the sample
                rover_with_sample = None
                for r, wps in self.current_have_rock.items():
                    if wp in wps:
                        rover_with_sample = r
                        break

                if rover_with_sample:
                    # Sample exists, need to communicate it
                    min_comm_cost = self.get_min_dist_rover_to_set(rover_with_sample, self.comm_wps)
                    if min_comm_cost == float('inf'):
                         h += self.UNREACHABLE_PENALTY # Cannot reach communication point
                    else:
                         h += min_comm_cost + 1 # Navigate + Communicate
                else:
                    # Sample does not exist, need to sample and communicate
                    if wp in self.current_at_rock_sample:
                        # Sample is at the waypoint, need to collect it
                        min_collect_comm_cost = float('inf')
                        for rover in self.rovers:
                            if 'rock' in self.rover_equipment.get(rover, set()):
                                # Rover is equipped for rock analysis
                                rover_loc = self.current_rover_locations.get(rover)
                                if not rover_loc: continue # Rover location unknown

                                dist_to_sample = self.get_distance(rover, rover_loc, wp)
                                if dist_to_sample == float('inf'):
                                     continue # Cannot reach sample location

                                store_is_full = (self.current_store_status.get(rover) == 'full')
                                cost_to_collect = dist_to_sample + (1 if store_is_full else 0) + 1 # Navigate + (Drop) + Sample

                                # After sampling, need to communicate from wp using the same rover
                                min_dist_sample_to_comm = self.get_min_dist_set_to_set(rover, {wp}, self.comm_wps)
                                if min_dist_sample_to_comm == float('inf'):
                                     continue # Cannot reach communication point from sample location

                                cost_to_communicate = min_dist_sample_to_comm + 1 # Navigate + Communicate

                                min_collect_comm_cost = min(min_collect_comm_cost, cost_to_collect + cost_to_communicate)

                        if min_collect_comm_cost == float('inf'):
                             h += self.UNREACHABLE_PENALTY # Cannot collect and communicate
                        else:
                             h += min_collect_comm_cost
                    else:
                        # Sample is gone from waypoint and no rover has it
                        h += self.UNREACHABLE_PENALTY # Sample likely lost, goal unreachable


            elif pred == 'communicated_image_data':
                obj, mode = args[0], args[1]
                # Check if any rover has the image
                rover_with_image = None
                for r, images in self.current_have_image.items():
                    if (obj, mode) in images:
                        rover_with_image = r
                        break

                if rover_with_image:
                    # Image exists, need to communicate it
                    min_comm_cost = self.get_min_dist_rover_to_set(rover_with_image, self.comm_wps)
                    if min_comm_cost == float('inf'):
                         h += self.UNREACHABLE_PENALTY # Cannot reach communication point
                    else:
                         h += min_comm_cost + 1 # Navigate + Communicate
                else:
                    # Image does not exist, need to take image and communicate
                    min_image_comm_cost = float('inf')

                    # Find suitable rovers and cameras
                    suitable_rovers_cameras = [] # list of (rover, camera)
                    for rover in self.rovers:
                        if 'imaging' in self.rover_equipment.get(rover, set()):
                            for camera, info in self.camera_info.items():
                                if info['rover'] == rover and mode in info['modes']:
                                    suitable_rovers_cameras.append((rover, camera))

                    if not suitable_rovers_cameras:
                         h += self.UNREACHABLE_PENALTY # No suitable equipment
                         continue

                    # Find imaging waypoints for the objective
                    img_wps = self.objective_visible_from.get(obj, set())
                    if not img_wps:
                         h += self.UNREACHABLE_PENALTY # No waypoint to view objective
                         continue

                    # Find communication waypoints (precomputed)
                    if not self.comm_wps:
                         h += self.UNREACHABLE_PENALTY # No communication waypoint
                         continue


                    for rover, camera in suitable_rovers_cameras:
                        cal_target = self.camera_info[camera].get('cal_target')
                        if not cal_target:
                             continue # Camera has no calibration target

                        # Find calibration waypoints for the camera
                        cal_wps = self.objective_visible_from.get(cal_target, set())
                        if not cal_wps:
                             continue # No waypoint to view calibration target

                        # Calculate min cost for this rover/camera combination
                        # Path: at(r) -> w (cal) -> p (img) -> x (comm)
                        # Cost = dist(at(r), w) + 1 + dist(w, p) + 1 + dist(p, x) + 1

                        rover_loc = self.current_rover_locations.get(rover)
                        if not rover_loc:
                             continue # Rover location unknown

                        min_cost_for_rc = float('inf')

                        # Minimize over calibration waypoints 'w'
                        for w in cal_wps:
                            dist_r_to_w = self.get_distance(rover, rover_loc, w)
                            if dist_r_to_w == float('inf'): continue

                            # Minimize over imaging waypoints 'p' from 'w'
                            min_cost_w_to_comm = float('inf')
                            for p in img_wps:
                                dist_w_to_p = self.get_distance(rover, w, p)
                                if dist_w_to_p == float('inf'): continue

                                # Minimize over communication waypoints 'x' from 'p'
                                min_dist_p_to_comm = self.get_min_dist_set_to_set(rover, {p}, self.comm_wps)
                                if min_dist_p_to_comm == float('inf'): continue

                                cost_w_to_comm = dist_w_to_p + 1 + min_dist_p_to_comm + 1 # Nav w->p + Img + Nav p->x + Comm
                                min_cost_w_to_comm = min(min_cost_w_to_comm, cost_w_to_comm)

                            if min_cost_w_to_comm != float('inf'):
                                 cost_r_to_comm = dist_r_to_w + 1 + min_cost_w_to_comm # Nav at(r)->w + Cal + (Nav w->p + Img + Nav p->x + Comm)
                                 min_cost_for_rc = min(min_cost_for_rc, cost_r_to_comm)

                        min_image_comm_cost = min(min_image_comm_cost, min_cost_for_rc)


                    if min_image_comm_cost == float('inf'):
                         h += self.UNREACHABLE_PENALTY # Cannot complete image task
                    else:
                         h += min_image_comm_cost


        return h
