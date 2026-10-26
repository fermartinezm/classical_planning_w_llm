import collections
from fnmatch import fnmatch
# Assuming heuristic_base is available in the execution environment
# from heuristics.heuristic_base import Heuristic

# Define a dummy Heuristic base class if not provided in the environment
# This is just for standalone testing or if the base class is simple.
# In the actual planning environment, the real Heuristic class will be used.
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    # Define a simple dummy class that matches the expected interface
    class Heuristic:
        def __init__(self, task):
            """Initializes the heuristic with task information."""
            self.task = task # Store task for potential use

        def __call__(self, node):
            """Computes the heuristic value for a given state node."""
            # This is a placeholder; actual heuristics override this.
            return 0 # Default to zero heuristic


# Helper functions (copied from Logistics example, slightly adapted)
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty strings or malformed facts gracefully
    if not fact or not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def bfs(graph, start_node, all_nodes):
    """
    Perform BFS to find shortest distances from start_node to all other nodes.

    Args:
        graph: Adjacency list representation {node: [neighbor1, neighbor2, ...]}
        start_node: The starting node for BFS.
        all_nodes: A set of all nodes in the graph.

    Returns:
        A dictionary {node: distance} from start_node to all reachable nodes.
        Distance is float('inf') for unreachable nodes.
    """
    distances = {node: float('inf') for node in all_nodes}
    if start_node not in all_nodes:
        # Start node is not in the graph, no paths possible
        return distances

    distances[start_node] = 0
    queue = collections.deque([start_node])

    while queue:
        current_node = queue.popleft()

        # Ensure current_node is in graph keys before accessing neighbors
        if current_node in graph:
            for neighbor in graph[current_node]:
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)

    return distances


class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the minimum number of actions required to achieve
    each uncommunicated goal (soil data, rock data, image data) independently,
    and sums these minimum costs. It considers whether the required data
    (sample or image) is already held by a rover or needs to be acquired,
    and the navigation cost to communication locations.

    # Assumptions
    - The cost of each action (navigate, sample, drop, calibrate, take_image, communicate) is 1.
    - Waypoint traversal costs 1 per step. Shortest paths are precomputed based on
      visible and traversable links.
    - If a soil/rock sample is no longer at its original waypoint, it must be held by a rover
      for the corresponding communication goal to be reachable in a solvable problem.
    - For image goals, a suitable waypoint exists from which both the objective and
      the calibration target for a suitable camera are visible in a solvable problem.
    - Calibration is assumed to cost 1 action whenever an image needs to be taken
      as taking an image consumes calibration.
    - The heuristic assumes solvable problems; unreachable goals due to missing samples/locations
      or impossible navigation will result in an infinite heuristic value.

    # Heuristic Initialization
    - Parses static facts to identify objects (rovers, waypoints, cameras, etc.).
    - Builds the waypoint graph based on `visible` and `can_traverse` facts, considering
      an edge exists if visible and traversable by at least one rover.
    - Precomputes all-pairs shortest paths between waypoints using BFS.
    - Identifies lander locations and communication waypoints (visible from landers).
    - Extracts static information about rovers (capabilities, store), cameras (on board, supports, calibration target),
      and objective/calibration target visibility from waypoints.

    # Step-By-Step Thinking for Computing Heuristic
    1.  Initialize total heuristic cost `h = 0`.
    2.  Parse the current state to determine dynamic facts: rover locations, store status,
        samples/images held by rovers, samples/rocks remaining at waypoints, camera calibration status.
    3.  Identify all unachieved goals from the task's goal set.
    4.  Group unachieved goals by the data they require: soil samples at specific waypoints,
        rock samples at specific waypoints, and images of specific objectives in specific modes.
    5.  For each waypoint `?w` requiring soil data communication (unachieved goal `(communicated_soil_data ?w)`):
        - Calculate the minimum cost to get the sample data for `?w` to a communication location.
        - This minimum cost is calculated considering two possibilities:
            a) A rover already has the sample `(have_soil_analysis ?r ?w)`. Find the minimum cost for such a rover to navigate from its current location to any communication waypoint and communicate (distance + 1).
            b) The sample is still at the waypoint `(at_soil_sample ?w)`. Find the minimum cost for a soil-equipped rover to navigate to `?w`, sample it (cost 1, plus 1 if store is full for drop), and then navigate from `?w` to any communication waypoint and communicate (distance + 1).
        - The cost for this soil goal is the minimum of costs from (a) and (b). If neither is possible (e.g., sample gone and not at waypoint), the cost is infinity.
        - Add this minimum cost to `h`.
    6.  Repeat step 5 for each waypoint `?w` requiring rock data communication.
    7.  For each objective-mode pair `(?o, ?m)` requiring image data communication (unachieved goal `(communicated_image_data ?o ?m)`):
        - Calculate the minimum cost to get the image data for `(?o, ?m)` to a communication location.
        - This minimum cost is calculated considering two possibilities:
            a) A rover already has the image `(have_image ?r ?o ?m)`. Find the minimum cost for such a rover to navigate from its current location to any communication waypoint and communicate (distance + 1).
            b) No rover has the image. Find the minimum cost for an imaging-equipped rover with a suitable camera to navigate to a waypoint `?p` visible from `?o` and the camera's calibration target, calibrate the camera (cost 1), take the image (cost 1), and then navigate from `?p` to any communication waypoint and communicate (distance + 1). This involves finding the best rover, camera, and waypoint `?p`.
        - The cost for this image goal is the minimum of costs from (a) and (b). If neither is possible, the cost is infinity.
        - Add this minimum cost to `h`.
    8.  Return the total accumulated cost `h`. If any required data item resulted in an infinite cost, return infinity.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # --- Precompute Static Information ---

        # Infer objects and types from predicates involving typed objects
        self.rovers = set()
        self.waypoints = set()
        self.stores = set()
        self.cameras = set()
        self.modes = set()
        self.landers = set()
        self.objectives = set()

        for fact in static_facts:
             parts = get_parts(fact)
             if not parts: continue
             predicate = parts[0]
             if predicate in ('at', 'can_traverse', 'equipped_for_soil_analysis',
                             'equipped_for_rock_analysis', 'equipped_for_imaging',
                             'store_of', 'on_board'):
                 if len(parts) > 1: self.rovers.add(parts[1])
             if predicate in ('at', 'at_lander', 'can_traverse', 'visible',
                             'at_soil_sample', 'at_rock_sample', 'visible_from'):
                 if len(parts) > 2: self.waypoints.add(parts[2])
                 # Waypoints can also be the second argument in 'visible' and 'at_lander'
                 if predicate in ('visible', 'at_lander') and len(parts) > 1:
                      self.waypoints.add(parts[1])
             if predicate in ('empty', 'full', 'store_of'):
                 if len(parts) > 1: self.stores.add(parts[1])
             if predicate in ('calibrated', 'supports', 'calibration_target', 'on_board'):
                 if len(parts) > 1: self.cameras.add(parts[1])
             if predicate in ('supports', 'have_image', 'communicated_image_data'):
                 if len(parts) > 2: self.modes.add(parts[2]) # mode is 3rd arg in supports/have_image/communicated_image_data
                 # mode is 4th arg in take_image, 5th in communicate_image_data - handled by get_parts correctly
             if predicate in ('calibration_target', 'visible_from', 'have_image',
                             'communicated_image_data'):
                 if len(parts) > 2: self.objectives.add(parts[2]) # obj is 3rd arg
                 # obj is 4th arg in take_image - handled by get_parts correctly
             if predicate == 'at_lander':
                 if len(parts) > 1: self.landers.add(parts[1])

        # Ensure all objects mentioned in goals are included, especially if they weren't in static init
        for goal in self.goals:
             parts = get_parts(goal)
             if not parts: continue
             predicate = parts[0]
             if predicate == 'communicated_soil_data' and len(parts) > 1:
                  self.waypoints.add(parts[1])
             elif predicate == 'communicated_rock_data' and len(parts) > 1:
                  self.waypoints.add(parts[1])
             elif predicate == 'communicated_image_data' and len(parts) > 2:
                  self.objectives.add(parts[1])
                  self.modes.add(parts[2])


        # Build waypoint graph based on visible and can_traverse facts
        graph = {wp: [] for wp in self.waypoints}
        # Collect all traversable edges (visible AND can_traverse by at least one rover)
        can_traverse_map = collections.defaultdict(set) # (w1, w2) -> {rover}
        for fact in static_facts:
            if match(fact, "can_traverse", "*", "*", "*"):
                _, r, w1, w2 = get_parts(fact)
                if w1 in self.waypoints and w2 in self.waypoints:
                     can_traverse_map[(w1, w2)].add(r)

        for fact in static_facts:
            if match(fact, "visible", "*", "*"):
                _, w1, w2 = get_parts(fact)
                if (w1, w2) in can_traverse_map: # Check if visible link is traversable by any rover
                     if w1 in graph and w2 in graph:
                         graph[w1].append(w2)

        # Remove duplicates from graph adjacency lists
        for wp in graph:
            graph[wp] = list(set(graph[wp]))

        # Precompute shortest paths
        self.dist = {}
        for start_wp in self.waypoints:
            self.dist[start_wp] = bfs(graph, start_wp, self.waypoints)

        # Identify lander locations (assuming static)
        self.lander_locs = {get_parts(fact)[2] for fact in static_facts if match(fact, "at_lander", "*", "*")}

        # Identify communication waypoints (visible from any lander location)
        self.comm_locs = set()
        for lander_loc in self.lander_locs:
            for fact in static_facts:
                if match(fact, "visible", "*", lander_loc):
                    comm_wp = get_parts(fact)[1]
                    if comm_wp in self.waypoints: # Ensure it's a known waypoint
                         self.comm_locs.add(comm_wp)


        # Store static info for quick lookup
        self.rover_capabilities = {}
        for rover in self.rovers:
            self.rover_capabilities[rover] = {
                'soil': f'(equipped_for_soil_analysis {rover})' in static_facts,
                'rock': f'(equipped_for_rock_analysis {rover})' in static_facts,
                'imaging': f'(equipped_for_imaging {rover})' in static_facts,
            }

        self.rover_to_store = {}
        self.store_to_rover = {}
        for fact in static_facts:
            if match(fact, "store_of", "*", "*"):
                _, store, rover = get_parts(fact)
                self.store_to_rover[store] = rover
                self.rover_to_store[rover] = store

        self.camera_on_rover = {}
        self.camera_modes = {}
        self.camera_cal_target = {}
        for camera in self.cameras:
            self.camera_modes[camera] = set()
            for fact in static_facts:
                if match(fact, "on_board", camera, "*"):
                    self.camera_on_rover[camera] = get_parts(fact)[2]
                elif match(fact, "supports", camera, "*"):
                    self.camera_modes[camera].add(get_parts(fact)[2])
                elif match(fact, "calibration_target", camera, "*"):
                    self.camera_cal_target[camera] = get_parts(fact)[2]

        self.objective_visibility = {}
        for objective in self.objectives:
            self.objective_visibility[objective] = set()
            for fact in static_facts:
                 if match(fact, "visible_from", objective, "*"):
                      wp = get_parts(fact)[2]
                      if wp in self.waypoints: # Ensure it's a known waypoint
                           self.objective_visibility[objective].add(wp)

        self.calibration_target_visibility = {}
        # Calibration targets are objectives. Reuse objective_visibility.
        # Map cal target name (objective name) to waypoints it's visible from.
        for cam, target_obj in self.camera_cal_target.items():
             if target_obj in self.objective_visibility:
                  self.calibration_target_visibility[target_obj] = self.objective_visibility[target_obj]
             else:
                  self.calibration_target_visibility[target_obj] = set() # Should not happen in solvable problems


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # --- Parse Dynamic State Information ---
        rover_locs = {} # {rover: waypoint}
        store_full = {} # {rover: True/False}
        has_soil = collections.defaultdict(lambda: collections.defaultdict(bool)) # {rover: {waypoint: True/False}}
        has_rock = collections.defaultdict(lambda: collections.defaultdict(bool)) # {rover: {waypoint: True/False}}
        has_image = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(bool))) # {rover: {objective: {mode: True/False}}}
        soil_at_w = collections.defaultdict(bool) # {waypoint: True/False}
        rock_at_w = collections.defaultdict(bool) # {waypoint: True/False}
        is_calibrated = collections.defaultdict(lambda: collections.defaultdict(bool)) # {camera: {rover: True/False}}

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue
            predicate = parts[0]
            if predicate == 'at' and len(parts) > 2 and parts[1] in self.rovers and parts[2] in self.waypoints:
                rover_locs[parts[1]] = parts[2]
            elif predicate == 'full' and len(parts) > 1 and parts[1] in self.stores:
                rover = self.store_to_rover.get(parts[1])
                if rover: store_full[rover] = True
            elif predicate == 'empty' and len(parts) > 1 and parts[1] in self.stores:
                 rover = self.store_to_rover.get(parts[1])
                 if rover: store_full[rover] = False
            elif predicate == 'have_soil_analysis' and len(parts) > 2 and parts[1] in self.rovers and parts[2] in self.waypoints:
                has_soil[parts[1]][parts[2]] = True
            elif predicate == 'have_rock_analysis' and len(parts) > 2 and parts[1] in self.rovers and parts[2] in self.waypoints:
                has_rock[parts[1]][parts[2]] = True
            elif predicate == 'have_image' and len(parts) > 3 and parts[1] in self.rovers and parts[2] in self.objectives and parts[3] in self.modes:
                has_image[parts[1]][parts[2]][parts[3]] = True
            elif predicate == 'at_soil_sample' and len(parts) > 1 and parts[1] in self.waypoints:
                soil_at_w[parts[1]] = True
            elif predicate == 'at_rock_sample' and len(parts) > 1 and parts[1] in self.waypoints:
                rock_at_w[parts[1]] = True
            elif predicate == 'calibrated' and len(parts) > 2 and parts[1] in self.cameras and parts[2] in self.rovers:
                is_calibrated[parts[1]][parts[2]] = True

        # Initialize store_full for rovers that might not have a store fact (implies empty)
        for rover in self.rovers:
             if rover not in store_full:
                  store_full[rover] = False # Default to empty if no state fact found


        # --- Calculate Heuristic ---
        h = 0
        unachieved_goals = self.goals - state

        # Group goals by type and target
        needed_soil_waypoints = {get_parts(g)[1] for g in unachieved_goals if match(g, "communicated_soil_data", "*")}
        needed_rock_waypoints = {get_parts(g)[1] for g in unachieved_goals if match(g, "communicated_rock_data", "*")}
        needed_images = {(get_parts(g)[1], get_parts(g)[2]) for g in unachieved_goals if match(g, "communicated_image_data", "*", "*")}

        # Cost for Soil Data Goals
        for w in needed_soil_waypoints:
            cost_w = float('inf')
            rovers_with_sample = {r for r in self.rovers if has_soil[r][w]}

            # Option 1: Communicate sample already held by a rover
            if rovers_with_sample:
                min_comm_cost = float('inf')
                for r in rovers_with_sample:
                    current_loc = rover_locs.get(r)
                    if current_loc and current_loc in self.dist:
                        for comm_loc in self.comm_locs:
                            if comm_loc in self.dist[current_loc]:
                                dist_to_comm = self.dist[current_loc][comm_loc]
                                if dist_to_comm != float('inf'):
                                    min_comm_cost = min(min_comm_cost, dist_to_comm + 1) # Navigate + Communicate
                if min_comm_cost != float('inf'):
                    cost_w = min(cost_w, min_comm_cost)

            # Option 2: Sample at waypoint, then communicate
            if soil_at_w[w]:
                soil_rovers = {r for r in self.rovers if self.rover_capabilities[r]['soil']}
                min_sample_comm_cost = float('inf')
                for r in soil_rovers:
                    current_loc = rover_locs.get(r)
                    if current_loc and current_loc in self.dist and w in self.dist[current_loc]:
                        dist_to_sample = self.dist[current_loc][w]
                        if dist_to_sample != float('inf'):
                            cost_r = dist_to_sample # Navigate to sample
                            cost_r += (1 if store_full.get(r, False) else 0) # Drop if needed
                            cost_r += 1 # Sample

                            # Now at w with sample, need to communicate from w
                            min_comm_cost_from_w = float('inf')
                            if w in self.dist: # Ensure w is a valid start node for BFS
                                for comm_loc in self.comm_locs:
                                    if comm_loc in self.dist[w]:
                                        dist_w_to_comm = self.dist[w][comm_loc]
                                        if dist_w_to_comm != float('inf'):
                                            min_comm_cost_from_w = min(min_comm_cost_from_w, dist_w_to_comm + 1) # Navigate + Communicate

                            if min_comm_cost_from_w != float('inf'):
                                min_sample_comm_cost = min(min_sample_comm_cost, cost_r + min_comm_cost_from_w)

                if min_sample_comm_cost != float('inf'):
                    cost_w = min(cost_w, min_sample_comm_cost)

            if cost_w == float('inf'):
                 # This goal is unreachable from this state
                 return float('inf')
            h += cost_w

        # Cost for Rock Data Goals (similar to Soil)
        for w in needed_rock_waypoints:
            cost_w = float('inf')
            rovers_with_sample = {r for r in self.rovers if has_rock[r][w]}

            # Option 1: Communicate sample already held by a rover
            if rovers_with_sample:
                min_comm_cost = float('inf')
                for r in rovers_with_sample:
                    current_loc = rover_locs.get(r)
                    if current_loc and current_loc in self.dist:
                        for comm_loc in self.comm_locs:
                            if comm_loc in self.dist[current_loc]:
                                dist_to_comm = self.dist[current_loc][comm_loc]
                                if dist_to_comm != float('inf'):
                                    min_comm_cost = min(min_comm_cost, dist_to_comm + 1) # Navigate + Communicate
                if min_comm_cost != float('inf'):
                    cost_w = min(cost_w, min_comm_cost)

            # Option 2: Sample at waypoint, then communicate
            if rock_at_w[w]:
                rock_rovers = {r for r in self.rovers if self.rover_capabilities[r]['rock']}
                min_sample_comm_cost = float('inf')
                for r in rock_rovers:
                    current_loc = rover_locs.get(r)
                    if current_loc and current_loc in self.dist and w in self.dist[current_loc]:
                        dist_to_sample = self.dist[current_loc][w]
                        if dist_to_sample != float('inf'):
                            cost_r = dist_to_sample # Navigate to sample
                            cost_r += (1 if store_full.get(r, False) else 0) # Drop if needed
                            cost_r += 1 # Sample

                            # Now at w with sample, need to communicate from w
                            min_comm_cost_from_w = float('inf')
                            if w in self.dist:
                                for comm_loc in self.comm_locs:
                                    if comm_loc in self.dist[w]:
                                        dist_w_to_comm = self.dist[w][comm_loc]
                                        if dist_w_to_comm != float('inf'):
                                            min_comm_cost_from_w = min(min_comm_cost_from_w, dist_w_to_comm + 1) # Navigate + Communicate

                            if min_comm_cost_from_w != float('inf'):
                                min_sample_comm_cost = min(min_sample_comm_cost, cost_r + min_comm_cost_from_w)

                if min_sample_comm_cost != float('inf'):
                    cost_w = min(cost_w, min_sample_comm_cost)

            if cost_w == float('inf'):
                 return float('inf')
            h += cost_w


        # Cost for Image Data Goals
        for o, m in needed_images:
            cost_om = float('inf')
            rovers_with_image = {r for r in self.rovers if has_image[r][o][m]}

            # Option 1: Communicate image already held by a rover
            if rovers_with_image:
                min_comm_cost = float('inf')
                for r in rovers_with_image:
                    current_loc = rover_locs.get(r)
                    if current_loc and current_loc in self.dist:
                        for comm_loc in self.comm_locs:
                            if comm_loc in self.dist[current_loc]:
                                dist_to_comm = self.dist[current_loc][comm_loc]
                                if dist_to_comm != float('inf'):
                                    min_comm_cost = min(min_comm_cost, dist_to_comm + 1) # Navigate + Communicate
                if min_comm_cost != float('inf'):
                    cost_om = min(cost_om, min_comm_cost)

            # Option 2: Take image, then communicate
            imaging_rovers = {r for r in self.rovers if self.rover_capabilities[r]['imaging']}
            min_take_comm_cost = float('inf')

            for r in imaging_rovers:
                current_loc = rover_locs.get(r)
                if not current_loc or current_loc not in self.dist: continue

                # Find suitable camera on this rover for this mode
                suitable_cameras = {i for i in self.cameras if self.camera_on_rover.get(i) == r and m in self.camera_modes.get(i, set())}

                for i in suitable_cameras:
                    cal_target = self.camera_cal_target.get(i)
                    if cal_target is None: continue # Camera needs calibration target

                    # Find suitable waypoint p for image and calibration
                    suitable_locs = self.objective_visibility.get(o, set()) & self.calibration_target_visibility.get(cal_target, set())

                    for p in suitable_locs:
                        if p in self.dist[current_loc]:
                            dist_to_p = self.dist[current_loc][p]
                            if dist_to_p != float('inf'):
                                cost_r_p = dist_to_p # Navigate to image/cal location
                                # Calibration cost: 1 (needed before taking image)
                                cost_r_p += 1
                                # Take image cost: 1
                                cost_r_p += 1

                                # Now at p with image, need to communicate from p
                                min_comm_cost_from_p = float('inf')
                                if p in self.dist:
                                    for comm_loc in self.comm_locs:
                                        if comm_loc in self.dist[p]:
                                            dist_p_to_comm = self.dist[p][comm_loc]
                                            if dist_p_to_comm != float('inf'):
                                                min_comm_cost_from_p = min(min_comm_cost_from_p, dist_p_to_comm + 1) # Navigate + Communicate

                                if min_comm_cost_from_p != float('inf'):
                                    min_take_comm_cost = min(min_take_comm_cost, cost_r_p + min_comm_cost_from_p)

            if min_take_comm_cost != float('inf'):
                cost_om = min(cost_om, min_take_comm_cost)

            if cost_om == float('inf'):
                 # This goal is unreachable from this state
                 return float('inf')
            h += cost_om

        return h
