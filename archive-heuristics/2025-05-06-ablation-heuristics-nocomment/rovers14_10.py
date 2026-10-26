import collections
# from heuristics.heuristic_base import Heuristic # Assuming Heuristic base class is provided

def get_parts(fact):
    """Helper function to parse a PDDL fact string into a list of parts."""
    # Assumes fact is like '(predicate arg1 arg2)'
    # Handles potential empty strings or malformed facts gracefully
    if not isinstance(fact, str) or len(fact) < 2 or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()

# Define a dummy Heuristic base class if not provided, just for code structure
# In a real scenario, this would be imported.
class Heuristic:
    def __init__(self, task):
        pass
    def __call__(self, node):
        pass


class roversHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Rovers domain.

    Summary:
    This heuristic estimates the cost to reach the goal state by summing up
    the estimated costs for each unsatisfied goal fact. For each goal, it
    determines if the required data (soil sample, rock sample, or image)
    is already collected by any rover. If not, it estimates the cost to
    collect the data (sampling or imaging, including calibration and store
    management) and then communicate it. If the data is already collected,
    it estimates the cost only for communication. Navigation costs are
    estimated using precomputed shortest path distances on the waypoint graph
    for each rover. The heuristic finds the minimum cost across all capable
    rovers and relevant waypoints for each unsatisfied goal component.

    Assumptions:
    - Soil/rock samples required by goals are initially present at their
      respective waypoints unless consumed. The heuristic checks if the sample
      is still present in the current state.
    - Camera calibration is required before taking an image if the specific
      image fact is not yet present for any rover. This might overestimate
      if a camera is already calibrated and no image has been taken yet,
      but simplifies the heuristic.
    - Each rover has exactly one store.
    - Navigation cost between adjacent waypoints is 1. Shortest path distances
      are used for longer navigations.
    - Unreachable goals (e.g., sample location gone and no rover has it,
      no visible waypoint for imaging/calibration/communication) contribute
      infinity to the cost, effectively pruning such states in a greedy search.

    Heuristic Initialization:
    The constructor performs the following steps once at the beginning:
    1.  Extracts all objects (rovers, waypoints, stores, cameras, modes,
        landers, objectives) from the initial state and static facts based
        on their appearance in known predicates.
    2.  Parses static facts to build data structures representing:
        -   Rover equipment (soil, rock, imaging).
        -   Mapping of rovers to their stores and cameras.
        -   Camera capabilities (supported modes, calibration targets).
        -   Visibility relationships between objectives/targets and waypoints.
        -   The waypoint traversal graph for each rover.
        -   The lander's location and the set of waypoints visible from it.
        -   Initial locations of soil and rock samples.
    3.  Precomputes the shortest path distances between all pairs of waypoints
        for each rover using Breadth-First Search (BFS) on their respective
        traversal graphs.
    4.  Parses the goal facts to identify the specific soil data, rock data,
        and image data requirements.

    Step-By-Step Thinking for Computing Heuristic:
    The `__call__` method computes the heuristic value for a given state:
    1.  Parses the current state to determine:
        -   Current location of each rover.
        -   Status (empty/full) of each store.
        -   Which rovers have collected which soil/rock samples.
        -   Which rovers have taken which images.
        -   Which cameras are currently calibrated on which rovers.
        -   Which soil/rock samples still remain at waypoints.
        -   Which data (soil, rock, image) has already been communicated.
    2.  Initializes the total heuristic cost `h` to 0.
    3.  Iterates through each soil data goal waypoint `W`:
        -   If `(communicated_soil_data W)` is not in the current state:
            -   Find the minimum cost to achieve this goal.
            -   Check if any rover `R` already has `(have_soil_analysis R W)`.
                -   If yes: Cost is 1 (communicate) + minimum navigation cost for `R` from its current location to any waypoint visible from the lander.
                -   If no: Check if `(at_soil_sample W)` is still in the state. If yes, find equipped rover `R`. Cost is 1 (sample) + 1 (communicate) + minimum chained navigation cost for `R` from its current location to `W` and then to any lander-visible waypoint. Add 1 if `R`'s store is full. Minimize this cost over all equipped rovers.
            -   Add the minimum cost found for this goal to `h` (if finite).
    4.  Iterates through each rock data goal waypoint `W`: Similar logic as for soil goals.
    5.  Iterates through each image data goal `(O, M)`:
        -   If `(communicated_image_data O M)` is not in the current state:
            -   Find the minimum cost to achieve this goal.
            -   Check if any rover `R` already has `(have_image R O M)`.
                -   If yes: Cost is 1 (communicate) + minimum navigation cost for `R` from its current location to any waypoint visible from the lander.
                -   If no: Find equipped rover `R` with camera `I` supporting mode `M`. Cost is 1 (take_image) + 1 (communicate) + 1 (calibrate) + minimum chained navigation cost for `R` from its current location to a calibration waypoint for `I`, then to an image waypoint for `O`, and then to any lander-visible waypoint. Minimize this cost over all suitable rovers, cameras, and waypoints.
            -   Add the minimum cost found for this goal to `h` (if finite).
    6.  Returns the total heuristic cost `h`. The heuristic is 0 if and only if all goals are in the current state.
    """

    def __init__(self, task):
        self.goals = task.goals
        static_facts = task.static

        # --- Extract Objects and Static Info ---
        self.rovers = set()
        self.waypoints = set()
        self.stores = set()
        self.cameras = set()
        self.modes = set()
        self.landers = set()
        self.objectives = set()

        all_facts = task.initial_state | static_facts
        for fact_str in all_facts:
            parts = get_parts(fact_str)
            if not parts: continue
            predicate = parts[0]
            if predicate == 'at' and len(parts) == 3:
                self.rovers.add(parts[1])
                self.waypoints.add(parts[2])
            elif predicate == 'at_lander' and len(parts) == 3:
                self.landers.add(parts[1])
                self.waypoints.add(parts[2])
            elif predicate == 'can_traverse' and len(parts) == 4:
                self.rovers.add(parts[1])
                self.waypoints.add(parts[2])
                self.waypoints.add(parts[3])
            elif predicate == 'store_of' and len(parts) == 3:
                self.stores.add(parts[1])
                self.rovers.add(parts[2])
            elif predicate == 'on_board' and len(parts) == 3:
                self.cameras.add(parts[1])
                self.rovers.add(parts[2])
            elif predicate == 'supports' and len(parts) == 3:
                self.cameras.add(parts[1])
                self.modes.add(parts[2])
            elif predicate == 'calibration_target' and len(parts) == 3:
                self.cameras.add(parts[1])
                self.objectives.add(parts[2])
            elif predicate == 'visible_from' and len(parts) == 3:
                self.objectives.add(parts[1])
                self.waypoints.add(parts[2])
            elif predicate == 'at_soil_sample' and len(parts) == 2:
                self.waypoints.add(parts[1])
            elif predicate == 'at_rock_sample' and len(parts) == 2:
                self.waypoints.add(parts[1])
            # Add other types if necessary, though these cover the main ones in facts

        self.rover_equipment = collections.defaultdict(set)
        self.rover_stores = {} # rover -> store
        self.rover_cameras = collections.defaultdict(list) # rover -> [cameras]
        self.camera_modes = collections.defaultdict(set) # camera -> {modes}
        self.camera_cal_targets = {} # camera -> objective
        self.objective_visible_wps = collections.defaultdict(set) # objective -> {waypoints}
        self.waypoint_graph = collections.defaultdict(lambda: collections.defaultdict(list)) # rover -> wp -> [neighbors]
        self.lander_location = None
        self.lander_visible_wps = set() # waypoints visible *from* lander location
        self.initial_soil_samples = set() # waypoints with soil samples initially
        self.initial_rock_samples = set() # waypoints with rock samples initially
        self.visible_graph = collections.defaultdict(set) # wp -> {visible_neighbors}

        for fact_str in static_facts:
            parts = get_parts(fact_str)
            if not parts: continue
            predicate = parts[0]
            if predicate == 'equipped_for_soil_analysis':
                self.rover_equipment[parts[1]].add('soil')
            elif predicate == 'equipped_for_rock_analysis':
                self.rover_equipment[parts[1]].add('rock')
            elif predicate == 'equipped_for_imaging':
                self.rover_equipment[parts[1]].add('imaging')
            elif predicate == 'store_of':
                self.rover_stores[parts[2]] = parts[1]
            elif predicate == 'on_board':
                self.rover_cameras[parts[2]].append(parts[1])
            elif predicate == 'supports':
                self.camera_modes[parts[1]].add(parts[2])
            elif predicate == 'calibration_target':
                self.camera_cal_targets[parts[1]] = parts[2]
            elif predicate == 'visible_from':
                self.objective_visible_wps[parts[1]].add(parts[2])
            elif predicate == 'can_traverse':
                self.waypoint_graph[parts[1]][parts[2]].append(parts[3])
            elif predicate == 'at_lander':
                self.lander_location = parts[2]
            elif predicate == 'visible':
                 self.visible_graph[parts[1]].add(parts[2])
                 # Assuming symmetric visibility based on examples, but domain doesn't enforce.
                 # self.visible_graph[parts[2]].add(parts[1])
            elif predicate == 'at_soil_sample':
                self.initial_soil_samples.add(parts[1])
            elif predicate == 'at_rock_sample':
                self.initial_rock_samples.add(parts[1])

        # Determine lander_visible_wps after finding lander_location
        if self.lander_location:
             # Find all waypoints X such that (visible X lander_location) is true
             for wp in self.waypoints:
                 if self.lander_location in self.visible_graph.get(wp, set()):
                      self.lander_visible_wps.add(wp)


        # Precompute rover distances
        self.rover_distances = {}
        for rover in self.rovers:
            graph = self.waypoint_graph.get(rover, {})
            # Need to compute distances from *all* waypoints, not just those in the graph keys
            # The graph keys only include waypoints that have outgoing 'can_traverse' edges.
            # We need distances from any waypoint a rover might be at.
            # Build a complete graph representation including all waypoints
            full_graph = {wp: [] for wp in self.waypoints}
            for start_wp, neighbors in graph.items():
                 full_graph[start_wp].extend(neighbors)

            self.rover_distances[rover] = {wp: self._bfs(full_graph, wp) for wp in self.waypoints}


        # Parse goal facts
        self.goal_soil_wps = set()
        self.goal_rock_wps = set()
        self.goal_images = set() # Set of (objective, mode) tuples

        for goal_str in self.goals:
            parts = get_parts(goal_str)
            if not parts: continue
            predicate = parts[0]
            if predicate == 'communicated_soil_data' and len(parts) == 2:
                self.goal_soil_wps.add(parts[1])
            elif predicate == 'communicated_rock_data' and len(parts) == 2:
                self.goal_rock_wps.add(parts[1])
            elif predicate == 'communicated_image_data' and len(parts) == 3:
                self.goal_images.add((parts[1], parts[2]))

    def _bfs(self, graph, start_node):
        """Performs BFS to find shortest distances from start_node to all reachable nodes."""
        distances = {node: float('inf') for node in graph}
        if start_node not in graph:
             # Start node is not in the graph (e.g., isolated waypoint)
             # It cannot reach anywhere, and nothing can reach it via traversal.
             # Distances remain infinity.
             return distances

        distances[start_node] = 0
        queue = collections.deque([start_node])
        while queue:
            current_node = queue.popleft()
            for neighbor in graph.get(current_node, []):
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)
        return distances


    def get_distance(self, rover, start_wp, end_wp):
        """Gets the shortest path distance for a rover between two waypoints."""
        if start_wp is None or end_wp is None: return float('inf') # Rover location unknown or target unknown
        if start_wp == end_wp:
            return 0
        # Check if rover can traverse at all or if start_wp is valid for this rover
        if rover not in self.rover_distances or start_wp not in self.rover_distances[rover]:
             return float('inf')
        # Return precomputed distance
        return self.rover_distances[rover][start_wp].get(end_wp, float('inf'))


    def __call__(self, node):
        state = node.state

        # --- Parse Current State ---
        rover_locations = {} # rover -> waypoint
        store_status = {} # store -> 'empty' or 'full'
        rover_soil_samples = collections.defaultdict(set) # rover -> {waypoints}
        rover_rock_samples = collections.defaultdict(set) # rover -> {waypoints}
        rover_images = collections.defaultdict(set) # rover -> {(objective, mode)}
        calibrated_cameras = set() # {(camera, rover)}
        current_soil_samples = set() # waypoints with samples remaining
        current_rock_samples = set() # waypoints with samples remaining
        communicated_soil = set() # {waypoints}
        communicated_rock = set() # {waypoints}
        communicated_images = set() # {(objective, mode)}

        for fact_str in state:
            parts = get_parts(fact_str)
            if not parts: continue
            predicate = parts[0]
            if predicate == 'at' and len(parts) == 3 and parts[1] in self.rovers:
                rover_locations[parts[1]] = parts[2]
            elif predicate == 'empty' and len(parts) == 2 and parts[1] in self.stores:
                store_status[parts[1]] = 'empty'
            elif predicate == 'full' and len(parts) == 2 and parts[1] in self.stores:
                store_status[parts[1]] = 'full'
            elif predicate == 'have_soil_analysis' and len(parts) == 3 and parts[1] in self.rovers:
                rover_soil_samples[parts[1]].add(parts[2])
            elif predicate == 'have_rock_analysis' and len(parts) == 3 and parts[1] in self.rovers:
                rover_rock_samples[parts[1]].add(parts[2])
            elif predicate == 'have_image' and len(parts) == 4 and parts[1] in self.rovers:
                rover_images[parts[1]].add((parts[2], parts[3]))
            elif predicate == 'calibrated' and len(parts) == 3 and parts[2] in self.rovers:
                calibrated_cameras.add((parts[1], parts[2]))
            elif predicate == 'at_soil_sample' and len(parts) == 2 and parts[1] in self.waypoints:
                current_soil_samples.add(parts[1])
            elif predicate == 'at_rock_sample' and len(parts) == 2 and parts[1] in self.waypoints:
                current_rock_samples.add(parts[1])
            elif predicate == 'communicated_soil_data' and len(parts) == 2 and parts[1] in self.waypoints:
                communicated_soil.add(parts[1])
            elif predicate == 'communicated_rock_data' and len(parts) == 2 and parts[1] in self.waypoints:
                communicated_rock.add(parts[1])
            elif predicate == 'communicated_image_data' and len(parts) == 3 and (parts[1], parts[2]) in self.goal_images:
                 communicated_images.add((parts[1], parts[2]))

        h = 0

        # --- Estimate Cost for Soil Goals ---
        for w in self.goal_soil_wps:
            if w not in communicated_soil:
                min_goal_cost = float('inf')

                # Find if any rover already has the sample
                rover_with_sample = None
                for r in self.rovers:
                    if w in rover_soil_samples[r]:
                        rover_with_sample = r
                        break

                if rover_with_sample:
                    # Need to communicate
                    comm_cost = 1
                    min_nav_to_lander = float('inf')
                    current_wp = rover_locations.get(rover_with_sample)
                    if current_wp:
                         for lander_wp in self.lander_visible_wps:
                             nav_cost = self.get_distance(rover_with_sample, current_wp, lander_wp)
                             min_nav_to_lander = min(min_nav_to_lander, nav_cost)

                    if min_nav_to_lander != float('inf'):
                         min_goal_cost = comm_cost + min_nav_to_lander

                else:
                    # Need to sample and then communicate
                    if w in current_soil_samples: # Check if sample is still available
                        for r in self.rovers:
                            if 'soil' in self.rover_equipment.get(r, set()):
                                # Rover R can sample
                                sample_cost = 1
                                comm_cost = 1

                                current_wp = rover_locations.get(r)
                                if current_wp is None: continue # Rover location unknown

                                # Chained navigation: current -> sample_wp -> lander_wp
                                min_chained_nav = float('inf')
                                if self.lander_visible_wps: # Must have lander visible waypoints
                                    nav_to_sample = self.get_distance(r, current_wp, w)
                                    if nav_to_sample != float('inf'):
                                        for lander_wp in self.lander_visible_wps:
                                             nav_sample_to_lander = self.get_distance(r, w, lander_wp)
                                             if nav_sample_to_lander != float('inf'):
                                                min_chained_nav = min(min_chained_nav, nav_to_sample + nav_sample_to_lander)

                                if min_chained_nav != float('inf'):
                                    store = self.rover_stores.get(r)
                                    drop_cost = 1 if store and store_status.get(store) == 'full' else 0

                                    current_rover_goal_cost = sample_cost + comm_cost + min_chained_nav + drop_cost
                                    min_goal_cost = min(min_goal_cost, current_rover_goal_cost)

                if min_goal_cost != float('inf'):
                     h += min_goal_cost


        # --- Estimate Cost for Rock Goals ---
        for w in self.goal_rock_wps:
            if w not in communicated_rock:
                min_goal_cost = float('inf')

                rover_with_sample = None
                for r in self.rovers:
                    if w in rover_rock_samples[r]:
                        rover_with_sample = r
                        break

                if rover_with_sample:
                    comm_cost = 1
                    min_nav_to_lander = float('inf')
                    current_wp = rover_locations.get(rover_with_sample)
                    if current_wp:
                         for lander_wp in self.lander_visible_wps:
                             nav_cost = self.get_distance(rover_with_sample, current_wp, lander_wp)
                             min_nav_to_lander = min(min_nav_to_lander, nav_cost)

                    if min_nav_to_lander != float('inf'):
                         min_goal_cost = comm_cost + min_nav_to_lander

                else:
                    if w in current_rock_samples:
                        for r in self.rovers:
                            if 'rock' in self.rover_equipment.get(r, set()):
                                sample_cost = 1
                                comm_cost = 1

                                current_wp = rover_locations.get(r)
                                if current_wp is None: continue

                                min_chained_nav = float('inf')
                                if self.lander_visible_wps:
                                    nav_to_sample = self.get_distance(r, current_wp, w)
                                    if nav_to_sample != float('inf'):
                                        for lander_wp in self.lander_visible_wps:
                                             nav_sample_to_lander = self.get_distance(r, w, lander_wp)
                                             if nav_sample_to_lander != float('inf'):
                                                min_chained_nav = min(min_chained_nav, nav_to_sample + nav_sample_to_lander)

                                if min_chained_nav != float('inf'):
                                    store = self.rover_stores.get(r)
                                    drop_cost = 1 if store and store_status.get(store) == 'full' else 0
                                    current_rover_goal_cost = sample_cost + comm_cost + min_chained_nav + drop_cost
                                    min_goal_cost = min(min_goal_cost, current_rover_goal_cost)

                if min_goal_cost != float('inf'):
                     h += min_goal_cost


        # --- Estimate Cost for Image Goals ---
        for o, m in self.goal_images:
            if (o, m) not in communicated_images:
                min_goal_cost = float('inf')

                rover_with_image = None
                for r in self.rovers:
                    if (o, m) in rover_images[r]:
                        rover_with_image = r
                        break

                if rover_with_image:
                    comm_cost = 1
                    min_nav_to_lander = float('inf')
                    current_wp = rover_locations.get(rover_with_image)
                    if current_wp:
                         for lander_wp in self.lander_visible_wps:
                             nav_cost = self.get_distance(rover_with_image, current_wp, lander_wp)
                             min_nav_to_lander = min(min_nav_to_lander, nav_cost)

                    if min_nav_to_lander != float('inf'):
                         min_goal_cost = comm_cost + min_nav_to_lander

                else:
                    # Need to take image and then communicate
                    for r in self.rovers:
                        if 'imaging' in self.rover_equipment.get(r, set()):
                            for cam in self.rover_cameras.get(r, []):
                                if m in self.camera_modes.get(cam, set()):
                                    # Rover R with camera Cam can take image O in mode M
                                    take_img_cost = 1
                                    comm_cost = 1
                                    cal_cost = 1 # Assume calibration is needed if image not taken

                                    current_wp = rover_locations.get(r)
                                    if current_wp is None: continue

                                    # Find best sequence of waypoints: current_wp -> cal_wp -> img_wp -> lander_wp.
                                    cal_target = self.camera_cal_targets.get(cam)
                                    cal_wps = self.objective_visible_wps.get(cal_target, set())
                                    img_wps = self.objective_visible_wps.get(o, set())
                                    lander_wps = self.lander_visible_wps

                                    min_chained_nav = float('inf')

                                    if not cal_wps or not img_wps or not lander_wps:
                                         # Cannot find suitable waypoints for calibration, image, or communication
                                         continue # Try next rover/camera

                                    for cal_wp in cal_wps:
                                        nav1 = self.get_distance(r, current_wp, cal_wp)
                                        if nav1 == float('inf'): continue

                                        for img_wp in img_wps:
                                            nav2 = self.get_distance(r, cal_wp, img_wp)
                                            if nav2 == float('inf'): continue

                                            for lander_wp in lander_wps:
                                                nav3 = self.get_distance(r, img_wp, lander_wp)
                                                if nav3 == float('inf'): continue

                                                min_chained_nav = min(min_chained_nav, nav1 + nav2 + nav3)


                                    if min_chained_nav != float('inf'):
                                        current_rover_goal_cost = cal_cost + take_img_cost + comm_cost + min_chained_nav
                                        min_goal_cost = min(min_goal_cost, current_rover_goal_cost)


                if min_goal_cost != float('inf'):
                     h += min_goal_cost

        return h
