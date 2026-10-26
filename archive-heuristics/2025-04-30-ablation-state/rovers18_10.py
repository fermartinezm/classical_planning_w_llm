import collections
from fnmatch import fnmatch
# Assuming heuristics.heuristic_base.Heuristic is available in the environment
from heuristics.heuristic_base import Heuristic

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact."""
    # Handle potential empty strings or malformed facts gracefully
    if not fact or not isinstance(fact, str) or len(fact) < 2 or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """Check if a PDDL fact matches a given pattern."""
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def bfs(graph, start_node):
    """
    Performs BFS on a graph to find shortest distances from start_node.
    Graph is represented as {node: [neighbor1, neighbor2, ...]}
    Returns {node: distance}
    """
    # Collect all nodes mentioned in the graph (keys and values)
    all_nodes = set(graph.keys())
    for neighbors in graph.values():
        all_nodes.update(neighbors)

    # If the start_node is not even in the set of known nodes, it's isolated.
    # Distance to itself is 0, others infinity.
    if start_node not in all_nodes:
         distances = {node: float('inf') for node in all_nodes}
         distances[start_node] = 0
         return distances

    distances = {node: float('inf') for node in all_nodes}
    distances[start_node] = 0
    queue = collections.deque([start_node])

    while queue:
        current_node = queue.popleft()

        # Check if current_node has outgoing edges in the graph
        if current_node in graph:
            for neighbor in graph[current_node]:
                # Only update if a shorter path is found (standard BFS)
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)
    return distances


class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve all goal conditions.
    It breaks down each unachieved goal (communicating soil, rock, or image data)
    into sequential stages: collecting the data (sampling or imaging) and then
    communicating the data. It sums the estimated minimum cost for each unachieved
    goal. The cost for each stage is estimated by summing the minimum navigation
    cost to the required location(s) and the cost of the specific action(s)
    (sample, drop, calibrate, take_image, communicate).

    # Assumptions:
    - Action costs are 1.
    - Rover capabilities (equipped_for_...) are static and precomputed.
    - Store management is simplified: a rover needs an empty store to sample;
      if its store is full, one 'drop' action is assumed to make it empty.
    - Camera calibration is needed before taking an image and is consumed by
      the take_image action.
    - Any waypoint visible from a calibration target can be used for calibration.
    - Any waypoint visible from an objective can be used for imaging that objective.
    - Any waypoint visible from the lander can be used for communication.
    - Navigation cost between waypoints for a specific rover is the shortest path
      distance (number of navigate actions) considering `can_traverse` and `visible`
      predicates. These distances are precomputed using BFS.
    - The cost to achieve a 'have_X_data' or 'have_image' fact is the minimum cost
      across all suitable rovers and required locations.
    - The cost to communicate is the minimum navigation cost for any rover to a
      communication point plus the communication action cost.
    - The total heuristic is the sum of the costs for each unachieved goal,
      assuming these goals can be pursued somewhat independently (h_add style).
    - If any required location is unreachable or a required capability is missing,
      the heuristic returns infinity.

    # Heuristic Initialization
    - Parses goal conditions.
    - Parses static facts to identify objects (rovers, waypoints, etc.) and
      relationships (capabilities, store_of, on_board, supports, calibration_target,
      visible, visible_from, can_traverse, at_lander).
    - Builds navigation graphs for each rover based on `can_traverse` and `visible`.
    - Computes all-pairs shortest paths for each rover on its navigation graph using BFS.
    - Identifies communication waypoints (visible from lander locations).
    - Stores mappings for quick lookup (store to rover, camera to rover, camera to modes,
      camera to calibration target, objective to visible waypoints).

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize total heuristic cost `h = 0`.
    2. Parse the current state to get dynamic facts: rover locations, store states,
       `have_..._analysis`, `have_image`, `calibrated`, and `communicated_..._data` facts.
    3. Iterate through each goal fact defined in the task.
    4. For an unachieved goal `(communicated_soil_data ?w)`:
       a. If `(have_soil_analysis ?r ?w)` is not true for any rover `?r`:
          i. Find equipped rovers for soil analysis. If none, return infinity.
          ii. Find the minimum navigation cost for any equipped rover from its current
              location to waypoint `?w`. If unreachable, this part is infinity.
          iii. Find the minimum cost to get an empty store for any equipped rover
               (0 if empty, 1 if full). If impossible, this part is infinity.
          iv. The cost to get the sample is (min_nav_to_w) + (cost_empty_store) + 1 (sample action).
       b. Find the minimum navigation cost for any rover from its current location
          to any communication waypoint. If unreachable, this part is infinity.
       c. The cost to communicate is (min_nav_to_comm_loc) + 1 (communicate action).
       d. Add (cost to get sample) + (cost to communicate) to `h`.
    5. For an unachieved goal `(communicated_rock_data ?w)`: Follow similar steps as for soil data.
    6. For an unachieved goal `(communicated_image_data ?o ?m)`:
       a. If `(have_image ?r ?o ?m)` is not true for any suitable rover `?r` and camera `?i`
          (equipped for imaging, camera on board, supports mode `?m`):
          i. Find suitable rover/camera pairs. If none, return infinity.
          ii. For each suitable pair (r, i):
              - Cost to calibrate: If `(calibrated i r)` is false, find the calibration
                target `?t` for `i`. Find waypoints visible from `?t`. Find minimum
                navigation cost for `r` from its current location to any of these
                waypoints. If unreachable, cost is infinity. Add 1 (calibrate action).
                If `(calibrated i r)` is true, cost is 0.
              - Cost to take image: Find waypoints visible from `?o`. Find minimum
                navigation cost for `r` from its current location to any of these
                waypoints. If unreachable, cost is infinity. Add 1 (take_image action).
          iii. The cost to get the image is the minimum of (cost to calibrate + cost to take image)
               over all suitable rover/camera pairs. If minimum is infinity, return infinity.
       b. Find the minimum navigation cost for any rover from its current location
          to any communication waypoint. If unreachable, this part is infinity.
       c. The cost to communicate is (min_nav_to_comm_loc) + 1 (communicate action).
       d. Add (cost to get image) + (cost to communicate) to `h`.
    7. Return `h`.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # Identify all objects by type from all facts (initial state + static)
        self.rovers = set()
        self.waypoints = set()
        self.stores = set()
        self.cameras = set()
        self.modes = set()
        self.landers = set()
        self.objectives = set()

        all_facts = task.initial_state | static_facts
        for fact in all_facts:
            parts = get_parts(fact)
            if not parts: continue
            pred = parts[0]
            args = parts[1:]

            # Populate object sets based on predicate arguments and domain types
            if pred == 'at' and len(args) == 2: self.rovers.add(args[0]); self.waypoints.add(args[1])
            elif pred == 'at_lander' and len(args) == 2: self.landers.add(args[0]); self.waypoints.add(args[1])
            elif pred == 'can_traverse' and len(args) == 3: self.rovers.add(args[0]); self.waypoints.add(args[1]); self.waypoints.add(args[2])
            elif pred == 'equipped_for_soil_analysis' and len(args) == 1: self.rovers.add(args[0])
            elif pred == 'equipped_for_rock_analysis' and len(args) == 1: self.rovers.add(args[0])
            elif pred == 'equipped_for_imaging' and len(args) == 1: self.rovers.add(args[0])
            elif pred == 'empty' and len(args) == 1: self.stores.add(args[0])
            elif pred == 'have_rock_analysis' and len(args) == 2: self.rovers.add(args[0]); self.waypoints.add(args[1])
            elif pred == 'have_soil_analysis' and len(args) == 2: self.rovers.add(args[0]); self.waypoints.add(args[1])
            elif pred == 'full' and len(args) == 1: self.stores.add(args[0])
            elif pred == 'calibrated' and len(args) == 2: self.cameras.add(args[0]); self.rovers.add(args[1])
            elif pred == 'supports' and len(args) == 2: self.cameras.add(args[0]); self.modes.add(args[1])
            elif pred == 'visible' and len(args) == 2: self.waypoints.add(args[0]); self.waypoints.add(args[1])
            elif pred == 'have_image' and len(args) == 3: self.rovers.add(args[0]); self.objectives.add(args[1]); self.modes.add(args[2])
            elif pred == 'communicated_soil_data' and len(args) == 1: self.waypoints.add(args[0])
            elif pred == 'communicated_rock_data' and len(args) == 1: self.waypoints.add(args[0])
            elif pred == 'communicated_image_data' and len(args) == 2: self.objectives.add(args[0]); self.modes.add(args[1])
            elif pred == 'at_soil_sample' and len(args) == 1: self.waypoints.add(args[0])
            elif pred == 'at_rock_sample' and len(args) == 1: self.waypoints.add(args[0])
            elif pred == 'visible_from' and len(args) == 2: self.objectives.add(args[0]); self.waypoints.add(args[1])
            elif pred == 'store_of' and len(args) == 2: self.stores.add(args[0]); self.rovers.add(args[1])
            elif pred == 'calibration_target' and len(args) == 2: self.cameras.add(args[0]); self.objectives.add(args[1])
            elif pred == 'on_board' and len(args) == 2: self.cameras.add(args[0]); self.rovers.add(args[1])


        # Store static relationships
        self.at_lander = {} # lander -> waypoint
        self.equipped_for_soil = set()
        self.equipped_for_rock = set()
        self.equipped_for_imaging = set()
        self.store_of = {} # store -> rover
        self.on_board = {} # camera -> rover
        self.supports = {} # camera -> set of modes
        self.calibration_target = {} # camera -> objective
        self.visible = set() # (wp1, wp2) pairs
        self.visible_from = {} # objective -> set of waypoints
        self.can_traverse = {} # rover -> set of (wp1, wp2) pairs

        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue
            pred = parts[0]
            args = parts[1:]

            if pred == 'at_lander' and len(args) == 2: self.at_lander[args[0]] = args[1]
            elif pred == 'equipped_for_soil_analysis' and len(args) == 1: self.equipped_for_soil.add(args[0])
            elif pred == 'equipped_for_rock_analysis' and len(args) == 1: self.equipped_for_rock.add(args[0])
            elif pred == 'equipped_for_imaging' and len(args) == 1: self.equipped_for_imaging.add(args[0])
            elif pred == 'store_of' and len(args) == 2: self.store_of[args[0]] = args[1]
            elif pred == 'on_board' and len(args) == 2: self.on_board[args[0]] = args[1]
            elif pred == 'supports' and len(args) == 2:
                camera, mode = args
                if camera not in self.supports: self.supports[camera] = set()
                self.supports[camera].add(mode)
            elif pred == 'calibration_target' and len(args) == 2: self.calibration_target[args[0]] = args[1]
            elif pred == 'visible' and len(args) == 2: self.visible.add((args[0], args[1]))
            elif pred == 'visible_from' and len(args) == 2:
                obj, wp = args
                if obj not in self.visible_from: self.visible_from[obj] = set()
                self.visible_from[obj].add(wp)
            elif pred == 'can_traverse' and len(args) == 3:
                rover, wp1, wp2 = args
                if rover not in self.can_traverse: self.can_traverse[rover] = set()
                self.can_traverse[rover].add((wp1, wp2))

        # Build navigation graphs and compute shortest paths
        self.dist = {} # rover -> start_wp -> end_wp -> distance
        for rover in self.rovers:
            rover_graph = {}
            # Initialize graph with all waypoints
            for wp in self.waypoints:
                rover_graph[wp] = []

            # Add edges if can_traverse AND visible
            if rover in self.can_traverse:
                 for w1, w2 in self.can_traverse[rover]:
                     if (w1, w2) in self.visible:
                         rover_graph[w1].append(w2)

            self.dist[rover] = {}
            # BFS from every waypoint to every other waypoint
            for start_wp in self.waypoints:
                 self.dist[rover][start_wp] = bfs(rover_graph, start_wp)

        # Compute communication waypoints (visible from any lander location)
        self.comm_waypoints = set()
        for lander_loc in self.at_lander.values():
            for wp1, wp2 in self.visible:
                if wp2 == lander_loc:
                    self.comm_waypoints.add(wp1)

        # Store mappings for quick lookup
        self._rover_has_store = {r: s for s, r in self.store_of.items()}
        self._rover_cameras = {} # rover -> set of cameras
        for cam, r in self.on_board.items():
            if r not in self._rover_cameras: self._rover_cameras[r] = set()
            self._rover_cameras[r].add(cam)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Parse current state for relevant facts
        state_rover_locations = {} # rover -> waypoint
        state_store_status = {} # store -> 'empty' or 'full'
        state_have_soil = set() # (rover, waypoint)
        state_have_rock = set() # (rover, waypoint)
        state_have_image = set() # (rover, objective, mode)
        state_calibrated = set() # (camera, rover)
        state_communicated_soil = set() # waypoint
        state_communicated_rock = set() # waypoint
        state_communicated_image = set() # (objective, mode)

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue
            pred = parts[0]
            args = parts[1:]

            if pred == 'at' and len(args) == 2 and args[0] in self.rovers and args[1] in self.waypoints:
                state_rover_locations[args[0]] = args[1]
            elif pred == 'empty' and len(args) == 1 and args[0] in self.stores:
                 state_store_status[args[0]] = 'empty'
            elif pred == 'full' and len(args) == 1 and args[0] in self.stores:
                 state_store_status[args[0]] = 'full'
            elif pred == 'have_soil_analysis' and len(args) == 2 and args[0] in self.rovers and args[1] in self.waypoints:
                state_have_soil.add(tuple(args))
            elif pred == 'have_rock_analysis' and len(args) == 2 and args[0] in self.rovers and args[1] in self.waypoints:
                state_have_rock.add(tuple(args))
            elif pred == 'have_image' and len(args) == 3 and args[0] in self.rovers and args[1] in self.objectives and args[2] in self.modes:
                state_have_image.add(tuple(args))
            elif pred == 'calibrated' and len(args) == 2 and args[0] in self.cameras and args[1] in self.rovers:
                state_calibrated.add(tuple(args))
            elif pred == 'communicated_soil_data' and len(args) == 1 and args[0] in self.waypoints:
                state_communicated_soil.add(args[0])
            elif pred == 'communicated_rock_data' and len(args) == 1 and args[0] in self.waypoints:
                state_communicated_rock.add(args[0])
            elif pred == 'communicated_image_data' and len(args) == 2 and args[0] in self.objectives and args[1] in self.modes:
                state_communicated_image.add(tuple(args))


        total_heuristic_cost = 0

        # Process each goal
        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue
            pred = parts[0]
            args = parts[1:]

            if pred == 'communicated_soil_data' and len(args) == 1:
                wp = args[0]
                if wp in state_communicated_soil:
                    continue # Goal already met

                # Cost to get sample
                sample_achieved = any(w == wp for r, w in state_have_soil)
                sample_cost = 0
                if not sample_achieved:
                    equipped_rovers = list(self.equipped_for_soil)
                    if not equipped_rovers:
                        # If no rover is equipped, this goal is impossible
                        return float('inf')

                    min_nav_to_sample_loc = float('inf')
                    cost_empty_store = float('inf')

                    for r in equipped_rovers:
                        current_loc_r = state_rover_locations.get(r)
                        if current_loc_r:
                             # Safe distance lookup
                             dist = self.dist.get(r, {}).get(current_loc_r, {}).get(wp, float('inf'))
                             min_nav_to_sample_loc = min(min_nav_to_sample_loc, dist)

                        store = self._rover_has_store.get(r)
                        if store:
                            if state_store_status.get(store) == 'empty':
                                cost_empty_store = 0
                            elif state_store_status.get(store) == 'full':
                                cost_empty_store = min(cost_empty_store, 1) # Cost of drop
                            # If store exists but is neither empty nor full, something is wrong, treat as impossible
                            # else: cost_empty_store remains float('inf')

                    if min_nav_to_sample_loc == float('inf') or cost_empty_store == float('inf'):
                         # Cannot reach sample location or get empty store with any equipped rover
                         return float('inf')

                    sample_cost = min_nav_to_sample_loc + cost_empty_store + 1 # navigate + drop (if needed) + sample

                # Cost to communicate
                min_nav_to_comm_loc = float('inf')
                if not self.comm_waypoints:
                    # No communication points available
                    return float('inf')

                for r in self.rovers: # Any rover can communicate
                    current_loc_r = state_rover_locations.get(r)
                    if current_loc_r:
                        for comm_wp in self.comm_waypoints:
                             # Safe distance lookup
                             dist = self.dist.get(r, {}).get(current_loc_r, {}).get(comm_wp, float('inf'))
                             min_nav_to_comm_loc = min(min_nav_to_comm_loc, dist)

                if min_nav_to_comm_loc == float('inf'):
                    # Cannot reach a communication point with any rover
                    return float('inf')

                comm_cost = min_nav_to_comm_loc + 1 # navigate + communicate

                total_heuristic_cost += sample_cost + comm_cost

            elif pred == 'communicated_rock_data' and len(args) == 1:
                wp = args[0]
                if wp in state_communicated_rock:
                    continue # Goal already met

                # Cost to get sample
                sample_achieved = any(w == wp for r, w in state_have_rock)
                sample_cost = 0
                if not sample_achieved:
                    equipped_rovers = list(self.equipped_for_rock)
                    if not equipped_rovers:
                        return float('inf')

                    min_nav_to_sample_loc = float('inf')
                    cost_empty_store = float('inf')

                    for r in equipped_rovers:
                        current_loc_r = state_rover_locations.get(r)
                        if current_loc_r:
                             dist = self.dist.get(r, {}).get(current_loc_r, {}).get(wp, float('inf'))
                             min_nav_to_sample_loc = min(min_nav_to_sample_loc, dist)

                        store = self._rover_has_store.get(r)
                        if store:
                            if state_store_status.get(store) == 'empty':
                                cost_empty_store = 0
                            elif state_store_status.get(store) == 'full':
                                cost_empty_store = min(cost_empty_store, 1) # Cost of drop
                            # else: cost_empty_store remains float('inf')

                    if min_nav_to_sample_loc == float('inf') or cost_empty_store == float('inf'):
                         return float('inf')

                    sample_cost = min_nav_to_sample_loc + cost_empty_store + 1 # navigate + drop (if needed) + sample

                # Cost to communicate (re-calculate or reuse min_nav_to_comm_loc)
                min_nav_to_comm_loc = float('inf')
                if not self.comm_waypoints: return float('inf')

                for r in self.rovers:
                    current_loc_r = state_rover_locations.get(r)
                    if current_loc_r:
                        for comm_wp in self.comm_waypoints:
                             dist = self.dist.get(r, {}).get(current_loc_r, {}).get(comm_wp, float('inf'))
                             min_nav_to_comm_loc = min(min_nav_to_comm_loc, dist)

                if min_nav_to_comm_loc == float('inf'):
                    return float('inf')

                comm_cost = min_nav_to_comm_loc + 1 # navigate + communicate

                total_heuristic_cost += sample_cost + comm_cost


            elif pred == 'communicated_image_data' and len(args) == 2:
                obj, mode = args
                if (obj, mode) in state_communicated_image:
                    continue # Goal already met

                # Cost to get image
                image_achieved = any(o == obj and m == mode for r, o, m in state_have_image)
                image_cost = 0
                if not image_achieved:
                    # Find suitable rovers/cameras
                    suitable_rovers_cameras = [] # List of (rover, camera) tuples
                    for r in self.equipped_for_imaging:
                        if r in self._rover_cameras:
                            for cam in self._rover_cameras[r]:
                                if mode in self.supports.get(cam, set()):
                                    suitable_rovers_cameras.append((r, cam))

                    if not suitable_rovers_cameras:
                        return float('inf') # Impossible

                    min_combined_image_prep_cost = float('inf')

                    for r, cam in suitable_rovers_cameras:
                         current_loc_r = state_rover_locations.get(r)
                         if not current_loc_r: continue # Rover location unknown

                         cal_cost_r_i = 0
                         if (cam, r) not in state_calibrated:
                             cal_target = self.calibration_target.get(cam)
                             if not cal_target:
                                 cal_cost_r_i = float('inf') # Camera has no calibration target defined
                             else:
                                 cal_wps = self.visible_from.get(cal_target, set())
                                 if not cal_wps:
                                     cal_cost_r_i = float('inf') # Calibration target not visible from anywhere
                                 else:
                                     min_nav_to_cal_loc = float('inf')
                                     for cal_wp in cal_wps:
                                         dist = self.dist.get(r, {}).get(current_loc_r, {}).get(cal_wp, float('inf'))
                                         min_nav_to_cal_loc = min(min_nav_to_cal_loc, dist)

                                     if min_nav_to_cal_loc == float('inf'):
                                         cal_cost_r_i = float('inf') # Cannot reach any calibration waypoint
                                     else:
                                         cal_cost_r_i = min_nav_to_cal_loc + 1 # navigate + calibrate

                         img_wps = self.visible_from.get(obj, set())
                         if not img_wps:
                             img_cost_r_i = float('inf') # Objective not visible from anywhere
                         else:
                             min_nav_to_img_loc = float('inf')
                             for img_wp in img_wps:
                                 dist = self.dist.get(r, {}).get(current_loc_r, {}).get(img_wp, float('inf'))
                                 min_nav_to_img_loc = min(min_nav_to_img_loc, dist)

                             if min_nav_to_img_loc == float('inf'):
                                 img_cost_r_i = float('inf') # Cannot reach any imaging waypoint
                             else:
                                 img_cost_r_i = min_nav_to_img_loc + 1 # navigate + take_image

                         if cal_cost_r_i != float('inf') and img_cost_r_i != float('inf'):
                             min_combined_image_prep_cost = min(min_combined_image_prep_cost, cal_cost_r_i + img_cost_r_i)

                    if min_combined_image_prep_cost == float('inf'):
                         # Cannot prepare image with any suitable rover/camera combination
                         return float('inf')

                    image_cost = min_combined_image_prep_cost

                # Cost to communicate (re-calculate or reuse min_nav_to_comm_loc)
                min_nav_to_comm_loc = float('inf')
                if not self.comm_waypoints: return float('inf')

                for r in self.rovers:
                    current_loc_r = state_rover_locations.get(r)
                    if current_loc_r:
                        for comm_wp in self.comm_waypoints:
                             dist = self.dist.get(r, {}).get(current_loc_r, {}).get(comm_wp, float('inf'))
                             min_nav_to_comm_loc = min(min_nav_to_comm_loc, dist)

                if min_nav_to_comm_loc == float('inf'):
                    return float('inf')

                comm_cost = min_nav_to_comm_loc + 1 # navigate + communicate

                total_heuristic_cost += image_cost + comm_cost

        return total_heuristic_cost
