from fnmatch import fnmatch
from collections import defaultdict, deque
from heuristics.heuristic_base import Heuristic # Assuming this base class exists


# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

# Helper function to match PDDL facts
def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at rover1 waypoint1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args) and '*' not in args:
         return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# Helper for shortest path (BFS)
def bfs(graph, start_node, all_nodes):
    """Computes shortest path distances from start_node to all other nodes using BFS."""
    distances = {node: float('inf') for node in all_nodes}
    if start_node not in all_nodes:
        return distances # Start node not in the graph's node set

    distances[start_node] = 0
    queue = deque([start_node])
    visited = {start_node}

    while queue:
        current_node = queue.popleft()

        if current_node in graph: # Check if node has any outgoing edges in this rover's graph
            for neighbor in graph[current_node]:
                if neighbor in all_nodes and neighbor not in visited: # Ensure neighbor is a known waypoint
                    visited.add(neighbor)
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)
    return distances

# Helper for minimum distance from a node to a set of nodes
def min_dist_to_set(start_node, target_nodes, rover_shortest_paths_for_rover):
    """Finds the minimum distance from start_node to any node in target_nodes using the provided shortest paths."""
    if not target_nodes:
        return float('inf')

    min_d = float('inf')
    if start_node in rover_shortest_paths_for_rover:
        for target_node in target_nodes:
            min_d = min(min_d, rover_shortest_paths_for_rover[start_node].get(target_node, float('inf')))
    return min_d

# Helper for minimum distance between two sets of nodes
def min_dist_between_sets(set1, set2, rover_shortest_paths_for_rover):
    """Finds the minimum distance between any node in set1 and any node in set2 using the provided shortest paths."""
    if not set1 or not set2:
        return float('inf')

    min_d = float('inf')
    for node1 in set1:
        if node1 in rover_shortest_paths_for_rover:
             for node2 in set2:
                min_d = min(min_d, rover_shortest_paths_for_rover[node1].get(node2, float('inf')))
    return min_d


class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the minimum number of actions required to achieve
    each uncommunicated goal predicate (soil data, rock data, image data)
    independently, and sums these minimum costs. It considers navigation,
    sampling/imaging, calibration, and communication actions. Navigation costs
    are estimated using precomputed shortest paths on the waypoint graph for
    each rover. Resource conflicts (like a single store or camera) are ignored
    by calculating the minimum cost assuming the best-suited available rover
    performs the task.

    # Assumptions
    - Each uncommunicated goal can be achieved independently by a suitable rover.
    - A suitable rover exists for each goal type (soil, rock, imaging).
    - Store state (full/empty) is mostly ignored for simplicity, except potentially for an initial drop action if a rover starts with a full store and needs to sample. (Decided to skip drop cost for simplicity in the final version).
    - Camera calibration, if needed, adds a fixed cost (1 action) and its navigation is assumed to be integrated with or similar to navigation for imaging.
    - There is at least one lander, and its location is static.
    - `can_traverse` implies `visible`. Communication requires `visible` between rover and lander locations.

    # Heuristic Initialization
    - Parses the static facts and initial state to identify:
        - All rovers, waypoints, landers, stores, cameras, objectives, modes.
        - Lander location(s).
        - Rover capabilities (soil, rock, imaging).
        - Store ownership (`store_of`).
        - Camera properties (`on_board`, `supports`, `calibration_target`).
        - Waypoints visible from objectives (`visible_from`).
        - Waypoint connectivity for each rover (`can_traverse`).
    - Precomputes shortest path distances between all pairs of waypoints for each rover based on `can_traverse` facts.
    - Identifies communication waypoints (visible from any lander location).
    - Maps objectives to imaging waypoints and calibration targets to calibration waypoints.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize total heuristic cost to 0.
    2. Extract dynamic information from the current state: rover locations, `have_soil_analysis`, `have_rock_analysis`, `have_image`, `calibrated` camera status.
    3. Iterate through each goal predicate specified in the task.
    4. For each goal predicate not yet satisfied in the current state:
        a. Determine the type of goal (soil, rock, image).
        b. Calculate the minimum estimated cost to achieve *this specific goal predicate*, considering all suitable rovers independently.
        c. For a `communicated_soil_data ?w` goal:
            - If the current rover being considered already has `(have_soil_analysis ?r ?w)`:
                - Cost = 1 (communicate) + navigation cost for that rover from its current location to any communication waypoint.
            - If the current rover does not have the sample:
                - Cost = 1 (sample) + 1 (communicate) + navigation cost for the rover from its current location to `?w` + navigation cost from `?w` to any communication waypoint.
            - Find the minimum cost among all suitable rovers.
        d. For a `communicated_rock_data ?w` goal:
            - Similar logic as soil data, using rock analysis capabilities and facts.
        e. For a `communicated_image_data ?o ?m` goal:
            - If the current rover being considered already has `(have_image ?r ?o ?m)`:
                - Cost = 1 (communicate) + navigation cost for that rover from its current location to any communication waypoint.
            - If the current rover does not have the image:
                - Cost = 1 (take_image) + 1 (communicate) + navigation cost for the rover from its current location to any imaging waypoint for `?o` + navigation cost from any imaging waypoint for `?o` to any communication waypoint.
                - If the camera needed for mode `?m` on that rover is not calibrated, add 1 for calibration action.
            - Find the minimum cost among all suitable rovers.
        f. Add the minimum estimated cost for this unachieved goal predicate to the total heuristic cost. If the goal is unreachable by any rover (e.g., no suitable rover, disconnected graph), the cost is infinite (or a large number), effectively pruning that state in greedy search.
    5. Return the total heuristic cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # --- Parse Objects ---
        # Infer objects from static facts and initial state facts.
        all_facts = set(static_facts) | set(task.initial_state)

        self.rovers = set()
        self.waypoints = set()
        self.landers = set()
        self.stores = set()
        self.cameras = set()
        self.modes = set()
        self.objectives = set()

        for fact in all_facts:
             parts = get_parts(fact)
             if len(parts) > 1:
                 pred = parts[0]
                 args = parts[1:]
                 if pred in ("at", "can_traverse", "equipped_for_soil_analysis",
                             "equipped_for_rock_analysis", "equipped_for_imaging",
                             "store_of", "on_board", "calibrated", "have_soil_analysis",
                             "have_rock_analysis", "have_image", "drop"):
                     if args[0].startswith('rover'): self.rovers.add(args[0])
                 if pred in ("at", "at_lander", "can_traverse", "visible",
                             "at_soil_sample", "at_rock_sample", "visible_from",
                             "communicate_soil_data", "communicate_rock_data",
                             "calibrate", "take_image"):
                     for arg in args:
                         if arg.startswith('waypoint'): self.waypoints.add(arg)
                 if pred == "at_lander":
                     if args[0].startswith('lander'): self.landers.add(args[0])
                 if pred in ("empty", "full", "store_of"):
                     if args[0].startswith('store'): self.stores.add(args[0])
                 if pred in ("calibrated", "supports", "calibration_target", "on_board", "take_image"):
                     if args[0].startswith('camera'): self.cameras.add(args[0])
                 if pred in ("supports", "take_image", "communicated_image_data"):
                     for arg in args:
                          if arg in ('high_res', 'low_res', 'colour'): self.modes.add(arg)
                 if pred in ("calibration_target", "visible_from", "have_image", "communicated_image_data"):
                     if args[0].startswith('objective'): self.objectives.add(args[0])


        # --- Parse Static Facts ---
        self.lander_location = {} # lander -> waypoint
        self.rover_capabilities = defaultdict(set) # rover -> set(capabilities: 'soil', 'rock', 'imaging')
        self.rover_stores = {} # rover -> store
        self.rover_cameras = defaultdict(set) # rover -> set(cameras)
        self.camera_info = {} # camera -> {'modes': set(), 'cal_target': objective, 'on_board': rover}
        self.objective_imaging_wps = defaultdict(set) # objective -> set(waypoint)
        self.objective_cal_wps = defaultdict(set) # camera -> set(waypoint) visible from its target
        self.communication_wps = set() # set(waypoint) visible from lander

        # Graph for navigation
        self.rover_graph = defaultdict(lambda: defaultdict(set)) # rover -> waypoint -> set(neighbor_waypoint)

        lander_locs_temp = set() # Collect lander locations first
        for fact in static_facts:
            parts = get_parts(fact)
            pred = parts[0]
            args = parts[1:]

            if pred == "at_lander":
                lander, wp = args
                self.lander_location[lander] = wp
                lander_locs_temp.add(wp)
            elif pred.startswith("equipped_for_"):
                cap_type = pred.split("_")[-2] # soil, rock, imaging
                rover = args[0]
                self.rover_capabilities[rover].add(cap_type)
            elif pred == "store_of":
                store, rover = args
                self.rover_stores[rover] = store
            elif pred == "on_board":
                camera, rover = args
                self.rover_cameras[rover].add(camera)
                if camera not in self.camera_info: self.camera_info[camera] = {}
                self.camera_info[camera]['on_board'] = rover
            elif pred == "supports":
                camera, mode = args
                if camera not in self.camera_info: self.camera_info[camera] = {}
                if 'modes' not in self.camera_info[camera]: self.camera_info[camera]['modes'] = set()
                self.camera_info[camera]['modes'].add(mode)
            elif pred == "calibration_target":
                camera, objective = args
                if camera not in self.camera_info: self.camera_info[camera] = {}
                self.camera_info[camera]['cal_target'] = objective
            elif pred == "visible_from":
                objective, wp = args
                self.objective_imaging_wps[objective].add(wp)
            elif pred == "can_traverse":
                rover, wp_from, wp_to = args
                self.rover_graph[rover][wp_from].add(wp_to)

        # Populate calibration waypoints based on calibration_target and visible_from
        for camera, info in self.camera_info.items():
             cal_target = info.get('cal_target')
             if cal_target and cal_target in self.objective_imaging_wps:
                 self.objective_cal_wps[camera] = self.objective_imaging_wps[cal_target]


        # Populate communication waypoints based on lander locations and visible
        for fact in static_facts:
             parts = get_parts(fact)
             pred = parts[0]
             args = parts[1:]
             if pred == "visible":
                 wp1, wp2 = args
                 if wp1 in lander_locs_temp:
                     self.communication_wps.add(wp2)
                 if wp2 in lander_locs_temp:
                     self.communication_wps.add(wp1)


        # --- Precompute Shortest Paths ---
        self.rover_shortest_paths = {} # rover -> start_wp -> end_wp -> distance
        for rover in self.rovers:
            self.rover_shortest_paths[rover] = {}
            rover_specific_graph = self.rover_graph.get(rover, {})
            for start_wp in self.waypoints:
                 self.rover_shortest_paths[rover][start_wp] = bfs(rover_specific_graph, start_wp, self.waypoints)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Check if goal is reached
        if self.goals <= state:
            return 0

        # --- Extract Dynamic State Information ---
        current_rover_pos = {} # rover -> waypoint
        have_soil = defaultdict(dict) # rover -> waypoint -> bool
        have_rock = defaultdict(dict) # rover -> waypoint -> bool
        have_image = defaultdict(lambda: defaultdict(dict)) # rover -> objective -> mode -> bool
        # store_status = {} # store -> 'empty' or 'full' # Not used in simplified heuristic
        calibrated_cameras = set() # set(camera)

        for fact in state:
            parts = get_parts(fact)
            pred = parts[0]
            args = parts[1:]

            if pred == "at" and args[0] in self.rovers:
                rover, wp = args
                current_rover_pos[rover] = wp
            elif pred == "have_soil_analysis":
                rover, wp = args
                have_soil[rover][wp] = True
            elif pred == "have_rock_analysis":
                rover, wp = args
                have_rock[rover][wp] = True
            elif pred == "have_image":
                rover, objective, mode = args
                have_image[rover][objective][mode] = True
            # elif pred in ("empty", "full"): # Not used in simplified heuristic
            #     store = args[0]
            #     store_status[store] = pred # 'empty' or 'full'
            elif pred == "calibrated":
                camera, rover = args
                calibrated_cameras.add(camera)

        total_cost = 0

        # --- Calculate Cost for Unachieved Goals ---
        for goal in self.goals:
            if goal in state:
                continue # Goal already achieved

            parts = get_parts(goal)
            pred = parts[0]

            if pred == "communicated_soil_data":
                wp = parts[1]
                min_goal_cost = float('inf')

                suitable_rovers = [r for r in self.rovers if 'soil' in self.rover_capabilities.get(r, set()) and r in self.rover_stores]

                if not suitable_rovers:
                    return float('inf')

                for rover in suitable_rovers:
                    rover_cost = 0
                    rover_current_wp = current_rover_pos.get(rover)

                    if rover_current_wp is None: continue

                    rover_paths = self.rover_shortest_paths.get(rover, {})

                    if have_soil.get(rover, {}).get(wp, False):
                         # This rover has the sample, just needs to communicate
                         rover_cost += 1 # communicate action
                         nav_to_comm = min_dist_to_set(rover_current_wp, self.communication_wps, rover_paths)
                         if nav_to_comm == float('inf'): rover_cost = float('inf')
                         else: rover_cost += nav_to_comm
                    else:
                        # This rover does NOT have the sample. Needs to sample and communicate.
                        rover_cost += 1 # sample action
                        rover_cost += 1 # communicate action

                        # Navigation to sample location
                        nav_to_sample = min_dist_to_set(rover_current_wp, {wp}, rover_paths)
                        if nav_to_sample == float('inf'):
                            rover_cost = float('inf')
                        else:
                            rover_cost += nav_to_sample

                            # Navigation from sample location to communication point
                            nav_sample_to_comm = min_dist_to_set(wp, self.communication_wps, rover_paths)
                            if nav_sample_to_comm == float('inf'):
                                rover_cost = float('inf')
                            else:
                                rover_cost += nav_sample_to_comm

                    min_goal_cost = min(min_goal_cost, rover_cost)

                if min_goal_cost == float('inf'):
                     return float('inf')

                total_cost += min_goal_cost

            elif pred == "communicated_rock_data":
                wp = parts[1]
                min_goal_cost = float('inf')

                suitable_rovers = [r for r in self.rovers if 'rock' in self.rover_capabilities.get(r, set()) and r in self.rover_stores]

                if not suitable_rovers:
                    return float('inf')

                for rover in suitable_rovers:
                    rover_cost = 0
                    rover_current_wp = current_rover_pos.get(rover)

                    if rover_current_wp is None: continue

                    rover_paths = self.rover_shortest_paths.get(rover, {})

                    if have_rock.get(rover, {}).get(wp, False):
                         # This rover has the sample, just needs to communicate
                         rover_cost += 1 # communicate action
                         nav_to_comm = min_dist_to_set(rover_current_wp, self.communication_wps, rover_paths)
                         if nav_to_comm == float('inf'): rover_cost = float('inf')
                         else: rover_cost += nav_to_comm
                    else:
                        # Need to sample and communicate
                        rover_cost += 1 # sample action
                        rover_cost += 1 # communicate action

                        # Navigation to sample location
                        nav_to_sample = min_dist_to_set(rover_current_wp, {wp}, rover_paths)
                        if nav_to_sample == float('inf'):
                            rover_cost = float('inf')
                        else:
                            rover_cost += nav_to_sample

                            # Navigation from sample location to communication point
                            nav_sample_to_comm = min_dist_to_set(wp, self.communication_wps, rover_paths)
                            if nav_sample_to_comm == float('inf'):
                                rover_cost = float('inf')
                            else:
                                rover_cost += nav_sample_to_comm

                    min_goal_cost = min(min_goal_cost, rover_cost)

                if min_goal_cost == float('inf'):
                     return float('inf')

                total_cost += min_goal_cost

            elif pred == "communicated_image_data":
                objective, mode = parts[1], parts[2]
                min_goal_cost = float('inf')

                suitable_rovers = []
                for r in self.rovers:
                    if 'imaging' in self.rover_capabilities.get(r, set()):
                        for cam in self.rover_cameras.get(r, set()):
                            if mode in self.camera_info.get(cam, {}).get('modes', set()):
                                suitable_rovers.append((r, cam)) # Store rover and the specific camera
                                break # Found a suitable camera for this rover/mode

                if not suitable_rovers:
                    return float('inf')

                for rover, camera in suitable_rovers:
                    rover_cost = 0
                    rover_current_wp = current_rover_pos.get(rover)

                    if rover_current_wp is None: continue

                    rover_paths = self.rover_shortest_paths.get(rover, {})

                    if have_image.get(rover, {}).get(objective, {}).get(mode, False):
                        # This rover has the image, just needs to communicate
                        rover_cost += 1 # communicate action
                        nav_to_comm = min_dist_to_set(rover_current_wp, self.communication_wps, rover_paths)
                        if nav_to_comm == float('inf'): rover_cost = float('inf')
                        else: rover_cost += nav_to_comm
                    else:
                        # Need to take image and communicate
                        rover_cost += 1 # take_image action
                        rover_cost += 1 # communicate action

                        # Check calibration state for this specific camera on this rover
                        if camera not in calibrated_cameras:
                            rover_cost += 1 # calibrate action
                            # Navigation for calibration is assumed integrated or minimal extra cost

                        # Navigation to imaging location
                        imaging_wps = self.objective_imaging_wps.get(objective, set())
                        if not imaging_wps: # No waypoint to view objective from
                             rover_cost = float('inf')
                        else:
                            nav_to_imaging = min_dist_to_set(rover_current_wp, imaging_wps, rover_paths)

                            if nav_to_imaging == float('inf'):
                                rover_cost = float('inf') # Cannot reach imaging location
                            else:
                                rover_cost += nav_to_imaging

                                # Navigation from imaging location to communication point
                                # Need min dist from *any* imaging wp to *any* comm wp
                                nav_imaging_to_comm = min_dist_between_sets(imaging_wps, self.communication_wps, rover_paths)
                                if nav_imaging_to_comm == float('inf'):
                                    rover_cost = float('inf') # Cannot reach communication point from imaging location
                                else:
                                    rover_cost += nav_imaging_to_comm

                    min_goal_cost = min(min_goal_cost, rover_cost)

                if min_goal_cost == float('inf'):
                     return float('inf')

                total_cost += min_goal_cost

        return total_cost
