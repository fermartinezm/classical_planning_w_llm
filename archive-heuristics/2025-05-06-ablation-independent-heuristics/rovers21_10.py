import collections
from heuristics.heuristic_base import Heuristic
# Assuming Task class is available in the environment where this heuristic runs

class roversHeuristic(Heuristic):
    """
    Summary:
        Domain-dependent heuristic for the rovers domain.
        Estimates the cost to reach a goal state by summing the estimated costs
        for each unsatisfied goal fact. The cost for each goal fact is estimated
        based on the minimum actions required by any suitable rover, including
        navigation, sampling/imaging, and communication. Navigation costs are
        estimated using precomputed shortest path distances (BFS) on the
        waypoint graph for each rover.

    Assumptions:
        - The heuristic assumes that if a soil/rock sample goal exists for a
          waypoint, the sample was initially present at that waypoint unless
          already collected.
        - The heuristic simplifies complex sequences (like calibrate -> image -> communicate)
          by estimating costs for sequential steps and picking the
          closest waypoint for intermediate steps (calibration/imaging) rather
          than finding the globally optimal path sequence.
        - Dropping a sample is always possible if a store is full.
        - Rovers and cameras required for goals exist and have the necessary static properties.
        - The heuristic returns a large finite value (1000) for goals deemed impossible
          (e.g., unreachable waypoints), ensuring the total heuristic is finite
          but large for unsolvable states or states where subgoals are unreachable.
        - The heuristic is non-admissible; it may overestimate costs.

    Heuristic Initialization:
        The constructor processes the static facts from the task description
        to build efficient data structures:
        - Mapping lander to its location.
        - Identifying rovers with specific equipment (soil, rock, imaging).
        - Mapping stores to their owning rovers.
        - Mapping cameras to their properties (on-board rover, supported modes, calibration target).
        - Mapping objectives to waypoints from which they are visible.
        - Identifying waypoints visible from the lander (communication points).
        - Building a traversal graph (adjacency list) for each rover based on 'can_traverse' facts.
        - Precomputing shortest path distances between all pairs of waypoints
          for each rover using BFS.
        - Identifying initial locations of soil and rock samples.

    Step-By-Step Thinking for Computing Heuristic:
        1.  Get the current state and the set of goal facts.
        2.  Identify the set of unsatisfied goal facts.
        3.  Initialize the total heuristic value `h` to 0.
        4.  Extract dynamic information from the current state (rover locations,
            store status, collected samples, calibrated cameras, collected images).
        5.  For each unsatisfied goal fact:
            a.  Parse the goal fact to determine its type (soil, rock, or image) and parameters.
            b.  Estimate the cost to achieve this specific goal fact from the current state.
                This estimation involves finding the minimum cost over all suitable rovers
                (and cameras for image goals). For a given rover/camera:
                i.  If the required sample or image is already held by this rover,
                    the cost to obtain it is 0. The starting location for communication
                    navigation is the current location of this rover.
                ii. If the item is not held, estimate the cost to obtain it sequentially:
                    -   **For soil/rock:** Calculate navigation cost from the rover's
                        current location to the sample waypoint. Add cost for dropping
                        sample if store is full (1) and the sample action (1). The
                        starting location for communication navigation is the sample waypoint.
                    -   **For image:** If calibration is needed, calculate navigation cost
                        from the rover's current location to the closest calibration waypoint
                        visible from the target, and add the calibrate action cost (1).
                        The rover's estimated location is updated to this calibration waypoint.
                        Then, calculate navigation cost from the rover's current estimated
                        location (after calibration, or initial location if calibrated)
                        to the closest imaging waypoint visible from the objective, and
                        add the take_image action cost (1). The rover's estimated location
                        is updated to this imaging waypoint.
                    -   The total estimated cost to obtain the item is the sum of these steps.
                iii. Estimate the cost to communicate: Calculate navigation cost from the
                     rover's current estimated location (after obtaining the item) to the
                     closest communication waypoint, and add the communicate action cost (1).
                iv. The total estimated cost for this rover/camera is the sum of the
                    estimated cost to obtain the item and the estimated cost to communicate.
                v.  Take the minimum total cost over all suitable rovers/cameras.
                vi. If the minimum cost is infinite (e.g., due to unreachable waypoints
                    or no suitable rover/camera), use a large finite value (1000).
            c.  Add the estimated cost for this goal fact to the total heuristic value `h`.
        6.  Return the total heuristic value `h`.
    """

    def __init__(self, task):
        super().__init__()
        self.goals = task.goals
        self.static = task.static
        self.initial_state = task.initial_state # Need initial state for sample locations

        # --- Process Static Information ---
        self.lander_location = {} # lander -> waypoint
        self.rover_capabilities = collections.defaultdict(set) # capability_type -> {rover}
        self.store_owners = {} # store -> rover
        self.camera_info = {} # camera -> {'rover': r, 'modes': {m}, 'calibration_target': t}
        self.objective_visibility = collections.defaultdict(set) # objective -> {waypoint}
        self.rover_traversal_graph = collections.defaultdict(lambda: collections.defaultdict(list)) # rover -> waypoint -> [neighbor_waypoint]

        # Temporary structure to build camera_info
        camera_modes = collections.defaultdict(set)
        camera_targets = {}
        camera_on_board = {}

        # Collect all waypoints mentioned in can_traverse to build complete graph nodes
        all_waypoints_in_traverse = set()

        for fact_str in self.static:
            fact = self.parse_fact(fact_str)
            if fact[0] == 'at_lander':
                self.lander_location[fact[1]] = fact[2]
            elif fact[0] == 'equipped_for_soil_analysis':
                self.rover_capabilities['soil'].add(fact[1])
            elif fact[0] == 'equipped_for_rock_analysis':
                self.rover_capabilities['rock'].add(fact[1])
            elif fact[0] == 'equipped_for_imaging':
                self.rover_capabilities['imaging'].add(fact[1])
            elif fact[0] == 'store_of':
                self.store_owners[fact[1]] = fact[2]
            elif fact[0] == 'visible_from':
                self.objective_visibility[fact[1]].add(fact[2])
            elif fact[0] == 'can_traverse':
                rover, wp1, wp2 = fact[1], fact[2], fact[3]
                self.rover_traversal_graph[rover][wp1].append(wp2)
                all_waypoints_in_traverse.add(wp1)
                all_waypoints_in_traverse.add(wp2)
            elif fact[0] == 'supports':
                camera_modes[fact[1]].add(fact[2])
            elif fact[0] == 'calibration_target':
                camera_targets[fact[1]] = fact[2]
            elif fact[0] == 'on_board':
                camera_on_board[fact[1]] = fact[2]

        # Ensure all waypoints in traversal graph have an entry, even if empty
        for rover in self.rover_traversal_graph:
             for wp in all_waypoints_in_traverse:
                 if wp not in self.rover_traversal_graph[rover]:
                     self.rover_traversal_graph[rover][wp] = []


        # Build camera_info
        all_cameras = set(camera_modes.keys()) | set(camera_targets.keys()) | set(camera_on_board.keys())
        for cam in all_cameras:
             if cam in camera_on_board: # Must be on board a rover to be useful
                 self.camera_info[cam] = {
                     'rover': camera_on_board.get(cam),
                     'modes': camera_modes.get(cam, set()),
                     'calibration_target': camera_targets.get(cam)
                 }

        # Identify communication waypoints (visible from any lander location)
        self.communication_waypoints = set()
        lander_locs = set(self.lander_location.values())

        # Need visible facts from static info
        visible_facts = set()
        for fact_str in self.static:
             if fact_str.startswith('(visible '):
                 visible_facts.add(self.parse_fact(fact_str))

        # Collect all waypoints mentioned in visible facts as potential communication points
        all_waypoints_in_visible = set()
        for fact in visible_facts:
             all_waypoints_in_visible.add(fact[1])
             all_waypoints_in_visible.add(fact[2])

        for wp1 in all_waypoints_in_visible:
             for wp2 in lander_locs:
                 if ('visible', wp1, wp2) in visible_facts:
                     self.communication_waypoints.add(wp1)
                     break # Found one lander it's visible from

        # Precompute distances for each rover
        self.rover_distances = {}
        for rover, graph in self.rover_traversal_graph.items():
            self.rover_distances[rover] = self.precompute_distances(graph)

    def parse_fact(self, fact_string):
        """Helper to parse PDDL fact string into a tuple."""
        # Removes leading/trailing parens and splits by space
        parts = fact_string[1:-1].split()
        return tuple(parts)

    def precompute_distances(self, graph):
        """Computes all-pairs shortest paths using BFS for a given graph."""
        distances = {}
        waypoints = list(graph.keys())
        for start_node in waypoints:
            distances[start_node] = {}
            queue = collections.deque([(start_node, 0)])
            visited = {start_node}
            distances[start_node][start_node] = 0

            while queue:
                current_node, dist = queue.popleft()
                if current_node in graph:
                    for neighbor in graph[current_node]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            distances[start_node][neighbor] = dist + 1
                            queue.append((neighbor, dist + 1))
        return distances

    def get_nav_distance(self, rover, start_wp, end_wp):
        """Gets precomputed navigation distance."""
        # Handle cases where start_wp or end_wp might not be in the precomputed map
        # (e.g., a waypoint only mentioned in initial state but not traversable)
        if rover not in self.rover_distances or start_wp not in self.rover_distances[rover] or end_wp not in self.rover_distances[rover].get(start_wp, {}):
             return float('inf')
        return self.rover_distances[rover][start_wp][end_wp]


    def get_min_nav_distance(self, rover, start_wp, target_wps):
        """Gets minimum precomputed navigation distance to any target waypoint."""
        min_dist = float('inf')
        rover_dist_map = self.rover_distances.get(rover, {})
        start_dist_map = rover_dist_map.get(start_wp, {})

        if not start_dist_map: # Start waypoint not in graph or unreachable
             return float('inf')

        for target_wp in target_wps:
            dist = start_dist_map.get(target_wp, float('inf'))
            min_dist = min(min_dist, dist)
        return min_dist

    def get_closest_waypoint(self, rover, start_wp, target_wps):
        """Finds the target waypoint closest to the start waypoint for a given rover."""
        min_dist = float('inf')
        closest_wp = None
        rover_dist_map = self.rover_distances.get(rover, {})
        start_dist_map = rover_dist_map.get(start_wp, {})

        if not start_dist_map: # Start waypoint not in graph or unreachable
             return None

        for target_wp in target_wps:
            dist = start_dist_map.get(target_wp, float('inf'))
            if dist < min_dist:
                min_dist = dist
                closest_wp = target_wp
        return closest_wp # Returns None if no target is reachable


    def __call__(self, node):
        state = node.state
        unsat_goals = self.goals - state
        h = 0

        # If no unsatisfied goals, heuristic is 0
        if not unsat_goals:
            return 0

        # Extract dynamic state information
        rover_locations = {} # rover -> waypoint
        store_status = {} # store -> 'empty' or 'full'
        have_soil = set() # (rover, waypoint)
        have_rock = set() # (rover, waypoint)
        calibrated_cameras = set() # (camera, rover)
        have_image = set() # (rover, objective, mode)
        # Communicated goals are implicitly handled by unsat_goals

        for fact_str in state:
            fact = self.parse_fact(fact_str)
            if fact[0] == 'at' and fact[1] in self.rover_traversal_graph: # Check if it's a rover with traversal info
                 rover_locations[fact[1]] = fact[2]
            elif fact[0] == 'empty':
                 store_status[fact[1]] = 'empty'
            elif fact[0] == 'full':
                 store_status[fact[1]] = 'full'
            elif fact[0] == 'have_soil_analysis':
                 have_soil.add((fact[1], fact[2]))
            elif fact[0] == 'have_rock_analysis':
                 have_rock.add((fact[1], fact[2]))
            elif fact[0] == 'calibrated':
                 calibrated_cameras.add((fact[1], fact[2]))
            elif fact[0] == 'have_image':
                 have_image.add((fact[1], fact[2], fact[3]))

        # Process unsatisfied goals
        for goal_str in unsat_goals:
            goal = self.parse_fact(goal_str)
            goal_type = goal[0]

            if goal_type == 'communicated_soil_data':
                waypoint = goal[1]
                h += self.estimate_soil_goal_cost(waypoint, rover_locations, store_status, have_soil)
            elif goal_type == 'communicated_rock_data':
                waypoint = goal[1]
                h += self.estimate_rock_goal_cost(waypoint, rover_locations, store_status, have_rock)
            elif goal_type == 'communicated_image_data':
                objective, mode = goal[1], goal[2]
                h += self.estimate_image_goal_cost(objective, mode, rover_locations, calibrated_cameras, have_image)

        return h

    def estimate_soil_goal_cost(self, waypoint, rover_locations, store_status, have_soil):
        # Check if sample is already collected by any rover
        rover_with_sample = None
        for r, w in have_soil:
            if w == waypoint:
                rover_with_sample = r
                break # Found one, use this rover for communication cost

        if rover_with_sample:
            # Sample already collected, just need to communicate
            rover = rover_with_sample
            rover_loc = rover_locations.get(rover)
            if rover_loc is None: return 1000 # Should not happen in a valid state

            # Cost to communicate = navigate from current loc to comm point + communicate action
            min_nav_to_comm = self.get_min_nav_distance(rover, rover_loc, self.communication_waypoints)
            if min_nav_to_comm == float('inf'): return 1000 # Cannot reach comm point
            cost_communicate = min_nav_to_comm + 1
            return cost_communicate

        else:
            # Need to collect sample and then communicate
            best_rover_cost = float('inf')

            for rover in self.rover_capabilities.get('soil', set()):
                rover_loc = rover_locations.get(rover)
                if rover_loc is None: continue # Rover location unknown

                # Cost to get sample = navigate to sample + (drop if needed) + sample action
                nav_to_sample_cost = self.get_nav_distance(rover, rover_loc, waypoint)
                if nav_to_sample_cost == float('inf'): continue # Cannot reach sample location

                store = None
                for s, r in self.store_owners.items():
                    if r == rover:
                        store = s
                        break
                drop_cost = 0
                if store and store_status.get(store) == 'full':
                     drop_cost = 1

                cost_get_sample = nav_to_sample_cost + drop_cost + 1

                # Now estimate cost to communicate *from the sample location*
                # Assuming the rover is at 'waypoint' after sampling
                min_nav_to_comm = self.get_min_nav_distance(rover, waypoint, self.communication_waypoints)
                if min_nav_to_comm == float('inf'): continue # Cannot reach comm point from sample location

                cost_communicate = min_nav_to_comm + 1

                total_cost_for_rover = cost_get_sample + cost_communicate

                best_rover_cost = min(best_rover_cost, total_cost_for_rover)

            return best_rover_cost if best_rover_cost != float('inf') else 1000

    def estimate_rock_goal_cost(self, waypoint, rover_locations, store_status, have_rock):
        # Check if sample is already collected by any rover
        rover_with_sample = None
        for r, w in have_rock:
            if w == waypoint:
                rover_with_sample = r
                break # Found one, use this rover for communication cost

        if rover_with_sample:
            # Sample already collected, just need to communicate
            rover = rover_with_sample
            rover_loc = rover_locations.get(rover)
            if rover_loc is None: return 1000 # Should not happen

            # Cost to communicate = navigate from current loc to comm point + communicate action
            min_nav_to_comm = self.get_min_nav_distance(rover, rover_loc, self.communication_waypoints)
            if min_nav_to_comm == float('inf'): return 1000 # Cannot reach comm point
            cost_communicate = min_nav_to_comm + 1
            return cost_communicate

        else:
            # Need to collect sample and then communicate
            best_rover_cost = float('inf')

            for rover in self.rover_capabilities.get('rock', set()):
                rover_loc = rover_locations.get(rover)
                if rover_loc is None: continue # Rover location unknown

                # Cost to get sample = navigate to sample + (drop if needed) + sample action
                nav_to_sample_cost = self.get_nav_distance(rover, rover_loc, waypoint)
                if nav_to_sample_cost == float('inf'): continue # Cannot reach sample location

                store = None
                for s, r in self.store_owners.items():
                    if r == rover:
                        store = s
                        break
                drop_cost = 0
                if store and store_status.get(store) == 'full':
                     drop_cost = 1

                cost_get_sample = nav_to_sample_cost + drop_cost + 1

                # Now estimate cost to communicate *from the sample location*
                # Assuming the rover is at 'waypoint' after sampling
                min_nav_to_comm = self.get_min_nav_distance(rover, waypoint, self.communication_waypoints)
                if min_nav_to_comm == float('inf'): continue # Cannot reach comm point from sample location

                cost_communicate = min_nav_to_comm + 1

                total_cost_for_rover = cost_get_sample + cost_communicate

                best_rover_cost = min(best_rover_cost, total_cost_for_rover)

            return best_rover_cost if best_rover_cost != float('inf') else 1000

    def estimate_image_goal_cost(self, objective, mode, rover_locations, calibrated_cameras, have_image):
        # Check if image is already collected by any rover
        rover_with_image = None
        for r, o, m in have_image:
            if o == objective and m == mode:
                rover_with_image = r
                break # Found one, use this rover for communication cost

        if rover_with_image:
            # Image already collected, just need to communicate
            rover = rover_with_image
            rover_loc = rover_locations.get(rover)
            if rover_loc is None: return 1000 # Should not happen

            # Cost to communicate = navigate from current loc to comm point + communicate action
            min_nav_to_comm = self.get_min_nav_distance(rover, rover_loc, self.communication_waypoints)
            if min_nav_to_comm == float('inf'): return 1000 # Cannot reach comm point
            cost_communicate = min_nav_to_comm + 1
            return cost_communicate

        else:
            # Need to collect image and then communicate
            best_rover_camera_cost = float('inf')

            for rover in self.rover_capabilities.get('imaging', set()):
                rover_loc = rover_locations.get(rover)
                if rover_loc is None: continue # Rover location unknown

                for camera_name, cam_info in self.camera_info.items():
                    if cam_info['rover'] == rover and mode in cam_info['modes']:
                        # Found a suitable rover/camera combo

                        # --- Estimate cost to get image ---
                        cost_get_image = 0
                        current_loc = rover_loc # Start point for the sequence of navs

                        # 1. Calibration (if needed)
                        calibrated = (camera_name, rover) in calibrated_cameras
                        if not calibrated:
                            cal_target = cam_info.get('calibration_target')
                            if cal_target is None: continue # Camera has no calibration target?
                            cal_waypoints = self.objective_visibility.get(cal_target, set())
                            if not cal_waypoints: continue # No waypoint to calibrate from?

                            # Find the closest calibration waypoint
                            min_nav_to_cal = self.get_min_nav_distance(rover, current_loc, cal_waypoints)
                            if min_nav_to_cal == float('inf'): continue

                            cost_get_image += min_nav_to_cal + 1 # Nav to cal + calibrate
                            # Update current location to the chosen calibration waypoint
                            closest_cal_wp = self.get_closest_waypoint(rover, current_loc, cal_waypoints)
                            if closest_cal_wp is None: continue # Should not happen if min_nav_to_cal was finite
                            current_loc = closest_cal_wp

                        # 2. Navigation to imaging point + Take image cost
                        img_waypoints = self.objective_visibility.get(objective, set())
                        if not img_waypoints: continue

                        min_nav_to_img = self.get_min_nav_distance(rover, current_loc, img_waypoints)
                        if min_nav_to_img == float('inf'): continue

                        cost_get_image += min_nav_to_img + 1 # Nav to img + take_image
                        # Update current location to the chosen imaging waypoint
                        closest_img_wp = self.get_closest_waypoint(rover, current_loc, img_waypoints)
                        if closest_img_wp is None: continue # Should not happen if min_nav_to_img was finite
                        current_loc = closest_img_wp

                        # --- Estimate cost to communicate ---
                        # Navigation to communication point + Communicate cost
                        comm_waypoints = self.communication_waypoints
                        if not comm_waypoints: continue

                        min_nav_to_comm = self.get_min_nav_distance(rover, current_loc, comm_waypoints)
                        if min_nav_to_comm == float('inf'): continue

                        cost_communicate = min_nav_to_comm + 1

                        total_cost_for_combo = cost_get_image + cost_communicate

                        best_rover_camera_cost = min(best_rover_camera_cost, total_cost_for_combo)

            return best_rover_camera_cost if best_rover_camera_cost != float('inf') else 1000
