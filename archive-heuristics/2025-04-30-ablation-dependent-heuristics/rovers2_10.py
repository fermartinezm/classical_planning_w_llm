import collections
import math

class roversHeuristic:
    """
    Domain-dependent heuristic for the Rovers domain.

    Summary:
        This heuristic estimates the cost to reach a goal state by summing
        the estimated costs for each unachieved goal predicate independently.
        For each unachieved goal (communicating soil data, rock data, or image data),
        the heuristic estimates the minimum number of actions required to achieve
        that specific goal from the current state, plus the minimum travel cost
        involved. Travel costs are estimated using precomputed shortest paths
        on the navigation graph, taking the minimum travel cost from *any*
        rover's current location to the required waypoint(s). The heuristic
        ignores negative effects of actions (like a camera becoming uncalibrated
        or a store becoming full) and resource constraints (like limited store
        capacity or which specific rover has a sample/image). It returns
        infinity if an unachieved goal is determined to be impossible
        (e.g., no equipped rover, unreachable location).

    Assumptions:
        - The problem instance is solvable if the heuristic is finite.
        - Each rover has at most one store (though store constraints are ignored).
        - The initial state correctly lists all available soil/rock samples.
        - Calibration targets are visible from the same waypoints as the objectives
          they are used to image (simplified travel cost for imaging).
        - Any rover can communicate data if it is at a waypoint visible from the lander.

    Heuristic Initialization:
        The constructor precomputes static information from the task, including:
        - Rover capabilities (soil, rock, imaging).
        - Store ownership.
        - Camera details (on which rover, calibration target, supported modes).
        - Lander locations.
        - Waypoint visibility graph for navigation.
        - Objective visibility waypoints for imaging.
        - Rover traversability graphs.
        - All-pairs shortest paths for each rover on its traversability graph using BFS.
        - Waypoints visible *from* lander locations (communication points).
        - Initial locations of soil and rock samples.

    Step-By-Step Thinking for Computing Heuristic:
        1.  Check if the current state is a goal state. If yes, the heuristic is 0.
        2.  Initialize the total heuristic value `h` to 0.
        3.  Parse the current state to extract dynamic information: rover locations,
            which rovers have which samples/images. (Store status and calibration
            status are ignored in this simplified heuristic).
        4.  Identify all waypoints from which communication is possible (visible from any lander).
        5.  Iterate through each goal predicate defined in the task's goals.
        6.  If a goal predicate `g` is already true in the current state, continue to the next goal.
        7.  If `g` is not true, calculate the estimated cost `cost_g` to achieve it independently:
            a.  **For `(communicated_soil_data ?w)`:**
                -   `cost_g` starts at 1 (for the `communicate_soil_data` action).
                -   Check if any rover currently has `(have_soil_analysis ?r ?w)`.
                -   If not:
                    -   Check if any rover is equipped for soil analysis and if the sample `(at_soil_sample ?w)` existed in the initial state. If not, this goal is impossible, return `float('inf')`.
                    -   Add 1 to `cost_g` (for the `sample_soil` action).
                    -   Add the minimum travel cost from any rover's current location to waypoint `w` (using precomputed distances). If unreachable, return `float('inf')`.
                -   Add the minimum travel cost from any rover's current location to any lander communication waypoint. If unreachable, return `float('inf')`.
            b.  **For `(communicated_rock_data ?w)`:** Similar logic as soil data, using rock-specific predicates and capabilities.
            c.  **For `(communicated_image_data ?o ?m)`:**
                -   `cost_g` starts at 1 (for the `communicate_image_data` action).
                -   Check if any rover currently has `(have_image ?r ?o ?m)`.
                -   If not:
                    -   Check if any rover is equipped for imaging and has a camera supporting mode `m` that can image objective `o` (i.e., `o` is visible from some waypoint reachable by the rover). If not, this goal is impossible, return `float('inf')`.
                    -   Add 1 to `cost_g` (for the `take_image` action).
                    -   Add 1 to `cost_g` (for the `calibrate` action - simplified, assumes one calibrate per image goal).
                    -   Find all waypoints visible from objective `o`. Add the minimum travel cost from any rover's current location to any of these waypoints. If no such waypoint exists or is unreachable, return `float('inf')`.
                -   Add the minimum travel cost from any rover's current location to any lander communication waypoint. If unreachable, return `float('inf')`.
        8.  Add `cost_g` to the total heuristic value `h`.
        9.  After iterating through all goals, return `h`.
    """

    def __init__(self, task):
        self.task = task
        self._parse_static_facts(task.static)
        self._compute_rover_distances()
        self._identify_lander_comm_points()
        self._store_initial_samples(task.initial_state)

    def _parse_fact(self, fact_str):
        """Parses a PDDL fact string into a list of strings."""
        # Remove surrounding brackets and split by space
        parts = fact_str[1:-1].split()
        return parts

    def _parse_static_facts(self, static_facts):
        self.rover_capabilities = {} # {rover: {'soil': bool, 'rock': bool, 'imaging': bool}}
        self.store_owner = {} # {store: rover}
        self.camera_info = {} # {camera: {'rover': rover, 'target': objective, 'modes': {mode}}}
        self.lander_location = {} # {lander: waypoint}
        self.waypoint_visibility_graph = collections.defaultdict(set) # {w1: {w2, ...}}
        self.objective_visibility_waypoints = collections.defaultdict(set) # {objective: {waypoint, ...}}
        self.rover_traversability_graph = collections.defaultdict(lambda: collections.defaultdict(set)) # {rover: {w1: {w2, ...}}}
        self.all_rovers = set()
        self.all_waypoints = set()
        self.all_objectives = set()
        self.all_modes = set()
        self.all_cameras = set()
        self.all_stores = set()
        self.all_landers = set()

        for fact_str in static_facts:
            parts = self._parse_fact(fact_str)
            pred = parts[0]

            if pred == 'at_lander':
                lander, wp = parts[1:]
                self.lander_location[lander] = wp
                self.all_landers.add(lander)
                self.all_waypoints.add(wp)
            elif pred == 'equipped_for_soil_analysis':
                rover = parts[1]
                if rover not in self.rover_capabilities: self.rover_capabilities[rover] = {'soil': False, 'rock': False, 'imaging': False}
                self.rover_capabilities[rover]['soil'] = True
                self.all_rovers.add(rover)
            elif pred == 'equipped_for_rock_analysis':
                rover = parts[1]
                if rover not in self.rover_capabilities: self.rover_capabilities[rover] = {'soil': False, 'rock': False, 'imaging': False}
                self.rover_capabilities[rover]['rock'] = True
                self.all_rovers.add(rover)
            elif pred == 'equipped_for_imaging':
                rover = parts[1]
                if rover not in self.rover_capabilities: self.rover_capabilities[rover] = {'soil': False, 'rock': False, 'imaging': False}
                self.rover_capabilities[rover]['imaging'] = True
                self.all_rovers.add(rover)
            elif pred == 'store_of':
                store, rover = parts[1:]
                self.store_owner[store] = rover
                self.all_stores.add(store)
                self.all_rovers.add(rover)
            elif pred == 'visible':
                w1, w2 = parts[1:]
                self.waypoint_visibility_graph[w1].add(w2)
                self.all_waypoints.add(w1)
                self.all_waypoints.add(w2)
            elif pred == 'can_traverse':
                rover, w1, w2 = parts[1:]
                self.rover_traversability_graph[rover][w1].add(w2)
                self.all_rovers.add(rover)
                self.all_waypoints.add(w1)
                self.all_waypoints.add(w2)
            elif pred == 'on_board':
                camera, rover = parts[1:]
                if camera not in self.camera_info: self.camera_info[camera] = {'rover': None, 'target': None, 'modes': set()}
                self.camera_info[camera]['rover'] = rover
                self.all_cameras.add(camera)
                self.all_rovers.add(rover)
            elif pred == 'calibration_target':
                camera, objective = parts[1:]
                if camera not in self.camera_info: self.camera_info[camera] = {'rover': None, 'target': None, 'modes': set()}
                self.camera_info[camera]['target'] = objective
                self.all_cameras.add(camera)
                self.all_objectives.add(objective)
            elif pred == 'supports':
                camera, mode = parts[1:]
                if camera not in self.camera_info: self.camera_info[camera] = {'rover': None, 'target': None, 'modes': set()}
                self.camera_info[camera]['modes'].add(mode)
                self.all_cameras.add(camera)
                self.all_modes.add(mode)
            elif pred == 'visible_from':
                objective, wp = parts[1:]
                self.objective_visibility_waypoints[objective].add(wp)
                self.all_objectives.add(objective)
                self.all_waypoints.add(wp)

        # Ensure all rovers/waypoints mentioned in traversability/capabilities/visibility are in sets
        for r, graph in self.rover_traversability_graph.items():
             self.all_rovers.add(r)
             for w1, neighbors in graph.items():
                 self.all_waypoints.add(w1)
                 for w2 in neighbors:
                     self.all_waypoints.add(w2)
        for r in self.rover_capabilities: self.all_rovers.add(r)
        for w1, neighbors in self.waypoint_visibility_graph.items():
             self.all_waypoints.add(w1)
             for w2 in neighbors: self.all_waypoints.add(w2)


    def _compute_rover_distances(self):
        self.rover_distances = {}
        for rover in self.all_rovers:
            self.rover_distances[rover] = {}
            graph = self.rover_traversability_graph.get(rover, {})

            for start_node in self.all_waypoints:
                self.rover_distances[rover][start_node] = {}
                # Perform BFS from start_node
                queue = collections.deque([(start_node, 0)])
                visited = {start_node: 0}

                while queue:
                    current_node, dist = queue.popleft()
                    self.rover_distances[rover][start_node][current_node] = dist

                    if current_node in graph:
                        for neighbor in graph[current_node]:
                            if neighbor not in visited:
                                visited[neighbor] = dist + 1
                                queue.append((neighbor, dist + 1))

                # Mark unreachable waypoints with infinity
                for end_node in self.all_waypoints:
                    if end_node not in self.rover_distances[rover][start_node]:
                         self.rover_distances[rover][start_node][end_node] = float('inf')


    def _identify_lander_comm_points(self):
        self.lander_comm_points = collections.defaultdict(set)
        # Communication requires (at ?r ?x), (at_lander ?l ?y), (visible ?x ?y)
        # This means waypoint ?y (lander location) must be visible *from* waypoint ?x (rover location).
        # We need waypoints ?x such that ?y is in self.waypoint_visibility_graph[?x].
        # This is equivalent to finding x such that x is in the reverse_visibility_graph of ?y.
        reverse_visibility_graph = collections.defaultdict(set)
        for w1, neighbors in self.waypoint_visibility_graph.items():
            for w2 in neighbors:
                reverse_visibility_graph[w2].add(w1)

        for lander, lander_wp in self.lander_location.items():
            self.lander_comm_points[lander] = reverse_visibility_graph.get(lander_wp, set())


    def _store_initial_samples(self, initial_state):
        self.initial_soil_samples = set()
        self.initial_rock_samples = set()
        for fact_str in initial_state:
            if fact_str.startswith('(at_soil_sample '):
                parts = self._parse_fact(fact_str)
                self.initial_soil_samples.add(parts[1])
            elif fact_str.startswith('(at_rock_sample '):
                parts = self._parse_fact(fact_str)
                self.initial_rock_samples.add(parts[1])

    def _get_rover_location(self, state, rover):
        """Finds the current waypoint of a given rover in the state."""
        for fact_str in state:
            if fact_str.startswith('(at '):
                parts = self._parse_fact(fact_str)
                if len(parts) == 3 and parts[1] == rover:
                    return parts[2]
        return None # Should not happen in valid states

    def _min_travel_cost_from_any_rover(self, state, target_waypoints):
        """
        Calculates the minimum travel cost from any rover's current location
        to any of the target waypoints.
        """
        if not self.all_rovers or not target_waypoints:
            return float('inf')

        min_dist = float('inf')
        current_rover_locations = {r: self._get_rover_location(state, r) for r in self.all_rovers}

        for rover in self.all_rovers:
            current_wp = current_rover_locations.get(rover)
            if current_wp is None: continue # Rover location unknown (shouldn't happen)

            if rover not in self.rover_distances or current_wp not in self.rover_distances[rover]:
                 # Rover or current waypoint not in precomputed distances
                 continue

            for target_wp in target_waypoints:
                if target_wp in self.rover_distances[rover][current_wp]:
                    dist = self.rover_distances[rover][current_wp][target_wp]
                    min_dist = min(min_dist, dist)
                # else: target_wp is unreachable from current_wp for this rover

        return min_dist

    def __call__(self, state):
        if self.task.goal_reached(state):
            return 0

        h = 0

        # Extract dynamic state information
        current_have_soil = set()
        current_have_rock = set()
        current_have_image = set()

        for fact_str in state:
            if fact_str.startswith('(have_soil_analysis '):
                parts = self._parse_fact(fact_str)
                if len(parts) == 3: current_have_soil.add(tuple(parts[1:])) # (rover, waypoint)
            elif fact_str.startswith('(have_rock_analysis '):
                parts = self._parse_fact(fact_str)
                if len(parts) == 3: current_have_rock.add(tuple(parts[1:])) # (rover, waypoint)
            elif fact_str.startswith('(have_image '):
                parts = self._parse_fact(fact_str)
                if len(parts) == 4: current_have_image.add(tuple(parts[1:])) # (rover, objective, mode)

        # Get all lander communication waypoints
        all_comm_wps = set().union(*self.lander_comm_points.values())
        # If no lander or no comm points, any communication goal is impossible
        if not all_comm_wps:
             # Check if any communication goal exists
             needs_comm_goal = False
             for g in self.task.goals:
                 if g not in state and (g.startswith('(communicated_soil_data ') or g.startswith('(communicated_rock_data ') or g.startswith('(communicated_image_data ')):
                     needs_comm_goal = True
                     break
             if needs_comm_goal:
                 return float('inf')


        for goal_fact_str in self.task.goals:
            if goal_fact_str in state:
                continue # Goal already achieved

            parts = self._parse_fact(goal_fact_str)
            pred = parts[0]

            if pred == 'communicated_soil_data':
                if len(parts) != 2: continue # Malformed goal
                w = parts[1]
                cost_g = 1 # communicate action

                # Check if sample needs to be obtained
                sample_needed = True
                if (any(r_w[1] == w for r_w in current_have_soil)):
                     sample_needed = False

                if sample_needed:
                    # Check if sampling is possible at all
                    if not any(self.rover_capabilities.get(r, {}).get('soil', False) for r in self.all_rovers):
                         return float('inf') # No rover equipped for soil
                    if w not in self.initial_soil_samples:
                         # Sample didn't exist initially and no one has it now. Impossible.
                         return float('inf')

                    cost_g += 1 # sample action
                    # Travel to sample location
                    travel_to_sample_cost = self._min_travel_cost_from_any_rover(state, {w})
                    if travel_to_sample_cost == float('inf'): return float('inf') # Unreachable sample waypoint
                    cost_g += travel_to_sample_cost
                    # (Ignore store/drop cost in this simplified heuristic)

                # Travel to communication location
                travel_to_comm_cost = self._min_travel_cost_from_any_rover(state, all_comm_wps)
                if travel_to_comm_cost == float('inf'): return float('inf') # Unreachable comm point
                cost_g += travel_to_comm_cost

                h += cost_g

            elif pred == 'communicated_rock_data':
                if len(parts) != 2: continue # Malformed goal
                w = parts[1]
                cost_g = 1 # communicate action

                # Check if sample needs to be obtained
                sample_needed = True
                if (any(r_w[1] == w for r_w in current_have_rock)):
                     sample_needed = False

                if sample_needed:
                    # Check if sampling is possible at all
                    if not any(self.rover_capabilities.get(r, {}).get('rock', False) for r in self.all_rovers):
                         return float('inf') # No rover equipped for rock
                    if w not in self.initial_rock_samples:
                         # Sample didn't exist initially and no one has it now. Impossible.
                         return float('inf')

                    cost_g += 1 # sample action
                    # Travel to sample location
                    travel_to_sample_cost = self._min_travel_cost_from_any_rover(state, {w})
                    if travel_to_sample_cost == float('inf'): return float('inf') # Unreachable sample waypoint
                    cost_g += travel_to_sample_cost
                    # (Ignore store/drop cost)

                # Travel to communication location
                travel_to_comm_cost = self._min_travel_cost_from_any_rover(state, all_comm_wps)
                if travel_to_comm_cost == float('inf'): return float('inf') # Unreachable comm point
                cost_g += travel_to_comm_cost

                h += cost_g

            elif pred == 'communicated_image_data':
                if len(parts) != 3: continue # Malformed goal
                o, m = parts[1:]
                cost_g = 1 # communicate action

                # Check if image needs to be obtained
                image_needed = True
                if (any(r_o_m[1] == o and r_o_m[2] == m for r_o_m in current_have_image)):
                     image_needed = False

                if image_needed:
                    # Check if imaging is possible at all for this objective/mode
                    can_image_goal = False
                    suitable_imaging_wps = self.objective_visibility_waypoints.get(o, set())

                    if suitable_imaging_wps: # Objective must be visible from somewhere
                        for r in self.all_rovers:
                            if self.rover_capabilities.get(r, {}).get('imaging', False):
                                for cam in self.camera_info:
                                    cam_info = self.camera_info[cam]
                                    if cam_info['rover'] == r and m in cam_info['modes']:
                                        # Found a rover with a camera supporting the mode
                                        # We also need a calibration target for this camera
                                        if cam_info['target'] is not None:
                                            # Assume calibration target is visible from imaging waypoint for simplicity
                                            can_image_goal = True
                                            break # Found a suitable camera on this rover
                                if can_image_goal: break # Found a suitable rover
                    if not can_image_goal:
                         return float('inf') # No rover can take this image

                    cost_g += 1 # take_image action
                    cost_g += 1 # calibrate action (simplified, assumes one calibrate per image goal)

                    # Travel to imaging location (any waypoint visible from o)
                    if not suitable_imaging_wps:
                         # This case should be caught by can_image_goal check, but double check
                         return float('inf') # Cannot image objective from anywhere

                    travel_to_image_cost = self._min_travel_cost_from_any_rover(state, suitable_imaging_wps)
                    if travel_to_image_cost == float('inf'): return float('inf') # Unreachable imaging waypoint
                    cost_g += travel_to_image_cost

                # Travel to communication location
                travel_to_comm_cost = self._min_travel_cost_from_any_rover(state, all_comm_wps)
                if travel_to_comm_cost == float('inf'): return float('inf') # Unreachable comm point
                cost_g += travel_to_comm_cost

                h += cost_g

            # Add other goal types if any (none in this domain)

        return h
