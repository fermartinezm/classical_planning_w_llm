# Note: This code assumes the existence of a base class `Heuristic`
# and a `Task` class with attributes `goals`, `static`, `initial_state`,
# and a `node` object with a `state` attribute (a frozenset of facts).
# The provided code defines the roversHeuristic class structure compatible
# with such a framework.

from fnmatch import fnmatch
from collections import deque

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    if not fact or not isinstance(fact, str) or len(fact) < 2 or fact[0] != '(' or fact[-1] != ')':
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


class roversHeuristic: # Inherits from Heuristic in the planner environment
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve all
    goal conditions. It calculates the cost for each unachieved goal
    independently and sums them up. The cost for a communication goal
    is estimated as the cost to first obtain the required data (sample or image)
    plus the cost to navigate to a lander-visible waypoint and communicate.
    Navigation costs are estimated using precomputed shortest paths (BFS).

    # Assumptions:
    - Each action has a cost of 1.
    - The cost of achieving different goals is independent (e.g., navigating
      to a waypoint for sampling is counted separately from navigating to
      a lander-visible waypoint for communication, even if the waypoints are the same).
    - Obtaining data (sampling or imaging) is only possible if the necessary
      conditions (sample exists, equipped rover, visible objective/calibration target)
      are met based on the current dynamic state and static facts.
    - A soil/rock sample, once sampled, is gone from that location. The heuristic
      checks the current state for sample availability.
    - Calibration is needed before taking an image unless the camera is already calibrated.
    - If a goal is unreachable (e.g., no equipped rover, no sample at location,
      no path to a required waypoint), the heuristic returns a large value.

    # Heuristic Initialization
    - Extracts static facts such as rover capabilities, store ownership,
      waypoint connectivity (`can_traverse`), waypoint visibility (`visible`),
      objective visibility (`visible_from`), camera information (on-board rover,
      calibration target, supported modes), and lander location.
    - Identifies all objects of relevant types (rovers, waypoints, cameras, etc.) from
      initial state, static facts, and goals.
    - Precomputes shortest path distances between all pairs of waypoints for each
      rover using BFS based on their `can_traverse` capabilities.
    - Identifies waypoints visible from the lander's location.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic calculates the sum of estimated costs for
    each goal predicate that is not yet satisfied in the state:

    1.  **Identify Unachieved Goals:** Iterate through the task's goal conditions.
        If a goal predicate is present in the current state, its cost is 0.
        Otherwise, estimate its cost and add it to the total heuristic value.

    2.  **Estimate Cost for `(communicated_soil_data ?w)`:**
        -   If `(have_soil_analysis ?r ?w)` is true for any rover `r` in the state:
            -   Cost to get data = 0.
        -   Else (need to sample soil at `?w`):
            -   If `(at_soil_sample ?w)` is not in the current state, this goal is unreachable (return large value).
            -   Find all rovers `r` equipped for soil analysis. If none, unreachable (return large value).
            -   Calculate the minimum cost among these rovers to:
                -   Navigate from the rover's current location to `?w`.
                -   Add 1 if the rover's store is full (for the `drop` action).
                -   Add 1 for the `sample_soil` action.
            -   Cost to get data = this minimum cost. If no such rover can reach `?w`, unreachable (return large value).
        -   Calculate the minimum cost among all rovers to:
            -   Navigate from the rover's current location to any waypoint `x` visible from the lander's location.
            -   Add 1 for the `communicate_soil_data` action.
        -   Cost to communicate = this minimum cost. If no rover can reach a lander-visible waypoint, unreachable (return large value).
        -   Total cost for this goal = Cost to get data + Cost to communicate.

    3.  **Estimate Cost for `(communicated_rock_data ?w)`:**
        -   Similar logic as for soil data, using rock-specific predicates (`equipped_for_rock_analysis`, `at_rock_sample`, `have_rock_analysis`, `communicate_rock_data`).

    4.  **Estimate Cost for `(communicated_image_data ?o ?m)`:**
        -   If `(have_image ?r ?o ?m)` is true for any rover `r` in the state:
            -   Cost to get data = 0.
        -   Else (need to take image of `?o` in mode `?m`):
            -   Find all rovers `r` equipped for imaging. If none, unreachable (return large value).
            -   Find cameras `c` on these rovers that support mode `?m`. If none, unreachable (return large value).
            -   Calculate the minimum cost among these rover-camera pairs to:
                -   Cost to calibrate (if camera `c` is not calibrated):
                    -   Find waypoints `w` visible from `c`'s calibration target. If none, unreachable (return large value).
                    -   Minimum cost to navigate from rover's current location to any such `w`.
                    -   Add 1 for the `calibrate` action.
                -   Cost to take image:
                    -   Find waypoints `p` visible from objective `?o`. If none, unreachable (return large value).
                    -   Minimum cost to navigate from rover's current location to any such `p`.
                    -   Add the calibration cost (calculated above).
                    -   Add 1 for the `take_image` action.
            -   Cost to get data = this minimum cost. If no suitable rover/camera/waypoint combination exists, unreachable (return large value).
        -   Calculate the minimum cost among all rovers to:
            -   Navigate from the rover's current location to any waypoint `x` visible from the lander's location.
            -   Add 1 for the `communicate_image_data` action.
        -   Cost to communicate = this minimum cost. If no rover can reach a lander-visible waypoint, unreachable (return large value).
        -   Total cost for this goal = Cost to get data + Cost to communicate.

    5.  **Sum Costs:** The total heuristic value is the sum of the estimated costs for all unachieved goals. If any unachieved goal is determined to be unreachable, the total heuristic is a large value (representing infinity).
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static
        initial_state_facts = task.initial_state

        # Use a large integer to represent infinity for unreachable goals
        self.infinity = 10**9

        # Extract all objects of relevant types
        all_objects = set()
        for fact in initial_state_facts | static_facts | self.goals:
            parts = get_parts(fact)
            # Add all arguments as potential objects, skipping the predicate name
            all_objects.update(parts[1:])

        self.all_waypoints = {obj for obj in all_objects if obj.startswith('waypoint')}
        self.all_rovers = {obj for obj in all_objects if obj.startswith('rover')}
        self.all_cameras = {obj for obj in all_objects if obj.startswith('camera')}
        self.all_objectives = {obj for obj in all_objects if obj.startswith('objective')}
        self.all_landers = {obj for obj in all_objects if obj.startswith('lander')}
        self.all_stores = {obj for obj in all_objects if obj.startswith('store')}
        self.all_modes = set() # Will be inferred from supports predicates


        # Parse static facts and build lookup structures
        self.lander_location = {} # {lander: waypoint}
        self.rover_capabilities = {} # {rover: {soil, rock, imaging}}
        self.rover_stores = {} # {rover: store}
        self.rover_can_traverse_graph = {} # {rover: {from_wp: {to_wp}}}
        self.waypoint_visibility = {} # {from_wp: {to_wp}}
        self.objective_visibility = {} # {objective: {waypoint}}
        self.camera_info = {} # {camera: {'rover': r, 'calibration_target': t, 'modes': {m}}}

        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]
            if predicate == "at_lander":
                lander, wp = parts[1], parts[2]
                self.lander_location[lander] = wp
            elif predicate == "equipped_for_soil_analysis":
                rover = parts[1]
                self.rover_capabilities.setdefault(rover, set()).add('soil')
            elif predicate == "equipped_for_rock_analysis":
                rover = parts[1]
                self.rover_capabilities.setdefault(rover, set()).add('rock')
            elif predicate == "equipped_for_imaging":
                rover = parts[1]
                self.rover_capabilities.setdefault(rover, set()).add('imaging')
            elif predicate == "store_of":
                store, rover = parts[1], parts[2]
                self.rover_stores[rover] = store
            elif predicate == "can_traverse":
                rover, wp1, wp2 = parts[1], parts[2], parts[3]
                self.rover_can_traverse_graph.setdefault(rover, {}).setdefault(wp1, set()).add(wp2)
            elif predicate == "visible":
                wp1, wp2 = parts[1], parts[2]
                self.waypoint_visibility.setdefault(wp1, set()).add(wp2)
                self.waypoint_visibility.setdefault(wp2, set()).add(wp1) # Visible is bidirectional
            elif predicate == "visible_from":
                objective, wp = parts[1], parts[2]
                self.objective_visibility.setdefault(objective, set()).add(wp)
            elif predicate == "calibration_target":
                camera, objective = parts[1], parts[2]
                self.camera_info.setdefault(camera, {})['calibration_target'] = objective
            elif predicate == "on_board":
                camera, rover = parts[1], parts[2]
                self.camera_info.setdefault(camera, {})['rover'] = rover
            elif predicate == "supports":
                camera, mode = parts[1], parts[2]
                self.camera_info.setdefault(camera, {}).setdefault('modes', set()).add(mode)
                self.all_modes.add(mode) # Infer modes

        # Precompute distances for each rover
        self.rover_distances = {} # {rover: {from_wp: {to_wp: dist}}}
        for rover in self.all_rovers:
            # BFS requires a graph where neighbors are reachable *by this rover*
            self.rover_distances[rover] = self._bfs(rover, self.all_waypoints, self.rover_can_traverse_graph)

        # Precompute calibration target visibility lookup
        self.calibration_target_visibility = {} # {objective: {waypoint}}
        for camera, info in self.camera_info.items():
            cal_target = info.get('calibration_target')
            if cal_target and cal_target in self.objective_visibility:
                self.calibration_target_visibility[cal_target] = self.objective_visibility[cal_target]
            else:
                 self.calibration_target_visibility[cal_target] = set() # No visible waypoints for calibration

        # Precompute lander visible waypoints (assuming one lander)
        lander_wp = list(self.lander_location.values())[0] if self.lander_location else None
        self.lander_visible_waypoints = self.waypoint_visibility.get(lander_wp, set()) if lander_wp else set()


    def _bfs(self, rover, waypoints, can_traverse_graph):
        """
        Computes shortest path distances from all waypoints to all other waypoints
        traversable by a specific rover.

        Args:
            rover (str): The name of the rover.
            waypoints (set): A set of all waypoint names.
            can_traverse_graph (dict): A graph representation {rover: {from_wp: {to_wp}}}.

        Returns:
            dict: A dictionary {start_wp: {end_wp: distance}}. Returns empty dict
                  if the rover has no traversal capabilities.
        """
        distances = {wp: {} for wp in waypoints}
        # Ensure the rover exists in the graph structure
        if rover not in can_traverse_graph:
            return distances # No traversable paths for this rover

        rover_graph = can_traverse_graph[rover]

        for start_node in waypoints:
            q = deque([(start_node, 0)])
            visited = {start_node}
            distances[start_node][start_node] = 0

            while q:
                current_node, dist = q.popleft()

                # Neighbors are waypoints reachable from current_node by the rover
                neighbors = rover_graph.get(current_node, set())

                for neighbor in neighbors:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        distances[start_node][neighbor] = dist + 1
                        q.append((neighbor, dist + 1))
        return distances


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state_facts = set(node.state)  # Current world state facts

        # Parse dynamic state information
        rover_locations = {} # {rover: waypoint}
        store_status = {} # {store: 'empty' or 'full'}
        soil_samples_at = set() # {waypoint}
        rock_samples_at = set() # {waypoint}
        rover_soil_data = {} # {rover: {waypoint}}
        rover_rock_data = {} # {rover: {waypoint}}
        rover_image_data = {} # {rover: {(objective, mode)}}
        camera_calibration = {} # {camera: bool}

        for fact in state_facts:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]
            if predicate == "at" and parts[1] in self.all_rovers: # Ensure it's a rover
                rover, wp = parts[1], parts[2]
                rover_locations[rover] = wp
            elif predicate == "empty" and parts[1] in self.all_stores:
                store = parts[1]
                store_status[store] = 'empty'
            elif predicate == "full" and parts[1] in self.all_stores:
                store = parts[1]
                store_status[store] = 'full'
            elif predicate == "at_soil_sample":
                wp = parts[1]
                soil_samples_at.add(wp)
            elif predicate == "at_rock_sample":
                wp = parts[1]
                rock_samples_at.add(wp)
            elif predicate == "have_soil_analysis" and parts[1] in self.all_rovers:
                rover, wp = parts[1], parts[2]
                rover_soil_data.setdefault(rover, set()).add(wp)
            elif predicate == "have_rock_analysis" and parts[1] in self.all_rovers:
                rover, wp = parts[1], parts[2]
                rover_rock_data.setdefault(rover, set()).add(wp)
            elif predicate == "have_image" and parts[1] in self.all_rovers:
                rover, obj, mode = parts[1], parts[2], parts[3]
                rover_image_data.setdefault(rover, set()).add((obj, mode))
            elif predicate == "calibrated" and parts[1] in self.all_cameras and parts[2] in self.all_rovers:
                camera, rover = parts[1], parts[2]
                camera_calibration[camera] = True

        # Ensure all rovers have a location recorded (should be true in valid states)
        # If a rover is missing, any goal requiring it is unreachable.
        # We check for None when getting rover_locations later.

        # Ensure all cameras/stores are in dynamic maps even if not calibrated/full/empty
        for cam in self.all_cameras:
            camera_calibration.setdefault(cam, False)
        for store in self.all_stores:
             store_status.setdefault(store, 'empty') # Assume empty if not explicitly full


        total_h = 0

        for goal in self.goals:
            if goal in state_facts:
                continue # Goal already achieved

            parts = get_parts(goal)
            if not parts: continue # Skip malformed goals

            predicate = parts[0]

            if predicate == "communicated_soil_data":
                wp = parts[1]
                goal_data_had = any(wp in data for data in rover_soil_data.values())

                cost_get_data = 0
                if not goal_data_had:
                    # Need to sample soil at wp
                    if wp not in soil_samples_at:
                         # Cannot sample if sample is not currently there
                         return self.infinity # Unsolvable goal

                    min_sample_cost = self.infinity
                    for rover in self.all_rovers:
                        if 'soil' in self.rover_capabilities.get(rover, set()):
                            current_r_loc = rover_locations.get(rover)
                            if current_r_loc is None: continue # Rover location unknown

                            dist_to_sample = self.rover_distances.get(rover, {}).get(current_r_loc, {}).get(wp, self.infinity)
                            if dist_to_sample == self.infinity: continue # Rover cannot reach sample location

                            store = self.rover_stores.get(rover)
                            if store is None: continue # Rover has no store

                            store_full_cost = 1 if store_status.get(store) == 'full' else 0

                            sample_cost = dist_to_sample + store_full_cost + 1 # Navigate + (Drop) + Sample
                            min_sample_cost = min(min_sample_cost, sample_cost)

                    if min_sample_cost == self.infinity:
                         return self.infinity # Cannot sample soil
                    cost_get_data = min_sample_cost

                # Need to communicate data from a lander-visible waypoint
                cost_communicate = self.infinity
                for rover in self.all_rovers:
                    current_r_loc = rover_locations.get(rover)
                    if current_r_loc is None: continue

                    min_dist_to_lander_visible = self.infinity
                    for lander_wp in self.lander_visible_waypoints:
                         dist = self.rover_distances.get(rover, {}).get(current_r_loc, {}).get(lander_wp, self.infinity)
                         min_dist_to_lander_visible = min(min_dist_to_lander_visible, dist)

                    if min_dist_to_lander_visible != self.infinity:
                         communicate_cost = min_dist_to_lander_visible + 1 # Navigate + Communicate
                         cost_communicate = min(cost_communicate, communicate_cost)

                if cost_communicate == self.infinity:
                     return self.infinity # Cannot reach lander visible waypoint

                total_h += cost_get_data + cost_communicate

            elif predicate == "communicated_rock_data":
                wp = parts[1]
                goal_data_had = any(wp in data for data in rover_rock_data.values())

                cost_get_data = 0
                if not goal_data_had:
                    # Need to sample rock at wp
                    if wp not in rock_samples_at:
                         return self.infinity # Unsolvable goal

                    min_sample_cost = self.infinity
                    for rover in self.all_rovers:
                        if 'rock' in self.rover_capabilities.get(rover, set()):
                            current_r_loc = rover_locations.get(rover)
                            if current_r_loc is None: continue

                            dist_to_sample = self.rover_distances.get(rover, {}).get(current_r_loc, {}).get(wp, self.infinity)
                            if dist_to_sample == self.infinity: continue

                            store = self.rover_stores.get(rover)
                            if store is None: continue

                            store_full_cost = 1 if store_status.get(store) == 'full' else 0

                            sample_cost = dist_to_sample + store_full_cost + 1 # Navigate + (Drop) + Sample
                            min_sample_cost = min(min_sample_cost, sample_cost)

                    if min_sample_cost == self.infinity:
                         return self.infinity # Cannot sample rock
                    cost_get_data = min_sample_cost

                # Need to communicate data from a lander-visible waypoint
                cost_communicate = self.infinity
                for rover in self.all_rovers:
                    current_r_loc = rover_locations.get(rover)
                    if current_r_loc is None: continue

                    min_dist_to_lander_visible = self.infinity
                    for lander_wp in self.lander_visible_waypoints:
                         dist = self.rover_distances.get(rover, {}).get(current_r_loc, {}).get(lander_wp, self.infinity)
                         min_dist_to_lander_visible = min(min_dist_to_lander_visible, dist)

                    if min_dist_to_lander_visible != self.infinity:
                         communicate_cost = min_dist_to_lander_visible + 1 # Navigate + Communicate
                         cost_communicate = min(cost_communicate, communicate_cost)

                if cost_communicate == self.infinity:
                     return self.infinity # Cannot reach lander visible waypoint

                total_h += cost_get_data + cost_communicate

            elif predicate == "communicated_image_data":
                obj, mode = parts[1], parts[2]
                goal_data_had = any((obj, mode) in data for data in rover_image_data.values())

                cost_get_data = 0
                if not goal_data_had:
                    # Need to take image of obj in mode
                    min_image_cost = self.infinity
                    for rover in self.all_rovers:
                        if 'imaging' in self.rover_capabilities.get(rover, set()):
                            current_r_loc = rover_locations.get(rover)
                            if current_r_loc is None: continue

                            # Find cameras on this rover supporting this mode
                            possible_cameras = []
                            for cam, info in self.camera_info.items():
                                if info.get('rover') == rover and mode in info.get('modes', set()):
                                    possible_cameras.append(cam)

                            if not possible_cameras: continue # Rover has no suitable camera

                            rover_camera_min_image_cost = self.infinity # Min cost for this rover + any of its cameras
                            for camera in possible_cameras:
                                cal_target = self.camera_info.get(camera, {}).get('calibration_target')
                                if not cal_target: continue # Camera has no calibration target

                                # Cost to calibrate (if needed)
                                calibration_cost = 0
                                if not camera_calibration.get(camera, False):
                                    min_dist_to_cal_wp = self.infinity
                                    cal_wps = self.calibration_target_visibility.get(cal_target, set())
                                    if not cal_wps:
                                         # Cannot calibrate if no waypoint visible from calibration target
                                         continue # Try next camera

                                    for cal_wp in cal_wps:
                                        dist = self.rover_distances.get(rover, {}).get(current_r_loc, {}).get(cal_wp, self.infinity)
                                        min_dist_to_cal_wp = min(min_dist_to_cal_wp, dist)

                                    if min_dist_to_cal_wp == self.infinity:
                                         # Cannot reach any calibration waypoint for this camera
                                         continue # Try next camera

                                    calibration_cost = min_dist_to_cal_wp + 1 # Navigate + Calibrate

                                # Cost to take image
                                min_dist_to_objective_wp = self.infinity
                                obj_wps = self.objective_visibility.get(obj, set())
                                if not obj_wps:
                                     # Cannot view objective if no waypoint visible from it
                                     continue # Try next camera

                                for obj_wp in obj_wps:
                                    dist = self.rover_distances.get(rover, {}).get(current_r_loc, {}).get(obj_wp, self.infinity)
                                    min_dist_to_objective_wp = min(min_dist_to_objective_wp, dist)

                                if min_dist_to_objective_wp == self.infinity:
                                     # Cannot reach any objective waypoint for this objective
                                     continue # Try next camera

                                take_image_cost = min_dist_to_objective_wp + calibration_cost + 1 # Navigate + (Calibrate) + Take_image
                                rover_camera_min_image_cost = min(rover_camera_min_image_cost, take_image_cost)

                            min_image_cost = min(min_image_cost, rover_camera_min_image_cost) # Min over all rovers and their cameras

                    if min_image_cost == self.infinity:
                         return self.infinity # Cannot take image
                    cost_get_data = min_image_cost

                # Need to communicate data from a lander-visible waypoint
                cost_communicate = self.infinity
                for rover in self.all_rovers:
                    current_r_loc = rover_locations.get(rover)
                    if current_r_loc is None: continue

                    min_dist_to_lander_visible = self.infinity
                    for lander_wp in self.lander_visible_waypoints:
                         dist = self.rover_distances.get(rover, {}).get(current_r_loc, {}).get(lander_wp, self.infinity)
                         min_dist_to_lander_visible = min(min_dist_to_lander_visible, dist)

                    if min_dist_to_lander_visible != self.infinity:
                         communicate_cost = min_dist_to_lander_visible + 1 # Navigate + Communicate
                         cost_communicate = min(cost_communicate, communicate_cost)

                if cost_communicate == self.infinity:
                     return self.infinity # Cannot reach lander visible waypoint

                total_h += cost_get_data + cost_communicate

            # Add other goal types if any (only communication goals in this domain)
            # else:
            #     # Unknown goal type - potentially unsolvable or unsupported
            #     return self.infinity


        return total_h
