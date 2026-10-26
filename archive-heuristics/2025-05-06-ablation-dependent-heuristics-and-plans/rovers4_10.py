from heuristics.heuristic_base import Heuristic
from fnmatch import fnmatch
from collections import deque
import math

# Helper functions
def get_parts(fact_string):
    """Parses a PDDL fact string into a list of parts."""
    # Remove parentheses and split by space
    # Handle potential empty strings or malformed facts defensively
    if not fact_string or not fact_string.startswith('(') or not fact_string.endswith(')'):
        return []
    return fact_string[1:-1].split()

def match(fact_string, *pattern_parts):
    """Checks if a PDDL fact matches a given pattern with wildcards."""
    parts = get_parts(fact_string)
    if len(parts) != len(pattern_parts):
        return False
    return all(fnmatch(part, pattern) for part, pattern in zip(parts, pattern_parts))

class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve all
    unsatisfied goal conditions. It sums the estimated costs for each goal
    independently, ignoring potential synergies (e.g., collecting multiple
    samples on one trip, communicating multiple data types at once). The
    cost for each goal is estimated based on the minimum actions required
    to collect the necessary data (sample or image) and then communicate it.
    Navigation costs are estimated using precomputed shortest paths between
    waypoints for each rover.

    # Assumptions
    - Each unsatisfied goal is treated independently.
    - Resource constraints (like store capacity, camera availability for multiple images)
      are simplified or ignored, except for the need for an empty store for sampling
      and camera calibration for imaging.
    - For sampling goals, an empty store is assumed available or requires one 'drop' action
      if the rover's store is currently full.
    - For imaging goals, calibration is assumed possible if a calibration target
      is visible from a reachable waypoint. The cost of calibration includes
      navigation to/from a calibration waypoint and the calibrate action.
    - If a required sample is no longer at its location and no rover has the data,
      the goal is considered unreachable (infinite cost).
    - Navigation between waypoints is possible only if explicitly allowed by
      'can_traverse' for the specific rover. Visibility between waypoints is
      symmetric for navigation and communication.

    # Heuristic Initialization
    - Parses static facts from the task definition.
    - Identifies lander location, rover capabilities (equipment, cameras),
      camera properties (modes, calibration targets), objective visibility,
      and navigation capabilities ('can_traverse').
    - Builds navigation graphs for each rover based on 'can_traverse'.
    - Precomputes all-pairs shortest path distances for each rover using BFS.
    - Identifies waypoints visible from the lander ('communication waypoints').
    - Stores the set of goal facts and static facts (for store mapping).

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic is computed as follows:

    1.  Initialize total heuristic cost to 0.
    2.  Extract current state information: rover locations, store statuses,
        collected soil/rock data, collected images, camera calibration statuses,
        and remaining soil/rock samples at waypoints.
    3.  Iterate through each goal fact defined in the task.
    4.  If a goal fact is already true in the current state, its cost is 0.
    5.  If a goal fact is not true, estimate the minimum cost to achieve it:
        -   **For `(communicated_soil_data ?w)`:**
            -   Check if any rover currently has `(have_soil_analysis ?r ?w)`.
                -   If yes: Cost is 1 (communicate) + minimum navigation cost for any such rover from its current location to any communication waypoint.
            -   If no rover has the data:
                -   Check if `(at_soil_sample ?w)` is still true in the state.
                    -   If yes: Find a rover equipped for soil analysis. Cost is 1 (sample) + 1 (communicate) + minimum navigation cost for an equipped rover from its current location to `?w` + minimum navigation cost from `?w` to any communication waypoint. Add 1 if the chosen rover's store is full (for a 'drop' action).
                    -   If no: The goal is unreachable from this state (sample is gone, data not collected). Cost is infinity.
        -   **For `(communicated_rock_data ?w)`:** Similar logic to soil data, using rock-specific predicates and equipment.
        -   **For `(communicated_image_data ?o ?m)`:**
            -   Check if any rover currently has `(have_image ?r ?o ?m)`.
                -   If yes: Cost is 1 (communicate) + minimum navigation cost for any such rover from its current location to any communication waypoint.
            -   If no rover has the image:
                -   Find a rover equipped for imaging, with a camera supporting mode `?m` on board.
                -   Find a waypoint `?w_img` where objective `?o` is visible.
                -   Find a communication waypoint `?w_comm`.
                -   Estimate the minimum cost over all suitable rovers, cameras, imaging waypoints, and communication waypoints:
                    -   Base cost = 1 (take_image) + 1 (communicate) + navigation cost from rover's current location to `?w_img` + navigation cost from `?w_img` to `?w_comm`.
                    -   Calibration cost: If the chosen camera is not calibrated for the rover: Find its calibration target `?t` and waypoints `?w_cal` where `?t` is visible. If such `?w_cal` exists and is reachable from `?w_img`, add the minimum cost of navigating from `?w_img` to `?w_cal`, calibrating (1 action), and navigating back to `?w_img`. If no suitable `?w_cal` is reachable, this option is impossible.
                    -   Total cost for this option = Base cost + Calibration cost.
                -   The goal cost is the minimum total cost over all valid options. If no valid option exists, the cost is infinity.
    6.  Sum the estimated costs for all unsatisfied goals to get the total heuristic value.
    7.  Return the total cost. If the total cost is infinity, return infinity.
    """

    def __init__(self, task):
        super().__init__(task)

        self.goals = task.goals
        self.static = task.static # Store static facts for easy access in __call__ if needed (e.g. store_of)

        # Parse static facts and build initial data structures
        self.lander_location = None
        self.rover_equipment = {} # {rover: {soil: bool, rock: bool, imaging: bool}}
        self.rover_cameras = {} # {rover: set(camera)}
        self.camera_modes = {} # {camera: set(mode)}
        self.camera_calibration_target = {} # {camera: objective}
        self.objective_visible_from = {} # {objective: set(waypoint)}
        self.can_traverse_graph = {} # {rover: {waypoint: set(waypoint)}}
        self.visible_graph = {} # {waypoint: set(waypoint)} # Used for communication waypoints
        self.rover_stores_map = {} # {rover: store} # Static mapping

        # Collect all relevant objects and waypoints from static facts
        all_waypoints = set()
        all_rovers = set()
        all_cameras = set()
        all_objectives = set()
        all_modes = set()
        all_stores = set()
        all_landers = set()

        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]
            if predicate == 'at_lander':
                all_landers.add(parts[1])
                all_waypoints.add(parts[2])
                self.lander_location = parts[2]
            elif predicate == 'can_traverse':
                rover, w1, w2 = parts[1:]
                all_rovers.add(rover)
                all_waypoints.add(w1)
                all_waypoints.add(w2)
                if rover not in self.can_traverse_graph:
                    self.can_traverse_graph[rover] = {}
                if w1 not in self.can_traverse_graph[rover]:
                    self.can_traverse_graph[rover][w1] = set()
                self.can_traverse_graph[rover][w1].add(w2)
            elif predicate == 'visible':
                w1, w2 = parts[1:]
                all_waypoints.add(w1)
                all_waypoints.add(w2)
                if w1 not in self.visible_graph:
                    self.visible_graph[w1] = set()
                if w2 not in self.visible_graph:
                    self.visible_graph[w2] = set()
                self.visible_graph[w1].add(w2)
                self.visible_graph[w2].add(w1) # Visibility is symmetric
            elif predicate.startswith('equipped_for_'):
                rover = parts[1]
                all_rovers.add(rover)
                if rover not in self.rover_equipment: self.rover_equipment[rover] = {}
                if predicate == 'equipped_for_soil_analysis':
                    self.rover_equipment[rover]['soil'] = True
                elif predicate == 'equipped_for_rock_analysis':
                    self.rover_equipment[rover]['rock'] = True
                elif predicate == 'equipped_for_imaging':
                    self.rover_equipment[rover]['imaging'] = True
            elif predicate == 'store_of':
                store, rover = parts[1:]
                all_stores.add(store)
                all_rovers.add(rover)
                self.rover_stores_map[rover] = store # Map rover to its store
            elif predicate == 'on_board':
                camera, rover = parts[1:]
                all_cameras.add(camera)
                all_rovers.add(rover)
                if rover not in self.rover_cameras: self.rover_cameras[rover] = set()
                self.rover_cameras[rover].add(camera)
            elif predicate == 'supports':
                camera, mode = parts[1:]
                all_cameras.add(camera)
                all_modes.add(mode)
                if camera not in self.camera_modes: self.camera_modes[camera] = set()
                self.camera_modes[camera].add(mode)
            elif predicate == 'calibration_target':
                camera, objective = parts[1:]
                all_cameras.add(camera)
                all_objectives.add(objective)
                self.camera_calibration_target[camera] = objective
            elif predicate == 'visible_from':
                objective, waypoint = parts[1:]
                all_objectives.add(objective)
                all_waypoints.add(waypoint)
                if objective not in self.objective_visible_from: self.objective_visible_from[objective] = set()
                self.objective_visible_from[objective].add(waypoint)

        # Initialize equipment flags to False if not present
        for r in all_rovers:
             if r not in self.rover_equipment: self.rover_equipment[r] = {}
             self.rover_equipment[r].setdefault('soil', False)
             self.rover_equipment[r].setdefault('rock', False)
             self.rover_equipment[r].setdefault('imaging', False)

        # Compute communication waypoints (visible from lander)
        self.communication_waypoints = set()
        if self.lander_location and self.lander_location in self.visible_graph:
             self.communication_waypoints.update(self.visible_graph[self.lander_location])

        # Compute navigation distances for each rover
        self.navigation_distances = {}
        for rover in all_rovers:
            self.navigation_distances[rover] = {}
            graph = self.can_traverse_graph.get(rover, {})
            # Compute distances from every waypoint to every other waypoint
            for start_node in all_waypoints:
                 self.navigation_distances[rover][start_node] = self._bfs(graph, start_node, all_waypoints)


    def _bfs(self, graph, start_node, all_nodes):
        """Performs BFS to find distances from start_node in a graph."""
        distances = {node: math.inf for node in all_nodes}

        # Ensure start_node is a known node before starting BFS
        if start_node not in all_nodes:
             return distances # Start node not in the set of known nodes

        distances[start_node] = 0
        queue = deque([start_node])

        while queue:
            current_node = queue.popleft()

            # Get neighbors from the graph, handle nodes with no outgoing edges
            neighbors = graph.get(current_node, set())

            for neighbor in neighbors:
                # Ensure neighbor is a known node (should be if graph is built from all_nodes)
                if neighbor in all_nodes and distances[neighbor] == math.inf:
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)
        return distances

    def get_distance(self, rover, w1, w2):
        """Safely get precomputed distance, return inf if not found or unreachable."""
        # Check if rover exists and waypoints exist in its distance map
        if rover in self.navigation_distances and w1 in self.navigation_distances[rover] and w2 in self.navigation_distances[rover][w1]:
            return self.navigation_distances[rover][w1][w2]
        # Return infinity if rover or waypoints are unknown or unreachable
        return math.inf

    def __call__(self, node):
        state = node.state
        total_cost = 0

        # Extract current state information
        rover_locations = {}
        store_status = {} # {store: 'empty' or 'full'}
        have_soil = set() # set of (rover, waypoint) tuples
        have_rock = set() # set of (rover, waypoint) tuples
        have_image = set() # set of (rover, objective, mode) tuples
        calibrated_cameras = set() # set of (camera, rover) tuples
        at_soil_samples = set() # set of waypoint strings
        at_rock_samples = set() # set of waypoint strings

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]
            if predicate == 'at' and parts[1].startswith('rover'):
                rover_locations[parts[1]] = parts[2]
            elif predicate == 'empty':
                 store_status[parts[1]] = 'empty'
            elif predicate == 'full':
                 store_status[parts[1]] = 'full'
            elif predicate == 'have_soil_analysis':
                 have_soil.add((parts[1], parts[2]))
            elif predicate == 'have_rock_analysis':
                 have_rock.add((parts[1], parts[2]))
            elif predicate == 'have_image':
                 have_image.add((parts[1], parts[2], parts[3]))
            elif predicate == 'calibrated':
                 calibrated_cameras.add((parts[1], parts[2]))
            elif predicate == 'at_soil_sample':
                 at_soil_samples.add(parts[1])
            elif predicate == 'at_rock_sample':
                 at_rock_samples.add(parts[1])

        # Calculate cost for each unsatisfied goal
        for goal_fact in self.goals:
            if goal_fact in state:
                continue # Goal already achieved

            parts = get_parts(goal_fact)
            predicate = parts[0]

            if predicate == 'communicated_soil_data':
                waypoint_p = parts[1]
                goal_cost = math.inf

                # Case 1: Data already collected by some rover
                rovers_with_data = [r for r, w in have_soil if w == waypoint_p]
                if rovers_with_data:
                    min_comm_nav_cost = math.inf
                    for rover in rovers_with_data:
                        w_curr = rover_locations.get(rover)
                        if w_curr is None: continue # Cannot use this rover if location unknown
                        for w_comm in self.communication_waypoints:
                            nav_cost = self.get_distance(rover, w_curr, w_comm)
                            min_comm_nav_cost = min(min_comm_nav_cost, nav_cost)
                    if min_comm_nav_cost != math.inf:
                         goal_cost = min(goal_cost, 1 + min_comm_nav_cost) # 1 for communicate

                # Case 2: Data needs to be collected and communicated
                # Only possible if the sample is still there
                if goal_cost == math.inf and waypoint_p in at_soil_samples:
                    equipped_rovers = [r for r, eq in self.rover_equipment.items() if eq.get('soil', False)]
                    if equipped_rovers:
                        min_collect_comm_cost = math.inf
                        for rover in equipped_rovers:
                            w_curr = rover_locations.get(rover)
                            if w_curr is None: continue

                            nav_to_sample = self.get_distance(rover, w_curr, waypoint_p)
                            if nav_to_sample == math.inf: continue # Cannot reach sample

                            # Cost to sample: nav_to_sample + 1 (sample) + drop_cost
                            drop_cost = 0
                            rover_store = self.rover_stores_map.get(rover) # Use static map
                            if rover_store and store_status.get(rover_store) == 'full':
                                drop_cost = 1 # Need to drop before sampling

                            # Cost to communicate after sampling: nav_from_sample_to_comm + 1 (communicate)
                            min_nav_sample_to_comm = math.inf
                            for w_comm in self.communication_waypoints:
                                nav_cost = self.get_distance(rover, waypoint_p, w_comm)
                                min_nav_sample_to_comm = min(min_nav_sample_to_comm, nav_cost)

                            if min_nav_sample_to_comm != math.inf:
                                current_option_cost = nav_to_sample + 1 + drop_cost + min_nav_sample_to_comm + 1
                                min_collect_comm_cost = min(min_collect_comm_cost, current_option_cost)

                        goal_cost = min(goal_cost, min_collect_comm_cost)

                total_cost += goal_cost

            elif predicate == 'communicated_rock_data':
                waypoint_p = parts[1]
                goal_cost = math.inf

                # Case 1: Data already collected
                rovers_with_data = [r for r, w in have_rock if w == waypoint_p]
                if rovers_with_data:
                    min_comm_nav_cost = math.inf
                    for rover in rovers_with_data:
                        w_curr = rover_locations.get(rover)
                        if w_curr is None: continue
                        for w_comm in self.communication_waypoints:
                            nav_cost = self.get_distance(rover, w_curr, w_comm)
                            min_comm_nav_cost = min(min_comm_nav_cost, nav_cost)
                    if min_comm_nav_cost != math.inf:
                         goal_cost = min(goal_cost, 1 + min_comm_nav_cost)

                # Case 2: Data needs to be collected and communicated
                # Only possible if the sample is still there
                if goal_cost == math.inf and waypoint_p in at_rock_samples:
                    equipped_rovers = [r for r, eq in self.rover_equipment.items() if eq.get('rock', False)]
                    if equipped_rovers:
                        min_collect_comm_cost = math.inf
                        for rover in equipped_rovers:
                            w_curr = rover_locations.get(rover)
                            if w_curr is None: continue

                            nav_to_sample = self.get_distance(rover, w_curr, waypoint_p)
                            if nav_to_sample == math.inf: continue

                            drop_cost = 0
                            rover_store = self.rover_stores_map.get(rover) # Use static map
                            if rover_store and store_status.get(rover_store) == 'full':
                                drop_cost = 1

                            min_nav_sample_to_comm = math.inf
                            for w_comm in self.communication_waypoints:
                                nav_cost = self.get_distance(rover, waypoint_p, w_comm)
                                min_nav_sample_to_comm = min(min_nav_sample_to_comm, nav_cost)

                            if min_nav_sample_to_comm != math.inf:
                                current_option_cost = nav_to_sample + 1 + drop_cost + min_nav_sample_to_comm + 1
                                min_collect_comm_cost = min(min_collect_comm_cost, current_option_cost)

                        goal_cost = min(goal_cost, min_collect_comm_cost)

                total_cost += goal_cost

            elif predicate == 'communicated_image_data':
                objective_o = parts[1]
                mode_m = parts[2]
                goal_cost = math.inf

                # Case 1: Image already taken by some rover
                rovers_with_image = [r for r, o, m in have_image if o == objective_o and m == mode_m]
                if rovers_with_image:
                    min_comm_nav_cost = math.inf
                    for rover in rovers_with_image:
                        w_curr = rover_locations.get(rover)
                        if w_curr is None: continue
                        for w_comm in self.communication_waypoints:
                            nav_cost = self.get_distance(rover, w_curr, w_comm)
                            min_comm_nav_cost = min(min_comm_nav_cost, nav_cost)
                    if min_comm_nav_cost != math.inf:
                         goal_cost = min(goal_cost, 1 + min_comm_nav_cost) # 1 for communicate

                # Case 2: Image needs to be taken and communicated
                if goal_cost == math.inf:
                    # Find suitable rovers (equipped for imaging)
                    equipped_rovers = [r for r, eq in self.rover_equipment.items() if eq.get('imaging', False)]
                    if equipped_rovers:
                        min_take_comm_cost = math.inf
                        for rover in equipped_rovers:
                            w_curr = rover_locations.get(rover)
                            if w_curr is None: continue

                            # Find suitable cameras on this rover supporting the mode
                            suitable_cameras = [
                                cam for cam in self.rover_cameras.get(rover, set())
                                if mode_m in self.camera_modes.get(cam, set())
                            ]

                            if suitable_cameras:
                                # Find suitable imaging waypoints for the objective
                                suitable_img_waypoints = self.objective_visible_from.get(objective_o, set())

                                if suitable_img_waypoints:
                                    for camera in suitable_cameras:
                                        # Find calibration target for the camera
                                        cal_target = self.camera_calibration_target.get(camera)
                                        suitable_cal_waypoints = set()
                                        if cal_target:
                                             suitable_cal_waypoints = self.objective_visible_from.get(cal_target, set())

                                        # Check if calibration is needed for this camera/rover
                                        calibration_needed = (camera, rover) not in calibrated_cameras

                                        # Find best imaging waypoint and communication waypoint for this rover/camera
                                        min_nav_img_comm = math.inf
                                        best_w_img = None

                                        for w_img in suitable_img_waypoints:
                                            nav_to_img = self.get_distance(rover, w_curr, w_img)
                                            if nav_to_img == math.inf: continue

                                            min_nav_img_to_comm = math.inf
                                            for w_comm in self.communication_waypoints:
                                                nav_img_to_comm = self.get_distance(rover, w_img, w_comm)
                                                min_nav_img_to_comm = min(min_nav_img_to_comm, nav_img_to_comm)

                                            if min_nav_img_to_comm != math.inf:
                                                 # Cost = Nav to img + Nav img to comm
                                                 current_nav_cost = nav_to_img + min_nav_img_to_comm
                                                 if current_nav_cost < min_nav_img_comm:
                                                     min_nav_img_comm = current_nav_cost
                                                     best_w_img = w_img # Store the best img waypoint for calibration cost calculation

                                        # Calculate calibration cost if needed and possible
                                        calibration_cost = 0
                                        if calibration_needed:
                                            min_cal_trip_cost = math.inf
                                            if best_w_img and suitable_cal_waypoints: # Need a valid img waypoint and cal waypoints
                                                for w_cal in suitable_cal_waypoints:
                                                    nav_img_to_cal = self.get_distance(rover, best_w_img, w_cal)
                                                    nav_cal_to_img = self.get_distance(rover, w_cal, best_w_img)
                                                    if nav_img_to_cal != math.inf and nav_cal_to_img != math.inf:
                                                        min_cal_trip_cost = min(min_cal_trip_cost, nav_img_to_cal + 1 + nav_cal_to_img) # +1 for calibrate action

                                            if min_cal_trip_cost == math.inf:
                                                 # Cannot calibrate this camera with this rover from the best img waypoint
                                                 continue # Skip this camera option
                                            else:
                                                 calibration_cost = min_cal_trip_cost


                                        if min_nav_img_comm != math.inf:
                                            # Cost = (Nav to img + Nav img to comm) + 1 (take_image) + 1 (communicate) + calibration_cost
                                            current_option_cost = min_nav_img_comm + 1 + 1 + calibration_cost
                                            min_take_comm_cost = min(min_take_comm_cost, current_option_cost)

                        goal_cost = min(goal_cost, min_take_comm_cost)

                total_cost += goal_cost

        # If total_cost is infinity, it means at least one goal is unreachable.
        # Return infinity in this case.
        return total_cost
