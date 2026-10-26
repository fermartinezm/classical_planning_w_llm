import sys
from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic

# Define a large value to represent infinity for unreachable states/goals
INF = float('inf')

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact string or malformed fact
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
        graph: Adjacency list representation (dict: node -> set(neighbors)).
        start_node: The starting node.
        all_nodes: A set of all possible nodes in the graph.

    Returns:
        A dictionary mapping node to its distance from start_node.
        Returns float('inf') for unreachable nodes.
    """
    distances = {node: INF for node in all_nodes}

    if start_node not in all_nodes:
         # Start node is not a known waypoint, likely an error or isolated object
         # Distance to itself is 0, others inf.
         if start_node in distances:
             distances[start_node] = 0
         return distances

    distances[start_node] = 0
    queue = deque([start_node])

    while queue:
        current_node = queue.popleft()

        # Get neighbors, handle nodes that might be in all_nodes but not graph (no visible edges)
        neighbors = graph.get(current_node, set())

        for neighbor in neighbors:
            if distances[neighbor] == INF:
                distances[neighbor] = distances[current_node] + 1
                queue.append(neighbor)

    return distances


class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve all
    unmet goal conditions. It sums the estimated costs for each individual
    unmet goal fact, ignoring potential positive interactions between goals
    (e.g., navigating to a waypoint that serves multiple purposes). The
    cost for each goal fact is estimated based on the minimum actions
    needed to acquire the required data (sample or image) and then
    communicate it to the lander. Navigation costs are estimated using
    precomputed shortest paths on the 'visible' graph.

    # Assumptions
    - The 'visible' predicate defines the traversable graph for navigation.
      (Note: The domain uses 'can_traverse' which is rover-specific, but
       for simplicity and efficiency, this heuristic uses the symmetric
       'visible' graph for all rover navigation cost estimates).
    - Samples (soil/rock) required by goals exist in the initial state.
    - Rovers equipped for sampling have a store available (either empty
      initially or can be emptied with one 'drop' action).
    - Cameras required for imaging goals have calibration targets and
      imaging locations visible from some waypoints.
    - If a goal requires data (sample/image) that is already acquired
      by a rover, the cost to acquire it is 0.
    - Communication happens from the location where the data was acquired
      (if acquired in this state) or from the current location of a rover
      that already possesses the data (if acquired previously).
    - Action costs are uniform (1 per action).

    # Heuristic Initialization
    The heuristic precomputes static information from the task definition:
    - The lander's waypoint.
    - The graph of waypoints connected by the 'visible' predicate.
    - Shortest path distances between all pairs of waypoints using BFS on the 'visible' graph.
    - Waypoints visible from the lander (communication waypoints).
    - Initial locations of soil and rock samples.
    - Waypoints visible from each objective (imaging waypoints).
    - Calibration targets for each camera.
    - Waypoints visible from each calibration target (calibration waypoints).
    - Capabilities (soil, rock, imaging) of each rover.
    - Cameras on board each rover and modes supported by each camera.
    - Store associated with each rover.
    - Identifies goals that are impossible to achieve based on static facts
      (e.g., required sample doesn't exist, no equipped rover, no imaging/calibration waypoints).

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic calculates the sum of estimated costs
    for each goal fact that is not yet satisfied:

    1.  **Identify Unmet Goals:** Iterate through the task's goal conditions. If a goal fact `g` is present in the current state, its contribution to the heuristic is 0. If `g` was determined to be impossible during initialization, return infinity.

    2.  **Estimate Cost for Soil/Rock Communication Goal `(communicated_X_data ?w)`:**
        -   Check if `(have_X_analysis ?r ?w)` is true for any rover `r` in the current state.
        -   **If sample exists:** Cost to acquire sample is 0. Cost to communicate is estimated as the minimum navigation cost from the current location of *any* rover holding the sample to the closest communication waypoint, plus 1 (for the communicate action).
        -   **If sample does not exist:** Cost to acquire sample is estimated as the minimum cost over all equipped rovers `r`: (navigation cost from `r`'s current location to `?w`) + 1 (sample action) + (1 if `r`'s store is full, for a drop action). Cost to communicate is estimated as the minimum navigation cost from `?w` (where the sample will be acquired) to the closest communication waypoint, plus 1 (for the communicate action).
        -   Add the sum of acquisition and communication costs to the total heuristic.

    3.  **Estimate Cost for Image Communication Goal `(communicated_image_data ?o ?m)`:**
        -   Check if `(have_image ?r ?o ?m)` is true for any rover `r` in the current state.
        -   **If image exists:** Cost to acquire image is 0. Cost to communicate is estimated as the minimum navigation cost from the current location of *any* rover holding the image to the closest communication waypoint, plus 1 (for the communicate action).
        -   **If image does not exist:** Cost to acquire image is estimated as the minimum cost over all equipped rovers `r` with a camera `i` supporting mode `m`:
            -   Cost to calibrate camera `i`: If `(calibrated i r)` is false, estimate as (navigation cost from `r`'s current location to the closest calibration waypoint for `i`) + 1 (calibrate action). If true, cost is 0.
            -   Cost to take image: Estimate as (navigation cost from location after calibration to the closest imaging waypoint for `o`) + 1 (take_image action).
            -   The minimum sum of calibration and take_image costs over all suitable rover/camera combinations is the estimated acquisition cost. Let `img_w_best` be the waypoint where the image is taken for the minimum cost.
            -   Cost to communicate is estimated as the minimum navigation cost from `img_w_best` to the closest communication waypoint, plus 1 (for the communicate action).
        -   Add the sum of acquisition and communication costs to the total heuristic.

    4.  **Summation:** The total heuristic value is the sum of the estimated costs for all unmet goal facts.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static_facts = task.static
        self.initial_state = task.initial_state

        # Data structures to store precomputed information
        self.lander_waypoint = None
        self.waypoints = set()
        self.visible_graph = {} # waypoint -> set(neighbor_waypoints)
        self.dist = {} # waypoint -> {waypoint -> distance}
        self.comm_waypoints = set()
        self.initial_soil_samples = set()
        self.initial_rock_samples = set()
        self.imaging_waypoints_by_objective = {} # objective -> set(waypoints)
        self.calibration_targets_by_camera = {} # camera -> objective
        self.calibration_waypoints_by_camera = {} # camera -> set(waypoints)
        self.rover_capabilities = {} # rover -> set(capabilities)
        self.rover_cameras = {} # rover -> set(cameras)
        self.camera_modes = {} # camera -> set(modes)
        self.rover_stores = {} # rover -> store
        self.rovers = set() # Collect all rover objects
        self.cameras = set() # Collect all camera objects
        self.objectives = set() # Collect all objective objects
        self.modes = set() # Collect all mode objects
        self.stores = set() # Collect all store objects

        # --- Parse Initial State and Static Facts ---
        all_facts = set(self.initial_state) | set(self.static_facts)

        for fact in all_facts:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]

            if predicate == "at_lander":
                self.lander_waypoint = parts[2]
                self.waypoints.add(parts[2])
            elif predicate == "at":
                # (at ?x - rover ?y - waypoint) or (at ?b - ball ?r - room) etc.
                # Need to check types if available, but relying on domain structure
                # For rovers, the first arg after predicate is usually the rover
                if len(parts) == 3: # (at obj loc)
                    obj, loc = parts[1], parts[2]
                    # Heuristic only cares about rover locations for navigation
                    # We collect all waypoints mentioned anywhere
                    self.waypoints.add(loc)
                    # Collect rovers
                    if match(fact, "at", "*", "*"): # Simple check, assumes rover is first arg
                         # This is not reliable. Let's collect rovers from equipped/store_of facts.
                         pass # Handled below
            elif predicate == "visible":
                w1, w2 = parts[1], parts[2]
                self.waypoints.add(w1)
                self.waypoints.add(w2)
                self.visible_graph.setdefault(w1, set()).add(w2)
                self.visible_graph.setdefault(w2, set()).add(w1) # Visible is symmetric
            elif predicate == "can_traverse":
                 # (can_traverse ?r - rover ?x - waypoint ?y - waypoint)
                 # We use visible graph for distance, but collect rovers/waypoints
                 if len(parts) == 4:
                     r, w1, w2 = parts[1], parts[2], parts[3]
                     self.rovers.add(r)
                     self.waypoints.add(w1)
                     self.waypoints.add(w2)
            elif predicate == "equipped_for_soil_analysis":
                r = parts[1]
                self.rovers.add(r)
                self.rover_capabilities.setdefault(r, set()).add('soil')
            elif predicate == "equipped_for_rock_analysis":
                r = parts[1]
                self.rovers.add(r)
                self.rover_capabilities.setdefault(r, set()).add('rock')
            elif predicate == "equipped_for_imaging":
                r = parts[1]
                self.rovers.add(r)
                self.rover_capabilities.setdefault(r, set()).add('imaging')
            elif predicate == "store_of":
                s, r = parts[1], parts[2]
                self.stores.add(s)
                self.rovers.add(r)
                self.rover_stores[r] = s
            elif predicate == "at_soil_sample":
                w = parts[1]
                self.waypoints.add(w)
                self.initial_soil_samples.add(w)
            elif predicate == "at_rock_sample":
                w = parts[1]
                self.waypoints.add(w)
                self.initial_rock_samples.add(w)
            elif predicate == "visible_from":
                o, w = parts[1], parts[2]
                self.objectives.add(o)
                self.waypoints.add(w)
                self.imaging_waypoints_by_objective.setdefault(o, set()).add(w)
            elif predicate == "calibration_target":
                i, o = parts[1], parts[2]
                self.cameras.add(i)
                self.objectives.add(o)
                self.calibration_targets_by_camera[i] = o
            elif predicate == "on_board":
                i, r = parts[1], parts[2]
                self.cameras.add(i)
                self.rovers.add(r)
                self.rover_cameras.setdefault(r, set()).add(i)
            elif predicate == "supports":
                i, m = parts[1], parts[2]
                self.cameras.add(i)
                self.modes.add(m)
                self.camera_modes.setdefault(i, set()).add(m)
            # Ignore goal predicates like communicated_soil_data, have_soil_analysis etc.
            # Ignore empty/full store predicates in init, they are dynamic.

        # Ensure all collected waypoints are in the graph structure, even if isolated
        for w in self.waypoints:
             self.visible_graph.setdefault(w, set())

        # --- Precompute Distances ---
        for start_w in self.waypoints:
            self.dist[start_w] = bfs(self.visible_graph, start_w, self.waypoints)

        # --- Precompute Communication Waypoints ---
        if self.lander_waypoint:
            for w in self.waypoints:
                # Check if (visible w lander_waypoint) is a static fact
                if f"(visible {w} {self.lander_waypoint})" in self.static_facts or \
                   f"(visible {self.lander_waypoint} {w})" in self.static_facts:
                    self.comm_waypoints.add(w)

        # --- Precompute Calibration Waypoints ---
        for camera, target_objective in self.calibration_targets_by_camera.items():
            if target_objective in self.imaging_waypoints_by_objective:
                self.calibration_waypoints_by_camera[camera] = self.imaging_waypoints_by_objective[target_objective]
            else:
                self.calibration_waypoints_by_camera[camera] = set() # No visible_from facts for target

        # --- Check Goal Feasibility ---
        self.impossible_goals = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue

            predicate = parts[0]

            if predicate == "communicated_soil_data":
                w = parts[1]
                # Requires sample at w initially AND at least one soil-equipped rover
                if w not in self.initial_soil_samples or \
                   not any('soil' in self.rover_capabilities.get(r, set()) for r in self.rovers):
                    self.impossible_goals.add(goal)
                # Also requires a path from w to a communication waypoint
                elif not self.comm_waypoints or \
                     all(self.dist[w].get(comm_w, INF) == INF for comm_w in self.comm_waypoints):
                     self.impossible_goals.add(goal)

            elif predicate == "communicated_rock_data":
                w = parts[1]
                # Requires sample at w initially AND at least one rock-equipped rover
                if w not in self.initial_rock_samples or \
                   not any('rock' in self.rover_capabilities.get(r, set()) for r in self.rovers):
                    self.impossible_goals.add(goal)
                # Also requires a path from w to a communication waypoint
                elif not self.comm_waypoints or \
                     all(self.dist[w].get(comm_w, INF) == INF for comm_w in self.comm_waypoints):
                     self.impossible_goals.add(goal)

            elif predicate == "communicated_image_data":
                o, m = parts[1], parts[2]
                is_possible = False
                # Requires an imaging-equipped rover with a camera supporting mode m
                # AND imaging waypoints for objective o
                # AND calibration waypoints for that camera's target
                for r in self.rovers:
                    if 'imaging' in self.rover_capabilities.get(r, set()):
                        for i in self.rover_cameras.get(r, set()):
                            if m in self.camera_modes.get(i, set()):
                                # Found a suitable rover/camera/mode combo
                                if o in self.imaging_waypoints_by_objective and self.imaging_waypoints_by_objective[o]:
                                    # Found imaging waypoints for the objective
                                    if i in self.calibration_targets_by_camera and \
                                       i in self.calibration_waypoints_by_camera and \
                                       self.calibration_waypoints_by_camera[i]:
                                        # Found calibration waypoints for the camera
                                        # Check if there's a path from any imaging waypoint for o
                                        # to any communication waypoint.
                                        can_communicate = False
                                        if self.comm_waypoints:
                                            for img_w in self.imaging_waypoints_by_objective[o]:
                                                if img_w in self.dist: # Ensure img_w is a known waypoint
                                                    if any(self.dist[img_w].get(comm_w, INF) != INF for comm_w in self.comm_waypoints):
                                                        can_communicate = True
                                                        break
                                        if can_communicate:
                                            is_possible = True
                                            break # Found a possible path for this goal
                if not is_possible:
                    self.impossible_goals.add(goal)

        # If no communication waypoints exist, all communication goals are impossible
        if not self.comm_waypoints:
             for goal in self.goals:
                 parts = get_parts(goal)
                 if parts and parts[0].startswith("communicated_"):
                     self.impossible_goals.add(goal)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        h = 0

        # Check if goal is reached
        if self.goals <= state:
            return 0

        # --- Parse current state for dynamic facts ---
        current_rover_location = {} # rover -> waypoint
        have_soil_analysis = {} # rover -> set(waypoints)
        have_rock_analysis = {} # rover -> set(waypoints)
        have_image = {} # rover -> set((objective, mode))
        calibrated_camera = set() # set of (camera, rover)
        store_full = set() # set of stores

        for r in self.rovers:
             have_soil_analysis[r] = set()
             have_rock_analysis[r] = set()
             have_image[r] = set()

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]

            if predicate == "at":
                 if len(parts) == 3 and parts[1] in self.rovers and parts[2] in self.waypoints:
                     current_rover_location[parts[1]] = parts[2]
            elif predicate == "have_soil_analysis":
                 if len(parts) == 3 and parts[1] in self.rovers and parts[2] in self.waypoints:
                     have_soil_analysis[parts[1]].add(parts[2])
            elif predicate == "have_rock_analysis":
                 if len(parts) == 3 and parts[1] in self.rovers and parts[2] in self.waypoints:
                     have_rock_analysis[parts[1]].add(parts[2])
            elif predicate == "have_image":
                 if len(parts) == 4 and parts[1] in self.rovers and parts[2] in self.objectives and parts[3] in self.modes:
                     have_image[parts[1]].add((parts[2], parts[3]))
            elif predicate == "calibrated":
                 if len(parts) == 3 and parts[1] in self.cameras and parts[2] in self.rovers:
                     calibrated_camera.add((parts[1], parts[2]))
            elif predicate == "full":
                 if len(parts) == 2 and parts[1] in self.stores:
                     store_full.add(parts[1])

        # --- Estimate cost for each unmet goal ---
        for goal in self.goals:
            if goal in state:
                continue # Goal already achieved

            if goal in self.impossible_goals:
                return INF # Goal is impossible

            parts = get_parts(goal)
            if not parts: continue # Should not happen for valid goals

            predicate = parts[0]

            if predicate == "communicated_soil_data":
                w = parts[1]
                cost_acquire = INF
                location_after_acquire = w # Assume sample is acquired at w

                # Check if sample already exists on any rover
                sample_exists_in_state = False
                rovers_with_sample = []
                for r in self.rovers:
                    if w in have_soil_analysis.get(r, set()):
                        sample_exists_in_state = True
                        rovers_with_sample.append(r)

                if sample_exists_in_state:
                    cost_acquire = 0
                    # Communication happens from the current location of a rover with the sample
                    min_comm_nav_cost = INF
                    for r_has in rovers_with_sample:
                         current_loc = current_rover_location.get(r_has)
                         if current_loc and current_loc in self.dist:
                             for comm_w in self.comm_waypoints:
                                 min_comm_nav_cost = min(min_comm_nav_cost, self.dist[current_loc].get(comm_w, INF))
                    cost_communicate = min_comm_nav_cost + 1 if min_comm_nav_cost != INF else INF

                else: # Need to sample
                    min_sample_cost_for_goal = INF
                    for r in self.rovers:
                        if 'soil' in self.rover_capabilities.get(r, set()):
                            current_loc = current_rover_location.get(r)
                            store_s = self.rover_stores.get(r)

                            if current_loc and current_loc in self.dist and store_s:
                                nav_cost = self.dist[current_loc].get(w, INF)
                                drop_cost = 1 if store_s in store_full else 0
                                sample_cost = nav_cost + 1 + drop_cost # Navigate + Sample + (Optional Drop)
                                min_sample_cost_for_goal = min(min_sample_cost_for_goal, sample_cost)

                    cost_acquire = min_sample_cost_for_goal
                    # Communication happens from the sample location w
                    min_comm_nav_cost = INF
                    if w in self.dist:
                        for comm_w in self.comm_waypoints:
                            min_comm_nav_cost = min(min_comm_nav_cost, self.dist[w].get(comm_w, INF))
                    cost_communicate = min_comm_nav_cost + 1 if min_comm_nav_cost != INF else INF

                h += (cost_acquire + cost_communicate) if cost_acquire != INF and cost_communicate != INF else INF


            elif predicate == "communicated_rock_data":
                w = parts[1]
                cost_acquire = INF
                location_after_acquire = w # Assume sample is acquired at w

                # Check if sample already exists on any rover
                sample_exists_in_state = False
                rovers_with_sample = []
                for r in self.rovers:
                    if w in have_rock_analysis.get(r, set()):
                        sample_exists_in_state = True
                        rovers_with_sample.append(r)

                if sample_exists_in_state:
                    cost_acquire = 0
                    # Communication happens from the current location of a rover with the sample
                    min_comm_nav_cost = INF
                    for r_has in rovers_with_sample:
                         current_loc = current_rover_location.get(r_has)
                         if current_loc and current_loc in self.dist:
                             for comm_w in self.comm_waypoints:
                                 min_comm_nav_cost = min(min_comm_nav_cost, self.dist[current_loc].get(comm_w, INF))
                    cost_communicate = min_comm_nav_cost + 1 if min_comm_nav_cost != INF else INF

                else: # Need to sample
                    min_sample_cost_for_goal = INF
                    for r in self.rovers:
                        if 'rock' in self.rover_capabilities.get(r, set()):
                            current_loc = current_rover_location.get(r)
                            store_s = self.rover_stores.get(r)

                            if current_loc and current_loc in self.dist and store_s:
                                nav_cost = self.dist[current_loc].get(w, INF)
                                drop_cost = 1 if store_s in store_full else 0
                                sample_cost = nav_cost + 1 + drop_cost # Navigate + Sample + (Optional Drop)
                                min_sample_cost_for_goal = min(min_sample_cost_for_goal, sample_cost)

                    cost_acquire = min_sample_cost_for_goal
                    # Communication happens from the sample location w
                    min_comm_nav_cost = INF
                    if w in self.dist:
                        for comm_w in self.comm_waypoints:
                            min_comm_nav_cost = min(min_comm_nav_cost, self.dist[w].get(comm_w, INF))
                    cost_communicate = min_comm_nav_cost + 1 if min_comm_nav_cost != INF else INF

                h += (cost_acquire + cost_communicate) if cost_acquire != INF and cost_communicate != INF else INF


            elif predicate == "communicated_image_data":
                o, m = parts[1], parts[2]
                cost_acquire = INF
                location_after_acquire = None # Will be the imaging waypoint

                # Check if image already exists on any rover
                image_exists_in_state = False
                rovers_with_image = []
                for r in self.rovers:
                    if (o, m) in have_image.get(r, set()):
                        image_exists_in_state = True
                        rovers_with_image.append(r)

                if image_exists_in_state:
                    cost_acquire = 0
                    # Communication happens from the current location of a rover with the image
                    min_comm_nav_cost = INF
                    for r_has in rovers_with_image:
                         current_loc = current_rover_location.get(r_has)
                         if current_loc and current_loc in self.dist:
                             for comm_w in self.comm_waypoints:
                                 min_comm_nav_cost = min(min_comm_nav_cost, self.dist[current_loc].get(comm_w, INF))
                    cost_communicate = min_comm_nav_cost + 1 if min_comm_nav_cost != INF else INF

                else: # Need to take image
                    min_image_cost_for_goal = INF
                    best_img_w_for_goal = None

                    for r in self.rovers:
                        if 'imaging' in self.rover_capabilities.get(r, set()):
                            current_loc = current_rover_location.get(r)
                            if not current_loc or current_loc not in self.dist: continue # Rover location unknown or isolated

                            for i in self.rover_cameras.get(r, set()):
                                if m in self.camera_modes.get(i, set()):
                                    # Found a suitable rover/camera/mode combo

                                    # Check if imaging is possible for this objective/camera/mode (redundant if impossible_goals check is perfect, but safe)
                                    if o not in self.imaging_waypoints_by_objective or not self.imaging_waypoints_by_objective[o]: continue
                                    if i not in self.calibration_targets_by_camera or i not in self.calibration_waypoints_by_camera or not self.calibration_waypoints_by_camera[i]: continue

                                    # Cost to calibrate
                                    cost_calibrate = 0
                                    location_after_calibration = current_loc
                                    if (i, r) not in calibrated_camera:
                                        min_cal_nav_cost = INF
                                        best_cal_w = None
                                        for cal_w in self.calibration_waypoints_by_camera[i]:
                                            if cal_w in self.dist[current_loc]:
                                                nav_cost = self.dist[current_loc][cal_w]
                                                if nav_cost < min_cal_nav_cost:
                                                    min_cal_nav_cost = nav_cost
                                                    best_cal_w = cal_w

                                        if min_cal_nav_cost != INF:
                                            cost_calibrate = min_cal_nav_cost + 1 # Navigate + Calibrate
                                            location_after_calibration = best_cal_w
                                        else:
                                            continue # Cannot calibrate, cannot take image

                                    # Cost to navigate to imaging waypoint and take image
                                    min_img_nav_cost = INF
                                    best_img_w_for_rover_camera = None
                                    if location_after_calibration in self.dist:
                                        for img_w in self.imaging_waypoints_by_objective[o]:
                                            if img_w in self.dist[location_after_calibration]:
                                                nav_cost = self.dist[location_after_calibration][img_w]
                                                if nav_cost < min_img_nav_cost:
                                                    min_img_nav_cost = nav_cost
                                                    best_img_w_for_rover_camera = img_w

                                    if min_img_nav_cost != INF:
                                        cost_take_image_action = min_img_nav_cost + 1 # Navigate + Take Image
                                        cost_for_this_rover_camera = cost_calibrate + cost_take_image_action

                                        if cost_for_this_rover_camera < min_image_cost_for_goal:
                                            min_image_cost_for_goal = cost_for_this_rover_camera
                                            best_img_w_for_goal = best_img_w_for_rover_camera # Store the waypoint where image will be taken
                                    # else: Cannot reach any imaging waypoint for this objective

                    cost_acquire = min_image_cost_for_goal
                    location_after_acquire = best_img_w_for_goal # Communication happens from here

                    # Cost to communicate
                    min_comm_nav_cost = INF
                    if location_after_acquire and location_after_acquire in self.dist:
                        for comm_w in self.comm_waypoints:
                            min_comm_nav_cost = min(min_comm_nav_cost, self.dist[location_after_acquire].get(comm_w, INF))
                    cost_communicate = min_comm_nav_cost + 1 if min_comm_nav_cost != INF else INF

                h += (cost_acquire + cost_communicate) if cost_acquire != INF and cost_communicate != INF else INF

        # Ensure heuristic is 0 only at goal state
        # If h is INF, it means at least one goal is impossible
        if h == INF:
             return INF
        # If h > 0, it's a non-goal state.
        # If h == 0, it should be a goal state (checked at the beginning).
        # This check ensures non-zero for non-goal states if the sum happens to be 0
        # due to all sub-costs being 0 (which implies goals are met).
        return h

