from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty string or malformed fact
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()


def match(fact_parts, *args):
    """
    Check if a list of fact parts matches a given pattern.

    - `fact_parts`: The components of a fact as a list, e.g., ["at", "ball1", "rooma"].
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact parts match the pattern, else `False`.
    """
    if len(fact_parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(fact_parts, args))


def bfs(graph, start_node, target_nodes):
    """
    Performs BFS to find the shortest distance from start_node to any node in target_nodes.
    Returns float('inf') if no target is reachable or target_nodes is empty.
    """
    if not target_nodes or start_node is None or start_node not in graph:
         return float('inf')

    q = deque([(start_node, 0)])
    visited = {start_node}

    while q:
        current_node, distance = q.popleft()

        if current_node in target_nodes:
            return distance

        # Ensure current_node exists in the graph keys before accessing neighbors
        if current_node in graph:
            for neighbor in graph[current_node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    q.append((neighbor, distance + 1))

    return float('inf') # No target node was reached


class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve all
    unmet goal conditions. It sums the estimated costs for each goal
    independently. The cost for a goal is estimated based on the minimum
    travel distance for a suitable rover to relevant locations (sample site,
    image location, calibration location, communication location) plus the
    necessary action costs (sample, take image, calibrate, communicate).
    Travel distances are computed using Breadth-First Search (BFS) on the
    rover-specific waypoint graph.

    # Assumptions:
    - The heuristic ignores resource constraints like store availability for
      sampling. It assumes an empty store is always available when needed for
      sampling or rock analysis.
    - The heuristic simplifies the image goal cost calculation by summing
      minimum travel costs from the rover's current location to the image
      location, calibration location (if needed), and communication location,
      plus action costs, regardless of the optimal sequence of visits. It does
      not accurately model sequential travel costs (e.g., travel from image
      location to communication location).
    - The heuristic assumes that if a soil or rock sample goal is not met,
      and the sample is no longer on the ground but was present initially,
      some equipped rover must possess the sample, and the cost is just
      communication.
    - If a goal requires a sample that was not initially present, or an image
      of an objective/mode combination that cannot be achieved (e.g., no
      visible_from waypoints for the objective/calibration target, no suitable
      camera), the heuristic assigns a large finite cost (1000) indicating
      impossibility.
    - All rovers are assumed to have a known location in the state.

    # Heuristic Initialization
    The heuristic precomputes static information from the task definition:
    - Lander locations.
    - Rover-specific waypoint graphs based on `visible` and `can_traverse` facts.
    - The set of waypoints from which communication with the lander is possible.
    - For each objective/mode goal, the set of waypoints from which the objective is visible (`visible_from`).
    - For each camera, the set of waypoints from which its calibration target is visible (`visible_from`).
    - The initial locations of soil and rock samples.
    - Which capabilities (soil, rock, imaging) each rover has.
    - Which cameras are on board each rover and which modes they support.
    - The calibration target for each camera.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic computes the sum of estimated costs for
    each goal condition that is not yet satisfied:

    1.  **Parse Current State:** Extract dynamic facts such as rover locations,
        collected samples (`have_soil_analysis`, `have_rock_analysis`), taken
        images (`have_image`), camera calibration status (`calibrated`), and
        remaining samples on the ground (`at_soil_sample`, `at_rock_sample`).

    2.  **Iterate Through Goals:** For each goal `G` in the task's goals:
        -   If `G` is true in the current state, the cost for this goal is 0.
        -   If `G` is `(communicated_soil_data ?w)` and is not true:
            -   Check if any rover `?r` currently has `(have_soil_analysis ?r ?w)`.
                -   If yes: The cost is the minimum travel distance for such a rover `?r` from its current location to any communication waypoint, plus 1 action (communicate). Minimize this over all rovers holding the sample.
                -   If no rover has the sample:
                    -   Check if `(at_soil_sample ?w)` is true in the *current state*.
                        -   If yes: The cost is the minimum travel distance for an equipped rover `?r` from its current location to `?w`, plus 1 action (sample), plus the minimum travel distance for `?r` from `?w` to any communication waypoint, plus 1 action (communicate). Minimize this over all equipped rovers.
                        -   If no (sample not on ground): Check if `?w` had a sample initially (`?w in self.soil_sample_wps_initial`).
                            -   If yes (sample was taken): Assume an equipped rover has it. The cost is the minimum travel distance for an equipped rover `?r` from its current location to any communication waypoint, plus 1 action (communicate). Minimize over equipped rovers.
                            -   If no (sample never existed): The goal is impossible. Add a large cost (1000).
        -   If `G` is `(communicated_rock_data ?w)` and is not true:
            -   Follow the same logic as for soil data, substituting rock predicates and capabilities.
        -   If `G` is `(communicated_image_data ?o ?m)` and is not true:
            -   Check if any rover `?r` currently has `(have_image ?r ?o ?m)`.
                -   If yes: The cost is the minimum travel distance for such a rover `?r` from its current location to any communication waypoint, plus 1 action (communicate). Minimize over all rovers holding the image.
                -   If no rover has the image:
                    -   Find suitable rover `?r` and camera `?i` (equipped `r`, camera `i` on `r` supporting `m`).
                    -   Identify relevant waypoints: image waypoints for `?o`, calibration waypoints for `?i`, and communication waypoints. If any required set of waypoints is empty, the goal is impossible. Add a large cost (1000).
                    -   Otherwise, for each suitable `(r, i)` pair, calculate the cost as the sum of:
                        -   Minimum travel distance from `r`'s current location to any image waypoint for `?o`.
                        -   Minimum travel distance from `r`'s current location to any communication waypoint.
                        -   Minimum travel distance from `r`'s current location to any calibration waypoint for `?i`, plus 1 action (calibrate), *only if* camera `?i` is not currently calibrated on rover `?r`.
                        -   1 action (take image).
                        -   1 action (communicate).
                    -   Minimize this total cost over all suitable `(r, i)` pairs. Add the minimum cost to the total heuristic value.

    3.  **Return Total Cost:** The sum accumulated across all unachieved goals is the heuristic value for the state. If the total cost is 0, it indicates the goal state has been reached.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # Data structures to build from static facts
        self.lander_locations = set()
        self.rover_graphs = {} # rover -> {waypoint -> {neighbor_waypoint}}
        self.comm_wps = set()
        self.image_wps = {} # (objective, mode) -> {waypoint}
        self.calib_wps = {} # camera -> {waypoint}
        self.soil_sample_wps_initial = set() # Initial locations
        self.rock_sample_wps_initial = set() # Initial locations
        self.rover_capabilities = {} # rover -> {soil, rock, imaging}
        self.rover_cameras = {} # rover -> {camera -> {mode}}
        self.camera_calib_target = {} # camera -> objective

        # Build static data structures
        all_waypoints = set()
        all_rovers = set()
        all_cameras = set()
        all_objectives = set()
        all_modes = set()
        all_landers = set()

        # First pass to identify objects and basic relationships
        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            if parts[0] == 'at_lander':
                lander, wp = parts[1], parts[2]
                self.lander_locations.add(wp)
                all_landers.add(lander)
                all_waypoints.add(wp)
            elif parts[0] == 'at': # Initial rover locations are not static, but waypoints are.
                 rover, wp = parts[1], parts[2]
                 all_rovers.add(rover)
                 all_waypoints.add(wp)
            elif parts[0] == 'can_traverse':
                 rover, wp1, wp2 = parts[1], parts[2], parts[3]
                 all_rovers.add(rover)
                 all_waypoints.add(wp1)
                 all_waypoints.add(wp2)
            elif parts[0] == 'visible':
                 wp1, wp2 = parts[1], parts[2]
                 all_waypoints.add(wp1)
                 all_waypoints.add(wp2)
            elif parts[0] == 'equipped_for_soil_analysis':
                 rover = parts[1]
                 self.rover_capabilities.setdefault(rover, set()).add('soil')
                 all_rovers.add(rover)
            elif parts[0] == 'equipped_for_rock_analysis':
                 rover = parts[1]
                 self.rover_capabilities.setdefault(rover, set()).add('rock')
                 all_rovers.add(rover)
            elif parts[0] == 'equipped_for_imaging':
                 rover = parts[1]
                 self.rover_capabilities.setdefault(rover, set()).add('imaging')
                 all_rovers.add(rover)
            elif parts[0] == 'at_soil_sample':
                 wp = parts[1]
                 self.soil_sample_wps_initial.add(wp)
                 all_waypoints.add(wp)
            elif parts[0] == 'at_rock_sample':
                 wp = parts[1]
                 self.rock_sample_wps_initial.add(wp)
                 all_waypoints.add(wp)
            elif parts[0] == 'visible_from':
                 obj, wp = parts[1], parts[2]
                 all_objectives.add(obj)
                 all_waypoints.add(wp)
            elif parts[0] == 'calibration_target':
                 camera, obj = parts[1], parts[2]
                 self.camera_calib_target[camera] = obj
                 all_cameras.add(camera)
                 all_objectives.add(obj)
            elif parts[0] == 'on_board':
                 camera, rover = parts[1], parts[2]
                 self.rover_cameras.setdefault(rover, {}).setdefault(camera, set())
                 all_cameras.add(camera)
                 all_rovers.add(rover)
            elif parts[0] == 'supports':
                 camera, mode = parts[1], parts[2]
                 all_cameras.add(camera)
                 all_modes.add(mode)

        # Second pass for complex relationships and graph
        visible_map = {} # wp -> {neighbor_wp}
        for fact in static_facts:
             parts = get_parts(fact)
             if not parts: continue
             if parts[0] == 'visible':
                 wp1, wp2 = parts[1], parts[2]
                 visible_map.setdefault(wp1, set()).add(wp2)
                 visible_map.setdefault(wp2, set()).add(wp1) # visible is symmetric

        for rover in all_rovers:
            self.rover_graphs[rover] = {}
            for wp in all_waypoints:
                self.rover_graphs[rover][wp] = set()

        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue
            if parts[0] == 'can_traverse':
                rover, wp1, wp2 = parts[1], parts[2], parts[3]
                # A rover can traverse from wp1 to wp2 if can_traverse is true AND visible is true
                if wp1 in visible_map and wp2 in visible_map[wp1]:
                     self.rover_graphs[rover][wp1].add(wp2)

        # Populate rover_cameras with modes
        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue
            if parts[0] == 'on_board':
                 camera, rover = parts[1], parts[2]
                 # Find modes supported by this camera
                 supported_modes = {get_parts(f)[2] for f in static_facts if match(get_parts(f), 'supports', camera, '*')}
                 self.rover_cameras.setdefault(rover, {})[camera] = supported_modes

        # Populate communication waypoints
        # Assuming there is at least one lander location
        if self.lander_locations:
            lander_loc = list(self.lander_locations)[0] # Assuming one lander
            # Communication is possible from any waypoint visible from the lander location
            self.comm_wps = visible_map.get(lander_loc, set())

        # Populate image waypoints based on goals
        for goal in self.goals:
            parts = get_parts(goal)
            if match(parts, 'communicated_image_data', '*', '*'):
                obj, mode = parts[1], parts[2]
                # Find all waypoints visible from this objective
                image_wps_for_obj = {get_parts(f)[2] for f in static_facts if match(get_parts(f), 'visible_from', obj, '*')}
                self.image_wps[(obj, mode)] = image_wps_for_obj

        # Populate calibration waypoints
        for camera, target_obj in self.camera_calib_target.items():
             # Find all waypoints visible from the calibration target objective
             calib_wps_for_target = {get_parts(f)[2] for f in static_facts if match(get_parts(f), 'visible_from', target_obj, '*')}
             self.calib_wps[camera] = calib_wps_for_target


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Parse dynamic state facts
        rover_locations = {}
        have_soil = set() # {(rover, waypoint)}
        have_rock = set() # {(rover, waypoint)}
        have_image = set() # {(rover, objective, mode)}
        calibrated_cameras = set() # {(camera, rover)}
        at_soil_sample_state = set() # {(waypoint)} - samples remaining on ground
        at_rock_sample_state = set() # {(waypoint)} - samples remaining on ground
        # empty_stores = set() # {store} # Not used in this heuristic simplification

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            if parts[0] == 'at' and parts[1].startswith('rover'): # Assuming objects starting with 'rover' are rovers
                rover_locations[parts[1]] = parts[2]
            elif parts[0] == 'have_soil_analysis':
                have_soil.add((parts[1], parts[2]))
            elif parts[0] == 'have_rock_analysis':
                have_rock.add((parts[1], parts[2]))
            elif parts[0] == 'have_image':
                have_image.add((parts[1], parts[2], parts[3]))
            elif parts[0] == 'calibrated':
                calibrated_cameras.add((parts[1], parts[2]))
            elif parts[0] == 'at_soil_sample':
                 at_soil_sample_state.add(parts[1])
            elif parts[0] == 'at_rock_sample':
                 at_rock_sample_state.add(parts[1])
            # elif parts[0] == 'empty':
            #      empty_stores.add(parts[1])


        total_cost = 0
        IMPOSSIBLE_COST = 1000 # Large finite cost for impossible goals

        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue

            goal_predicate = parts[0]

            if goal_predicate == 'communicated_soil_data':
                waypoint = parts[1]
                if goal not in state:
                    min_goal_cost = float('inf')

                    # Option 1: Find a rover that already has the sample
                    rover_with_sample = next(((r, w) for r, w in have_soil if w == waypoint), None)
                    if rover_with_sample:
                        rover, _ = rover_with_sample
                        current_loc = rover_locations.get(rover)
                        if current_loc:
                            move_cost = bfs(self.rover_graphs.get(rover, {}), current_loc, self.comm_wps)
                            if move_cost != float('inf'):
                                min_goal_cost = min(min_goal_cost, move_cost + 1) # +1 for communicate action

                    # Option 2: Sample the soil and communicate
                    equipped_rovers = [r for r, caps in self.rover_capabilities.items() if 'soil' in caps]
                    if equipped_rovers:
                        # Check if sample is currently on the ground
                        if waypoint in at_soil_sample_state:
                            for rover in equipped_rovers:
                                current_loc = rover_locations.get(rover)
                                if current_loc:
                                    move_to_sample_cost = bfs(self.rover_graphs.get(rover, {}), current_loc, {waypoint})
                                    move_to_comm_cost = bfs(self.rover_graphs.get(rover, {}), waypoint, self.comm_wps)
                                    if move_to_sample_cost != float('inf') and move_to_comm_cost != float('inf'):
                                        # Sample action requires empty store - ignored for simplicity
                                        current_goal_cost = move_to_sample_cost + 1 + move_to_comm_cost + 1
                                        min_goal_cost = min(min_goal_cost, current_goal_cost)
                        elif waypoint in self.soil_sample_wps_initial:
                             # Sample was initially present but is no longer on the ground.
                             # Assume a rover collected it (even if not explicitly tracked in have_soil).
                             # Cost is just communication cost from rover's current location.
                             for rover in equipped_rovers: # Assume any equipped rover *could* have sampled it
                                 current_loc = rover_locations.get(rover)
                                 if current_loc:
                                     move_to_comm_cost = bfs(self.rover_graphs.get(rover, {}), current_loc, self.comm_wps)
                                     if move_to_comm_cost != float('inf'):
                                          current_goal_cost = move_to_comm_cost + 1
                                          min_goal_cost = min(min_goal_cost, current_goal_cost)
                        # Else: Sample was never at this waypoint initially. Impossible. min_goal_cost remains inf.


                    # Add the minimum cost found for this goal
                    total_cost += (min_goal_cost if min_goal_cost != float('inf') else IMPOSSIBLE_COST)


            elif goal_predicate == 'communicated_rock_data':
                waypoint = parts[1]
                if goal not in state:
                    min_goal_cost = float('inf')

                    # Option 1: Find a rover that already has the sample
                    rover_with_sample = next(((r, w) for r, w in have_rock if w == waypoint), None)
                    if rover_with_sample:
                        rover, _ = rover_with_sample
                        current_loc = rover_locations.get(rover)
                        if current_loc:
                            move_cost = bfs(self.rover_graphs.get(rover, {}), current_loc, self.comm_wps)
                            if move_cost != float('inf'):
                                min_goal_cost = min(min_goal_cost, move_cost + 1) # +1 for communicate action

                    # Option 2: Sample the rock and communicate
                    equipped_rovers = [r for r, caps in self.rover_capabilities.items() if 'rock' in caps]
                    if equipped_rovers:
                        # Check if sample is currently on the ground (using initial locations as reference)
                        if waypoint in at_rock_sample_state:
                             # Sample is still on the ground, need to sample and communicate
                             for rover in equipped_rovers:
                                 current_loc = rover_locations.get(rover)
                                 if current_loc:
                                     # Cost to sample: move to sample waypoint + sample action
                                     move_to_sample_cost = bfs(self.rover_graphs.get(rover, {}), current_loc, {waypoint})
                                     # Cost to communicate after sampling: move from sample waypoint to comm waypoint + communicate action
                                     move_to_comm_cost = bfs(self.rover_graphs.get(rover, {}), waypoint, self.comm_wps)
                                     if move_to_sample_cost != float('inf') and move_to_comm_cost != float('inf'):
                                         # Sample action requires empty store - ignored for simplicity
                                         current_goal_cost = move_to_sample_cost + 1 + move_to_comm_cost + 1
                                         min_goal_cost = min(min_goal_cost, current_goal_cost)
                        elif waypoint in self.rock_sample_wps_initial:
                             # Sample was initially present but is no longer on the ground.
                             # Assume a rover collected it (even if not explicitly tracked in have_rock).
                             # Cost is just communication cost from rover's current location.
                             for rover in equipped_rovers: # Assume any equipped rover *could* have sampled it
                                 current_loc = rover_locations.get(rover)
                                 if current_loc:
                                     move_to_comm_cost = bfs(self.rover_graphs.get(rover, {}), current_loc, self.comm_wps)
                                     if move_to_comm_cost != float('inf'):
                                          current_goal_cost = move_to_comm_cost + 1
                                          min_goal_cost = min(min_goal_cost, current_goal_cost)
                        # Else: Sample was never at this waypoint initially. Impossible. min_goal_cost remains inf.

                    # Add the minimum cost found for this goal
                    total_cost += (min_goal_cost if min_goal_cost != float('inf') else IMPOSSIBLE_COST)


            elif goal_predicate == 'communicated_image_data':
                obj, mode = parts[1], parts[2]

                if goal not in state:
                    min_goal_cost = float('inf')

                    # Option 1: Find a rover that already has the image
                    rover_with_image = next(((r, o, m) for r, o, m in have_image if o == obj and m == mode), None)
                    if rover_with_image:
                        rover, _, _ = rover_with_image
                        current_loc = rover_locations.get(rover)
                        if current_loc:
                            move_cost = bfs(self.rover_graphs.get(rover, {}), current_loc, self.comm_wps)
                            if move_cost != float('inf'):
                                min_goal_cost = min(min_goal_cost, move_cost + 1) # +1 for communicate action

                    # Option 2: Take the image and communicate
                    suitable_rovers_cameras = []
                    for rover, caps in self.rover_capabilities.items():
                        if 'imaging' in caps:
                            for camera, modes in self.rover_cameras.get(rover, {}).items():
                                if mode in modes:
                                    suitable_rovers_cameras.append((rover, camera)) # Found a rover/camera combo

                    if suitable_rovers_cameras:
                        image_wps_for_goal = self.image_wps.get((obj, mode), set())
                        comm_wps_for_goal = self.comm_wps # Communication waypoints are global

                        # Check if goal is possible at all based on static facts
                        if not image_wps_for_goal or not comm_wps_for_goal:
                             # Cannot take image (no visible_from waypoint) or cannot communicate (no comm waypoint)
                             # min_goal_cost remains inf, will be set to IMPOSSIBLE_COST later
                             pass # Handled below
                        else:
                            for rover, camera in suitable_rovers_cameras:
                                current_loc = rover_locations.get(rover)
                                if not current_loc:
                                     continue # Cannot use this rover if location is unknown

                                # Get calibration waypoints for this camera
                                calib_wps_for_camera = self.calib_wps.get(camera, set())

                                # Check if calibration is possible if needed
                                calibration_needed = (camera, rover) not in calibrated_cameras
                                if calibration_needed and not calib_wps_for_camera:
                                     # Calibration is needed but no waypoint exists to calibrate from
                                     continue # Cannot use this rover/camera for this goal

                                # Calculate cost components for this rover/camera
                                # Component 1: Move to image location
                                move_to_image_cost = bfs(self.rover_graphs.get(rover, {}), current_loc, image_wps_for_goal)

                                # Component 2: Calibration cost (move + action) if needed
                                calibration_cost = 0
                                if calibration_needed:
                                     move_to_calib_cost = bfs(self.rover_graphs.get(rover, {}), current_loc, calib_wps_for_camera)
                                     # If cannot reach any calibration waypoint, this rover/camera cannot achieve the goal
                                     if move_to_calib_cost == float('inf'):
                                          continue
                                     calibration_cost = move_to_calib_cost + 1 # Add 1 for calibrate action

                                # Component 3: Take image action
                                take_image_action_cost = 1

                                # Component 4: Move to communication location
                                move_to_comm_cost = bfs(self.rover_graphs.get(rover, {}), current_loc, comm_wps_for_goal)

                                # Component 5: Communicate action
                                communicate_action_cost = 1

                                # If cannot reach image waypoint or comm waypoint, this rover/camera cannot achieve the goal
                                if move_to_image_cost == float('inf') or move_to_comm_cost == float('inf'):
                                     continue

                                # Total cost for this rover/camera combination (sum of components)
                                current_goal_cost = move_to_image_cost + \
                                                    calibration_cost + \
                                                    take_image_action_cost + \
                                                    move_to_comm_cost + \
                                                    communicate_action_cost

                                min_goal_cost = min(min_goal_cost, current_goal_cost)

                    # Add the minimum cost found for this goal
                    total_cost += (min_goal_cost if min_goal_cost != float('inf') else IMPOSSIBLE_COST)


        # The heuristic value is the sum of costs for unachieved goals.
        # If all goals are achieved, total_cost is 0.
        # If some goals are impossible, total_cost will include IMPOSSIBLE_COST.
        # For a greedy best-first search, any state with IMPOSSIBLE_COST will be
        # worse than any state without it, which is a reasonable behavior.
        # If the initial state has IMPOSSIBLE_COST, the problem is unsolvable.
        # The heuristic should be finite for solvable states.
        # If the problem is solvable, the BFS will eventually find paths,
        # and min_goal_cost will be finite for all goals.
        # So, total_cost will be finite for solvable states.

        return total_cost
