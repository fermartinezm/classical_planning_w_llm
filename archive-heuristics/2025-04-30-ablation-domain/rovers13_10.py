from collections import deque
# Assuming Heuristic base class is available in heuristics.heuristic_base
# If not, define a placeholder as done below
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    class Heuristic:
        def __init__(self, task):
            pass
        def __call__(self, node):
            raise NotImplementedError

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential invalid fact strings
    if not isinstance(fact, str) or len(fact) < 2 or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()

# Helper function for BFS
def bfs(start_node, graph, target_nodes=None):
    """
    Performs BFS to find shortest path distances from start_node.
    If target_nodes is provided, returns the minimum distance to any target node.
    Returns float('inf') if start_node is not in graph or no target is reachable.
    """
    if start_node not in graph:
         return float('inf') # Cannot start from a node not in graph

    distances = {start_node: 0}
    queue = deque([start_node])
    visited = {start_node}

    min_dist_to_target = float('inf')
    found_target = False

    # Check if start node is a target
    if target_nodes and start_node in target_nodes:
         min_dist_to_target = 0
         found_target = True

    while queue:
        current_node = queue.popleft()

        # Optimization: If we already found a target and the current distance is greater than the min found,
        # we can stop exploring from this branch. This is valid for finding the minimum distance
        # to *any* node in the target set.
        if found_target and distances[current_node] >= min_dist_to_target:
             continue

        neighbors = graph.get(current_node, [])
        for neighbor in neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                distances[neighbor] = distances[current_node] + 1
                queue.append(neighbor)
                if target_nodes and neighbor in target_nodes:
                     min_dist_to_target = min(min_dist_to_target, distances[neighbor])
                     found_target = True

    if target_nodes:
        return min_dist_to_target if found_target else float('inf')
    else:
        # This case is not strictly needed for the heuristic as designed,
        # but could be used to get all distances from start_node.
        # The current heuristic only uses the target_nodes version.
        return distances # Should return distances to all reachable nodes


class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the minimum number of actions required to satisfy
    each unsatisfied goal predicate independently and sums these estimates.
    It considers the steps needed to collect samples or images and communicate
    the data, including movement costs (estimated by shortest path) and action costs.
    Calibration for imaging is included as a fixed cost if needed.

    # Assumptions
    - Movement between waypoints has a cost of 1 per step, estimated by shortest path in the `can_traverse` graph.
    - Taking a sample (soil/rock) costs 1 action.
    - Taking an image costs 1 action.
    - Calibrating a camera costs a fixed amount (estimated as 2 actions: move to calibration target waypoint + calibrate). This is a simplification; the actual cost depends on the rover's location and the calibration target's location.
    - Communicating data costs 1 action.
    - Any rover equipped for a task can perform it.
    - Store constraints for samples are ignored for simplicity.
    - Optimal choices of rovers, cameras, and intermediate waypoints (for imaging/calibration/communication) are assumed by taking minimum path costs over available options.
    - The lander location is static. Communication can happen from any waypoint visible from the lander's location.
    - The `can_traverse` graph is the same for all rovers and is symmetric.

    # Heuristic Initialization
    The constructor pre-processes static information from the task:
    - Builds a graph of waypoints based on `can_traverse` facts (assuming symmetry and same graph for all rovers).
    - Identifies the lander's waypoint and the set of communication waypoints (visible from the lander).
    - Stores which rovers are equipped for soil, rock, and imaging.
    - Stores camera information (on which rover, supported modes, calibration target).
    - Stores objective visibility from waypoints.
    - Stores soil and rock sample locations.
    - Stores the set of goal predicates.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic value is calculated as follows:
    1. Initialize total heuristic cost to 0.
    2. Parse the current state to get dynamic facts like rover locations, collected data (`have_...`), and calibration status (`calibrated`).
    3. For each goal predicate in the task's goals:
        a. If the goal predicate is already true in the current state, add 0 to the total cost and continue to the next goal.
        b. If the goal is `(communicated_soil_data ?w)` and not true:
            i. Check if `(have_soil_analysis ?r ?w)` is true for any rover `?r`.
               - If yes: Find the minimum path cost from the location of any such rover `?r` to any communication waypoint. Add this cost + 1 (for communication) to the total. If no such rover is currently located or no path exists, this goal is impossible (return infinity).
            ii. Else (need to take sample):
                - Find equipped soil analysis rovers `R_soil`. If `R_soil` is empty or no soil sample at `?w`, this goal is impossible (return infinity).
                - Find the minimum cost over all `?r_eq` in `R_soil`: path cost from `at(?r_eq)` to `?w` + 1 (take sample) + minimum path cost from `?w` to any communication waypoint + 1 (communicate). Add this minimum cost to the total. If no path exists for any equipped rover, this goal is impossible (return infinity).
        c. If the goal is `(communicated_rock_data ?w)` and not true: Similar logic as soil data, using rock-specific predicates and equipment.
        d. If the goal is `(communicated_image_data ?o ?m)` and not true:
            i. Check if `(have_image ?r ?o ?m)` is true for any rover `?r`.
               - If yes: Find the minimum path cost from the location of any such rover `?r` to any communication waypoint. Add this cost + 1 (for communication) to the total. If no such rover is currently located or no path exists, this goal is impossible (return infinity).
            ii. Else (need to take image):
                - Find suitable rover/camera combos `RC_image = {(r, c) | equipped_for_imaging r, on_board c r, supports c m}`. If `RC_image` is empty, this goal is impossible (return infinity).
                - Find image waypoints `W_image = {w | visible_from o w}`. If `W_image` is empty, this goal is impossible (return infinity).
                - Find the minimum cost over all `(r, c)` in `RC_image` and `w_image` in `W_image`:
                    - Path cost from `at(r)` to `w_image`.
                    - Add 1 for the `take_image` action.
                    - Add minimum path cost from `w_image` to any communication waypoint.
                    - Add 1 for the `communicate_image_data` action.
                    - Add calibration cost: If `(calibrated c r)` is *not* in the state, add a fixed cost of 2.
                - Add this minimum total cost for the image goal to the overall total. If no path exists for any combination, this goal is impossible (return infinity).
    4. Return the total heuristic cost. If at any point a goal was determined to be impossible, infinity would have been returned already.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # --- Pre-process Static Facts ---

        # Waypoint graph for pathfinding (assuming symmetric can_traverse for simplicity)
        self.waypoint_graph = {}
        # Lander location
        self.lander_waypoint = None
        # Waypoints visible from lander (communication points)
        self.comm_waypoints = set()
        # Rover equipment
        self.rover_equipment = {} # rover -> set(equipment_type_str)
        # Camera information
        self.camera_info = {} # camera -> {'on_board': rover, 'supports': set(modes), 'calib_target': objective}
        # Objective visibility
        self.objective_visibility = {} # objective -> set(waypoints)
        # Sample locations
        self.soil_sample_locations = set()
        self.rock_sample_locations = set()

        all_waypoints_mentioned = set()

        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue # Skip invalid facts

            predicate = parts[0]

            if predicate == 'can_traverse':
                # Assuming can_traverse is listed for all rovers and is symmetric
                # We build a single undirected graph
                if len(parts) == 4: # (can_traverse rover waypoint1 waypoint2)
                    r, w1, w2 = parts[1:]
                    self.waypoint_graph.setdefault(w1, []).append(w2)
                    self.waypoint_graph.setdefault(w2, []).append(w1) # Assume symmetric
                    all_waypoints_mentioned.add(w1)
                    all_waypoints_mentioned.add(w2)

            elif predicate == 'at_lander':
                # Assuming only one lander
                if len(parts) == 3: # (at_lander lander waypoint)
                    self.lander_waypoint = parts[2]
                    all_waypoints_mentioned.add(self.lander_waypoint)

            elif predicate == 'visible':
                 # These define visibility between waypoints
                 # Used later with lander_waypoint to find comm_waypoints
                 if len(parts) == 3: # (visible waypoint1 waypoint2)
                     w1, w2 = parts[1:]
                     all_waypoints_mentioned.add(w1)
                     all_waypoints_mentioned.add(w2)

            elif predicate.startswith('equipped_for_'):
                if len(parts) == 2: # (equipped_for_imaging rover)
                    eq_type = predicate[len('equipped_for_'):] # e.g., 'soil_analysis'
                    rover = parts[1]
                    self.rover_equipment.setdefault(rover, set()).add(eq_type)

            elif predicate == 'on_board':
                if len(parts) == 3: # (on_board camera rover)
                    camera, rover = parts[1:]
                    self.camera_info.setdefault(camera, {})['on_board'] = rover

            elif predicate == 'supports':
                if len(parts) == 3: # (supports camera mode)
                    camera, mode = parts[1:]
                    self.camera_info.setdefault(camera, {}).setdefault('supports', set()).add(mode)

            elif predicate == 'calibration_target':
                if len(parts) == 3: # (calibration_target camera objective)
                    camera, objective = parts[1:]
                    self.camera_info.setdefault(camera, {})['calib_target'] = objective

            elif predicate == 'visible_from':
                if len(parts) == 3: # (visible_from objective waypoint)
                    objective, waypoint = parts[1:]
                    self.objective_visibility.setdefault(objective, set()).add(waypoint)
                    all_waypoints_mentioned.add(waypoint)

            elif predicate == 'at_soil_sample':
                if len(parts) == 2: # (at_soil_sample waypoint)
                    self.soil_sample_locations.add(parts[1])
                    all_waypoints_mentioned.add(parts[1])

            elif predicate == 'at_rock_sample':
                if len(parts) == 2: # (at_rock_sample waypoint)
                    self.rock_sample_locations.add(parts[1])
                    all_waypoints_mentioned.add(parts[1])

        # Now process visible facts to find comm_waypoints
        if self.lander_waypoint:
             for fact in static_facts:
                 parts = get_parts(fact)
                 if parts and parts[0] == 'visible' and len(parts) == 3:
                     w1, w2 = parts[1:]
                     if w1 == self.lander_waypoint:
                         self.comm_waypoints.add(w2)
                     if w2 == self.lander_waypoint:
                         self.comm_waypoints.add(w1)
             # Lander waypoint itself is also a communication point if a rover can be there
             self.comm_waypoints.add(self.lander_waypoint)


        # Ensure all waypoints mentioned in static facts are in the graph, even if isolated
        for w in all_waypoints_mentioned:
             self.waypoint_graph.setdefault(w, [])


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # --- Extract Dynamic Facts from State ---
        rover_locations = {}
        have_soil_analysis_facts = set()
        have_rock_analysis_facts = set()
        have_image_facts = set()
        calibrated_facts = set()
        communicated_facts = set()

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]

            if predicate == 'at' and len(parts) == 3 and parts[1].startswith('rover'):
                rover, waypoint = parts[1:]
                rover_locations[rover] = waypoint
            elif predicate == 'have_soil_analysis' and len(parts) == 3:
                have_soil_analysis_facts.add(fact)
            elif predicate == 'have_rock_analysis' and len(parts) == 3:
                have_rock_analysis_facts.add(fact)
            elif predicate == 'have_image' and len(parts) == 4:
                have_image_facts.add(fact)
            elif predicate == 'calibrated' and len(parts) == 3:
                calibrated_facts.add(fact)
            elif predicate.startswith('communicated_') and len(parts) >= 2:
                 communicated_facts.add(fact)

        total_cost = 0

        # --- Calculate Cost for Each Goal ---
        for goal in self.goals:
            if goal in communicated_facts:
                continue # Goal already achieved

            parts = get_parts(goal)
            if not parts or len(parts) < 2:
                 # Invalid goal format, return infinity as it's an unachievable goal.
                 return float('inf')

            goal_predicate = parts[0]

            if goal_predicate == 'communicated_soil_data':
                if len(parts) != 2: return float('inf') # Invalid goal format
                waypoint_w = parts[1]

                # Check if analysis data is already collected by any rover
                rovers_with_data = [get_parts(f)[1] for f in have_soil_analysis_facts if len(get_parts(f)) > 2 and get_parts(f)[2] == waypoint_w]

                if rovers_with_data:
                    # Data collected, need to communicate
                    min_comm_cost = float('inf')
                    for r in rovers_with_data:
                        if r in rover_locations:
                            current_rover_loc = rover_locations[r]
                            cost_to_comm = bfs(current_rover_loc, self.waypoint_graph, self.comm_waypoints)
                            if cost_to_comm != float('inf'):
                                 min_comm_cost = min(min_comm_cost, cost_to_comm + 1) # +1 for communicate action

                    if min_comm_cost == float('inf'):
                         return float('inf') # Cannot communicate from current location(s)
                    total_cost += min_comm_cost

                else:
                    # Need to take sample and communicate
                    min_total_soil_cost = float('inf')
                    equipped_rovers = [r for r, eq_types in self.rover_equipment.items() if 'soil_analysis' in eq_types]

                    if not equipped_rovers or waypoint_w not in self.soil_sample_locations:
                        return float('inf') # Cannot take sample (no equipped rover or no sample at waypoint)

                    for r_eq in equipped_rovers:
                        if r_eq in rover_locations:
                            current_rover_loc = rover_locations[r_eq]
                            # Cost to move to sample location
                            cost_move_to_sample = bfs(current_rover_loc, self.waypoint_graph, [waypoint_w])
                            if cost_move_to_sample == float('inf'): continue # Cannot reach sample

                            # Cost to take sample
                            cost_take_sample = 1

                            # Cost to move from sample location to comm point
                            cost_move_to_comm = bfs(waypoint_w, self.waypoint_graph, self.comm_waypoints)
                            if cost_move_to_comm == float('inf'): continue # Cannot reach comm point

                            # Cost to communicate
                            cost_communicate = 1

                            min_total_soil_cost = min(min_total_soil_cost, cost_move_to_sample + cost_take_sample + cost_move_to_comm + cost_communicate)

                    if min_total_soil_cost == float('inf'):
                         return float('inf') # Cannot achieve goal (no path for any equipped rover)

                    total_cost += min_total_soil_cost

            elif goal_predicate == 'communicated_rock_data':
                if len(parts) != 2: return float('inf') # Invalid goal format
                waypoint_w = parts[1]

                # Check if analysis data is already collected by any rover
                rovers_with_data = [get_parts(f)[1] for f in have_rock_analysis_facts if len(get_parts(f)) > 2 and get_parts(f)[2] == waypoint_w]

                if rovers_with_data:
                    # Data collected, need to communicate
                    min_comm_cost = float('inf')
                    for r in rovers_with_data:
                        if r in rover_locations:
                            current_rover_loc = rover_locations[r]
                            cost_to_comm = bfs(current_rover_loc, self.waypoint_graph, self.comm_waypoints)
                            if cost_to_comm != float('inf'):
                                 min_comm_cost = min(min_comm_cost, cost_to_comm + 1) # +1 for communicate action

                    if min_comm_cost == float('inf'):
                         return float('inf') # Cannot communicate from current location(s)
                    total_cost += min_comm_cost

                else:
                    # Need to take sample and communicate
                    min_total_rock_cost = float('inf')
                    equipped_rovers = [r for r, eq_types in self.rover_equipment.items() if 'rock_analysis' in eq_types]

                    if not equipped_rovers or waypoint_w not in self.rock_sample_locations:
                        return float('inf') # Cannot take sample (no equipped rover or no sample at waypoint)

                    for r_eq in equipped_rovers:
                        if r_eq in rover_locations:
                            current_rover_loc = rover_locations[r_eq]
                            # Cost to move to sample location
                            cost_move_to_sample = bfs(current_rover_loc, self.waypoint_graph, [waypoint_w])
                            if cost_move_to_sample == float('inf'): continue # Cannot reach sample

                            # Cost to take sample
                            cost_take_sample = 1

                            # Cost to move from sample location to comm point
                            cost_move_to_comm = bfs(waypoint_w, self.waypoint_graph, self.comm_waypoints)
                            if cost_move_to_comm == float('inf'): continue # Cannot reach comm point

                            # Cost to communicate
                            cost_communicate = 1

                            min_total_rock_cost = min(min_total_rock_cost, cost_move_to_sample + cost_take_sample + cost_move_to_comm + cost_communicate)

                    if min_total_rock_cost == float('inf'):
                         return float('inf') # Cannot achieve goal (no path for any equipped rover)

                    total_cost += min_total_rock_cost

            elif goal_predicate == 'communicated_image_data':
                if len(parts) != 3: return float('inf') # Invalid goal format
                objective_o, mode_m = parts[1:]

                # Check if image data is already collected by any rover
                rovers_with_data = [get_parts(f)[1] for f in have_image_facts if len(get_parts(f)) > 3 and get_parts(f)[2] == objective_o and get_parts(f)[3] == mode_m]

                if rovers_with_data:
                    # Image taken, need to communicate
                    min_comm_cost = float('inf')
                    for r in rovers_with_data:
                        if r in rover_locations:
                            current_rover_loc = rover_locations[r]
                            cost_to_comm = bfs(current_rover_loc, self.waypoint_graph, self.comm_waypoints)
                            if cost_to_comm != float('inf'):
                                 min_comm_cost = min(min_comm_cost, cost_to_comm + 1) # +1 for communicate action

                    if min_comm_cost == float('inf'):
                         return float('inf') # Cannot communicate from current location(s)
                    total_cost += min_comm_cost

                else:
                    # Need to take image and communicate
                    min_total_image_cost = float('inf')

                    # Find suitable rover/camera combos
                    suitable_rc_combos = []
                    for r, eq_types in self.rover_equipment.items():
                        if 'imaging' in eq_types:
                            for c, c_info in self.camera_info.items():
                                if c_info.get('on_board') == r and mode_m in c_info.get('supports', set()):
                                    suitable_rc_combos.append((r, c))

                    if not suitable_rc_combos:
                        return float('inf') # No suitable rover/camera

                    # Find image waypoints
                    image_waypoints = self.objective_visibility.get(objective_o, set())
                    if not image_waypoints:
                        return float('inf') # Objective not visible from anywhere

                    # Iterate through all suitable combos and image waypoints to find the minimum cost
                    for r_eq, c in suitable_rc_combos:
                        if r_eq not in rover_locations: continue # Should not happen in valid state

                        current_rover_loc = rover_locations[r_eq]

                        # Check calibration status for this specific camera on this rover
                        is_calibrated = f'(calibrated {c} {r_eq})' in calibrated_facts
                        calib_cost = 0
                        if not is_calibrated:
                            # Add fixed cost for calibration if needed for this camera/rover
                            calib_cost = 2 # Move to calib point + calibrate (simplified)

                        for w_image in image_waypoints:
                            # Cost to move to image location
                            cost_move_to_image = bfs(current_rover_loc, self.waypoint_graph, [w_image])
                            if cost_move_to_image == float('inf'): continue # Cannot reach image waypoint

                            # Cost to take image
                            cost_take_image = 1

                            # Cost to move from image location to comm point
                            cost_move_to_comm = bfs(w_image, self.waypoint_graph, self.comm_waypoints)
                            if cost_move_to_comm == float('inf'): continue # Cannot reach comm point

                            # Cost to communicate
                            cost_communicate = 1

                            # Total cost for this combo and image waypoint
                            current_combo_cost = cost_move_to_image + calib_cost + cost_take_image + cost_move_to_comm + cost_communicate
                            min_total_image_cost = min(min_total_image_cost, current_combo_cost)

                    if min_total_image_cost == float('inf'):
                         return float('inf') # Cannot achieve goal (no path for any combination)

                    total_cost += min_total_image_cost

            # Add other goal types if necessary (e.g., store goals, although not in examples)
            # Based on examples, only communicated_... goals exist.

        # If we reached here, total_cost is the sum of finite costs for all unsatisfied goals.
        # If any goal was impossible, we would have returned float('inf') earlier.
        # If total_cost is 0, it means all goals were satisfied.
        return total_cost
