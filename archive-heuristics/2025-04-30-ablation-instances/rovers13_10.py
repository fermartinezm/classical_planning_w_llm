from fnmatch import fnmatch
from collections import deque, defaultdict
from heuristics.heuristic_base import Heuristic # Assuming this path is correct

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Check if the number of parts is at least the number of args
    if len(parts) < len(args):
        return False
    # Check if all args match the corresponding parts
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# Helper functions to check store status
def has_empty_store(state, rover_name, rover_stores):
    """Helper to check if a rover has an empty store."""
    store = rover_stores.get(rover_name)
    if store:
        return f"(empty {store})" in state
    # If rover has no store, it effectively has no empty store for sampling purposes
    return False

def has_full_store(state, rover_name, rover_stores):
    """Helper to check if a rover has a full store."""
    store = rover_stores.get(rover_name)
    if store:
        return f"(full {store})" in state
    # If rover has no store, it effectively has no full store
    return False

class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the total number of actions required to achieve all
    unmet goal conditions. It breaks down the cost for each type of goal
    (communicating soil data, rock data, or image data) by estimating the
    navigation steps and the required actions (sample, drop, calibrate, take_image, communicate).
    It considers the current state of rovers, samples, images, stores, and cameras.

    # Assumptions
    - The waypoint graph (defined by visible and can_traverse) is connected enough
      to reach necessary locations for solvable problems.
    - Navigation cost is 1 per step (edge in the graph).
    - Action costs (sample, drop, calibrate, take_image, communicate) are 1.
    - The heuristic sums the estimated costs for each unachieved goal independently,
      ignoring potential positive or negative interactions between achieving different goals
      (e.g., a single navigation might serve multiple goals, a drop might enable sampling for multiple goals).
    - If a sample (soil/rock) is required for a goal but is no longer present at the waypoint
      and no rover has the analysis data, the heuristic assumes this path to the goal is
      impossible from the current state via sampling at that waypoint.
    - Calibration is considered per image goal: if a camera is calibrated, the first image
      action for a specific objective/mode pair might skip the calibration cost, but subsequent
      image actions for the same pair (if needed for another goal) would require recalibration.
      The current heuristic simplifies this further: it checks calibration *at the current state*
      when estimating the cost for an unachieved image goal. If calibrated, it assumes no
      calibration cost is needed for *this specific goal's first image*. If not calibrated,
      it adds the calibration cost. It doesn't track calibration state *after* taking the image
      within the heuristic calculation for other goals.

    # Heuristic Initialization
    - Parses static facts to build the waypoint graph (based on `visible` and `can_traverse`).
    - Identifies the lander location and determines the set of waypoints visible from the lander
      (communication points).
    - Stores rover capabilities (equipped for soil, rock, imaging).
    - Stores which store belongs to which rover.
    - Stores which cameras are on board which rovers and which modes they support.
    - Stores camera calibration targets.
    - Stores which objectives are visible from which waypoints.
    - Collects all unique waypoints mentioned in static facts and goals.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1.  Identify all goal conditions that are not met in the current state.
    2.  Initialize total heuristic cost to 0.
    3.  For each unachieved goal:
        -   **Goal: `(communicated_soil_data ?w)`**
            -   Check if any rover `r` currently has `(have_soil_analysis ?r ?w)`.
            -   If yes, find the minimum cost for such a rover `r`: `dist(r_loc, comm_loc) + 1` (communicate), where `comm_loc` is any waypoint visible from the lander. Add this minimum cost to the total.
            -   If no rover has the data, check if `(at_soil_sample ?w)` is in the state.
            -   If the sample is present, find the minimum cost over equipped rovers `r`: `dist(r_loc, w) + 1` (sample) + `dist(w, comm_loc) + 1` (communicate). Add 1 if the rover needs to drop a full store. Add this minimum cost to the total.
            -   If the sample is not present and no rover has the data, assume this goal is unreachable via sampling at `w` from this state and add a large penalty.
        -   **Goal: `(communicated_rock_data ?w)`**
            -   Analogous calculation to `communicated_soil_data ?w`.
        -   **Goal: `(communicated_image_data ?o ?m)`**
            -   Check if any rover `r` currently has `(have_image ?r ?o ?m)`.
            -   If yes, find the minimum cost for such a rover `r`: `dist(r_loc, comm_loc) + 1` (communicate). Add this minimum cost to the total.
            -   If no rover has the image, find the minimum cost over equipped rovers `r` with suitable cameras `i` (`on_board i r`, `supports i m`):
                -   Find potential image waypoints `p` for `o` (`visible_from o p`).
                -   Find potential calibration waypoints `w` for camera `i` (via its target `t`).
                -   If camera `i` is currently calibrated on rover `r`: Cost is `dist(r_loc, p) + 1` (take image) + `dist(p, comm_loc) + 1` (communicate). Find minimum over `p`.
                -   If camera `i` is not calibrated on rover `r`: Cost is `dist(r_loc, w) + 1` (calibrate) + `dist(w, p) + 1` (take image) + `dist(p, comm_loc) + 1` (communicate). Find minimum over `w` and `p`.
                -   Add the minimum cost found for this rover/camera combination to the total.
            -   If no path to getting/communicating the image is found (e.g., no suitable waypoints, no equipped rover/camera), add a large penalty.
        -   **Goal: `(at ?r ?w)`** (Less common, but handle)
            -   If `(at ?r ?w)` is not in state, cost is `dist(r_loc, w)`. Add this to the total.
    4.  Return the total heuristic cost. If any necessary navigation was impossible (BFS returned infinity) or a goal was deemed unreachable, return a large number.
    """

    def __init__(self, task):
        self.goals = task.goals
        static_facts = task.static

        self.lander_location = None
        self.comm_waypoints = set()
        self.rover_capabilities = defaultdict(lambda: {'soil': False, 'rock': False, 'imaging': False})
        self.rover_stores = {} # rover -> store
        self.rover_cameras = defaultdict(list) # rover -> [camera1, ...]
        self.camera_modes = defaultdict(list) # camera -> [mode1, ...]
        self.camera_calibration_targets = {} # camera -> objective
        self.objective_visible_from = defaultdict(list) # objective -> [waypoint1, ...]
        self.waypoint_graph = defaultdict(list) # waypoint -> [neighbor1, ...]
        self.all_waypoints = set() # Keep track of all waypoints

        # Parse static facts
        for fact in static_facts:
            parts = get_parts(fact)
            predicate = parts[0]

            if predicate == "at_lander":
                # (at_lander ?x - lander ?y - waypoint)
                self.lander_location = parts[2]
                self.all_waypoints.add(self.lander_location)
            elif predicate == "equipped_for_soil_analysis":
                self.rover_capabilities[parts[1]]['soil'] = True
            elif predicate == "equipped_for_rock_analysis":
                self.rover_capabilities[parts[1]]['rock'] = True
            elif predicate == "equipped_for_imaging":
                self.rover_capabilities[parts[1]]['imaging'] = True
            elif predicate == "store_of":
                self.rover_stores[parts[2]] = parts[1]
            elif predicate == "on_board":
                self.rover_cameras[parts[2]].append(parts[1])
            elif predicate == "supports":
                self.camera_modes[parts[1]].append(parts[2])
            elif predicate == "calibration_target":
                self.camera_calibration_targets[parts[1]] = parts[2]
            elif predicate == "visible_from":
                # (visible_from ?o - objective ?w - waypoint)
                self.objective_visible_from[parts[1]].append(parts[2])
                self.all_waypoints.add(parts[2]) # Add waypoint to set
            elif predicate == "visible":
                 # (visible ?w - waypoint ?p - waypoint)
                 w1, w2 = parts[1], parts[2]
                 self.waypoint_graph[w1].append(w2)
                 self.waypoint_graph[w2].append(w1) # Assuming visible is symmetric
                 self.all_waypoints.add(w1)
                 self.all_waypoints.add(w2)
            elif predicate == "can_traverse":
                 # (can_traverse ?r - rover ?x - waypoint ?y - waypoint)
                 # Assuming can_traverse is symmetric and applies to all rovers for graph structure
                 w1, w2 = parts[2], parts[3]
                 # Add edges only if not already added by 'visible'
                 if w2 not in self.waypoint_graph[w1]:
                     self.waypoint_graph[w1].append(w2)
                 if w1 not in self.waypoint_graph[w2]:
                     self.waypoint_graph[w2].append(w1)
                 self.all_waypoints.add(w1)
                 self.all_waypoints.add(w2)
            # Add other types to all_waypoints if they are waypoints
            elif predicate in ["at_soil_sample", "at_rock_sample"]:
                 self.all_waypoints.add(parts[1])
            # 'at' facts in static might include initial rover locations
            elif predicate == "at":
                 # (at ?x - rover ?y - waypoint) or (at ?obj ?loc)
                 # Only add waypoint if it's a waypoint type
                 if parts[2].startswith('waypoint'):
                     self.all_waypoints.add(parts[2])


        # Identify communication waypoints (visible from lander)
        if self.lander_location:
             self.comm_waypoints = {
                 get_parts(fact)[1] for fact in static_facts
                 if match(fact, "visible", "*", self.lander_location)
             }
             self.comm_waypoints.update({
                 get_parts(fact)[2] for fact in static_facts
                 if match(fact, "visible", self.lander_location, "*")
             })


        # Collect all unique waypoints mentioned in goals
        for goal in self.goals:
             parts = get_parts(goal)
             if parts[0] in ["communicated_soil_data", "communicated_rock_data", "at"]:
                 self.all_waypoints.add(parts[1])
             # For image goals, the relevant waypoints (image/calibration) are added from visible_from facts.


    def __call__(self, node):
        state = node.state
        total_cost = 0
        LARGE_PENALTY = 1000 # Penalty for potentially unreachable goals

        # Dynamic info parsing
        rover_locations = {}
        have_soil = set() # (rover, waypoint)
        have_rock = set() # (rover, waypoint)
        have_image = set() # (rover, objective, mode)
        soil_samples_at = set() # waypoint
        rock_samples_at = set() # waypoint
        rover_store_status = {} # rover -> 'empty' or 'full'
        camera_calibrated = set() # (camera, rover)

        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == "at":
                # (at ?x - rover ?y - waypoint) or (at ?obj ?loc)
                # We only care about rover locations here
                if parts[1] in self.rover_capabilities: # Check if it's a rover by checking capabilities dict keys
                     rover_locations[parts[1]] = parts[2]
            elif predicate == "have_soil_analysis":
                have_soil.add((parts[1], parts[2]))
            elif predicate == "have_rock_analysis":
                have_rock.add((parts[1], parts[2]))
            elif predicate == "have_image":
                have_image.add((parts[1], parts[2], parts[3]))
            elif predicate == "at_soil_sample":
                soil_samples_at.add(parts[1])
            elif predicate == "at_rock_sample":
                rock_samples_at.add(parts[1])
            elif predicate == "empty":
                 # (empty ?s - store)
                 # Find the rover for this store
                 for rover, store in self.rover_stores.items():
                     if store == parts[1]:
                         rover_store_status[rover] = 'empty'
                         break
            elif predicate == "full":
                 # (full ?s - store)
                 for rover, store in self.rover_stores.items():
                     if store == parts[1]:
                         rover_store_status[rover] = 'full'
                         break
            elif predicate == "calibrated":
                 # (calibrated ?c - camera ?r - rover)
                 camera_calibrated.add((parts[1], parts[2]))

        # Helper BFS function within the call method to use the current graph
        def get_dist(start_node, target_nodes_set):
             """
             Performs BFS to find the shortest distance from start_node to any node in target_nodes_set.
             Returns float('inf') if no target is reachable or if start_node/targets are invalid.
             """
             if start_node is None or not target_nodes_set:
                 return float('inf')

             # Ensure start node is a known waypoint
             if start_node not in self.all_waypoints:
                 # This can happen if a rover is at a location not defined as a waypoint in static facts or goals.
                 # Should not happen in valid PDDL instances? Assume all locations are waypoints.
                 # If it happens, the rover is effectively stranded from the known graph.
                 return float('inf')

             # Filter target_nodes_set to include only known waypoints
             valid_target_nodes_set = {target for target in target_nodes_set if target in self.all_waypoints}
             if not valid_target_nodes_set:
                 return float('inf') # No valid targets in the graph

             # Handle case where start is already a target
             if start_node in valid_target_nodes_set:
                 return 0

             queue = deque([(start_node, 0)])
             visited = {start_node}

             while queue:
                 current_node, current_dist = queue.popleft()

                 # Check neighbors
                 for neighbor in self.waypoint_graph.get(current_node, []):
                     if neighbor not in visited:
                         visited.add(neighbor)
                         if neighbor in valid_target_nodes_set:
                             return current_dist + 1 # Found shortest path to *a* target
                         queue.append((neighbor, current_dist + 1))

             return float('inf') # No path found

        # Process goals
        for goal in self.goals:
            if goal in state:
                continue # Goal already achieved

            parts = get_parts(goal)
            predicate = parts[0]

            if predicate == "communicated_soil_data":
                # (communicated_soil_data ?w - waypoint)
                waypoint_w = parts[1]
                min_goal_cost = float('inf')

                # Option 1: Data already collected by some rover
                rovers_with_data = [r for r, w in have_soil if w == waypoint_w]
                if rovers_with_data:
                    for r in rovers_with_data:
                        r_loc = rover_locations.get(r)
                        if r_loc:
                            dist_to_comm = get_dist(r_loc, self.comm_waypoints)
                            if dist_to_comm != float('inf'):
                                cost = dist_to_comm + 1 # navigate + communicate
                                min_goal_cost = min(min_goal_cost, cost)

                # Option 2: Need to sample and communicate
                # Only consider sampling if the sample is currently available
                if min_goal_cost == float('inf') and waypoint_w in soil_samples_at:
                     equipped_rovers = [r for r, caps in self.rover_capabilities.items() if caps['soil']]
                     for r in equipped_rovers:
                         r_loc = rover_locations.get(r)
                         if r_loc:
                             dist_r_to_w = get_dist(r_loc, {waypoint_w})
                             if dist_r_to_w != float('inf'):
                                 cost_sample = dist_r_to_w + 1 # navigate + sample
                                 # Add cost for dropping if needed (rover has a full store and no empty one)
                                 if has_full_store(state, r, self.rover_stores) and not has_empty_store(state, r, self.rover_stores):
                                     cost_sample += 1 # drop

                                 dist_w_to_comm = get_dist(waypoint_w, self.comm_waypoints)
                                 if dist_w_to_comm != float('inf'):
                                     cost_comm = dist_w_to_comm + 1 # navigate + communicate
                                     min_goal_cost = min(min_goal_cost, cost_sample + cost_comm)

                # Add cost for this goal
                if min_goal_cost == float('inf'):
                    total_cost += LARGE_PENALTY # Cannot achieve this goal via known methods
                else:
                    total_cost += min_goal_cost

            elif predicate == "communicated_rock_data":
                # (communicated_rock_data ?w - waypoint)
                waypoint_w = parts[1]
                min_goal_cost = float('inf')

                # Option 1: Data already collected by some rover
                rovers_with_data = [r for r, w in have_rock if w == waypoint_w]
                if rovers_with_data:
                    for r in rovers_with_data:
                        r_loc = rover_locations.get(r)
                        if r_loc:
                            dist_to_comm = get_dist(r_loc, self.comm_waypoints)
                            if dist_to_comm != float('inf'):
                                cost = dist_to_comm + 1 # navigate + communicate
                                min_goal_cost = min(min_goal_cost, cost)

                # Option 2: Need to sample and communicate
                # Only consider sampling if the sample is currently available
                if min_goal_cost == float('inf') and waypoint_w in rock_samples_at:
                     equipped_rovers = [r for r, caps in self.rover_capabilities.items() if caps['rock']]
                     for r in equipped_rovers:
                         r_loc = rover_locations.get(r)
                         if r_loc:
                             dist_r_to_w = get_dist(r_loc, {waypoint_w})
                             if dist_r_to_w != float('inf'):
                                 cost_sample = dist_r_to_w + 1 # navigate + sample
                                 # Add cost for dropping if needed (rover has a full store and no empty one)
                                 if has_full_store(state, r, self.rover_stores) and not has_empty_store(state, r, self.rover_stores):
                                     cost_sample += 1 # drop

                                 dist_w_to_comm = get_dist(waypoint_w, self.comm_waypoints)
                                 if dist_w_to_comm != float('inf'):
                                     cost_comm = dist_w_to_comm + 1 # navigate + communicate
                                     min_goal_cost = min(min_goal_cost, cost_sample + cost_comm)

                # Add cost for this goal
                if min_goal_cost == float('inf'):
                    total_cost += LARGE_PENALTY # Cannot achieve this goal via known methods
                else:
                    total_cost += min_goal_cost

            elif predicate == "communicated_image_data":
                # (communicated_image_data ?o - objective ?m - mode)
                objective_o = parts[1]
                mode_m = parts[2]
                min_goal_cost = float('inf')

                # Option 1: Image already taken by some rover
                rovers_with_image = [r for r, o, m_ in have_image if o == objective_o and m_ == mode_m]
                if rovers_with_image:
                    for r in rovers_with_image:
                        r_loc = rover_locations.get(r)
                        if r_loc:
                            dist_to_comm = get_dist(r_loc, self.comm_waypoints)
                            if dist_to_comm != float('inf'):
                                cost = dist_to_comm + 1 # navigate + communicate
                                min_goal_cost = min(min_goal_cost, cost)

                # Option 2: Need to take image and communicate
                if min_goal_cost == float('inf'):
                    equipped_rovers = [r for r, caps in self.rover_capabilities.items() if caps['imaging']]
                    for r in equipped_rovers:
                        r_loc = rover_locations.get(r)
                        if r_loc:
                            suitable_cameras = [
                                cam for cam in self.rover_cameras.get(r, [])
                                if mode_m in self.camera_modes.get(cam, [])
                            ]
                            for camera_i in suitable_cameras:
                                image_waypoints = set(self.objective_visible_from.get(objective_o, []))
                                if not image_waypoints: continue # Cannot image this objective from anywhere

                                cal_target = self.camera_calibration_targets.get(camera_i)
                                cal_waypoints = set(self.objective_visible_from.get(cal_target, [])) if cal_target else set()
                                # If camera needs calibration but has no target or target not visible from anywhere, cannot calibrate
                                if cal_target and not cal_waypoints: continue

                                min_cost_for_rover_camera = float('inf')

                                # Case A: Camera is currently calibrated
                                if (camera_i, r) in camera_calibrated:
                                     for p in image_waypoints:
                                         dist_r_to_p = get_dist(r_loc, {p})
                                         if dist_r_to_p != float('inf'):
                                             cost_image = dist_r_to_p + 1 # navigate + take image
                                             dist_p_to_comm = get_dist(p, self.comm_waypoints)
                                             if dist_p_to_comm != float('inf'):
                                                 cost_comm = dist_p_to_comm + 1 # navigate + communicate
                                                 min_cost_for_rover_camera = min(min_cost_for_rover_camera, cost_image + cost_comm)

                                # Case B: Camera is not calibrated (or needs recalibration)
                                # If not calibrated, we must calibrate first.
                                if (camera_i, r) not in camera_calibrated and cal_waypoints:
                                     for w in cal_waypoints:
                                         for p in image_waypoints:
                                             dist_r_to_w = get_dist(r_loc, {w})
                                             if dist_r_to_w != float('inf'):
                                                 cost_calibrate = dist_r_to_w + 1 # navigate + calibrate
                                                 dist_w_to_p = get_dist(w, {p})
                                                 if dist_w_to_p != float('inf'):
                                                     cost_image = dist_w_to_p + 1 # navigate + take image
                                                     dist_p_to_comm = get_dist(p, self.comm_waypoints)
                                                     if dist_p_to_comm != float('inf'):
                                                         cost_comm = dist_p_to_comm + 1 # navigate + communicate
                                                         min_cost_for_rover_camera = min(min_cost_for_rover_camera, cost_calibrate + cost_image + cost_comm)
                                elif (camera_i, r) not in camera_calibrated and not cal_waypoints:
                                    # Cannot calibrate this camera
                                    pass # min_cost_for_rover_camera remains infinity


                                if min_cost_for_rover_camera != float('inf'):
                                    min_goal_cost = min(min_goal_cost, min_cost_for_rover_camera)


                # Add cost for this goal
                if min_goal_cost == float('inf'):
                    total_cost += LARGE_PENALTY # Cannot achieve this goal via known methods
                else:
                    total_cost += min_goal_cost

            elif predicate == "at":
                 # (at ?r - rover ?w - waypoint)
                 rover_r = parts[1]
                 waypoint_w = parts[2]
                 r_loc = rover_locations.get(rover_r)
                 # Only calculate cost if rover exists and is not already at the target waypoint
                 if r_loc and r_loc != waypoint_w:
                     dist = get_dist(r_loc, {waypoint_w})
                     if dist == float('inf'):
                         total_cost += LARGE_PENALTY # Cannot reach target waypoint
                     else:
                         total_cost += dist # navigate actions
                 # If r_loc is None, the rover might not exist or not be at a waypoint.
                 # Assuming rovers are always at waypoints and exist if they are in goals/static facts.
                 # If r_loc == waypoint_w, cost is 0, which is handled by the outer 'if goal in state' check.


            # Add other goal types if necessary (based on domain goals, these three seem primary)
            # If there are other goal types, they would need specific cost estimation logic here.
            # Assuming only 'communicated_...' and 'at' goals based on typical rovers problems.


        # If any penalty was added, return a large value.
        # This helps greedy search avoid paths leading to seemingly impossible goals.
        if total_cost >= LARGE_PENALTY:
             return LARGE_PENALTY # Or a larger value if needed

        return total_cost
