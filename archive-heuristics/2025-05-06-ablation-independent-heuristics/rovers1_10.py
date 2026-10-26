import collections
import heapq # Used for BFS, although deque is also fine

from heuristics.heuristic_base import Heuristic
# Assuming Task and Operator classes are available in the environment
# from task import Operator, Task


def parse_fact(fact_string):
    """
    Helper function to parse a PDDL fact string into a tuple.
    e.g., '(at rover1 waypoint1)' -> ('at', 'rover1', 'waypoint1')
    """
    # Remove surrounding parentheses and split by space
    parts = fact_string[1:-1].split()
    return tuple(parts)

def bfs(graph, start_node):
    """
    Performs Breadth-First Search to find shortest distances from a start node
    in an unweighted graph.
    """
    distances = {node: float('inf') for node in graph}
    if start_node in graph:
        distances[start_node] = 0
        queue = collections.deque([start_node])
        while queue:
            current_node = queue.popleft()
            for neighbor in graph.get(current_node, []):
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)
    return distances

def compute_all_pairs_shortest_paths(graph):
    """
    Computes shortest distances between all pairs of nodes in a graph
    using BFS starting from each node.
    """
    all_distances = {}
    for start_node in graph:
        all_distances[start_node] = bfs(graph, start_node)
    return all_distances


class roversHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Rovers domain.

    Summary:
    This heuristic estimates the cost to reach a goal state by summing up
    the estimated costs for each unsatisfied goal fact. For each unsatisfied
    goal, it calculates the minimum number of actions required to achieve
    that specific goal fact, considering different rovers and action sequences,
    and adds this minimum cost to the total heuristic value. Navigation costs
    are estimated using precomputed shortest paths on the waypoint graph for
    each rover.

    Assumptions:
    - The heuristic is non-admissible. It aims to guide a greedy best-first
      search and prioritizes minimizing expanded nodes over guaranteeing
      optimality.
    - Delete effects are largely ignored in the cost estimation (e.g., a
      sampled location is assumed to remain sampled for the purpose of
      calculating communication cost from that location, camera uncalibration
      after taking an image is handled by potentially adding a calibration
      cost if needed for the *next* image goal, but not strictly enforced
      within a single image goal sequence).
    - Store capacity is simplified: for sampling, it's assumed a rover can
      always find or make a store empty (e.g., by dropping, although drop
      cost is ignored).
    - Goals are treated independently: the cost for each unsatisfied goal
      is calculated assuming it can be achieved by the most suitable available
      rover/resources, without considering conflicts or resource sharing
      between goals.
    - Connectivity is assumed where paths are calculated. Unreachable parts
      of the graph result in infinite costs, which are handled by assigning
      a large penalty to the goal if no finite path is found.
    - The heuristic value is 0 if and only if the state is a goal state.

    Heuristic Initialization:
    In the constructor (__init__), the heuristic pre-processes the static
    information from the task description to build necessary data structures:
    1.  Identifies all objects by type (rovers, waypoints, landers, etc.)
        by parsing all possible facts defined in the task.
    2.  Parses static facts to extract relationships like lander location,
        waypoint visibility, rover equipment, store ownership, camera
        properties (on-board, supported modes, calibration target),
        objective visibility, and initial sample locations.
    3.  Constructs a navigation graph for each rover based on `can_traverse`
        and `visible` static facts. An edge exists from waypoint A to B
        for rover R if `(can_traverse R A B)` and `(visible A B)` are true.
    4.  Computes all-pairs shortest paths for each rover on its navigation
        graph using BFS. These distances are stored for quick lookup.
    5.  Identifies waypoints visible from the lander location, as these are
        potential communication points.
    6.  Stores the set of goal facts.

    Step-By-Step Thinking for Computing Heuristic (__call__):
    For a given state (node):
    1.  Extract dynamic information from the state: current rover locations,
        store statuses, which rovers have soil/rock data or images, which
        cameras are calibrated, and current locations of soil/rock samples.
    2.  Initialize the total heuristic value `h` to 0.
    3.  Iterate through each goal fact defined in the task.
    4.  If a goal fact is already satisfied in the current state, skip it.
    5.  If a goal fact is not satisfied, calculate the minimum estimated cost
        to achieve it:
        a.  **For `(communicated_soil_data W)`:**
            - Check if any rover already has `(have_soil_analysis R W)`. If yes, the cost is the minimum navigation cost for such a rover from its current location to any communication waypoint, plus 1 (for the communicate action).
            - If no rover has the data, check if `(at_soil_sample W)` is true in the current state. If yes, the cost is the minimum navigation cost for a soil-equipped rover from its current location to W, plus 1 (for sample_soil), plus the minimum navigation cost from W to any communication waypoint, plus 1 (for communicate). Store full cost is ignored.
            - The minimum cost for this goal is the minimum of the above options (communicate existing data vs. sample+communicate).
        b.  **For `(communicated_rock_data W)`:** Similar logic as soil data, using rock-specific predicates and equipment.
        c.  **For `(communicated_image_data O M)`:**
            - Check if any rover already has `(have_image R O M)`. If yes, the cost is the minimum navigation cost for such a rover from its current location to any communication waypoint, plus 1 (for the communicate action).
            - If no rover has the image, iterate through suitable combinations of imaging-equipped rovers `R`, cameras `I` on board `R` supporting mode `M`, and waypoints `P` visible from objective `O`. For each combination:
                - Calculate the cost:
                    - If camera `I` is not calibrated for rover `R`, add the minimum navigation cost for `R` from its current location to any waypoint visible from `I`'s calibration target, plus 1 (for calibrate). Update the rover's effective location to the chosen calibration waypoint.
                    - Add the navigation cost for `R` from its current (potentially updated) location to `P`, plus 1 (for take_image). Update the rover's effective location to `P`.
                    - Add the minimum navigation cost for `R` from `P` to any communication waypoint, plus 1 (for communicate).
                - The cost for this combination is the sum of navigation and action costs.
            - The minimum cost for this goal is the minimum cost found across all suitable combinations.
        d.  If the minimum cost calculated for a goal is infinity (meaning no path or required condition exists in this relaxation), add a large penalty (e.g., 1000) to `h`. Otherwise, add the calculated minimum cost to `h`.
    6.  Return the total heuristic value `h`.
    """

    def __init__(self, task):
        super().__init__()
        self.goals = task.goals

        # --- Pre-processing Static Information ---

        # 1. Identify all objects by type
        self.rovers = set()
        self.waypoints = set()
        self.landers = set()
        self.stores = set()
        self.cameras = set()
        self.modes = set()
        self.objectives = set()

        # Mapping predicate name to tuple of object types (based on domain definition)
        predicate_signatures = {
            'at': ('rover', 'waypoint'),
            'at_lander': ('lander', 'waypoint'),
            'can_traverse': ('rover', 'waypoint', 'waypoint'),
            'equipped_for_soil_analysis': ('rover',),
            'equipped_for_rock_analysis': ('rover',),
            'equipped_for_imaging': ('rover',),
            'empty': ('store',),
            'have_rock_analysis': ('rover', 'waypoint'),
            'have_soil_analysis': ('rover', 'waypoint'),
            'full': ('store',),
            'calibrated': ('camera', 'rover'),
            'supports': ('camera', 'mode'),
            'visible': ('waypoint', 'waypoint'),
            'have_image': ('rover', 'objective', 'mode'),
            'communicated_soil_data': ('waypoint',),
            'communicated_rock_data': ('waypoint',),
            'communicated_image_data': ('objective', 'mode'),
            'at_soil_sample': ('waypoint',),
            'at_rock_sample': ('waypoint',),
            'visible_from': ('objective', 'waypoint'),
            'store_of': ('store', 'rover'),
            'calibration_target': ('camera', 'objective'),
            'on_board': ('camera', 'rover'),
        }

        type_sets = {
            'rover': self.rovers,
            'waypoint': self.waypoints,
            'lander': self.landers,
            'store': self.stores,
            'camera': self.cameras,
            'mode': self.modes,
            'objective': self.objectives,
        }

        # Collect all fact strings from initial state, goals, static, and operators
        all_fact_strings = set(task.initial_state) | set(task.goals) | set(task.static)
        for op in task.operators:
             all_fact_strings |= set(op.preconditions) | set(op.add_effects) | set(op.del_effects)

        # Populate object sets based on predicate signatures
        for fact_string in all_fact_strings:
            parsed = parse_fact(fact_string)
            predicate = parsed[0]
            args = parsed[1:]
            if predicate in predicate_signatures:
                types = predicate_signatures[predicate]
                # Basic check for arity match
                if len(args) == len(types):
                     for i, obj_name in enumerate(args):
                         obj_type = types[i]
                         if obj_type in type_sets:
                             type_sets[obj_type].add(obj_name)
                # Handle predicates with multiple args of the same type like visible/can_traverse
                elif predicate in ['visible'] and len(args) == 2 and types == ('waypoint', 'waypoint'):
                     type_sets['waypoint'].add(args[0])
                     type_sets['waypoint'].add(args[1])
                elif predicate == 'can_traverse' and len(args) == 3 and types == ('rover', 'waypoint', 'waypoint'):
                     type_sets['rover'].add(args[0])
                     type_sets['waypoint'].add(args[1])
                     type_sets['waypoint'].add(args[2])


        # 2. Parse static facts and build static data structures
        self.lander_location = None
        self.visible_wps = collections.defaultdict(set) # waypoint -> set(visible_waypoints)
        self.rover_equipped_soil = set()
        self.rover_equipped_rock = set()
        self.rover_equipped_imaging = set()
        self.rover_stores = {} # rover -> store
        self.camera_on_board_rover = {} # camera -> rover
        self.camera_supports_mode = collections.defaultdict(set) # camera -> set(modes)
        self.camera_calibration_target = {} # camera -> objective
        self.objective_visible_from = collections.defaultdict(set) # objective -> set(waypoints)

        # Temporary structure to hold can_traverse facts
        can_traverse_facts = collections.defaultdict(list) # rover -> list((wp1, wp2))

        for fact_string in task.static:
            parsed = parse_fact(fact_string)
            predicate = parsed[0]
            if predicate == 'at_lander':
                # Assuming only one lander and its location is static
                if len(parsed) == 3 and parsed[1] in self.landers and parsed[2] in self.waypoints:
                    self.lander_location = parsed[2]
            elif predicate == 'can_traverse':
                 if len(parsed) == 4 and parsed[1] in self.rovers and parsed[2] in self.waypoints and parsed[3] in self.waypoints:
                    rover, wp1, wp2 = parsed[1:]
                    can_traverse_facts[rover].append((wp1, wp2))
            elif predicate == 'visible':
                 if len(parsed) == 3 and parsed[1] in self.waypoints and parsed[2] in self.waypoints:
                    wp1, wp2 = parsed[1:]
                    self.visible_wps[wp1].add(wp2)
            elif predicate == 'equipped_for_soil_analysis':
                 if len(parsed) == 2 and parsed[1] in self.rovers:
                    self.rover_equipped_soil.add(parsed[1])
            elif predicate == 'equipped_for_rock_analysis':
                 if len(parsed) == 2 and parsed[1] in self.rovers:
                    self.rover_equipped_rock.add(parsed[1])
            elif predicate == 'equipped_for_imaging':
                 if len(parsed) == 2 and parsed[1] in self.rovers:
                    self.rover_equipped_imaging.add(parsed[1])
            elif predicate == 'store_of':
                 if len(parsed) == 3 and parsed[1] in self.stores and parsed[2] in self.rovers:
                    store, rover = parsed[1:]
                    self.rover_stores[rover] = store # Assuming one store per rover
            elif predicate == 'on_board':
                 if len(parsed) == 3 and parsed[1] in self.cameras and parsed[2] in self.rovers:
                    camera, rover = parsed[1:]
                    self.camera_on_board_rover[camera] = rover # Assuming camera is on only one rover
            elif predicate == 'supports':
                 if len(parsed) == 3 and parsed[1] in self.cameras and parsed[2] in self.modes:
                    camera, mode = parsed[1:]
                    self.camera_supports_mode[camera].add(mode)
            elif predicate == 'calibration_target':
                 if len(parsed) == 3 and parsed[1] in self.cameras and parsed[2] in self.objectives:
                    camera, objective = parsed[1:]
                    self.camera_calibration_target[camera] = objective # Assuming camera has one target
            elif predicate == 'visible_from':
                 if len(parsed) == 3 and parsed[1] in self.objectives and parsed[2] in self.waypoints:
                    objective, waypoint = parsed[1:]
                    self.objective_visible_from[objective].add(waypoint)

        # 3. Build rover navigation graphs considering both can_traverse and visible
        self.rover_nav_graphs = collections.defaultdict(lambda: collections.defaultdict(list))
        for rover, edges in can_traverse_facts.items():
            for wp1, wp2 in edges:
                # An edge exists if the rover can traverse AND the destination is visible from the source
                if wp2 in self.visible_wps.get(wp1, set()):
                    self.rover_nav_graphs[rover][wp1].append(wp2)
            # Ensure all waypoints the rover can potentially be at are in its graph dict, even if no edges
            for wp in self.waypoints:
                 if wp not in self.rover_nav_graphs[rover]:
                      self.rover_nav_graphs[rover][wp] = []


        # 4. Compute all-pairs shortest paths for each rover
        self.rover_distances = {}
        for rover in self.rovers:
            self.rover_distances[rover] = compute_all_pairs_shortest_paths(self.rover_nav_graphs.get(rover, {}))

        # 5. Identify communication waypoints (visible from lander)
        self.comm_waypoint_options = self.visible_wps.get(self.lander_location, set())

        # 6. Store goals
        self.goals = task.goals

    def __call__(self, node):
        """
        Computes the domain-dependent heuristic value for the given state.
        """
        h = 0
        state = node.state

        # --- Extract Dynamic State Information ---
        rover_locations = {}
        store_status = {} # store -> 'empty' or 'full'
        rover_soil_data = collections.defaultdict(set) # rover -> set(waypoints)
        rover_rock_data = collections.defaultdict(set) # rover -> set(waypoints)
        rover_images = collections.defaultdict(set) # rover -> set((objective, mode))
        rover_calibrated_cameras = collections.defaultdict(set) # rover -> set(cameras)
        soil_samples_at_wp_current = set() # waypoints
        rock_samples_at_wp_current = set() # waypoints

        for fact_string in state:
            parsed = parse_fact(fact_string)
            predicate = parsed[0]
            if predicate == 'at' and len(parsed) == 3 and parsed[1] in self.rovers and parsed[2] in self.waypoints:
                rover_locations[parsed[1]] = parsed[2]
            elif predicate == 'empty' and len(parsed) == 2 and parsed[1] in self.stores:
                store_status[parsed[1]] = 'empty'
            elif predicate == 'full' and len(parsed) == 2 and parsed[1] in self.stores:
                store_status[parsed[1]] = 'full'
            elif predicate == 'have_soil_analysis' and len(parsed) == 3 and parsed[1] in self.rovers and parsed[2] in self.waypoints:
                rover_soil_data[parsed[1]].add(parsed[2])
            elif predicate == 'have_rock_analysis' and len(parsed) == 3 and parsed[1] in self.rovers and parsed[2] in self.waypoints:
                rover_rock_data[parsed[1]].add(parsed[2])
            elif predicate == 'have_image' and len(parsed) == 4 and parsed[1] in self.rovers and parsed[2] in self.objectives and parsed[3] in self.modes:
                rover_images[parsed[1]].add((parsed[2], parsed[3]))
            elif predicate == 'calibrated' and len(parsed) == 3 and parsed[1] in self.cameras and parsed[2] in self.rovers:
                 rover_calibrated_cameras[parsed[2]].add(parsed[1]) # calibrated camera I for rover R
            elif predicate == 'at_soil_sample' and len(parsed) == 2 and parsed[1] in self.waypoints:
                soil_samples_at_wp_current.add(parsed[1])
            elif predicate == 'at_rock_sample' and len(parsed) == 2 and parsed[1] in self.waypoints:
                rock_samples_at_wp_current.add(parsed[1])

        # Helper to get distance using precomputed paths
        def get_distance(rover, start_wp, end_wp):
            if rover not in self.rover_distances or start_wp not in self.rover_distances[rover]:
                 return float('inf') # Rover might not have any navigation edges
            return self.rover_distances[rover][start_wp].get(end_wp, float('inf'))

        # --- Calculate Cost for Unsatisfied Goals ---
        for goal_string in self.goals:
            if goal_string in state:
                continue # Goal already satisfied

            parsed_goal = parse_fact(goal_string)
            predicate = parsed_goal[0]

            min_goal_cost = float('inf')

            if predicate == 'communicated_soil_data' and len(parsed_goal) == 2:
                wp = parsed_goal[1]
                if wp not in self.waypoints: continue # Invalid goal waypoint

                # Option 1: Communicate existing data
                rovers_with_data = [r for r, wps in rover_soil_data.items() if wp in wps]
                if rovers_with_data:
                    for r_have in rovers_with_data:
                        current_wp = rover_locations.get(r_have)
                        if current_wp:
                            min_comm_nav_cost = float('inf')
                            for comm_wp in self.comm_waypoint_options:
                                dist = get_distance(r_have, current_wp, comm_wp)
                                if dist != float('inf'):
                                    min_comm_nav_cost = min(min_comm_nav_cost, dist)
                            if min_comm_nav_cost != float('inf'):
                                 min_goal_cost = min(min_goal_cost, min_comm_nav_cost + 1) # +1 for communicate action

                # Option 2: Sample and Communicate
                if wp in soil_samples_at_wp_current:
                     equipped_rovers = list(self.rover_equipped_soil)
                     if equipped_rovers:
                         for r_sample in equipped_rovers:
                             current_wp = rover_locations.get(r_sample)
                             if current_wp:
                                 sample_nav_cost = get_distance(r_sample, current_wp, wp)
                                 if sample_nav_cost != float('inf'):
                                     # Assume store is or can be made empty (ignore drop cost)
                                     sample_cost = sample_nav_cost + 1 # +1 for sample_soil action

                                     min_comm_nav_cost = float('inf')
                                     for comm_wp in self.comm_waypoint_options:
                                         dist = get_distance(r_sample, wp, comm_wp) # Nav from sample location
                                         if dist != float('inf'):
                                             min_comm_nav_cost = min(min_comm_nav_cost, dist)

                                     if min_comm_nav_cost != float('inf'):
                                         total_cost = sample_cost + min_comm_nav_cost + 1 # +1 for communicate action
                                         min_goal_cost = min(min_goal_cost, total_cost)

            elif predicate == 'communicated_rock_data' and len(parsed_goal) == 2:
                wp = parsed_goal[1]
                if wp not in self.waypoints: continue # Invalid goal waypoint

                # Option 1: Communicate existing data
                rovers_with_data = [r for r, wps in rover_rock_data.items() if wp in wps]
                if rovers_with_data:
                    for r_have in rovers_with_data:
                        current_wp = rover_locations.get(r_have)
                        if current_wp:
                            min_comm_nav_cost = float('inf')
                            for comm_wp in self.comm_waypoint_options:
                                dist = get_distance(r_have, current_wp, comm_wp)
                                if dist != float('inf'):
                                    min_comm_nav_cost = min(min_comm_nav_cost, dist)
                            if min_comm_nav_cost != float('inf'):
                                 min_goal_cost = min(min_goal_cost, min_comm_nav_cost + 1) # +1 for communicate action

                # Option 2: Sample and Communicate
                if wp in rock_samples_at_wp_current:
                     equipped_rovers = list(self.rover_equipped_rock)
                     if equipped_rovers:
                         for r_sample in equipped_rovers:
                             current_wp = rover_locations.get(r_sample)
                             if current_wp:
                                 sample_nav_cost = get_distance(r_sample, current_wp, wp)
                                 if sample_nav_cost != float('inf'):
                                     # Assume store is or can be made empty (ignore drop cost)
                                     sample_cost = sample_nav_cost + 1 # +1 for sample_rock action

                                     min_comm_nav_cost = float('inf')
                                     for comm_wp in self.comm_waypoint_options:
                                         dist = get_distance(r_sample, wp, comm_wp) # Nav from sample location
                                         if dist != float('inf'):
                                             min_comm_nav_cost = min(min_comm_nav_cost, dist)

                                     if min_comm_nav_cost != float('inf'):
                                         total_cost = sample_cost + min_comm_nav_cost + 1 # +1 for communicate action
                                         min_goal_cost = min(min_goal_cost, total_cost)

            elif predicate == 'communicated_image_data' and len(parsed_goal) == 3:
                obj, mode = parsed_goal[1:]
                if obj not in self.objectives or mode not in self.modes: continue # Invalid goal objective/mode

                # Option 1: Communicate existing image
                rovers_with_image = [r for r, images in rover_images.items() if (obj, mode) in images]
                if rovers_with_image:
                    for r_have in rovers_with_image:
                        current_wp = rover_locations.get(r_have)
                        if current_wp:
                            min_comm_nav_cost = float('inf')
                            for comm_wp in self.comm_waypoint_options:
                                dist = get_distance(r_have, current_wp, comm_wp)
                                if dist != float('inf'):
                                    min_comm_nav_cost = min(min_comm_nav_cost, dist)
                            if min_comm_nav_cost != float('inf'):
                                 min_goal_cost = min(min_goal_cost, min_comm_nav_cost + 1) # +1 for communicate action

                # Option 2: Take image and Communicate
                # Find suitable rovers/cameras/waypoints
                suitable_imaging_options = [] # List of (rover, camera, imaging_wp)
                for r_img in self.rover_equipped_imaging:
                    # Find cameras on this rover that support the mode
                    suitable_cameras = [
                        cam for cam, r_on_board in self.camera_on_board_rover.items()
                        if r_on_board == r_img and mode in self.camera_supports_mode.get(cam, set())
                    ]
                    for cam in suitable_cameras:
                        # Find waypoints visible from the objective
                        imaging_wps = self.objective_visible_from.get(obj, set())
                        for img_wp in imaging_wps:
                            suitable_imaging_options.append((r_img, cam, img_wp))

                if suitable_imaging_options:
                    for r_img, cam, img_wp in suitable_imaging_options:
                        current_wp = rover_locations.get(r_img)
                        if not current_wp: continue # Rover location unknown

                        cost = 0
                        cal_wp_after_nav = current_wp # Start location for imaging nav

                        # Step 1: Calibration (if needed)
                        calibration_needed = cam not in rover_calibrated_cameras.get(r_img, set())

                        if calibration_needed:
                            cal_target = self.camera_calibration_target.get(cam)
                            if not cal_target: continue # Cannot calibrate this camera

                            cal_wps = self.objective_visible_from.get(cal_target, set())
                            if not cal_wps: continue # No waypoint to calibrate from

                            min_cal_nav_cost = float('inf')
                            best_cal_wp = None
                            for cal_wp in cal_wps:
                                dist = get_distance(r_img, current_wp, cal_wp)
                                if dist != float('inf') and dist < min_cal_nav_cost:
                                    min_cal_nav_cost = dist
                                    best_cal_wp = cal_wp

                            if min_cal_nav_cost == float('inf'): continue # Cannot reach any calibration point

                            cost += min_cal_nav_cost + 1 # +1 for calibrate action
                            cal_wp_after_nav = best_cal_wp # Rover is now at the chosen calibration waypoint

                        # Step 2: Imaging
                        img_nav_cost = get_distance(r_img, cal_wp_after_nav, img_wp)
                        if img_nav_cost == float('inf'): continue # Cannot reach imaging point

                        cost += img_nav_cost + 1 # +1 for take_image action

                        # Step 3: Communication
                        min_comm_nav_cost = float('inf')
                        for comm_wp in self.comm_waypoint_options:
                            dist = get_distance(r_img, img_wp, comm_wp) # Nav from imaging location
                            if dist != float('inf'):
                                min_comm_nav_cost = min(min_comm_nav_cost, dist)

                        if min_comm_nav_cost == float('inf'): continue # Cannot reach communication point

                        cost += min_comm_nav_cost + 1 # +1 for communicate action

                        # Update minimum goal cost
                        min_goal_cost = min(min_goal_cost, cost)

            # Add the minimum cost for this goal to the total heuristic
            if min_goal_cost != float('inf'):
                h += min_goal_cost
            else:
                # If a goal is unachievable in this relaxation, add a large penalty
                # This helps distinguish dead ends from states closer to the goal.
                h += 1000 # Penalty value

        # The heuristic is 0 iff all goals are satisfied.
        # If h is 0, it means all goals were found in the state set during the loop.
        # If self.goals <= state is true, h will be 0.
        # If self.goals <= state is false but h is 0, something is wrong with the logic,
        # but the requirement is H=0 only for goal states.
        # We trust the sum logic implies H=0 only when all goals are satisfied.

        return h

