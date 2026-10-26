import collections
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic # Assuming this base class exists

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle empty fact string case defensively
    if not fact or fact.strip() == '()':
        return []
    # Remove outer parentheses and split by spaces
    return fact.strip()[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Check if the number of parts matches the number of args, unless args is empty
    if args and len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions and travel steps required
    to achieve each uncommunicated goal fact. It sums the minimum estimated
    cost for each goal independently, considering the necessary steps
    (calibration, sample collection, imaging, movement to required locations,
    and communication) and selecting the best available rover for each goal.
    Travel cost is estimated using shortest path distances between waypoints.

    # Assumptions
    - All `can_traverse` facts define a single undirected graph for all rovers.
    - Stores are effectively infinite or can be emptied without significant cost
      or complex state tracking for heuristic purposes (i.e., the 'empty'
      precondition for sample taking is ignored or assumed met).
    - Any data in a full store is relevant to a goal or can be ignored for
      heuristic calculation simplicity.
    - All necessary objects (rovers, cameras, landers, waypoints, samples,
      objectives) and static relationships required to achieve the goals
      exist and are correctly defined in the problem instance.
    - The heuristic does not track the state of individual rovers precisely
      across multiple steps for a single goal; it sums estimated costs for
      sequential abstract steps (e.g., move to calibrate, calibrate, move
      to image, image, move to communicate, communicate).

    # Heuristic Initialization
    - Extracts all static facts to build data structures:
        - Waypoint graph from `visible` and `can_traverse` facts.
        - Shortest path distances between all waypoints using BFS.
        - Lander locations.
        - Waypoints with soil/rock samples.
        - Objective visibility from waypoints.
        - Rover capabilities (`equipped_for_...`).
        - Camera information (`on_board`, `supports`, `calibration_target`).
    - Stores the set of goal facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize total heuristic cost to 0.
    2. Identify all goal facts that are not yet true in the current state.
    3. For each unachieved goal fact:
        a. Initialize the minimum cost for this goal (`cost_G`) to infinity.
        b. If the goal is `(communicated_image_data ?o ?m)`:
            i. Find all (rover `?r`, camera `?c`) pairs such that `?c` is on board `?r`, `?c` supports mode `?m`, and `?c` is a calibration target for objective `?o`.
            ii. For each suitable (rover `?r`, camera `?c`) pair:
                - Get the current location `?w_curr` of rover `?r`.
                - Calculate the estimated cost for this rover/camera to achieve this goal:
                    - Initialize step cost = 0, path cost = 0.
                    - `current_waypoint` = `?w_curr`.
                    - If `(calibrated ?c ?r)` is not in the state:
                        - Find the waypoint `?w_cal` visible from `?o` that is closest to `current_waypoint`.
                        - Add `dist(current_waypoint, ?w_cal)` to path cost.
                        - Add 1 (for `calibrate` action) to step cost.
                        - Update `current_waypoint` to `?w_cal`.
                    - If `(have_image ?r ?o ?m)` is not in the state:
                        - Find the waypoint `?w_img` visible from `?o` that is closest to `current_waypoint`.
                        - Add `dist(current_waypoint, ?w_img)` to path cost.
                        - Add 1 (for `take_image` action) to step cost.
                        - Update `current_waypoint` to `?w_img`.
                    - Find the lander waypoint `?w_comm` closest to `current_waypoint`.
                    - Add `dist(current_waypoint, ?w_comm)` to path cost.
                    - Add 1 (for `communicate_image_data` action) to step cost.
                    - The estimated cost for this rover/camera is `path_cost + step_cost`.
                - Update `cost_G = min(cost_G, estimated_cost_for_this_rover)`.
        c. If the goal is `(communicated_soil_data ?w_soil)`:
            i. Find all rovers `?r` equipped for soil analysis.
            ii. For each suitable rover `?r`:
                - Get the current location `?w_curr` of rover `?r`.
                - Calculate the estimated cost for this rover:
                    - Initialize step cost = 0, path cost = 0.
                    - `current_waypoint` = `?w_curr`.
                    - If `(have_soil_analysis ?r ?w_soil)` is not in the state:
                        - Add `dist(current_waypoint, ?w_soil)` to path cost.
                        - Add 1 (for `take_soil_sample` action) to step cost.
                        - Update `current_waypoint` to `?w_soil`.
                    - Find the lander waypoint `?w_comm` closest to `current_waypoint`.
                    - Add `dist(current_waypoint, ?w_comm)` to path cost.
                    - Add 1 (for `communicate_soil_data` action) to step cost.
                    - The estimated cost for this rover is `path_cost + step_cost`.
                - Update `cost_G = min(cost_G, estimated_cost_for_this_rover)`.
        d. If the goal is `(communicated_rock_data ?w_rock)`:
            i. Find all rovers `?r` equipped for rock analysis.
            ii. For each suitable rover `?r`:
                - Get the current location `?w_curr` of rover `?r`.
                - Calculate the estimated cost for this rover:
                    - Initialize step cost = 0, path cost = 0.
                    - `current_waypoint` = `?w_curr`.
                    - If `(have_rock_analysis ?r ?w_rock)` is not in the state:
                        - Add `dist(current_waypoint, ?w_rock)` to path cost.
                        - Add 1 (for `take_rock_sample` action) to step cost.
                        - Update `current_waypoint` to `?w_rock`.
                    - Find the lander waypoint `?w_comm` closest to `current_waypoint`.
                    - Add `dist(current_waypoint, ?w_comm)` to path cost.
                    - Add 1 (for `communicate_rock_data` action) to step cost.
                    - The estimated cost for this rover is `path_cost + step_cost`.
                - Update `cost_G = min(cost_G, estimated_cost_for_this_rover)`.
        e. Add `cost_G` to the total heuristic cost.
    4. Return the total heuristic cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # Data structures to store static information
        self.lander_waypoints = set()
        self.objective_visible_from = collections.defaultdict(set)
        self.waypoint_graph = collections.defaultdict(set)
        self.rover_capabilities = collections.defaultdict(lambda: {'imaging': False, 'soil': False, 'rock': False})
        self.rover_cameras = collections.defaultdict(set) # rover -> set(camera)
        self.camera_info = collections.defaultdict(lambda: {'supports': set(), 'calibration_target': None}) # camera -> {supports: set(modes), calibration_target: objective}
        self.soil_sample_waypoints = set()
        self.rock_sample_waypoints = set()
        self.all_waypoints = set() # Collect all waypoints to build graph

        # Parse static facts
        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue # Skip empty facts

            predicate = parts[0]

            if predicate == "at_lander":
                lander, waypoint = parts[1], parts[2]
                self.lander_waypoints.add(waypoint)
                self.all_waypoints.add(waypoint)
            elif predicate == "equipped_for_imaging":
                rover = parts[1]
                self.rover_capabilities[rover]['imaging'] = True
            elif predicate == "equipped_for_soil_analysis":
                rover = parts[1]
                self.rover_capabilities[rover]['soil'] = True
            elif predicate == "equipped_for_rock_analysis":
                rover = parts[1]
                self.rover_capabilities[rover]['rock'] = True
            # store_of is not strictly needed for this simplified heuristic
            # elif predicate == "store_of":
            #     store, rover = parts[1], parts[2]
            #     self.rover_stores[rover] = store
            elif predicate == "visible":
                w1, w2 = parts[1], parts[2]
                self.waypoint_graph[w1].add(w2)
                self.waypoint_graph[w2].add(w1) # Visible is bidirectional
                self.all_waypoints.add(w1)
                self.all_waypoints.add(w2)
            elif predicate == "can_traverse":
                rover, w1, w2 = parts[1], parts[2], parts[3]
                # Assuming can_traverse defines the general graph for all rovers
                self.waypoint_graph[w1].add(w2)
                self.waypoint_graph[w2].add(w1) # Assuming traverse is bidirectional if visible is
                self.all_waypoints.add(w1)
                self.all_waypoints.add(w2)
            elif predicate == "calibration_target":
                camera, objective = parts[1], parts[2]
                self.camera_info[camera]['calibration_target'] = objective
            elif predicate == "on_board":
                camera, rover = parts[1], parts[2]
                self.rover_cameras[rover].add(camera)
            elif predicate == "supports":
                camera, mode = parts[1], parts[2]
                self.camera_info[camera]['supports'].add(mode)
            elif predicate == "visible_from":
                objective, waypoint = parts[1], parts[2]
                self.objective_visible_from[objective].add(waypoint)
                self.all_waypoints.add(waypoint)
            elif predicate == "at_rock_sample":
                waypoint = parts[1]
                self.rock_sample_waypoints.add(waypoint)
                self.all_waypoints.add(waypoint)
            elif predicate == "at_soil_sample":
                waypoint = parts[1]
                self.soil_sample_waypoints.add(waypoint)
                self.all_waypoints.add(waypoint)

        # Add waypoints from initial state if any were missed (e.g., rover start loc)
        for fact in task.initial_state:
             if match(fact, "at", "*", "*"):
                 self.all_waypoints.add(get_parts(fact)[2])

        # Compute shortest paths between all waypoints
        self.shortest_paths = self.compute_shortest_paths()

    def build_graph(self):
        """Builds the waypoint graph from static facts."""
        # This is already done during parsing in __init__
        pass

    def compute_shortest_paths(self):
        """Computes all-pairs shortest paths using BFS."""
        distances = {}
        for start_node in self.all_waypoints:
            q = collections.deque([(start_node, 0)])
            visited = {start_node}
            distances[(start_node, start_node)] = 0

            while q:
                current_node, dist = q.popleft()

                for neighbor in self.waypoint_graph.get(current_node, set()):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        distances[(start_node, neighbor)] = dist + 1
                        q.append((neighbor, dist + 1))
        return distances

    def get_distance(self, w1, w2):
        """Returns the shortest distance between two waypoints."""
        if w1 is None or w2 is None:
             return float('inf') # Cannot travel from/to None
        return self.shortest_paths.get((w1, w2), float('inf'))

    def get_rover_location(self, state, rover):
        """Finds the current waypoint of a given rover in the state."""
        for fact in state:
            if match(fact, "at", rover, "*"):
                return get_parts(fact)[2]
        return None # Rover location not found (should not happen in valid states)

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        total_heuristic = 0

        # Convert state to a set for faster lookups
        state_set = set(state)

        # Find current rover locations
        rover_locations = {}
        for fact in state_set:
            if match(fact, "at", "*", "*"):
                rover = get_parts(fact)[1]
                location = get_parts(fact)[2]
                rover_locations[rover] = location

        # Find current rover data/calibration status
        rover_soil_data = collections.defaultdict(set) # rover -> set(waypoint)
        rover_rock_data = collections.defaultdict(set) # rover -> set(waypoint)
        rover_image_data = collections.defaultdict(set) # rover -> set((objective, mode))
        rover_calibrated_cameras = collections.defaultdict(set) # rover -> set(camera)

        for fact in state_set:
            parts = get_parts(fact)
            if not parts: continue
            predicate = parts[0]
            if predicate == "have_soil_analysis":
                rover, waypoint = parts[1], parts[2]
                rover_soil_data[rover].add(waypoint)
            elif predicate == "have_rock_analysis":
                rover, waypoint = parts[1], parts[2]
                rover_rock_data[rover].add(waypoint)
            elif predicate == "have_image":
                rover, objective, mode = parts[1], parts[2], parts[3]
                rover_image_data[rover].add((objective, mode))
            elif predicate == "calibrated":
                 camera, rover = parts[1], parts[2]
                 rover_calibrated_cameras[rover].add(camera)

        # Iterate through unachieved goals
        for goal in self.goals:
            if goal in state_set:
                continue # Goal already achieved

            parts = get_parts(goal)
            if not parts: continue
            predicate = parts[0]

            cost_G = float('inf') # Minimum cost for this specific goal

            if predicate == "communicated_image_data":
                objective, mode = parts[1], parts[2]
                
                # Find suitable rover/camera pairs for this goal
                suitable_rover_camera_pairs = []
                for rover, cameras in self.rover_cameras.items():
                    if not self.rover_capabilities[rover]['imaging']: continue # Rover must be equipped
                    for camera in cameras:
                        cam_info = self.camera_info.get(camera)
                        # Check if camera exists, supports mode, and calibrates for the target objective
                        if cam_info and mode in cam_info['supports'] and cam_info['calibration_target'] == objective:
                             suitable_rover_camera_pairs.append((rover, camera))

                if not suitable_rover_camera_pairs:
                    # This goal is impossible if no suitable equipment exists.
                    # Return inf if any goal is impossible.
                    return float('inf')

                # Waypoints visible from the objective
                W_o = self.objective_visible_from.get(objective, set())
                if not W_o:
                     # Objective not visible from anywhere - impossible goal?
                     return float('inf')

                # Lander waypoints
                W_lander = self.lander_waypoints
                if not W_lander:
                     # No lander - impossible goal?
                     return float('inf')

                for rover, camera in suitable_rover_camera_pairs:
                    rover_loc = rover_locations.get(rover)
                    if rover_loc is None: continue # Should not happen

                    current_waypoint = rover_loc
                    step_cost = 0
                    path_cost = 0

                    # Step 1: Calibrate (if needed)
                    if camera not in rover_calibrated_cameras[rover]:
                        min_dist_to_cal = float('inf')
                        best_w_cal = None
                        for w_cal in W_o:
                            dist = self.get_distance(current_waypoint, w_cal)
                            if dist < min_dist_to_cal:
                                min_dist_to_cal = dist
                                best_w_cal = w_cal

                        if best_w_cal is not None and min_dist_to_cal != float('inf'):
                            path_cost += min_dist_to_cal
                            step_cost += 1 # calibrate action
                            current_waypoint = best_w_cal # Rover is conceptually here now
                        else:
                            # Cannot reach any calibration waypoint from current location
                            continue # This rover/camera cannot achieve this goal

                    # Step 2: Take image (if needed)
                    if (objective, mode) not in rover_image_data[rover]:
                        min_dist_to_img = float('inf')
                        best_w_img = None
                        for w_img in W_o:
                            dist = self.get_distance(current_waypoint, w_img)
                            if dist < min_dist_to_img:
                                min_dist_to_img = dist
                                best_w_img = w_img

                        if best_w_img is not None and min_dist_to_img != float('inf'):
                            path_cost += min_dist_to_img
                            step_cost += 1 # take_image action
                            current_waypoint = best_w_img # Rover is conceptually here now
                        else:
                             # Cannot reach any image waypoint from current location (or cal spot)
                            continue # This rover/camera cannot achieve this goal

                    # Step 3: Communicate
                    min_dist_to_comm = float('inf')
                    best_w_comm = None
                    for w_comm in W_lander:
                        dist = self.get_distance(current_waypoint, w_comm)
                        if dist < min_dist_to_comm:
                            min_dist_to_comm = dist
                            best_w_comm = w_comm

                    if best_w_comm is not None and min_dist_to_comm != float('inf'):
                        path_cost += min_dist_to_comm
                        step_cost += 1 # communicate action
                        # current_waypoint = best_w_comm # Rover is conceptually here now
                    else:
                        # Cannot reach any lander waypoint from current location (or img spot)
                        continue # This rover/camera cannot achieve this goal

                    # Update minimum cost for this goal
                    cost_G = min(cost_G, path_cost + step_cost)

            elif predicate == "communicated_soil_data":
                waypoint_soil = parts[1]

                # Find suitable rovers
                suitable_rovers = [r for r, caps in self.rover_capabilities.items() if caps['soil']]

                if not suitable_rovers:
                    return float('inf')

                # Lander waypoints
                W_lander = self.lander_waypoints
                if not W_lander:
                     return float('inf')

                # Check if the soil sample exists at the waypoint
                if waypoint_soil not in self.soil_sample_waypoints:
                     return float('inf') # Cannot take sample if none exists

                for rover in suitable_rovers:
                    rover_loc = rover_locations.get(rover)
                    if rover_loc is None: continue # Should not happen

                    current_waypoint = rover_loc
                    step_cost = 0
                    path_cost = 0

                    # Step 1: Take sample (if needed)
                    if waypoint_soil not in rover_soil_data[rover]:
                        # Need to take sample. Travel to waypoint_soil.
                        dist_to_sample = self.get_distance(current_waypoint, waypoint_soil)
                        if dist_to_sample != float('inf'):
                            path_cost += dist_to_sample
                            step_cost += 1 # take_soil_sample action
                            current_waypoint = waypoint_soil # Rover is conceptually here now
                        else:
                            # Cannot reach sample waypoint
                            continue # This rover cannot achieve this goal

                    # Step 2: Communicate
                    min_dist_to_comm = float('inf')
                    best_w_comm = None
                    for w_comm in W_lander:
                        dist = self.get_distance(current_waypoint, w_comm)
                        if dist < min_dist_to_comm:
                            min_dist_to_comm = dist
                            best_w_comm = w_comm

                    if best_w_comm is not None and min_dist_to_comm != float('inf'):
                        path_cost += min_dist_to_comm
                        step_cost += 1 # communicate action
                        # current_waypoint = best_w_comm # Rover is conceptually here now
                    else:
                        # Cannot reach lander
                        continue # This rover cannot achieve this goal

                    # Update minimum cost for this goal
                    cost_G = min(cost_G, path_cost + step_cost)


            elif predicate == "communicated_rock_data":
                waypoint_rock = parts[1]

                # Find suitable rovers
                suitable_rovers = [r for r, caps in self.rover_capabilities.items() if caps['rock']]

                if not suitable_rovers:
                    return float('inf')

                # Lander waypoints
                W_lander = self.lander_waypoints
                if not W_lander:
                     return float('inf')

                # Check if the rock sample exists at the waypoint
                if waypoint_rock not in self.rock_sample_waypoints:
                     return float('inf') # Cannot take sample if none exists

                for rover in suitable_rovers:
                    rover_loc = rover_locations.get(rover)
                    if rover_loc is None: continue # Should not happen

                    current_waypoint = rover_loc
                    step_cost = 0
                    path_cost = 0

                    # Step 1: Take sample (if needed)
                    if waypoint_rock not in rover_rock_data[rover]:
                        # Need to take sample. Travel to waypoint_rock.
                        dist_to_sample = self.get_distance(current_waypoint, waypoint_rock)
                        if dist_to_sample != float('inf'):
                            path_cost += dist_to_sample
                            step_cost += 1 # take_rock_sample action
                            current_waypoint = waypoint_rock # Rover is conceptually here now
                        else:
                            # Cannot reach sample waypoint
                            continue # This rover cannot achieve this goal

                    # Step 2: Communicate
                    min_dist_to_comm = float('inf')
                    best_w_comm = None
                    for w_comm in W_lander:
                        dist = self.get_distance(current_waypoint, w_comm)
                        if dist < min_dist_to_comm:
                            min_dist_to_comm = dist
                            best_w_comm = w_comm

                    if best_w_comm is not None and min_dist_to_comm != float('inf'):
                        path_cost += min_dist_to_comm
                        step_cost += 1 # communicate action
                        # current_waypoint = best_w_comm # Rover is conceptually here now
                    else:
                        # Cannot reach lander
                        continue # This rover cannot achieve this goal

                    # Update minimum cost for this goal
                    cost_G = min(cost_G, path_cost + step_cost)

            # If cost_G is still infinity, it means this goal is impossible
            # for any suitable rover. Return infinity for the total heuristic.
            if cost_G == float('inf'):
                 return float('inf')

            # Add the minimum cost for this goal to the total heuristic
            total_heuristic += cost_G

        return total_heuristic
