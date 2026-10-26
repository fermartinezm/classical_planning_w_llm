from collections import deque
from fnmatch import fnmatch
# Assuming heuristic_base is in a 'heuristics' directory relative to where this file is used
# from heuristics.heuristic_base import Heuristic

# If the above import fails, you might need to adjust the path or provide a dummy class
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    print("Warning: heuristics.heuristic_base not found. Using a dummy Heuristic class.")
    class Heuristic:
        def __init__(self, task):
            self.goals = task.goals
            self.static = task.static
        def __call__(self, node):
            # A simple relaxed goal count heuristic as a fallback
            state = node.state
            h = 0
            for goal in self.goals:
                # Goal is a string like '(communicated_soil_data waypoint1)'
                # Check if the goal string is present in the state frozenset
                if goal not in state:
                    h += 1
            return h


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact strings or malformed facts gracefully
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at rover1 waypoint1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def shortest_path(graph, start, end):
    """
    Find the shortest path distance between two nodes in a graph using BFS.

    Args:
        graph: Adjacency list representation (dict: node -> list of neighbors).
        start: Starting node.
        end: Target node.

    Returns:
        The shortest distance (number of edges) or float('inf') if no path exists.
    """
    if start == end:
        return 0
    if start not in graph or end not in graph:
        return float('inf')

    queue = deque([(start, 0)])
    visited = {start}
    while queue:
        current_node, distance = queue.popleft()

        # Ensure current_node is a valid key before accessing graph[current_node]
        if current_node not in graph:
             continue # Should not happen if start was in graph and graph is well-formed

        for neighbor in graph[current_node]:
            if neighbor == end:
                return distance + 1
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, distance + 1))
    return float('inf')

class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the total number of actions required to satisfy all
    unmet goal conditions. It calculates the minimum estimated cost for each
    individual goal fact independently and sums these minimums. The cost for
    each goal fact is estimated based on the actions needed to achieve it,
    including navigation (estimated by shortest path distance), sampling,
    imaging (including calibration), and communication.

    # Assumptions
    - Each goal fact can be pursued independently.
    - Navigation cost is the shortest path distance in the rover's traversal graph.
    - A rover needs an empty store to sample soil or rock. If its store is full,
      a 'drop' action is assumed necessary before sampling.
    - A camera needs to be calibrated before taking an image. Calibration is
      lost after taking an image.
    - Communication requires the rover to be at a waypoint visible from the lander.
    - The heuristic assumes that if a soil/rock sample is required for a goal
      and is not yet collected by any rover, it is still available at its
      original waypoint (`at_soil_sample` or `at_rock_sample`).
    - The heuristic assumes that if an image is required for a goal and is not
      yet taken by any rover, the objective is still visible from its designated
      waypoints and the calibration target is visible from its designated waypoints.

    # Heuristic Initialization
    The constructor extracts static information from the task:
    - The lander's waypoint.
    - Waypoints visible from the lander (communication points).
    - Rover capabilities (soil, rock, imaging).
    - Rover store mapping.
    - Camera information (which rover it's on, supported modes, calibration target).
    - Visibility relationships for objectives and calibration targets.
    - Rover-specific traversal graphs for navigation cost calculation.
    - Sets of all objects by type (rovers, waypoints, cameras, etc.) for easier iteration.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic computes the total estimated cost by summing
    the minimum estimated costs for each goal fact that is not yet satisfied.

    For each unsatisfied goal fact:
    1.  **Identify the type of goal:** `communicated_soil_data`, `communicated_rock_data`, or `communicated_image_data`.
    2.  **For `communicated_soil_data ?w`:**
        - If any rover `r` already has `(have_soil_analysis r ?w)`:
            - The cost is the minimum navigation cost for such a rover `r` from its current location to any lander communication waypoint, plus 1 (for the `communicate_soil_data` action).
        - Otherwise (sample needs collecting):
            - The cost is the minimum over all soil-equipped rovers `r` of:
                - (1 if `r`'s store is full, for a `drop` action)
                - + navigation cost from `r`'s current location to `?w` (sample location)
                - + 1 (for the `sample_soil` action)
                - + navigation cost from `?w` to any lander communication waypoint
                - + 1 (for the `communicate_soil_data` action).
            - This calculation is only possible if `(at_soil_sample ?w)` is true in the current state. If the sample is gone and no rover has it, the goal is unreachable.
    3.  **For `communicated_rock_data ?w`:**
        - Similar logic to `communicated_soil_data`, replacing soil actions/predicates with rock ones and checking `(at_rock_sample ?w)`.
    4.  **For `communicated_image_data ?o ?m`:**
        - If any rover `r` already has `(have_image r ?o ?m)`:
            - The cost is the minimum navigation cost for such a rover `r` from its current location to any lander communication waypoint, plus 1 (for the `communicate_image_data` action).
        - Otherwise (image needs taking):
            - The cost is the minimum over all imaging-equipped rovers `r` with a camera `i` supporting mode `m` of:
                - If camera `i` is not calibrated:
                    - navigation cost from `r`'s current location to any waypoint `w_cal` visible from `i`'s calibration target
                    - + 1 (for the `calibrate` action)
                    - + navigation cost from `w_cal` to any waypoint `w_img` visible from objective `o`
                - If camera `i` is calibrated:
                    - navigation cost from `r`'s current location to any waypoint `w_img` visible from objective `o`
                - + 1 (for the `take_image` action)
                - + navigation cost from `w_img` to any lander communication waypoint
                - + 1 (for the `communicate_image_data` action).
            - This calculation is only possible if there are waypoints visible from the objective `o` and (if calibration is needed) from the calibration target.
    5.  **Summation:** The total heuristic value is the sum of these minimum costs for all unsatisfied goals. If any goal is unreachable (e.g., no path exists, sample gone, no suitable rover/camera), the heuristic returns infinity.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals

        # --- Static Information Extraction ---
        self.lander_waypoint = None
        self.general_visibility_graph = {} # For finding lander comm points
        self.rover_traversal_graph = {} # Rover-specific graphs
        self.rover_capabilities = {} # rover -> set of capabilities ('soil', 'rock', 'imaging')
        self.rover_stores = {} # rover -> store
        self.camera_info = {} # camera -> {'rover': r, 'supports': set(modes), 'calibration_target': t}
        self.objective_visibility = {} # objective -> set(waypoints)
        self.calibration_target_visibility = {} # cal_target -> set(waypoints)

        # Sets of all objects by type (populated during static fact parsing)
        self.rovers = set()
        self.waypoint_names = set() # Renamed to avoid conflict with waypoint objects if they existed
        self.cameras = set()
        self.objectives = set()
        self.modes = set()
        self.stores = set()
        self.landers = set()
        self.cal_targets = set()

        # First pass to identify objects and build basic graphs/mappings
        for fact in task.static:
            parts = get_parts(fact)
            if not parts: continue

            pred = parts[0]
            if pred == 'at_lander':
                self.landers.add(parts[1])
                self.lander_waypoint = parts[2]
                self.waypoint_names.add(parts[2])
            elif pred == 'at': # Initial rover locations are static in initial state, but dynamic in subsequent states
                 # We collect object types here, location will be dynamic
                 obj_name = parts[1]
                 loc_name = parts[2]
                 self.waypoint_names.add(loc_name)
                 # We don't know the type from this fact alone, rely on other predicates
            elif pred == 'can_traverse':
                rover, w1, w2 = parts[1:]
                self.rovers.add(rover)
                self.waypoint_names.add(w1)
                self.waypoint_names.add(w2)
                if rover not in self.rover_traversal_graph:
                    self.rover_traversal_graph[rover] = {}
                if w1 not in self.rover_traversal_graph[rover]:
                    self.rover_traversal_graph[rover][w1] = []
                if w2 not in self.rover_traversal_graph[rover]:
                    self.rover_traversal_graph[rover][w2] = []
                self.rover_traversal_graph[rover][w1].append(w2)
            elif pred == 'visible':
                w1, w2 = parts[1:]
                self.waypoint_names.add(w1)
                self.waypoint_names.add(w2)
                if w1 not in self.general_visibility_graph:
                    self.general_visibility_graph[w1] = []
                if w2 not in self.general_visibility_graph:
                    self.general_visibility_graph[w2] = []
                self.general_visibility_graph[w1].append(w2)
            elif pred == 'equipped_for_soil_analysis':
                rover = parts[1]
                self.rovers.add(rover)
                if rover not in self.rover_capabilities: self.rover_capabilities[rover] = set()
                self.rover_capabilities[rover].add('soil')
            elif pred == 'equipped_for_rock_analysis':
                rover = parts[1]
                self.rovers.add(rover)
                if rover not in self.rover_capabilities: self.rover_capabilities[rover] = set()
                self.rover_capabilities[rover].add('rock')
            elif pred == 'equipped_for_imaging':
                rover = parts[1]
                self.rovers.add(rover)
                if rover not in self.rover_capabilities: self.rover_capabilities[rover] = set()
                self.rover_capabilities[rover].add('imaging')
            elif pred == 'store_of':
                store, rover = parts[1:]
                self.stores.add(store)
                self.rovers.add(rover)
                self.rover_stores[rover] = store
            elif pred == 'on_board':
                camera, rover = parts[1:]
                self.cameras.add(camera)
                self.rovers.add(rover)
                if camera not in self.camera_info: self.camera_info[camera] = {}
                self.camera_info[camera]['rover'] = rover
            elif pred == 'supports':
                camera, mode = parts[1:]
                self.cameras.add(camera)
                self.modes.add(mode)
                if camera not in self.camera_info: self.camera_info[camera] = {}
                if 'supports' not in self.camera_info[camera]: self.camera_info[camera]['supports'] = set()
                self.camera_info[camera]['supports'].add(mode)
            elif pred == 'calibration_target':
                camera, target = parts[1:]
                self.cameras.add(camera)
                self.cal_targets.add(target)
                if camera not in self.camera_info: self.camera_info[camera] = {}
                self.camera_info[camera]['calibration_target'] = target
            elif pred == 'visible_from':
                obj, waypoint = parts[1:]
                self.waypoint_names.add(waypoint)
                # Check if obj is a known calibration target
                is_cal_target = obj in self.cal_targets

                if is_cal_target:
                    if obj not in self.calibration_target_visibility: self.calibration_target_visibility[obj] = set()
                    self.calibration_target_visibility[obj].add(waypoint)
                else: # Assume it's an objective
                    self.objectives.add(obj)
                    if obj not in self.objective_visibility: self.objective_visibility[obj] = set()
                    self.objective_visibility[obj].add(waypoint)

        # Calculate lander communication waypoints.
        # Rover must be at a waypoint ?x such that (visible ?x ?y) is true, where ?y is lander loc.
        self.lander_comm_waypoints = set()
        lander_loc = self.lander_waypoint
        if lander_loc:
            # Need to find waypoints W such that (visible W lander_loc) is true.
            for fact in task.static:
                 parts = get_parts(fact)
                 if match(fact, "visible", "*", lander_loc):
                     self.lander_comm_waypoints.add(parts[1])


        # Ensure all waypoints seen in can_traverse are in the graph dicts, even if no edges
        for rover in self.rovers:
            if rover not in self.rover_traversal_graph:
                 self.rover_traversal_graph[rover] = {}
            for w in self.waypoint_names:
                 if w not in self.rover_traversal_graph[rover]:
                     self.rover_traversal_graph[rover][w] = []

        # Ensure all waypoints seen in visible are in the general graph dicts
        for w in self.waypoint_names:
             if w not in self.general_visibility_graph:
                 self.general_visibility_graph[w] = []


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # --- Dynamic Information Extraction from Current State ---
        current_rover_locations = {} # rover -> waypoint
        rover_soil_samples = {} # rover -> set(waypoints)
        rover_rock_samples = {} # rover -> set(waypoints)
        rover_images = {} # rover -> set((objective, mode))
        rover_store_status = {} # rover -> bool (True if full)
        camera_calibration_status = {} # (camera, rover) -> bool (True if calibrated)
        soil_samples_at_waypoint = set() # set(waypoints)
        rock_samples_at_waypoint = set() # set(waypoints)
        communicated_soil = set() # set(waypoints)
        communicated_rock = set() # set(waypoints)
        communicated_image = set() # set((objective, mode))

        # Initialize dynamic dicts/sets for all known objects
        for r in self.rovers:
            rover_soil_samples[r] = set()
            rover_rock_samples[r] = set()
            rover_images[r] = set()
            rover_store_status[r] = False # Assume empty unless 'full' is seen
        for c in self.cameras:
             # Only initialize if camera info was found in static facts
             if c in self.camera_info and 'rover' in self.camera_info[c]:
                 r = self.camera_info[c]['rover']
                 camera_calibration_status[(c, r)] = False # Assume not calibrated unless 'calibrated' is seen

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            pred = parts[0]
            if pred == 'at':
                obj, loc = parts[1:]
                if obj in self.rovers: # Only track rover locations
                    current_rover_locations[obj] = loc
            elif pred == 'have_soil_analysis':
                rover, waypoint = parts[1:]
                if rover in rover_soil_samples: # Ensure rover is known
                    rover_soil_samples[rover].add(waypoint)
            elif pred == 'have_rock_analysis':
                rover, waypoint = parts[1:]
                if rover in rover_rock_samples: # Ensure rover is known
                    rover_rock_samples[rover].add(waypoint)
            elif pred == 'have_image':
                rover, obj, mode = parts[1:]
                if rover in rover_images: # Ensure rover is known
                    rover_images[rover].add((obj, mode))
            elif pred == 'full':
                store = parts[1]
                # Find which rover this store belongs to
                for r, s in self.rover_stores.items():
                    if s == store:
                        rover_store_status[r] = True
                        break
            elif pred == 'empty':
                 store = parts[1]
                 # Find which rover this store belongs to
                 for r, s in self.rover_stores.items():
                     if s == store:
                         rover_store_status[r] = False
                         break
            elif pred == 'calibrated':
                camera, rover = parts[1:]
                if (camera, rover) in camera_calibration_status: # Ensure camera/rover pair is known
                    camera_calibration_status[(camera, rover)] = True
            elif pred == 'at_soil_sample':
                waypoint = parts[1]
                soil_samples_at_waypoint.add(waypoint)
            elif pred == 'at_rock_sample':
                waypoint = parts[1]
                rock_samples_at_waypoint.add(waypoint)
            elif pred == 'communicated_soil_data':
                waypoint = parts[1]
                communicated_soil.add(waypoint)
            elif pred == 'communicated_rock_data':
                waypoint = parts[1]
                communicated_rock.add(waypoint)
            elif pred == 'communicated_image_data':
                obj, mode = parts[1:]
                communicated_image.add((obj, mode))

        total_cost = 0

        # --- Calculate Cost for each Goal ---
        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue

            pred = parts[0]

            if pred == 'communicated_soil_data':
                w = parts[1]
                if w in communicated_soil:
                    continue # Goal already met

                min_goal_cost = float('inf')

                # Case 1: Sample is already collected by a rover
                for r in self.rovers:
                    if w in rover_soil_samples.get(r, set()):
                        current_w = current_rover_locations.get(r)
                        if current_w:
                            for comm_w in self.lander_comm_waypoints:
                                nav_cost = shortest_path(self.rover_traversal_graph.get(r, {}), current_w, comm_w)
                                if nav_cost is not float('inf'):
                                    min_goal_cost = min(min_goal_cost, nav_cost + 1) # +1 for communicate

                # Case 2: Sample needs to be collected
                # This is only possible if the sample is still at the waypoint
                if min_goal_cost == float('inf') and w in soil_samples_at_waypoint:
                     for r in self.rovers:
                         if 'soil' in self.rover_capabilities.get(r, set()):
                             current_w = current_rover_locations.get(r)
                             if current_w:
                                 drop_cost = 1 if rover_store_status.get(r, False) else 0 # Cost to drop if store is full

                                 nav_cost_to_sample = shortest_path(self.rover_traversal_graph.get(r, {}), current_w, w)
                                 if nav_cost_to_sample is not float('inf'):
                                     # Now calculate cost from sample location to comm point
                                     for comm_w in self.lander_comm_waypoints:
                                         nav_cost_to_comm = shortest_path(self.rover_traversal_graph.get(r, {}), w, comm_w)
                                         if nav_cost_to_comm is not float('inf'):
                                             cost = drop_cost + nav_cost_to_sample + 1 + nav_cost_to_comm + 1 # drop + nav + sample + nav + communicate
                                             min_goal_cost = min(min_goal_cost, cost)

                total_cost += min_goal_cost # Add infinity if unreachable

            elif pred == 'communicated_rock_data':
                w = parts[1]
                if w in communicated_rock:
                    continue # Goal already met

                min_goal_cost = float('inf')

                # Case 1: Sample is already collected by a rover
                for r in self.rovers:
                    if w in rover_rock_samples.get(r, set()):
                        current_w = current_rover_locations.get(r)
                        if current_w:
                            for comm_w in self.lander_comm_waypoints:
                                nav_cost = shortest_path(self.rover_traversal_graph.get(r, {}), current_w, comm_w)
                                if nav_cost is not float('inf'):
                                    min_goal_cost = min(min_goal_cost, nav_cost + 1) # +1 for communicate

                # Case 2: Sample needs to be collected
                # This is only possible if the sample is still at the waypoint
                if min_goal_cost == float('inf') and w in rock_samples_at_waypoint:
                     for r in self.rovers:
                         if 'rock' in self.rover_capabilities.get(r, set()):
                             current_w = current_rover_locations.get(r)
                             if current_w:
                                 drop_cost = 1 if rover_store_status.get(r, False) else 0 # Cost to drop if store is full

                                 nav_cost_to_sample = shortest_path(self.rover_traversal_graph.get(r, {}), current_w, w)
                                 if nav_cost_to_sample is not float('inf'):
                                     # Now calculate cost from sample location to comm point
                                     for comm_w in self.lander_comm_waypoints:
                                         nav_cost_to_comm = shortest_path(self.rover_traversal_graph.get(r, {}), w, comm_w)
                                         if nav_cost_to_comm is not float('inf'):
                                             cost = drop_cost + nav_cost_to_sample + 1 + nav_cost_to_comm + 1 # drop + nav + sample + nav + communicate
                                             min_goal_cost = min(min_goal_cost, cost)

                total_cost += min_goal_cost # Add infinity if unreachable

            elif pred == 'communicated_image_data':
                o, m = parts[1:]
                if (o, m) in communicated_image:
                    continue # Goal already met

                min_goal_cost = float('inf')

                # Case 1: Image is already taken by a rover
                for r in self.rovers:
                    if (o, m) in rover_images.get(r, set()):
                        current_w = current_rover_locations.get(r)
                        if current_w:
                            for comm_w in self.lander_comm_waypoints:
                                nav_cost = shortest_path(self.rover_traversal_graph.get(r, {}), current_w, comm_w)
                                if nav_cost is not float('inf'):
                                    min_goal_cost = min(min_goal_cost, nav_cost + 1) # +1 for communicate

                # Case 2: Image needs to be taken
                # This is only possible if objective is visible from somewhere
                img_waypoints = self.objective_visibility.get(o, set())
                if min_goal_cost == float('inf') and img_waypoints:
                    for r in self.rovers:
                        if 'imaging' in self.rover_capabilities.get(r, set()):
                            current_w = current_rover_locations.get(r)
                            if current_w:
                                for cam, cam_info in self.camera_info.items():
                                    if cam_info.get('rover') == r and m in cam_info.get('supports', set()):
                                        cal_target = cam_info.get('calibration_target')
                                        cal_waypoints = self.calibration_target_visibility.get(cal_target, set())

                                        # Need calibration if camera is not calibrated
                                        needs_calibration = not camera_calibration_status.get((cam, r), False)

                                        # Calibration is only possible if there are cal_waypoints
                                        if needs_calibration and not cal_waypoints:
                                            continue # Cannot calibrate, cannot take image

                                        for w_img in img_waypoints:
                                            for comm_w in self.lander_comm_waypoints:
                                                cost = 0
                                                nav_start_img = current_w
                                                nav_cost_to_img = float('inf')

                                                if needs_calibration:
                                                    # Find min nav cost to any calibration waypoint
                                                    min_nav_to_cal = float('inf')
                                                    best_w_cal = None
                                                    for w_cal in cal_waypoints:
                                                        nav = shortest_path(self.rover_traversal_graph.get(r, {}), current_w, w_cal)
                                                        if nav < min_nav_to_cal:
                                                            min_nav_to_cal = nav
                                                            best_w_cal = w_cal

                                                    if min_nav_to_cal is not float('inf'):
                                                        cost += min_nav_to_cal + 1 # nav + calibrate
                                                        nav_start_img = best_w_cal # Start next nav from calibration point
                                                    else:
                                                        continue # Cannot reach any calibration point

                                                # Nav from current loc (or cal point) to image point
                                                nav_cost_to_img = shortest_path(self.rover_traversal_graph.get(r, {}), nav_start_img, w_img)

                                                if nav_cost_to_img is not float('inf'):
                                                    cost += nav_cost_to_img + 1 # nav + take_image
                                                    nav_start_comm = w_img # Start next nav from image point

                                                    # Nav from image point to comm point
                                                    nav_cost_to_comm = shortest_path(self.rover_traversal_graph.get(r, {}), nav_start_comm, comm_w)
                                                    if nav_cost_to_comm is not float('inf'):
                                                        cost += nav_cost_to_comm + 1 # nav + communicate
                                                        min_goal_cost = min(min_goal_cost, cost)
                                                    # else: continue # Cannot reach comm point
                                                # else: continue # Cannot reach image point


                total_cost += min_goal_cost # Add infinity if unreachable


        # If total_cost is infinity, it means at least one goal is unreachable
        if total_cost == float('inf'):
            return float('inf') # Or a large number like 1e9

        return total_cost
