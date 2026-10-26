from fnmatch import fnmatch
from collections import deque
import sys

# Assuming Heuristic base class is available in heuristics.heuristic_base
# If not, a dummy class is defined below for standalone testing.
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    print("Warning: heuristics.heuristic_base not found. Using a dummy base class.")
    class Heuristic:
        def __init__(self, task):
            pass
        def __call__(self, node):
            raise NotImplementedError

def get_parts(fact):
    """Extract the components of a PDDL fact."""
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
         # Handle unexpected input format
         return []
    # Remove parentheses and split by space
    return fact[1:-1].split()

def match(fact_parts, *args):
    """
    Check if parsed fact parts match a given pattern.
    - `fact_parts`: List of strings (e.g., ['at', 'rover1', 'waypoint1']).
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    if len(fact_parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(fact_parts, args))

def bfs(graph, start_node):
    """
    Perform BFS to find shortest paths from start_node in a graph.
    Graph is {node: [neighbor1, neighbor2, ...]}
    Returns {node: distance}
    """
    # Handle cases where start_node is not in the graph or graph is empty
    if start_node not in graph or not graph:
         # Return distances indicating unreachable for all nodes in the graph
         return {node: float('inf') for node in graph}

    distances = {node: float('inf') for node in graph}
    distances[start_node] = 0
    queue = deque([start_node])

    while queue:
        current_node = queue.popleft()

        # Check if current_node has neighbors in the graph
        if current_node in graph:
            for neighbor in graph[current_node]:
                # Ensure neighbor is a valid node in the graph distances dictionary
                if neighbor in distances and distances[neighbor] == float('inf'):
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)
    return distances


class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    Estimates the cost to reach the goal by summing the estimated costs
    for each unachieved goal predicate independently.

    Costs considered for each goal:
    1. Navigation to required locations (sample, image, calibrate, communicate).
       Uses precomputed shortest paths for each rover.
    2. Action costs (sample, image, calibrate, communicate, drop).
       Adds 1 for each required action.
    3. Store availability for sampling (adds 1 for drop if the chosen rover's
       store is full when sampling is needed).
    4. Camera calibration for imaging (adds 1 for calibrate if needed, plus
       additional navigation cost to visit calibration waypoint).

    Simplifications:
    - Assumes the same rover that collects data/image also communicates it.
    - Greedily picks the "best" rover/waypoint for each sub-task independently.
    - Does not account for resource contention (multiple goals needing the same rover/store/camera).
    - Assumes calibration happens before imaging if both are needed for an image goal,
      and calculates navigation cost accordingly (curr -> cal_wp -> img_wp).
    """

    def __init__(self, task):
        """Initialize the heuristic by precomputing navigation costs and parsing static facts."""
        self.goals = task.goals
        self.static_facts = task.static # frozenset of strings
        self.static_predicates = [get_parts(fact) for fact in self.static_facts if get_parts(fact)] # List of lists/tuples, filter empty parts

        # Collect all relevant objects from static predicates and goals
        self.rovers = set()
        self.waypoints = set()
        self.cameras = set()
        self.objectives = set()
        self.modes = set()
        self.stores = set()
        self.landers = set()

        for p in self.static_predicates:
            if not p: continue
            if p[0] in ['at', 'equipped_for_soil_analysis', 'equipped_for_rock_analysis', 'equipped_for_imaging'] and len(p) > 1: self.rovers.add(p[1])
            if p[0] in ['at', 'at_lander', 'can_traverse', 'visible', 'visible_from', 'at_soil_sample', 'at_rock_sample']:
                 if len(p) > 1: self.waypoints.add(p[1])
                 if len(p) > 2: self.waypoints.add(p[2])
                 if len(p) > 3: self.waypoints.add(p[3]) # can_traverse has 3 waypoints
            if p[0] in ['on_board', 'calibration_target', 'supports'] and len(p) > 1: self.cameras.add(p[1])
            if p[0] in ['calibration_target', 'visible_from']:
                 if len(p) > 1: self.objectives.add(p[1])
                 if len(p) > 2: self.objectives.add(p[2])
            if p[0] in ['supports'] and len(p) > 2:
                 self.modes.add(p[2])
            if p[0] in ['store_of'] and len(p) > 1: self.stores.add(p[1])
            if p[0] in ['at_lander'] and len(p) > 1: self.landers.add(p[1])

        # Add objects from goals
        for goal_str in self.goals:
            goal_parts = get_parts(goal_str)
            if not goal_parts: continue
            if goal_parts[0] in ['communicated_soil_data', 'communicated_rock_data']:
                if len(goal_parts) > 1: self.waypoints.add(goal_parts[1])
            elif goal_parts[0] == 'communicated_image_data':
                if len(goal_parts) > 1: self.objectives.add(goal_parts[1])
                if len(goal_parts) > 2: self.modes.add(goal_parts[2])

        # Ensure all collected objects are added to their respective sets
        # (This step might be redundant if parsing is thorough, but adds safety)
        # Example: if a waypoint only appears in a goal, it should be in self.waypoints


        # Build rover-specific navigation graphs
        self.rover_graphs = {rover: {wp: [] for wp in self.waypoints} for rover in self.rovers}
        for p in self.static_predicates:
            if p[0] == 'can_traverse' and len(p) == 4:
                r, y, z = p[1], p[2], p[3]
                # Check if visible Y Z exists
                is_visible = any(match(vis_p, 'visible', y, z) for vis_p in self.static_predicates)
                if is_visible and r in self.rover_graphs and y in self.rover_graphs[r] and z in self.waypoints: # Ensure z is a known waypoint
                     self.rover_graphs[r][y].append(z)

        # Precompute shortest path distances for each rover
        self.rover_nav_costs = {} # {rover: {from_wp: {to_wp: distance}}}
        for rover in self.rovers:
            self.rover_nav_costs[rover] = {}
            if rover in self.rover_graphs:
                for start_wp in self.waypoints:
                    self.rover_nav_costs[rover][start_wp] = bfs(self.rover_graphs[rover], start_wp)

        # Identify lander location(s)
        self.lander_locations = {p[2] for p in self.static_predicates if p[0] == 'at_lander' and len(p) == 3}

        # Identify communication waypoints (visible from any lander location)
        self.comm_waypoints = set()
        for lander_loc in self.lander_locations:
            for p in self.static_predicates:
                if p[0] == 'visible' and len(p) == 3:
                    wp1, wp2 = p[1], p[2]
                    if wp1 == lander_loc and wp2 in self.waypoints: self.comm_waypoints.add(wp2)
                    if wp2 == lander_loc and wp1 in self.waypoints: self.comm_waypoints.add(wp1) # visible is symmetric in example, but PDDL doesn't require it
        # Add lander locations themselves if they are waypoints and can be communication points
        self.comm_waypoints.update(l for l in self.lander_locations if l in self.waypoints)


        # Store static relationships for quick lookup
        self.equipped_for_soil = {p[1] for p in self.static_predicates if p[0] == 'equipped_for_soil_analysis' and len(p) == 2}
        self.equipped_for_rock = {p[1] for p in self.static_predicates if p[0] == 'equipped_for_rock_analysis' and len(p) == 2}
        self.equipped_for_imaging = {p[1] for p in self.static_predicates if p[0] == 'equipped_for_imaging' and len(p) == 2}
        self.store_of_rover = {p[1]: p[2] for p in self.static_predicates if p[0] == 'store_of' and len(p) == 3} # store -> rover
        self.camera_on_rover = {p[1]: p[2] for p in self.static_predicates if p[0] == 'on_board' and len(p) == 3} # camera -> rover
        self.camera_supports_mode = {} # camera -> {mode1, mode2, ...}
        for p in self.static_predicates:
            if p[0] == 'supports' and len(p) == 3:
                c, m = p[1], p[2]
                if c not in self.camera_supports_mode:
                    self.camera_supports_mode[c] = set()
                self.camera_supports_mode[c].add(m)
        self.camera_cal_target = {p[1]: p[2] for p in self.static_predicates if p[0] == 'calibration_target' and len(p) == 3} # camera -> objective
        self.objective_visible_from = {} # objective -> {wp1, wp2, ...}
        for p in self.static_predicates:
            if p[0] == 'visible_from' and len(p) == 3:
                o, w = p[1], p[2]
                if o not in self.objective_visible_from:
                    self.objective_visible_from[o] = set()
                self.objective_visible_from[o].add(w)


    def __call__(self, node):
        state = node.state
        total_cost = 0

        # Extract dynamic state information
        rover_at = {} # rover -> waypoint
        store_full = set() # {store1, store2, ...}
        camera_calibrated = set() # {(camera, rover), ...}
        have_soil = set() # {waypoint1, waypoint2, ...}
        have_rock = set() # {waypoint1, waypoint2, ...}
        have_image = set() # {(objective, mode), ...}

        for fact_str in state:
            parts = get_parts(fact_str)
            if not parts: continue
            if parts[0] == 'at' and len(parts) == 3 and parts[1] in self.rovers:
                rover_at[parts[1]] = parts[2]
            elif parts[0] == 'full' and len(parts) == 2 and parts[1] in self.stores:
                store_full.add(parts[1])
            elif parts[0] == 'calibrated' and len(parts) == 3 and parts[1] in self.cameras and parts[2] in self.rovers:
                camera_calibrated.add((parts[1], parts[2]))
            elif parts[0] == 'have_soil_analysis' and len(parts) == 3 and parts[2] in self.waypoints:
                have_soil.add(parts[2])
            elif parts[0] == 'have_rock_analysis' and len(parts) == 3 and parts[2] in self.waypoints:
                have_rock.add(parts[2])
            elif parts[0] == 'have_image' and len(parts) == 4 and parts[2] in self.objectives and parts[3] in self.modes:
                have_image.add((parts[2], parts[3]))

        # Process each goal
        for goal_str in self.goals:
            goal_parts = get_parts(goal_str)
            if not goal_parts: continue
            goal_type = goal_parts[0]

            if goal_type == 'communicated_soil_data' and len(goal_parts) == 2:
                w = goal_parts[1]
                if goal_str not in state:
                    cost_this_goal = 0
                    if w not in have_soil:
                        # Need to sample
                        cost_this_goal += 1 # sample_soil action

                        # Find an equipped rover R that can reach W, minimizing nav_cost + drop_cost
                        equipped_rovers = self.equipped_for_soil
                        best_rover = None
                        min_nav_plus_drop_cost = float('inf')

                        for rover in equipped_rovers:
                            if rover not in rover_at or rover_at[rover] not in self.waypoints: continue # Rover location unknown or invalid
                            if rover not in self.rover_nav_costs or rover_at[rover] not in self.rover_nav_costs[rover] or w not in self.rover_nav_costs[rover][rover_at[rover]]:
                                continue # Rover cannot reach W

                            nav_cost = self.rover_nav_costs[rover][rover_at[rover]][w]

                            # Check store status for this rover
                            store = next((s for s, r in self.store_of_rover.items() if r == rover), None)
                            store_is_full = (store and store in store_full)
                            drop_cost = 1 if store_is_full else 0

                            current_nav_plus_drop_cost = nav_cost + drop_cost

                            if current_nav_plus_drop_cost < min_nav_plus_drop_cost:
                                min_nav_plus_drop_cost = current_nav_plus_drop_cost
                                best_rover = rover

                        if best_rover is None: return -1 # Cannot sample
                        cost_this_goal += min_nav_plus_drop_cost # Includes drop cost if needed
                        rover_for_comm = best_rover # Assume same rover communicates
                        nav_start_loc_comm = w # Rover is at W after sampling

                    else: # Already have the sample
                        # Find *a* rover that has the sample
                        rover_with_sample = next((get_parts(fact)[1] for fact in state if match(get_parts(fact), 'have_soil_analysis', '*', w)), None)
                        if not rover_with_sample or rover_with_sample not in rover_at or rover_at[rover_with_sample] not in self.waypoints: return -1 # Should not happen or invalid state
                        rover_for_comm = rover_with_sample
                        nav_start_loc_comm = rover_at[rover_for_comm] # Rover is at its current location

                    # Need to communicate
                    cost_this_goal += 1 # communicate_soil_data action
                    # Find nearest communication waypoint for rover_for_comm from nav_start_loc_comm
                    best_nav_cost_to_comm = float('inf')
                    if rover_for_comm and nav_start_loc_comm and rover_for_comm in self.rover_nav_costs and nav_start_loc_comm in self.rover_nav_costs[rover_for_comm]:
                        for comm_wp in self.comm_waypoints:
                             if comm_wp in self.rover_nav_costs[rover_for_comm][nav_start_loc_comm]:
                                  best_nav_cost_to_comm = min(best_nav_cost_to_comm, self.rover_nav_costs[rover_for_comm][nav_start_loc_comm][comm_wp])

                    if best_nav_cost_to_comm == float('inf'): return -1 # Cannot reach communication point
                    cost_this_goal += best_nav_cost_to_comm

                    total_cost += cost_this_goal

            elif goal_type == 'communicated_rock_data' and len(goal_parts) == 2:
                w = goal_parts[1]
                if goal_str not in state:
                    cost_this_goal = 0
                    if w not in have_rock:
                        # Need to sample
                        cost_this_goal += 1 # sample_rock action

                        # Find an equipped rover R that can reach W, minimizing nav_cost + drop_cost
                        equipped_rovers = self.equipped_for_rock
                        best_rover = None
                        min_nav_plus_drop_cost = float('inf')

                        for rover in equipped_rovers:
                            if rover not in rover_at or rover_at[rover] not in self.waypoints: continue # Rover location unknown or invalid
                            if rover not in self.rover_nav_costs or rover_at[rover] not in self.rover_nav_costs[rover] or w not in self.rover_nav_costs[rover][rover_at[rover]]:
                                continue # Rover cannot reach W

                            nav_cost = self.rover_nav_costs[rover][rover_at[rover]][w]

                            # Check store status for this rover
                            store = next((s for s, r in self.store_of_rover.items() if r == rover), None)
                            store_is_full = (store and store in store_full)
                            drop_cost = 1 if store_is_full else 0

                            current_nav_plus_drop_cost = nav_cost + drop_cost

                            if current_nav_plus_drop_cost < min_nav_plus_drop_cost:
                                min_nav_plus_drop_cost = current_nav_plus_drop_cost
                                best_rover = rover

                        if best_rover is None: return -1 # Cannot sample
                        cost_this_goal += min_nav_plus_drop_cost # Includes drop cost if needed
                        rover_for_comm = best_rover # Assume same rover communicates
                        nav_start_loc_comm = w # Rover is at W after sampling

                    else: # Already have the sample
                        rover_with_sample = next((get_parts(fact)[1] for fact in state if match(get_parts(fact), 'have_rock_analysis', '*', w)), None)
                        if not rover_with_sample or rover_with_sample not in rover_at or rover_at[rover_with_sample] not in self.waypoints: return -1
                        rover_for_comm = rover_with_sample
                        nav_start_loc_comm = rover_at[rover_for_comm]

                    # Need to communicate
                    cost_this_goal += 1 # communicate_rock_data action
                    best_nav_cost_to_comm = float('inf')
                    if rover_for_comm and nav_start_loc_comm and rover_for_comm in self.rover_nav_costs and nav_start_loc_comm in self.rover_nav_costs[rover_for_comm]:
                        for comm_wp in self.comm_waypoints:
                             if comm_wp in self.rover_nav_costs[rover_for_comm][nav_start_loc_comm]:
                                  best_nav_cost_to_comm = min(best_nav_cost_to_comm, self.rover_nav_costs[rover_for_comm][nav_start_loc_comm][comm_wp])

                            if best_nav_cost_to_comm == float('inf'): return -1
                            cost_this_goal += best_nav_cost_to_comm

                            total_cost += cost_this_goal

            elif goal_type == 'communicated_image_data' and len(goal_parts) == 3:
                o, m = goal_parts[1], goal_parts[2]
                if goal_str not in state:
                    cost_this_goal = 0

                    # Check if the image is already taken by *any* rover
                    image_already_taken = any(match(get_parts(fact), 'have_image', '*', o, m) for fact in state)

                    if not image_already_taken:
                        # Need to take image
                        cost_this_goal += 1 # take_image action

                        # Find suitable rovers/cameras for this image goal
                        suitable_rovers_cameras = [] # List of (rover, camera)
                        for camera, rover in self.camera_on_rover.items():
                            if rover in self.equipped_for_imaging and camera in self.camera_supports_mode and m in self.camera_supports_mode[camera]:
                                suitable_rovers_cameras.append((rover, camera))

                        if not suitable_rovers_cameras: return -1 # No rover can take this image

                        # Find best image waypoint (visible from o) and best rover/camera
                        best_nav_cost_to_image = float('inf')
                        image_waypoint = None
                        best_rover_camera = None # (rover, camera)

                        # Find waypoints visible from objective O
                        visible_wps = self.objective_visible_from.get(o, set())
                        if not visible_wps: return -1 # No waypoint to view objective

                        for rover, camera in suitable_rovers_cameras:
                            if rover not in rover_at or rover_at[rover] not in self.waypoints: continue # Rover location unknown or invalid
                            if rover not in self.rover_nav_costs or rover_at[rover] not in self.rover_nav_costs[rover]: continue

                            for img_wp in visible_wps:
                                if img_wp in self.rover_nav_costs[rover][rover_at[rover]]:
                                    nav_cost = self.rover_nav_costs[rover][rover_at[rover]][img_wp]
                                    if nav_cost < best_nav_cost_to_image:
                                        best_nav_cost_to_image = nav_cost
                                        image_waypoint = img_wp
                                        best_rover_camera = (rover, camera)

                        if best_rover_camera is None: return -1 # Cannot reach any image location
                        rover_for_image, camera_for_image = best_rover_camera

                        # Check calibration for the chosen rover/camera
                        needs_calibration = (camera_for_image, rover_for_image) not in camera_calibrated

                        if needs_calibration:
                            # Need to calibrate
                            cost_this_goal += 1 # calibrate action
                            # Find calibration target T for camera_for_image
                            cal_target = self.camera_cal_target.get(camera_for_image)
                            if not cal_target: return -1 # Camera has no calibration target

                            # Find best calibration waypoint (visible from cal_target) for rover_for_image
                            best_nav_cost_to_cal = float('inf')
                            cal_waypoint = None
                            # Find waypoints visible from calibration target T
                            visible_wps_cal = self.objective_visible_from.get(cal_target, set())
                            if not visible_wps_cal: return -1 # No waypoint to view calibration target

                            nav_start_loc_cal = rover_at[rover_for_image]

                            for cal_wp in visible_wps_cal:
                                if cal_wp in self.rover_nav_costs[rover_for_image][nav_start_loc_cal]:
                                    nav_cost = self.rover_nav_costs[rover_for_image][nav_start_loc_cal][cal_wp]
                                    if nav_cost < best_nav_cost_to_cal:
                                        best_nav_cost_to_cal = nav_cost
                                        cal_waypoint = cal_wp

                            if cal_waypoint is None: return -1 # Cannot reach any calibration location

                            # Total navigation cost for imaging and calibration:
                            # From current location to cal_wp, then from cal_wp to image_wp.
                            # Or from current location to image_wp, then from image_wp to cal_wp.
                            nav_start = rover_at[rover_for_image]
                            nav_cost_option1 = float('inf')
                            if nav_start in self.rover_nav_costs[rover_for_image] and cal_waypoint in self.rover_nav_costs[rover_for_image][nav_start] and cal_waypoint in self.rover_nav_costs[rover_for_image] and image_waypoint in self.rover_nav_costs[rover_for_image][cal_waypoint]:
                                nav_cost_option1 = self.rover_nav_costs[rover_for_image][nav_start][cal_waypoint] + self.rover_nav_costs[rover_for_image][cal_waypoint][image_waypoint]

                            nav_cost_option2 = float('inf')
                            if nav_start in self.rover_nav_costs[rover_for_image] and image_waypoint in self.rover_nav_costs[rover_for_image][nav_start] and image_waypoint in self.rover_nav_costs[rover_for_image] and cal_waypoint in self.rover_nav_costs[rover_for_image][image_waypoint]:
                                nav_cost_option2 = self.rover_nav_costs[rover_for_image][nav_start][image_waypoint] + self.rover_nav_costs[rover_for_image][image_waypoint][cal_waypoint]

                            best_total_nav_cost_for_imaging = min(nav_cost_option1, nav_cost_option2)
                            if best_total_nav_cost_for_imaging == float('inf'): return -1
                            cost_this_goal += best_total_nav_cost_for_imaging

                        else: # Calibration is not needed
                            # Only need to navigate to the image waypoint
                            cost_this_goal += best_nav_cost_to_image # This was already calculated as nav from curr to image_wp

                        rover_for_comm = rover_for_image # Assume same rover communicates
                        nav_start_loc_comm = image_waypoint # Rover is at image_waypoint after imaging

                    else: # Image is already taken (have_image is true for some rover)
                         # Find *a* rover that has the image
                         rover_with_image = next((get_parts(fact)[1] for fact in state if match(get_parts(fact), 'have_image', '*', o, m)), None)
                         if not rover_with_image or rover_with_image not in rover_at or rover_at[rover_with_image] not in self.waypoints: return -1
                         rover_for_comm = rover_with_image
                         nav_start_loc_comm = rover_at[rover_for_comm] # Rover is at its current location


                    # Need to communicate
                    cost_this_goal += 1 # communicate_image_data action
                    # Find nearest communication waypoint for rover_for_comm from nav_start_loc_comm
                    best_nav_cost_to_comm = float('inf')
                    if rover_for_comm and nav_start_loc_comm and rover_for_comm in self.rover_nav_costs and nav_start_loc_comm in self.rover_nav_costs[rover_for_comm]:
                        for comm_wp in self.comm_waypoints:
                             if comm_wp in self.rover_nav_costs[rover_for_comm][nav_start_loc_comm]:
                                  best_nav_cost_to_comm = min(best_nav_cost_to_comm, self.rover_nav_costs[rover_for_comm][nav_start_loc_comm][comm_wp])

                    if best_nav_cost_to_comm == float('inf'): return -1 # Cannot reach communication point
                    cost_this_goal += best_nav_cost_to_comm

                    total_cost += cost_this_goal

        return total_cost

