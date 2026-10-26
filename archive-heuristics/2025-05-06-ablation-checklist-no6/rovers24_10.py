from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import collections

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def bfs(graph, start_node):
    """
    Performs Breadth-First Search on a graph to find shortest distances from a start node.
    Graph is represented as an adjacency list: {node: {neighbor1, neighbor2, ...}}
    Returns a dictionary {node: distance}.
    """
    distances = {node: float('inf') for node in graph}
    # Add start_node to distances if it's not already in the graph keys but might be a valid node
    if start_node not in distances:
        distances[start_node] = float('inf')

    distances[start_node] = 0
    queue = collections.deque([start_node])

    while queue:
        current_node = queue.popleft()
        current_dist = distances[current_node]

        # Check if current_node is in the graph before iterating neighbors
        if current_node in graph:
            for neighbor in graph[current_node]:
                # Ensure neighbor is also in the distances dictionary
                if neighbor not in distances:
                     distances[neighbor] = float('inf')

                if distances[neighbor] == float('inf'):
                    distances[neighbor] = current_dist + 1
                    queue.append(neighbor)
    return distances


class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    Estimates the number of actions required to achieve uncommunicated goals.
    It sums the estimated costs for each unachieved goal fact independently.
    The cost for each goal is estimated by finding the minimum cost path
    for a suitable rover to perform the necessary steps (sample/image, then communicate),
    including navigation, sampling/imaging, calibration (if needed), and communication.
    Navigation costs are precomputed shortest paths using BFS on the rover's traversal graph.
    Resource constraints (like store capacity beyond the immediate need to drop) and
    rover conflicts are ignored for simplicity and efficiency.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static
        initial_state = task.initial_state # Need initial state for sample locations

        # --- Parse Static Facts ---
        self.rover_capabilities = collections.defaultdict(set)
        self.store_owners = {} # {store: rover}
        self.lander_location = None
        self.waypoint_graph = collections.defaultdict(set) # For visible predicate
        self.rover_traversal_graphs = collections.defaultdict(lambda: collections.defaultdict(set)) # {rover: {wp: {traversable_wps}}}
        self.camera_info = {} # {camera: {'rover': rover, 'supports': {modes}, 'calibration_target': objective}}
        self.calibration_targets = {} # {camera: objective}
        self.objective_visibility = collections.defaultdict(set) # {objective: {visible_wps}}
        self.soil_sample_wps_initial = set() # Waypoints with soil samples initially
        self.rock_sample_wps_initial = set() # Waypoints with rock samples initially
        self.all_waypoints = set() # Collect all waypoints mentioned in init/static

        # Collect all waypoints mentioned in static facts
        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue # Skip empty facts

            pred = parts[0]
            if pred == 'at_lander':
                self.lander_location = parts[2]
                self.all_waypoints.add(parts[2])
            elif pred == 'visible':
                wp1, wp2 = parts[1], parts[2]
                self.waypoint_graph[wp1].add(wp2)
                self.waypoint_graph[wp2].add(wp1) # Assuming visible is symmetric
                self.all_waypoints.add(wp1)
                self.all_waypoints.add(wp2)
            elif pred == 'can_traverse':
                rover, wp1, wp2 = parts[1], parts[2], parts[3]
                self.rover_traversal_graphs[rover][wp1].add(wp2)
                self.rover_traversal_graphs[rover][wp2].add(wp1) # Assuming can_traverse is symmetric
                self.all_waypoints.add(wp1)
                self.all_waypoints.add(wp2)
            elif pred == 'visible_from':
                objective, waypoint = parts[1], parts[2]
                self.objective_visibility[objective].add(waypoint)
                self.all_waypoints.add(waypoint)
            # Collect other static info
            elif pred == 'equipped_for_soil_analysis':
                self.rover_capabilities[parts[1]].add('soil')
            elif pred == 'equipped_for_rock_analysis':
                self.rover_capabilities[parts[1]].add('rock')
            elif pred == 'equipped_for_imaging':
                self.rover_capabilities[parts[1]].add('imaging')
            elif pred == 'store_of':
                self.store_owners[parts[1]] = parts[2]
            elif pred == 'on_board':
                camera, rover = parts[1], parts[2]
                if camera not in self.camera_info: self.camera_info[camera] = {}
                self.camera_info[camera]['rover'] = rover
            elif pred == 'supports':
                camera, mode = parts[1], parts[2]
                if camera not in self.camera_info: self.camera_info[camera] = {}
                if 'supports' not in self.camera_info[camera]: self.camera_info[camera]['supports'] = set()
                self.camera_info[camera]['supports'].add(mode)
            elif pred == 'calibration_target':
                camera, objective = parts[1], parts[2]
                self.calibration_targets[camera] = objective
                if camera not in self.camera_info: self.camera_info[camera] = {}
                self.camera_info[camera]['calibration_target'] = objective


        # Collect initial sample locations and rover/lander locations from initial state
        for fact in initial_state:
             parts = get_parts(fact)
             if not parts: continue
             pred = parts[0]
             if pred == 'at_soil_sample':
                 self.soil_sample_wps_initial.add(parts[1])
                 self.all_waypoints.add(parts[1])
             elif pred == 'at_rock_sample':
                 self.rock_sample_wps_initial.add(parts[1])
                 self.all_waypoints.add(parts[1])
             elif pred == 'at': # Collect all waypoints rovers start at
                 if len(parts) == 3 and parts[1].startswith('rover'):
                     self.all_waypoints.add(parts[2])
             elif pred == 'at_lander': # Collect lander waypoint (already done for static, but safety)
                  self.all_waypoints.add(parts[2])


        # Precompute calibration waypoints based on calibration targets and visibility
        for camera, target in self.calibration_targets.items():
            if target in self.objective_visibility:
                self.calibration_wps[camera] = self.objective_visibility[target]
            else:
                 self.calibration_wps[camera] = set() # No visible points for target

        # Precompute communication waypoints (visible from lander)
        self.communication_wps = self.waypoint_graph.get(self.lander_location, set())


        # --- Precompute Rover Distances ---
        self.rover_distances = {} # {rover: { (start_wp, end_wp): distance }}
        all_rovers = set(self.rover_traversal_graphs.keys()) # Get all rovers mentioned in can_traverse

        # Also add rovers mentioned in initial state but not necessarily in can_traverse
        for fact in initial_state:
             if match(fact, "at", "*", "*"):
                 parts = get_parts(fact)
                 obj_name = parts[1]
                 if obj_name.startswith('rover'):
                     all_rovers.add(obj_name)

        # Ensure all waypoints are in the graph keys for BFS even if they have no outgoing edges
        # This is important so BFS can calculate distance *to* them.
        graph_with_all_wps = collections.defaultdict(set)
        for wp in self.all_waypoints:
             graph_with_all_wps[wp] = set() # Initialize all known WPs

        for rover in all_rovers:
            # Start with the base graph containing all waypoints
            rover_graph = collections.defaultdict(set, graph_with_all_wps)
            # Add rover-specific traversal edges
            for wp, neighbors in self.rover_traversal_graphs.get(rover, {}).items():
                 rover_graph[wp].update(neighbors)
                 # Ensure neighbors are also in the graph_with_all_wps structure
                 for neighbor in neighbors:
                      if neighbor not in rover_graph:
                           rover_graph[neighbor] = set()


            self.rover_distances[rover] = {}
            # Run BFS from every waypoint in the problem, using the rover's specific graph
            for start_wp in self.all_waypoints:
                 distances_from_start = bfs(rover_graph, start_wp)
                 for end_wp in self.all_waypoints:
                     dist = distances_from_start.get(end_wp, float('inf'))
                     if dist != float('inf'):
                         self.rover_distances[rover][(start_wp, end_wp)] = dist


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        total_cost = 0

        # Map current rover locations
        rover_locations = {}
        for fact in state:
            if match(fact, "at", "*", "*"):
                parts = get_parts(fact)
                obj_name, wp_name = parts[1], parts[2]
                # Check if the object is a rover (simple check based on naming convention)
                if obj_name.startswith('rover'):
                     rover_locations[obj_name] = wp_name

        # Map current store status
        store_status = {} # {store: 'empty' or 'full'}
        for fact in state:
             if match(fact, "empty", "*"):
                 store_status[get_parts(fact)[1]] = 'empty'
             elif match(fact, "full", "*"):
                 store_status[get_parts(fact)[1]] = 'full'

        # Map current analysis/image status
        have_soil = set() # {(rover, waypoint)}
        have_rock = set() # {(rover, waypoint)}
        have_image = set() # {(rover, objective, mode)}
        calibrated_cameras = set() # {(camera, rover)}

        for fact in state:
             if match(fact, "have_soil_analysis", "*", "*"):
                 have_soil.add(tuple(get_parts(fact)[1:]))
             elif match(fact, "have_rock_analysis", "*", "*"):
                 have_rock.add(tuple(get_parts(fact)[1:]))
             elif match(fact, "have_image", "*", "*", "*"):
                 have_image.add(tuple(get_parts(fact)[1:]))
             elif match(fact, "calibrated", "*", "*"):
                 calibrated_cameras.add(tuple(get_parts(fact)[1:]))

        # Track which sample/image goals are already communicated
        communicated_soil = {get_parts(fact)[1] for fact in state if match(fact, "communicated_soil_data", "*")}
        communicated_rock = {get_parts(fact)[1] for fact in state if match(fact, "communicated_rock_data", "*")}
        communicated_image = {tuple(get_parts(fact)[1:]) for fact in state if match(fact, "communicated_image_data", "*", "*")}


        # Consider each goal fact
        for goal_fact in self.goals:
            # Check if the goal is already fully communicated
            parts = get_parts(goal_fact)
            pred = parts[0]

            if pred == 'communicated_soil_data' and parts[1] in communicated_soil: continue
            if pred == 'communicated_rock_data' and parts[1] in communicated_rock: continue
            if pred == 'communicated_image_data' and tuple(parts[1:]) in communicated_image: continue

            # If the goal fact itself is in the state, it means the communication is done.
            # But the goal is the *communicated* fact, so we check the communicated sets above.
            # If the goal fact is not in the communicated sets, we need to calculate the cost.

            min_cost_for_goal = float('inf')

            if pred == 'communicated_soil_data':
                waypoint = parts[1]

                # Need to achieve (communicated_soil_data waypoint)
                # This requires (have_soil_analysis rover waypoint) and communication
                # Check if soil sample exists initially at this waypoint
                if waypoint not in self.soil_sample_wps_initial:
                    min_cost_for_goal = float('inf') # Cannot sample if no sample exists
                else:
                    # Find equipped rovers
                    equipped_rovers = [r for r, caps in self.rover_capabilities.items() if 'soil' in caps]
                    if not equipped_rovers:
                         min_cost_for_goal = float('inf') # No rover can do soil analysis
                    else:
                        for rover in equipped_rovers:
                            current_w = rover_locations.get(rover)
                            if current_w is None: continue # Rover location unknown (shouldn't happen in valid states)

                            cost_get_analysis = float('inf')
                            location_after_analysis = current_w # Default location if analysis is already had

                            if (rover, waypoint) in have_soil:
                                cost_get_analysis = 0 # Already have the analysis
                            else:
                                # Need to sample soil
                                # Check if store is full for this rover
                                rover_store = next((s for s, r in self.store_owners.items() if r == rover), None)
                                store_is_full = (rover_store is not None and store_status.get(rover_store) == 'full')

                                nav_cost_to_sample = self.rover_distances.get(rover, {}).get((current_w, waypoint), float('inf'))
                                if nav_cost_to_sample is not float('inf'):
                                    cost_get_analysis = nav_cost_to_sample + (1 if store_is_full else 0) + 1 # Nav + Drop (if needed) + Sample
                                    location_after_analysis = waypoint # Rover is at waypoint after sampling

                            if cost_get_analysis is float('inf'): continue # Cannot get analysis with this rover

                            # Now cost to communicate from location_after_analysis
                            cost_communicate = float('inf')
                            if self.communication_wps:
                                min_nav_to_comm = float('inf')
                                for comm_w in self.communication_wps:
                                    nav_cost = self.rover_distances.get(rover, {}).get((location_after_analysis, comm_w), float('inf'))
                                    min_nav_to_comm = min(min_nav_to_comm, nav_cost)

                                if min_nav_to_comm is not float('inf'):
                                     cost_communicate = min_nav_to_comm + 1 # Nav + Communicate

                            if cost_communicate is not float('inf'):
                                min_cost_for_goal = min(min_cost_for_goal, cost_get_analysis + cost_communicate)


            elif pred == 'communicated_rock_data':
                waypoint = parts[1]

                # Need to achieve (communicated_rock_data waypoint)
                # This requires (have_rock_analysis rover waypoint) and communication
                # Check if rock sample exists initially at this waypoint
                if waypoint not in self.rock_sample_wps_initial:
                    min_cost_for_goal = float('inf') # Cannot sample if no sample exists
                else:
                    # Find equipped rovers
                    equipped_rovers = [r for r, caps in self.rover_capabilities.items() if 'rock' in caps]
                    if not equipped_rovers:
                         min_cost_for_goal = float('inf') # No rover can do rock analysis
                    else:
                        for rover in equipped_rovers:
                            current_w = rover_locations.get(rover)
                            if current_w is None: continue

                            cost_get_analysis = float('inf')
                            location_after_analysis = current_w # Default location if analysis is already had

                            if (rover, waypoint) in have_rock:
                                cost_get_analysis = 0 # Already have the analysis
                            else:
                                # Need to sample rock
                                # Check if store is full for this rover
                                rover_store = next((s for s, r in self.store_owners.items() if r == rover), None)
                                store_is_full = (rover_store is not None and store_status.get(rover_store) == 'full')

                                nav_cost_to_sample = self.rover_distances.get(rover, {}).get((current_w, waypoint), float('inf'))
                                if nav_cost_to_sample is not float('inf'):
                                    cost_get_analysis = nav_cost_to_sample + (1 if store_is_full else 0) + 1 # Nav + Drop (if needed) + Sample
                                    location_after_analysis = waypoint # Rover is at waypoint after sampling

                            if cost_get_analysis is float('inf'): continue # Cannot get analysis with this rover

                            # Now cost to communicate from location_after_analysis
                            cost_communicate = float('inf')
                            if self.communication_wps:
                                min_nav_to_comm = float('inf')
                                for comm_w in self.communication_wps:
                                    nav_cost = self.rover_distances.get(rover, {}).get((location_after_analysis, comm_w), float('inf'))
                                    min_nav_to_comm = min(min_nav_to_comm, nav_cost)

                                if min_nav_to_comm is not float('inf'):
                                     cost_communicate = min_nav_to_comm + 1 # Nav + Communicate

                            if cost_communicate is not float('inf'):
                                min_cost_for_goal = min(min_cost_for_goal, cost_get_analysis + cost_communicate)


            elif pred == 'communicated_image_data':
                objective, mode = parts[1], parts[2]

                # Need to achieve (communicated_image_data objective mode)
                # This requires (have_image rover objective mode) and communication

                # Find rovers with imaging capability and cameras supporting the mode for this objective
                suitable_rover_camera = [] # List of (rover, camera) tuples
                for camera, info in self.camera_info.items():
                    rover = info.get('rover')
                    supports_modes = info.get('supports', set())
                    cal_target = info.get('calibration_target')

                    if (rover is not None and 'imaging' in self.rover_capabilities.get(rover, set()) and
                        mode in supports_modes and cal_target is not None): # Must have a calibration target
                         suitable_rover_camera.append((rover, camera))

                if not suitable_rover_camera:
                    min_cost_for_goal = float('inf') # No rover/camera can take this image
                else:
                    for rover, camera in suitable_rover_camera:
                        current_w = rover_locations.get(rover)
                        if current_w is None: continue

                        cost_get_image = float('inf')
                        location_after_image = current_w # Default location if image is already had

                        if (rover, objective, mode) in have_image:
                            cost_get_image = 0 # Already have the image
                        else:
                            # Need to take the image
                            # Requires calibration and being at a visible_from waypoint

                            # 1. Calibration cost
                            cost_calibrate = float('inf')
                            location_after_calibrate = current_w # Default if already calibrated

                            if (camera, rover) in calibrated_cameras:
                                cost_calibrate = 0 # Already calibrated
                            else:
                                # Need to calibrate
                                cal_wps = self.calibration_wps.get(camera, set())
                                if not cal_wps: continue # Cannot calibrate this camera (no visible wps for target)

                                min_nav_to_cal = float('inf')
                                best_cal_wp = None
                                for cal_w in cal_wps:
                                    nav_cost = self.rover_distances.get(rover, {}).get((current_w, cal_w), float('inf'))
                                    if nav_cost is not float('inf'):
                                        if nav_cost < min_nav_to_cal:
                                            min_nav_to_cal = nav_cost
                                            best_cal_wp = cal_w

                                if best_cal_wp is not None:
                                    cost_calibrate = min_nav_to_cal + 1 # Nav + Calibrate
                                    location_after_calibrate = best_cal_wp # Rover is at this WP after calibrating
                                else:
                                    continue # Cannot reach any calibration waypoint

                            if cost_calibrate is float('inf'): continue # Cannot calibrate

                            # 2. Take Image cost (after calibration)
                            img_wps = self.objective_visibility.get(objective, set())
                            if not img_wps: continue # Cannot image this objective (no visible wps)

                            min_nav_to_img = float('inf')
                            best_img_wp = None
                            for img_w in img_wps:
                                nav_cost = self.rover_distances.get(rover, {}).get((location_after_calibrate, img_w), float('inf'))
                                if nav_cost is not float('inf'):
                                    if nav_cost < min_nav_to_img:
                                        min_nav_to_img = nav_cost
                                        best_img_wp = img_w

                            if best_img_wp is not None:
                                cost_get_image = cost_calibrate + min_nav_to_img + 1 # Nav + Take Image
                                location_after_image = best_img_wp # Rover is at this WP after imaging
                            else:
                                continue # Cannot reach any image waypoint

                        if cost_get_image is float('inf'): continue # Cannot get image with this rover/camera

                        # 3. Communicate cost (after imaging)
                        cost_communicate = float('inf')
                        if self.communication_wps:
                            min_nav_to_comm = float('inf')
                            for comm_w in self.communication_wps:
                                nav_cost = self.rover_distances.get(rover, {}).get((location_after_image, comm_w), float('inf'))
                                min_nav_to_comm = min(min_nav_to_comm, nav_cost)

                            if min_nav_to_comm is not float('inf'):
                                 cost_communicate = min_nav_to_comm + 1 # Nav + Communicate

                        if cost_communicate is not float('inf'):
                            min_cost_for_goal = min(min_cost_for_goal, cost_get_image + cost_communicate)


            # Add the minimum cost found for this goal to the total heuristic
            # If min_cost_for_goal is infinity, it means this specific goal is unreachable
            # by any single rover sequence considered by this heuristic.
            # Adding infinity correctly makes the total heuristic infinity if any goal is unreachable.
            total_cost += min_cost_for_goal

        # Return a large number for infinity, otherwise the calculated cost.
        return total_cost if total_cost != float('inf') else 1000000
