from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque

# Helper functions
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

def bfs(graph, start_node, all_nodes):
    """
    Performs Breadth-First Search to find shortest paths from a start node
    in a graph represented as an adjacency list.
    Returns a dictionary {end_node: distance}.
    """
    distances = {node: float('inf') for node in all_nodes}
    if start_node not in all_nodes:
        return distances # Start node not in the set of nodes

    distances[start_node] = 0
    queue = deque([start_node])
    while queue:
        current_node = queue.popleft()
        current_dist = distances[current_node]
        if current_node in graph:
            for neighbor in graph[current_node]:
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = current_dist + 1
                    queue.append(neighbor)
    return distances

class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve all goal
    conditions. It sums up the estimated costs for each unachieved goal fact
    independently, considering the steps needed (sampling/imaging, calibration,
    navigation, communication). Navigation costs are estimated using precomputed
    shortest paths for each rover.

    # Assumptions
    - The heuristic is non-admissible and designed for greedy best-first search.
    - All necessary equipment (soil/rock/imaging), cameras, and stores exist
      on some rover(s) to achieve the goals.
    - Waypoints required for sampling, imaging, calibration, and communication
      are reachable by equipped rovers in solvable instances.
    - Store capacity is a soft constraint; a drop action is added only if all
      stores of equipped rovers for a task type are full.
    - Calibration is needed for an image goal if no suitable camera is currently
      calibrated. This simplifies the calibration state tracking and assumes one
      calibration action is sufficient per image goal if needed (overestimating
      if one calibration serves multiple images, underestimating if calibration
      is consumed and needs repeating).
    - Communication requires the rover with the data/image to be at a waypoint
      visible from the lander.

    # Heuristic Initialization
    The heuristic precomputes and stores static information from the task:
    - Lists of rovers, waypoints, landers, stores, cameras, modes, objectives.
    - Lander location.
    - Rover capabilities (soil, rock, imaging).
    - Store ownership by rover.
    - Camera ownership by rover and supported modes.
    - Camera calibration targets.
    - Waypoints visible from objectives and calibration targets.
    - Waypoint visibility for communication.
    - Navigation graphs for each rover based on `can_traverse`.
    - All-pairs shortest paths between waypoints for each rover using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic calculates the cost as follows:
    1. Initialize total cost to 0.
    2. Identify the current location of each rover.
    3. Identify the lander location and communication waypoints (visible from lander).
    4. Iterate through each unachieved goal condition:
       - If the goal is `(communicated_soil_data ?w)`:
         - If `(have_soil_analysis ?r ?w)` is not true for any rover:
           - Add 1 for `sample_soil`.
           - Find a soil-equipped rover. Add the shortest path cost from its current location to `?w`.
           - Check if all stores of soil-equipped rovers are full. If so, add 1 for `drop`.
         - Else (`have_soil_analysis` exists):
           - Add 1 for `communicate_soil_data`.
           - Find the rover(s) with the analysis. Add the minimum shortest path cost from their current location to any communication waypoint.
       - If the goal is `(communicated_rock_data ?w)`:
         - Similar logic as soil data, using rock-equipped rovers and `have_rock_analysis`.
       - If the goal is `(communicated_image_data ?o ?m)`:
         - If `(have_image ?r ?o ?m)` is not true for any rover:
           - Add 1 for `take_image`.
           - Find imaging-equipped rovers with cameras supporting mode `?m`.
           - Check if *any* suitable camera is calibrated in the current state. If not:
             - Add 1 for `calibrate`.
             - Find the camera's calibration target and visible waypoints. Add the minimum shortest path cost from a suitable rover's current location to any calibration waypoint.
           - Find waypoints visible from objective `?o`. Add the minimum shortest path cost from a suitable rover's current location to any image waypoint.
         - Else (`have_image` exists):
           - Add 1 for `communicate_image_data`.
           - Find the rover(s) with the image. Add the minimum shortest path cost from their current location to any communication waypoint.
    5. Return the total accumulated cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # --- Extract Objects and Static Information ---
        self.rovers = set()
        self.waypoints = set()
        self.landers = set()
        self.stores = set()
        self.cameras = set()
        self.modes = set()
        self.objectives = set()
        calibration_targets_set = set() # Keep track of calibration targets

        self.lander_location = {} # {lander: waypoint}
        self.rover_capabilities = {} # {rover: set of capabilities}
        self.rover_stores = {} # {rover: set of stores}
        self.rover_cameras = {} # {rover: set of cameras}
        self.camera_modes = {} # {camera: set of modes}
        self.camera_calibration_target = {} # {camera: target} - Assuming 1:1 based on example
        self.objective_visible_waypoints = {} # {objective: set of waypoints}
        self.calibration_target_visible_waypoints = {} # {target: set of waypoints}
        self.waypoint_visibility = set() # {(wp1, wp2)} where wp1 is visible from wp2
        self.rover_navigation_graphs = {} # {rover: {wp: set of neighbors}}

        # Collect all facts to identify objects and static relations
        all_facts = set(task.initial_state) | set(static_facts) | set(self.goals)

        # First pass: Identify all objects by type based on predicate structure
        for fact_str in all_facts:
            parts = get_parts(fact_str)
            if not parts: continue
            pred = parts[0]
            if pred == 'at' and len(parts) == 3:
                 obj, loc = parts[1], parts[2]
                 if obj.startswith('rover'): self.rovers.add(obj)
                 if loc.startswith('waypoint'): self.waypoints.add(loc)
            elif pred == 'at_lander' and len(parts) == 3:
                 lander, loc = parts[1], parts[2]
                 self.landers.add(lander)
                 if loc.startswith('waypoint'): self.waypoints.add(loc)
            elif pred == 'can_traverse' and len(parts) == 4:
                 rover, wp1, wp2 = parts[1], parts[2], parts[3]
                 if rover.startswith('rover'): self.rovers.add(rover)
                 if wp1.startswith('waypoint'): self.waypoints.add(wp1)
                 if wp2.startswith('waypoint'): self.waypoints.add(wp2)
            elif pred in ['equipped_for_soil_analysis', 'equipped_for_rock_analysis', 'equipped_for_imaging'] and len(parts) == 2:
                 rover = parts[1]
                 if rover.startswith('rover'): self.rovers.add(rover)
            elif pred in ['empty', 'full'] and len(parts) == 2:
                 store = parts[1]
                 if store.startswith('store'): self.stores.add(store)
            elif pred == 'have_rock_analysis' and len(parts) == 3:
                 rover, wp = parts[1], parts[2]
                 if rover.startswith('rover'): self.rovers.add(rover)
                 if wp.startswith('waypoint'): self.waypoints.add(wp)
            elif pred == 'have_soil_analysis' and len(parts) == 3:
                 rover, wp = parts[1], parts[2]
                 if rover.startswith('rover'): self.rovers.add(rover)
                 if wp.startswith('waypoint'): self.waypoints.add(wp)
            elif pred == 'calibrated' and len(parts) == 3:
                 camera, rover = parts[1], parts[2]
                 if camera.startswith('camera'): self.cameras.add(camera)
                 if rover.startswith('rover'): self.rovers.add(rover)
            elif pred == 'supports' and len(parts) == 3:
                 camera, mode = parts[1], parts[2]
                 if camera.startswith('camera'): self.cameras.add(camera)
                 self.modes.add(mode) # Add any mode mentioned
            elif pred == 'visible' and len(parts) == 3:
                 wp1, wp2 = parts[1], parts[2]
                 if wp1.startswith('waypoint'): self.waypoints.add(wp1)
                 if wp2.startswith('waypoint'): self.waypoints.add(wp2)
            elif pred == 'have_image' and len(parts) == 4:
                 rover, obj, mode = parts[1], parts[2], parts[3]
                 if rover.startswith('rover'): self.rovers.add(rover)
                 if obj.startswith('objective'): self.objectives.add(obj)
                 self.modes.add(mode)
            elif pred in ['communicated_soil_data', 'communicated_rock_data'] and len(parts) == 2:
                 wp = parts[1]
                 if wp.startswith('waypoint'): self.waypoints.add(wp)
            elif pred == 'communicated_image_data' and len(parts) == 3:
                 obj, mode = parts[1], parts[2]
                 if obj.startswith('objective'): self.objectives.add(obj)
                 self.modes.add(mode)
            elif pred in ['at_soil_sample', 'at_rock_sample'] and len(parts) == 2:
                 wp = parts[1]
                 if wp.startswith('waypoint'): self.waypoints.add(wp)
            elif pred == 'visible_from' and len(parts) == 3:
                 obj_or_target, wp = parts[1], parts[2]
                 # Assume anything visible_from might be an objective or a cal target
                 self.objectives.add(obj_or_target)
                 calibration_targets_set.add(obj_or_target)
                 if wp.startswith('waypoint'): self.waypoints.add(wp)
            elif pred == 'store_of' and len(parts) == 3:
                 store, rover = parts[1], parts[2]
                 if store.startswith('store'): self.stores.add(store)
                 if rover.startswith('rover'): self.rovers.add(rover)
            elif pred == 'calibration_target' and len(parts) == 3:
                 camera, target = parts[1], parts[2]
                 if camera.startswith('camera'): self.cameras.add(camera)
                 calibration_targets_set.add(target) # Explicitly mark as calibration target

        # Initialize dictionaries/graphs for static info based on identified objects
        self.rover_capabilities = {r: set() for r in self.rovers}
        self.rover_stores = {r: set() for r in self.rovers}
        self.rover_cameras = {r: set() for r in self.rovers}
        self.camera_modes = {c: set() for c in self.cameras}
        self.objective_visible_waypoints = {o: set() for o in self.objectives}
        self.calibration_target_visible_waypoints = {t: set() for t in calibration_targets_set}
        self.rover_navigation_graphs = {r: {wp: set() for wp in self.waypoints} for r in self.rovers}


        # Second pass: Populate static data structures from static_facts
        for fact_str in static_facts:
            parts = get_parts(fact_str)
            if not parts: continue
            pred = parts[0]

            if pred == 'at_lander' and len(parts) == 3:
                 lander, loc = parts[1], parts[2]
                 if lander in self.landers: self.lander_location[lander] = loc
            elif pred == 'equipped_for_soil_analysis' and len(parts) == 2:
                 if parts[1] in self.rovers: self.rover_capabilities[parts[1]].add('soil')
            elif pred == 'equipped_for_rock_analysis' and len(parts) == 2:
                 if parts[1] in self.rovers: self.rover_capabilities[parts[1]].add('rock')
            elif pred == 'equipped_for_imaging' and len(parts) == 2:
                 if parts[1] in self.rovers: self.rover_capabilities[parts[1]].add('imaging')
            elif pred == 'store_of' and len(parts) == 3:
                 store, rover = parts[1], parts[2]
                 if rover in self.rovers and store in self.stores: self.rover_stores[rover].add(store)
            elif pred == 'on_board' and len(parts) == 3:
                 camera, rover = parts[1], parts[2]
                 if rover in self.rovers and camera in self.cameras: self.rover_cameras[rover].add(camera)
            elif pred == 'supports' and len(parts) == 3:
                 camera, mode = parts[1], parts[2]
                 if camera in self.cameras and mode in self.modes: self.camera_modes[camera].add(mode)
            elif pred == 'calibration_target' and len(parts) == 3:
                 camera, target = parts[1], parts[2]
                 if camera in self.cameras and target in calibration_targets_set: self.camera_calibration_target[camera] = target
            elif pred == 'visible_from' and len(parts) == 3:
                 obj_or_target, wp = parts[1], parts[2]
                 if wp in self.waypoints:
                     if obj_or_target in self.objectives:
                         self.objective_visible_waypoints[obj_or_target].add(wp)
                     if obj_or_target in calibration_targets_set:
                         self.calibration_target_visible_waypoints[obj_or_target].add(wp)
            elif pred == 'visible' and len(parts) == 3:
                 wp1, wp2 = parts[1], parts[2]
                 if wp1 in self.waypoints and wp2 in self.waypoints: self.waypoint_visibility.add((wp1, wp2))
            elif pred == 'can_traverse' and len(parts) == 4:
                 rover, wp1, wp2 = parts[1], parts[2], parts[3]
                 if rover in self.rovers and wp1 in self.waypoints and wp2 in self.waypoints:
                     self.rover_navigation_graphs[rover][wp1].add(wp2)

        # Precompute shortest paths for each rover
        self.rover_shortest_paths = {}
        for rover in self.rovers:
            self.rover_shortest_paths[rover] = {}
            graph = self.rover_navigation_graphs.get(rover, {})
            for start_wp in self.waypoints:
                 self.rover_shortest_paths[rover][start_wp] = bfs(graph, start_wp, self.waypoints)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        total_cost = 0

        # Get current rover locations
        rover_locations = {}
        for fact_str in state:
            parts = get_parts(fact_str)
            if match(fact_str, 'at', '*', '*'):
                 obj, loc = parts[1], parts[2]
                 if obj in self.rovers:
                     rover_locations[obj] = loc

        # Get lander location (static)
        # Assuming only one lander for simplicity based on examples
        lander = next(iter(self.landers), None) # Get the first lander if any
        lander_wp = self.lander_location.get(lander) if lander else None

        # Find communication waypoints visible from lander location
        comm_wps = {wp1 for wp1, wp2 in self.waypoint_visibility if wp2 == lander_wp} if lander_wp else set()

        # Process unachieved goals
        unachieved_goals = self.goals - state

        for goal_fact_str in unachieved_goals:
            parts = get_parts(goal_fact_str)
            pred = parts[0]

            if pred == 'communicated_soil_data':
                w = parts[1]
                # Check if soil analysis exists for this waypoint
                soil_analysis_exists = any(f'(have_soil_analysis {r} {w})' in state for r in self.rovers)

                if not soil_analysis_exists:
                    # Need to sample soil
                    total_cost += 1 # sample_soil action
                    # Add navigation cost to sample location ?w
                    min_nav_to_sample = float('inf')
                    equipped_rovers = [r for r in self.rovers if 'soil' in self.rover_capabilities.get(r, set())]
                    for r in equipped_rovers:
                        current_loc = rover_locations.get(r)
                        if current_loc and w in self.rover_shortest_paths.get(r, {}).get(current_loc, {}):
                             min_nav_to_sample = min(min_nav_to_sample, self.rover_shortest_paths[r][current_loc][w])

                    if min_nav_to_sample != float('inf'):
                        total_cost += min_nav_to_sample
                    else:
                         # Unreachable sample location for any equipped rover
                         return float('inf') # Indicate unsolvable state

                    # Store cost: Check if all stores of soil-equipped rovers are full
                    all_stores_full = True
                    if equipped_rovers:
                        for r in equipped_rovers:
                            # Assuming one store per rover, get the first one
                            store = next(iter(self.rover_stores.get(r, set())), None)
                            if store and f'(empty {store})' in state:
                                all_stores_full = False
                                break
                        if all_stores_full:
                             total_cost += 1 # drop action needed

                # Communication step is always needed if goal not achieved
                total_cost += 1 # communicate_soil_data action
                # Add navigation cost to communication waypoint
                min_nav_to_comm = float('inf')
                # Find rovers that *could* communicate this data (either have it or are equipped to get it)
                potential_rovers = [r for r in self.rovers if f'(have_soil_analysis {r} {w})' in state]
                if not potential_rovers: # If no one has it yet, assume an equipped rover will get it
                    potential_rovers = [r for r in self.rovers if 'soil' in self.rover_capabilities.get(r, set())]

                if not potential_rovers: # No rover can ever get/communicate this
                     return float('inf')

                for r in potential_rovers:
                    current_loc = rover_locations.get(r)
                    if current_loc and comm_wps:
                        for comm_wp in comm_wps:
                            if comm_wp in self.rover_shortest_paths.get(r, {}).get(current_loc, {}):
                                 min_nav_to_comm = min(min_nav_to_comm, self.rover_shortest_paths[r][current_loc][comm_wp])

                if min_nav_to_comm != float('inf'):
                    total_cost += min_nav_to_comm
                elif comm_wps: # If there are communication waypoints but none are reachable
                     return float('inf') # Unreachable communication location
                # If comm_wps is empty, communication is impossible. Assume solvable instances have comm_wps.


            elif pred == 'communicated_rock_data':
                w = parts[1]
                # Check if rock analysis exists for this waypoint
                rock_analysis_exists = any(f'(have_rock_analysis {r} {w})' in state for r in self.rovers)

                if not rock_analysis_exists:
                    # Need to sample rock
                    total_cost += 1 # sample_rock action
                    # Add navigation cost to sample location ?w
                    min_nav_to_sample = float('inf')
                    equipped_rovers = [r for r in self.rovers if 'rock' in self.rover_capabilities.get(r, set())]
                    for r in equipped_rovers:
                        current_loc = rover_locations.get(r)
                        if current_loc and w in self.rover_shortest_paths.get(r, {}).get(current_loc, {}):
                             min_nav_to_sample = min(min_nav_to_sample, self.rover_shortest_paths[r][current_loc][w])

                    if min_nav_to_sample != float('inf'):
                        total_cost += min_nav_to_sample
                    else:
                         return float('inf') # Unreachable sample location

                    # Store cost: Check if all stores of rock-equipped rovers are full
                    all_stores_full = True
                    if equipped_rovers:
                        for r in equipped_rovers:
                            store = next(iter(self.rover_stores.get(r, set())), None)
                            if store and f'(empty {store})' in state:
                                all_stores_full = False
                                break
                        if all_stores_full:
                             total_cost += 1 # drop action needed

                # Communication step is always needed if goal not achieved
                total_cost += 1 # communicate_rock_data action
                # Add navigation cost to communication waypoint
                min_nav_to_comm = float('inf')
                potential_rovers = [r for r in self.rovers if f'(have_rock_analysis {r} {w})' in state]
                if not potential_rovers:
                    potential_rovers = [r for r in self.rovers if 'rock' in self.rover_capabilities.get(r, set())]

                if not potential_rovers:
                     return float('inf')

                for r in potential_rovers:
                    current_loc = rover_locations.get(r)
                    if current_loc and comm_wps:
                        for comm_wp in comm_wps:
                            if comm_wp in self.rover_shortest_paths.get(r, {}).get(current_loc, {}):
                                 min_nav_to_comm = min(min_nav_to_comm, self.rover_shortest_paths[r][current_loc][comm_wp])

                if min_nav_to_comm != float('inf'):
                    total_cost += min_nav_to_comm
                elif comm_wps:
                     return float('inf') # Unreachable communication location


            elif pred == 'communicated_image_data':
                o, m = parts[1], parts[2]
                # Check if image exists for this objective and mode
                image_exists = any(f'(have_image {r} {o} {m})' in state for r in self.rovers)

                # Find suitable rovers/cameras for this image task
                suitable_rovers_cameras = [] # [(rover, camera)]
                for r in self.rovers:
                    if 'imaging' in self.rover_capabilities.get(r, set()):
                        for i in self.rover_cameras.get(r, set()):
                            if m in self.camera_modes.get(i, set()):
                                suitable_rovers_cameras.append((r, i))

                if not suitable_rovers_cameras:
                     # No rover/camera can take this image
                     return float('inf') # Unsolvable

                if not image_exists:
                    # Need to take image
                    total_cost += 1 # take_image action

                    # Calibration cost: If *any* suitable camera is *not* calibrated, assume one calibration is needed for *this specific image task*.
                    any_suitable_calibrated = any(f'(calibrated {i} {r})' in state for r, i in suitable_rovers_cameras)

                    if not any_suitable_calibrated:
                        total_cost += 1 # calibrate action
                        # Navigation to calibration waypoint
                        min_nav_to_cal = float('inf')
                        # Find nearest calibration waypoint for a suitable rover/camera
                        for r, i in suitable_rovers_cameras:
                            cal_target = self.camera_calibration_target.get(i)
                            if cal_target:
                                cal_wps = self.calibration_target_visible_waypoints.get(cal_target, set())
                                current_loc = rover_locations.get(r)
                                if current_loc and cal_wps:
                                    for cal_wp in cal_wps:
                                        if cal_wp in self.rover_shortest_paths.get(r, {}).get(current_loc, {}):
                                            min_nav_to_cal = min(min_nav_to_cal, self.rover_shortest_paths[r][current_loc][cal_wp])

                        if min_nav_to_cal != float('inf'):
                            total_cost += min_nav_to_cal
                        else:
                             return float('inf') # Unreachable calibration location

                    # Navigation to image waypoint
                    min_nav_to_image = float('inf')
                    img_wps = self.objective_visible_waypoints.get(o, set())
                    if img_wps:
                        # Find nearest image waypoint for any suitable rover
                        for r, i in suitable_rovers_cameras:
                            current_loc = rover_locations.get(r)
                            if current_loc:
                                for img_wp in img_wps:
                                    if img_wp in self.rover_shortest_paths.get(r, {}).get(current_loc, {}):
                                        min_nav_to_image = min(min_nav_to_image, self.rover_shortest_paths[r][current_loc][img_wp])

                    if min_nav_to_image != float('inf'):
                        total_cost += min_nav_to_image
                    elif img_wps:
                         return float('inf') # Unreachable image location


                # Communication step is always needed if goal not achieved
                total_cost += 1 # communicate_image_data action
                # Add navigation cost to communication waypoint
                min_nav_to_comm = float('inf')
                potential_rovers = [r for r in self.rovers if f'(have_image {r} {o} {m})' in state]
                if not potential_rovers: # If no one has it yet, assume a suitable rover will get it
                    potential_rovers = [r for r, i in suitable_rovers_cameras] # Any rover capable of taking the image

                if not potential_rovers:
                     return float('inf')

                for r in potential_rovers:
                    current_loc = rover_locations.get(r)
                    if current_loc and comm_wps:
                        for comm_wp in comm_wps:
                            if comm_wp in self.rover_shortest_paths.get(r, {}).get(current_loc, {}):
                                 min_nav_to_comm = min(min_nav_to_comm, self.rover_shortest_paths[r][current_loc][comm_wp])

                if min_nav_to_comm != float('inf'):
                    total_cost += min_nav_to_comm
                elif comm_wps:
                     return float('inf') # Unreachable communication location


        # Heuristic is 0 if all goals are achieved
        return total_cost

