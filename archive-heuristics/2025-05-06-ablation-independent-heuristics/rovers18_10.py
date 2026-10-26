# Imports
from heuristics.heuristic_base import Heuristic
from task import Task # Although not strictly used, good for type hinting
from collections import deque
import math # For float('inf')

# Helper function to parse PDDL fact strings
def parse_fact(fact_string):
    """Parses a PDDL fact string into a tuple."""
    # Remove parentheses and split by space
    parts = fact_string[1:-1].split()
    return tuple(parts)

# Helper function for Breadth-First Search
def bfs(start_node, graph):
    """
    Performs BFS to find shortest paths from start_node in a graph.
    Returns a dictionary mapping reachable nodes to their distance from start_node.
    """
    distances = {node: float('inf') for node in graph}
    distances[start_node] = 0
    queue = deque([start_node])

    while queue:
        current_node = queue.popleft()

        if current_node in graph: # Handle nodes with no outgoing edges
            for neighbor in graph[current_node]:
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)
    return distances

# The heuristic class
class roversHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Rovers domain.

    Summary:
    Estimates the cost to reach the goal by summing the minimum estimated costs
    for each unachieved goal fact. The cost for each goal fact is estimated
    by finding the minimum cost across all suitable rovers (and cameras for image goals)
    to first acquire the necessary data/image (if not already possessed) and then
    communicate it to a lander. Navigation costs are precomputed using BFS.

    Assumptions:
    - Waypoint visibility is symmetric (based on examples).
    - Rover traversal is symmetric if specified as bidirectional in static facts (based on examples).
    - A rover can drop a sample anywhere if its store is full (cost 1).
    - Calibration state is consumed by taking an image.
    - The heuristic ignores potential positive interactions (e.g., sharing navigation paths,
      calibrating once for multiple images) and most negative interactions (e.g., store becoming full,
      camera becoming uncalibrated), except for the immediate need for drop/calibration.
    - If a required resource (like a soil/rock sample at a specific waypoint, or a visible waypoint
      for an objective/calibration target) is missing based on the initial state and static facts,
      the corresponding goal component is considered unreachable (infinite cost).
    - If no communication waypoint is reachable from necessary locations, the goal component is infinite.

    Heuristic Initialization:
    - Stores the task object and goals.
    - Parses static facts to identify lander locations, rover capabilities, store ownership,
      waypoint visibility, rover traversal capabilities, camera properties (on-board, modes,
      calibration targets), and objective visibility. Collects all objects by type.
    - Builds a navigation graph for each rover based on static can_traverse and visible facts.
      Assumes symmetric traversal if both (can_traverse r w1 w2) and (can_traverse r w2 w1)
      are present, and symmetric visibility if both (visible w1 w2) and (visible w2 w1) are present.
      The navigation graph edge (w1 -> w2) exists if (can_traverse r w1 w2) AND (visible w1 w2).
    - Precomputes all-pairs shortest paths for each rover using BFS on their respective
      navigation graphs.
    - Identifies communication waypoints visible from each lander.

    Step-By-Step Thinking for Computing Heuristic:
    1. Check if the current state is a goal state using `self.task.goal_reached(state)`. If yes, return 0.
    2. Extract dynamic information from the current state: rover locations, full stores,
       locations of soil/rock samples, data/images held by rovers, and calibrated cameras.
    3. Initialize the total heuristic value `h` to 0.
    4. Iterate through each goal fact string in `self.goals`.
    5. If a goal fact string is already present in the current state, skip it.
    6. For an unachieved goal fact string, parse it to get the predicate and arguments.
    7. Initialize the minimum cost for this specific goal fact (`min_goal_cost`) to infinity.
    8. Based on the predicate:
        - If `communicated_soil_data ?w`: Iterate through all rovers equipped for soil analysis.
            - Get the rover's current location.
            - **If the rover already has `(have_soil_analysis r w)`:** Calculate the cost to communicate: minimum navigation cost from the rover's current location to any waypoint visible from any lander, plus 1 for communication. Update `min_goal_cost`.
            - **If the rover does not have the data and `(at_soil_sample w)` is true:** Calculate the cost to acquire the data: navigation cost from current location to `w`, plus 1 for sampling, plus 1 if the rover's store is full (for dropping). If navigation is impossible, acquisition cost is infinity. If acquisition cost is finite, calculate the cost to communicate from `?w`: minimum navigation cost from `?w` to any waypoint visible from any lander, plus 1 for communication. If navigation is impossible, communication cost is infinity. If both costs are finite, sum them and update `min_goal_cost`.
            - If `(at_soil_sample w)` is false and the rover doesn't have the data, this rover cannot acquire it by sampling; this path is infinite.
    - If `communicated_rock_data ?w`: Similar logic as soil data, using rock analysis capabilities, rock samples, and `have_rock_analysis`.
    - If `communicated_image_data ?o ?m`: Iterate through all rovers equipped for imaging. For each such rover, iterate through its cameras supporting mode `?m`.
        - Get the rover's current location.
        - **If the rover/camera already has `(have_image r o m)`:** Calculate the cost to communicate: minimum navigation cost from the rover's current location to any waypoint visible from any lander, plus 1 for communication. Update `min_goal_cost`.
        - **If the rover/camera does not have the image:** Find waypoints visible from `?o` (`image_wps`). If calibration is needed (`(calibrated camera rover)` is false), find waypoints visible from the calibration target (`calib_wps`). Calculate the cost to acquire the image: minimum navigation cost from current location to a `calib_wp` then to an `image_wp`, plus 1 for calibrate and 1 for take_image. If calibration is not needed, calculate minimum navigation cost from current location to an `image_wp`, plus 1 for take_image. If required waypoints don't exist or navigation is impossible, acquisition cost is infinity. If acquisition cost is finite, calculate the cost to communicate from an `image_wp`: minimum navigation cost from any `image_wp` to any waypoint visible from any lander, plus 1 for communication. If navigation is impossible, communication cost is infinity. If both costs are finite, sum them and update `min_goal_cost`.
    9. Add `min_goal_cost` to the total heuristic value `h`. If `min_goal_cost` was infinity for any goal, `h` becomes infinity.
    10. After iterating through all goal facts, return `h`.
    """

    def __init__(self, task: Task):
        super().__init__()
        self.task = task
        self.goals = task.goals
        self.static_facts = task.static

        # Data structures to store parsed static info
        self.lander_location = {} # lander -> waypoint
        self.equipped_soil = set() # rovers
        self.equipped_rock = set() # rovers
        self.equipped_imaging = set() # rovers
        self.store_owner = {} # store -> rover
        self.visible_wps = {} # wp -> set of visible wps
        self.rover_traversal_graph = {} # rover -> {wp -> set of neighbor wps}
        self.camera_on_board = {} # camera -> rover
        self.camera_modes = {} # camera -> set of modes
        self.camera_calib_target = {} # camera -> objective
        self.objective_visible_from = {} # objective -> set of wps

        # Collect all objects by type
        self.rovers = set()
        self.waypoints = set()
        self.stores = set()
        self.cameras = set()
        self.modes = set()
        self.objectives = set()
        self.landers = set()

        # Parse static facts and populate structures
        for fact_str in self.static_facts:
            fact = parse_fact(fact_str)
            pred = fact[0]
            if pred == 'at_lander':
                lander, wp = fact[1], fact[2]
                self.landers.add(lander)
                self.waypoints.add(wp)
                self.lander_location[lander] = wp
            elif pred == 'equipped_for_soil_analysis':
                rover = fact[1]
                self.rovers.add(rover)
                self.equipped_soil.add(rover)
            elif pred == 'equipped_for_rock_analysis':
                rover = fact[1]
                self.rovers.add(rover)
                self.equipped_rock.add(rover)
            elif pred == 'equipped_for_imaging':
                rover = fact[1]
                self.rovers.add(rover)
                self.equipped_imaging.add(rover)
            elif pred == 'store_of':
                store, rover = fact[1], fact[2]
                self.stores.add(store)
                self.rovers.add(rover)
                self.store_owner[store] = rover
            elif pred == 'visible':
                wp1, wp2 = fact[1], fact[2]
                self.waypoints.add(wp1)
                self.waypoints.add(wp2)
                self.visible_wps.setdefault(wp1, set()).add(wp2)
                # Assuming symmetric visibility based on examples
                self.visible_wps.setdefault(wp2, set()).add(wp1)
            elif pred == 'can_traverse':
                rover, wp1, wp2 = fact[1], fact[2], fact[3]
                self.rovers.add(rover)
                self.waypoints.add(wp1)
                self.waypoints.add(wp2)
                # Store directed traversal graph first
                self.rover_traversal_graph.setdefault(rover, {}).setdefault(wp1, set()).add(wp2)
                # Assuming symmetric traversal based on example instances
                self.rover_traversal_graph.setdefault(rover, {}).setdefault(wp2, set()).add(wp1)
            elif pred == 'on_board':
                camera, rover = fact[1], fact[2]
                self.cameras.add(camera)
                self.rovers.add(rover)
                self.camera_on_board[camera] = rover
            elif pred == 'supports':
                camera, mode = fact[1], fact[2]
                self.cameras.add(camera)
                self.modes.add(mode)
                self.camera_modes.setdefault(camera, set()).add(mode)
            elif pred == 'calibration_target':
                camera, objective = fact[1], fact[2]
                self.cameras.add(camera)
                self.objectives.add(objective)
                self.camera_calib_target[camera] = objective
            elif pred == 'visible_from':
                objective, wp = fact[1], fact[2]
                self.objectives.add(objective)
                self.waypoints.add(wp)
                self.objective_visible_from.setdefault(objective, set()).add(wp)

        # Ensure all waypoints are in traversal graph nodes for each rover
        for rover in self.rovers:
            self.rover_traversal_graph.setdefault(rover, {})
            for wp in self.waypoints:
                 self.rover_traversal_graph[rover].setdefault(wp, set())

        # Build the actual navigation graph for each rover using can_traverse and visible
        self.rover_navigation_graph = {} # rover -> {wp -> set of neighbor wps}
        for rover in self.rovers:
            self.rover_navigation_graph[rover] = {}
            for wp1 in self.waypoints:
                self.rover_navigation_graph[rover][wp1] = set()
                if wp1 in self.rover_traversal_graph.get(rover, {}):
                     for wp2 in self.rover_traversal_graph[rover][wp1]:
                         if wp2 in self.visible_wps.get(wp1, set()):
                             self.rover_navigation_graph[rover][wp1].add(wp2)

        # Precompute all-pairs shortest paths for each rover
        self.rover_paths = {} # rover -> {start_wp -> {end_wp -> distance}}
        for rover in self.rovers:
            self.rover_paths[rover] = {}
            for start_wp in self.waypoints:
                self.rover_paths[rover][start_wp] = bfs(start_wp, self.rover_navigation_graph[rover])

        # Identify communication waypoints for each lander
        self.comm_wps = {} # lander -> set of comm wps
        for lander, lander_wp in self.lander_location.items():
            self.comm_wps[lander] = self.visible_wps.get(lander_wp, set())

        # Check if any lander exists and has communication points
        self.has_comm_points = any(len(wps) > 0 for wps in self.comm_wps.values())


    def __call__(self, node):
        state = node.state

        # Check if goal is reached
        if self.task.goal_reached(state):
            return 0

        # Extract dynamic state information
        rover_location = {} # rover -> waypoint
        full_stores = set() # stores
        soil_samples_at = set() # waypoints
        rock_samples_at = set() # waypoints
        have_soil_data = set() # (rover, waypoint)
        have_rock_data = set() # (rover, waypoint)
        have_image_data = set() # (rover, objective, mode)
        calibrated_cameras = set() # (camera, rover)

        for fact_str in state:
            fact = parse_fact(fact_str)
            pred = fact[0]
            if pred == 'at':
                rover, wp = fact[1], fact[2]
                rover_location[rover] = wp
            elif pred == 'full':
                store = fact[1]
                full_stores.add(store)
            elif pred == 'at_soil_sample':
                wp = fact[1]
                soil_samples_at.add(wp)
            elif pred == 'at_rock_sample':
                wp = fact[1]
                rock_samples_at.add(wp)
            elif pred == 'have_soil_analysis':
                rover, wp = fact[1], fact[2]
                have_soil_data.add((rover, wp))
            elif pred == 'have_rock_analysis':
                rover, wp = fact[1], fact[2]
                have_rock_data.add((rover, wp))
            elif pred == 'have_image':
                rover, obj, mode = fact[1], fact[2], fact[3]
                have_image_data.add((rover, obj, mode))
            elif pred == 'calibrated':
                camera, rover = fact[1], fact[2]
                calibrated_cameras.add((camera, rover))

        h = 0

        # Check each goal fact
        for goal_fact_str in self.goals:

            if goal_fact_str in state:
                continue # Goal already achieved

            goal_fact = parse_fact(goal_fact_str)
            pred = goal_fact[0]

            min_goal_cost = float('inf')

            if pred == 'communicated_soil_data':
                wp = goal_fact[1]

                for rover in self.rovers:
                    if rover not in self.equipped_soil: continue

                    curr_wp = rover_location.get(rover)
                    if curr_wp is None: continue # Should not happen in valid states

                    # Option 1: Already have data
                    if (rover, wp) in have_soil_data:
                        min_nav_to_comm = float('inf')
                        if self.has_comm_points:
                            for lander, comm_wps in self.comm_wps.items():
                                for comm_wp in comm_wps:
                                     nav_cost = self.rover_paths[rover][curr_wp].get(comm_wp, float('inf'))
                                     min_nav_to_comm = min(min_nav_to_comm, nav_cost)
                        if min_nav_to_comm != float('inf'):
                            min_goal_cost = min(min_goal_cost, min_nav_to_comm + 1) # +1 communicate

                    # Option 2: Need to sample
                    elif wp in soil_samples_at:
                         # Cost to sample: navigate to wp + (drop if store full) + sample
                         nav_cost_to_sample = self.rover_paths[rover][curr_wp].get(wp, float('inf'))
                         if nav_cost_to_sample == float('inf'): continue

                         store = None
                         for s, r_owner in self.store_owner.items():
                             if r_owner == rover:
                                 store = s
                                 break
                         if store is None: continue # Should not happen in valid problems

                         store_full = (store in full_stores)
                         store_cost = 1 if store_full else 0

                         cost_to_get_data = nav_cost_to_sample + store_cost + 1 # +1 for sample action

                         # Cost to communicate from wp (where sample was taken)
                         min_nav_to_comm = float('inf')
                         if self.has_comm_points:
                             for lander, comm_wps in self.comm_wps.items():
                                 for comm_wp in comm_wps:
                                      nav_cost = self.rover_paths[rover][wp].get(comm_wp, float('inf'))
                                      min_nav_to_comm = min(min_nav_to_comm, nav_cost)

                         if min_nav_to_comm != float('inf'):
                             total_cost_for_rover = cost_to_get_data + min_nav_to_comm + 1 # +1 for communicate action
                             min_goal_cost = min(min_goal_cost, total_cost_for_rover)

            elif pred == 'communicated_rock_data':
                wp = goal_fact[1]

                for rover in self.rovers:
                    if rover not in self.equipped_rock: continue

                    curr_wp = rover_location.get(rover)
                    if curr_wp is None: continue

                    # Option 1: Already have data
                    if (rover, wp) in have_rock_data:
                        min_nav_to_comm = float('inf')
                        if self.has_comm_points:
                            for lander, comm_wps in self.comm_wps.items():
                                for comm_wp in comm_wps:
                                     nav_cost = self.rover_paths[rover][curr_wp].get(comm_wp, float('inf'))
                                     min_nav_to_comm = min(min_nav_to_comm, nav_cost)
                        if min_nav_to_comm != float('inf'):
                            min_goal_cost = min(min_goal_cost, min_nav_to_comm + 1) # +1 communicate

                    # Option 2: Need to sample
                    elif wp in rock_samples_at:
                         # Cost to sample: navigate to wp + (drop if store full) + sample
                         nav_cost_to_sample = self.rover_paths[rover][curr_wp].get(wp, float('inf'))
                         if nav_cost_to_sample == float('inf'): continue

                         store = None
                         for s, r_owner in self.store_owner.items():
                             if r_owner == rover:
                                 store = s
                                 break
                         if store is None: continue

                         store_full = (store in full_stores)
                         store_cost = 1 if store_full else 0

                         cost_to_get_data = nav_cost_to_sample + store_cost + 1 # +1 for sample action

                         # Cost to communicate from wp (where sample was taken)
                         min_nav_to_comm = float('inf')
                         if self.has_comm_points:
                             for lander, comm_wps in self.comm_wps.items():
                                 for comm_wp in comm_wps:
                                      nav_cost = self.rover_paths[rover][wp].get(comm_wp, float('inf'))
                                      min_nav_to_comm = min(min_nav_to_comm, nav_cost)

                         if min_nav_to_comm != float('inf'):
                             total_cost_for_rover = cost_to_get_data + min_nav_to_comm + 1 # +1 for communicate action
                             min_goal_cost = min(min_goal_cost, total_cost_for_rover)

            elif pred == 'communicated_image_data':
                obj, mode = goal_fact[1], goal_fact[2]

                for rover in self.rovers:
                    if rover not in self.equipped_imaging: continue

                    curr_wp = rover_location.get(rover)
                    if curr_wp is None: continue

                    suitable_cameras = [
                        cam for cam in self.cameras
                        if self.camera_on_board.get(cam) == rover and mode in self.camera_modes.get(cam, set())
                    ]
                    if not suitable_cameras: continue

                    for camera in suitable_cameras:
                        # Option 1: Already have image
                        if (rover, obj, mode) in have_image_data:
                            min_nav_to_comm = float('inf')
                            if self.has_comm_points:
                                for lander, comm_wps in self.comm_wps.items():
                                    for comm_wp in comm_wps:
                                         nav_cost = self.rover_paths[rover][curr_wp].get(comm_wp, float('inf'))
                                         min_nav_to_comm = min(min_nav_to_comm, nav_cost)
                            if min_nav_to_comm != float('inf'):
                                min_goal_cost = min(min_goal_cost, min_nav_to_comm + 1) # +1 communicate

                        # Option 2: Need to take image
                        else:
                            image_wps = self.objective_visible_from.get(obj, set())
                            if not image_wps: continue

                            calib_needed = (camera, rover) not in calibrated_cameras
                            cost_to_get_image = float('inf')

                            if calib_needed:
                                calib_target = self.camera_calib_target.get(camera)
                                if calib_target is None: continue

                                calib_wps = self.objective_visible_from.get(calib_target, set())
                                if not calib_wps: continue

                                min_path_cost = float('inf')
                                for cwp in calib_wps:
                                    nav_curr_to_cwp = self.rover_paths[rover][curr_wp].get(cwp, float('inf'))
                                    if nav_curr_to_cwp == float('inf'): continue
                                    for iwp in image_wps:
                                        nav_cwp_to_iwp = self.rover_paths[rover][cwp].get(iwp, float('inf'))
                                        if nav_cwp_to_iwp == float('inf'): continue
                                        min_path_cost = min(min_path_cost, nav_curr_to_cwp + nav_cwp_to_iwp)

                                if min_path_cost != float('inf'):
                                     cost_to_get_image = min_path_cost + 2 # +1 calibrate, +1 take_image
                                # else: cost_to_get_image remains infinity

                            else: # Already calibrated
                                min_path_cost = float('inf')
                                for iwp in image_wps:
                                    nav_curr_to_iwp = self.rover_paths[rover][curr_wp].get(iwp, float('inf'))
                                    min_path_cost = min(min_path_cost, nav_curr_to_iwp)

                                if min_path_cost != float('inf'):
                                     cost_to_get_image = min_path_cost + 1 # +1 take_image
                                # else: cost_to_get_image remains infinity

                            if cost_to_get_image == float('inf'): continue

                            # Cost to communicate from an image waypoint (one of image_wps)
                            min_nav_image_to_comm = float('inf')
                            if self.has_comm_points:
                                for iwp in image_wps: # Image is taken at one of these wps
                                    for lander, comm_wps in self.comm_wps.items():
                                        for comm_wp in comm_wps:
                                            nav_cost = self.rover_paths[rover][iwp].get(comm_wp, float('inf'))
                                            min_nav_image_to_comm = min(min_nav_image_to_comm, nav_cost)

                            if min_nav_image_to_comm != float('inf'):
                                total_cost_for_rover_camera = cost_to_get_image + min_nav_image_to_comm + 1 # +1 communicate action
                                min_goal_cost = min(min_goal_cost, total_cost_for_rover_camera)


            # Add the minimum cost for this goal to the total heuristic
            h += min_goal_cost

        return h
