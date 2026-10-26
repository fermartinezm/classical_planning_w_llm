from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential nested structures if needed, but simple rovers facts are flat.
    # For now, assume simple space-separated parts.
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

def bfs_distance(graph, start_node, goal_nodes):
    """
    Performs Breadth-First Search to find the shortest distance from a start_node
    to any node in the goal_nodes set within a given graph.

    Args:
        graph: An adjacency list representation (dict mapping node to list of neighbors).
        start_node: The starting node for the search.
        goal_nodes: A set of target nodes.

    Returns:
        The shortest distance (number of edges) to any goal node, or float('inf') if
        no goal node is reachable.
    """
    if start_node in goal_nodes:
        return 0
    queue = deque([(start_node, 0)])
    visited = {start_node}
    while queue:
        current_node, dist = queue.popleft()
        if current_node in graph:
            for neighbor in graph[current_node]:
                if neighbor in goal_nodes:
                    return dist + 1
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
    return float('inf') # Unreachable

class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to satisfy all
    unmet goal conditions. It calculates the minimum cost for each individual
    unmet goal (communicating soil, rock, or image data) assuming it can be
    achieved independently, and sums these minimum costs. The cost for each
    goal involves navigation steps (estimated by BFS) and specific actions
    like sampling, calibrating, taking images, and communicating.

    # Assumptions
    - The cost of each action (navigate, sample, drop, calibrate, take_image, communicate) is 1.
    - Navigation cost between waypoints for a specific rover is the shortest path length
      in the graph defined by the rover's `can_traverse` predicates.
    - Goals are independent for heuristic calculation (ignores potential conflicts or synergies,
      e.g., multiple goals requiring the same sample or image).
    - A rover's store needs to be empty to sample. If full, a drop action is needed.
    - An imaging camera needs to be calibrated before taking an image. Taking an image
      uncalibrates the camera.
    - Communication requires the rover to be at a waypoint visible from the lander's location.
    - If a goal requires sampling a soil/rock sample that is no longer `at_soil_sample`
      or `at_rock_sample` in the current state, it is considered unreachable by this
      heuristic (returns infinity for that goal component). This simplifies the heuristic
      by not considering complex sequences like dropping samples to re-sample.

    # Heuristic Initialization
    The heuristic pre-processes static information from the task definition:
    - Identifies the lander's location and waypoints visible from it.
    - Maps rovers to their capabilities (soil, rock, imaging) and stores.
    - Stores camera information (on-board rover, supported modes, calibration target).
    - Maps objectives to waypoints they are visible from.
    - Builds navigation graphs (adjacency lists) for each rover based on `can_traverse` facts.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic calculates the sum of estimated costs for each
    goal fact that is not yet satisfied in the state.

    For each unmet goal:
    1.  **Identify Goal Type:** Determine if it's a soil, rock, or image communication goal.
    2.  **Check if Data is Already Collected:**
        -   If the required data (soil sample, rock sample, or image) is already collected by *any* rover:
            -   The cost is 1 (for communication) plus the minimum navigation cost for a rover holding the data to reach *any* waypoint visible from the lander.
    3.  **If Data Needs Collection:**
        -   **Soil/Rock Goal `(communicated_soil_data ?w)` or `(communicated_rock_data ?w)`:**
            -   Find a rover capable of the required analysis (`equipped_for_soil_analysis` or `equipped_for_rock_analysis`).
            -   If no such rover exists, the goal is unreachable (cost is infinity).
            -   Otherwise, find the minimum cost over all capable rovers:
                -   Cost = 1 (sample) + 1 (communicate).
                -   Add 1 if the rover's store is currently full (for the `drop` action needed before sampling).
                -   Add the navigation cost from the rover's current location to the sample waypoint (`?w`).
                -   Add the navigation cost from the sample waypoint (`?w`) to *any* waypoint visible from the lander.
                -   If the sample is no longer at `?w` in the current state, this path is considered blocked (cost is infinity for this rover).
            -   The cost for this goal is the minimum cost found among all capable rovers.
        -   **Image Goal `(communicated_image_data ?o ?m)`:**
            -   Find a rover capable of imaging (`equipped_for_imaging`) that has a camera supporting mode `?m`.
            -   If no such rover/camera exists, the goal is unreachable (cost is infinity).
            -   Otherwise, find the minimum cost over all suitable rover/camera combinations:
                -   Cost = 1 (take_image) + 1 (communicate).
                -   Find a waypoint `?img_wp` from which objective `?o` is visible. If none exist, this path is blocked (cost is infinity). Choose the one closest to the rover's current location.
                -   Add the navigation cost from the rover's current location to `?img_wp`.
                -   Add the navigation cost from `?img_wp` to *any* waypoint visible from the lander.
                -   If the camera is *not* currently calibrated:
                    -   Find the camera's calibration target `?t`.
                    -   Find a waypoint `?cal_wp` from which `?t` is visible. If none exist, this path is blocked (cost is infinity). Choose the one closest to the rover's current location.
                    -   Add 1 (calibrate).
                    -   Add the navigation cost from the rover's current location to `?cal_wp`.
                    -   Add the navigation cost from `?cal_wp` to `?img_wp`.
                    -   (Note: If calibrated, the navigation cost is just to `?img_wp`).
            -   The cost for this goal is the minimum cost found among all suitable rover/camera combinations.
    4.  **Sum Costs:** The total heuristic value is the sum of the minimum costs calculated for each unmet goal. If any goal is unreachable (cost is infinity), the total heuristic is infinity.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and static facts.
        """
        self.goals = task.goals
        static_facts = task.static

        self.lander_location = None
        self.lander_visible_waypoints = set()
        self.rover_capabilities = {} # rover -> set of capabilities ('soil', 'rock', 'imaging')
        self.rover_stores = {} # rover -> store
        self.camera_info = {} # camera -> {'on_board_rover': r, 'supported_modes': set(), 'calibration_target': t}
        self.objective_visibility = {} # objective -> set of waypoints
        self.calibration_targets = {} # camera -> objective
        self.rover_traverse_graphs = {} # rover -> waypoint -> list of waypoints

        # First pass to identify objects and basic relations
        for fact in static_facts:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == 'at_lander':
                lander, wp = parts[1:]
                self.lander_location = wp
            elif pred == 'equipped_for_soil_analysis':
                rover = parts[1]
                self.rover_capabilities.setdefault(rover, set()).add('soil')
            elif pred == 'equipped_for_rock_analysis':
                rover = parts[1]
                self.rover_capabilities.setdefault(rover, set()).add('rock')
            elif pred == 'equipped_for_imaging':
                rover = parts[1]
                self.rover_capabilities.setdefault(rover, set()).add('imaging')
            elif pred == 'store_of':
                store, rover = parts[1:]
                self.rover_stores[rover] = store
            elif pred == 'on_board':
                camera, rover = parts[1:]
                self.camera_info.setdefault(camera, {}).update({'on_board_rover': rover})
            elif pred == 'supports':
                camera, mode = parts[1:]
                self.camera_info.setdefault(camera, {}).setdefault('supported_modes', set()).add(mode)
            elif pred == 'calibration_target':
                camera, target_obj = parts[1:]
                self.camera_info.setdefault(camera, {}).update({'calibration_target': target_obj})
                self.calibration_targets[camera] = target_obj
            elif pred == 'visible_from':
                obj, wp = parts[1:]
                self.objective_visibility.setdefault(obj, set()).add(wp)
            elif pred == 'can_traverse':
                rover, y, z = parts[1:]
                if rover not in self.rover_traverse_graphs:
                    self.rover_traverse_graphs[rover] = {}
                if y not in self.rover_traverse_graphs[rover]:
                    self.rover_traverse_graphs[rover][y] = []
                self.rover_traverse_graphs[rover][y].append(z)

        # Compute lander visible waypoints
        if self.lander_location:
             for fact in static_facts:
                parts = get_parts(fact)
                if parts[0] == 'visible':
                    wp1, wp2 = parts[1:]
                    if wp1 == self.lander_location:
                        self.lander_visible_waypoints.add(wp2)
                    if wp2 == self.lander_location:
                         self.lander_visible_waypoints.add(wp1) # Visible is symmetric

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Parse current state facts
        rover_locations = {}
        soil_samples_at = set()
        rock_samples_at = set()
        have_soil = {} # rover -> set of waypoints
        have_rock = {} # rover -> set of waypoints
        have_image = {} # rover -> set of (objective, mode)
        calibrated_cameras = set() # set of (camera, rover)
        store_status = {} # store -> status ('empty' or 'full')
        communicated_soil = set() # set of waypoints
        communicated_rock = set() # set of waypoints
        communicated_image = set() # set of (objective, mode)

        # Initialize have_soil, have_rock, have_image for all known rovers
        all_known_rovers = set(self.rover_capabilities.keys()) | set(self.rover_stores.keys()) | set(r_info['on_board_rover'] for r_info in self.camera_info.values() if 'on_board_rover' in r_info)
        for rover in all_known_rovers:
             have_soil[rover] = set()
             have_rock[rover] = set()
             have_image[rover] = set()

        for fact in state:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == 'at':
                obj, loc = parts[1:]
                if obj in all_known_rovers: # Only track known rovers
                    rover_locations[obj] = loc
            elif pred == 'at_soil_sample':
                soil_samples_at.add(parts[1])
            elif pred == 'at_rock_sample':
                rock_samples_at.add(parts[1])
            elif pred == 'have_soil_analysis':
                rover, wp = parts[1:]
                if rover in have_soil: have_soil[rover].add(wp)
            elif pred == 'have_rock_analysis':
                rover, wp = parts[1:]
                if rover in have_rock: have_rock[rover].add(wp)
            elif pred == 'have_image':
                rover, obj, mode = parts[1:]
                if rover in have_image: have_image[rover].add((obj, mode))
            elif pred == 'calibrated':
                camera, rover = parts[1:]
                calibrated_cameras.add((camera, rover))
            elif pred == 'empty':
                store_status[parts[1]] = 'empty'
            elif pred == 'full':
                store_status[parts[1]] = 'full'
            elif pred == 'communicated_soil_data':
                communicated_soil.add(parts[1])
            elif pred == 'communicated_rock_data':
                communicated_rock.add(parts[1])
            elif pred == 'communicated_image_data':
                obj, mode = parts[1:]
                communicated_image.add((obj, mode))

        total_cost = 0

        # Iterate through goal conditions
        for goal in self.goals:
            parts = get_parts(goal)
            pred = parts[0]

            if pred == 'communicated_soil_data':
                goal_wp = parts[1]
                if goal_wp not in communicated_soil:
                    min_goal_cost = float('inf')
                    # Find rovers capable of soil analysis
                    capable_rovers = [r for r, caps in self.rover_capabilities.items() if 'soil' in caps]

                    for rover in capable_rovers:
                        r_loc = rover_locations.get(rover)
                        if r_loc is None: continue # Rover location unknown

                        rover_cost = float('inf')

                        # Case 1: Already have the soil analysis
                        if goal_wp in have_soil.get(rover, set()):
                            # Need to move to lander view and communicate
                            nav_cost_to_comm = bfs_distance(self.rover_traverse_graphs.get(rover, {}), r_loc, self.lander_visible_waypoints)
                            if nav_cost_to_comm != float('inf'):
                                rover_cost = min(rover_cost, nav_cost_to_comm + 1) # 1 for communicate

                        # Case 2: Need to sample and communicate
                        else:
                            # Check if sample is still available at the waypoint
                            if goal_wp in soil_samples_at:
                                # Need to move to sample location, sample, move to lander view, communicate
                                nav_cost_to_sample = bfs_distance(self.rover_traverse_graphs.get(rover, {}), r_loc, {goal_wp})
                                if nav_cost_to_sample != float('inf'):
                                    nav_cost_from_sample_to_comm = bfs_distance(self.rover_traverse_graphs.get(rover, {}), goal_wp, self.lander_visible_waypoints)
                                    if nav_cost_from_sample_to_comm != float('inf'):
                                         sample_comm_cost = nav_cost_to_sample + 1 + nav_cost_from_sample_to_comm + 1 # sample + communicate

                                         # Add cost for dropping if store is full
                                         store = self.rover_stores.get(rover)
                                         drop_cost = 0
                                         if store and store_status.get(store) == 'full':
                                             drop_cost = 1 # Assumes drop can happen at current location or sample location

                                         rover_cost = min(rover_cost, sample_comm_cost + drop_cost)

                    min_goal_cost = min(min_goal_cost, rover_cost)

                    total_cost += min_goal_cost

            elif pred == 'communicated_rock_data':
                goal_wp = parts[1]
                if goal_wp not in communicated_rock:
                    min_goal_cost = float('inf')
                    # Find rovers capable of rock analysis
                    capable_rovers = [r for r, caps in self.rover_capabilities.items() if 'rock' in caps]

                    for rover in capable_rovers:
                        r_loc = rover_locations.get(rover)
                        if r_loc is None: continue # Rover location unknown

                        rover_cost = float('inf')

                        # Case 1: Already have the rock analysis
                        if goal_wp in have_rock.get(rover, set()):
                            # Need to move to lander view and communicate
                            nav_cost_to_comm = bfs_distance(self.rover_traverse_graphs.get(rover, {}), r_loc, self.lander_visible_waypoints)
                            if nav_cost_to_comm != float('inf'):
                                rover_cost = min(rover_cost, nav_cost_to_comm + 1) # 1 for communicate

                        # Case 2: Need to sample and communicate
                        else:
                             # Check if sample is still available at the waypoint
                            if goal_wp in rock_samples_at:
                                # Need to move to sample location, sample, move to lander view, communicate
                                nav_cost_to_sample = bfs_distance(self.rover_traverse_graphs.get(rover, {}), r_loc, {goal_wp})
                                if nav_cost_to_sample != float('inf'):
                                    nav_cost_from_sample_to_comm = bfs_distance(self.rover_traverse_graphs.get(rover, {}), goal_wp, self.lander_visible_waypoints)
                                    if nav_cost_from_sample_to_comm != float('inf'):
                                         sample_comm_cost = nav_cost_to_sample + 1 + nav_cost_from_sample_to_comm + 1 # sample + communicate

                                         # Add cost for dropping if store is full
                                         store = self.rover_stores.get(rover)
                                         drop_cost = 0
                                         if store and store_status.get(store) == 'full':
                                             drop_cost = 1 # Assumes drop can happen at current location or sample location

                                         rover_cost = min(rover_cost, sample_comm_cost + drop_cost)

                    min_goal_cost = min(min_goal_cost, rover_cost)
                    total_cost += min_goal_cost

            elif pred == 'communicated_image_data':
                goal_obj, goal_mode = parts[1], parts[2]
                if (goal_obj, goal_mode) not in communicated_image:
                    min_goal_cost = float('inf')
                    # Find rovers capable of imaging
                    imaging_rovers = [r for r, caps in self.rover_capabilities.items() if 'imaging' in caps]

                    for rover in imaging_rovers:
                        r_loc = rover_locations.get(rover)
                        if r_loc is None: continue # Rover location unknown

                        # Find cameras on this rover supporting the mode
                        suitable_cameras = [
                            cam for cam, info in self.camera_info.items()
                            if info.get('on_board_rover') == rover and goal_mode in info.get('supported_modes', set())
                        ]
                        if not suitable_cameras:
                             continue # This rover can't achieve this image goal

                        for camera in suitable_cameras:
                             camera_rover_cost = float('inf')

                             # Case 1: Already have the image
                             if (goal_obj, goal_mode) in have_image.get(rover, set()):
                                 # Need to move to lander view and communicate
                                 nav_cost_to_comm = bfs_distance(self.rover_traverse_graphs.get(rover, {}), r_loc, self.lander_visible_waypoints)
                                 if nav_cost_to_comm != float('inf'):
                                     camera_rover_cost = min(camera_rover_cost, nav_cost_to_comm + 1) # 1 for communicate

                             # Case 2: Need to take image and communicate
                             else:
                                 # Need to move to image location, take image, move to lander view, communicate
                                 image_wps = self.objective_visibility.get(goal_obj, set())
                                 if not image_wps:
                                     continue # No waypoint to view objective from

                                 # Find best image waypoint (closest to current location)
                                 min_nav_cost_to_image_wp = float('inf')
                                 best_img_wp = None
                                 for img_wp in image_wps:
                                     cost = bfs_distance(self.rover_traverse_graphs.get(rover, {}), r_loc, {img_wp})
                                     if cost < min_nav_cost_to_image_wp:
                                         min_nav_cost_to_image_wp = cost
                                         best_img_wp = img_wp

                                 if best_img_wp is None or min_nav_cost_to_image_wp == float('inf'):
                                     continue # Cannot reach any image waypoint

                                 # Cost to take image = (calibration_cost) + 1 (take_image)
                                 # Cost to communicate = nav_cost_from_image_loc_to_comm + 1 (communicate)

                                 nav_cost_from_image_to_comm = bfs_distance(self.rover_traverse_graphs.get(rover, {}), best_img_wp, self.lander_visible_waypoints)
                                 if nav_cost_from_image_to_comm == float('inf'):
                                     continue # Cannot reach lander view from image waypoint

                                 take_comm_cost = 1 + nav_cost_from_image_to_comm + 1 # take_image + communicate

                                 # Add calibration cost if needed
                                 calibration_cost = 0
                                 if (camera, rover) not in calibrated_cameras:
                                     # Need to calibrate first
                                     cal_target = self.calibration_targets.get(camera)
                                     if cal_target is None:
                                         continue # Camera has no calibration target

                                     cal_wps = self.objective_visibility.get(cal_target, set())
                                     if not cal_wps:
                                         continue # No waypoint to view calibration target from

                                     # Find best calibration waypoint (closest to current location)
                                     min_nav_cost_to_cal_wp = float('inf')
                                     best_cal_wp = None
                                     for cal_wp in cal_wps:
                                         cost = bfs_distance(self.rover_traverse_graphs.get(rover, {}), r_loc, {cal_wp})
                                         if cost < min_nav_cost_to_cal_wp:
                                             min_nav_cost_to_cal_wp = cost
                                             best_cal_wp = cal_wp

                                     if best_cal_wp is None or min_nav_cost_to_cal_wp == float('inf'):
                                         continue # Cannot reach any calibration waypoint

                                     # Nav cost from calibration wp to image wp
                                     nav_cost_cal_to_image = bfs_distance(self.rover_traverse_graphs.get(rover, {}), best_cal_wp, {best_img_wp})
                                     if nav_cost_cal_to_image == float('inf'):
                                         continue # Cannot reach image waypoint from calibration waypoint

                                     calibration_cost = min_nav_cost_to_cal_wp + 1 + nav_cost_cal_to_image # nav to cal + calibrate + nav to image

                                 else: # Camera is already calibrated
                                     # The cost to take the image starts from the current location, navigating to the image waypoint.
                                     # The calibration cost is effectively zero, but we still need to reach the image waypoint.
                                     # The navigation cost to the image waypoint is already calculated as min_nav_cost_to_image_wp.
                                     calibration_cost = min_nav_cost_to_image_wp # This represents the navigation part before taking the image

                                 total_action_cost_for_image = calibration_cost + take_comm_cost
                                 camera_rover_cost = min(camera_rover_cost, total_action_cost_for_image)

                             min_goal_cost = min(min_goal_cost, camera_rover_cost)

                    total_cost += min_goal_cost

        # If any goal was unreachable, total_cost will be infinity
        return total_cost
