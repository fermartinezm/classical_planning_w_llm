from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic # Assuming this base class exists

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact."""
    # Ensure fact is treated as a string and handle parentheses
    fact_str = str(fact).strip()
    if fact_str.startswith('(') and fact_str.endswith(')'):
        return fact_str[1:-1].split()
    # Fallback for non-standard formats, though PDDL facts are typically (pred arg1 ...)
    return [fact_str]

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    Wildcards `*` allowed in args.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# BFS function for shortest path
def bfs_shortest_path(graph, start, end):
    """
    Find the shortest path distance between start and end in a graph using BFS.
    Graph is an adjacency list: {node: [neighbor1, neighbor2, ...]}
    Returns distance, or float('inf') if no path exists.
    """
    if start == end:
        return 0
    if start not in graph or end not in graph:
        return float('inf') # Start or end node not in graph structure

    visited = {start}
    queue = deque([(start, 0)]) # (node, distance)

    while queue:
        current_node, distance = queue.popleft()

        # Check if current_node is in graph keys before iterating neighbors
        if current_node in graph:
            for neighbor in graph[current_node]:
                if neighbor == end:
                    return distance + 1
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, distance + 1))

    return float('inf') # No path found

# Heuristic class definition
class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the total number of actions required to achieve all
    uncommunicated goal predicates. It breaks down each goal (communicating
    soil, rock, or image data) into sequential steps: acquiring the data
    (sampling or imaging/calibrating) and then communicating the data.
    Navigation costs between required locations are estimated using precomputed
    shortest paths on the rover's traverse graph. The heuristic sums the minimum
    estimated cost for each uncommunicated goal, considering the capabilities
    and current state of available rovers.

    # Assumptions
    - All goal predicates are initially achievable in the problem instance.
    - Navigation is possible between any two waypoints connected by a path
      traversable by the specific rover, and the cost is the shortest path
      distance (number of `navigate` actions).
    - A rover can only hold one sample at a time in its store.
    - A camera becomes uncalibrated after taking an image.
    - Communication requires the rover to be at a waypoint visible from the lander.
    - The heuristic assumes that any capable rover can be used for any goal,
      and it greedily picks the rover/waypoint combination that minimizes the
      estimated cost for that specific goal, ignoring potential conflicts or
      synergies with other goals. This makes it non-admissible but potentially
      more informative than simpler heuristics.

    # Heuristic Initialization
    The heuristic precomputes static information from the task definition:
    - Lander location.
    - Which store belongs to which rover.
    - Equipment (soil, rock, imaging) possessed by each rover.
    - Cameras on board each rover and modes they support.
    - Calibration target for each camera.
    - Waypoints from which each objective is visible (for imaging and calibration).
    - Waypoint visibility graph (for identifying communication points).
    - Rover navigation graphs based on `can_traverse` facts.
    - Shortest path distances between all pairs of waypoints for each rover
      using Breadth-First Search (BFS).
    - Set of waypoints visible from the lander (communication points).

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic calculates the sum of estimated costs
    for each goal predicate that is not yet satisfied in that state.

    For each uncommunicated goal (e.g., `(communicated_soil_data waypoint_w)`):
    1.  Initialize the minimum cost for this goal to infinity.
    2.  Identify all rovers capable of achieving this goal (based on equipment).
    3.  For each capable rover:
        a.  Estimate the cost to acquire the necessary data (`have_...` predicate).
            -   If the data is already acquired by this rover, the acquisition cost is 0.
            -   Otherwise (e.g., need to sample soil at `waypoint_w`):
                -   Add 1 action if the rover's store is full (for `drop`).
                -   Calculate navigation cost from the rover's current location to the required location (e.g., `waypoint_w` for sampling, or a calibration waypoint then an image waypoint for imaging). This uses precomputed shortest paths.
                -   Add 1 action for the core acquisition step (`sample_soil`, `sample_rock`, or `take_image`).
                -   If imaging, also estimate calibration cost if the camera is not calibrated: navigate to a calibration waypoint visible from the target, plus 1 action for `calibrate`. The navigation cost for imaging is then calculated from the calibration waypoint.
            -   Determine the rover's location after acquisition.
        b.  Estimate the cost to communicate the data.
            -   Find the lander's location.
            -   Identify waypoints visible from the lander (communication points).
            -   Calculate navigation cost from the rover's location after acquisition to the closest communication point (using precomputed shortest paths).
            -   Add 1 action for the communication step (`communicate_..._data`).
        c.  The total estimated cost for this rover/goal combination is the sum of acquisition and communication costs.
        d.  Update the minimum cost for this goal if the current rover's cost is lower.
    4.  If no capable rover can reach the necessary locations, the minimum cost remains infinity, indicating the goal is likely unreachable. The total heuristic returns infinity in this case.
    5.  Add the minimum estimated cost for this goal to the total heuristic value.

    The final heuristic value is the sum of these minimum costs across all
    uncommunicated goals.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # Precompute static information
        self.lander_location = {}
        self.rover_stores = {} # {rover: store}
        self.rover_equipment = {} # {rover: {soil, rock, imaging}}
        self.rover_cameras = {} # {rover: [camera1, camera2]}
        self.camera_modes = {} # {camera: {mode1, mode2}}
        self.camera_calibration_target = {} # {camera: objective}
        self.objective_visible_from = {} # {objective: {waypoint1, waypoint2}}
        self.waypoint_visibility = {} # {wp1: {wp2, wp3}}
        self.rover_navigation_graph = {} # {rover: {wp1: [wp2, wp3]}}
        self.rover_shortest_paths = {} # {rover: {(start_wp, end_wp): distance}}
        self.communication_waypoints = set() # Waypoints visible from lander

        # Collect all object names by type from initial state, goals, and static facts
        # This is a workaround as task object doesn't expose typed objects directly
        all_rovers = set()
        all_waypoints = set()
        all_objectives = set()
        all_modes = set()
        all_cameras = set()
        all_landers = set()
        all_stores = set()

        # Helper to add objects based on predicate structure
        def add_objects_from_fact(fact_str):
            parts = get_parts(fact_str)
            if not parts: return
            predicate = parts[0]
            args = parts[1:]

            if predicate == 'at' and len(args) == 2:
                 if args[0].startswith('rover'): all_rovers.add(args[0])
                 if args[1].startswith('waypoint'): all_waypoints.add(args[1])
            elif predicate == 'at_lander' and len(args) == 2:
                 all_landers.add(args[0])
                 all_waypoints.add(args[1])
            elif predicate == 'can_traverse' and len(args) == 3:
                 all_rovers.add(args[0])
                 all_waypoints.add(args[1])
                 all_waypoints.add(args[2])
            elif predicate.startswith('equipped_for_') and len(args) == 1:
                 all_rovers.add(args[0])
            elif predicate in ['empty', 'full'] and len(args) == 1:
                 all_stores.add(args[0])
            elif predicate.startswith('have_') and len(args) == 2:
                 all_rovers.add(args[0])
                 all_waypoints.add(args[1])
            elif predicate == 'calibrated' and len(args) == 2:
                 all_cameras.add(args[0])
                 all_rovers.add(args[1])
            elif predicate == 'supports' and len(args) == 2:
                 all_cameras.add(args[0])
                 all_modes.add(args[1])
            elif predicate == 'visible' and len(args) == 2:
                 all_waypoints.add(args[0])
                 all_waypoints.add(args[1])
            elif predicate == 'have_image' and len(args) == 3:
                 all_rovers.add(args[0])
                 all_objectives.add(args[1])
                 all_modes.add(args[2])
            elif predicate.startswith('communicated_') and len(args) >= 1:
                 if predicate in ['communicated_soil_data', 'communicated_rock_data'] and len(args) == 1:
                     all_waypoints.add(args[0])
                 elif predicate == 'communicated_image_data' and len(args) == 2:
                     all_objectives.add(args[0])
                     all_modes.add(args[1])
            elif predicate.startswith('at_') and len(args) == 1: # at_soil_sample, at_rock_sample
                 all_waypoints.add(args[0])
            elif predicate == 'visible_from' and len(args) == 2:
                 all_objectives.add(args[0])
                 all_waypoints.add(args[1])
            elif predicate == 'store_of' and len(args) == 2:
                 all_stores.add(args[0])
                 all_rovers.add(args[1])
            elif predicate == 'calibration_target' and len(args) == 2:
                 all_cameras.add(args[0])
                 all_objectives.add(args[1])
            elif predicate == 'on_board' and len(args) == 2:
                 all_cameras.add(args[0])
                 all_rovers.add(args[1])

        # Parse facts to populate object sets
        for fact_str in task.initial_state: add_objects_from_fact(fact_str)
        for fact_str in task.goals: add_objects_from_fact(fact_str)
        for fact_str in static_facts: add_objects_from_fact(fact_str)


        # Initialize static data structures based on identified objects
        for rover in all_rovers:
            self.rover_equipment[rover] = set()
            self.rover_navigation_graph[rover] = {wp: [] for wp in all_waypoints}
            self.rover_shortest_paths[rover] = {}
            self.rover_cameras[rover] = []
        for camera in all_cameras:
            self.camera_modes[camera] = set()
        for objective in all_objectives:
            self.objective_visible_from[objective] = set()
        for wp in all_waypoints:
             self.waypoint_visibility[wp] = set()
        # Add waypoints from can_traverse/visible that might not be in init/goals
        # This ensures graph includes all relevant waypoints mentioned in static facts
        for fact in static_facts:
             if match(fact, "can_traverse", "*", "*", "*"):
                 _, r, w1, w2 = get_parts(fact)
                 if r in self.rover_navigation_graph: # Ensure rover exists
                     if w1 not in self.rover_navigation_graph[r]: self.rover_navigation_graph[r][w1] = []
                     if w2 not in self.rover_navigation_graph[r]: self.rover_navigation_graph[r][w2] = []
             elif match(fact, "visible", "*", "*"):
                 _, w1, w2 = get_parts(fact)
                 if w1 not in self.waypoint_visibility: self.waypoint_visibility[w1] = set()
                 if w2 not in self.waypoint_visibility: self.waypoint_visibility[w2] = set()


        lander_wp = None
        for fact in static_facts:
            if match(fact, "at_lander", "*", "*"):
                lander, wp = get_parts(fact)[1:]
                self.lander_location[lander] = wp
                lander_wp = wp # Assuming only one lander relevant for communication
            elif match(fact, "store_of", "*", "*"):
                store, rover = get_parts(fact)[1:]
                self.rover_stores[rover] = store # Assuming one store per rover
            elif match(fact, "equipped_for_soil_analysis", "*"):
                rover = get_parts(fact)[1]
                if rover in self.rover_equipment: self.rover_equipment[rover].add('soil')
            elif match(fact, "equipped_for_rock_analysis", "*"):
                rover = get_parts(fact)[1]
                if rover in self.rover_equipment: self.rover_equipment[rover].add('rock')
            elif match(fact, "equipped_for_imaging", "*"):
                rover = get_parts(fact)[1]
                if rover in self.rover_equipment: self.rover_equipment[rover].add('imaging')
            elif match(fact, "on_board", "*", "*"):
                camera, rover = get_parts(fact)[1:]
                if rover in self.rover_cameras: self.rover_cameras[rover].append(camera)
            elif match(fact, "supports", "*", "*"):
                camera, mode = get_parts(fact)[1:]
                if camera in self.camera_modes: self.camera_modes[camera].add(mode)
            elif match(fact, "calibration_target", "*", "*"):
                camera, objective = get_parts(fact)[1:]
                if camera in self.camera_calibration_target: # Avoid overwriting if multiple targets? Domain suggests one.
                    pass # Or handle error/warning
                self.camera_calibration_target[camera] = objective
            elif match(fact, "visible_from", "*", "*"):
                objective, wp = get_parts(fact)[1:]
                if objective in self.objective_visible_from: self.objective_visible_from[objective].add(wp)
            elif match(fact, "visible", "*", "*"):
                wp1, wp2 = get_parts(fact)[1:]
                if wp1 in self.waypoint_visibility: self.waypoint_visibility[wp1].add(wp2)
                if wp2 in self.waypoint_visibility: self.waypoint_visibility[wp2].add(wp1) # Assuming symmetric
            elif match(fact, "can_traverse", "*", "*", "*"):
                rover, wp1, wp2 = get_parts(fact)[1:]
                if rover in self.rover_navigation_graph:
                    if wp1 in self.rover_navigation_graph[rover]:
                        self.rover_navigation_graph[rover][wp1].append(wp2)
                    if wp2 in self.rover_navigation_graph[rover]:
                        self.rover_navigation_graph[rover][wp2].append(wp1) # Assuming symmetric

        # Precompute communication waypoints (visible from lander)
        if lander_wp and lander_wp in self.waypoint_visibility:
             self.communication_waypoints = self.waypoint_visibility[lander_wp]

        # Precompute shortest paths for each rover
        for rover, graph in self.rover_navigation_graph.items():
            for start_wp in graph:
                # Run BFS from start_wp to all reachable waypoints
                distances = {wp: float('inf') for wp in graph}
                distances[start_wp] = 0
                queue = deque([start_wp])
                visited = {start_wp}

                while queue:
                    current_wp = queue.popleft()

                    # Check if current_wp is in graph keys before iterating neighbors
                    if current_wp in graph:
                        for neighbor_wp in graph[current_wp]:
                            if neighbor_wp not in visited:
                                visited.add(neighbor_wp)
                                distances[neighbor_wp] = distances[current_wp] + 1
                                queue.append(neighbor_wp)

                # Store distances
                for end_wp, dist in distances.items():
                    self.rover_shortest_paths[rover][(start_wp, end_wp)] = dist

    def get_navigation_cost(self, rover, start_wp, end_wp):
        """Lookup precomputed shortest path distance."""
        if start_wp == end_wp:
            return 0
        # Return infinity if rover or waypoints are not in the precomputed paths
        return self.rover_shortest_paths.get(rover, {}).get((start_wp, end_wp), float('inf'))


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Extract current state information
        current_rover_locations = {}
        current_store_status = {} # {store: 'empty' or 'full'}
        current_have_soil = set() # {(rover, waypoint)}
        current_have_rock = set() # {(rover, waypoint)}
        current_have_image = set() # {(rover, objective, mode)}
        current_calibrated_cameras = set() # {(camera, rover)}
        # current_at_soil_sample and current_at_rock_sample are not strictly needed
        # for the heuristic logic as designed (it assumes samples are available if needed)
        # but could be used for a more accurate check if a goal is truly unreachable.
        # current_at_soil_sample = set() # {waypoint}
        # current_at_rock_sample = set() # {waypoint}

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue
            predicate = parts[0]
            args = parts[1:]

            if predicate == 'at' and len(args) == 2 and args[0].startswith('rover'):
                current_rover_locations[args[0]] = args[1]
            elif predicate == 'empty' and len(args) == 1 and args[0].startswith('store'):
                current_store_status[args[0]] = 'empty'
            elif predicate == 'full' and len(args) == 1 and args[0].startswith('store'):
                current_store_status[args[0]] = 'full'
            elif predicate == 'have_soil_analysis' and len(args) == 2:
                current_have_soil.add(tuple(args))
            elif predicate == 'have_rock_analysis' and len(args) == 2:
                current_have_rock.add(tuple(args))
            elif predicate == 'have_image' and len(args) == 3:
                current_have_image.add(tuple(args))
            elif predicate == 'calibrated' and len(args) == 2:
                current_calibrated_cameras.add(tuple(args))
            # elif predicate == 'at_soil_sample' and len(args) == 1:
            #      current_at_soil_sample.add(args[0])
            # elif predicate == 'at_rock_sample' and len(args) == 1:
            #      current_at_rock_sample.add(args[0])

        total_heuristic_cost = 0

        # Process each goal
        for goal_fact_str in self.goals:
            if goal_fact_str in state:
                continue # Goal already achieved

            parts = get_parts(goal_fact_str)
            if not parts: continue
            goal_predicate = parts[0]
            goal_args = parts[1:]

            min_goal_cost = float('inf')

            if goal_predicate == 'communicated_soil_data' and len(goal_args) == 1:
                waypoint_w = goal_args[0]

                # Find suitable rovers
                capable_rovers = [r for r, equipment in self.rover_equipment.items() if 'soil' in equipment]
                if not capable_rovers:
                    min_goal_cost = float('inf') # Goal unreachable by any rover
                else:
                    for rover in capable_rovers:
                        rover_cost = 0
                        curr_wp = current_rover_locations.get(rover)
                        if curr_wp is None: continue # Rover location unknown or rover doesn't exist

                        wp_after_acq = curr_wp # Where the rover is after acquisition (if needed)

                        # 1. Acquisition Step: Get have_soil_analysis
                        if (rover, waypoint_w) not in current_have_soil:
                            # Need to sample soil at waypoint_w
                            store = self.rover_stores.get(rover)
                            if store is None: continue # Rover has no store

                            if current_store_status.get(store) == 'full':
                                rover_cost += 1 # drop action

                            nav_cost_to_sample = self.get_navigation_cost(rover, curr_wp, waypoint_w)
                            if nav_cost_to_sample == float('inf'): continue # Cannot reach sample location

                            rover_cost += nav_cost_to_sample
                            rover_cost += 1 # sample_soil action
                            wp_after_acq = waypoint_w # Rover is now at the sample location

                        # 2. Communication Step: Communicate data
                        lander_wp = list(self.lander_location.values())[0] # Assuming one lander
                        if not self.communication_waypoints:
                             continue # No waypoints visible from lander

                        min_nav_cost_to_comm = float('inf')
                        best_comm_wp = None
                        # Calculate navigation from the waypoint *after* acquisition
                        nav_start_wp_for_comm = wp_after_acq
                        for comm_wp in self.communication_waypoints:
                             nav_cost = self.get_navigation_cost(rover, nav_start_wp_for_comm, comm_wp)
                             if nav_cost < min_nav_cost_to_comm:
                                  min_nav_cost_to_comm = nav_cost
                                  best_comm_wp = comm_wp

                        if best_comm_wp is None or min_nav_cost_to_comm == float('inf'):
                             continue # Cannot reach any communication waypoint

                        rover_cost += min_nav_cost_to_comm
                        rover_cost += 1 # communicate_soil_data action

                        min_goal_cost = min(min_goal_cost, rover_cost)

            elif goal_predicate == 'communicated_rock_data' and len(goal_args) == 1:
                waypoint_w = goal_args[0]

                # Find suitable rovers
                capable_rovers = [r for r, equipment in self.rover_equipment.items() if 'rock' in equipment]
                if not capable_rovers:
                    min_goal_cost = float('inf') # Goal unreachable by any rover
                else:
                    for rover in capable_rovers:
                        rover_cost = 0
                        curr_wp = current_rover_locations.get(rover)
                        if curr_wp is None: continue

                        wp_after_acq = curr_wp

                        # 1. Acquisition Step: Get have_rock_analysis
                        if (rover, waypoint_w) not in current_have_rock:
                            # Need to sample rock at waypoint_w
                            store = self.rover_stores.get(rover)
                            if store is None: continue # Cannot sample

                            if current_store_status.get(store) == 'full':
                                rover_cost += 1 # drop action

                            nav_cost_to_sample = self.get_navigation_cost(rover, curr_wp, waypoint_w)
                            if nav_cost_to_sample == float('inf'): continue

                            rover_cost += nav_cost_to_sample
                            rover_cost += 1 # sample_rock action
                            wp_after_acq = waypoint_w # Rover is now at the sample location

                        # 2. Communication Step: Communicate data
                        lander_wp = list(self.lander_location.values())[0] # Assuming one lander
                        if not self.communication_waypoints: continue

                        min_nav_cost_to_comm = float('inf')
                        best_comm_wp = None
                        # Calculate navigation from the waypoint *after* acquisition
                        nav_start_wp_for_comm = wp_after_acq
                        for comm_wp in self.communication_waypoints:
                             nav_cost = self.get_navigation_cost(rover, nav_start_wp_for_comm, comm_wp)
                             if nav_cost < min_nav_cost_to_comm:
                                  min_nav_cost_to_comm = nav_cost
                                  best_comm_wp = comm_wp

                        if best_comm_wp is None or min_nav_cost_to_comm == float('inf'): continue

                        rover_cost += min_nav_cost_to_comm
                        rover_cost += 1 # communicate_rock_data action

                        min_goal_cost = min(min_goal_cost, rover_cost)

            elif goal_predicate == 'communicated_image_data' and len(goal_args) == 2:
                objective_o, mode_m = goal_args

                # Find suitable rover/camera pairs
                suitable_pairs = [] # [(rover, camera)]
                for rover, equipment in self.rover_equipment.items():
                    if 'imaging' in equipment:
                        for camera in self.rover_cameras.get(rover, []):
                            if mode_m in self.camera_modes.get(camera, set()):
                                suitable_pairs.append((rover, camera))

                if not suitable_pairs:
                    min_goal_cost = float('inf') # Goal unreachable
                else:
                    for rover, camera in suitable_pairs:
                        pair_cost = 0
                        curr_wp = current_rover_locations.get(rover)
                        if curr_wp is None: continue

                        wp_after_cal = curr_wp # Where rover is after calibration (if needed)
                        wp_after_acq = curr_wp # Where rover is after image acquisition (if needed)

                        # 1. Acquisition Step: Get have_image
                        if (rover, objective_o, mode_m) not in current_have_image:
                            # Need to take image
                            cal_target = self.camera_calibration_target.get(camera)
                            if cal_target is None:
                                # Camera has no calibration target? Cannot calibrate.
                                continue # Cannot take image

                            # 1a. Calibration Step: Get calibrated
                            if (camera, rover) not in current_calibrated_cameras:
                                # Need to calibrate camera
                                cal_wps = self.objective_visible_from.get(cal_target, set())
                                if not cal_wps:
                                    # Calibration target not visible from anywhere? Cannot calibrate.
                                    continue

                                min_nav_cost_to_cal = float('inf')
                                best_cal_wp = None
                                # Calculate navigation from the rover's current location
                                nav_start_wp_for_cal = curr_wp
                                for cal_wp in cal_wps:
                                    nav_cost = self.get_navigation_cost(rover, nav_start_wp_for_cal, cal_wp)
                                    if nav_cost < min_nav_cost_to_cal:
                                        min_nav_cost_to_cal = nav_cost
                                        best_cal_wp = cal_wp

                                if best_cal_wp is None or min_nav_cost_to_cal == float('inf'):
                                    continue # Cannot reach any calibration waypoint

                                pair_cost += min_nav_cost_to_cal
                                pair_cost += 1 # calibrate action
                                wp_after_cal = best_cal_wp # Rover is now at calibration location

                            # 1b. Take Image Step:
                            img_wps = self.objective_visible_from.get(objective_o, set())
                            if not img_wps:
                                # Objective not visible from anywhere? Cannot take image.
                                continue

                            min_nav_cost_to_img = float('inf')
                            best_img_wp = None
                            # Calculate navigation from the waypoint *after* calibration
                            nav_start_wp_for_img = wp_after_cal
                            for img_wp in img_wps:
                                nav_cost = self.get_navigation_cost(rover, nav_start_wp_for_img, img_wp)
                                if nav_cost < min_nav_cost_to_img:
                                    min_nav_cost_to_img = nav_cost
                                    best_img_wp = img_wp

                            if best_img_wp is None or min_nav_cost_to_img == float('inf'):
                                continue # Cannot reach any image waypoint

                            pair_cost += min_nav_cost_to_img
                            pair_cost += 1 # take_image action
                            wp_after_acq = best_img_wp # Rover is now at image location

                        # 2. Communication Step: Communicate data
                        lander_wp = list(self.lander_location.values())[0] # Assuming one lander
                        if not self.communication_waypoints: continue

                        min_nav_cost_to_comm = float('inf')
                        best_comm_wp = None
                        # Calculate navigation from the waypoint *after* acquisition
                        nav_start_wp_for_comm = wp_after_acq
                        for comm_wp in self.communication_waypoints:
                             nav_cost = self.get_navigation_cost(rover, nav_start_wp_for_comm, comm_wp)
                             if nav_cost < min_nav_cost_to_comm:
                                  min_nav_cost_to_comm = nav_cost
                                  best_comm_wp = comm_wp

                        if best_comm_wp is None or min_nav_cost_to_comm == float('inf'): continue

                        pair_cost += min_nav_cost_to_comm
                        pair_cost += 1 # communicate_image_data action

                        min_goal_cost = min(min_goal_cost, pair_cost)

            # Add the minimum cost found for this goal (or infinity if unreachable)
            if min_goal_cost == float('inf'):
                 # If a goal is unreachable, the problem is unsolvable.
                 # Return infinity to prune this path.
                 return float('inf')
            total_heuristic_cost += min_goal_cost

        return total_heuristic_cost
