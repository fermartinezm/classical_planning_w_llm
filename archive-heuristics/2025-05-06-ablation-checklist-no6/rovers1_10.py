from fnmatch import fnmatch
from collections import deque, defaultdict
from heuristics.heuristic_base import Heuristic

# Define a large cost to represent unreachable goals
UNREACHABLE_COST = 1000000

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty facts or malformed strings gracefully
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def bfs(graph, start_node, all_nodes):
    """
    Computes shortest path distances from a start_node to all other nodes
    in a graph using Breadth-First Search.

    Args:
        graph: Adjacency list representation (dict: node -> list of neighbors).
        start_node: The starting node for the BFS.
        all_nodes: A set of all possible nodes in the graph.

    Returns:
        A dictionary mapping reachable nodes to their shortest distance from start_node.
        Nodes in all_nodes that are unreachable will not be in the dictionary.
    """
    distances = {}
    if start_node not in all_nodes:
         # Start node is not a known waypoint, cannot start BFS
         return {}

    distances[start_node] = 0
    queue = deque([start_node])
    visited = {start_node}

    while queue:
        current_node = queue.popleft()

        # Check if current_node has neighbors in the graph
        if current_node in graph:
            for neighbor in graph[current_node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)

    return distances


class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    Estimates the number of actions required to reach the goal state by summing
    up the estimated costs for each unachieved goal fact.

    The heuristic considers the actions needed for sampling/imaging and communication,
    including navigation costs estimated by shortest paths on the 'visible' graph.
    It makes simplifying assumptions, such as ignoring specific rover assignments
    for tasks (considering any equipped rover) and relaxing resource constraints
    (like store capacity beyond needing one drop action if any store is full,
    or camera calibration state beyond needing one calibrate action if uncalibrated).
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and static facts,
        and precomputing navigation shortest paths.
        """
        self.goals = task.goals
        static_facts = task.static
        initial_state = task.initial_state # Needed to know initial sample locations

        self.rover_info = defaultdict(lambda: {'equipped_soil': False, 'equipped_rock': False, 'equipped_imaging': False, 'stores': [], 'cameras': []})
        self.camera_info = {} # camera -> {'rover': rover, 'modes': [mode], 'calibration_target': objective}
        self.objective_info = defaultdict(list) # objective -> [waypoint] (visible_from)
        self.calibration_target_locations = defaultdict(list) # objective (target) -> [waypoint] (visible_from)
        self.waypoint_graph = defaultdict(list) # waypoint -> [neighbor_waypoint] (visible)
        self.lander_location = None
        self.all_waypoints = set() # Collect all waypoints mentioned

        # Parse initial state to collect all waypoints and initial sample locations
        for fact in initial_state:
             parts = get_parts(fact)
             if not parts: continue
             predicate = parts[0]
             if predicate == "at_soil_sample":
                 # We don't need initial sample locations in __init__ for this heuristic logic
                 # as we check current state for 'have_soil_analysis'.
                 # But we collect waypoints.
                 self.all_waypoints.add(parts[1])
             elif predicate == "at_rock_sample":
                 self.all_waypoints.add(parts[1])
             elif predicate == "at":
                 # Rover location
                 self.all_waypoints.add(parts[2])
             elif predicate == "at_lander":
                 # Lander location
                 self.all_waypoints.add(parts[2])


        # Parse static facts
        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]

            if predicate == "at_lander":
                self.lander_location = parts[2]
                self.all_waypoints.add(self.lander_location)
            elif predicate == "visible":
                wp1, wp2 = parts[1], parts[2]
                self.waypoint_graph[wp1].append(wp2)
                # Assuming visible is symmetric based on examples
                self.waypoint_graph[wp2].append(wp1)
                self.all_waypoints.add(wp1)
                self.all_waypoints.add(wp2)
            elif predicate == "equipped_for_soil_analysis":
                self.rover_info[parts[1]]['equipped_soil'] = True
            elif predicate == "equipped_for_rock_analysis":
                self.rover_info[parts[1]]['equipped_rock'] = True
            elif predicate == "equipped_for_imaging":
                self.rover_info[parts[1]]['equipped_imaging'] = True
            elif predicate == "store_of":
                store, rover = parts[1], parts[2]
                self.rover_info[rover]['stores'].append(store)
            elif predicate == "on_board":
                camera, rover = parts[1], parts[2]
                self.camera_info[camera] = self.camera_info.get(camera, {}) # Ensure dict exists
                self.camera_info[camera]['rover'] = rover
                self.rover_info[rover]['cameras'].append(camera)
            elif predicate == "supports":
                camera, mode = parts[1], parts[2]
                self.camera_info[camera] = self.camera_info.get(camera, {})
                self.camera_info[camera]['modes'] = self.camera_info[camera].get('modes', []) + [mode]
            elif predicate == "calibration_target":
                camera, objective = parts[1], parts[2]
                self.camera_info[camera] = self.camera_info.get(camera, {})
                self.camera_info[camera]['calibration_target'] = objective
            elif predicate == "visible_from":
                objective, waypoint = parts[1], parts[2]
                self.objective_info[objective].append(waypoint)
                self.all_waypoints.add(waypoint)
                # Check if this objective is a calibration target for any camera
                for cam, c_info in self.camera_info.items():
                    if c_info.get('calibration_target') == objective:
                        self.calibration_target_locations[objective].append(waypoint)

        # Compute communication waypoints (visible from lander)
        self.communication_waypoints = set()
        if self.lander_location:
             # Waypoints visible from lander
             if self.lander_location in self.waypoint_graph:
                 self.communication_waypoints.update(self.waypoint_graph[self.lander_location])
             # The lander location itself is also a communication point
             self.communication_waypoints.add(self.lander_location)

        # Compute all-pairs shortest paths on the visible graph
        self.shortest_paths = {}
        for start_wp in self.all_waypoints:
            self.shortest_paths[start_wp] = bfs(self.waypoint_graph, start_wp, self.all_waypoints)

    def get_distance(self, start_wp, end_wp):
        """Helper to get shortest distance, returns UNREACHABLE_COST if unreachable."""
        # Check if start_wp was even processed by BFS (i.e., is a known waypoint)
        if start_wp not in self.shortest_paths:
            return UNREACHABLE_COST
        # Check if end_wp was reached from start_wp
        if end_wp not in self.shortest_paths[start_wp]:
            return UNREACHABLE_COST
        return self.shortest_paths[start_wp][end_wp]

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Parse current state to get dynamic facts
        current_rover_locations = {}
        current_have_soil = set() # (rover, waypoint)
        current_have_rock = set() # (rover, waypoint)
        current_have_image = set() # (rover, objective, mode)
        current_calibrated_cameras = set() # (camera, rover)
        current_empty_stores = set() # store

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]
            if predicate == "at":
                # Only track rover locations, not lander
                if parts[1].startswith('rover'):
                    current_rover_locations[parts[1]] = parts[2]
            elif predicate == "have_soil_analysis":
                current_have_soil.add((parts[1], parts[2]))
            elif predicate == "have_rock_analysis":
                current_have_rock.add((parts[1], parts[2]))
            elif predicate == "have_image":
                current_have_image.add((parts[1], parts[2], parts[3]))
            elif predicate == "calibrated":
                current_calibrated_cameras.add((parts[1], parts[2]))
            elif predicate == "empty":
                current_empty_stores.add(parts[1])

        total_cost = 0

        # Check each goal
        for goal in self.goals:
            if goal in state:
                continue # Goal already achieved

            parts = get_parts(goal)
            if not parts: continue

            predicate = parts[0]

            if predicate == "communicated_soil_data":
                waypoint = parts[1]
                goal_cost = 0

                # Cost for sampling
                # Check if sample is already collected by *any* rover
                already_have_sample = any((r, waypoint) in current_have_soil for r in self.rover_info)

                if not already_have_sample:
                    # Need to sample: sample_soil action + navigation + store prep
                    goal_cost += 1 # sample_soil

                    # Need an equipped rover
                    equipped_rovers = [r for r, info in self.rover_info.items() if info['equipped_soil']]
                    if not equipped_rovers: return UNREACHABLE_COST # Should not happen in solvable problems

                    # Need an empty store on an equipped rover. Add 1 if any store on any equipped rover is full.
                    needs_drop = False
                    for r in equipped_rovers:
                        for s in self.rover_info[r]['stores']:
                            if s not in current_empty_stores: # Store is full
                                needs_drop = True
                                break
                        if needs_drop: break
                    if needs_drop: goal_cost += 1 # drop action

                    # Need to navigate to sample waypoint
                    min_path_to_sample = UNREACHABLE_COST
                    for r in equipped_rovers:
                        r_loc = current_rover_locations.get(r)
                        if r_loc:
                            min_path_to_sample = min(min_path_to_sample, self.get_distance(r_loc, waypoint))
                    if min_path_to_sample == UNREACHABLE_COST: return UNREACHABLE_COST # Sample waypoint unreachable
                    goal_cost += min_path_to_sample

                # Cost for communication
                goal_cost += 1 # communicate_soil_data

                # Need to navigate to a communication waypoint
                min_path_to_comm = UNREACHABLE_COST
                if not self.communication_waypoints: return UNREACHABLE_COST # No communication points

                # Find the rover(s) that could potentially communicate.
                # If sample is already held, use the rover(s) holding it.
                # If not held, use any rover equipped for soil (as they could get it).
                potential_communicators = {r for r, wp in current_have_soil if wp == waypoint}
                if not potential_communicators:
                     potential_communicators = {r for r, info in self.rover_info.items() if info['equipped_soil']}

                if not potential_communicators: return UNREACHABLE_COST # No rover can communicate this data

                for r in potential_communicators:
                    r_loc = current_rover_locations.get(r)
                    if r_loc:
                        min_path_r_to_comm = UNREACHABLE_COST
                        for comm_wp in self.communication_waypoints:
                            min_path_r_to_comm = min(min_path_r_to_comm, self.get_distance(r_loc, comm_wp))
                        min_path_to_comm = min(min_path_to_comm, min_path_r_to_comm)

                if min_path_to_comm == UNREACHABLE_COST: return UNREACHABLE_COST # Communication points unreachable
                goal_cost += min_path_to_comm

                total_cost += goal_cost

            elif predicate == "communicated_rock_data":
                waypoint = parts[1]
                goal_cost = 0

                # Cost for sampling
                already_have_sample = any((r, waypoint) in current_have_rock for r in self.rover_info)

                if not already_have_sample:
                    goal_cost += 1 # sample_rock

                    equipped_rovers = [r for r, info in self.rover_info.items() if info['equipped_rock']]
                    if not equipped_rovers: return UNREACHABLE_COST

                    needs_drop = False
                    for r in equipped_rovers:
                        for s in self.rover_info[r]['stores']:
                            if s not in current_empty_stores:
                                needs_drop = True
                                break
                        if needs_drop: break
                    if needs_drop: goal_cost += 1 # drop action

                    min_path_to_sample = UNREACHABLE_COST
                    for r in equipped_rovers:
                        r_loc = current_rover_locations.get(r)
                        if r_loc:
                            min_path_to_sample = min(min_path_to_sample, self.get_distance(r_loc, waypoint))
                    if min_path_to_sample == UNREACHABLE_COST: return UNREACHABLE_COST
                    goal_cost += min_path_to_sample

                # Cost for communication
                goal_cost += 1 # communicate_rock_data

                min_path_to_comm = UNREACHABLE_COST
                if not self.communication_waypoints: return UNREACHABLE_COST

                potential_communicators = {r for r, wp in current_have_rock if wp == waypoint}
                if not potential_communicators:
                     potential_communicators = {r for r, info in self.rover_info.items() if info['equipped_rock']}

                if not potential_communicators: return UNREACHABLE_COST

                for r in potential_communicators:
                    r_loc = current_rover_locations.get(r)
                    if r_loc:
                        min_path_r_to_comm = UNREACHABLE_COST
                        for comm_wp in self.communication_waypoints:
                            min_path_r_to_comm = min(min_path_r_to_comm, self.get_distance(r_loc, comm_wp))
                        min_path_to_comm = min(min_path_to_comm, min_path_r_to_comm)

                if min_path_to_comm == UNREACHABLE_COST: return UNREACHABLE_COST
                goal_cost += min_path_to_comm

                total_cost += goal_cost

            elif predicate == "communicated_image_data":
                objective, mode = parts[1], parts[2]
                goal_cost = 0

                # Cost for imaging
                already_have_image = any((r, objective, mode) in current_have_image for r in self.rover_info)

                if not already_have_image:
                    goal_cost += 1 # take_image action

                    # Find suitable rover/camera/calibration target combinations
                    suitable_combos = []
                    for r, r_info in self.rover_info.items():
                        if r_info['equipped_imaging']:
                            for cam in r_info['cameras']:
                                c_info = self.camera_info.get(cam)
                                if c_info and mode in c_info.get('modes', []):
                                    suitable_combos.append((r, cam, c_info.get('calibration_target')))

                    if not suitable_combos: return UNREACHABLE_COST # No rover/camera can take this image

                    # Find waypoints visible from objective
                    image_wps = self.objective_info.get(objective, [])
                    if not image_wps: return UNREACHABLE_COST # Cannot observe objective

                    min_imaging_prep_cost_over_rovers = UNREACHABLE_COST # Cost to get to image location and calibrate

                    for r, cam, cal_target in suitable_combos:
                        r_loc = current_rover_locations.get(r)
                        if not r_loc: continue # Rover not found in state? Should not happen.

                        current_rover_prep_cost = UNREACHABLE_COST

                        if (cam, r) in current_calibrated_cameras:
                            # Camera is calibrated, just need to go to image waypoint
                            min_path_r_to_image_wps = UNREACHABLE_COST
                            for img_wp in image_wps:
                                min_path_r_to_image_wps = min(min_path_r_to_image_wps, self.get_distance(r_loc, img_wp))
                            current_rover_prep_cost = min_path_r_to_image_wps
                        else:
                            # Camera needs calibration, need to go to cal waypoint then image waypoint
                            cal_wps = self.calibration_target_locations.get(cal_target, [])
                            if not cal_wps: continue # No waypoint to calibrate from

                            min_path_r_cal_img = UNREACHABLE_COST
                            for cal_wp in cal_wps:
                                path_r_to_cal = self.get_distance(r_loc, cal_wp)
                                if path_r_to_cal == UNREACHABLE_COST: continue
                                for img_wp in image_wps:
                                    path_cal_to_img = self.get_distance(cal_wp, img_wp)
                                    if path_cal_to_img == UNREACHABLE_COST: continue
                                    min_path_r_cal_img = min(min_path_r_cal_img, path_r_to_cal + path_cal_to_img)

                            if min_path_r_cal_img != UNREACHABLE_COST:
                                 current_rover_prep_cost = 1 + min_path_r_cal_img # 1 for calibrate action

                        min_imaging_prep_cost_over_rovers = min(min_imaging_prep_cost_over_rovers, current_rover_prep_cost)

                    if min_imaging_prep_cost_over_rovers == UNREACHABLE_COST: return UNREACHABLE_COST # Cannot perform imaging prep
                    goal_cost += min_imaging_prep_cost_over_rovers

                # Cost for communication
                goal_cost += 1 # communicate_image_data

                min_path_to_comm = UNREACHABLE_COST
                if not self.communication_waypoints: return UNREACHABLE_COST

                # Find the rover(s) that could potentially communicate.
                # If image is already held, use the rover(s) holding it.
                # If not held, use any rover equipped for imaging (as they could get it).
                potential_communicators = {r for r, o, m in current_have_image if o == objective and m == mode}
                if not potential_communicators:
                     potential_communicators = {r for r, info in self.rover_info.items() if info['equipped_imaging']}

                if not potential_communicators: return UNREACHABLE_COST # No rover can communicate this data

                for r in potential_communicators:
                    r_loc = current_rover_locations.get(r)
                    if r_loc:
                        min_path_r_to_comm = UNREACHABLE_COST
                        for comm_wp in self.communication_waypoints:
                            min_path_r_to_comm = min(min_path_r_to_comm, self.get_distance(r_loc, comm_wp))
                        min_path_to_comm = min(min_path_to_comm, min_path_r_to_comm)

                if min_path_to_comm == UNREACHABLE_COST: return UNREACHABLE_COST
                goal_cost += min_path_to_comm

                total_cost += goal_cost

        return total_cost
