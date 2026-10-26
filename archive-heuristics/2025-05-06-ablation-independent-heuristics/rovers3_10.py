from heuristics.heuristic_base import Heuristic
from task import Task
from collections import defaultdict, deque
import math

# Helper parsing functions
def _parse_fact(fact_string):
    """Removes leading/trailing parens and splits by space."""
    # Ensure fact_string is a string and has expected format
    if not isinstance(fact_string, str) or not fact_string.startswith('(') or not fact_string.endswith(')'):
        # Handle unexpected format, maybe log a warning or raise an error
        # For this problem, we assume valid PDDL fact strings
        return None, []
    parts = fact_string[1:-1].split()
    if not parts:
        return None, []
    return parts[0], parts[1:]

def _parse_at_fact(fact_string):
    _, rover, waypoint = _parse_fact(fact_string)
    return rover, waypoint

def _parse_at_lander_fact(fact_string):
    _, lander, waypoint = _parse_fact(fact_string)
    return lander, waypoint

def _parse_equipped_fact(fact_string):
    pred, args = _parse_fact(fact_string)
    # Example: 'equipped_for_soil_analysis' -> 'soil_analysis'
    capability = '_'.join(pred.split('_')[2:])
    rover = args[0]
    return rover, capability

def _parse_store_fact(fact_string):
    _, store, rover = _parse_fact(fact_string)
    return store, rover

def _parse_on_board_fact(fact_string):
    _, camera, rover = _parse_fact(fact_string)
    return camera, rover

def _parse_supports_fact(fact_string):
    _, camera, mode = _parse_fact(fact_string)
    return camera, mode

def _parse_calibration_target_fact(fact_string):
    _, camera, objective = _parse_fact(fact_string)
    return camera, objective

def _parse_visible_from_fact(fact_string):
    _, objective, waypoint = _parse_fact(fact_string)
    return objective, waypoint

def _parse_visible_fact(fact_string):
    _, wp1, wp2 = _parse_fact(fact_string)
    return wp1, wp2

def _parse_can_traverse_fact(fact_string):
    _, rover, wp1, wp2 = _parse_fact(fact_string)
    return rover, wp1, wp2

def _parse_soil_goal(goal_string):
    _, waypoint = _parse_fact(goal_string)
    return waypoint[0]

def _parse_rock_goal(goal_string):
    _, waypoint = _parse_fact(goal_string)
    return waypoint[0]

def _parse_image_goal(goal_string):
    _, objective, mode = _parse_fact(goal_string)
    return objective, mode

def _parse_full_fact(fact_string):
    _, store = _parse_fact(fact_string)
    return store[0]

def _parse_calibrated_fact(fact_string):
    _, camera, rover = _parse_fact(fact_string)
    return camera, rover

def _parse_sample_fact(fact_string):
    _, rover, waypoint = _parse_fact(fact_string)
    return rover, waypoint

def _parse_image_fact(fact_string):
    _, rover, objective, mode = _parse_fact(fact_string)
    return rover, objective, mode

# BFS helper function
def bfs(graph, start_node, all_nodes):
    """
    Performs BFS on a graph to find shortest distances from start_node to all other nodes.
    Args:
        graph: Adjacency list representation (dict: node -> set of neighbors).
        start_node: The starting node for BFS.
        all_nodes: A set of all possible nodes in the graph.
    Returns:
        A dictionary mapping each node in all_nodes to its shortest distance from start_node.
    """
    distances = {node: math.inf for node in all_nodes}

    if start_node not in all_nodes:
         # Start node is not a known waypoint
         return distances

    distances[start_node] = 0
    queue = deque([start_node])

    while queue:
        current_node = queue.popleft()
        current_dist = distances[current_node]

        # Check if current_node has outgoing edges in the graph
        if current_node in graph:
            for neighbor in graph[current_node]:
                if distances[neighbor] == math.inf:
                    distances[neighbor] = current_dist + 1
                    queue.append(neighbor)
    return distances


class roversHeuristic(Heuristic):
    """
    Summary:
        Domain-dependent heuristic for the rovers domain. Estimates the total
        number of actions required to achieve all unachieved goals by summing
        up the estimated costs for each individual goal. The cost for each goal
        is estimated by finding the minimum cost sequence of actions (navigate,
        sample/image, communicate) for the most suitable rover (and camera).
        Navigation costs are estimated using precomputed shortest paths on the
        rover's traversal graph.

    Assumptions:
        - The problem instance is solvable.
        - Rovers are always located at some waypoint in any reachable state.
        - Soil/rock samples required by goals are present at the specified
          waypoints in the initial state.
        - Calibration targets required by cameras are present as objectives
          and visible from some waypoints.
        - Lander is located at some waypoint.
        - All necessary objects (rovers, cameras, stores, waypoints, objectives,
          modes, landers) exist and are correctly typed and related in the
          initial/static state.
        - The 'can_traverse' predicate defines a connected graph for relevant
          rovers between necessary waypoints, or the heuristic will return
          infinity for unreachable goals.

    Heuristic Initialization:
        The constructor precomputes static information from the task definition
        and initial state, including:
        - Lander location(s).
        - Rover capabilities (soil, rock, imaging).
        - Rover-store mapping.
        - Rover-camera mapping.
        - Camera-mode support mapping.
        - Camera-calibration target mapping.
        - Objective-visible-from waypoint mapping.
        - Waypoint visibility graph (for communication points).
        - Rover-specific traversal graphs ('can_traverse').
        - Initial locations of soil and rock samples.

        It then precomputes shortest path distances between all pairs of
        waypoints for each rover's traversal graph using BFS. It also identifies
        sets of waypoints relevant for communication (visible from lander),
        imaging (visible from objectives), and calibration (visible from
        calibration targets).

    Step-By-Step Thinking for Computing Heuristic:
        1.  Get the current state and the set of goal facts.
        2.  Identify the set of unachieved goal facts. If this set is empty,
            the heuristic value is 0.
        3.  Extract dynamic information from the current state: current rover
            locations, which stores are full, which cameras are calibrated,
            which samples have been collected, and which images have been taken.
        4.  Initialize the total heuristic value `h` to 0.
        5.  For each unachieved goal fact:
            a.  Parse the goal fact to identify its type (soil, rock, image)
                and parameters (waypoint, objective, mode).
            b.  Estimate the minimum cost to achieve this specific goal from
                the current state, considering all available and capable rovers
                (and cameras for image goals).
            c.  For a soil/rock goal `(communicated_soil_data/rock_data ?w)`:
                i.  Find all rovers equipped for the required analysis.
                ii. For each capable rover `?r`:
                    - Get its current location `?current_w`.
                    - Calculate the cost:
                        - Navigation to sample waypoint `?w`: Shortest path from `?current_w` to `?w` using `?r`'s graph.
                        - Sample action: 1. Add 1 if `?r`'s store is currently full.
                        - Navigation to communication waypoint: Shortest path from `?w` to any waypoint visible from the lander using `?r`'s graph.
                        - Communicate action: 1.
                    - Sum these costs. If any navigation is impossible, the cost for this rover is infinity.
                iii. Find the minimum cost among all capable rovers. If no capable rover or minimum cost is infinity, the goal is unreachable (return infinity for the total heuristic).
            d.  For an image goal `(communicated_image_data ?o ?m)`:
                i.  Find all rovers equipped for imaging.
                ii. For each capable rover `?r`:
                    - Find all cameras `?i` on board `?r` that support mode `?m`.
                    - For each suitable camera `?i`:
                        - Get `?r`'s current location `?current_w`.
                        - Calculate the cost:
                            - Navigation to imaging waypoint: Shortest path from `?current_w` to any waypoint from which `?o` is visible using `?r`'s graph. Let the chosen waypoint be `?img_w`.
                            - Calibration cost: If camera `?i` is not calibrated, add 1 (calibrate action). If the chosen `?img_w` is not a valid calibration location for `?i`, add navigation cost from `?img_w` to any valid calibration location for `?i` using `?r`'s graph.
                            - Take image action: 1.
                            - Navigation to communication waypoint: Shortest path from `?img_w` to any waypoint visible from the lander using `?r`'s graph.
                            - Communicate action: 1.
                        - Sum these costs. If any navigation is impossible, the cost for this rover/camera is infinity.
                    - Find the minimum cost among all suitable cameras for rover `?r`. If no suitable camera or minimum cost is infinity, the cost for this rover is infinity.
                iii. Find the minimum cost among all capable rovers. If no capable rover or minimum cost is infinity, the goal is unreachable (return infinity for the total heuristic).
            e.  Add the minimum estimated cost for the current goal to the total `h`.
        6.  Return the total heuristic value `h`.
    """

    def __init__(self, task):
        super().__init__()
        self.goals = task.goals
        self.static = task.static
        self.all_waypoints = set() # Collect all waypoints

        # --- Parse Static Information ---
        self.lander_location = {} # lander -> waypoint (Assuming one lander)
        self.rover_capabilities = defaultdict(set) # rover -> set of capabilities ('soil_analysis', 'rock_analysis', 'imaging')
        self.rover_stores = {} # rover -> store (Assuming one store per rover)
        self.rover_cameras = defaultdict(set) # rover -> set of cameras
        self.camera_modes = defaultdict(set) # camera -> set of modes
        self.camera_calibration_target = {} # camera -> objective (Assuming one target per camera)
        self.objective_visible_from = defaultdict(set) # objective -> set of waypoints
        self.waypoint_visibility_graph = defaultdict(set) # waypoint -> set of visible waypoints (for communication)
        self.rover_traversal_graphs = defaultdict(lambda: defaultdict(set)) # rover -> waypoint -> set of reachable waypoints (1 step)
        self.soil_sample_locations_init = set() # waypoint with soil sample initially
        self.rock_sample_locations_init = set() # waypoint with rock sample initially

        # Collect all waypoints from static and initial state
        all_facts = set(self.static) | set(task.initial_state)
        for fact in all_facts:
             if fact.startswith('(at_lander '):
                lander, waypoint = _parse_at_lander_fact(fact)
                self.lander_location[lander] = waypoint
                self.all_waypoints.add(waypoint)
             elif fact.startswith('(at '):
                 _, rover, waypoint = _parse_at_fact(fact)
                 self.all_waypoints.add(waypoint)
             elif fact.startswith('(visible_from '):
                objective, waypoint = _parse_visible_from_fact(fact)
                self.objective_visible_from[objective].add(waypoint)
                self.all_waypoints.add(waypoint)
             elif fact.startswith('(visible '):
                wp1, wp2 = _parse_visible_fact(fact)
                self.waypoint_visibility_graph[wp1].add(wp2)
                self.all_waypoints.add(wp1)
                self.all_waypoints.add(wp2)
             elif fact.startswith('(can_traverse '):
                rover, wp1, wp2 = _parse_can_traverse_fact(fact)
                self.rover_traversal_graphs[rover][wp1].add(wp2)
                self.all_waypoints.add(wp1)
                self.all_waypoints.add(wp2)
             elif fact.startswith('(at_soil_sample '):
                _, waypoint = _parse_fact(fact)
                self.soil_sample_locations_init.add(waypoint[0])
                self.all_waypoints.add(waypoint[0])
             elif fact.startswith('(at_rock_sample '):
                _, waypoint = _parse_fact(fact)
                self.rock_sample_locations_init.add(waypoint[0])
                self.all_waypoints.add(waypoint[0])
             # Parse other static facts
             elif fact.startswith('(equipped_for_'):
                rover, capability = _parse_equipped_fact(fact)
                self.rover_capabilities[rover].add(capability)
             elif fact.startswith('(store_of '):
                store, rover = _parse_store_fact(fact)
                self.rover_stores[rover] = store
             elif fact.startswith('(on_board '):
                camera, rover = _parse_on_board_fact(fact)
                self.rover_cameras[rover].add(camera)
             elif fact.startswith('(supports '):
                camera, mode = _parse_supports_fact(fact)
                self.camera_modes[camera].add(mode)
             elif fact.startswith('(calibration_target '):
                camera, objective = _parse_calibration_target_fact(fact)
                self.camera_calibration_target[camera] = objective


        # --- Precompute Shortest Paths ---
        self.rover_shortest_paths = {} # rover -> start_wp -> end_wp -> distance
        for rover in self.rover_traversal_graphs.keys(): # Iterate over rovers that have traversal info
            graph = self.rover_traversal_graphs[rover]
            self.rover_shortest_paths[rover] = {}
            # Compute paths from *every* known waypoint
            for start_wp in self.all_waypoints:
                 self.rover_shortest_paths[rover][start_wp] = bfs(graph, start_wp, self.all_waypoints)

        # --- Precompute Useful Waypoint Sets ---
        self.comm_waypoint_set = set()
        # Assuming there is at least one lander
        if self.lander_location:
            lander_wp = list(self.lander_location.values())[0]
            # Find waypoints visible *from* the lander location (reversed visible predicate)
            # Need to iterate through all visible facts to find wps visible from lander_wp
            for fact in self.static:
                if fact.startswith('(visible '):
                    wp1, wp2 = _parse_visible_fact(fact)
                    if wp2 == lander_wp:
                        self.comm_waypoint_set.add(wp1)

        self.objective_imaging_waypoint_sets = {} # objective -> set of waypoints
        for obj, wps in self.objective_visible_from.items():
             self.objective_imaging_waypoint_sets[obj] = wps

        self.camera_calibration_waypoint_sets = {} # camera -> set of waypoints
        for camera, obj in self.camera_calibration_target.items():
             self.camera_calibration_waypoint_sets[camera] = self.objective_visible_from.get(obj, set())


    def get_shortest_path(self, rover, start_wp, end_wp):
        """Helper to get precomputed shortest path."""
        if rover not in self.rover_shortest_paths or \
           start_wp not in self.rover_shortest_paths[rover] or \
           end_wp not in self.rover_shortest_paths[rover][start_wp]:
            return math.inf # Rover doesn't exist or start_wp not in its path table or end_wp unreachable
        return self.rover_shortest_paths[rover][start_wp][end_wp]

    def get_shortest_path_to_set(self, rover, start_wp, end_wp_set):
        """Helper to get shortest path to any waypoint in a set."""
        if not end_wp_set:
            return math.inf
        min_dist = math.inf
        for end_wp in end_wp_set:
            dist = self.get_shortest_path(rover, start_wp, end_wp)
            min_dist = min(min_dist, dist)
        return min_dist

    def get_shortest_path_to_set_with_target(self, rover, start_wp, end_wp_set):
        """Helper to get shortest path to any waypoint in a set, returning dist and target."""
        if not end_wp_set:
            return math.inf, None
        min_dist = math.inf
        target_wp = None
        for end_wp in end_wp_set:
            dist = self.get_shortest_path(rover, start_wp, end_wp)
            if dist < min_dist:
                min_dist = dist
                target_wp = end_wp
        return min_dist, target_wp


    def __call__(self, node):
        state = node.state
        unachieved_goals = self.goals - state

        if not unachieved_goals:
            return 0 # Goal reached

        # --- Extract Dynamic Information from State ---
        rover_locations = {} # rover -> waypoint
        rover_stores_full = {} # rover -> True if full
        camera_calibrated = {} # camera -> True if calibrated

        for fact in state:
            if fact.startswith('(at '):
                rover, waypoint = _parse_at_fact(fact)
                rover_locations[rover] = waypoint
            elif fact.startswith('(full '):
                store = _parse_full_fact(fact)
                # Find which rover this store belongs to
                for r, s in self.rover_stores.items():
                    if s == store:
                        rover_stores_full[r] = True
                        break
            elif fact.startswith('(calibrated '):
                camera, rover = _parse_calibrated_fact(fact)
                camera_calibrated[camera] = True
            # Note: We don't need to track have_soil/rock/image or at_soil/rock_sample
            # because the heuristic estimates cost from current state to goal state,
            # assuming initial sample locations are fixed and have_X facts are
            # temporary steps towards communication goals.

        h = 0
        for goal in unachieved_goals:
            goal_cost = math.inf

            if goal.startswith('(communicated_soil_data '):
                waypoint_w = _parse_soil_goal(goal)
                # Check if sample was initially available (basic solvability check)
                if waypoint_w not in self.soil_sample_locations_init:
                     return math.inf # Goal requires sample that wasn't initially there

                capable_rovers = [r for r, caps in self.rover_capabilities.items() if 'soil_analysis' in caps]
                if not capable_rovers:
                    return math.inf # No rover can do soil analysis

                min_rover_cost = math.inf
                for rover in capable_rovers:
                    current_w = rover_locations.get(rover)
                    if current_w is None: continue # Rover location unknown (shouldn't happen in valid states)

                    # Cost to reach sample
                    dist_to_sample = self.get_shortest_path(rover, current_w, waypoint_w)
                    if dist_to_sample == math.inf: continue # Cannot reach sample waypoint

                    # Cost to sample (includes dropping if store is full)
                    sample_cost = 1 + (1 if rover_stores_full.get(rover, False) else 0)

                    # Cost to reach communication point
                    dist_to_comm = self.get_shortest_path_to_set(rover, waypoint_w, self.comm_waypoint_set)
                    if dist_to_comm == math.inf: continue # Cannot reach any communication waypoint

                    # Cost to communicate
                    comm_cost = 1

                    current_rover_cost = dist_to_sample + sample_cost + dist_to_comm + comm_cost
                    min_rover_cost = min(min_rover_cost, current_rover_cost)

                goal_cost = min_rover_cost

            elif goal.startswith('(communicated_rock_data '):
                waypoint_w = _parse_rock_goal(goal)
                 # Check if sample was initially available
                if waypoint_w not in self.rock_sample_locations_init:
                     return math.inf # Goal requires sample that wasn't initially there

                capable_rovers = [r for r, caps in self.rover_capabilities.items() if 'rock_analysis' in caps]
                if not capable_rovers:
                    return math.inf # No rover can do rock analysis

                min_rover_cost = math.inf
                for rover in capable_rovers:
                    current_w = rover_locations.get(rover)
                    if current_w is None: continue # Rover location unknown

                    # Cost to reach sample
                    dist_to_sample = self.get_shortest_path(rover, current_w, waypoint_w)
                    if dist_to_sample == math.inf: continue # Cannot reach sample waypoint

                    # Cost to sample (includes dropping if store is full)
                    sample_cost = 1 + (1 if rover_stores_full.get(rover, False) else 0)

                    # Cost to reach communication point
                    dist_to_comm = self.get_shortest_path_to_set(rover, waypoint_w, self.comm_waypoint_set)
                    if dist_to_comm == math.inf: continue # Cannot reach any communication waypoint

                    # Cost to communicate
                    comm_cost = 1

                    current_rover_cost = dist_to_sample + sample_cost + dist_to_comm + comm_cost
                    min_rover_cost = min(min_rover_cost, current_rover_cost)

                goal_cost = min_rover_cost

            elif goal.startswith('(communicated_image_data '):
                objective_o, mode_m = _parse_image_goal(goal)

                capable_rovers = [r for r, caps in self.rover_capabilities.items() if 'imaging' in caps]
                if not capable_rovers:
                    return math.inf # No rover can do imaging

                min_rover_cost = math.inf
                for rover in capable_rovers:
                    current_w = rover_locations.get(rover)
                    if current_w is None: continue # Rover location unknown

                    suitable_cameras = [c for c in self.rover_cameras.get(rover, []) if mode_m in self.camera_modes.get(c, [])]
                    if not suitable_cameras: continue # Rover has no suitable camera for this mode

                    min_camera_cost = math.inf
                    for camera in suitable_cameras:
                        # Find imaging waypoints for objective
                        img_wps = self.objective_imaging_waypoint_sets.get(objective_o, set())
                        if not img_wps: continue # Objective not visible from anywhere

                        # Cost to reach imaging point
                        dist_to_img_wp, img_wp = self.get_shortest_path_to_set_with_target(rover, current_w, img_wps)
                        if dist_to_img_wp == math.inf: continue # Cannot reach any imaging waypoint

                        # Cost to take image
                        take_cost = 1

                        # Cost to calibrate
                        cal_cost = 0
                        if not camera_calibrated.get(camera, False):
                            cal_cost += 1 # Calibrate action cost
                            cal_wps = self.camera_calibration_waypoint_sets.get(camera, set())
                            if not cal_wps: continue # Camera has no calibration target or target not visible from anywhere
                            if img_wp not in cal_wps:
                                # Need to navigate to a calibration waypoint first
                                dist_to_cal_wp = self.get_shortest_path_to_set(rover, img_wp, cal_wps)
                                if dist_to_cal_wp == math.inf: continue # Cannot reach any calibration waypoint
                                cal_cost += dist_to_cal_wp
                                # Note: We don't add cost to navigate back to img_wp for simplicity (non-admissible)

                        # Cost to reach communication point
                        dist_to_comm = self.get_shortest_path_to_set(rover, img_wp, self.comm_waypoint_set)
                        if dist_to_comm == math.inf: continue # Cannot reach communication point

                        # Cost to communicate
                        comm_cost = 1

                        current_camera_cost = dist_to_img_wp + cal_cost + take_cost + dist_to_comm + comm_cost
                        min_camera_cost = min(min_camera_cost, current_camera_cost)

                    if min_camera_cost != math.inf:
                        min_rover_cost = min(min_rover_cost, min_camera_cost)

                goal_cost = min_rover_cost

            # Add the minimum cost for this goal to the total heuristic
            if goal_cost == math.inf:
                 # If any goal is unreachable, the whole state is likely a dead end
                 # or the problem is unsolvable. Return infinity.
                 return math.inf
            h += goal_cost

        return h
