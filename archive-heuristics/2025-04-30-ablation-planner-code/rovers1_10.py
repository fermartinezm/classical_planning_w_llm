from fnmatch import fnmatch
from collections import deque
import math

# Assume Heuristic base class is available in heuristics/heuristic_base.py
# from heuristics.heuristic_base import Heuristic

# If running standalone for testing, uncomment this dummy class
class Heuristic:
    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static

    def __call__(self, node):
        raise NotImplementedError


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty facts or malformed strings gracefully
    if not fact or not isinstance(fact, str) or len(fact) < 2:
        return []
    # Remove surrounding parentheses and split by whitespace
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at rover1 waypoint1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Check if the number of parts matches the number of pattern arguments
    if len(parts) != len(args):
        return False
    # Check if each part matches the corresponding pattern argument
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


def bfs(graph, start_node):
    """
    Perform Breadth-First Search to find shortest distances from a start node
    in a graph represented as an adjacency list.

    Args:
        graph: A dictionary where keys are nodes and values are sets of neighbors.
        start_node: The node to start the BFS from.

    Returns:
        A dictionary mapping each reachable node to its distance from the start_node.
        Nodes not reachable will not be in the dictionary.
    """
    distances = {start_node: 0}
    queue = deque([start_node])

    while queue:
        current_node = queue.popleft()
        current_dist = distances[current_node]

        # Check if current_node exists in the graph keys before iterating neighbors
        if current_node in graph:
            for neighbor in graph[current_node]:
                if neighbor not in distances:
                    distances[neighbor] = current_dist + 1
                    queue.append(neighbor)

    return distances


class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve all
    uncommunicated goals (soil data, rock data, image data). It sums the
    estimated cost for each unachieved goal independently. The cost for each
    goal includes the final communication action, plus actions and minimum
    movement costs (based on precomputed shortest paths) required to collect
    the necessary data if it hasn't been collected yet.

    # Assumptions:
    - Movement cost between adjacent waypoints is 1.
    - The shortest path distance is a reasonable estimate for movement cost.
    - Store capacity is ignored for image goals.
    - Store capacity is ignored for soil/rock sampling goals for simplicity.
    - The heuristic assumes a capable rover/camera exists and can reach
      necessary locations if the data is not yet collected. If no such option
      exists or locations are unreachable, the goal is considered impossible
      (infinite cost).
    - The heuristic does not account for potential conflicts or synergies
      between goals (e.g., multiple goals requiring the same rover or waypoint).
    - The heuristic is non-admissible.

    # Heuristic Initialization
    The initialization phase precomputes static information from the task:
    - Identifies rover capabilities (soil, rock, imaging).
    - Maps stores to rovers.
    - Maps cameras to rovers, supported modes, and calibration targets.
    - Maps objectives/calibration targets to waypoints from which they are visible.
    - Finds the lander location and identifies communication points (waypoints
      visible from the lander).
    - Builds a traversal graph (adjacency list) for each rover based on
      `can_traverse` facts.
    - Computes all-pairs shortest path distances for each rover's graph using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic is computed as follows:
    1.  Identify all goal predicates that are not yet true in the current state.
    2.  Initialize the total heuristic cost to 0.
    3.  For each unachieved goal predicate:
        -   Estimate the cost to achieve this single goal independently.
        -   If the goal is `(communicated_soil_data W)`:
            -   Add 1 for the `communicate_soil_data` action.
            -   If `(have_soil_analysis R W)` is false for all rovers R:
                -   Add 1 for the `sample_soil` action.
                -   Find the minimum shortest path distance from any soil-capable
                    rover's current location to waypoint W. Add this distance.
            -   Else (soil data is collected by some rover):
                -   Find the minimum shortest path distance from any rover holding
                    the soil data to any communication point. Add this distance.
        -   If the goal is `(communicated_rock_data W)`:
            -   Similar logic as for soil data, using rock-specific predicates
                and capabilities.
        -   If the goal is `(communicated_image_data O M)`:
            -   Add 1 for the `communicate_image_data` action.
            -   If `(have_image R O M)` is false for all rovers R:
                -   Add 1 for the `take_image` action.
                -   Find the minimum cost across all imaging-capable rovers
                    with cameras supporting mode M to take the image:
                    -   If the camera is not calibrated: Add 1 for `calibrate`,
                        plus the minimum distance from the rover's current location
                        to a calibration waypoint, plus the minimum distance from
                        that calibration waypoint to an image waypoint.
                    -   If the camera is calibrated: Add the minimum distance from
                        the rover's current location to an image waypoint.
                    -   The minimum cost is taken over all suitable rover/camera/
                        waypoint combinations.
            -   Else (image is taken by some rover):
                -   Find the minimum shortest path distance from any rover holding
                    the image data to any communication point. Add this distance.
        -   If at any point a required location is unreachable or a capable
            rover/camera combination does not exist, the cost for this goal
            is considered infinite.
        -   Add the estimated cost for this goal to the total heuristic cost.
    4.  If the total cost becomes infinite, return infinity.
    5.  Return the total heuristic cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # --- Parse Static Facts ---
        self.rover_capabilities = {}  # rover -> set of capabilities ('soil', 'rock', 'imaging')
        self.rover_stores = {}        # rover -> list of stores (usually one)
        self.camera_info = {}         # camera -> {'rover': R, 'supports': {modes}, 'calibration_target': T}
        self.visible_from_map = {}    # objective/target -> set of waypoints
        self.lander_location = None
        self.comm_points = set()      # waypoints visible from lander location
        self.rover_graphs = {}        # rover -> {waypoint -> set of connected waypoints}
        self.rover_distances = {}     # rover -> {(start_wp, end_wp) -> distance}
        self.rovers = set()
        self.waypoints = set()
        self.cameras = set()
        self.objectives = set()
        self.modes = set()
        self.stores = set()
        self.landers = set()

        # First pass to identify objects and basic relationships
        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue

            pred = parts[0]
            if pred == 'at_lander':
                self.lander_location = parts[2] # (at_lander ?x - lander ?y - waypoint)
                self.landers.add(parts[1])
                self.waypoints.add(parts[2])
            elif pred == 'at': # Should be rover location in initial state, but also appears in state facts
                 if parts[1].startswith('rover'): self.rovers.add(parts[1])
                 self.waypoints.add(parts[2])
            elif pred == 'equipped_for_soil_analysis':
                rover = parts[1]
                self.rovers.add(rover)
                self.rover_capabilities.setdefault(rover, set()).add('soil')
            elif pred == 'equipped_for_rock_analysis':
                rover = parts[1]
                self.rovers.add(rover)
                self.rover_capabilities.setdefault(rover, set()).add('rock')
            elif pred == 'equipped_for_imaging':
                rover = parts[1]
                self.rovers.add(rover)
                self.rover_capabilities.setdefault(rover, set()).add('imaging')
            elif pred == 'store_of':
                store, rover = parts[1], parts[2]
                self.stores.add(store)
                self.rovers.add(rover)
                self.rover_stores.setdefault(rover, []).append(store)
            elif pred == 'on_board':
                camera, rover = parts[1], parts[2]
                self.cameras.add(camera)
                self.rovers.add(rover)
                self.camera_info.setdefault(camera, {})['rover'] = rover
            elif pred == 'supports':
                camera, mode = parts[1], parts[2]
                self.cameras.add(camera)
                self.modes.add(mode)
                self.camera_info.setdefault(camera, {}).setdefault('supports', set()).add(mode)
            elif pred == 'calibration_target':
                camera, target = parts[1], parts[2]
                self.cameras.add(camera)
                self.objectives.add(target) # Calibration targets are objectives
                self.camera_info.setdefault(camera, {})['calibration_target'] = target
            elif pred == 'visible_from':
                obj, wp = parts[1], parts[2]
                self.objectives.add(obj)
                self.waypoints.add(wp)
                self.visible_from_map.setdefault(obj, set()).add(wp)
            elif pred == 'visible':
                 wp1, wp2 = parts[1], parts[2]
                 self.waypoints.add(wp1)
                 self.waypoints.add(wp2)
                 # Visibility is used for communication points
            elif pred == 'can_traverse':
                rover, wp1, wp2 = parts[1], parts[2], parts[3]
                self.rovers.add(rover)
                self.waypoints.add(wp1)
                self.waypoints.add(wp2)
                self.rover_graphs.setdefault(rover, {}).setdefault(wp1, set()).add(wp2)
                # Assuming can_traverse is symmetric if not specified otherwise, but PDDL example shows both directions
                # self.rover_graphs[rover].setdefault(wp2, set()).add(wp1) # Only add if domain guarantees symmetry

        # Identify communication points (waypoints visible from lander location)
        if self.lander_location:
             for fact in static_facts:
                 if match(fact, "visible", "*", self.lander_location):
                     self.comm_points.add(get_parts(fact)[1])
                 if match(fact, "visible", self.lander_location, "*"):
                      self.comm_points.add(get_parts(fact)[2]) # Should be the same, but just in case

        # Ensure all waypoints mentioned in can_traverse are in the graph structure
        # even if they have no outgoing edges
        for rover, graph in self.rover_graphs.items():
             for wp in self.waypoints:
                  graph.setdefault(wp, set())

        # Precompute shortest path distances for each rover
        for rover, graph in self.rover_graphs.items():
            self.rover_distances[rover] = {}
            for start_wp in graph:
                distances_from_start = bfs(graph, start_wp)
                for end_wp, dist in distances_from_start.items():
                    self.rover_distances[rover][(start_wp, end_wp)] = dist

    def get_distance(self, rover, start_wp, end_wp):
        """Looks up precomputed distance or returns infinity if unreachable."""
        # Ensure rover exists and has distance info
        if rover not in self.rover_distances:
            return math.inf
        # Ensure start and end waypoints are valid and path exists
        return self.rover_distances[rover].get((start_wp, end_wp), math.inf)

    def get_min_distance_to_comm(self, rover, start_wp):
        """Finds the minimum distance from start_wp to any communication point for the given rover."""
        min_dist = math.inf
        if not self.comm_points: # No communication points defined
             return math.inf

        for comm_wp in self.comm_points:
            dist = self.get_distance(rover, start_wp, comm_wp)
            min_dist = min(min_dist, dist)
        return min_dist

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state (frozenset of fact strings)

        # --- Parse Current State ---
        current_rover_locations = {} # rover -> waypoint
        soil_samples_at_wp = set()   # waypoint with soil sample
        rock_samples_at_wp = set()   # waypoint with rock sample
        have_soil_analysis_facts = set() # (have_soil_analysis R W)
        have_rock_analysis_facts = set() # (have_rock_analysis R W)
        have_image_facts = set()         # (have_image R O M)
        full_stores = set()          # store name
        calibrated_cameras = set()   # (calibrated C R)
        communicated_soil_goals = set() # (communicated_soil_data W)
        communicated_rock_goals = set() # (communicated_rock_data W)
        communicated_image_goals = set() # (communicated_image_data O M)

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            pred = parts[0]
            if pred == 'at' and parts[1].startswith('rover'):
                current_rover_locations[parts[1]] = parts[2]
            elif pred == 'at_soil_sample':
                soil_samples_at_wp.add(parts[1])
            elif pred == 'at_rock_sample':
                rock_samples_at_wp.add(parts[1])
            elif pred == 'have_soil_analysis':
                have_soil_analysis_facts.add(fact)
            elif pred == 'have_rock_analysis':
                have_rock_analysis_facts.add(fact)
            elif pred == 'have_image':
                have_image_facts.add(fact)
            elif pred == 'full':
                full_stores.add(parts[1])
            elif pred == 'calibrated':
                calibrated_cameras.add(fact)
            elif pred == 'communicated_soil_data':
                communicated_soil_goals.add(fact)
            elif pred == 'communicated_rock_data':
                communicated_rock_goals.add(fact)
            elif pred == 'communicated_image_data':
                communicated_image_goals.add(fact)

        # --- Compute Heuristic Cost ---
        total_cost = 0

        # Iterate through all goal predicates defined in the task
        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue

            pred = parts[0]

            # Goal: Communicate Soil Data
            if pred == 'communicated_soil_data':
                waypoint_w = parts[1]
                if goal not in communicated_soil_goals:
                    goal_cost = 1 # Cost for communicate_soil_data action

                    # Check if soil analysis data is already collected by any rover
                    data_collected = any(match(fact, "have_soil_analysis", "*", waypoint_w) for fact in have_soil_analysis_facts)

                    if not data_collected:
                        # Need to sample soil
                        goal_cost += 1 # Cost for sample_soil action

                        # Find min distance for a capable rover to reach the sample waypoint
                        min_dist_to_sample = math.inf
                        capable_rover_found = False
                        for rover in self.rovers:
                            if 'soil' in self.rover_capabilities.get(rover, set()):
                                capable_rover_found = True
                                current_pos = current_rover_locations.get(rover)
                                if current_pos:
                                    dist = self.get_distance(rover, current_pos, waypoint_w)
                                    min_dist_to_sample = min(min_dist_to_sample, dist)

                        if not capable_rover_found or min_dist_to_sample == math.inf:
                            # Goal is unreachable if no capable rover or waypoint is unreachable
                            return math.inf
                        goal_cost += min_dist_to_sample
                        # Simplified: Ignore store capacity cost

                    else: # Data is collected, need to communicate
                        # Find min distance for a rover holding the data to reach a communication point
                        min_dist_to_comm = math.inf
                        data_holding_rover_found = False
                        for fact in have_soil_analysis_facts:
                            f_parts = get_parts(fact)
                            if match(fact, "have_soil_analysis", "*", waypoint_w):
                                rover = f_parts[1]
                                data_holding_rover_found = True
                                current_pos = current_rover_locations.get(rover)
                                if current_pos:
                                    dist = self.get_min_distance_to_comm(rover, current_pos)
                                    min_dist_to_comm = min(min_dist_to_comm, dist)

                        if not data_holding_rover_found or min_dist_to_comm == math.inf:
                             # Should not happen if data_collected is True, but safety check
                             return math.inf
                        goal_cost += min_dist_to_comm

                    total_cost += goal_cost

            # Goal: Communicate Rock Data
            elif pred == 'communicated_rock_data':
                 waypoint_w = parts[1]
                 if goal not in communicated_rock_goals:
                    goal_cost = 1 # Cost for communicate_rock_data action

                    # Check if rock analysis data is already collected by any rover
                    data_collected = any(match(fact, "have_rock_analysis", "*", waypoint_w) for fact in have_rock_analysis_facts)

                    if not data_collected:
                        # Need to sample rock
                        goal_cost += 1 # Cost for sample_rock action

                        # Find min distance for a capable rover to reach the sample waypoint
                        min_dist_to_sample = math.inf
                        capable_rover_found = False
                        for rover in self.rovers:
                            if 'rock' in self.rover_capabilities.get(rover, set()):
                                capable_rover_found = True
                                current_pos = current_rover_locations.get(rover)
                                if current_pos:
                                    dist = self.get_distance(rover, current_pos, waypoint_w)
                                    min_dist_to_sample = min(min_dist_to_sample, dist)

                        if not capable_rover_found or min_dist_to_sample == math.inf:
                            return math.inf
                        goal_cost += min_dist_to_sample
                        # Simplified: Ignore store capacity cost

                    else: # Data is collected, need to communicate
                        # Find min distance for a rover holding the data to reach a communication point
                        min_dist_to_comm = math.inf
                        data_holding_rover_found = False
                        for fact in have_rock_analysis_facts:
                            f_parts = get_parts(fact)
                            if match(fact, "have_rock_analysis", "*", waypoint_w):
                                rover = f_parts[1]
                                data_holding_rover_found = True
                                current_pos = current_rover_locations.get(rover)
                                if current_pos:
                                    dist = self.get_min_distance_to_comm(rover, current_pos)
                                    min_dist_to_comm = min(min_dist_to_comm, dist)

                        if not data_holding_rover_found or min_dist_to_comm == math.inf:
                             return math.inf
                        goal_cost += min_dist_to_comm

                    total_cost += goal_cost

            # Goal: Communicate Image Data
            elif pred == 'communicated_image_data':
                objective_o, mode_m = parts[1], parts[2]
                if goal not in communicated_image_goals:
                    goal_cost = 1 # Cost for communicate_image_data action

                    # Check if image data is already collected by any rover
                    data_collected = any(match(fact, "have_image", "*", objective_o, mode_m) for fact in have_image_facts)

                    if not data_collected:
                        # Need to take image
                        goal_cost += 1 # Cost for take_image action

                        # Find min cost for a capable rover/camera combo to take the image
                        min_cost_to_image = math.inf
                        capable_option_found = False

                        # Iterate through rovers equipped for imaging
                        for rover in self.rovers:
                            if 'imaging' in self.rover_capabilities.get(rover, set()):
                                current_pos = current_rover_locations.get(rover)
                                if not current_pos: continue # Rover location unknown? Should not happen.

                                # Iterate through cameras on this rover supporting the mode
                                for camera in self.cameras:
                                    cam_info = self.camera_info.get(camera)
                                    if cam_info and cam_info.get('rover') == rover and mode_m in cam_info.get('supports', set()):
                                        capable_option_found = True

                                        cal_target = cam_info.get('calibration_target')
                                        cal_wps = self.visible_from_map.get(cal_target, set()) if cal_target else set()
                                        img_wps = self.visible_from_map.get(objective_o, set())

                                        if not img_wps: continue # Cannot image this objective from anywhere

                                        cost_r_c = 0
                                        is_calibrated = f"(calibrated {camera} {rover})" in calibrated_cameras

                                        if not is_calibrated:
                                            if not cal_wps: continue # Cannot calibrate this camera

                                            cost_r_c += 1 # Calibrate action

                                            # Min dist from current pos to any calibration waypoint
                                            min_dist_to_cal = math.inf
                                            for w_cal in cal_wps:
                                                min_dist_to_cal = min(min_dist_to_cal, self.get_distance(rover, current_pos, w_cal))
                                            if min_dist_to_cal == math.inf: continue # Cannot reach any calibration point
                                            cost_r_c += min_dist_to_cal

                                            # Min dist from a calibration waypoint to an image waypoint
                                            min_dist_cal_to_img = math.inf
                                            for w_cal in cal_wps:
                                                for w_img in img_wps:
                                                    dist = self.get_distance(rover, w_cal, w_img)
                                                    if dist != math.inf:
                                                        min_dist_cal_to_img = min(min_dist_cal_to_img, dist)
                                            if min_dist_cal_to_img == math.inf: continue # Cannot reach image point after calibration
                                            cost_r_c += min_dist_cal_to_img
                                        else: # Already calibrated
                                            # Min dist from current pos to any image waypoint
                                            min_dist_to_img = math.inf
                                            for w_img in img_wps:
                                                min_dist_to_img = min(min_dist_to_img, self.get_distance(rover, current_pos, w_img))
                                            if min_dist_to_img == math.inf: continue # Cannot reach image point
                                            cost_r_c += min_dist_to_img

                                        min_cost_to_image = min(min_cost_to_image, cost_r_c)

                        if not capable_option_found or min_cost_to_image == math.inf:
                            return math.inf # Goal is unreachable
                        goal_cost += min_cost_to_image

                    else: # Image is taken, need to communicate
                        # Find min distance for a rover holding the image to reach a communication point
                        min_dist_to_comm = math.inf
                        data_holding_rover_found = False
                        for fact in have_image_facts:
                            f_parts = get_parts(fact)
                            if match(fact, "have_image", "*", objective_o, mode_m):
                                rover = f_parts[1]
                                data_holding_rover_found = True
                                current_pos = current_rover_locations.get(rover)
                                if current_pos:
                                    dist = self.get_min_distance_to_comm(rover, current_pos)
                                    min_dist_to_comm = min(min_dist_to_comm, dist)

                        if not data_holding_rover_found or min_dist_to_comm == math.inf:
                             return math.inf
                        goal_cost += min_dist_to_comm

                    total_cost += goal_cost

        # The heuristic is 0 only if all goals are achieved.
        # If total_cost is 0 here, it means the loop over unachieved goals didn't add anything,
        # which implies all goals were already in the communicated_* sets.
        return total_cost

