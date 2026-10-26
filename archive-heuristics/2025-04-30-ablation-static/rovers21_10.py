from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic

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

def bfs(graph, start_node, all_nodes):
    """
    Performs Breadth-First Search to find shortest distances from a start node
    in a graph.

    Args:
        graph: Adjacency list representation {node: [neighbor1, ...]}
        start_node: The starting node for the BFS.
        all_nodes: A set of all possible nodes in the graph.

    Returns:
        A dictionary {node: distance} containing shortest distances from start_node.
        Distance is float('inf') if a node is unreachable.
    """
    distances = {node: float('inf') for node in all_nodes}
    if start_node not in all_nodes:
         # Start node is not in the graph (e.g., rover not at a known waypoint)
         # This case shouldn't happen in valid states, but handle defensively.
         return distances

    distances[start_node] = 0
    queue = deque([start_node])

    while queue:
        current_node = queue.popleft()
        # Ensure current_node is in graph keys before accessing neighbors
        if current_node in graph:
            for neighbor in graph[current_node]:
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)
    return distances


class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to satisfy all
    goal conditions. It breaks down each unsatisfied communication goal
    into necessary steps (collecting data, moving to communication location,
    communicating) and sums the estimated costs for each goal independently.
    Navigation costs are estimated using precomputed shortest path distances
    on the waypoint graph for each rover. Other action costs (sample, image,
    calibrate, drop, communicate) are counted as 1.

    # Assumptions
    - The heuristic assumes a solvable problem instance. Unreachable goals
      are assigned a large penalty.
    - Navigation costs are based on shortest paths assuming any `can_traverse`
      fact implies a bidirectional edge unless explicitly stated otherwise
      (the domain file suggests symmetric visibility and can_traverse).
    - The cost of satisfying different goals is independent (e.g., collecting
      one sample doesn't affect the cost of collecting another, except for
      store usage). This is a relaxation for efficiency.
    - Calibration is required before each image taken with that camera.
    - Initial sample locations (`at_soil_sample`, `at_rock_sample`) are static
      facts indicating where samples *were* initially. The sample action removes
      these facts. The heuristic assumes a sample goal `(communicated_soil_data W)`
      is only possible if `(at_soil_sample W)` was true in the initial state.

    # Heuristic Initialization
    The heuristic precomputes static information from the task:
    - The lander's waypoint.
    - The set of waypoints visible from the lander's waypoint.
    - Rover capabilities (soil, rock, imaging).
    - Rover-store mapping.
    - Camera information (on-board rover, supported modes, calibration target).
    - Objective information (waypoints visible from the objective).
    - Initial sample locations (soil and rock).
    - The set of all waypoints in the problem.
    - Connectivity graph for each rover based on `can_traverse` facts.
    - Shortest path distances between all pairs of waypoints for each rover
      using BFS on their respective connectivity graphs.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic calculates the cost as follows:

    1. Initialize total cost to 0.
    2. Identify the current state of the world: rover locations, store contents,
       camera calibration status, and collected data (`have_soil_analysis`,
       `have_rock_analysis`, `have_image`).
    3. Iterate through each goal condition defined in the task.
    4. If a goal condition is already satisfied in the current state, add 0 cost
       for this goal and continue to the next goal.
    5. If a goal condition is not satisfied, estimate the minimum cost to satisfy it:
       - **For `(communicated_soil_data W)`:**
         - Check if `(have_soil_analysis R W)` is true for any rover R in the current state.
           - If yes: The cost is 1 (communicate action) plus the minimum navigation cost
             for rover R from its current location to any waypoint visible from the lander.
           - If no: Check if `W` was an initial soil sample location. If not, this goal
             is likely impossible (add a large penalty). If yes, find a soil-equipped
             rover R. The cost is the minimum over all such R:
             (1 if R's store is full else 0 for drop action) +
             navigation cost from R's current location to W +
             1 (sample action) +
             navigation cost from W to any waypoint visible from the lander +
             1 (communicate action).
         - Take the minimum cost found across all applicable scenarios and rovers.
       - **For `(communicated_rock_data W)`:**
         - Similar logic as soil data, using rock-specific predicates and locations.
       - **For `(communicated_image_data O M)`:**
         - Check if `(have_image R O M)` is true for any rover R in the current state.
           - If yes: The cost is 1 (communicate action) plus the minimum navigation cost
             for rover R from its current location to any waypoint visible from the lander.
           - If no: Find an imaging-equipped rover R with a camera I that is on-board R
             and supports mode M. The cost is the minimum over all such R/I pairs:
             minimum navigation cost from R's current location to any calibration waypoint
             for camera I +
             1 (calibrate action) +
             minimum navigation cost from any calibration waypoint for I to any imaging
             waypoint for objective O +
             1 (take_image action) +
             minimum navigation cost from any imaging waypoint for O to any waypoint
             visible from the lander +
             1 (communicate action).
             (Note: The calibration cost is always included if the image is not yet taken,
             as calibration is consumed by the take_image action).
         - Take the minimum cost found across all applicable scenarios and rovers/cameras.
       - If no path or suitable rover/camera exists for a goal, add a large penalty
         (e.g., 1000) to the total cost.
    6. Sum the minimum estimated costs for all unsatisfied goals to get the total
       heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static_facts = task.static

        # --- Extract Static Information ---
        self.lander_waypoint = None
        self.lander_visible_waypoints = set()
        self.rover_capabilities = {} # {rover: {cap1, cap2, ...}}
        self.rover_stores = {}       # {rover: store}
        self.camera_info = {}        # {camera: {rover: r, modes: {m1, ...}, cal_target: t}}
        self.objective_info = {}     # {objective: {visible_from: {wp1, ...}}}
        self.sample_locations = {'soil': set(), 'rock': set()} # {type: {wp1, ...}}
        self.waypoints = set()       # Set of all waypoints
        self.rover_connectivity = {} # {rover: {wp: [neighbor1, ...], ...}}

        # Collect all waypoints first
        for fact in self.static_facts:
            parts = get_parts(fact)
            if not parts: continue
            pred = parts[0]
            if pred in ['at_lander', 'at', 'visible', 'can_traverse', 'at_soil_sample', 'at_rock_sample', 'visible_from']:
                 for part in parts[1:]:
                     # Simple check if it looks like a waypoint (starts with 'waypoint')
                     # This might need refinement based on actual object types if available
                     # but based on examples, 'waypoint' prefix is common.
                     # A more robust way would be to parse object types from the domain/instance.
                     # For this problem, let's rely on the common naming convention.
                     if part.startswith('waypoint'):
                         self.waypoints.add(part)

        # Parse static facts
        for fact in self.static_facts:
            parts = get_parts(fact)
            if not parts: continue
            pred = parts[0]

            if pred == 'at_lander':
                # Assuming only one lander
                self.lander_waypoint = parts[2]
            elif pred == 'visible':
                wp1, wp2 = parts[1], parts[2]
                if wp2 == self.lander_waypoint:
                    self.lander_visible_waypoints.add(wp1)
            elif pred.startswith('equipped_for_'):
                rover = parts[1]
                capability = pred
                self.rover_capabilities.setdefault(rover, set()).add(capability)
            elif pred == 'store_of':
                store, rover = parts[1], parts[2]
                self.rover_stores[rover] = store
            elif pred == 'on_board':
                camera, rover = parts[1], parts[2]
                self.camera_info.setdefault(camera, {})['rover'] = rover
            elif pred == 'supports':
                camera, mode = parts[1], parts[2]
                self.camera_info.setdefault(camera, {}).setdefault('modes', set()).add(mode)
            elif pred == 'calibration_target':
                camera, objective = parts[1], parts[2]
                self.camera_info.setdefault(camera, {})['cal_target'] = objective
            elif pred == 'visible_from':
                objective, waypoint = parts[1], parts[2]
                self.objective_info.setdefault(objective, {}).setdefault('visible_from', set()).add(waypoint)
            elif pred == 'at_soil_sample':
                self.sample_locations['soil'].add(parts[1])
            elif pred == 'at_rock_sample':
                self.sample_locations['rock'].add(parts[1])
            elif pred == 'can_traverse':
                rover, wp1, wp2 = parts[1], parts[2], parts[3]
                self.rover_connectivity.setdefault(rover, {}).setdefault(wp1, []).append(wp2)
                # Assuming bidirectional traverse if visible is bidirectional
                # The domain file shows visible is often symmetric, let's assume can_traverse is too.
                self.rover_connectivity[rover].setdefault(wp2, []).append(wp1)

        # Precompute shortest path distances for each rover
        self.rover_distances = {}
        for rover, graph in self.rover_connectivity.items():
            self.rover_distances[rover] = {}
            for start_wp in self.waypoints: # Compute distances from all waypoints
                 self.rover_distances[rover][start_wp] = bfs(graph, start_wp, self.waypoints)

        # Penalty for impossible goals or unreachable locations
        self.UNREACHABLE_PENALTY = 1000 # Use a large finite number

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        total_cost = 0

        # --- Extract Current State Information ---
        current_rover_locations = {} # {rover: waypoint}
        current_store_status = {}    # {store: 'empty' or 'full'}
        current_calibrated_cameras = set() # {(camera, rover)}
        current_have_soil = set()    # {(rover, waypoint)}
        current_have_rock = set()    # {(rover, waypoint)}
        current_have_image = set()   # {(rover, objective, mode)}

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue
            pred = parts[0]

            if pred == 'at':
                obj, loc = parts[1], parts[2]
                # Assuming anything starting with 'rover' is a rover
                if obj.startswith('rover'):
                    current_rover_locations[obj] = loc
            elif pred in ['empty', 'full']:
                 store = parts[1]
                 current_store_status[store] = pred
            elif pred == 'calibrated':
                 camera, rover = parts[1], parts[2]
                 current_calibrated_cameras.add((camera, rover))
            elif pred == 'have_soil_analysis':
                 rover, wp = parts[1], parts[2]
                 current_have_soil.add((rover, wp))
            elif pred == 'have_rock_analysis':
                 rover, wp = parts[1], parts[2]
                 current_have_rock.add((rover, wp))
            elif pred == 'have_image':
                 rover, obj, mode = parts[1], parts[2], parts[3]
                 current_have_image.add((rover, obj, mode))

        # --- Estimate Cost for Each Unsatisfied Goal ---
        for goal in self.goals:
            if goal in state:
                continue # Goal already achieved

            parts = get_parts(goal)
            if not parts: continue
            pred = parts[0]

            goal_cost = float('inf') # Minimum cost to achieve this specific goal

            if pred == 'communicated_soil_data':
                w = parts[1]
                # Scenario 1: Data already collected
                min_cost_if_have = float('inf')
                for rover, wp_data in current_have_soil:
                    if wp_data == w:
                        # Rover has the data, needs to communicate
                        rover_loc = current_rover_locations.get(rover)
                        if rover_loc and rover_loc in self.rover_distances.get(rover, {}):
                            # Find min nav cost to a lander-visible waypoint
                            min_nav_to_lander = float('inf')
                            for lw in self.lander_visible_waypoints:
                                dist = self.rover_distances[rover][rover_loc].get(lw, float('inf'))
                                min_nav_to_lander = min(min_nav_to_lander, dist)

                            if min_nav_to_lander != float('inf'):
                                cost = 1 + min_nav_to_lander # 1 for communicate + nav
                                min_cost_if_have = min(min_cost_if_have, cost)

                if min_cost_if_have != float('inf'):
                    goal_cost = min(goal_cost, min_cost_if_have)

                # Scenario 2: Data needs to be collected
                if w in self.sample_locations['soil']: # Check if sample was initially available
                    min_cost_if_sample = float('inf')
                    # Find suitable rovers
                    for rover, capabilities in self.rover_capabilities.items():
                        if 'equipped_for_soil_analysis' in capabilities:
                            rover_loc = current_rover_locations.get(rover)
                            store = self.rover_stores.get(rover)

                            if rover_loc and store and rover_loc in self.rover_distances.get(rover, {}):
                                # Cost to ensure store is empty
                                store_cost = 1 if current_store_status.get(store) == 'full' else 0

                                # Nav cost to sample location W
                                nav1 = self.rover_distances[rover][rover_loc].get(w, float('inf'))

                                # Nav cost from sample location W to lander-visible waypoint
                                min_nav2_from_sample = float('inf')
                                if w in self.rover_distances.get(rover, {}):
                                    for lw in self.lander_visible_waypoints:
                                        dist = self.rover_distances[rover][w].get(lw, float('inf'))
                                        min_nav2_from_sample = min(min_nav2_from_sample, dist)

                                if nav1 != float('inf') and min_nav2_from_sample != float('inf'):
                                    cost = store_cost + nav1 + 1 + min_nav2_from_sample + 1 # drop + nav1 + sample + nav2 + communicate
                                    min_cost_if_sample = min(min_cost_if_sample, cost)

                    if min_cost_if_sample != float('inf'):
                         goal_cost = min(goal_cost, min_cost_if_sample)
                # else: sample location W was not in initial state, goal is impossible

            elif pred == 'communicated_rock_data':
                w = parts[1]
                # Scenario 1: Data already collected
                min_cost_if_have = float('inf')
                for rover, wp_data in current_have_rock:
                    if wp_data == w:
                        # Rover has the data, needs to communicate
                        rover_loc = current_rover_locations.get(rover)
                        if rover_loc and rover_loc in self.rover_distances.get(rover, {}):
                            min_nav_to_lander = float('inf')
                            for lw in self.lander_visible_waypoints:
                                dist = self.rover_distances[rover][rover_loc].get(lw, float('inf'))
                                min_nav_to_lander = min(min_nav_to_lander, dist)

                            if min_nav_to_lander != float('inf'):
                                cost = 1 + min_nav_to_lander # 1 for communicate + nav
                                min_cost_if_have = min(min_cost_if_have, cost)

                if min_cost_if_have != float('inf'):
                    goal_cost = min(goal_cost, min_cost_if_have)

                # Scenario 2: Data needs to be collected
                if w in self.sample_locations['rock']: # Check if sample was initially available
                    min_cost_if_sample = float('inf')
                    # Find suitable rovers
                    for rover, capabilities in self.rover_capabilities.items():
                        if 'equipped_for_rock_analysis' in capabilities:
                            rover_loc = current_rover_locations.get(rover)
                            store = self.rover_stores.get(rover)

                            if rover_loc and store and rover_loc in self.rover_distances.get(rover, {}):
                                # Cost to ensure store is empty
                                store_cost = 1 if current_store_status.get(store) == 'full' else 0

                                # Nav cost to sample location W
                                nav1 = self.rover_distances[rover][rover_loc].get(w, float('inf'))

                                # Nav cost from sample location W to lander-visible waypoint
                                min_nav2_from_sample = float('inf')
                                if w in self.rover_distances.get(rover, {}):
                                    for lw in self.lander_visible_waypoints:
                                        dist = self.rover_distances[rover][w].get(lw, float('inf'))
                                        min_nav2_from_sample = min(min_nav2_from_sample, dist)

                                if nav1 != float('inf') and min_nav2_from_sample != float('inf'):
                                    cost = store_cost + nav1 + 1 + min_nav2_from_sample + 1 # drop + nav1 + sample + nav2 + communicate
                                    min_cost_if_sample = min(min_cost_if_sample, cost)

                    if min_cost_if_sample != float('inf'):
                         goal_cost = min(goal_cost, min_cost_if_sample)
                # else: sample location W was not in initial state, goal is impossible

            elif pred == 'communicated_image_data':
                o, m = parts[1], parts[2]
                # Scenario 1: Image already collected
                min_cost_if_have = float('inf')
                for rover, obj_data, mode_data in current_have_image:
                    if obj_data == o and mode_data == m:
                        # Rover has the image, needs to communicate
                        rover_loc = current_rover_locations.get(rover)
                        if rover_loc and rover_loc in self.rover_distances.get(rover, {}):
                            min_nav_to_lander = float('inf')
                            for lw in self.lander_visible_waypoints:
                                dist = self.rover_distances[rover][rover_loc].get(lw, float('inf'))
                                min_nav_to_lander = min(min_nav_to_lander, dist)

                            if min_nav_to_lander != float('inf'):
                                cost = 1 + min_nav_to_lander # 1 for communicate + nav
                                min_cost_if_have = min(min_cost_if_have, cost)

                if min_cost_if_have != float('inf'):
                    goal_cost = min(goal_cost, min_cost_if_have)

                # Scenario 2: Image needs to be collected
                min_cost_if_image = float('inf')
                # Find suitable rovers and cameras
                for camera, cam_info in self.camera_info.items():
                    rover = cam_info.get('rover')
                    supported_modes = cam_info.get('modes', set())
                    cal_target = cam_info.get('cal_target')

                    if (rover and m in supported_modes and
                        rover in self.rover_capabilities and
                        'equipped_for_imaging' in self.rover_capabilities[rover]):

                        rover_loc = current_rover_locations.get(rover)
                        if not rover_loc or rover_loc not in self.rover_distances.get(rover, {}):
                            continue # Rover location unknown or not in graph

                        # Find calibration waypoints for this camera/target
                        cal_waypoints = set()
                        if cal_target and cal_target in self.objective_info:
                             cal_waypoints = self.objective_info[cal_target].get('visible_from', set())

                        # Find imaging waypoints for this objective
                        img_waypoints = set()
                        if o in self.objective_info:
                            img_waypoints = self.objective_info[o].get('visible_from', set())

                        if not cal_waypoints or not img_waypoints or not self.lander_visible_waypoints:
                            continue # Cannot calibrate, image, or communicate

                        # Nav cost to calibration waypoint
                        min_nav1_to_cal = float('inf')
                        for cwp in cal_waypoints:
                            dist = self.rover_distances[rover][rover_loc].get(cwp, float('inf'))
                            min_nav1_to_cal = min(min_nav1_to_cal, dist)

                        # Nav cost from calibration waypoint to imaging waypoint
                        min_nav2_cal_to_img = float('inf')
                        for cwp in cal_waypoints:
                            if cwp in self.rover_distances.get(rover, {}):
                                for iwp in img_waypoints:
                                    dist = self.rover_distances[rover][cwp].get(iwp, float('inf'))
                                    min_nav2_cal_to_img = min(min_nav2_cal_to_img, dist)

                        # Nav cost from imaging waypoint to lander-visible waypoint
                        min_nav3_img_to_lander = float('inf')
                        for iwp in img_waypoints:
                            if iwp in self.rover_distances.get(rover, {}):
                                for lw in self.lander_visible_waypoints:
                                    dist = self.rover_distances[rover][iwp].get(lw, float('inf'))
                                    min_nav3_img_to_lander = min(min_nav3_img_to_lander, dist)

                        if (min_nav1_to_cal != float('inf') and
                            min_nav2_cal_to_img != float('inf') and
                            min_nav3_img_to_lander != float('inf')):

                            # Cost = nav(to cal) + calibrate + nav(cal to img) + take_image + nav(img to lander) + communicate
                            cost = min_nav1_to_cal + 1 + min_nav2_cal_to_img + 1 + min_nav3_img_to_lander + 1
                            min_cost_if_image = min(min_cost_if_image, cost)

                if min_cost_if_image != float('inf'):
                    goal_cost = min(goal_cost, min_cost_if_image)
                # else: no suitable rover/camera, goal is impossible

            # Add the cost for this goal, using penalty if impossible
            if goal_cost == float('inf'):
                 total_cost += self.UNREACHABLE_PENALTY
            else:
                 total_cost += goal_cost

        return total_cost

