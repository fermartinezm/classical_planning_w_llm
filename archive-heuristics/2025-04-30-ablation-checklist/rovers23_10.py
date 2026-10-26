from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic
import sys

# Define a large value to represent infinity for unreachable states
INF = float('inf')

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty strings or malformed facts gracefully
    if not fact or not isinstance(fact, str) or len(fact) < 2:
        return []
    # Remove outer parentheses and split by whitespace
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at rover1 waypoint1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Check if the number of parts matches the number of arguments in the pattern
    if len(parts) != len(args):
        return False
    # Use fnmatch for wildcard matching on each part
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve all
    unmet goal conditions. It decomposes the problem by goal type (soil, rock, image)
    and estimates the minimum cost for each, summing them up. Navigation costs
    are estimated using precomputed shortest paths on the traversability graph
    for each rover. Other action costs (sample, drop, calibrate, take_image, communicate)
    are estimated based on preconditions and dependencies.

    # Heuristic Initialization
    - Parses static facts to identify objects, lander location, rover capabilities,
      camera information, visibility facts, and communication waypoints.
    - Precomputes shortest path distances between all pairs of waypoints for each
      rover using BFS based on the `can_traverse` facts.
    - Stores initial locations of soil and rock samples from the initial state.

    # Step-By-Step Thinking for Computing Heuristic (for a given state)
    1. Initialize total heuristic cost to 0.
    2. Identify the current location of each rover and the status of cameras (calibrated)
       and stores (full) from the current state.
    3. Identify which `have_soil_analysis`, `have_rock_analysis`, and `have_image`
       facts are true in the current state.
    4. Iterate through each goal fact specified in the task:
       - If the goal fact is already true in the current state, add 0 cost for this goal.
       - If the goal fact is `(communicated_soil_data ?w)` and not in state:
         - Estimate the minimum cost to achieve `(communicated_soil_data ?w)`.
         - This cost is the minimum of two options, over all suitable rovers:
           a) If `(have_soil_analysis ?r ?w)` is already true for rover `?r`:
              Cost = `nav_cost(?r, current_loc(?r), comm_wp) + communicate_cost`.
              Minimize `nav_cost` over all communication waypoints `comm_wp`.
           b) If `(have_soil_analysis ?r ?w)` is not true, but `(at_soil_sample ?w)`
              was in the initial state and rover `?r` is equipped for soil:
              Cost = `nav_cost(?r, current_loc(?r), ?w) + sample_cost + (drop_cost if store full) + nav_cost(?r, ?w, comm_wp) + communicate_cost`.
              Minimize `nav_cost` over all communication waypoints `comm_wp`.
         - Add the minimum estimated cost for this goal to the total. If no path is possible, return infinity.
       - If the goal fact is `(communicated_rock_data ?w)` and not in state:
         - Similar logic to soil data, using rock analysis and rock samples.
       - If the goal fact is `(communicated_image_data ?o ?m)` and not in state:
         - Estimate the minimum cost to achieve `(communicated_image_data ?o ?m)`.
         - This cost is the minimum of two options, over all suitable rovers `?r`
           with a camera `?i` supporting mode `?m`:
           a) If `(have_image ?r ?o ?m)` is already true:
              Cost = `nav_cost(?r, current_loc(?r), comm_wp) + communicate_cost`.
              Minimize `nav_cost` over all communication waypoints `comm_wp`.
           b) If `(have_image ?r ?o ?m)` is not true, and rover `?r` is equipped for imaging
              with camera `?i` supporting `?m`, and objective `?o` is visible from some
              waypoint `?p`, and camera `?i` has a calibration target `?t` visible
              from some waypoint `?w_cal`:
              Cost to take image:
                - If `(calibrated ?i ?r)` is true: `nav_cost(?r, current_loc(?r), ?p) + take_image_cost`.
                - If `(calibrated ?i ?r)` is false: `nav_cost(?r, current_loc(?r), ?w_cal) + calibrate_cost + nav_cost(?r, ?w_cal, ?p) + take_image_cost`.
                Minimize navigation costs over suitable `?p` and `?w_cal`.
              Cost to communicate: `nav_cost(?r, ?p, comm_wp) + communicate_cost`.
              Minimize `nav_cost` over suitable `?p` and `comm_wp`.
              Total cost = Cost to take image + Cost to communicate.
         - Add the minimum estimated cost for this goal to the total. If no path is possible, return infinity.
    5. Return the total estimated cost.

    # Action Costs (assumed 1 for all actions in this domain)
    - navigate: 1
    - sample_soil: 1
    - sample_rock: 1
    - drop: 1
    - calibrate: 1
    - take_image: 1
    - communicate_soil_data: 1
    - communicate_rock_data: 1
    - communicate_image_data: 1
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information and precomputing
        navigation distances.
        """
        self.goals = task.goals
        self.static_facts = task.static
        self.initial_state = task.initial_state # Need initial state for sample locations

        # --- Parse Static and Initial Facts ---
        all_facts = set(self.initial_state) | set(self.static_facts)

        # Extract objects by type (parsing from facts is safer than assuming types)
        self.rovers = {p[1] for p in map(get_parts, all_facts) if len(p) > 1 and p[0] == 'at' and p[1].startswith('rover')}
        self.waypoints = {p[2] for p in map(get_parts, all_facts) if len(p) > 2 and p[0] in ['at', 'at_lander', 'can_traverse', 'visible', 'at_soil_sample', 'at_rock_sample', 'visible_from']}
        self.landers = {p[1] for p in map(get_parts, all_facts) if len(p) > 1 and p[0] == 'at_lander'}
        self.cameras = {p[1] for p in map(get_parts, all_facts) if len(p) > 1 and p[0] in ['on_board', 'calibration_target', 'supports']}
        self.objectives = {p[2] for p in map(get_parts, all_facts) if len(p) > 2 and p[0] in ['calibration_target', 'visible_from']}
        self.modes = {p[2] for p in map(get_parts, all_facts) if len(p) > 2 and p[0] == 'supports'}
        self.stores = {p[1] for p in map(get_parts, all_facts) if len(p) > 1 and p[0] == 'store_of'}

        # Lander location (assuming one lander)
        lander_fact = next((f for f in self.static_facts if match(f, 'at_lander', '*', '*')), None)
        self.lander_location = get_parts(lander_fact)[2] if lander_fact else None

        # Rover capabilities and equipment
        self.rover_capabilities = {}
        for r in self.rovers:
            self.rover_capabilities[r] = {
                'soil': f'(equipped_for_soil_analysis {r})' in self.static_facts,
                'rock': f'(equipped_for_rock_analysis {r})' in self.static_facts,
                'imaging': f'(equipped_for_imaging {r})' in self.static_facts,
                'store': next((s for s in self.stores if f'(store_of {s} {r})' in self.static_facts), None),
                'cameras': [c for c in self.cameras if f'(on_board {c} {r})' in self.static_facts]
            }

        # Camera information
        self.camera_info = {}
        for c in self.cameras:
            self.camera_info[c] = {
                'supports': [get_parts(f)[2] for f in self.static_facts if match(f, 'supports', c, '*')],
                'calibration_target': next((get_parts(f)[2] for f in self.static_facts if match(f, 'calibration_target', c, '*')), None)
            }

        # Visibility information
        self.visible_from_objective = {} # objective -> [waypoints]
        for o in self.objectives:
            self.visible_from_objective[o] = [get_parts(f)[2] for f in self.static_facts if match(f, 'visible_from', o, '*')]

        self.visible_from_calibration_target = {} # objective (target) -> [waypoints]
        for c in self.cameras:
             target = self.camera_info[c]['calibration_target']
             if target:
                 self.visible_from_calibration_target[target] = [get_parts(f)[2] for f in self.static_facts if match(f, 'visible_from', target, '*')]

        # Communication waypoints (visible from lander)
        self.comm_waypoints = [get_parts(f)[1] for f in self.static_facts if self.lander_location and match(f, 'visible', '*', self.lander_location)]

        # Initial sample locations (needed to know if sampling is possible)
        self.initial_soil_samples = {get_parts(f)[1] for f in self.initial_state if match(f, 'at_soil_sample', '*')}
        self.initial_rock_samples = {get_parts(f)[1] for f in self.initial_state if match(f, 'at_rock_sample', '*')}

        # --- Precompute Navigation Distances ---
        self._precompute_distances()

    def _precompute_distances(self):
        """
        Builds navigation graphs for each rover and computes all-pairs shortest paths
        using BFS. Stores results in self.distances[rover][start_wp][end_wp].
        """
        self.distances = {} # distances[rover][start_wp][end_wp]

        for rover in self.rovers:
            graph = {}
            for wp in self.waypoints:
                graph[wp] = []
            for fact in self.static_facts:
                if match(fact, 'can_traverse', rover, '*', '*'):
                    _, r, wp1, wp2 = get_parts(fact)
                    graph[wp1].append(wp2)

            self.distances[rover] = {}
            for start_wp in self.waypoints:
                self.distances[rover][start_wp] = self._bfs(graph, start_wp)

    def _bfs(self, graph, start_node):
        """
        Performs Breadth-First Search on a graph to find distances from a start node.
        """
        distances = {node: INF for node in graph}
        if start_node not in graph: # Handle cases where start_node might not be in the graph (e.g., isolated waypoint)
             return distances

        distances[start_node] = 0
        queue = deque([start_node])

        while queue:
            current_node = queue.popleft()
            for neighbor in graph.get(current_node, []):
                if distances[neighbor] == INF:
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)
        return distances

    def get_distance(self, rover, start_wp, end_wp):
         """Retrieves precomputed distance, returns INF if not found."""
         return self.distances.get(rover, {}).get(start_wp, {}).get(end_wp, INF)

    def __call__(self, node):
        """
        Computes the domain-dependent heuristic value for the given state.
        """
        state = node.state
        total_cost = 0

        # --- Extract State-Dependent Information ---
        rover_locations = {}
        for fact in state:
            if match(fact, 'at', '*', '*'):
                _, r, wp = get_parts(fact)
                if r in self.rovers:
                    rover_locations[r] = wp

        calibrated_cameras = {get_parts(f)[1] for f in state if match(f, 'calibrated', '*', '*')}
        full_stores = {get_parts(f)[1] for f in state if match(f, 'full', '*')}

        have_soil = {(get_parts(f)[2], get_parts(f)[3]) for f in state if match(f, 'have_soil_analysis', '*', '*')} # (rover, waypoint)
        have_rock = {(get_parts(f)[2], get_parts(f)[3]) for f in state if match(f, 'have_rock_analysis', '*', '*')} # (rover, waypoint)
        have_image = {(get_parts(f)[2], get_parts(f)[3], get_parts(f)[4]) for f in state if match(f, 'have_image', '*', '*', '*')} # (rover, objective, mode)

        # --- Estimate Cost for Each Unachieved Goal ---
        for goal in self.goals:
            if goal in state:
                continue # Goal already achieved

            predicate, *args = get_parts(goal)

            if predicate == 'communicated_soil_data':
                waypoint = args[0]
                min_goal_cost = INF

                # Option 1: Data is already collected by some rover
                rovers_with_data = [r for r, w in have_soil if w == waypoint]
                for r in rovers_with_data:
                     curr_wp = rover_locations.get(r)
                     if curr_wp:
                         min_nav_to_comm = min((self.get_distance(r, curr_wp, comm_wp) for comm_wp in self.comm_waypoints), default=INF)
                         if min_nav_to_comm != INF:
                             cost = min_nav_to_comm + 1 # communicate action cost
                             min_goal_cost = min(min_goal_cost, cost)

                # Option 2: Data needs to be collected and then communicated
                # Check if sample exists initially at the waypoint
                if waypoint in self.initial_soil_samples:
                    equipped_rovers = [r for r, caps in self.rover_capabilities.items() if caps['soil']]
                    for r in equipped_rovers:
                        curr_wp = rover_locations.get(r)
                        store = self.rover_capabilities[r]['store']
                        if curr_wp and store:
                            # Cost to sample
                            nav_to_sample_cost = self.get_distance(r, curr_wp, waypoint)
                            if nav_to_sample_cost == INF: continue

                            sample_action_cost = 1 # sample_soil action cost
                            if store in full_stores:
                                sample_action_cost += 1 # drop action cost if store is full

                            # Cost to communicate after sampling
                            nav_to_comm_cost = min((self.get_distance(r, waypoint, comm_wp) for comm_wp in self.comm_waypoints), default=INF)
                            if nav_to_comm_cost == INF: continue

                            cost = nav_to_sample_cost + sample_action_cost + nav_to_comm_cost + 1 # communicate action cost
                            min_goal_cost = min(min_goal_cost, cost)

                if min_goal_cost == INF:
                    # If a goal is unreachable, return infinity. Greedy search won't expand this path.
                    return INF
                total_cost += min_goal_cost


            elif predicate == 'communicated_rock_data':
                waypoint = args[0]
                min_goal_cost = INF

                # Option 1: Data is already collected by some rover
                rovers_with_data = [r for r, w in have_rock if w == waypoint]
                for r in rovers_with_data:
                     curr_wp = rover_locations.get(r)
                     if curr_wp:
                         min_nav_to_comm = min((self.get_distance(r, curr_wp, comm_wp) for comm_wp in self.comm_waypoints), default=INF)
                         if min_nav_to_comm != INF:
                             cost = min_nav_to_comm + 1 # communicate
                             min_goal_cost = min(min_goal_cost, cost)

                # Option 2: Data needs to be collected and then communicated
                if waypoint in self.initial_rock_samples: # Check if sample exists initially
                    equipped_rovers = [r for r, caps in self.rover_capabilities.items() if caps['rock']]
                    for r in equipped_rovers:
                        curr_wp = rover_locations.get(r)
                        store = self.rover_capabilities[r]['store']
                        if curr_wp and store:
                            # Cost to sample
                            nav_to_sample_cost = self.get_distance(r, curr_wp, waypoint)
                            if nav_to_sample_cost == INF: continue

                            sample_action_cost = 1 # sample_rock action cost
                            if store in full_stores:
                                sample_action_cost += 1 # drop action cost

                            # Cost to communicate after sampling
                            nav_to_comm_cost = min((self.get_distance(r, waypoint, comm_wp) for comm_wp in self.comm_waypoints), default=INF)
                            if nav_to_comm_cost == INF: continue

                            cost = nav_to_sample_cost + sample_action_cost + nav_to_comm_cost + 1 # communicate action cost
                            min_goal_cost = min(min_goal_cost, cost)

                if min_goal_cost == INF:
                    return INF
                total_cost += min_goal_cost


            elif predicate == 'communicated_image_data':
                objective, mode = args
                min_goal_cost = INF

                # Option 1: Image is already taken by some rover
                rovers_with_image = [r for r, o, m in have_image if o == objective and m == mode]
                for r in rovers_with_image:
                     curr_wp = rover_locations.get(r)
                     if curr_wp:
                         min_nav_to_comm = min((self.get_distance(r, curr_wp, comm_wp) for comm_wp in self.comm_waypoints), default=INF)
                         if min_nav_to_comm != INF:
                             cost = min_nav_to_comm + 1 # communicate action cost
                             min_goal_cost = min(min_goal_cost, cost)

                # Option 2: Image needs to be taken and then communicated
                equipped_rovers = [r for r, caps in self.rover_capabilities.items() if caps['imaging']]
                for r in equipped_rovers:
                    curr_wp = rover_locations.get(r)
                    if not curr_wp: continue # Rover location unknown

                    # Find suitable cameras on this rover that support the mode
                    suitable_cameras = [c for c in self.rover_capabilities[r]['cameras'] if mode in self.camera_info.get(c, {}).get('supports', [])]
                    if not suitable_cameras: continue # No camera on this rover supports the mode

                    # Find waypoints where objective is visible
                    image_wps = self.visible_from_objective.get(objective, [])
                    if not image_wps: continue # Cannot take image of this objective from anywhere

                    for camera in suitable_cameras:
                        cal_target = self.camera_info[camera].get('calibration_target')
                        cal_wps = self.visible_from_calibration_target.get(cal_target, []) if cal_target else []

                        # Cost to take image with this camera
                        cost_take = INF

                        # Case 1: Camera is currently calibrated
                        if camera in calibrated_cameras:
                            nav_to_image_cost = min((self.get_distance(r, curr_wp, p) for p in image_wps), default=INF)
                            if nav_to_image_cost != INF:
                                cost_take = nav_to_image_cost + 1 # take_image action cost
                        # Case 2: Camera needs calibration
                        elif cal_wps: # Must have calibration waypoints for this camera's target
                            # Find best sequence: curr -> cal_wp -> image_wp
                            min_nav_cal_image = INF
                            for w_cal in cal_wps:
                                nav_curr_to_cal = self.get_distance(r, curr_wp, w_cal)
                                if nav_curr_to_cal == INF: continue
                                for p in image_wps:
                                    nav_cal_to_image = self.get_distance(r, w_cal, p)
                                    if nav_cal_to_image == INF: continue
                                    min_nav_cal_image = min(min_nav_cal_image, nav_curr_to_cal + nav_cal_to_image)

                            if min_nav_cal_image != INF:
                                 cost_take = min_nav_cal_image + 1 # calibrate action cost
                                 cost_take += 1 # take_image action cost

                        if cost_take == INF: continue # Cannot take image with this camera (unreachable waypoints or no cal target/wp)

                        # Cost to communicate after taking image
                        # Assume rover is at one of the image_wps after taking the image.
                        # We need the min cost from *any* image_wp to *any* comm_wp.
                        min_nav_image_to_comm = INF
                        for p in image_wps:
                            min_nav_image_to_comm = min(min_nav_image_to_comm, min((self.get_distance(r, p, comm_wp) for comm_wp in self.comm_waypoints), default=INF))

                        if min_nav_image_to_comm == INF: continue # Cannot communicate from image waypoints

                        cost = cost_take + min_nav_image_to_comm + 1 # communicate action cost
                        min_goal_cost = min(min_goal_cost, cost)

                if min_goal_cost == INF:
                    return INF
                total_cost += min_goal_cost

            # Add other goal types if necessary (though rovers domain goals are usually just communicated_*)
            # else:
            #     # Handle other potential goal predicates if they exist and are not achieved
            #     # For this domain, we only expect communicated_* goals.
            #     # If an unexpected unachieved goal exists, the heuristic might be incomplete.
            #     # Returning INF or a large number indicates this goal is not handled.
            #     # For simplicity in this domain, we assume only communicated_* goals.
            #     pass


        return total_cost

