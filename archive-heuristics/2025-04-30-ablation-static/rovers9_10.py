from fnmatch import fnmatch
from collections import defaultdict, deque
from heuristics.heuristic_base import Heuristic

# Helper functions to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact strings or malformed facts defensively
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

# Helper function for Breadth-First Search
def bfs(graph, start_node):
    """Computes shortest path distances from start_node to all reachable nodes."""
    distances = {node: float('inf') for node in graph}
    if start_node in graph:
        distances[start_node] = 0
        queue = deque([start_node])
        while queue:
            current_node = queue.popleft()
            # Handle nodes that might be in the graph keys but have no neighbors
            if current_node in graph:
                for neighbor in graph[current_node]:
                    if distances[neighbor] == float('inf'):
                        distances[neighbor] = distances[current_node] + 1
                        queue.append(neighbor)
    return distances

def compute_all_pairs_shortest_paths(graph, all_nodes):
    """Computes shortest path distances between all pairs of nodes in the graph."""
    all_distances = {}
    # Ensure all nodes are keys in the graph for BFS, even if they have no edges
    graph_with_all_nodes = {node: graph.get(node, set()) for node in all_nodes}

    for start_node in all_nodes:
        all_distances[start_node] = bfs(graph_with_all_nodes, start_node)
    return all_distances


class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the cost to reach the goal state by summing the estimated costs
    for each unachieved communication goal. The cost for each goal is estimated based on
    whether the required data (soil sample, rock sample, or image) has already been acquired.
    If data is acquired, the cost is moving the rover with the data to a communication
    waypoint and communicating. If data is not acquired, the cost includes acquiring the
    data (sampling or imaging, potentially including calibration and dropping samples)
    plus moving to a communication waypoint and communicating. Movement costs are estimated
    using precomputed shortest path distances for each rover.

    # Assumptions:
    - The heuristic assumes that any required data can be acquired by *some* suitable rover
      and communicated. It finds the minimum cost among suitable rovers and locations for
      each individual goal requirement.
    - Resource constraints like store capacity are only partially considered (a single drop
      action is added if a rover's store is full before sampling).
    - Camera calibration status is tracked.
    - Movement cost between waypoints is the shortest path distance for the specific rover.
    - The lander stays at its initial position.
    - The heuristic sums costs for individual goals, potentially overestimating if actions
      contribute to multiple goals (non-admissible).

    # Heuristic Initialization
    - Extracts goal conditions.
    - Parses static facts and initial state facts to build data structures:
        - Rover-specific traversal graphs based on `can_traverse`.
        - All-pairs shortest path distances for each rover's graph.
        - Lander location and waypoints visible from the lander.
        - Initial locations of soil and rock samples.
        - Rover capabilities (`equipped_for_...`).
        - Store ownership (`store_of`).
        - Camera details (`on_board`, `supports`, `calibration_target`).
        - Objective visibility from waypoints (`visible_from`).
    - Identifies all unique objects (rovers, waypoints, etc.) from the facts for graph building.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current location of each rover, the status of their stores (full/empty),
       which data (soil, rock, images) each rover possesses, and which cameras are calibrated.
    2. Initialize the total heuristic cost to 0.
    3. Iterate through each goal fact `(communicated_X_data ...)` specified in the problem.
    4. If a goal fact is already true in the current state, it contributes 0 to the heuristic.
    5. If a goal fact is not true:
       a.  **Determine if the required data is already acquired:** Check if any rover has the
           necessary `(have_soil_analysis ?w)`, `(have_rock_analysis ?w)`, or `(have_image ?o ?m)` fact.
       b.  **If data IS acquired (by rover R_acq):**
           -   Estimate cost: Move R_acq from its current location to the nearest waypoint
               visible from the lander + 1 (for the `communicate_X_data` action). Use precomputed
               shortest path distances for R_acq.
       c.  **If data is NOT acquired:**
           -   Estimate the minimum cost to acquire the data using a suitable rover.
           -   **For Soil/Rock data at waypoint W:** Find a soil/rock-equipped rover R_s that
               can reach W. Minimum acquisition cost = `dist(R_s.current_pos, W)` + (1 if R_s's
               store is full) + 1 (`sample_X` action). Minimize this over all suitable R_s.
               The acquisition waypoint is W.
           -   **For Image data of Objective O in Mode M:** Find an imaging-equipped rover R_i
                with camera C supporting M. Find waypoints P visible from O.
               -   If camera C is NOT calibrated on R_i: Find calibration target T for C and
                   waypoints W_c visible from T. Calibration cost = `dist(R_i.current_pos, W_c)`
                   + 1 (`calibrate` action). The rover is then at W_c.
               -   Move to image waypoint cost = `dist(R_i.pos_after_cal, P)` + 1 (`take_image` action).
               -   Total acquisition cost = (Calibration cost if needed) + (Move to image wp cost).
               -   Minimize this over all suitable R_i, C, P (and W_c). The acquisition waypoint is P.
           -   Estimate the minimum cost to move from the acquisition waypoint (W for samples,
               P for images) to the nearest waypoint visible from the lander. Assume the rover
               that acquired the data performs this move. Use precomputed distances for that rover.
           -   Estimate total cost for this goal: (Min Acquisition Cost) + (Min Move to Comm Waypoint Cost) + 1 (for the `communicate_X_data` action).
       d.  Add the estimated cost for this unachieved goal to the total heuristic cost.
    6. Return the total heuristic cost. If any necessary step (like reaching a waypoint) is impossible
       for all suitable rovers, the cost for that goal is infinite, and the total heuristic is infinite.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals

        # Data structures to store parsed static and initial state information
        self.rover_traversal_graphs = defaultdict(lambda: defaultdict(set))
        self.lander_waypoint = None
        self.waypoints_visible_from_lander = set()
        self.soil_sample_locations = set()
        self.rock_sample_locations = set()
        self.rover_capabilities = defaultdict(set) # {rover: {'soil', 'rock', 'imaging'}}
        self.rover_stores = {} # {rover: store}
        self.rover_cameras = defaultdict(set) # {rover: {camera}}
        self.camera_modes = defaultdict(set) # {camera: {mode}}
        self.camera_calibration_targets = {} # {camera: objective}
        self.objective_visibility_waypoints = defaultdict(set) # {objective: {waypoint}}

        # Collect all unique objects by type for graph building and lookups
        self.all_rovers = set()
        self.all_waypoints = set()
        self.all_cameras = set()
        self.all_objectives = set()
        self.all_modes = set()
        self.all_stores = set()
        self.all_landers = set()

        # Parse static facts
        for fact in task.static:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            args = parts[1:]

            if predicate == 'can_traverse':
                r, w1, w2 = args
                self.rover_traversal_graphs[r][w1].add(w2)
                self.all_rovers.add(r)
                self.all_waypoints.add(w1)
                self.all_waypoints.add(w2)
            elif predicate == 'visible':
                w1, w2 = args
                # Store global visibility graph if needed, but can_traverse is rover specific
                # We only need visible from lander waypoint later
                self.all_waypoints.add(w1)
                self.all_waypoints.add(w2)
            elif predicate == 'at_lander':
                l, w = args
                self.lander_waypoint = w
                self.all_landers.add(l)
                self.all_waypoints.add(w)
            elif predicate == 'equipped_for_soil_analysis':
                self.rover_capabilities[args[0]].add('soil')
                self.all_rovers.add(args[0])
            elif predicate == 'equipped_for_rock_analysis':
                self.rover_capabilities[args[0]].add('rock')
                self.all_rovers.add(args[0])
            elif predicate == 'equipped_for_imaging':
                self.rover_capabilities[args[0]].add('imaging')
                self.all_rovers.add(args[0])
            elif predicate == 'store_of':
                s, r = args
                self.rover_stores[r] = s
                self.all_rovers.add(r)
                self.all_stores.add(s)
            elif predicate == 'on_board':
                c, r = args
                self.rover_cameras[r].add(c)
                self.all_rovers.add(r)
                self.all_cameras.add(c)
            elif predicate == 'supports':
                c, m = args
                self.camera_modes[c].add(m)
                self.all_cameras.add(c)
                self.all_modes.add(m)
            elif predicate == 'calibration_target':
                c, o = args
                self.camera_calibration_targets[c] = o
                self.all_cameras.add(c)
                self.all_objectives.add(o)
            elif predicate == 'visible_from':
                o, w = args
                self.objective_visibility_waypoints[o].add(w)
                self.all_objectives.add(o)
                self.all_waypoints.add(w)
            # Add other object types if they appear in static facts not covered above
            elif predicate == 'rover': self.all_rovers.add(args[0])
            elif predicate == 'waypoint': self.all_waypoints.add(args[0])
            elif predicate == 'store': self.all_stores.add(args[0])
            elif predicate == 'camera': self.all_cameras.add(args[0])
            elif predicate == 'mode': self.all_modes.add(args[0])
            elif predicate == 'lander': self.all_landers.add(args[0])
            elif predicate == 'objective': self.all_objectives.add(args[0])


        # Parse initial state facts for initial sample locations
        for fact in task.initial_state:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]
            args = parts[1:]

            if predicate == 'at_soil_sample':
                self.soil_sample_locations.add(args[0])
                self.all_waypoints.add(args[0])
            elif predicate == 'at_rock_sample':
                self.rock_sample_locations.add(args[0])
                self.all_waypoints.add(args[0])
            # Add other object types if they appear in initial state facts not covered above
            elif predicate == 'at':
                 obj, loc = args
                 if obj.startswith('rover'): self.all_rovers.add(obj)
                 elif obj.startswith('lander'): self.all_landers.add(obj)
                 elif obj.startswith('package'): pass # Logistics specific, ignore
                 self.all_waypoints.add(loc)
            elif predicate == 'empty': self.all_stores.add(args[0])
            elif predicate == 'full': self.all_stores.add(args[0])
            elif predicate == 'have_soil_analysis':
                 self.all_rovers.add(args[0])
                 self.all_waypoints.add(args[1])
            elif predicate == 'have_rock_analysis':
                 self.all_rovers.add(args[0])
                 self.all_waypoints.add(args[1])
            elif predicate == 'have_image':
                 self.all_rovers.add(args[0])
                 self.all_objectives.add(args[1])
                 self.all_modes.add(args[2])
            elif predicate == 'calibrated':
                 self.all_cameras.add(args[0])
                 self.all_rovers.add(args[1])
            elif predicate == 'communicated_soil_data': self.all_waypoints.add(args[0])
            elif predicate == 'communicated_rock_data': self.all_waypoints.add(args[0])
            elif predicate == 'communicated_image_data':
                 self.all_objectives.add(args[0])
                 self.all_modes.add(args[1])
            # Add other predicates if they introduce new objects

        # Compute all-pairs shortest path distances for each rover
        self.rover_distances = {}
        for rover in self.all_rovers:
             # Compute distances only if the rover has any traversal edges defined
             if rover in self.rover_traversal_graphs:
                 self.rover_distances[rover] = compute_all_pairs_shortest_paths(self.rover_traversal_graphs[rover], self.all_waypoints)
             else:
                 # If a rover cannot traverse, distance is 0 to itself, inf otherwise
                 self.rover_distances[rover] = {wp1: {wp2: (0 if wp1 == wp2 else float('inf')) for wp2 in self.all_waypoints} for wp1 in self.all_waypoints}


        # Compute waypoints visible from lander (using global visible facts)
        # Note: visible facts are static, already processed above.
        # We need to iterate through static facts again to find visible waypoints relative to lander
        if self.lander_waypoint:
            for fact in task.static:
                parts = get_parts(fact)
                if parts and parts[0] == 'visible':
                    w1, w2 = parts[1:]
                    if w1 == self.lander_waypoint:
                        self.waypoints_visible_from_lander.add(w2)
                    elif w2 == self.lander_waypoint:
                        self.waypoints_visible_from_lander.add(w1)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state (frozenset of fact strings)

        # Data structures to store current state information
        current_rover_locations = {} # {rover: waypoint}
        rover_stores_full = set() # {rover}
        rover_soil_data = defaultdict(set) # {rover: {waypoint}}
        rover_rock_data = defaultdict(set) # {rover: {waypoint}}
        rover_image_data = defaultdict(set) # {rover: {(objective, mode)}}
        rover_calibrated_cameras = defaultdict(set) # {rover: {camera}}

        # Parse current state facts
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]
            args = parts[1:]

            if predicate == 'at' and args and args[0].startswith('rover'):
                current_rover_locations[args[0]] = args[1]
            elif predicate == 'full' and args and args[0].startswith('store'):
                store_name = args[0]
                # Find which rover owns this store
                for r, s in self.rover_stores.items():
                    if s == store_name:
                        rover_stores_full.add(r)
                        break
            elif predicate == 'have_soil_analysis' and len(args) == 2:
                rover, waypoint = args
                rover_soil_data[rover].add(waypoint)
            elif predicate == 'have_rock_analysis' and len(args) == 2:
                rover, waypoint = args
                rover_rock_data[rover].add(waypoint)
            elif predicate == 'have_image' and len(args) == 3:
                rover, objective, mode = args
                rover_image_data[rover].add((objective, mode))
            elif predicate == 'calibrated' and len(args) == 2:
                camera, rover = args
                rover_calibrated_cameras[rover].add(camera)

        total_cost = 0  # Initialize action cost counter.

        # Identify which goals are already achieved
        achieved_goals = set(state)

        # Iterate through each goal fact
        for goal in self.goals:
            if goal in achieved_goals:
                continue # Goal is already satisfied

            parts = get_parts(goal)
            if not parts: continue # Skip malformed goals

            goal_type = parts[0]

            if goal_type == 'communicated_soil_data' and len(parts) == 2:
                waypoint_goal = parts[1]

                # Check if data is already acquired by any rover
                data_acquired = False
                acquiring_rover = None
                for r, waypoints_with_data in rover_soil_data.items():
                    if waypoint_goal in waypoints_with_data:
                        data_acquired = True
                        acquiring_rover = r # Assume this rover will communicate
                        break

                if data_acquired:
                    # Cost is moving the acquiring rover to a comm waypoint + communicate
                    rover_pos = current_rover_locations.get(acquiring_rover)
                    if rover_pos is None or acquiring_rover not in self.rover_distances:
                         cost = float('inf') # Rover location unknown or cannot move
                    else:
                        min_move_to_comm = float('inf')
                        if rover_pos in self.rover_distances[acquiring_rover]:
                            for w_comm in self.waypoints_visible_from_lander:
                                dist = self.rover_distances[acquiring_rover][rover_pos].get(w_comm, float('inf'))
                                min_move_to_comm = min(min_move_to_comm, dist)
                        cost = min_move_to_comm + 1 # +1 for communicate action

                else:
                    # Cost is acquiring data + moving to comm waypoint + communicate
                    min_total_goal_cost = float('inf')

                    # Find suitable rovers (equipped for soil, can reach sample waypoint)
                    suitable_rovers = [r for r, caps in self.rover_capabilities.items() if 'soil' in caps]

                    for r_s in suitable_rovers:
                         rover_pos = current_rover_locations.get(r_s)
                         if rover_pos is None or r_s not in self.rover_distances: continue # Rover location unknown or cannot move

                         if rover_pos in self.rover_distances[r_s]:
                             dist_to_sample = self.rover_distances[r_s][rover_pos].get(waypoint_goal, float('inf'))

                             if dist_to_sample != float('inf'):
                                 # Cost to sample: move + (drop if full) + sample
                                 acquisition_cost_part = dist_to_sample + (1 if r_s in rover_stores_full else 0) + 1

                                 # After sampling, rover is at waypoint_goal.
                                 # Need to move from waypoint_goal to a comm waypoint.
                                 min_move_from_sample_to_comm = float('inf')
                                 if waypoint_goal in self.rover_distances[r_s]: # Check if rover can move *from* sample waypoint
                                     for w_comm in self.waypoints_visible_from_lander:
                                         dist = self.rover_distances[r_s][waypoint_goal].get(w_comm, float('inf'))
                                         min_move_from_sample_to_comm = min(min_move_from_sample_to_comm, dist)

                                 if min_move_from_sample_to_comm != float('inf'):
                                     total_goal_cost = acquisition_cost_part + min_move_from_sample_to_comm + 1 # +1 for communicate
                                     min_total_goal_cost = min(min_total_goal_cost, total_goal_cost)

                    cost = min_total_goal_cost

                total_cost += cost

            elif goal_type == 'communicated_rock_data' and len(parts) == 2:
                waypoint_goal = parts[1]

                # Check if data is already acquired by any rover
                data_acquired = False
                acquiring_rover = None
                for r, waypoints_with_data in rover_rock_data.items():
                    if waypoint_goal in waypoints_with_data:
                        data_acquired = True
                        acquiring_rover = r
                        break

                if data_acquired:
                    rover_pos = current_rover_locations.get(acquiring_rover)
                    if rover_pos is None or acquiring_rover not in self.rover_distances:
                         cost = float('inf')
                    else:
                        min_move_to_comm = float('inf')
                        if rover_pos in self.rover_distances[acquiring_rover]:
                            for w_comm in self.waypoints_visible_from_lander:
                                dist = self.rover_distances[acquiring_rover][rover_pos].get(w_comm, float('inf'))
                                min_move_to_comm = min(min_move_to_comm, dist)
                        cost = min_move_to_comm + 1
                else:
                    min_total_goal_cost = float('inf')
                    suitable_rovers = [r for r, caps in self.rover_capabilities.items() if 'rock' in caps]

                    for r_s in suitable_rovers:
                         rover_pos = current_rover_locations.get(r_s)
                         if rover_pos is None or r_s not in self.rover_distances: continue

                         if rover_pos in self.rover_distances[r_s]:
                             dist_to_sample = self.rover_distances[r_s][rover_pos].get(waypoint_goal, float('inf'))

                             if dist_to_sample != float('inf'):
                                 acquisition_cost_part = dist_to_sample + (1 if r_s in rover_stores_full else 0) + 1 # Move + Drop + Sample

                                 min_move_from_sample_to_comm = float('inf')
                                 if waypoint_goal in self.rover_distances[r_s]:
                                     for w_comm in self.waypoints_visible_from_lander:
                                         dist = self.rover_distances[r_s][waypoint_goal].get(w_comm, float('inf'))
                                         min_move_from_sample_to_comm = min(min_move_from_sample_to_comm, dist)

                                 if min_move_from_sample_to_comm != float('inf'):
                                     total_goal_cost = acquisition_cost_part + min_move_from_sample_to_comm + 1
                                     min_total_goal_cost = min(min_total_goal_cost, total_goal_cost)

                    cost = min_total_goal_cost
                total_cost += cost

            elif goal_type == 'communicated_image_data' and len(parts) == 3:
                objective_goal, mode_goal = parts[1:]
                image_tuple_goal = (objective_goal, mode_goal)

                # Check if data is already acquired by any rover
                data_acquired = False
                acquiring_rover = None
                for r, images_data in rover_image_data.items():
                    if image_tuple_goal in images_data:
                        data_acquired = True
                        acquiring_rover = r
                        break

                if data_acquired:
                    rover_pos = current_rover_locations.get(acquiring_rover)
                    if rover_pos is None or acquiring_rover not in self.rover_distances:
                         cost = float('inf')
                    else:
                        min_move_to_comm = float('inf')
                        if rover_pos in self.rover_distances[acquiring_rover]:
                            for w_comm in self.waypoints_visible_from_lander:
                                dist = self.rover_distances[acquiring_rover][rover_pos].get(w_comm, float('inf'))
                                min_move_to_comm = min(min_move_to_comm, dist)
                        cost = min_move_to_comm + 1
                else:
                    min_total_goal_cost = float('inf')

                    # Find suitable rovers and cameras
                    # Need imaging-equipped rover R_i, camera C on R_i, C supports M
                    suitable_rover_camera_pairs = []
                    for r_i in self.all_rovers:
                        if 'imaging' in self.rover_capabilities.get(r_i, set()):
                            for c in self.rover_cameras.get(r_i, set()):
                                if mode_goal in self.camera_modes.get(c, set()):
                                    suitable_rover_camera_pairs.append((r_i, c))

                    # Find waypoints visible from the objective
                    suitable_image_waypoints = self.objective_visibility_waypoints.get(objective_goal, set())

                    if not suitable_rover_camera_pairs or not suitable_image_waypoints:
                         cost = float('inf') # Cannot take image
                    else:
                        for r_i, c in suitable_rover_camera_pairs:
                            rover_pos = current_rover_locations.get(r_i)
                            if rover_pos is None or r_i not in self.rover_distances: continue # Rover location unknown or cannot move

                            if rover_pos not in self.rover_distances[r_i]:
                                 continue # Rover cannot move from its current position

                            # Cost involves calibration (if needed) and taking the image
                            calibration_needed = c not in rover_calibrated_cameras.get(r_i, set())
                            calibration_cost_part = 0
                            calibration_waypoint = None # Where calibration happens

                            if calibration_needed:
                                # Find calibration target for this camera
                                cal_target = self.camera_calibration_targets.get(c)
                                if cal_target is None:
                                     calibration_cost_part = float('inf') # Cannot calibrate
                                else:
                                    # Find waypoints visible from calibration target
                                    suitable_cal_waypoints = self.objective_visibility_waypoints.get(cal_target, set())
                                    if not suitable_cal_waypoints:
                                         calibration_cost_part = float('inf') # Cannot calibrate
                                    else:
                                        min_move_to_cal = float('inf')
                                        best_cal_wp = None
                                        # Rover starts from its current position for calibration move
                                        start_pos_for_cal_move = rover_pos
                                        if start_pos_for_cal_move in self.rover_distances[r_i]:
                                            for w_c in suitable_cal_waypoints:
                                                 dist = self.rover_distances[r_i][start_pos_for_cal_move].get(w_c, float('inf'))
                                                 if dist != float('inf'):
                                                     if dist < min_move_to_cal:
                                                         min_move_to_cal = dist
                                                         best_cal_wp = w_c

                                        if min_move_to_cal == float('inf'):
                                             calibration_cost_part = float('inf') # Cannot reach any cal waypoint
                                        else:
                                             calibration_cost_part = min_move_to_cal + 1 # Move + Calibrate
                                             calibration_waypoint = best_cal_wp # Rover is at this waypoint after calibration

                            if calibration_cost_part == float('inf'):
                                 continue # Cannot calibrate, so cannot take image

                            # Cost involves moving to image waypoint and taking image
                            min_image_cost_part = float('inf')
                            best_image_wp = None

                            # Rover starts from rover_pos if no calibration, or from calibration_waypoint if calibrated
                            start_pos_for_image_move = calibration_waypoint if calibration_needed else rover_pos

                            if start_pos_for_image_move not in self.rover_distances.get(r_i, {}):
                                 continue # Rover cannot move from this position

                            for p in suitable_image_waypoints:
                                 dist_to_image_wp = self.rover_distances[r_i][start_pos_for_image_move].get(p, float('inf'))
                                 if dist_to_image_wp != float('inf'):
                                     image_cost_part = dist_to_image_wp + 1 # Move + Take Image
                                     if image_cost_part < min_image_cost_part:
                                         min_image_cost_part = image_cost_part
                                         best_image_wp = p # Rover is at this waypoint after taking image

                            if min_image_cost_part == float('inf'):
                                 continue # Cannot reach any image waypoint

                            # Total acquisition cost for this rover/camera/path combination
                            acquisition_cost = calibration_cost_part + min_image_cost_part

                            # Cost to move from image waypoint to comm waypoint
                            min_move_from_image_to_comm = float('inf')
                            if best_image_wp in self.rover_distances.get(r_i, {}):
                                for w_comm in self.waypoints_visible_from_lander:
                                    dist = self.rover_distances[r_i][best_image_wp].get(w_comm, float('inf'))
                                    min_move_from_image_to_comm = min(min_move_from_image_to_comm, dist)

                            if min_move_from_image_to_comm != float('inf'):
                                total_goal_cost = acquisition_cost + min_move_from_image_to_comm + 1 # +1 for communicate
                                min_total_goal_cost = min(min_total_goal_cost, total_goal_cost)

                        cost = min_total_goal_cost

                total_cost += cost

            # If at any point the cost for a goal is infinite, the total cost is infinite
            if total_cost == float('inf'):
                return float('inf')

        # The heuristic is 0 if and only if all goals are achieved (loop finishes with total_cost = 0)
        return total_cost

