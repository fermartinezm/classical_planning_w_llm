import collections

class roversHeuristic:
    """
    Domain-dependent heuristic for the Rovers domain.

    Summary:
    The heuristic estimates the total number of actions required to reach the goal
    state by summing up the estimated costs for each unachieved goal fact.
    For soil/rock data goals, the cost is estimated based on navigating to the
    sample location (if needed), sampling, and then navigating to a communication
    point and communicating.
    For image data goals, the cost is estimated based on navigating to a calibration
    location, calibrating, navigating to an image location, taking the image,
    and then navigating to a communication point and communicating.
    Navigation costs are estimated using shortest paths on the rover-specific
    traversal graph. The minimum cost is taken over available rovers and relevant
    waypoints.

    Assumptions:
    - If a soil or rock sample is required for a goal but is no longer at its
      initial location and no rover currently holds the analysis, the goal is
      considered unreachable from this state (sample is assumed lost).
    - If an objective or calibration target required for an image goal has no
      visible_from waypoints, the image goal is considered unreachable.
    - If a camera required for an image goal has no calibration target, the
      image goal is considered unreachable with that camera.
    - If a rover needs to sample but all its stores are full, it needs one
      'drop' action first (assuming it has something to drop).
    - The navigation graph for each rover is defined by the 'can_traverse' facts.
      'visible' facts are used only to identify communication points.
    - Shortest path distances are precomputed using BFS.

    Heuristic Initialization:
    The constructor precomputes static information from the task definition:
    - Lander locations.
    - Rover capabilities (equipped for soil, rock, imaging).
    - Store ownership.
    - Initial locations of soil and rock samples.
    - Camera information (on board, supported modes, calibration target).
    - Objective visible_from waypoints.
    - Communication points (waypoints visible from lander locations).
    - Rover-specific navigation graphs based on 'can_traverse' facts.
    - All-pairs shortest paths for each rover on its navigation graph using BFS.

    Step-By-Step Thinking for Computing Heuristic:
    1. Check if the current state is a goal state. If yes, return 0.
    2. Initialize the total heuristic cost to 0.
    3. Parse the current state to extract dynamic facts such as rover locations,
       store contents, 'have' predicates (soil, rock, image), and calibration status.
    4. Iterate through each goal fact specified in the task.
    5. If a goal fact is already true in the current state, skip it.
    6. If the goal is `(communicated_soil_data ?w)`:
       - Find the minimum estimated cost to achieve this goal among all rovers
         equipped for soil analysis.
       - For each suitable rover `?r`:
         - If `(have_soil_analysis ?r ?w)` is true in the state: The cost is
           estimated as the navigation cost from `?r`'s current location to
           the nearest communication point plus 1 (for the communicate action).
         - If `(at_soil_sample ?w)` was true initially (checked during init):
           The cost is estimated as the navigation cost from `?r`'s current
           location to `?w` plus 1 (for sample action), plus 1 if `?r`'s store
           is full (for drop action), plus the navigation cost from `?w` to
           the nearest communication point plus 1 (for communicate action).
         - If neither condition is met, this rover cannot achieve this goal.
       - Take the minimum cost over all suitable rovers. If no rover can achieve
         the goal, the heuristic for this goal is infinity.
       - Add this minimum cost to the total heuristic.
    7. If the goal is `(communicated_rock_data ?w)`: Follow a similar process as
       for soil data goals, using rock-specific predicates and capabilities.
    8. If the goal is `(communicated_image_data ?o ?m)`:
       - Find the minimum estimated cost among all rover/camera combinations
         where the rover is equipped for imaging and the camera is on board
         and supports mode `?m`.
       - For each suitable rover `?r` and camera `?i`:
         - If `(have_image ?r ?o ?m)` is true in the state: The cost is estimated
           as the navigation cost from `?r`'s current location to the nearest
           communication point plus 1 (for the communicate action).
         - If false: The cost is estimated based on the sequence: navigate to
           calibration waypoint, calibrate, navigate to image waypoint, take
           image, navigate to communication waypoint, communicate. The cost is
           minimized over all valid calibration waypoints (`?w`), image waypoints
           (`?p`), and communication waypoints (`?c`):
           `min (dist(?r.location, ?w) + 1 + dist(?w, ?p) + 1 + dist(?p, ?c) + 1)`.
           If no valid sequence of waypoints exists, this combo cannot achieve
           the goal.
       - Take the minimum cost over all suitable rover/camera combinations. If
         none can achieve the goal, the heuristic for this goal is infinity.
       - Add this minimum cost to the total heuristic.
    9. If the total heuristic cost is infinity (meaning at least one goal is
       unreachable), return infinity. Otherwise, return the total cost.
    """

    def __init__(self, task):
        self.task = task
        self.goals = task.goals
        self.static = task.static
        self.initial_state = task.initial_state # Need initial state for sample locations

        # Precompute static information
        self.lander_location = {} # {lander: waypoint}
        self.equipped_soil = set() # {rover}
        self.equipped_rock = set() # {rover}
        self.equipped_imaging = set() # {rover}
        self.store_owner = {} # {store: rover}
        self.rover_stores = collections.defaultdict(set) # {rover: {store}}
        self.initial_soil_samples = set() # {waypoint}
        self.initial_rock_samples = set() # {waypoint}
        self.can_traverse_graph = collections.defaultdict(lambda: collections.defaultdict(set)) # {rover: {wp1: {wp2, ...}}}
        self.visible_graph = collections.defaultdict(set) # {wp1: {wp2, ...}} (undirected)
        self.camera_modes = collections.defaultdict(set) # {camera: {mode}}
        self.camera_calib_target = {} # {camera: objective}
        self.camera_on_rover = {} # {camera: rover}
        self.rover_cameras = collections.defaultdict(set) # {rover: {camera}}
        self.objective_visible_from = collections.defaultdict(set) # {objective: {waypoint}}

        # Parse static facts
        for fact_str in self.static:
            pred, objs = self._parse_fact(fact_str)
            if pred == 'at_lander':
                self.lander_location[objs[0]] = objs[1]
            elif pred == 'equipped_for_soil_analysis':
                self.equipped_soil.add(objs[0])
            elif pred == 'equipped_for_rock_analysis':
                self.equipped_rock.add(objs[0])
            elif pred == 'equipped_for_imaging':
                self.equipped_imaging.add(objs[0])
            elif pred == 'store_of':
                self.store_owner[objs[0]] = objs[1]
                self.rover_stores[objs[1]].add(objs[0])
            elif pred == 'can_traverse':
                self.can_traverse_graph[objs[0]][objs[1]].add(objs[2])
            elif pred == 'visible':
                self.visible_graph[objs[0]].add(objs[1])
                self.visible_graph[objs[1]].add(objs[0]) # Visible is symmetric
            elif pred == 'supports':
                self.camera_modes[objs[0]].add(objs[1])
            elif pred == 'calibration_target':
                self.camera_calib_target[objs[0]] = objs[1]
            elif pred == 'on_board':
                self.camera_on_rover[objs[0]] = objs[1]
                self.rover_cameras[objs[1]].add(objs[0])
            elif pred == 'visible_from':
                self.objective_visible_from[objs[0]].add(objs[1])

        # Parse initial state for initial sample locations
        for fact_str in self.initial_state:
             pred, objs = self._parse_fact(fact_str)
             if pred == 'at_soil_sample':
                 self.initial_soil_samples.add(objs[0])
             elif pred == 'at_rock_sample':
                 self.initial_rock_samples.add(objs[0])

        # Identify communication points
        self.communication_points = collections.defaultdict(set) # {lander: {waypoint}}
        all_waypoints = set()
        # Collect all waypoints mentioned in can_traverse, visible, objective_visible_from, lander_location
        for rover_graph in self.can_traverse_graph.values():
            all_waypoints.update(rover_graph.keys())
            for dest_wps in rover_graph.values():
                 all_waypoints.update(dest_wps)
        for wp1, connected_wps in self.visible_graph.items():
             all_waypoints.add(wp1)
             all_waypoints.update(connected_wps)
        for obj_wps in self.objective_visible_from.values():
             all_waypoints.update(obj_wps)
        all_waypoints.update(self.lander_location.values())


        for lander, lander_wp in self.lander_location.items():
            # A waypoint is a communication point for a lander if it's visible from the lander's location
            # The 'communicate' action requires `(visible ?x ?y)` where ?x is rover loc, ?y is lander loc.
            # So, communication points for lander at Y are waypoints X such that (visible X Y) is true.
            if lander_wp in self.visible_graph:
                 self.communication_points[lander].update(self.visible_graph[lander_wp])
            # Also check the reverse if visible graph wasn't fully symmetric in static facts
            for wp1, connected_wps in self.visible_graph.items():
                if lander_wp in connected_wps:
                    self.communication_points[lander].add(wp1)


        # Compute shortest paths for each rover
        self.rover_dist = {} # {rover: {start_wp: {end_wp: dist}}}
        all_rovers = set(self.can_traverse_graph.keys())
        # Include rovers from equipped facts even if they have no can_traverse
        all_rovers.update(self.equipped_soil)
        all_rovers.update(self.equipped_rock)
        all_rovers.update(self.equipped_imaging)

        for rover in all_rovers:
            # Build the graph for this specific rover
            rover_specific_graph = self.can_traverse_graph.get(rover, collections.defaultdict(set))
            self.rover_dist[rover] = self._compute_all_pairs_shortest_paths(rover_specific_graph, all_waypoints)

    def _parse_fact(self, fact_str):
        """Helper to parse a fact string into predicate and objects."""
        # Remove parentheses and split by space
        parts = fact_str[1:-1].split()
        predicate = parts[0]
        objects = parts[1:]
        return predicate, objects

    def _compute_all_pairs_shortest_paths(self, graph, all_waypoints):
        """Computes shortest paths from all relevant nodes to all nodes using BFS."""
        distances = {}
        # Consider all waypoints that are part of the graph or known in the domain
        nodes_to_bfs_from = set(graph.keys())
        for dest_nodes in graph.values():
            nodes_to_bfs_from.update(dest_nodes)
        nodes_to_bfs_from.update(all_waypoints) # Ensure all known waypoints are potential start points

        for start_node in nodes_to_bfs_from:
            distances[start_node] = self._bfs(graph, start_node)
        return distances

    def _bfs(self, graph, start_node):
        """Performs BFS from a start node to find distances to all reachable nodes."""
        dist = {start_node: 0}
        queue = collections.deque([start_node])

        # If the start_node is not in the graph keys (i.e., has no outgoing edges),
        # the loop won't run, and dist will correctly contain only {start_node: 0}.
        # If it is in graph keys but has no neighbors, the loop will run once and stop.

        queue = collections.deque([start_node])
        visited = {start_node}
        dist = {start_node: 0}

        while queue:
            current_node = queue.popleft()

            # Get neighbors from the graph. If current_node is not a key, it has no outgoing edges.
            neighbors = graph.get(current_node, set())

            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    dist[neighbor] = dist[current_node] + 1
                    queue.append(neighbor)

        return dist

    def get_dist(self, rover_name, start_wp, end_wp):
        """Gets the precomputed shortest distance between two waypoints for a rover."""
        # If the rover doesn't exist or has no navigation info, it can't move.
        if rover_name not in self.rover_dist:
             return 0 if start_wp == end_wp else None # None represents infinity

        # Check if the start waypoint was a valid starting point for BFS for this rover
        if start_wp not in self.rover_dist[rover_name]:
             # This waypoint might be isolated for this rover, or wasn't included in BFS starts.
             # Our _compute_all_pairs_shortest_paths tries to include all known waypoints.
             # If it's still not found, assume unreachable unless start == end.
             return 0 if start_wp == end_wp else None # None represents infinity

        # Get the distance from the precomputed table. .get() returns None if end_wp is not reachable.
        return self.rover_dist[rover_name][start_wp].get(end_wp)

    def get_min_dist_to_comm(self, rover_name, start_wp):
        """Gets the minimum distance from start_wp to any communication point for a rover."""
        min_dist = float('inf')
        found_comm_point = False
        # Iterate through all communication points for all landers
        for lander_wps in self.communication_points.values():
            for comm_wp in lander_wps:
                dist = self.get_dist(rover_name, start_wp, comm_wp)
                if dist is not None:
                    min_dist = min(min_dist, dist)
                    found_comm_point = True
        return min_dist if found_comm_point else None # Represents infinity

    def __call__(self, state):
        """
        Computes the heuristic value for the given state.

        @param state The current state (frozenset of facts).
        @return The estimated number of actions to reach a goal state, or float('inf').
        """
        # Check if goal is reached
        if self.task.goal_reached(state):
            return 0

        # Parse dynamic state facts
        state_rover_locations = {} # {rover: waypoint}
        state_empty_stores = set() # {store}
        state_full_stores = set() # {store}
        state_have_soil = set() # {(rover, waypoint)}
        state_have_rock = set() # {(rover, waypoint)}
        state_have_image = set() # {(rover, objective, mode)}
        state_calibrated_cameras = set() # {(camera, rover)}
        state_comm_soil = set() # {waypoint}
        state_comm_rock = set() # {waypoint}
        state_comm_image = set() # {(objective, mode)}
        # state_soil_samples_at = set() # {waypoint} # Not needed for heuristic logic based on initial samples
        # state_rock_samples_at = set() # {waypoint} # Not needed for heuristic logic based on initial samples


        for fact_str in state:
            pred, objs = self._parse_fact(fact_str)
            if pred == 'at':
                state_rover_locations[objs[0]] = objs[1]
            elif pred == 'empty':
                state_empty_stores.add(objs[0])
            elif pred == 'full':
                state_full_stores.add(objs[0])
            elif pred == 'have_soil_analysis':
                state_have_soil.add((objs[0], objs[1]))
            elif pred == 'have_rock_analysis':
                state_have_rock.add((objs[0], objs[1]))
            elif pred == 'have_image':
                state_have_image.add((objs[0], objs[1], objs[2]))
            elif pred == 'calibrated':
                state_calibrated_cameras.add((objs[0], objs[1]))
            elif pred == 'communicated_soil_data':
                state_comm_soil.add(objs[0])
            elif pred == 'communicated_rock_data':
                state_comm_rock.add(objs[0])
            elif pred == 'communicated_image_data':
                state_comm_image.add((objs[0], objs[1]))
            # elif pred == 'at_soil_sample': # Not used in heuristic logic
            #      state_soil_samples_at.add(objs[0])
            # elif pred == 'at_rock_sample': # Not used in heuristic logic
            #      state_rock_samples_at.add(objs[0])


        total_cost = 0

        # Process Soil Goals
        soil_goals_needed = set()
        for goal_fact_str in self.goals:
            pred, objs = self._parse_fact(goal_fact_str)
            if pred == 'communicated_soil_data':
                w_name = objs[0]
                if w_name not in state_comm_soil:
                    soil_goals_needed.add(w_name)

        for w_name in soil_goals_needed:
            min_goal_cost = float('inf')
            for r_name in self.equipped_soil:
                if r_name not in state_rover_locations: continue # Rover not in state? Should not happen.
                rover_loc = state_rover_locations[r_name]
                current_rover_cost = float('inf')

                if (r_name, w_name) in state_have_soil:
                    # Need to communicate
                    min_comm_dist = self.get_min_dist_to_comm(r_name, rover_loc)
                    if min_comm_dist is not None:
                        current_rover_cost = min_comm_dist + 1 # +1 for communicate action
                elif w_name in self.initial_soil_samples:
                    # Need to sample and communicate
                    dist_to_sample = self.get_dist(r_name, rover_loc, w_name)
                    if dist_to_sample is not None:
                        sample_cost = dist_to_sample + 1 # +1 for sample action

                        # Check if rover's store is full
                        store_full = False
                        for store in self.rover_stores.get(r_name, []):
                            if store in state_full_stores:
                                store_full = True
                                break
                        if store_full:
                            sample_cost += 1 # +1 for drop action

                        # After sampling, rover is at w_name
                        min_comm_dist = self.get_min_dist_to_comm(r_name, w_name)
                        if min_comm_dist is not None:
                            communicate_cost = min_comm_dist + 1 # +1 for communicate action
                            current_rover_cost = sample_cost + communicate_cost

                min_goal_cost = min(min_goal_cost, current_rover_cost)

            if min_goal_cost == float('inf'):
                return float('inf') # Unreachable goal
            total_cost += min_goal_cost

        # Process Rock Goals
        rock_goals_needed = set()
        for goal_fact_str in self.goals:
            pred, objs = self._parse_fact(goal_fact_str)
            if pred == 'communicated_rock_data':
                w_name = objs[0]
                if w_name not in state_comm_rock:
                    rock_goals_needed.add(w_name)

        for w_name in rock_goals_needed:
            min_goal_cost = float('inf')
            for r_name in self.equipped_rock:
                if r_name not in state_rover_locations: continue
                rover_loc = state_rover_locations[r_name]
                current_rover_cost = float('inf')

                if (r_name, w_name) in state_have_rock:
                    # Need to communicate
                    min_comm_dist = self.get_min_dist_to_comm(r_name, rover_loc)
                    if min_comm_dist is not None:
                        current_rover_cost = min_comm_dist + 1 # +1 for communicate action
                elif w_name in self.initial_rock_samples:
                    # Need to sample and communicate
                    dist_to_sample = self.get_dist(r_name, rover_loc, w_name)
                    if dist_to_sample is not None:
                        sample_cost = dist_to_sample + 1 # +1 for sample action

                        # Check if rover's store is full
                        store_full = False
                        for store in self.rover_stores.get(r_name, []):
                            if store in state_full_stores:
                                store_full = True
                                break
                        if store_full:
                            sample_cost += 1 # +1 for drop action

                        # After sampling, rover is at w_name
                        min_comm_dist = self.get_min_dist_to_comm(r_name, w_name)
                        if min_comm_dist is not None:
                            communicate_cost = min_comm_dist + 1 # +1 for communicate action
                            current_rover_cost = sample_cost + communicate_cost

                min_goal_cost = min(min_goal_cost, current_rover_cost)

            if min_goal_cost == float('inf'):
                return float('inf') # Unreachable goal
            total_cost += min_goal_cost


        # Process Image Goals
        image_goals_needed = set() # {(objective, mode)}
        for goal_fact_str in self.goals:
            pred, objs = self._parse_fact(goal_fact_str)
            if pred == 'communicated_image_data':
                o_name, m_name = objs
                if (o_name, m_name) not in state_comm_image:
                    image_goals_needed.add((o_name, m_name))

        for o_name, m_name in image_goals_needed:
            min_goal_cost = float('inf')
            for r_name in self.equipped_imaging:
                if r_name not in state_rover_locations: continue
                rover_loc = state_rover_locations[r_name]

                for i_name in self.rover_cameras.get(r_name, set()):
                    if m_name not in self.camera_modes.get(i_name, set()): continue # Camera doesn't support mode

                    current_rover_camera_cost = float('inf')

                    if (r_name, o_name, m_name) in state_have_image:
                        # Need to communicate
                        min_comm_dist = self.get_min_dist_to_comm(r_name, rover_loc)
                        if min_comm_dist is not None:
                            current_rover_camera_cost = min_comm_dist + 1 # +1 for communicate action
                    else:
                        # Need to take image and communicate
                        # Find best image waypoint ?p
                        image_wps = self.objective_visible_from.get(o_name, set())
                        if not image_wps: continue # Cannot see objective ?o

                        # Find best calib waypoint ?w
                        calib_target = self.camera_calib_target.get(i_name)
                        if not calib_target: continue # Camera ?i has no calib target
                        calib_wps = self.objective_visible_from.get(calib_target, set()) # calib targets are objectives
                        if not calib_wps: continue # Cannot see calib target

                        # Find best comm waypoint ?c
                        comm_wps = set()
                        for lander_wp_set in self.communication_points.values():
                            comm_wps.update(lander_wp_set)
                        if not comm_wps: continue # No communication points

                        # Minimize cost over choices of ?p, ?w, ?c
                        # Cost = dist(?r.location, ?w) + 1 + dist(?w, ?p) + 1 + dist(?p, ?c) + 1
                        min_path_cost = float('inf')
                        for w in calib_wps:
                            dist_r_w = self.get_dist(r_name, rover_loc, w)
                            if dist_r_w is None: continue
                            for p in image_wps:
                                dist_w_p = self.get_dist(r_name, w, p)
                                if dist_w_p is None: continue
                                for c in comm_wps:
                                    dist_p_c = self.get_dist(r_name, p, c)
                                    if dist_p_c is None: continue
                                    path_cost = dist_r_w + dist_w_p + dist_p_c
                                    min_path_cost = min(min_path_cost, path_cost)

                        if min_path_cost != float('inf'):
                            # Add action costs: 1 (calib) + 1 (take_image) + 1 (communicate)
                            current_rover_camera_cost = min_path_cost + 3

                    min_goal_cost = min(min_goal_cost, current_rover_camera_cost)

            if min_goal_cost == float('inf'):
                return float('inf') # Unreachable goal
            total_cost += min_goal_cost


        return total_cost
