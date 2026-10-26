import collections
import heapq
import logging

from heuristics.heuristic_base import Heuristic
from task import Operator, Task


# Helper function for parsing PDDL facts
def parse_fact(fact_string):
    """
    Parses a PDDL fact string into a predicate name and a list of arguments.
    e.g., '(at rover1 waypoint1)' -> ('at', ['rover1', 'waypoint1'])
    """
    # Removes outer parentheses and splits by space
    parts = fact_string.strip()[1:-1].split()
    if not parts: # Handle empty fact string if somehow encountered
        return None, []
    return parts[0], parts[1:] # predicate, arguments

# Helper function for Breadth-First Search (BFS)
def bfs(graph, start_node):
    """
    Performs BFS on a graph to find shortest path distances from a start node.
    Graph is represented as dict(node -> set(neighbor)).
    Returns a dict(node -> distance).
    """
    # Collect all nodes in the graph
    all_nodes = set(graph.keys())
    for neighbors in graph.values():
        all_nodes.update(neighbors)

    distances = {node: float('inf') for node in all_nodes}

    if start_node in distances: # Ensure start_node is a valid node in the graph
        distances[start_node] = 0
        queue = collections.deque([start_node])

        while queue:
            current_node = queue.popleft()

            # Check if current_node has neighbors in the graph (it should, based on how all_nodes is built)
            if current_node in graph:
                for neighbor in graph[current_node]:
                    if distances[neighbor] == float('inf'):
                        distances[neighbor] = distances[current_node] + 1
                        queue.append(neighbor)
    return distances

# Helper function to compute all-pairs shortest paths using BFS
def compute_all_pairs_shortest_paths(graph):
    """
    Computes shortest path distances between all pairs of nodes in the graph
    using BFS from each node.
    Returns a dict(start_node -> dict(end_node -> distance)).
    """
    all_paths = {}
    # Collect all nodes in the graph
    all_nodes = set(graph.keys())
    for neighbors in graph.values():
        all_nodes.update(neighbors)

    for start_node in all_nodes:
        all_paths[start_node] = bfs(graph, start_node)
    return all_paths


class roversHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Rovers domain.

    Summary:
        This heuristic estimates the cost to reach a goal state by summing the
        estimated minimum costs for each currently unachieved goal fact.
        The cost for each goal fact (communicated_soil_data, communicated_rock_data,
        communicated_image_data) is estimated based on the required sequence of
        actions: navigating to relevant locations (sample site, imaging site,
        calibration site, communication site), performing the necessary data
        collection actions (sample, calibrate, take_image), and finally
        communicating the data. Navigation costs are estimated using precomputed
        shortest path distances on the waypoint graph for each rover. The heuristic
        considers rover capabilities, camera properties, store status, and the
        availability of samples. It finds the minimum cost among all rovers
        (and cameras for imaging goals) capable of achieving the goal.

    Assumptions:
        - This heuristic is domain-dependent for the PDDL 'rovers' domain.
        - It is non-admissible, as it sums costs for individual goals which may
          share required sub-tasks (e.g., navigation to a common waypoint).
        - It is designed for use with greedy best-first search, prioritizing
          states that appear closer to the goal based on this cost estimate.
        - Soil and rock samples are consumed upon sampling.
        - Camera calibration is consumed upon taking an image.
        - The PDDL structure and fact names conform to the provided domain file.

    Heuristic Initialization:
        The constructor performs precomputation to efficiently calculate heuristic
        values during the search. This involves:
        1.  Parsing static facts (like `at_lander`, `equipped_for_soil_analysis`,
            `can_traverse`, `visible`, `calibration_target`, `on_board`, `supports`,
            `visible_from`) and initial state facts (like `at_soil_sample`,
            `at_rock_sample`, `at`) to build internal data structures.
        2.  Identifying all waypoints and rovers present in the problem.
        3.  Building a navigation graph for each rover. An edge exists from waypoint A
            to waypoint B for rover R if `(can_traverse R A B)` and `(visible A B)`
            are true in the static facts.
        4.  Computing all-pairs shortest path distances for each rover's navigation
            graph using Breadth-First Search (BFS), as navigation actions have unit cost.
        5.  Identifying communication waypoints for each rover: waypoints reachable
            by the rover that are visible from the lander's location.
        6.  Identifying waypoints visible from each objective (for imaging).
        7.  Identifying waypoints visible from the calibration target for each camera
            (for calibration).
        These precomputed structures allow for quick lookups during the heuristic
        calculation for any given state.

    Step-By-Step Thinking for Computing Heuristic:
        The `__call__` method computes the heuristic value for a given state node:
        1.  Identify the set of goal facts that are not yet true in the current state.
        2.  If the set of unachieved goals is empty, the state is a goal state, and the
            heuristic value is 0.
        3.  Initialize the total heuristic value `h` to 0.
        4.  Parse the current state facts to get the current location of each rover,
            the status of each store (empty/full), the locations of remaining soil/rock
            samples, the data collected (`have_soil_analysis`, `have_rock_analysis`,
            `have_image`), and the calibration status of cameras.
        5.  Iterate through each unachieved goal fact:
            a.  For a `(communicated_soil_data ?w)` goal:
                -   Find the minimum cost among all soil-equipped rovers.
                -   The cost for a rover R is estimated as:
                    -   Cost to get `(have_soil_analysis R ?w)` + Cost to communicate it.
                    -   Cost to get `(have_soil_analysis R ?w)`: 0 if already true. If not, and if a sample was initially at `?w` and is still there: shortest path cost from R's current location to `?w` + (1 if R's store is full, for `drop`) + 1 (for `sample_soil`). If the sample was initially present but is now gone, or never existed, this cost is infinity.
                    -   Cost to communicate: shortest path cost from R's current location to the nearest communication waypoint + 1 (for `communicate_soil_data`).
                -   The minimum cost over all suitable rovers is added to `h`.
            b.  For a `(communicated_rock_data ?w)` goal:
                -   Similar logic to soil data, using rock-equipped rovers, rock samples, `have_rock_analysis`, `sample_rock`, and `communicate_rock_data`.
            c.  For a `(communicated_image_data ?o ?m)` goal:
                -   Find the minimum cost among all imaging-equipped rovers R with a camera I supporting mode `?m`.
                -   The cost for a rover R and camera I is estimated as:
                    -   Cost to get `(have_image R ?o ?m)` + Cost to communicate it.
                    -   Cost to get `(have_image R ?o ?m)`: 0 if already true. If not: shortest path cost from R's current location to the nearest waypoint visible from `?o` + Cost to calibrate camera I + 1 (for `take_image`).
                    -   Cost to calibrate camera I: 0 if already calibrated. If not: shortest path cost from R's current location to the nearest waypoint visible from I's calibration target + 1 (for `calibrate`). If the camera has no calibration target defined, this cost is infinity.
                    -   If any navigation or calibration step is impossible (e.g., no visible waypoint reachable), the cost is infinity.
                    -   If the image was already taken, the cost to get the image is 0.
                    -   Cost to communicate: shortest path cost from R's current location to the nearest communication waypoint + 1 (for `communicate_image_data`).
                -   The minimum cost over all suitable rover/camera pairs is added to `h`.
        6.  If the minimum cost for any unachieved goal is infinity (meaning it's unreachable), the total heuristic `h` becomes infinity.
        7.  Return the total heuristic value `h`.
    """

    def __init__(self, task):
        super().__init__()
        self.goals = task.goals
        self.static_facts = task.static
        self.initial_state = task.initial_state # Need initial state for sample locations

        # --- Precomputation ---
        self._precompute_static_info()
        self._precompute_navigation_costs()
        self._precompute_communication_wps()
        self._precompute_objective_visible_wps() # Populated in _precompute_static_info
        self._precompute_calibration_visible_wps() # Populated based on _precompute_static_info

    def _precompute_static_info(self):
        self.lander_location = None
        self.rover_capabilities = collections.defaultdict(set)
        self.rover_stores = {} # rover -> store
        self.rover_cameras = collections.defaultdict(list) # rover -> list of cameras
        self.camera_modes = collections.defaultdict(set) # camera -> set of modes
        self.camera_calibration_target = {} # camera -> objective
        self.objective_visible_from = collections.defaultdict(set) # objective -> set of waypoints
        self.calibration_target_visible_from = collections.defaultdict(set) # objective -> set of waypoints
        self.initial_soil_sample_locations = set() # waypoints
        self.initial_rock_sample_locations = set() # waypoints

        all_waypoints = set()
        all_rovers = set()

        # Parse initial state for initial sample locations and rover positions (to get all waypoints/rovers)
        for fact_string in self.initial_state:
             pred, args = parse_fact(fact_string)
             if pred == 'at_soil_sample':
                 self.initial_soil_sample_locations.add(args[0])
                 all_waypoints.add(args[0])
             elif pred == 'at_rock_sample':
                 self.initial_rock_sample_locations.add(args[0])
                 all_waypoints.add(args[0])
             elif pred == 'at':
                 all_rovers.add(args[0])
                 all_waypoints.add(args[1])
             elif pred == 'at_lander':
                 all_waypoints.add(args[1])

        # Parse static facts to populate data structures and collect all waypoints/rovers
        visible_pairs = set()
        can_traverse_facts = collections.defaultdict(set) # rover -> set((from_wp, to_wp))

        for fact_string in self.static_facts:
            pred, args = parse_fact(fact_string)
            if pred == 'at_lander':
                self.lander_location = args[1]
                all_waypoints.add(args[1])
            elif pred == 'equipped_for_soil_analysis':
                self.rover_capabilities[args[0]].add('soil')
                all_rovers.add(args[0])
            elif pred == 'equipped_for_rock_analysis':
                self.rover_capabilities[args[0]].add('rock')
                all_rovers.add(args[0])
            elif pred == 'equipped_for_imaging':
                self.rover_capabilities[args[0]].add('imaging')
                all_rovers.add(args[0])
            elif pred == 'store_of':
                self.rover_stores[args[1]] = args[0] # rover -> store
            elif pred == 'can_traverse':
                 rover, wp_from, wp_to = args
                 can_traverse_facts[rover].add((wp_from, wp_to))
                 all_rovers.add(rover)
                 all_waypoints.add(wp_from)
                 all_waypoints.add(wp_to)
            elif pred == 'visible':
                 wp1, wp2 = args
                 visible_pairs.add((wp1, wp2))
                 all_waypoints.add(wp1)
                 all_waypoints.add(wp2)
            elif pred == 'calibration_target':
                 self.camera_calibration_target[args[0]] = args[1] # camera -> objective
            elif pred == 'on_board':
                 self.rover_cameras[args[1]].append(args[0]) # rover -> camera list
                 all_rovers.add(args[1])
            elif pred == 'supports':
                 self.camera_modes[args[0]].add(args[1]) # camera -> mode set
            elif pred == 'visible_from':
                 objective, waypoint = args
                 self.objective_visible_from[objective].add(waypoint)
                 all_waypoints.add(waypoint)

        # Build the navigation graph for each rover
        self.rover_waypoint_graph = collections.defaultdict(lambda: collections.defaultdict(set))
        for rover in all_rovers:
             # Initialize graph with all waypoints, even if isolated
             for wp in all_waypoints:
                  self.rover_waypoint_graph[rover][wp] = set()

             # Add edges where both can_traverse and visible hold
             for wp_from, wp_to in can_traverse_facts.get(rover, set()):
                  if (wp_from, wp_to) in visible_pairs:
                       self.rover_waypoint_graph[rover][wp_from].add(wp_to)

    def _precompute_navigation_costs(self):
        self.shortest_path_cost = {}
        for rover, graph in self.rover_waypoint_graph.items():
            self.shortest_path_cost[rover] = compute_all_pairs_shortest_paths(graph)

    def _precompute_communication_wps(self):
        self.communication_wps = collections.defaultdict(set)
        if self.lander_location:
            visible_from_lander = set()
            for fact_string in self.static_facts:
                pred, args = parse_fact(fact_string)
                if pred == 'visible':
                    wp1, wp2 = args
                    if wp1 == self.lander_location:
                        visible_from_lander.add(wp2)
                    elif wp2 == self.lander_location:
                        visible_from_lander.add(wp1)

            # A rover can communicate from any waypoint visible from the lander
            # provided that waypoint exists in the rover's navigation graph nodes.
            all_waypoints_in_graphs = set()
            for rover, graph in self.rover_waypoint_graph.items():
                 all_waypoints_in_graphs.update(graph.keys())

            for rover in self.rover_waypoint_graph:
                 for comm_wp in visible_from_lander:
                      if comm_wp in all_waypoints_in_graphs: # Check if this waypoint is part of the rover's possible locations
                           self.communication_wps[rover].add(comm_wp)

    def _precompute_objective_visible_wps(self):
         # objective_visible_from is already populated in _precompute_static_info
         pass # This step is just a placeholder for clarity

    def _precompute_calibration_visible_wps(self):
         # Need to map calibration target objective to waypoints visible from it
         # calibration_target_visible_from[objective] = set(waypoints)
         # This requires knowing which objective is a calibration target for which camera
         # and the visible_from facts for those objectives.
         cal_objectives = set(self.camera_calibration_target.values())
         for obj in cal_objectives:
              if obj in self.objective_visible_from:
                   self.calibration_target_visible_from[obj] = self.objective_visible_from[obj]
              else:
                   self.calibration_target_visible_from[obj] = set() # Should not happen if domain is well-formed


    def __call__(self, node):
        state = node.state
        h = 0
        unachieved_goals = self.goals - state

        if not unachieved_goals:
            return 0 # Goal reached

        # --- Parse current state ---
        current_rover_location = {}
        current_store_status = {} # store_name -> 'empty' or 'full'
        current_soil_samples = set() # waypoints with samples
        current_rock_samples = set() # waypoints with samples
        current_have_soil_analysis = set() # (rover, waypoint)
        current_have_rock_analysis = set() # (rover, waypoint)
        current_have_image = set() # (rover, objective, mode)
        current_calibrated_cameras = set() # (camera, rover)

        for fact_string in state:
            pred, args = parse_fact(fact_string)
            if pred == 'at':
                current_rover_location[args[0]] = args[1]
            elif pred == 'empty':
                current_store_status[args[0]] = 'empty'
            elif pred == 'full':
                current_store_status[args[0]] = 'full'
            elif pred == 'at_soil_sample':
                current_soil_samples.add(args[0])
            elif pred == 'at_rock_sample':
                current_rock_samples.add(args[0])
            elif pred == 'have_soil_analysis':
                current_have_soil_analysis.add((args[0], args[1]))
            elif pred == 'have_rock_analysis':
                current_have_rock_analysis.add((args[0], args[1]))
            elif pred == 'have_image':
                current_have_image.add((args[0], args[1], args[2]))
            elif pred == 'calibrated':
                current_calibrated_cameras.add((args[0], args[1]))

        # --- Calculate heuristic for each unachieved goal ---
        for goal_string in unachieved_goals:
            pred, args = parse_fact(goal_string)
            goal_cost = float('inf') # Minimum cost for this specific goal

            if pred == 'communicated_soil_data':
                waypoint_to_communicate = args[0]
                # Find the minimum cost over all suitable rovers
                for rover in self.rover_capabilities:
                    if 'soil' in self.rover_capabilities[rover]:
                        rover_cost = float('inf')

                        # Cost to get (have_soil_analysis rover waypoint_to_communicate)
                        have_soil_fact = (rover, waypoint_to_communicate)
                        cost_get_sample = float('inf')
                        if have_soil_fact in current_have_soil_analysis:
                            cost_get_sample = 0 # Already have the sample data

                        if cost_get_sample == float('inf'): # Need to sample
                            if waypoint_to_communicate in self.initial_soil_sample_locations: # Check if sample was initially present
                                if waypoint_to_communicate in current_soil_samples: # Check if sample is still there
                                    # Need to navigate to waypoint_to_communicate
                                    current_wp = current_rover_location.get(rover)
                                    if current_wp is not None and rover in self.shortest_path_cost and current_wp in self.shortest_path_cost[rover]:
                                        nav_cost = self.shortest_path_cost[rover][current_wp].get(waypoint_to_communicate, float('inf'))
                                        if nav_cost < float('inf'):
                                            # Need empty store
                                            store = self.rover_stores.get(rover)
                                            store_cost = 0
                                            if store and current_store_status.get(store) == 'full':
                                                store_cost = 1 # Cost of drop action
                                            # Total cost to sample = nav_cost + store_cost + 1 (sample action)
                                            cost_get_sample = nav_cost + store_cost + 1
                                else:
                                     # Sample was there initially but is now gone (consumed)
                                     cost_get_sample = float('inf') # Cannot sample again
                            else:
                                # Sample was never at this waypoint initially
                                cost_get_sample = float('inf') # Cannot sample

                        if cost_get_sample < float('inf'):
                            # Cost to communicate
                            cost_communicate = float('inf')
                            current_wp = current_rover_location.get(rover)
                            if current_wp is not None and rover in self.shortest_path_cost and current_wp in self.shortest_path_cost[rover]:
                                # Find cheapest path to a communication waypoint
                                min_comm_nav_cost = float('inf')
                                if self.communication_wps.get(rover):
                                     for comm_wp in self.communication_wps[rover]:
                                          nav_cost = self.shortest_path_cost[rover][current_wp].get(comm_wp, float('inf'))
                                          min_comm_nav_cost = min(min_comm_nav_cost, nav_cost)

                                if min_comm_nav_cost < float('inf'):
                                     cost_communicate = min_comm_nav_cost + 1 # communicate action

                            if cost_communicate < float('inf'):
                                rover_cost = cost_get_sample + cost_communicate

                        goal_cost = min(goal_cost, rover_cost)

            elif pred == 'communicated_rock_data':
                waypoint_to_communicate = args[0]
                # Find the minimum cost over all suitable rovers
                for rover in self.rover_capabilities:
                    if 'rock' in self.rover_capabilities[rover]:
                        rover_cost = float('inf')

                        # Cost to get (have_rock_analysis rover waypoint_to_communicate)
                        have_rock_fact = (rover, waypoint_to_communicate)
                        cost_get_sample = float('inf')
                        if have_rock_fact in current_have_rock_analysis:
                            cost_get_sample = 0 # Already have the sample data

                        if cost_get_sample == float('inf'): # Need to sample
                            if waypoint_to_communicate in self.initial_rock_sample_locations: # Check if sample was initially present
                                if waypoint_to_communicate in current_rock_samples: # Check if sample is still there
                                    # Need to navigate to waypoint_to_communicate
                                    current_wp = current_rover_location.get(rover)
                                    if current_wp is not None and rover in self.shortest_path_cost and current_wp in self.shortest_path_cost[rover]:
                                        nav_cost = self.shortest_path_cost[rover][current_wp].get(waypoint_to_communicate, float('inf'))
                                        if nav_cost < float('inf'):
                                            # Need empty store
                                            store = self.rover_stores.get(rover)
                                            store_cost = 0
                                            if store and current_store_status.get(store) == 'full':
                                                store_cost = 1 # Cost of drop action
                                            # Total cost to sample = nav_cost + store_cost + 1 (sample action)
                                            cost_get_sample = nav_cost + store_cost + 1
                                else:
                                     # Sample was there initially but is now gone (consumed)
                                     cost_get_sample = float('inf') # Cannot sample again
                            else:
                                # Sample was never at this waypoint initially
                                cost_get_sample = float('inf') # Cannot sample


                        if cost_get_sample < float('inf'):
                            # Cost to communicate
                            cost_communicate = float('inf')
                            current_wp = current_rover_location.get(rover)
                            if current_wp is not None and rover in self.shortest_path_cost and current_wp in self.shortest_path_cost[rover]:
                                # Find cheapest path to a communication waypoint
                                min_comm_nav_cost = float('inf')
                                if self.communication_wps.get(rover):
                                     for comm_wp in self.communication_wps[rover]:
                                          nav_cost = self.shortest_path_cost[rover][current_wp].get(comm_wp, float('inf'))
                                          min_comm_nav_cost = min(min_comm_nav_cost, nav_cost)

                                if min_comm_nav_cost < float('inf'):
                                     cost_communicate = min_comm_nav_cost + 1 # communicate action

                            if cost_communicate < float('inf'):
                                rover_cost = cost_get_sample + cost_communicate

                        goal_cost = min(goal_cost, rover_cost)

            elif pred == 'communicated_image_data':
                objective_to_communicate = args[0]
                mode_to_communicate = args[1]
                # Find the minimum cost over all suitable rover/camera pairs
                for rover in self.rover_capabilities:
                    if 'imaging' in self.rover_capabilities[rover]:
                        for camera in self.rover_cameras.get(rover, []):
                            if mode_to_communicate in self.camera_modes.get(camera, set()):
                                # Found a suitable rover and camera
                                pair_cost = float('inf')

                                # Cost to get (have_image rover objective_to_communicate mode_to_communicate)
                                have_image_fact = (rover, objective_to_communicate, mode_to_communicate)
                                cost_get_image = float('inf')
                                if have_image_fact in current_have_image:
                                    cost_get_image = 0 # Already have the image data

                                if cost_get_image == float('inf'): # Need to take image
                                    # Need to navigate to a waypoint visible from the objective
                                    current_wp = current_rover_location.get(rover)
                                    if current_wp is not None and rover in self.shortest_path_cost and current_wp in self.shortest_path_cost[rover]:
                                        min_obj_nav_cost = float('inf')
                                        if self.objective_visible_from.get(objective_to_communicate):
                                             for obj_wp in self.objective_visible_from[objective_to_communicate]:
                                                  nav_cost = self.shortest_path_cost[rover][current_wp].get(obj_wp, float('inf'))
                                                  min_obj_nav_cost = min(min_obj_nav_cost, nav_cost)

                                        if min_obj_nav_cost < float('inf'):
                                            # Need camera calibrated
                                            calibrated_fact = (camera, rover)
                                            cost_calibrate = float('inf')
                                            if calibrated_fact in current_calibrated_cameras:
                                                cost_calibrate = 0 # Already calibrated
                                            else:
                                                # Need to calibrate
                                                cal_target_obj = self.camera_calibration_target.get(camera)
                                                if cal_target_obj:
                                                    # Need to navigate to a waypoint visible from the calibration target
                                                    min_cal_nav_cost = float('inf')
                                                    if self.calibration_target_visible_from.get(cal_target_obj):
                                                         for cal_wp in self.calibration_target_visible_from[cal_target_obj]:
                                                              nav_cost = self.shortest_path_cost[rover][current_wp].get(cal_wp, float('inf'))
                                                              min_cal_nav_cost = min(min_cal_nav_cost, nav_cost)

                                                    if min_cal_nav_cost < float('inf'):
                                                        cost_calibrate = min_cal_nav_cost + 1 # calibrate action
                                                else:
                                                     # Camera has no calibration target defined
                                                     cost_calibrate = float('inf') # Cannot calibrate

                                            if cost_calibrate < float('inf'):
                                                # Total cost to take image = min_obj_nav_cost + cost_calibrate + 1 (take_image action)
                                                cost_get_image = min_obj_nav_cost + cost_calibrate + 1


                                if cost_get_image < float('inf'):
                                    # Cost to communicate
                                    cost_communicate = float('inf')
                                    current_wp = current_rover_location.get(rover)
                                    if current_wp is not None and rover in self.shortest_path_cost and current_wp in self.shortest_path_cost[rover]:
                                        # Find cheapest path to a communication waypoint
                                        min_comm_nav_cost = float('inf')
                                        if self.communication_wps.get(rover):
                                             for comm_wp in self.communication_wps[rover]:
                                                  nav_cost = self.shortest_path_cost[rover][current_wp].get(comm_wp, float('inf'))
                                                  min_comm_nav_cost = min(min_comm_nav_cost, nav_cost)

                                        if min_comm_nav_cost < float('inf'):
                                             cost_communicate = min_comm_nav_cost + 1 # communicate action

                                    if cost_communicate < float('inf'):
                                        pair_cost = cost_get_image + cost_communicate

                                goal_cost = min(goal_cost, pair_cost)


            # Add the minimum cost for this goal to the total heuristic
            if goal_cost == float('inf'):
                 # If a goal is unreachable, the heuristic should reflect this.
                 # For greedy best-first, infinity is fine.
                 h = float('inf')
                 break # No need to check other goals if one is unreachable
            else:
                 h += goal_cost

        return h
