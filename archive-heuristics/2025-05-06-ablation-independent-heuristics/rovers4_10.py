import collections
from heuristics.heuristic_base import Heuristic
from task import Task # Keep this import as Task structure is implicitly used

# Helper function to parse PDDL facts
def parse_fact(fact_string):
    """Parses a PDDL fact string into predicate and objects."""
    # Remove surrounding brackets and split by space
    parts = fact_string[1:-1].split()
    predicate = parts[0]
    objects = parts[1:]
    return predicate, objects

# Helper function for BFS
def bfs(graph, start_node):
    """Performs BFS to find shortest distances from start_node to all reachable nodes."""
    distances = {node: float('inf') for node in graph}
    distances[start_node] = 0
    queue = collections.deque([start_node])
    while queue:
        current_node = queue.popleft()
        current_dist = distances[current_node]
        # Check if current_node has outgoing edges in the graph
        if current_node in graph:
            for neighbor in graph[current_node]:
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = current_dist + 1
                    queue.append(neighbor)
    return distances


class roversHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Rovers domain.

    Summary:
    The heuristic estimates the remaining cost to reach the goal state by summing
    the estimated minimum costs for each unachieved goal predicate. For each
    unachieved goal (communicating soil data, rock data, or image data), it
    calculates the minimum number of actions required to achieve that specific
    goal from the current state, considering available rovers, equipment,
    locations, and precomputed shortest paths. Navigation costs are estimated
    using precomputed shortest path distances (number of steps) between waypoints
    for each rover. Other action costs (sampling, dropping, calibrating, taking
    image, communicating) are counted as 1. The heuristic considers two main
    cases for data goals: either the required data/image is already collected
    by some rover, or it needs to be collected/taken. It takes the minimum
    estimated cost over all applicable rovers and necessary waypoints (sample,
    image, calibration, communication).

    Assumptions:
    - The PDDL domain structure and action costs (all 1) are as provided.
    - The navigation graph for a rover `r` from waypoint `y` to `z` exists
      if and only if `(can_traverse ?r y z)` and `(visible y z)` are true.
      The heuristic uses this graph to compute shortest paths.
    - `visible` is assumed symmetric if facts are provided in both directions.
      `can_traverse` is assumed symmetric if facts are provided in both directions.
      The graph building uses the directed edges defined by the navigate precondition.
    - Each rover has at most one store.
    - The lander location is static and given by `at_lander`.
    - Initial soil/rock samples are static and given by `at_soil_sample`/`at_rock_sample`
      in the initial state. If a goal requires a sample not present initially,
      it's unreachable.
    - Objectives visible from waypoints (`visible_from`) are static.
    - Camera properties (on_board, supports, calibration_target) are static.
    - The heuristic does not account for negative interactions like store
      capacity limits beyond the current state, or calibration being lost
      after taking an image when estimating costs for multiple image goals.
      It sums independent goal costs.
    - Unreachable goals (e.g., sample doesn't exist, objective not visible
      from any waypoint, no communication waypoint) result in an infinite
      heuristic value.

    Heuristic Initialization:
    1.  Parse all static facts from `task.static` and `task.initial_state`.
    2.  Store static information in dedicated data structures: lander location,
        rover equipment, store-rover mapping, camera properties (modes,
        calibration target, on-board status), objective visibility from waypoints,
        and waypoint visibility.
    3.  Store initial locations of soil and rock samples from `task.initial_state`.
    4.  Identify all relevant waypoints from `can_traverse`, `visible`, and
        initial rover locations.
    5.  For each rover, build its specific directed navigation graph. An edge
        exists from waypoint `y` to `z` for rover `r` if `(can_traverse ?r y z)`
        and `(visible y z)` are true. Include all identified waypoints as nodes,
        even if isolated for a specific rover.
    6.  For each rover and each waypoint, run BFS on the rover's navigation
        graph to precompute shortest path distances (number of navigation actions)
        to all other reachable waypoints. Store these distances.

    Step-By-Step Thinking for Computing Heuristic:
    1.  Check if the current state is a goal state. If yes, return 0.
    2.  Identify the lander waypoint and communication waypoints (visible from lander). If no communication waypoints exist and there are communication goals, return infinity.
    3.  Initialize the total heuristic value `h` to 0.
    4.  Iterate through each goal predicate in `self.goals`.
    5.  If the goal predicate is already true in the current state, skip it.
    6.  If the goal is `(communicated_soil_data ?w)`:
        a.  Initialize minimum cost for this goal `min_cost_g` to infinity.
        b.  Check if any soil-equipped rover currently holds `(have_soil_analysis ?r ?w)`. If yes, calculate the minimum cost for any such rover to navigate from its current location to a communication waypoint and communicate (navigate cost + 1). Minimize this over suitable rovers and communication waypoints. Update `min_cost_g`.
        c.  If no rover holds the sample, check if `(at_soil_sample ?w)` was in the initial state. If not, this goal is unreachable; set `min_cost_g` to infinity.
        d.  If the sample exists initially, calculate the minimum cost for any soil-equipped rover to navigate from its current location to `?w`, sample (cost 1, plus 1 if store is full), navigate from `?w` to a communication waypoint, and communicate (cost 1). Minimize this over all soil-equipped rovers and communication waypoints. Update `min_cost_g` with the minimum found.
        e.  The minimum cost for this goal is the minimum of the costs found in steps b and d. Add this minimum cost `min_cost_g` to the total heuristic `h`.
    7.  If the goal is `(communicated_rock_data ?w)`: Follow the same logic as soil data, replacing soil with rock.
    8.  If the goal is `(communicated_image_data ?o ?m)`:
        a.  Initialize minimum cost for this goal `min_cost_g` to infinity.
        b.  Find the minimum cost to communicate if the image is already held by *any* imaging-equipped rover. Check if `(have_image ?r ?o ?m)` is in the state for any rover `r`. If yes, calculate the minimum cost for any such rover to navigate from its current location to a communication waypoint and communicate (navigate cost + 1). Minimize this over suitable rovers and communication waypoints. Update `min_cost_g`.
        c.  If no rover holds the image, calculate the minimum cost to acquire and communicate it. This involves finding an imaging-equipped rover with a suitable camera (on board, supports mode), navigating from its current location to a calibration waypoint for the camera's target, calibrating (cost 1), navigating from the calibration waypoint to an image waypoint for objective `?o`, taking the image (cost 1), navigating from the image waypoint to a communication waypoint, and communicating (cost 1). Minimize this total cost over all suitable rovers, cameras, calibration waypoints, image waypoints, and communication waypoints. If objective `?o` or the camera's calibration target is not visible from any waypoint, or no suitable camera/rover exists, this option's cost is infinity. Update `min_cost_g` with the minimum found.
        d.  The minimum cost for this goal is the minimum of the costs found in steps b and c. Add this minimum cost `min_cost_g` to the total heuristic `h`.
    9.  After processing all unachieved goals, if the total heuristic `h` is infinity (because any single goal was unreachable), return infinity. Otherwise, return `h`.
    """

    def __init__(self, task):
        super().__init__()
        self.goals = task.goals
        self.initial_state = task.initial_state # Store initial state for sample locations

        # --- Precompute Static Information ---
        self.lander_waypoint = None
        self.rover_equipment = collections.defaultdict(set) # rover -> {soil, rock, imaging}
        self.store_rover = {} # store -> rover
        self.camera_modes = collections.defaultdict(set) # camera -> {mode1, mode2}
        self.camera_calibration_target = {} # camera -> objective
        self.rover_cameras = collections.defaultdict(set) # rover -> {camera1, camera2}
        self.objective_visible_from = collections.defaultdict(set) # objective -> {waypoint1, waypoint2}
        self.waypoint_visibility = collections.defaultdict(set) # waypoint -> {visible_waypoint1, visible_waypoint2}
        self.all_waypoints = set() # Collect all waypoints
        rover_can_traverse_facts = collections.defaultdict(set) # Store can_traverse facts per rover

        # Store initial sample locations
        self.initial_soil_samples = set()
        self.initial_rock_samples = set()

        # Parse static facts
        for fact_string in task.static:
            predicate, objects = parse_fact(fact_string)
            if predicate == 'at_lander':
                self.lander_waypoint = objects[1]
            elif predicate == 'equipped_for_soil_analysis':
                self.rover_equipment[objects[0]].add('soil')
            elif predicate == 'equipped_for_rock_analysis':
                self.rover_equipment[objects[0]].add('rock')
            elif predicate == 'equipped_for_imaging':
                self.rover_equipment[objects[0]].add('imaging')
            elif predicate == 'store_of':
                self.store_rover[objects[0]] = objects[1]
            elif predicate == 'supports':
                self.camera_modes[objects[0]].add(objects[1])
            elif predicate == 'calibration_target':
                self.camera_calibration_target[objects[0]] = objects[1]
            elif predicate == 'on_board':
                self.rover_cameras[objects[1]].add(objects[0])
            elif predicate == 'visible_from':
                self.objective_visible_from[objects[0]].add(objects[1])
            elif predicate == 'visible':
                 # Store visible facts (assuming symmetric based on examples)
                 self.waypoint_visibility[objects[0]].add(objects[1])
                 self.waypoint_visibility[objects[1]].add(objects[0])
                 self.all_waypoints.add(objects[0])
                 self.all_waypoints.add(objects[1])
            elif predicate == 'can_traverse':
                 # Store can_traverse facts per rover
                 rover, wp_from, wp_to = objects
                 rover_can_traverse_facts[rover].add((wp_from, wp_to))
                 self.all_waypoints.add(wp_from)
                 self.all_waypoints.add(wp_to)

        # Parse initial state for sample locations and initial rover locations
        initial_rover_locations = {}
        for fact_string in task.initial_state:
             predicate, objects = parse_fact(fact_string)
             if predicate == 'at_soil_sample':
                 self.initial_soil_samples.add(objects[0])
             elif predicate == 'at_rock_sample':
                 self.initial_rock_samples.add(objects[0])
             elif predicate == 'at':
                 rover, wp = objects
                 initial_rover_locations[rover] = wp
                 self.all_waypoints.add(wp)

        # Build traversal graph for each rover using the intersection of can_traverse and visible
        self.rover_traversal_graph = collections.defaultdict(lambda: collections.defaultdict(set))

        # Get all rovers mentioned anywhere (init, equipment, store, cameras, can_traverse)
        all_rovers = set(initial_rover_locations.keys()).union(set(self.rover_equipment.keys())).union(set(self.store_rover.values())).union(set(self.rover_cameras.keys())).union(set(rover_can_traverse_facts.keys()))

        for rover in all_rovers:
             # Ensure all waypoints are nodes in the graph structure for this rover
             self.rover_traversal_graph[rover] = collections.defaultdict(set)
             for wp in self.all_waypoints:
                  self.rover_traversal_graph[rover][wp] = set() # Add all waypoints as isolated nodes initially

             # Add edges based on can_traverse and visible
             if rover in rover_can_traverse_facts:
                  for wp_from, wp_to in rover_can_traverse_facts[rover]:
                       # Check if wp_to is visible from wp_from for the navigate precondition
                       if wp_to in self.waypoint_visibility.get(wp_from, set()):
                            self.rover_traversal_graph[rover][wp_from].add(wp_to)


        # Precompute shortest paths for each rover
        self.rover_shortest_paths = {} # rover -> (start_wp, end_wp) -> distance
        for rover, graph in self.rover_traversal_graph.items():
            self.rover_shortest_paths[rover] = {}
            # BFS from every waypoint that is a node in the graph structure for this rover
            for start_wp in graph.keys(): # Iterate over keys in the graph dict (all_waypoints)
                 distances = bfs(graph, start_wp)
                 for end_wp, dist in distances.items():
                    self.rover_shortest_paths[rover][(start_wp, end_wp)] = dist


    def get_rover_location(self, state, rover):
        """Helper to find the current location of a rover in the state."""
        for fact_string in state:
            predicate, objects = parse_fact(fact_string)
            if predicate == 'at' and objects[0] == rover:
                return objects[1]
        return None # Should not happen in valid states

    def get_rover_store(self, rover):
        """Helper to find the store for a given rover."""
        # Assuming each rover has exactly one store based on domain
        return self.store_rover.get(rover) # Returns None if rover has no store (shouldn't happen)

    def get_distance(self, rover, start_wp, end_wp):
        """Helper to get precomputed distance, handling unreachable."""
        # Check if rover exists and waypoints are valid in precomputed paths
        if rover in self.rover_shortest_paths:
             return self.rover_shortest_paths[rover].get((start_wp, end_wp), float('inf'))
        # If rover not in precomputed paths (e.g., not mentioned in init/static), it cannot traverse
        return float('inf')


    def __call__(self, node):
        """
        Computes the heuristic value for the given state.
        """
        state = node.state

        # Check if goal is reached
        if self.goals <= state:
            return 0

        # Find lander location
        lander_wp = self.lander_waypoint
        if lander_wp is None:
             # Lander location not found in static facts, problem likely unsolvable
             return float('inf')

        # Find communication waypoints (visible from lander)
        comm_wayps = self.waypoint_visibility.get(lander_wp, set())
        # If no communication waypoints exist, any communication goal is impossible
        if not comm_wayps and any(g.startswith('(communicated_') for g in self.goals):
             return float('inf')

        total_h_value = 0

        # Iterate over unachieved goals
        for goal in self.goals:
            if goal in state:
                continue # Goal already achieved

            predicate, objects = parse_fact(goal)
            min_cost_goal = float('inf')

            if predicate == 'communicated_soil_data':
                waypoint_p = objects[0] # Waypoint where sample is needed

                # Option A: Sample already collected by *some* rover
                cost_if_have_sample = float('inf')
                for rover in self.rover_equipment: # Iterate all rovers
                    if 'soil' in self.rover_equipment[rover]:
                        if f'(have_soil_analysis {rover} {waypoint_p})' in state:
                            current_wp = self.get_rover_location(state, rover)
                            if current_wp:
                                # Estimate cost to communicate
                                for comm_wp in comm_wayps:
                                    cost = self.get_distance(rover, current_wp, comm_wp) + 1 # navigate + communicate
                                    cost_if_have_sample = min(cost_if_have_sample, cost)

                # Option B: Need to collect soil sample
                cost_if_need_sample = float('inf')
                if waypoint_p in self.initial_soil_samples: # Check if sample exists initially
                    for rover in self.rover_equipment:
                        if 'soil' in self.rover_equipment[rover]:
                            current_wp = self.get_rover_location(state, rover)
                            store = self.get_rover_store(rover)
                            if current_wp and store:
                                # Cost to navigate to sample location
                                cost_to_sample_wp = self.get_distance(rover, current_wp, waypoint_p)
                                if cost_to_sample_wp == float('inf'): continue # Cannot reach sample location

                                # Cost to sample (includes drop if store is full)
                                cost_sample = 1 # sample_soil action
                                if f'(full {store})' in state:
                                    cost_sample += 1 # drop action

                                # Estimate cost to communicate after sampling
                                for comm_wp in comm_wayps:
                                    cost_to_comm_wp = self.get_distance(rover, waypoint_p, comm_wp)
                                    if cost_to_comm_wp == float('inf'): continue # Cannot reach communication location

                                    total_cost = cost_to_sample_wp + cost_sample + cost_to_comm_wp + 1 # navigate + sample (+drop) + navigate + communicate
                                    cost_if_need_sample = min(cost_if_need_sample, total_cost)

                # The minimum cost for this goal is the minimum of the two options
                min_cost_goal = min(cost_if_have_sample, cost_if_need_sample)

                total_h_value += min_cost_goal


            elif predicate == 'communicated_rock_data':
                waypoint_p = objects[0] # Waypoint where sample is needed

                # Option A: Sample already collected by *some* rover
                cost_if_have_sample = float('inf')
                for rover in self.rover_equipment: # Iterate all rovers
                    if 'rock' in self.rover_equipment[rover]:
                        if f'(have_rock_analysis {rover} {waypoint_p})' in state:
                            current_wp = self.get_rover_location(state, rover)
                            if current_wp:
                                # Estimate cost to communicate
                                for comm_wp in comm_wayps:
                                    cost = self.get_distance(rover, current_wp, comm_wp) + 1 # navigate + communicate
                                    cost_if_have_sample = min(cost_if_have_sample, cost)

                # Option B: Need to collect rock sample
                cost_if_need_sample = float('inf')
                if waypoint_p in self.initial_rock_samples: # Check if sample exists initially
                    for rover in self.rover_equipment:
                        if 'rock' in self.rover_equipment[rover]:
                            current_wp = self.get_rover_location(state, rover)
                            store = self.get_rover_store(rover)
                            if current_wp and store:
                                # Cost to navigate to sample location
                                cost_to_sample_wp = self.get_distance(rover, current_wp, waypoint_p)
                                if cost_to_sample_wp == float('inf'): continue # Cannot reach sample location

                                # Cost to sample (includes drop if store is full)
                                cost_sample = 1 # sample_rock action
                                if f'(full {store})' in state:
                                    cost_sample += 1 # drop action

                                # Estimate cost to communicate after sampling
                                for comm_wp in comm_wayps:
                                    cost_to_comm_wp = self.get_distance(rover, waypoint_p, comm_wp)
                                    if cost_to_comm_wp == float('inf'): continue # Cannot reach communication location

                                    total_cost = cost_to_sample_wp + cost_sample + cost_to_comm_wp + 1 # navigate + sample (+drop) + navigate + communicate
                                    cost_if_need_sample = min(cost_if_need_sample, total_cost)

                # The minimum cost for this goal is the minimum of the two options
                min_cost_goal = min(cost_if_have_sample, cost_if_need_sample)

                total_h_value += min_cost_goal


            elif predicate == 'communicated_image_data':
                objective_o = objects[0]
                mode_m = objects[1]

                # Find minimum cost to communicate if image is already held by *any* rover
                cost_if_have_image = float('inf')
                for rover in self.rover_equipment:
                    if 'imaging' in self.rover_equipment[rover]:
                         # Check if this rover has the image
                         if f'(have_image {rover} {objective_o} {mode_m})' in state:
                             current_wp = self.get_rover_location(state, rover)
                             if current_wp:
                                 for comm_wp in comm_wayps:
                                     cost = self.get_distance(rover, current_wp, comm_wp) + 1 # navigate + communicate
                                     cost_if_have_image = min(cost_if_have_image, cost)

                # If image is not held by any rover, find minimum cost to acquire and communicate
                cost_if_need_image = float('inf')
                image_wayps = self.objective_visible_from.get(objective_o, set())

                if image_wayps: # Only proceed if objective is visible from somewhere
                    for rover in self.rover_equipment:
                        if 'imaging' in self.rover_equipment[rover]:
                            current_wp = self.get_rover_location(state, rover)
                            if not current_wp: continue

                            for camera in self.rover_cameras.get(rover, set()):
                                if mode_m in self.camera_modes.get(camera, set()):
                                    calibration_target = self.camera_calibration_target.get(camera)
                                    if not calibration_target: continue

                                    calibration_wayps = self.objective_visible_from.get(calibration_target, set())

                                    if calibration_wayps: # Only proceed if calibration target is visible from somewhere
                                        # Find best calibration waypoint
                                        min_cost_to_cal = float('inf')
                                        best_cal_wp = None
                                        for cal_wp in calibration_wayps:
                                             cost = self.get_distance(rover, current_wp, cal_wp)
                                             if cost < min_cost_to_cal:
                                                  min_cost_to_cal = cost
                                                  best_cal_wp = cal_wp

                                        if best_cal_wp is None or min_cost_to_cal == float('inf'): continue

                                        cost_calibrate = 1

                                        # Find best image waypoint reachable from best_cal_wp
                                        min_cost_to_img = float('inf')
                                        best_img_wp = None
                                        for img_wp in image_wayps:
                                             cost = self.get_distance(rover, best_cal_wp, img_wp)
                                             if cost < min_cost_to_img:
                                                  min_cost_to_img = cost
                                                  best_img_wp = img_wp

                                        if best_img_wp is None or min_cost_to_img == float('inf'): continue

                                        cost_take_image = 1

                                        # Find best communication waypoint reachable from best_img_wp
                                        min_cost_to_comm = float('inf')
                                        for comm_wp in comm_wayps:
                                             cost = self.get_distance(rover, best_img_wp, comm_wp)
                                             min_cost_to_comm = min(min_cost_to_comm, cost)

                                        if min_cost_to_comm == float('inf'): continue

                                        total_cost = min_cost_to_cal + cost_calibrate + min_cost_to_img + cost_take_image + min_cost_to_comm + 1 # navigate + calibrate + navigate + take_image + navigate + communicate
                                        cost_if_need_image = min(cost_if_need_image, total_cost)


                # The minimum cost for this goal is the minimum of the two options
                min_cost_goal = min(cost_if_have_image, cost_if_need_image)

                total_h_value += min_cost_goal

            # If any goal is unreachable, the total heuristic is infinity
            if min_cost_goal == float('inf'):
                 return float('inf')

        return total_h_value
