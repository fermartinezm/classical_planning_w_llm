from fnmatch import fnmatch
from collections import deque, defaultdict
from heuristics.heuristic_base import Heuristic

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    if not isinstance(fact, str) or len(fact) < 2:
        return []
    # Remove outer parentheses and split by whitespace
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
    Perform BFS to find shortest distances from start_node to all reachable nodes.
    Returns a dictionary mapping node to distance.
    """
    distances = {node: float('inf') for node in all_nodes}
    if start_node not in all_nodes:
         return distances # Start node not in the known waypoint set

    distances[start_node] = 0
    queue = deque([start_node])

    while queue:
        current_node = queue.popleft()
        # Ensure current_node is a valid key in the graph
        if current_node in graph:
            for neighbor in graph[current_node]:
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)
    return distances

# The main heuristic class
class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the required number of actions to reach a goal state
    by summing the estimated costs for each unachieved goal predicate.
    For each goal (communicating soil, rock, or image data), it estimates the
    minimum cost to acquire the necessary data (sampling or imaging, including
    calibration) and then move to a communication location and communicate.
    It considers rover capabilities and precomputes navigation costs using BFS.

    # Assumptions
    - The cost of each action (navigate, sample, drop, calibrate, take_image, communicate) is 1.
    - Navigation cost between waypoints is the shortest path distance in the
      rover's traversal graph. Distances are precomputed using BFS.
    - For sampling goals, the heuristic assumes the rover samples at the sample
      location and communicates from there. It accounts for dropping a full store.
    - For imaging goals, the heuristic assumes the rover follows a sequence:
      move to calibration waypoint, calibrate, move to image waypoint, take image,
      move to communication waypoint, communicate. It finds the minimum cost path
      over all valid intermediate waypoints and suitable rovers/cameras.
    - The heuristic does not account for resource contention (e.g., multiple
      rovers needing the same sample or store) or complex interactions like
      dropping collected data to sample a new location.
    - If a goal requires a resource (like a soil sample or visible_from waypoint)
      that is not available in the current state, that path to the goal is
      considered infinitely expensive for the current state.

    # Heuristic Initialization
    - Parses static facts to build data structures:
        - Rover capabilities (equipped_for_soil, rock, imaging).
        - Store ownership.
        - Camera info (on_board, supports, calibration_target).
        - Objective visibility from waypoints.
        - Lander location and communication waypoints (visible from lander).
        - Rover traversal graphs.
    - Collects all possible waypoints in the problem.
    - Precomputes all-pairs shortest path distances for each rover using BFS
      on their respective traversal graphs.
    - Stores the set of goal predicates.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize total heuristic cost `h = 0`.
    2. Extract current state information: rover locations, store status (empty/full),
       data held by rovers (have_soil, have_rock, have_image), camera calibration status,
       and locations of remaining soil/rock samples.
    3. Iterate through each goal predicate defined in the problem.
    4. For an unachieved goal predicate `G`:
       a. Initialize the minimum cost for this goal `goal_h = infinity`.
       b. If `G` is `(communicated_soil_data W)`:
          - Iterate through all rovers `R` equipped for soil analysis.
          - If `R` already has `(have_soil_analysis R W)`:
            - Calculate cost to communicate: `min_{W_comm} dist(current_R_loc, W_comm) + 1`. Update `goal_h` with the minimum.
          - Else if `(at_soil_sample W)` is true in the current state:
            - Calculate cost to sample: `dist(current_R_loc, W) + (1 if R's store is full else 0) + 1`.
            - Calculate cost to communicate from `W`: `min_{W_comm} dist(W, W_comm) + 1`.
            - Total cost for this rover: cost to sample + cost to communicate. Update `goal_h` with the minimum over rovers.
       c. If `G` is `(communicated_rock_data W)`: Similar logic as soil data, using rock-specific predicates and equipment.
       d. If `G` is `(communicated_image_data O M)`:
          - Iterate through all rovers `R` equipped for imaging.
          - Iterate through all cameras `C` on `R` that support mode `M`.
          - If `R` already has `(have_image R O M)`:
            - Calculate cost to communicate: `min_{W_comm} dist(current_R_loc, W_comm) + 1`. Update `goal_h` with the minimum.
          - Else:
            - Find calibration target `T` for `C`.
            - Find sets of waypoints `W_calib_set` visible from `T` and `W_image_set` visible from `O`.
            - If both sets are non-empty and communication waypoints exist:
              - Calculate the minimum path cost for the sequence: `current_R_loc -> w_calib -> w_image -> w_comm`.
              - Min path cost = `min_{w_calib in W_calib_set, w_image in W_image_set, w_comm in comm_waypoints} (dist(current_R_loc, w_calib) + dist(w_calib, w_image) + dist(w_image, w_comm))`.
              - Total cost for this rover/camera: min path cost + 3 (for calibrate, take_image, communicate actions). Update `goal_h` with the minimum over rovers and cameras.
       e. If `goal_h` is still infinity (no viable path found for this goal), it contributes 0 to the total heuristic (non-admissible). Otherwise, add `goal_h` to `total_h`.
    5. Return `total_h`.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals

        # Data structures for static information
        self.lander_waypoint = None
        self.rover_equipment = defaultdict(set) # rover -> {soil, rock, imaging}
        self.store_rover_map = {} # store -> rover
        self.rover_store_map = {} # rover -> store
        self.camera_rover_map = {} # camera -> rover
        self.rover_camera_map = defaultdict(set) # rover -> {camera}
        self.camera_modes = defaultdict(set) # camera -> {mode}
        self.camera_calibration_target = {} # camera -> objective
        self.objective_visible_from = defaultdict(set) # objective -> {waypoint}
        self.rover_traversal_graph = defaultdict(lambda: defaultdict(set)) # rover -> waypoint -> {neighbor_waypoint}

        # Collect all waypoints from initial state and static facts
        all_waypoints = set()
        for fact in task.initial_state | task.static:
            parts = get_parts(fact)
            if not parts: continue
            pred = parts[0]
            if pred == 'at' and len(parts) > 2: all_waypoints.add(parts[2])
            elif pred == 'at_lander' and len(parts) > 2: all_waypoints.add(parts[2])
            elif pred == 'can_traverse' and len(parts) > 3:
                all_waypoints.add(parts[2])
                all_waypoints.add(parts[3])
            elif pred == 'visible' and len(parts) > 2:
                all_waypoints.add(parts[1])
                all_waypoints.add(parts[2])
            elif pred == 'at_soil_sample' and len(parts) > 1: all_waypoints.add(parts[1])
            elif pred == 'at_rock_sample' and len(parts) > 1: all_waypoints.add(parts[1])
            elif pred == 'visible_from' and len(parts) > 2: all_waypoints.add(parts[2])

        self.all_waypoints = list(all_waypoints) # Use a list for consistent iteration if needed

        # Parse static facts
        visible_graph = defaultdict(set) # Graph for visible predicate
        for fact in task.static:
            parts = get_parts(fact)
            if not parts: continue
            pred = parts[0]
            if pred == 'at_lander' and len(parts) > 2:
                self.lander_waypoint = parts[2]
            elif pred == 'can_traverse' and len(parts) > 3:
                rover, wp_from, wp_to = parts[1], parts[2], parts[3]
                self.rover_traversal_graph[rover][wp_from].add(wp_to)
            elif pred == 'equipped_for_soil_analysis' and len(parts) > 1:
                self.rover_equipment[parts[1]].add('soil')
            elif pred == 'equipped_for_rock_analysis' and len(parts) > 1:
                self.rover_equipment[parts[1]].add('rock')
            elif pred == 'equipped_for_imaging' and len(parts) > 1:
                self.rover_equipment[parts[1]].add('imaging')
            elif pred == 'store_of' and len(parts) > 2:
                store, rover = parts[1], parts[2]
                self.store_rover_map[store] = rover
                self.rover_store_map[rover] = store
            elif pred == 'supports' and len(parts) > 2:
                camera, mode = parts[1], parts[2]
                self.camera_modes[camera].add(mode)
            elif pred == 'visible_from' and len(parts) > 2:
                objective, waypoint = parts[1], parts[2]
                self.objective_visible_from[objective].add(waypoint)
            elif pred == 'calibration_target' and len(parts) > 2:
                camera, objective = parts[1], parts[2]
                self.camera_calibration_target[camera] = objective
            elif pred == 'on_board' and len(parts) > 2:
                camera, rover = parts[1], parts[2]
                self.camera_rover_map[camera] = rover
                self.rover_camera_map[rover].add(camera)
            elif pred == 'visible' and len(parts) > 2:
                 wp1, wp2 = parts[1], parts[2]
                 visible_graph[wp1].add(wp2)
                 visible_graph[wp2].add(wp1) # Assuming visibility is symmetric

        # Precompute all-pairs shortest paths for each rover
        self.rover_distances = {}
        # Collect all rovers mentioned in can_traverse or equipped facts
        all_rovers = set(self.rover_traversal_graph.keys()) | set(self.rover_equipment.keys())
        for rover in all_rovers:
             self.rover_distances[rover] = {}
             # Need to compute distances from *all* waypoints to all others
             # using the rover's specific traversal graph
             for start_wp in self.all_waypoints:
                 self.rover_distances[rover][start_wp] = bfs(self.rover_traversal_graph.get(rover, {}), start_wp, self.all_waypoints)


        # Precompute communication waypoints (visible from lander)
        self.comm_waypoints = set()
        if self.lander_waypoint and self.lander_waypoint in self.all_waypoints:
            # Find all waypoints reachable from the lander's waypoint in the visible graph
            lander_visible_distances = bfs(visible_graph, self.lander_waypoint, self.all_waypoints)
            # Communication is possible from any waypoint visible *from* the rover's location *to* the lander's location.
            # The visible graph is symmetric. So, any waypoint W where (visible W lander_wp) is true is a comm waypoint.
            # We can find these by checking neighbors of lander_wp in the visible graph.
            self.comm_waypoints = visible_graph.get(self.lander_waypoint, set())


    def get_distance(self, rover, wp_from, wp_to):
        """Helper to get precomputed distance, returning inf if rover or waypoints are invalid/unreachable."""
        if rover not in self.rover_distances:
            return float('inf')
        if wp_from not in self.rover_distances[rover] or wp_to not in self.rover_distances[rover][wp_from]:
             # This case should ideally not happen if all_waypoints is correct and BFS covers all nodes
             return float('inf')
        return self.rover_distances[rover][wp_from][wp_to]


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Extract current state information
        current_rover_location = {} # rover -> waypoint
        current_store_status = {} # store -> 'empty' or 'full'
        current_have_soil = set() # (rover, waypoint)
        current_have_rock = set() # (rover, waypoint)
        current_have_image = set() # (rover, objective, mode)
        current_calibrated = set() # (camera, rover)
        current_soil_samples_at = set() # waypoint
        current_rock_samples_at = set() # waypoint

        # Collect all rovers present in the state to handle cases where a rover might not be in static equipped facts
        all_rovers_in_state = set()

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue
            pred = parts[0]
            if pred == 'at' and len(parts) > 2:
                 # Check if the object is a rover based on its presence in equipped or traversal facts
                 obj_at = parts[1]
                 loc_at = parts[2]
                 if obj_at in self.rover_equipment or obj_at in self.rover_traversal_graph:
                     current_rover_location[obj_at] = loc_at
                     all_rovers_in_state.add(obj_at)
            elif pred == 'empty' and len(parts) > 1:
                 current_store_status[parts[1]] = 'empty'
            elif pred == 'full' and len(parts) > 1:
                 current_store_status[parts[1]] = 'full'
            elif pred == 'have_soil_analysis' and len(parts) > 2:
                 current_have_soil.add((parts[1], parts[2]))
            elif pred == 'have_rock_analysis' and len(parts) > 2:
                 current_have_rock.add((parts[1], parts[2]))
            elif pred == 'have_image' and len(parts) > 3:
                 current_have_image.add((parts[1], parts[2], parts[3]))
            elif pred == 'calibrated' and len(parts) > 2:
                 current_calibrated.add((parts[1], parts[2]))
            elif pred == 'at_soil_sample' and len(parts) > 1:
                 current_soil_samples_at.add(parts[1])
            elif pred == 'at_rock_sample' and len(parts) > 1:
                 current_rock_samples_at.add(parts[1])

        total_h = 0

        # Iterate through goals
        for goal in self.goals:
            if goal in state:
                continue # Goal already achieved

            parts = get_parts(goal)
            if not parts: continue
            pred = parts[0]

            goal_h = float('inf')

            # Goal: Communicate Soil Data (communicated_soil_data ?w)
            if pred == 'communicated_soil_data' and len(parts) > 1:
                waypoint_w = parts[1]
                # Find best rover to achieve this goal
                for rover in all_rovers_in_state: # Consider rovers present in the state
                    if 'soil' in self.rover_equipment.get(rover, set()) and rover in current_rover_location:
                        h_rover = float('inf')
                        rover_curr_wp = current_rover_location[rover]

                        # Option 1: Rover already has the data
                        if (rover, waypoint_w) in current_have_soil:
                            # Cost = move to comm waypoint + communicate
                            min_dist_to_comm = float('inf')
                            for comm_wp in self.comm_waypoints:
                                dist = self.get_distance(rover, rover_curr_wp, comm_wp)
                                min_dist_to_comm = min(min_dist_to_comm, dist)
                            if min_dist_to_comm != float('inf'):
                                h_rover = min_dist_to_comm + 1 # communicate action

                        # Option 2: Sample is available at W
                        elif waypoint_w in current_soil_samples_at:
                            # Cost = move to W + sample (+ drop) + move to comm waypoint + communicate
                            dist_to_sample = self.get_distance(rover, rover_curr_wp, waypoint_w)
                            if dist_to_sample != float('inf'):
                                cost_sample = dist_to_sample + 1 # sample action
                                store = self.rover_store_map.get(rover)
                                if store and current_store_status.get(store) == 'full':
                                    cost_sample += 1 # drop action

                                # Communicate from W
                                min_dist_sample_to_comm = float('inf')
                                for comm_wp in self.comm_waypoints:
                                    dist = self.get_distance(rover, waypoint_w, comm_wp)
                                    min_dist_sample_to_comm = min(min_dist_sample_to_comm, dist)

                                if min_dist_sample_to_comm != float('inf'):
                                     h_rover = cost_sample + min_dist_sample_to_comm + 1 # communicate action

                        goal_h = min(goal_h, h_rover)

            # Goal: Communicate Rock Data (communicated_rock_data ?w)
            elif pred == 'communicated_rock_data' and len(parts) > 1:
                waypoint_w = parts[1]
                # Find best rover to achieve this goal
                for rover in all_rovers_in_state: # Consider rovers present in the state
                    if 'rock' in self.rover_equipment.get(rover, set()) and rover in current_rover_location:
                        h_rover = float('inf')
                        rover_curr_wp = current_rover_location[rover]

                        # Option 1: Rover already has the data
                        if (rover, waypoint_w) in current_have_rock:
                            # Cost = move to comm waypoint + communicate
                            min_dist_to_comm = float('inf')
                            for comm_wp in self.comm_waypoints:
                                dist = self.get_distance(rover, rover_curr_wp, comm_wp)
                                min_dist_to_comm = min(min_dist_to_comm, dist)
                            if min_dist_to_comm != float('inf'):
                                h_rover = min_dist_to_comm + 1 # communicate action

                        # Option 2: Sample is available at W
                        elif waypoint_w in current_rock_samples_at:
                            # Cost = move to W + sample (+ drop) + move to comm waypoint + communicate
                            dist_to_sample = self.get_distance(rover, rover_curr_wp, waypoint_w)
                            if dist_to_sample != float('inf'):
                                cost_sample = dist_to_sample + 1 # sample action
                                store = self.rover_store_map.get(rover)
                                if store and current_store_status.get(store) == 'full':
                                    cost_sample += 1 # drop action

                                # Communicate from W
                                min_dist_sample_to_comm = float('inf')
                                for comm_wp in self.comm_waypoints:
                                    dist = self.get_distance(rover, waypoint_w, comm_wp)
                                    min_dist_sample_to_comm = min(min_dist_sample_to_comm, dist)

                                if min_dist_sample_to_comm != float('inf'):
                                     h_rover = cost_sample + min_dist_sample_to_comm + 1 # communicate action

                        goal_h = min(goal_h, h_rover)


            # Goal: Communicate Image Data (communicated_image_data ?o ?m)
            elif pred == 'communicated_image_data' and len(parts) > 2:
                objective_o, mode_m = parts[1], parts[2]

                # Find best rover/camera to achieve this goal
                for rover in all_rovers_in_state: # Consider rovers present in the state
                    if 'imaging' in self.rover_equipment.get(rover, set()) and rover in current_rover_location:
                        rover_curr_wp = current_rover_location[rover]
                        for camera in self.rover_camera_map.get(rover, set()):
                            if mode_m in self.camera_modes.get(camera, set()):
                                h_rc = float('inf')

                                # Option 1: Rover already has the image
                                if (rover, objective_o, mode_m) in current_have_image:
                                    # Cost = move to comm waypoint + communicate
                                    min_dist_to_comm = float('inf')
                                    for comm_wp in self.comm_waypoints:
                                        dist = self.get_distance(rover, rover_curr_wp, comm_wp)
                                        min_dist_to_comm = min(min_dist_to_comm, dist)
                                    if min_dist_to_comm != float('inf'):
                                        h_rc = min_dist_to_comm + 1 # communicate action

                                # Option 2: Need to take the image
                                else:
                                    calib_target = self.camera_calibration_target.get(camera)
                                    w_calib_set = self.objective_visible_from.get(calib_target, set())
                                    w_image_set = self.objective_visible_from.get(objective_o, set())

                                    if w_calib_set and w_image_set and self.comm_waypoints:
                                        # Cost sequence: curr -> w_calib -> w_image -> w_comm
                                        min_path_cost = float('inf')
                                        for w_calib in w_calib_set:
                                            for w_image in w_image_set:
                                                for w_comm in self.comm_waypoints:
                                                    d1 = self.get_distance(rover, rover_curr_wp, w_calib)
                                                    d2 = self.get_distance(rover, w_calib, w_image)
                                                    d3 = self.get_distance(rover, w_image, w_comm)
                                                    if d1 != float('inf') and d2 != float('inf') and d3 != float('inf'):
                                                        min_path_cost = min(min_path_cost, d1 + d2 + d3)

                                        if min_path_cost != float('inf'):
                                            h_rc = min_path_cost + 3 # calibrate, take_image, communicate actions

                                goal_h = min(goal_h, h_rc)

            # Add the cost for this goal if a finite path was found
            if goal_h != float('inf'):
                total_h += goal_h

        return total_h
