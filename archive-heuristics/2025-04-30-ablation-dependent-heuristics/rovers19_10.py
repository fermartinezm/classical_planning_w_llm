import collections
import math

# Helper function to parse PDDL facts
def parse_fact(fact_string):
    """Parses a PDDL fact string into predicate and arguments."""
    # Removes leading/trailing brackets and splits by space
    parts = fact_string[1:-1].split()
    predicate = parts[0]
    objects = parts[1:]
    return predicate, objects

# BFS function to compute shortest paths
def bfs(graph, start_node, all_nodes):
    """Computes shortest path distances from start_node in a graph."""
    distances = {node: math.inf for node in all_nodes}
    if start_node not in all_nodes:
         # Start node is not in the set of relevant nodes, cannot start BFS
         return distances

    distances[start_node] = 0
    queue = collections.deque([start_node])

    # Ensure graph only contains nodes from all_nodes
    graph_subset = {node: [n for n in neighbors if n in all_nodes] for node, neighbors in graph.items() if node in all_nodes}

    while queue:
        current_node = queue.popleft()
        # Use graph_subset to only consider relevant nodes and edges
        for neighbor in graph_subset.get(current_node, []):
            if distances[neighbor] == math.inf:
                distances[neighbor] = distances[current_node] + 1
                queue.append(neighbor)
    return distances

class roversHeuristic:
    """
    Domain-dependent heuristic for the Rovers domain.

    Summary:
    Estimates the cost to reach the goal by summing the estimated costs
    for each unachieved goal fact. The cost for each goal fact is estimated
    as the sum of the cost to acquire the necessary data (sample or image)
    and the cost to communicate that data. Navigation costs are estimated
    using precomputed shortest paths for each rover on its traversable graph.

    Assumptions:
    - The heuristic assumes that if a soil or rock sample is required by a goal
      and is no longer present on the ground (`at_soil_sample` or `at_rock_sample`
      is false in the current state) and no rover currently holds the sample
      (`have_soil_analysis` or `have_rock_analysis` is false for all rovers),
      then the goal is impossible to achieve in a solvable problem instance.
      (Note: The refined logic in the code assumes it *is* possible to sample
      if a rover is equipped, regardless of `at_soil_sample` in the current state,
      estimating the cost as if the sample were still there if needed. This is a relaxation).
    - The heuristic assumes that image calibration is required before taking a new image,
      unless the camera is already calibrated (though the `take_image` action uncalibrates).
      For simplicity, when calculating the cost to get a *new* image, it assumes a calibration
      action is always needed immediately before the `take_image` action.
    - The heuristic calculates costs for each unachieved goal independently and sums them
      (additive heuristic). This ignores potential positive interactions (e.g., one trip
      to the lander can communicate multiple items) and thus is not admissible but
      can be effective for greedy search.
    - The heuristic assumes communication happens from the location where the data was
      acquired (sample location or image location) if the data is not already held
      by a rover. If the data is already held, communication cost is estimated from
      the holding rover's current location.

    Heuristic Initialization:
    The `__init__` method parses the static facts from the PDDL problem. It extracts:
    - The lander's location.
    - Each rover's capabilities (soil, rock, imaging).
    - Which store belongs to which rover.
    - Information about each camera (which rover it's on, supported modes, calibration target).
    - The `can_traverse` graph for each rover.
    - The `visible` graph between waypoints.
    - Waypoints from which each objective is visible (`visible_from`).
    - Waypoints from which each camera's calibration target is visible.
    It then precomputes:
    - Shortest path distances between all pairs of relevant waypoints for each rover
      using Breadth-First Search (BFS) on the rover's `can_traverse` graph.
    - The set of waypoints from which the lander is visible (`lander_comm_wps`).

    Step-By-Step Thinking for Computing Heuristic:
    1.  Check if the current state `state` satisfies all goal conditions (`task.goal_reached(state)`). If yes, the heuristic value is 0.
    2.  Initialize the total heuristic value `h` to 0.
    3.  Identify the set of goal facts that are present in `task.goals` but not in the current `state`. These are the unachieved goals.
    4.  Extract dynamic information from the current `state`: rover locations, store statuses, which samples/images rovers have, and which cameras are calibrated.
    5.  For each unachieved goal fact:
        -   **If the goal is `(communicated_X_data ?w)`:**
            -   If any rover currently in `state` has `(have_X_analysis ?r ?w)`, the cost to get the sample is 0. The remaining cost is only to communicate it. Find the minimum cost for any rover *currently holding the sample* to navigate to a waypoint visible from the lander and perform the `communicate_soil_data` action (+1 cost).
            -   If no rover currently holds the sample, the cost includes getting the sample *and* communicating it.
                -   Cost to get sample: Find the minimum cost for a soil-equipped rover to navigate from its current location to `?w`, perform `sample_soil/rock` (+1 cost), potentially performing `drop` (+1 cost) if its store is full. Use precomputed distances.
                -   Cost to communicate: If a rover currently holds the sample, find the minimum cost for that rover to navigate to a lander communication waypoint and communicate (+1 cost). If no rover holds the sample (meaning it needs to be sampled), estimate the cost for *any* rover to navigate from `?w` (the sample location) to a waypoint visible from the lander, and perform `communicate_soil_data` (+1 cost). This minimum is taken over all rovers capable of traversing from `?w` and all lander communication waypoints.
                -   The total cost for this goal is the sum of the minimum cost to get the sample (over suitable rovers) and the minimum cost to communicate from `?w` (over suitable rovers and communication points).
            -   If any required step is impossible (e.g., no equipped rover, no path), a large penalty (`IMPOSSIBLE_COST`) is added to `h`.
        -   **If the goal is `(communicated_rock_data ?w)`:** The logic is analogous to `communicated_soil_data`.
        -   **If the goal is `(communicated_image_data ?o ?m)`:**
            -   If any rover currently in `state` has `(have_image ?r ?o ?m)`, the cost to get the image is 0. The remaining cost is only to communicate it. Find the minimum cost for any rover *currently holding the image* to navigate to a waypoint visible from the lander and perform the `communicate_image_data` action (+1 cost).
            -   If no rover currently holds the image, the cost includes getting the image *and* communicating it.
                -   Cost to get image: If any rover currently in `state` has `(have_image ?r ?o ?m)`, cost_get = 0. Otherwise, find the minimum cost for a suitable rover/camera pair to navigate from its current location to a calibration waypoint for the camera's target, perform `calibrate` (+1 cost), navigate to a waypoint visible from objective `?o`, and perform `take_image` (+1 cost). This minimum is taken over suitable rovers, cameras, calibration waypoints, and image waypoints.
                -   Cost to communicate: If a rover currently holds the image, find the minimum cost for that rover to navigate to a lander communication waypoint and communicate (+1 cost). If no rover holds the image (meaning it needs to be taken), estimate the cost for *any* rover to navigate from the image waypoint (where it was taken) to a lander communication waypoint and communicate (+1 cost).
                -   The total cost for this goal is the sum of the minimum cost to get the image (over suitable rover/camera/waypoint combinations) and the minimum cost to communicate from the image waypoint (over suitable rovers and communication points).
            -   If any required step is impossible, a large penalty (`IMPOSSIBLE_COST`) is added to `h`.
    6.  Return the total heuristic value `h`.

        Args:
            state: A frozenset of PDDL facts representing the current state.
            task: The planning task object.

        Returns:
            An integer estimate of the remaining cost to reach the goal.
        """
        # 1. Check if goal reached
        if task.goal_reached(state):
            return 0

        # Define a large cost for impossible subgoals
        IMPOSSIBLE_COST = 1000

        # 4. Extract dynamic information from state
        rover_locations = {} # rover -> waypoint
        store_status = {} # store -> 'empty' or 'full'
        have_soil_analysis = collections.defaultdict(set) # rover -> {waypoint}
        have_rock_analysis = collections.defaultdict(set) # rover -> {waypoint}
        have_image = collections.defaultdict(set) # rover -> {(objective, mode)}
        calibrated_cameras = collections.defaultdict(set) # rover -> {camera}
        # at_soil_sample_wps = set() # {waypoint} # Not strictly needed for this heuristic logic
        # at_rock_sample_wps = set() # {waypoint} # Not strictly needed for this heuristic logic

        for fact_string in state:
            pred, objs = parse_fact(fact_string)
            if pred == 'at':
                # (at ?x - rover ?y - waypoint)
                rover_locations[objs[0]] = objs[1]
            elif pred == 'empty':
                # (empty ?s - store)
                store_status[objs[0]] = 'empty'
            elif pred == 'full':
                # (full ?s - store)
                store_status[objs[0]] = 'full'
            elif pred == 'have_soil_analysis':
                # (have_soil_analysis ?r - rover ?w - waypoint)
                have_soil_analysis[objs[0]].add(objs[1])
            elif pred == 'have_rock_analysis':
                # (have_rock_analysis ?r - rover ?w - waypoint)
                have_rock_analysis[objs[0]].add(objs[1])
            elif pred == 'have_image':
                # (have_image ?r - rover ?o - objective ?m - mode)
                have_image[objs[0]].add((objs[1], objs[2]))
            elif pred == 'calibrated':
                # (calibrated ?c - camera ?r - rover)
                calibrated_cameras[objs[1]].add(objs[0])
            # elif pred == 'at_soil_sample':
            #      at_soil_sample_wps.add(objs[0])
            # elif pred == 'at_rock_sample':
            #      at_rock_sample_wps.add(objs[0])


        # 2. Initialize heuristic
        h = 0

        # 3. Identify unachieved goals
        unachieved_goals = task.goals - state

        # 5. Estimate cost for each unachieved goal
        for goal_fact_string in unachieved_goals:
            pred, objs = parse_fact(goal_fact_string)

            if pred == 'communicated_soil_data':
                # (communicated_soil_data ?w - waypoint)
                waypoint = objs[0]

                # Cost to get sample from waypoint
                cost_get_sample = float('inf')
                # Check if any rover currently has the sample
                rovers_with_sample = [r for r in have_soil_analysis if waypoint in have_soil_analysis[r]]

                if rovers_with_sample:
                    cost_get_sample = 0
                else:
                    # Need to sample it. Find the best rover to do it.
                    for rover in self.rover_capabilities:
                        if 'soil' in self.rover_capabilities[rover]:
                            current_w = rover_locations.get(rover)
                            if current_w: # Rover location must be known
                                nav_cost = self.rover_distances.get(rover, {}).get(current_w, {}).get(waypoint, float('inf'))
                                if nav_cost != float('inf'):
                                    sample_cost = nav_cost + 1 # navigate + sample_soil
                                    # Check if store is full, requiring a drop action
                                    store_for_rover = None
                                    for s, r_owner in self.store_owner.items():
                                        if r_owner == rover:
                                            store_for_rover = s
                                            break
                                    if store_for_rover and store_status.get(store_for_rover) == 'full':
                                         sample_cost += 1 # Add cost for drop action

                                    cost_get_sample = min(cost_get_sample, sample_cost)

                # Cost to communicate data
                cost_communicate = float('inf')
                if rovers_with_sample:
                    # If a rover currently has the sample, estimate communication cost from its current location
                    for rover_comm in rovers_with_sample:
                        current_x = rover_locations.get(rover_comm)
                        if current_x:
                            min_dist_to_comm = float('inf')
                            for comm_w in self.lander_comm_wps:
                                dist = self.rover_distances.get(rover_comm, {}).get(current_x, {}).get(comm_w, float('inf'))
                                min_dist_to_comm = min(min_dist_to_comm, dist)
                            if min_dist_to_comm != float('inf'):
                                cost_communicate = min(cost_communicate, min_dist_to_comm + 1) # navigate + communicate
                else:
                    # If sample needs to be acquired, estimate communication cost from the sample location 'waypoint'
                    min_dist_sample_to_comm = float('inf')
                    for rover_nav in self.rover_can_traverse: # Any rover that can navigate from 'waypoint'
                         # Ensure waypoint is a valid start node for this rover's graph
                         if waypoint in self.rover_distances.get(rover_nav, {}):
                             for comm_w in self.lander_comm_wps:
                                  dist = self.rover_distances.get(rover_nav, {}).get(waypoint, {}).get(comm_w, float('inf'))
                                  min_dist_sample_to_comm = min(min_dist_sample_to_comm, dist)
                    if min_dist_sample_to_comm != float('inf'):
                         cost_communicate = min_dist_sample_to_comm + 1 # navigate + communicate


                # Total cost for this goal
                if cost_get_sample == float('inf') or cost_communicate == float('inf'):
                     h += IMPOSSIBLE_COST
                else:
                     h += cost_get_sample + cost_communicate


            elif pred == 'communicated_rock_data':
                # (communicated_rock_data ?w - waypoint)
                waypoint = objs[0]

                # Cost to get sample from waypoint
                cost_get_sample = float('inf')
                rovers_with_sample = [r for r in have_rock_analysis if waypoint in have_rock_analysis[r]]

                if rovers_with_sample:
                    cost_get_sample = 0
                else:
                    # Need to sample it. Find the best rover to do it.
                    for rover in self.rover_capabilities:
                        if 'rock' in self.rover_capabilities[rover]:
                            current_w = rover_locations.get(rover)
                            if current_w: # Rover location must be known
                                nav_cost = self.rover_distances.get(rover, {}).get(current_w, {}).get(waypoint, float('inf'))
                                if nav_cost != float('inf'):
                                    sample_cost = nav_cost + 1 # navigate + sample_rock
                                    # Check if store is full, requiring a drop action
                                    store_for_rover = None
                                    for s, r_owner in self.store_owner.items():
                                        if r_owner == rover:
                                            store_for_rover = s
                                            break
                                    if store_for_rover and store_status.get(store_for_rover) == 'full':
                                         sample_cost += 1 # Add cost for drop action

                                    cost_get_sample = min(cost_get_sample, sample_cost)

                # Cost to communicate data
                cost_communicate = float('inf')
                if rovers_with_sample:
                    # If a rover currently has the sample, estimate communication cost from its current location
                    for rover_comm in rovers_with_sample:
                        current_x = rover_locations.get(rover_comm)
                        if current_x:
                            min_dist_to_comm = float('inf')
                            for comm_w in self.lander_comm_wps:
                                dist = self.rover_distances.get(rover_comm, {}).get(current_x, {}).get(comm_w, float('inf'))
                                min_dist_to_comm = min(min_dist_to_comm, dist)
                            if min_dist_to_comm != float('inf'):
                                cost_communicate = min(cost_communicate, min_dist_to_comm + 1) # navigate + communicate
                else:
                    # If sample needs to be acquired, estimate communication cost from the sample location 'waypoint'
                    min_dist_sample_to_comm = float('inf')
                    for rover_nav in self.rover_can_traverse: # Any rover that can navigate from 'waypoint'
                         # Ensure waypoint is a valid start node for this rover's graph
                         if waypoint in self.rover_distances.get(rover_nav, {}):
                             for comm_w in self.lander_comm_wps:
                                  dist = self.rover_distances.get(rover_nav, {}).get(waypoint, {}).get(comm_w, float('inf'))
                                  min_dist_sample_to_comm = min(min_dist_sample_to_comm, dist)
                    if min_dist_sample_to_comm != float('inf'):
                         cost_communicate = min_dist_sample_to_comm + 1 # navigate + communicate

                # Total cost for this goal
                if cost_get_sample == float('inf') or cost_communicate == float('inf'):
                     h += IMPOSSIBLE_COST
                else:
                     h += cost_get_sample + cost_communicate


            elif pred == 'communicated_image_data':
                # (communicated_image_data ?o - objective ?m - mode)
                objective, mode = objs

                # Cost to get image
                cost_get_image = float('inf')
                # Check if any rover currently has the image
                rovers_with_image = [r for r in have_image if (objective, mode) in have_image[r]]

                if rovers_with_image:
                    cost_get_image = 0
                else:
                    # Need to take image. Find the best rover/camera.
                    min_cost_acquire_image = float('inf')
                    best_image_w_for_acquire = None # Track the image_w that gives the min acquire cost

                    for rover in self.rover_capabilities:
                        if 'imaging' in self.rover_capabilities[rover]:
                            current_w = rover_locations.get(rover)
                            if not current_w: continue

                            for camera, cam_info in self.camera_info.items():
                                if cam_info['rover'] == rover and mode in cam_info['modes']:
                                    cal_target = cam_info['calibration_target']
                                    if not cal_target: continue # Camera has no calibration target

                                    cal_wps = self.visible_from_objective.get(cal_target, set())
                                    image_wps = self.visible_from_objective.get(objective, set())

                                    if not cal_wps or not image_wps: continue # Cannot calibrate or see objective

                                    min_cost_this_pair = float('inf')
                                    current_best_image_w_this_pair = None

                                    for cal_w in cal_wps:
                                        for image_w in image_wps:
                                            nav1_cost = self.rover_distances.get(rover, {}).get(current_w, {}).get(cal_w, float('inf'))
                                            if nav1_cost == float('inf'): continue

                                            nav2_cost = self.rover_distances.get(rover, {}).get(cal_w, {}).get(image_w, float('inf'))
                                            if nav2_cost == float('inf'): continue

                                            # Cost: nav to cal_w + calibrate + nav to image_w + take_image
                                            # Assume calibration is needed before taking a new image.
                                            cost_path = nav1_cost + 1 + nav2_cost + 1

                                            if cost_path < min_cost_this_pair:
                                                 min_cost_this_pair = cost_path
                                                 current_best_image_w_this_pair = image_w

                                    if min_cost_this_pair != float('inf'):
                                         if min_cost_this_pair < min_cost_acquire_image:
                                              min_cost_acquire_image = min_cost_this_pair
                                              best_image_w_for_acquire = current_best_image_w_this_pair

                    cost_get_image = min_cost_acquire_image


                # Cost to communicate image
                cost_communicate = float('inf')
                if rovers_with_image:
                    # If a rover currently has the image, estimate communication cost from its current location
                    for rover_comm in rovers_with_image:
                        current_x = rover_locations.get(rover_comm)
                        if current_x:
                            min_dist_to_comm = float('inf')
                            for comm_w in self.lander_comm_wps:
                                dist = self.rover_distances.get(rover_comm, {}).get(current_x, {}).get(comm_w, float('inf'))
                                min_dist_to_comm = min(min_dist_to_comm, dist)
                            if min_dist_to_comm != float('inf'):
                                cost_communicate = min(cost_communicate, min_dist_to_comm + 1) # navigate + communicate
                elif best_image_w_for_acquire is not None:
                    # If image needs to be acquired, estimate communication cost from the best image location found
                    image_w = best_image_w_for_acquire
                    min_dist_image_to_comm = float('inf')
                    for rover_nav in self.rover_can_traverse: # Any rover that can navigate from 'image_w'
                         # Ensure image_w is a valid start node for this rover's graph
                         if image_w in self.rover_distances.get(rover_nav, {}):
                             for comm_w in self.lander_comm_wps:
                                  dist = self.rover_distances.get(rover_nav, {}).get(image_w, {}).get(comm_w, float('inf'))
                                  min_dist_image_to_comm = min(min_dist_image_to_comm, dist)
                    if min_dist_image_to_comm != float('inf'):
                         cost_communicate = min_dist_image_to_comm + 1 # navigate + communicate
                # else: # If best_image_w_for_acquire is None, it means acquire was impossible, cost_communicate remains inf

                # Total cost for this goal
                if cost_get_image == float('inf') or cost_communicate == float('inf'):
                     h += IMPOSSIBLE_COST
                else:
                     h += cost_get_image + cost_communicate

            # Add cost for this goal to total heuristic
            # (Costs were added directly inside the if/elif blocks)

        # Ensure heuristic is 0 only for goal states (checked at the beginning)
        # Ensure heuristic is finite for solvable states (handled by IMPOSSIBLE_COST)

        return h
