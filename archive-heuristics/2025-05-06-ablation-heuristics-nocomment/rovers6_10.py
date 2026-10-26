from heuristics.heuristic_base import Heuristic
from collections import deque

# Helper function to parse facts
def get_parts(fact):
    """Extracts predicate and arguments from a fact string."""
    # Remove parentheses and split by space
    return fact[1:-1].split()

class roversHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Rovers domain.

    Summary:
    This heuristic estimates the remaining cost to reach the goal state by
    summing the estimated costs for each individual unachieved goal fact.
    For each unachieved goal, it calculates the minimum cost required for
    any suitable rover to complete the necessary steps (sampling/imaging
    and communication) and takes the minimum cost over all suitable rovers
    (and cameras for imaging goals). Navigation costs are precomputed using
    Breadth-First Search (BFS) on the rover-specific traversable waypoint graphs.
    Action costs (sample, drop, calibrate, take_image, communicate) are assumed to be 1.

    Assumptions:
    - Navigation cost between adjacent visible waypoints traversable by a rover is 1.
    - Action costs (sample, drop, calibrate, take_image, communicate) are 1.
    - A single rover is primarily responsible for achieving and communicating a single goal fact.
    - Samples (soil/rock) are consumed upon sampling from the waypoint.
    - Dropped samples cannot be re-picked up.
    - Calibration is consumed by taking an image.
    - Navigation distances are precomputed using BFS on the rover's traversable graph.
    - If a required waypoint (sample, image, cal, comm) is unreachable for a rover,
      that rover cannot achieve the goal via that path. If no rover can achieve
      the goal, the cost is infinite.
    - If a soil/rock sample was initially at a waypoint but is now gone from the
      waypoint and not held by any rover, it is considered lost, and the goal
      requiring it is unreachable.

    Heuristic Initialization:
    The `__init__` method performs the following precomputation steps:
    - Stores the initial state of the task to check initial sample locations.
    - Parses static facts to extract:
        - Lander location.
        - Waypoints visible from the lander location (communication points).
        - Rover capabilities (soil, rock, imaging).
        - Store ownership for each rover.
        - Cameras on board each rover.
        - Supported modes for each camera.
        - Calibration target for each camera.
        - Waypoints from which each objective is visible.
    - Builds a navigation graph for each rover based on the `can_traverse` facts.
    - Precomputes all-pairs shortest path distances for each rover within its
      traversable graph using BFS.

    Step-By-Step Thinking for Computing Heuristic:
    The `__call__` method computes the heuristic value for a given state:
    1. Initialize the total heuristic value `h = 0`.
    2. Check if the current state is a goal state. If `self.goals` is a subset
       of the current `state`, return 0.
    3. Iterate through each goal fact `g` in `self.goals`.
    4. If `g` is already present in the current `state`, this goal is achieved;
       continue to the next goal.
    5. If `g` is `(communicated_soil_data ?w)`:
       a. Initialize `min_goal_cost = infinity`.
       b. Check if any rover `r_check` currently has the sample `(have_soil_analysis r_check w)`.
          - If yes (a `rover_with_sample` is found): The cost is for this rover
            to navigate from its current location to the nearest communication
            waypoint and perform the `communicate_soil_data` action (cost 1).
            Calculate `dist(rover_with_sample, loc_r_with_sample, comm_wp) + 1`,
            minimizing over all `comm_wp`. Update `min_goal_cost`.
          - If no rover has the sample: Check if `(at_soil_sample w)` is in the
            current state.
            - If yes (sample is at the waypoint): A capable rover needs to sample
              and communicate. Iterate through rovers `r` with 'soil' capability
              and a store. For each capable rover `r`:
                - Calculate cost to sample: `dist(r, loc_r, w)` (navigate to sample)
                  + (1 if rover's store is full, for `drop` action) + 1 (`sample_soil`).
                  The rover is then at waypoint `w`.
                - Calculate cost to communicate from `w`: Find minimum
                  `dist(r, w, comm_wp)` over `comm_wp`. Add `min_comm_nav_dist + 1`
                  (`communicate_soil_data`).
                - Sum these costs (`current_rover_cost`). If all navigation steps
                  were possible, update `min_goal_cost = min(min_goal_cost, current_rover_cost)`.
            - If no (sample is gone from waypoint): The goal is unreachable via
              sampling from this waypoint. `min_goal_cost` remains infinity.
       c. If `min_goal_cost` is still infinity after checking all possibilities,
          return infinity (the problem is likely unsolvable from this state).
       d. Add `min_goal_cost` to the total heuristic value `h`.
    6. If `g` is `(communicated_rock_data ?w)`: Follow the same logic as soil data,
       using 'rock' capability and `sample_rock` action.
    7. If `g` is `(communicated_image_data ?o ?m)`:
       a. Initialize `min_goal_cost = infinity`.
       b. Check if any rover `r_check` currently has the image `(have_image r_check o m)`.
          - If yes (a `rover_with_image` is found): The cost is for this rover
            to navigate from its current location to the nearest communication
            waypoint and perform the `communicate_image_data` action (cost 1).
            Calculate `dist(rover_with_image, loc_r_with_image, comm_wp) + 1`,
            minimizing over all `comm_wp`. Update `min_goal_cost`.
          - If no rover has the image: Need to take the image and communicate.
            Iterate through rovers `r` with imaging capability and cameras `c`
            supporting mode `m` on rover `r`.
            - For each suitable (r, c) pair:
               i. Find `loc_r`.
               ii. Calculate cost to take image (requires calibration first):
                   - If camera `c` is not calibrated `(calibrated c r)`:
                     Find `cal_target` for `c`. Find minimum `dist(r, loc_r, w_cal)`
                     over `w_cal` visible from `cal_target`. Add `min_cal_nav_dist + 1`
                     (`calibrate`). The rover is then at `best_cal_wp`. Update
                     `loc_after_cal = best_cal_wp`.
                   - If camera `c` is already calibrated: `loc_after_cal = loc_r`.
                   - If calibration step was possible (a reachable `w_cal` was found or already calibrated):
                     Find minimum `dist(r, loc_after_cal, w_img)` over `w_img` visible
                     from `o`. Add `min_img_nav_dist + 1` (`take_image`). The rover is
                     then at `best_img_wp`. Update `loc_after_image = best_img_wp`.
                   - If image taking step was possible (a reachable `w_img` was found):
                     Calculate cost to communicate: Find minimum `dist(r, loc_after_image, comm_wp)`
                     over `comm_wp`. Add `min_comm_nav_dist + 1` (`communicate_image_data`).
                     If communication was possible, update `min_goal_cost = min(min_goal_cost, current_rover_cost)`.
       c. If `min_goal_cost` is still infinity after checking all possibilities,
          return infinity.
       d. Add `min_goal_cost` to `h`.
    8. Return the total heuristic value `h`.
    """

    def __init__(self, task):
        self.task_initial_state = task.initial_state # Store initial state
        self.goals = task.goals
        static_facts = task.static

        # --- Precompute Static Information ---
        self.lander_location = None
        self.comm_waypoints = set() # Waypoints visible from lander
        self.rover_capabilities = {} # {rover: {soil, rock, imaging}}
        self.rover_stores = {} # {rover: store}
        self.rover_cameras = {} # {rover: {camera}}
        self.camera_modes = {} # {camera: {mode}}
        self.camera_cal_target = {} # {camera: objective}
        self.objective_visible_from = {} # {objective: {waypoint}}
        self.waypoint_graph = {} # {rover: {waypoint: {neighbor_waypoint}}}
        self.waypoints = set() # All waypoints mentioned in can_traverse

        # Identify all rovers
        rovers = set()
        for fact in static_facts:
             parts = get_parts(fact)
             if parts[0] == "store_of": rovers.add(parts[2])
             elif parts[0] == "on_board": rovers.add(parts[2])
             elif parts[0] == "equipped_for_soil_analysis": rovers.add(parts[1])
             elif parts[0] == "equipped_for_rock_analysis": rovers.add(parts[1])
             elif parts[0] == "equipped_for_imaging": rovers.add(parts[1])
        self.rovers = rovers

        for rover in self.rovers:
             self.rover_capabilities[rover] = set()
             self.waypoint_graph[rover] = {}

        for fact in static_facts:
            parts = get_parts(fact)
            predicate = parts[0]

            if predicate == "at_lander":
                self.lander_location = parts[2]
            elif predicate == "can_traverse":
                 rover, wp1, wp2 = parts[1], parts[2], parts[3]
                 self.waypoint_graph[rover].setdefault(wp1, set()).add(wp2)
                 self.waypoints.add(wp1)
                 self.waypoints.add(wp2)
            elif predicate == "equipped_for_soil_analysis":
                self.rover_capabilities[parts[1]].add("soil")
            elif predicate == "equipped_for_rock_analysis":
                self.rover_capabilities[parts[1]].add("rock")
            elif predicate == "equipped_for_imaging":
                self.rover_capabilities[parts[1]].add("imaging")
            elif predicate == "store_of":
                self.rover_stores[parts[2]] = parts[1]
            elif predicate == "on_board":
                self.rover_cameras.setdefault(parts[2], set()).add(parts[1])
            elif predicate == "supports":
                self.camera_modes.setdefault(parts[1], set()).add(parts[2])
            elif predicate == "calibration_target":
                self.camera_cal_target[parts[1]] = parts[2]
            elif predicate == "visible_from":
                self.objective_visible_from.setdefault(parts[1], set()).add(parts[2])

        # Determine communication waypoints (visible from lander)
        if self.lander_location:
             for fact in static_facts:
                 parts = get_parts(fact)
                 if parts[0] == "visible":
                     if parts[1] == self.lander_location:
                         self.comm_waypoints.add(parts[2])
                     if parts[2] == self.lander_location:
                          self.comm_waypoints.add(parts[1]) # Visible is symmetric

        # --- Precompute Navigation Distances (BFS) ---
        self.rover_distances = {} # {rover: {start_wp: {end_wp: dist}}}
        for rover in self.rovers:
            self.rover_distances[rover] = {}
            # Run BFS from all waypoints that appear in the rover's graph
            all_relevant_wps = set(self.waypoint_graph[rover].keys())
            # Also include waypoints that might only have incoming edges
            for fact in static_facts:
                 parts = get_parts(fact)
                 if parts[0] == "can_traverse" and parts[1] == rover:
                      all_relevant_wps.add(parts[2])
                      all_relevant_wps.add(parts[3])

            for start_wp in all_relevant_wps:
                 self.rover_distances[rover][start_wp] = self._bfs(rover, start_wp)

    def _bfs(self, rover, start_waypoint):
        """Performs BFS for a given rover starting from a waypoint."""
        distances = {start_waypoint: 0}
        queue = deque([start_waypoint])
        graph = self.waypoint_graph.get(rover, {})

        while queue:
            current_wp = queue.popleft()

            # Ensure current_wp exists in the graph for this rover
            if current_wp not in graph:
                continue

            for neighbor_wp in graph[current_wp]:
                if neighbor_wp not in distances:
                    distances[neighbor_wp] = distances[current_wp] + 1
                    queue.append(neighbor_wp)
        return distances

    def get_distance(self, rover, start_wp, end_wp):
        """Looks up precomputed distance or returns infinity."""
        if rover not in self.rover_distances or start_wp not in self.rover_distances[rover]:
             return float('inf') # Rover cannot start here or doesn't exist in graph distances
        return self.rover_distances[rover].get(end_wp, float('inf'))

    def find_rover_location(self, state, rover):
        """Finds the current waypoint of a rover in the state."""
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at" and parts[1] == rover:
                return parts[2]
        return None # Should not happen in a valid state representation

    def __call__(self, node):
        """
        Computes the heuristic value for the given state.

        Args:
            node: The search node containing the state.

        Returns:
            The estimated cost (integer) to reach the goal.
        """
        state = node.state
        h = 0

        # Check if goal is reached
        if self.goals <= state:
            return 0

        for goal in self.goals:
            if goal in state:
                continue

            parts = get_parts(goal)
            predicate = parts[0]

            min_goal_cost = float('inf')

            if predicate == "communicated_soil_data":
                waypoint_w = parts[1]

                # Check if any rover currently has the sample
                rover_with_sample = None
                for r_check in self.rovers:
                    if f'(have_soil_analysis {r_check} {waypoint_w})' in state:
                        rover_with_sample = r_check
                        break

                if rover_with_sample:
                    # Sample is held by rover_with_sample. Need to communicate it.
                    current_loc_r_with_sample = self.find_rover_location(state, rover_with_sample)
                    if current_loc_r_with_sample is None: continue

                    comm_cost = 0
                    min_comm_nav_dist = float('inf')
                    for comm_wp in self.comm_waypoints:
                        dist = self.get_distance(rover_with_sample, current_loc_r_with_sample, comm_wp)
                        min_comm_nav_dist = min(min_comm_nav_dist, dist)

                    if min_comm_nav_dist != float('inf'):
                        comm_cost += min_comm_nav_dist # Navigate to comm point
                        comm_cost += 1 # Communicate
                        min_goal_cost = min(min_goal_cost, comm_cost)

                else:
                    # No rover has the sample. Check if it's still at the waypoint.
                    sample_at_waypoint = f'(at_soil_sample {waypoint_w})' in state

                    if sample_at_waypoint:
                        # Sample is at the waypoint. Need to sample and communicate.
                        for rover in self.rovers:
                            if "soil" not in self.rover_capabilities.get(rover, set()):
                                continue # Rover cannot sample soil
                            store_r = self.rover_stores.get(rover)
                            if store_r is None:
                                continue # Rover has no store

                            current_loc_r = self.find_rover_location(state, rover)
                            if current_loc_r is None: continue

                            current_rover_cost = 0

                            # Need to sample
                            dist_to_sample = self.get_distance(rover, current_loc_r, waypoint_w)
                            if dist_to_sample == float('inf'): continue # Cannot reach sample location

                            current_rover_cost += dist_to_sample # Navigate to sample
                            loc_at_sample = waypoint_w

                            if f'(full {store_r})' in state:
                                current_rover_cost += 1 # Drop

                            current_rover_cost += 1 # Sample
                            loc_after_sample = loc_at_sample # Rover is now at sample location

                            # Now handle communication for this rover
                            min_comm_nav_dist = float('inf')
                            for comm_wp in self.comm_waypoints:
                                dist = self.get_distance(rover, loc_after_sample, comm_wp)
                                min_comm_nav_dist = min(min_comm_nav_dist, dist)

                            if min_comm_nav_dist == float('inf'):
                                continue # Cannot reach communication point

                            current_rover_cost += min_comm_nav_dist # Navigate to comm point
                            current_rover_cost += 1 # Communicate

                            min_goal_cost = min(min_goal_cost, current_rover_cost)
                    # Else: Sample is gone from waypoint and not held. Goal is unreachable. min_goal_cost remains infinity.


                if min_goal_cost == float('inf'):
                     return float('inf') # Goal unreachable
                h += min_goal_cost

            elif predicate == "communicated_rock_data":
                waypoint_w = parts[1]

                rover_with_sample = None
                for r_check in self.rovers:
                    if f'(have_rock_analysis {r_check} {waypoint_w})' in state:
                        rover_with_sample = r_check
                        break

                if rover_with_sample:
                    current_loc_r_with_sample = self.find_rover_location(state, rover_with_sample)
                    if current_loc_r_with_sample is None: continue

                    comm_cost = 0
                    min_comm_nav_dist = float('inf')
                    for comm_wp in self.comm_waypoints:
                        dist = self.get_distance(rover_with_sample, current_loc_r_with_sample, comm_wp)
                        min_comm_nav_dist = min(min_comm_nav_dist, dist)

                    if min_comm_nav_dist != float('inf'):
                        comm_cost += min_comm_nav_dist
                        comm_cost += 1
                        min_goal_cost = min(min_goal_cost, comm_cost)

                else:
                    sample_at_waypoint = f'(at_rock_sample {waypoint_w})' in state

                    if sample_at_waypoint:
                        for rover in self.rovers:
                            if "rock" not in self.rover_capabilities.get(rover, set()):
                                continue
                            store_r = self.rover_stores.get(rover)
                            if store_r is None:
                                continue # Rover has no store

                            current_loc_r = self.find_rover_location(state, rover)
                            if current_loc_r is None: continue

                            current_rover_cost = 0

                            dist_to_sample = self.get_distance(rover, current_loc_r, waypoint_w)
                            if dist_to_sample == float('inf'): continue

                            current_rover_cost += dist_to_sample
                            loc_at_sample = waypoint_w

                            if f'(full {store_r})' in state:
                                current_rover_cost += 1

                            current_rover_cost += 1
                            loc_after_sample = loc_at_sample

                            min_comm_nav_dist = float('inf')
                            for comm_wp in self.comm_waypoints:
                                dist = self.get_distance(rover, loc_after_sample, comm_wp)
                                min_comm_nav_dist = min(min_comm_nav_dist, dist)

                            if min_comm_nav_dist == float('inf'):
                                continue

                            current_rover_cost += min_comm_nav_dist
                            current_rover_cost += 1

                            min_goal_cost = min(min_goal_cost, current_rover_cost)
                    # Else: Sample is gone from waypoint and not held. Goal is unreachable.

                if min_goal_cost == float('inf'):
                     return float('inf')
                h += min_goal_cost


            elif predicate == "communicated_image_data":
                objective_o = parts[1]
                mode_m = parts[2]

                # Check if any rover currently has the image
                rover_with_image = None
                for r_check in self.rovers:
                    if f'(have_image {r_check} {objective_o} {mode_m})' in state:
                        rover_with_image = r_check
                        break

                if rover_with_image:
                    # Image is held by rover_with_image. Need to communicate it.
                    current_loc_r_with_image = self.find_rover_location(state, rover_with_image)
                    if current_loc_r_with_image is None: continue

                    comm_cost = 0
                    min_comm_nav_dist = float('inf')
                    for comm_wp in self.comm_waypoints:
                        dist = self.get_distance(rover_with_image, current_loc_r_with_image, comm_wp)
                        min_comm_nav_dist = min(min_comm_nav_dist, dist)

                    if min_comm_nav_dist != float('inf'):
                        comm_cost += min_comm_nav_dist
                        comm_cost += 1
                        min_goal_cost = min(min_goal_cost, comm_cost)

                else:
                    # Image is not held by any rover. Need to take it and communicate.
                    for rover in self.rovers:
                        if "imaging" not in self.rover_capabilities.get(rover, set()):
                            continue

                        current_loc_r = self.find_rover_location(state, rover)
                        if current_loc_r is None: continue

                        for camera in self.rover_cameras.get(rover, set()):
                            if mode_m not in self.camera_modes.get(camera, set()):
                                continue

                            current_rover_cost = 0

                            # Need to take image (requires calibration first)
                            is_calibrated = f'(calibrated {camera} {rover})' in state
                            loc_after_cal = current_loc_r # Assume calibration starts from current location

                            if not is_calibrated:
                                # Need to calibrate
                                cal_target = self.camera_cal_target.get(camera)
                                if cal_target is None: continue

                                cal_wps = self.objective_visible_from.get(cal_target, set())
                                if not cal_wps: continue

                                min_cal_nav_dist = float('inf')
                                best_cal_wp = None
                                for cal_wp in cal_wps:
                                    dist = self.get_distance(rover, current_loc_r, cal_wp)
                                    if dist < min_cal_nav_dist:
                                        min_cal_nav_dist = dist
                                        best_cal_wp = cal_wp

                                if best_cal_wp == float('inf'): continue

                                current_rover_cost += min_cal_nav_dist # Navigate to cal point
                                current_rover_cost += 1 # Calibrate
                                loc_after_cal = best_cal_wp # Rover is now at cal location
                            else:
                                # Already calibrated, start image navigation from current location
                                loc_after_cal = current_loc_r

                            # Need to take image
                            img_wps = self.objective_visible_from.get(objective_o, set())
                            if not img_wps: continue

                            min_img_nav_dist = float('inf')
                            best_img_wp = None
                            for img_wp in img_wps:
                                dist = self.get_distance(rover, loc_after_cal, img_wp)
                                if dist < min_img_nav_dist:
                                    min_img_nav_dist = dist
                                    best_img_wp = img_wp

                            if best_img_wp == float('inf'): continue

                            current_rover_cost += min_img_nav_dist # Navigate to image point
                            current_rover_cost += 1 # Take image
                            loc_after_image = best_img_wp # Rover is now at image location

                            # Now handle communication
                            min_comm_nav_dist = float('inf')
                            for comm_wp in self.comm_waypoints:
                                dist = self.get_distance(rover, loc_after_image, comm_wp)
                                min_comm_nav_dist = min(min_comm_nav_dist, dist)

                            if min_comm_nav_dist == float('inf'):
                                continue # Cannot reach communication point

                            current_rover_cost += min_comm_nav_dist # Navigate to comm point
                            current_rover_cost += 1 # Communicate

                            min_goal_cost = min(min_goal_cost, current_rover_cost)

                if min_goal_cost == float('inf'):
                    return float('inf') # Goal unreachable
                h += min_goal_cost

        return h
