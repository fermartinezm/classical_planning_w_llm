import collections
import math

class roversHeuristic:
    """
    Domain-dependent heuristic for the Rovers domain.

    Summary:
    The heuristic estimates the cost to reach the goal by summing up the
    estimated costs for each unachieved goal predicate. For each goal,
    it calculates the minimum number of actions (sample/image, calibrate,
    drop, communicate) and the minimum navigation cost required to satisfy
    that specific goal, assuming other goals can be pursued independently
    or concurrently by different rovers. Navigation costs are precomputed
    using BFS on the traverse graph for each rover.

    Assumptions:
    - The PDDL task is well-formed according to the rovers domain.
    - Navigation actions have a cost of 1.
    - Sample, Drop, Calibrate, Take Image, Communicate actions have a cost of 1.
    - The heuristic assumes optimal choices of rovers, waypoints, and cameras
      for achieving each individual goal predicate.
    - It assumes that if a sample/image is needed, it will be obtained by
      an equipped rover, and then communicated by a rover that possesses it
      (either the one that just sampled/imaged or one that already had it).
      The navigation cost for communication is estimated from the location
      where the sample/image is obtained (or the current location if already held)
      to the nearest communication waypoint.
    - Calibration cost is estimated by finding the minimum navigation cost
      from the rover's current location to a calibration waypoint for the camera,
      plus the calibration action, plus navigation from the calibration waypoint
      to the image waypoint, plus the take_image action.

    Heuristic Initialization:
    The constructor pre-processes the static facts from the PDDL task:
    - Identifies all objects of each type (rovers, waypoints, etc.) by inspecting
      the arguments of static predicates.
    - Stores rover capabilities (equipped_for_...).
    - Stores camera information (on_board, supports, calibration_target).
    - Stores store information (store_of).
    - Stores lander location.
    - Builds the navigation graph (can_traverse) for each rover.
    - Precomputes shortest path distances between all pairs of waypoints
      for each rover using BFS.
    - Identifies communication waypoints (visible from lander).
    - Identifies waypoints visible from objectives and calibration targets.

    Step-By-Step Thinking for Computing Heuristic (__call__):
    1. Check if the state is a goal state. If yes, return 0.
    2. Initialize total heuristic value `h = 0`.
    3. Get the current location of each rover from the state.
    4. Iterate through each goal predicate `G` in `task.goals`.
    5. If `G` is already true in the current `state`, skip this goal.
    6. If `G` is `(communicated_soil_data w)`:
        a. Initialize goal cost `cost = 1` (for the communicate action).
        b. Check if any rover currently has `(have_soil_analysis r w)`.
        c. If no rover has the sample:
            i. Add 1 to `cost` (for the sample_soil action).
            ii. Find the minimum navigation cost for any equipped soil rover
               to reach waypoint `w`. Add this cost to `cost`. If unreachable,
               return infinity for the total heuristic.
            iii. Add 1 to `cost` if any equipped soil rover has a full store
                (approximation for drop action).
            iv. Find the minimum navigation cost from waypoint `w` (where sample
                is taken) to any communication waypoint, considering any rover's
                traverse capabilities. Add this cost to `cost`. If unreachable,
                return infinity for the total heuristic.
        d. If some rover has the sample:
            i. Find the minimum navigation cost for any rover that has the sample
               to reach any communication waypoint from its current location.
               Add this cost to `cost`. If unreachable, return infinity for the
               total heuristic.
        e. Add `cost` for this goal to the total heuristic `h`.
    7. If `G` is `(communicated_rock_data w)`:
        a. Follow similar logic as for soil data, using rock-specific predicates
           and equipped rock rovers.
    8. If `G` is `(communicated_image_data o m)`:
        a. Initialize goal cost `cost = 1` (for the communicate action).
        b. Check if any rover currently has `(have_image r o m)`.
        c. If no rover has the image:
            i. Add 1 to `cost` (for the take_image action).
            ii. Find the minimum cost to get the image: Iterate over equipped
                imaging rovers `r`, their cameras `i` supporting mode `m`,
                and waypoints `p` where objective `o` is visible.
                Calculate the cost for rover `r` to take image `(o m)` at `p`
                with camera `i`. This cost includes navigation to `p` and
                potentially calibration.
                - If camera `i` on rover `r` is calibrated: Cost is `dist(at(r), p)`.
                - If camera `i` on rover `r` is not calibrated: Find the minimum
                  navigation cost from `at(r)` to a calibration waypoint `w'`
                  for camera `i`, plus 1 (calibrate action), plus navigation
                  from `w'` to `p`. Minimize this over all suitable `w'`.
                - The minimum cost to get the image is the minimum of the above
                  over all suitable `r`, `i`, `p`. Add this minimum cost to `cost`.
                  If unreachable, return infinity for the total heuristic.
            iii. Find the minimum navigation cost from any waypoint `p` where
                 objective `o` is visible (where the image was taken) to any
                 communication waypoint, considering any rover's traverse
                 capabilities. Add this cost to `cost`. If unreachable,
                 return infinity for the total heuristic.
        d. If some rover has the image:
            i. Find the minimum navigation cost for any rover that has the image
               to reach any communication waypoint from its current location.
               Add this cost to `cost`. If unreachable, return infinity for the
               total heuristic.
        e. Add `cost` for this goal to the total heuristic `h`.
    9. Return the total heuristic value `h`. If any step determined a goal
       was unreachable, the total heuristic is infinity.
    """

    def __init__(self, task):
        self.task = task
        self.static_facts = task.static
        self.goals = task.goals

        # --- Preprocessing Static Facts ---
        # Infer objects from static facts
        self.rovers = set()
        self.waypoints = set()
        self.cameras = set()
        self.objectives = set()
        self.modes = set()
        self.landers = set()
        self.stores = set()

        self.lander_location = None
        self.rover_store_map = {} # rover -> store
        self.equipped_soil = set() # rovers
        self.equipped_rock = set() # rovers
        self.equipped_imaging = set() # rovers
        self.camera_rover_map = {} # camera -> rover
        self.camera_modes_map = collections.defaultdict(set) # camera -> set of modes
        self.calib_target_map = {} # camera -> objective
        self.obj_visibility_wps = collections.defaultdict(set) # objective -> set of waypoints
        self.calib_visibility_wps = collections.defaultdict(set) # camera -> set of waypoints
        self.rover_traverse_graph = collections.defaultdict(lambda: collections.defaultdict(set)) # rover -> from_wp -> set of to_wps

        for fact_str in self.static_facts:
            pred, args = self._parse_fact(fact_str)
            if pred == 'at_lander':
                if len(args) == 2:
                    self.landers.add(args[0])
                    self.waypoints.add(args[1])
                    self.lander_location = args[1]
            elif pred == 'can_traverse':
                if len(args) == 3:
                    self.rovers.add(args[0])
                    self.waypoints.add(args[1])
                    self.waypoints.add(args[2])
                    self.rover_traverse_graph[args[0]][args[1]].add(args[2])
            elif pred == 'equipped_for_soil_analysis':
                if len(args) == 1: self.rovers.add(args[0]); self.equipped_soil.add(args[0])
            elif pred == 'equipped_for_rock_analysis':
                if len(args) == 1: self.rovers.add(args[0]); self.equipped_rock.add(args[0])
            elif pred == 'equipped_for_imaging':
                if len(args) == 1: self.rovers.add(args[0]); self.equipped_imaging.add(args[0])
            elif pred == 'store_of':
                if len(args) == 2:
                    self.stores.add(args[0])
                    self.rovers.add(args[1])
                    self.rover_store_map[args[1]] = args[0]
            elif pred == 'supports':
                if len(args) == 2:
                    self.cameras.add(args[0])
                    self.modes.add(args[1])
                    self.camera_modes_map[args[0]].add(args[1])
            elif pred == 'visible':
                if len(args) == 2:
                    self.waypoints.add(args[0])
                    self.waypoints.add(args[1])
            elif pred == 'visible_from':
                if len(args) == 2:
                    self.waypoints.add(args[1])
                    self.objectives.add(args[0]) # Assume anything visible_from is an objective or target
                    self.obj_visibility_wps[args[0]].add(args[1])
            elif pred == 'calibration_target':
                if len(args) == 2:
                    self.cameras.add(args[0])
                    self.objectives.add(args[1]) # Calibration target is an objective
                    self.calib_target_map[args[0]] = args[1]
            elif pred == 'on_board':
                if len(args) == 2:
                    self.cameras.add(args[0])
                    self.rovers.add(args[1])
                    self.camera_rover_map[args[0]] = args[1]

        # Refine calib_visibility_wps using calib_target_map
        for camera, target in self.calib_target_map.items():
             if target in self.obj_visibility_wps: # Targets are objectives
                 self.calib_visibility_wps[camera] = self.obj_visibility_wps[target]

        # Identify communication waypoints (visible from lander)
        self.communication_wps = set()
        if self.lander_location:
            # Check visible facts where the second argument is the lander location
            for fact_str in self.static_facts:
                pred, args = self._parse_fact(fact_str)
                if pred == 'visible' and len(args) == 2 and args[1] == self.lander_location:
                    self.communication_wps.add(args[0])

        # --- Precompute Navigation Distances ---
        self.dist = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(lambda: float('inf'))))
        # Need to ensure all waypoints are included, even if a rover cannot traverse from/to them initially
        all_waypoints_list = list(self.waypoints) # Use a list for consistent indexing if needed, though dict is fine
        for rover in self.rovers:
            for start_wp in self.waypoints:
                self.dist[rover][start_wp][start_wp] = 0
                queue = collections.deque([start_wp])
                visited = {start_wp}
                while queue:
                    curr_wp = queue.popleft()
                    # Use the rover's specific traverse graph
                    for next_wp in self.rover_traverse_graph[rover].get(curr_wp, set()):
                        if next_wp not in visited:
                            visited.add(next_wp)
                            self.dist[rover][start_wp][next_wp] = self.dist[rover][start_wp][curr_wp] + 1
                            queue.append(next_wp)

    def _parse_fact(self, fact_string):
        """Helper to parse a fact string into predicate and arguments."""
        # Remove surrounding brackets and split by space
        parts = fact_string.strip("()").split()
        if not parts: # Handle empty string after strip
            return None, []
        return parts[0], parts[1:]

    def get_rover_location(self, state, rover):
        """Finds the current waypoint of a rover in the state."""
        for fact_str in state:
            pred, args = self._parse_fact(fact_str)
            if pred == 'at' and len(args) == 2 and args[0] == rover:
                return args[1]
        return None # Should not happen in a valid state representation

    def __call__(self, state):
        """
        Computes the heuristic value for the given state.
        """
        # If goal is reached, heuristic is 0
        if self.task.goal_reached(state):
            return 0

        h = 0
        unreachable = False

        # Get current rover locations
        rover_locations = {r: self.get_rover_location(state, r) for r in self.rovers}

        for goal_str in self.goals:
            if goal_str in state:
                continue # Goal already achieved

            pred, args = self._parse_fact(goal_str)

            if pred == 'communicated_soil_data':
                if len(args) != 1: continue # Malformed goal?
                soil_wp = args[0]
                goal_cost = 1 # communicate action

                # Cost to get the sample (if needed)
                has_sample = any(f'(have_soil_analysis {r} {soil_wp})' in state for r in self.rovers)
                if not has_sample:
                    goal_cost += 1 # sample action
                    # Min cost to get sample: nav to w + drop (if needed)
                    min_sample_prep_cost = float('inf')
                    for r in self.equipped_soil:
                        current_wp = rover_locations.get(r)
                        if current_wp and soil_wp in self.dist[r][current_wp]:
                            nav_cost = self.dist[r][current_wp][soil_wp]
                            if nav_cost != float('inf'):
                                rover_store = self.rover_store_map.get(r)
                                drop_cost = 1 if rover_store and f'(full {rover_store})' in state else 0
                                min_sample_prep_cost = min(min_sample_prep_cost, nav_cost + drop_cost)

                    if min_sample_prep_cost == float('inf'):
                        unreachable = True
                        break # Cannot get sample, goal is unreachable
                    goal_cost += min_sample_prep_cost

                    # Min cost to communicate after sampling: nav from w to comm_wp
                    min_nav_sample_to_comm = float('inf')
                    for r in self.rovers: # Any rover can communicate if it has the sample
                        for comm_wp in self.communication_wps:
                            if soil_wp in self.dist[r] and comm_wp in self.dist[r][soil_wp]:
                                min_nav_sample_to_comm = min(min_nav_sample_to_comm, self.dist[r][soil_wp][comm_wp])

                    if min_nav_sample_to_comm == float('inf'):
                        unreachable = True
                        break # Cannot reach comm from sample location
                    goal_cost += min_nav_sample_to_comm
                else: # Sample is already held by some rover
                    # Min cost to communicate: nav from rover location to comm_wp
                    min_nav_have_to_comm = float('inf')
                    for r in self.rovers:
                        if f'(have_soil_analysis {r} {soil_wp})' in state:
                            current_wp = rover_locations.get(r)
                            if current_wp:
                                for comm_wp in self.communication_wps:
                                    if current_wp in self.dist[r] and comm_wp in self.dist[r][current_wp]:
                                        min_nav_have_to_comm = min(min_nav_have_to_comm, self.dist[r][current_wp][comm_wp])
                    if min_nav_have_to_comm == float('inf'):
                        unreachable = True
                        break # Cannot reach comm from rover location
                    goal_cost += min_nav_have_to_comm

                h += goal_cost

            elif pred == 'communicated_rock_data':
                if len(args) != 1: continue # Malformed goal?
                rock_wp = args[0]
                goal_cost = 1 # communicate action

                # Cost to get the sample (if needed)
                has_sample = any(f'(have_rock_analysis {r} {rock_wp})' in state for r in self.rovers)
                if not has_sample:
                    goal_cost += 1 # sample action
                    min_sample_prep_cost = float('inf')
                    for r in self.equipped_rock:
                        current_wp = rover_locations.get(r)
                        if current_wp and rock_wp in self.dist[r][current_wp]:
                            nav_cost = self.dist[r][current_wp][rock_wp]
                            if nav_cost != float('inf'):
                                rover_store = self.rover_store_map.get(r)
                                drop_cost = 1 if rover_store and f'(full {rover_store})' in state else 0
                                min_sample_prep_cost = min(min_sample_prep_cost, nav_cost + drop_cost)

                    if min_sample_prep_cost == float('inf'):
                        unreachable = True
                        break # Cannot get sample
                    goal_cost += min_sample_prep_cost

                    min_nav_sample_to_comm = float('inf')
                    for r in self.rovers: # Any rover can communicate
                        for comm_wp in self.communication_wps:
                            if rock_wp in self.dist[r] and comm_wp in self.dist[r][rock_wp]:
                                min_nav_sample_to_comm = min(min_nav_sample_to_comm, self.dist[r][rock_wp][comm_wp])

                    if min_nav_sample_to_comm == float('inf'):
                        unreachable = True
                        break # Cannot reach comm from sample location
                    goal_cost += min_nav_sample_to_comm
                else: # Sample is already held
                    min_nav_have_to_comm = float('inf')
                    for r in self.rovers:
                        if f'(have_rock_analysis {r} {rock_wp})' in state:
                            current_wp = rover_locations.get(r)
                            if current_wp:
                                for comm_wp in self.communication_wps:
                                    if current_wp in self.dist[r] and comm_wp in self.dist[r][current_wp]:
                                        min_nav_have_to_comm = min(min_nav_have_to_comm, self.dist[r][current_wp][comm_wp])
                    if min_nav_have_to_comm == float('inf'):
                        unreachable = True
                        break # Cannot reach comm from rover location
                    goal_cost += min_nav_have_to_comm

                h += goal_cost

            elif pred == 'communicated_image_data':
                if len(args) != 2: continue # Malformed goal?
                objective = args[0]
                mode = args[1]
                goal_cost = 1 # communicate action

                # Cost to get the image (if needed)
                has_image = any(f'(have_image {r} {objective} {mode})' in state for r in self.rovers)
                if not has_image:
                    goal_cost += 1 # take_image action
                    # Min cost to get image: nav to image_wp + calibrate (if needed) + take_image
                    min_image_prep_cost = float('inf')
                    for r in self.equipped_imaging:
                        current_wp = rover_locations.get(r)
                        if current_wp:
                            for i in self.cameras:
                                # Check if camera i is on rover r and supports mode m
                                if self.camera_rover_map.get(i) == r and mode in self.camera_modes_map.get(i, set()):
                                    # Find best image waypoint p for objective o
                                    for p in self.obj_visibility_wps.get(objective, set()):
                                        if current_wp in self.dist[r] and p in self.dist[r][current_wp]:
                                            nav_to_image_wp_cost = self.dist[r][current_wp][p]
                                            if nav_to_image_wp_cost != float('inf'):
                                                calib_cost = 0
                                                if f'(calibrated {i} {r})' not in state:
                                                    # Need to calibrate camera i on rover r
                                                    calib_target = self.calib_target_map.get(i)
                                                    if calib_target:
                                                        # Find best calibration waypoint w' for camera i
                                                        min_nav_current_to_calib_to_p = float('inf')
                                                        for w_prime in self.calib_visibility_wps.get(i, set()):
                                                            if current_wp in self.dist[r] and w_prime in self.dist[r][current_wp] and \
                                                               w_prime in self.dist[r] and p in self.dist[r][w_prime]:
                                                                nav1 = self.dist[r][current_wp][w_prime]
                                                                nav2 = self.dist[r][w_prime][p]
                                                                if nav1 != float('inf') and nav2 != float('inf'):
                                                                    min_nav_current_to_calib_to_p = min(min_nav_current_to_calib_to_p, nav1 + nav2)

                                                        if min_nav_current_to_calib_to_p != float('inf'):
                                                            calib_cost = min_nav_current_to_calib_to_p + 1 # nav + calibrate + nav
                                                        else:
                                                            calib_cost = float('inf') # Cannot reach calib_wp or image_wp via calib_wp
                                                    else:
                                                        calib_cost = float('inf') # No calib target for camera i?

                                                if calib_cost != float('inf'):
                                                    # Cost to get image = (nav current to p) + (calib cost if needed) + 1 (take_image)
                                                    # If calib_cost is 0, it's just dist(current_wp, p) + 1.
                                                    # If calib_cost > 0, it's the full path current -> calib_wp -> p + 1.
                                                    current_image_cost = nav_to_image_wp_cost + 1 # nav + take_image
                                                    if f'(calibrated {i} {r})' not in state:
                                                         current_image_cost = calib_cost + 1 # nav + calibrate + nav + take_image

                                                    min_image_prep_cost = min(min_image_prep_cost, current_image_cost)

                    if min_image_prep_cost == float('inf'):
                        unreachable = True
                        break # Cannot get image
                    goal_cost += min_image_prep_cost

                    # Min cost to communicate after taking image: nav from image_wp p to comm_wp
                    min_nav_image_to_comm = float('inf')
                    # We need the minimum distance from *any* image waypoint for objective to *any* comm_wp, over *any* rover.
                    # This is an approximation. Ideally, we'd use the specific image_wp 'p' chosen above for min_image_prep_cost.
                    # But finding that 'p' adds complexity. Let's use the min over all possible image wps for the objective.
                    for p in self.obj_visibility_wps.get(objective, set()):
                         for r in self.rovers:
                            for comm_wp in self.communication_wps:
                                if p in self.dist[r] and comm_wp in self.dist[r][p]:
                                    min_nav_image_to_comm = min(min_nav_image_to_comm, self.dist[r][p][comm_wp])

                    if min_nav_image_to_comm == float('inf'):
                        unreachable = True
                        break # Cannot reach comm waypoint
                    goal_cost += min_nav_image_to_comm

                else: # Image is already held by some rover
                    # Min cost to communicate: nav from rover location to comm_wp
                    min_nav_have_to_comm = float('inf')
                    for r in self.rovers:
                        if f'(have_image {r} {objective} {mode})' in state:
                            current_wp = rover_locations.get(r)
                            if current_wp:
                                for comm_wp in self.communication_wps:
                                    if current_wp in self.dist[r] and comm_wp in self.dist[r][current_wp]:
                                        min_nav_have_to_comm = min(min_nav_have_to_comm, self.dist[r][current_wp][comm_wp])
                    if min_nav_have_to_comm == float('inf'):
                        unreachable = True
                        break # Cannot reach comm from rover location
                    goal_cost += min_nav_have_to_comm

                h += goal_cost

            # Add other goal types if necessary (although the problem description only shows communicated_...)
            # The provided examples only have communicated_ goals.

        if unreachable:
            return float('inf')

        return h
