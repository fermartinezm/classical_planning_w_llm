import heapq
from collections import defaultdict
from fnmatch import fnmatch
import math # Use math.inf

# Try to import the base class, provide a dummy if not found
try:
    # Assumes the heuristic base class is available in the specified path
    from heuristics.heuristic_base import Heuristic
except ImportError:
    # Define a dummy base class if the import fails (e.g., for standalone testing)
    class Heuristic:
        """Dummy base class for Heuristic if import fails."""
        def __init__(self, task):
            """Initializes the heuristic with the planning task."""
            pass
        def __call__(self, node):
            """Calculates the heuristic value for a given node."""
            raise NotImplementedError

def get_parts(fact):
    """
    Extracts predicate and arguments from a PDDL fact string.
    Example: "(pred arg1 arg2)" -> ["pred", "arg1", "arg2"]
    Handles potential surrounding whitespace.
    """
    cleaned_fact = fact.strip()
    if not cleaned_fact or cleaned_fact[0] != '(' or cleaned_fact[-1] != ')':
        # Return empty list for malformed facts or non-facts
        return []
    return cleaned_fact[1:-1].split()

class RoversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the PDDL Rovers domain.

    # Summary
    This heuristic estimates the remaining cost to achieve all goals by summing
    the estimated costs for each individual unsatisfied goal. It considers the
    necessary sequence of actions (navigation, sampling/imaging, communication)
    and leverages precomputed shortest path distances between waypoints for each rover.
    The heuristic aims to be informative for Greedy Best-First Search by providing
    a reasonable estimate of actions needed, without needing to be admissible.

    # Assumptions
    - The map (waypoints, visibility, traversability) is static.
    - Each required action (navigate, sample_soil, sample_rock, calibrate,
      take_image, communicate_*, drop) has a cost of 1.
    - Rovers must be at the correct location and satisfy equipment/state
      preconditions for actions.
    - Communication requires the rover to be at a waypoint visible from the
      lander's location.
    - Taking an image requires calibration, which might require navigating to a
      specific location first. Taking an image uncalibrates the camera.
    - Sampling requires the rover's store to be empty; if full, a 'drop' action
      (cost 1) is assumed to be needed before sampling. Samples are consumed
      upon collection in the PDDL, but the heuristic assumes they are available
      if needed for a goal (i.e., it doesn't track `at_soil_sample` state).

    # Heuristic Initialization
    - Parses static facts provided in `task.static` to build internal knowledge:
        - Identifies all objects (rovers, waypoints, cameras, etc.).
        - Rover capabilities (soil analysis, rock analysis, imaging).
        - Camera properties (which rover it's on, supported modes, calibration targets).
        - Waypoint graph structure (visibility between waypoints, traversability
          links per rover).
        - Location of the lander (`at_lander`).
        - Locations from which objectives are visible (`visible_from`).
        - Waypoints from which communication with the lander is possible.
    - Precomputes all-pairs shortest paths (APSP) for each rover using BFS on the
      traversable waypoint graph. The distance represents the minimum number of
      'navigate' actions required between any two waypoints for that specific rover.
      Stores distances in `self.shortest_paths[rover][from_wp][to_wp]`.

    # Step-By-Step Thinking for Computing Heuristic
    1. Check if the current state (`node.state`) satisfies all goals (`self.task.goals`).
       If yes, the goal is reached, return 0.
    2. Parse the current state to extract dynamic information efficiently:
        - Current location of each rover (`at`).
        - Whether each rover's store is full (`full`).
        - Soil/rock samples currently held by each rover (`have_soil_analysis`, `have_rock_analysis`).
        - Images currently held by each rover (`have_image`).
        - Calibration status of each camera (`calibrated`).
    3. Initialize `total_cost = 0`. This will accumulate the estimated cost.
    4. Iterate through each goal predicate `g` defined in `self.task.goals`:
        a. If `g` is already present in the current `state`, this goal is satisfied,
           so continue to the next goal.
        b. If `g` is not satisfied, estimate the minimum cost `cost(g)` required
           to achieve it from the current state, ignoring potential conflicts or
           interactions with achieving other goals:
            i. **For `communicated_soil_data(w)`:**
               - Calculate the minimum cost by considering two main scenarios:
                 1. *Sample already held*: If a rover `r` currently has the soil
                    sample (`have_soil_analysis r w`), the cost is the minimum
                    navigation steps for `r` from its current location `w_r` to
                    any communication waypoint `x`, plus 1 action for `communicate_soil_data`.
                 2. *Sample needs collection*: If no rover has the sample, find
                    the minimum cost across all rovers `r` equipped for soil analysis.
                    The estimated sequence cost is: (1 action for `drop` if store is full)
                    + nav(`w_r` to `w`) + 1 action for `sample_soil` + nav(`w` to `x`)
                    + 1 action for `communicate_soil_data`.
               - Take the minimum cost found across these scenarios and rovers.
            ii. **For `communicated_rock_data(w)`:**
                - Analogous to soil data, but using rock analysis capabilities,
                  `have_rock_analysis`, `sample_rock`, and `communicate_rock_data`.
            iii. **For `communicated_image_data(o, m)`:**
                - Calculate the minimum cost considering two main scenarios:
                  1. *Image already held*: If a rover `r` currently has the image
                     (`have_image r o m`), the cost is the minimum navigation steps
                     for `r` from `w_r` to any communication waypoint `x`, plus 1
                     action for `communicate_image_data`.
                  2. *Image needs to be taken*: Find the minimum cost across all
                     rovers `r` equipped for imaging with a suitable camera `c`
                     (on board `r`, supports mode `m`).
                     - *If camera `c` is already calibrated*: Estimated sequence cost is
                       nav(`w_r` to `p`) + 1 action for `take_image` + nav(`p` to `x`)
                       + 1 action for `communicate_image_data`. Minimize over possible
                       image waypoints `p` (where `o` is visible) and comm waypoints `x`.
                     - *If camera `c` is not calibrated*: Estimated sequence cost is
                       nav(`w_r` to `w_calib`) + 1 action for `calibrate`
                       + nav(`w_calib` to `p`) + 1 action for `take_image`
                       + nav(`p` to `x`) + 1 action for `communicate_image_data`.
                       Minimize over calibration waypoints `w_calib`, image waypoints `p`,
                       and comm waypoints `x`.
               - Take the minimum cost found across these scenarios, rovers, cameras,
                 and waypoints.
        c. If the calculated `cost(g)` for the unsatisfied goal `g` is finite
           (i.e., not `math.inf`), add it to `total_cost`.
        d. If `cost(g)` is `math.inf`, it implies this heuristic estimates the goal
           as unreachable from the current state (e.g., due to lack of paths or
           equipment). In this case, the state is considered a dead-end by the
           heuristic, so return `math.inf` immediately.
    5. After checking all goals, return the accumulated `total_cost`.
    """
    def __init__(self, task):
        super().__init__(task)
        self.task = task
        self.goals = task.goals
        self.static = task.static

        # Precompute data structures from static facts
        self._parse_static_facts()
        # Precompute shortest paths for navigation
        self._precompute_shortest_paths()
        # Cache for navigation costs to sets of waypoints, reset per state evaluation
        self._min_nav_cache = {}


    def _parse_static_facts(self):
        """Extracts static information into convenient data structures."""
        # Initialize sets and dictionaries to store parsed info
        self.rovers = set()
        self.waypoints = set()
        self.cameras = set()
        self.objectives = set()
        self.modes = set()
        self.stores = set()
        self.lander = None
        self.lander_location = None

        self.rover_equipment_soil = set()
        self.rover_equipment_rock = set()
        self.rover_equipment_imaging = set()
        self.rover_stores = {} # rover -> store
        self.store_rovers = {} # store -> rover
        self.camera_on_board = {} # camera -> rover
        self.rover_cameras = defaultdict(set) # rover -> {cameras}
        self.camera_supports = defaultdict(set) # camera -> {modes}
        self.camera_calibration_target = {} # camera -> objective
        self.objective_calibration_camera = defaultdict(set) # objective -> {cameras}

        self.visibility = defaultdict(set) # wp -> {visible wps}
        self.can_traverse = defaultdict(lambda: defaultdict(set)) # rover -> from_wp -> {to_wp}
        self.objective_visible_from = defaultdict(set) # objective -> {wps}
        self.waypoint_visible_objectives = defaultdict(set) # wp -> {objectives}

        # Populate sets and dictionaries by parsing static facts
        for fact in self.static:
            try:
                parts = get_parts(fact)
                if not parts: continue # Skip empty or invalid facts
                pred = parts[0]
                args = parts[1:]

                # Infer object types (useful for iterating later)
                # Note: Types are often declared in :types, but parsing predicates helps too
                if pred == 'rover': self.rovers.add(args[0])
                elif pred == 'waypoint': self.waypoints.add(args[0])
                elif pred == 'camera': self.cameras.add(args[0])
                elif pred == 'objective': self.objectives.add(args[0])
                elif pred == 'mode': self.modes.add(args[0])
                elif pred == 'store': self.stores.add(args[0])
                elif pred == 'lander': self.lander = args[0]

                # Parse relationships between objects based on predicates
                if pred == 'at_lander' and len(args) == 2:
                    self.lander = args[0]
                    self.lander_location = args[1]
                    self.waypoints.add(args[1]) # Ensure lander location is known waypoint
                elif pred == 'equipped_for_soil_analysis' and len(args) == 1: self.rover_equipment_soil.add(args[0])
                elif pred == 'equipped_for_rock_analysis' and len(args) == 1: self.rover_equipment_rock.add(args[0])
                elif pred == 'equipped_for_imaging' and len(args) == 1: self.rover_equipment_imaging.add(args[0])
                elif pred == 'store_of' and len(args) == 2:
                    self.stores.add(args[0])
                    self.rovers.add(args[1])
                    self.store_rovers[args[0]] = args[1]
                    self.rover_stores[args[1]] = args[0]
                elif pred == 'on_board' and len(args) == 2:
                    self.cameras.add(args[0])
                    self.rovers.add(args[1])
                    self.camera_on_board[args[0]] = args[1]
                    self.rover_cameras[args[1]].add(args[0])
                elif pred == 'supports' and len(args) == 2:
                    self.cameras.add(args[0])
                    self.modes.add(args[1])
                    self.camera_supports[args[0]].add(args[1])
                elif pred == 'calibration_target' and len(args) == 2:
                    self.cameras.add(args[0])
                    self.objectives.add(args[1])
                    self.camera_calibration_target[args[0]] = args[1]
                    self.objective_calibration_camera[args[1]].add(args[0])
                elif pred == 'visible' and len(args) == 2:
                    self.waypoints.add(args[0])
                    self.waypoints.add(args[1])
                    self.visibility[args[0]].add(args[1])
                elif pred == 'can_traverse' and len(args) == 3:
                    self.rovers.add(args[0])
                    self.waypoints.add(args[1])
                    self.waypoints.add(args[2])
                    self.can_traverse[args[0]][args[1]].add(args[2])
                elif pred == 'visible_from' and len(args) == 2:
                    self.objectives.add(args[0])
                    self.waypoints.add(args[1])
                    self.objective_visible_from[args[0]].add(args[1])
                    self.waypoint_visible_objectives[args[1]].add(args[0])
            except IndexError:
                 # Silently skip facts that don't match expected structure
                 # print(f"Warning: Skipping potentially malformed static fact: {fact}")
                 pass


        # Identify waypoints suitable for communication with the lander
        self.comm_waypoints = set()
        if self.lander_location:
            # Communication action requires rover at x, lander at y, visible(x, y)
            for wp_from in self.waypoints:
                 if self.lander_location in self.visibility.get(wp_from, set()):
                     self.comm_waypoints.add(wp_from)


    def _precompute_shortest_paths(self):
        """
        Computes All-Pairs Shortest Paths (APSP) using Breadth-First Search (BFS)
        for each rover. The distance is the number of 'navigate' actions.
        Handles the requirement that navigation needs both `can_traverse` and `visible`.
        Stores results in self.shortest_paths[rover][from_wp][to_wp].
        """
        self.shortest_paths = defaultdict(lambda: defaultdict(lambda: math.inf))

        for r in self.rovers:
            rover_paths = self.shortest_paths[r] # Reference to the inner dict for this rover
            for start_node in self.waypoints:
                # Initialize distances from start_node for this rover
                current_rover_start_paths = rover_paths[start_node]
                current_rover_start_paths[start_node] = 0

                queue = [(0, start_node)] # Use a list as a queue for BFS: (cost, node)
                visited = {start_node: 0} # Keep track of visited nodes and their costs

                head = 0
                while head < len(queue):
                    cost, curr_wp = queue[head]
                    head += 1

                    # Find neighbors based on rover's ability to traverse from curr_wp
                    possible_next_wps = self.can_traverse.get(r, {}).get(curr_wp, set())

                    for next_wp in possible_next_wps:
                        # The 'navigate' action also requires visibility from curr_wp to next_wp
                        if next_wp in self.visibility.get(curr_wp, set()):
                            new_cost = cost + 1
                            # If we found a shorter path to next_wp, update and add to queue
                            if new_cost < visited.get(next_wp, math.inf):
                                visited[next_wp] = new_cost
                                current_rover_start_paths[next_wp] = new_cost
                                queue.append((new_cost, next_wp))


    def _parse_state(self, state):
        """Parses the dynamic state facts into usable dictionaries for quick lookup."""
        # Initialize dictionaries to hold current state info
        rover_locations = {}
        store_full = set() # Stores that are full
        have_soil = defaultdict(set) # rover -> {waypoints where soil sample is held}
        have_rock = defaultdict(set) # rover -> {waypoints where rock sample is held}
        have_image = defaultdict(lambda: defaultdict(set)) # rover -> objective -> {modes of image held}
        calibrated_cameras = defaultdict(set) # rover -> {cameras that are calibrated}

        # Parse each fact in the current state
        for fact in state:
            try:
                parts = get_parts(fact)
                if not parts: continue
                pred = parts[0]
                args = parts[1:]

                # Update state dictionaries based on predicate
                if pred == 'at' and len(args) == 2 and args[0] in self.rovers:
                    rover_locations[args[0]] = args[1]
                elif pred == 'full' and len(args) == 1:
                    store_full.add(args[0])
                elif pred == 'have_soil_analysis' and len(args) == 2:
                    have_soil[args[0]].add(args[1])
                elif pred == 'have_rock_analysis' and len(args) == 2:
                    have_rock[args[0]].add(args[1])
                elif pred == 'have_image' and len(args) == 3:
                    have_image[args[0]][args[1]].add(args[2])
                elif pred == 'calibrated' and len(args) == 2:
                    # args[0] is camera, args[1] is rover
                    # Verify the camera is indeed on this rover (consistency check)
                    cam_rover = self.camera_on_board.get(args[0])
                    if cam_rover == args[1]:
                         calibrated_cameras[args[1]].add(args[0])
            except IndexError:
                 # Silently skip malformed facts
                 # print(f"Warning: Skipping potentially malformed dynamic fact: {fact}")
                 pass

        # Return a dictionary containing all parsed dynamic information
        return {
            'rover_locations': rover_locations,
            'store_full': store_full,
            'have_soil': have_soil,
            'have_rock': have_rock,
            'have_image': have_image,
            'calibrated_cameras': calibrated_cameras
        }

    def _get_min_nav_cost(self, rover, from_wp, target_wps):
        """
        Calculates the minimum navigation cost (number of 'navigate' actions)
        for a specific rover to travel from 'from_wp' to any waypoint in the
        set 'target_wps'. Uses precomputed shortest paths and caches results.
        Returns math.inf if no target waypoint is reachable.
        """
        # Use frozenset for cache key as sets are not hashable
        target_wps_fs = frozenset(target_wps)
        if not target_wps_fs: # No targets means unreachable
            return math.inf

        # Check cache first
        cache_key = (rover, from_wp, target_wps_fs)
        if cache_key in self._min_nav_cache:
            return self._min_nav_cache[cache_key]

        min_cost = math.inf
        # Access precomputed shortest paths for the rover starting from from_wp
        # Ensure rover and from_wp exist in the precomputed data structure
        if rover in self.shortest_paths and from_wp in self.shortest_paths[rover]:
            paths_from_wp = self.shortest_paths[rover][from_wp]
            # Find the minimum cost to any of the target waypoints
            for target_wp in target_wps_fs:
                 cost = paths_from_wp.get(target_wp, math.inf) # Default to infinity if path doesn't exist
                 min_cost = min(min_cost, cost)

        # Store result in cache and return
        self._min_nav_cache[cache_key] = min_cost
        return min_cost


    def __call__(self, node):
        """Computes the heuristic value for the given state node."""
        state = node.state
        # If the goal is already reached in the current state, heuristic value is 0
        if self.task.goal_reached(state):
            return 0

        # Clear the navigation cost cache for this new state evaluation
        self._min_nav_cache.clear()
        # Parse the dynamic facts of the current state
        parsed_state = self._parse_state(state)
        # Initialize total estimated cost
        total_cost = 0

        # Iterate through all goal conditions defined in the task
        for goal in self.goals:
            # Skip goals that are already satisfied in the current state
            if goal in state:
                continue

            # Parse the goal predicate and arguments
            try:
                parts = get_parts(goal)
                if not parts: continue # Skip if goal is malformed
                pred = parts[0]
                args = parts[1:]
            except IndexError:
                # print(f"Warning: Skipping potentially malformed goal: {goal}")
                continue

            # Estimate the cost to achieve this specific unsatisfied goal
            goal_cost = math.inf # Initialize cost for this goal to infinity

            # --- Estimate cost for `communicated_soil_data(w)` ---
            if pred == 'communicated_soil_data' and len(args) == 1:
                w_goal = args[0]
                current_min_cost = math.inf # Min cost found so far for this goal

                # Scenario 1: A rover already has the required soil sample
                for r in self.rovers:
                    # Check if rover 'r' has the soil sample from 'w_goal'
                    if w_goal in parsed_state['have_soil'].get(r, set()):
                        w_r = parsed_state['rover_locations'].get(r)
                        # Rover must have a location to navigate from
                        if w_r:
                            # Calculate cost: navigate to comms range + communicate action
                            nav_cost = self._get_min_nav_cost(r, w_r, self.comm_waypoints)
                            if nav_cost != math.inf:
                                current_min_cost = min(current_min_cost, nav_cost + 1)

                # Scenario 2: Sample needs to be collected and then communicated
                for r in self.rover_equipment_soil: # Iterate through capable rovers
                    w_r = parsed_state['rover_locations'].get(r)
                    store = self.rover_stores.get(r)
                    # Rover needs a location and an associated store
                    if w_r and store:
                        is_full = store in parsed_state['store_full']
                        drop_cost = 1 if is_full else 0 # Cost for 'drop' if store is full

                        # Cost to navigate from current location to sample location
                        nav_cost1 = self.shortest_paths[r].get(w_r, {}).get(w_goal, math.inf)
                        if nav_cost1 != math.inf:
                            # Cost to navigate from sample location to comms range
                            nav_cost2 = self._get_min_nav_cost(r, w_goal, self.comm_waypoints)
                            if nav_cost2 != math.inf:
                                # Total cost: drop? + nav1 + sample + nav2 + communicate
                                cost = drop_cost + nav_cost1 + 1 + nav_cost2 + 1
                                current_min_cost = min(current_min_cost, cost)

                goal_cost = min(goal_cost, current_min_cost)


            # --- Estimate cost for `communicated_rock_data(w)` ---
            elif pred == 'communicated_rock_data' and len(args) == 1:
                w_goal = args[0]
                current_min_cost = math.inf # Min cost found so far for this goal

                # Scenario 1: A rover already has the required rock sample
                for r in self.rovers:
                    if w_goal in parsed_state['have_rock'].get(r, set()):
                        w_r = parsed_state['rover_locations'].get(r)
                        if w_r:
                            nav_cost = self._get_min_nav_cost(r, w_r, self.comm_waypoints)
                            if nav_cost != math.inf:
                                current_min_cost = min(current_min_cost, nav_cost + 1)

                # Scenario 2: Sample needs to be collected and then communicated
                for r in self.rover_equipment_rock: # Iterate through capable rovers
                    w_r = parsed_state['rover_locations'].get(r)
                    store = self.rover_stores.get(r)
                    if w_r and store:
                        is_full = store in parsed_state['store_full']
                        drop_cost = 1 if is_full else 0

                        nav_cost1 = self.shortest_paths[r].get(w_r, {}).get(w_goal, math.inf)
                        if nav_cost1 != math.inf:
                            nav_cost2 = self._get_min_nav_cost(r, w_goal, self.comm_waypoints)
                            if nav_cost2 != math.inf:
                                # Total cost: drop? + nav1 + sample + nav2 + communicate
                                cost = drop_cost + nav_cost1 + 1 + nav_cost2 + 1
                                current_min_cost = min(current_min_cost, cost)

                goal_cost = min(goal_cost, current_min_cost)


            # --- Estimate cost for `communicated_image_data(o, m)` ---
            elif pred == 'communicated_image_data' and len(args) == 2:
                o_goal, m_goal = args[0], args[1]
                current_min_cost = math.inf # Min cost found so far for this goal

                # Scenario 1: A rover already has the required image
                for r in self.rovers:
                    # Check if rover 'r' has the image of 'o_goal' in mode 'm_goal'
                    if m_goal in parsed_state['have_image'].get(r, {}).get(o_goal, set()):
                        w_r = parsed_state['rover_locations'].get(r)
                        if w_r:
                            # Cost: navigate to comms range + communicate action
                            nav_cost = self._get_min_nav_cost(r, w_r, self.comm_waypoints)
                            if nav_cost != math.inf:
                                current_min_cost = min(current_min_cost, nav_cost + 1)

                # Scenario 2: Image needs to be taken and then communicated
                # Find waypoints from which the objective 'o_goal' is visible
                W_visible_o = self.objective_visible_from.get(o_goal, set())
                # If objective is never visible, this goal cannot be achieved
                if not W_visible_o: continue

                # Iterate through rovers equipped for imaging
                for r in self.rover_equipment_imaging:
                    w_r = parsed_state['rover_locations'].get(r)
                    if not w_r: continue # Rover needs a location

                    # Iterate through cameras on this rover
                    for c in self.rover_cameras.get(r, set()):
                        # Check if the camera supports the required mode
                        if m_goal not in self.camera_supports.get(c, set()): continue

                        # Check if the camera is currently calibrated
                        is_calibrated = c in parsed_state['calibrated_cameras'].get(r, set())

                        if is_calibrated:
                            # Path: w_r -> p (take_image) -> x (communicate)
                            min_path_cost = math.inf
                            # Find the best waypoint 'p' to take the image from
                            for p in W_visible_o:
                                nav1 = self.shortest_paths[r].get(w_r, {}).get(p, math.inf)
                                if nav1 == math.inf: continue
                                # Find the cost to navigate from 'p' to comms range 'x'
                                nav2 = self._get_min_nav_cost(r, p, self.comm_waypoints)
                                if nav2 == math.inf: continue
                                # Update minimum combined navigation cost
                                min_path_cost = min(min_path_cost, nav1 + nav2)

                            if min_path_cost != math.inf:
                                # Total cost: nav1 + take_image + nav2 + communicate
                                cost = min_path_cost + 1 + 1
                                current_min_cost = min(current_min_cost, cost)
                        else:
                            # Camera needs calibration first
                            calib_target = self.camera_calibration_target.get(c)
                            # Camera must have a calibration target defined
                            if not calib_target: continue
                            # Find waypoints 'w_c' from which the calibration target is visible
                            W_calib = self.objective_visible_from.get(calib_target, set())
                            # If calibration target is never visible, cannot calibrate
                            if not W_calib: continue

                            # Path: w_r -> w_c (calibrate) -> p (take_image) -> x (communicate)
                            min_path_cost = math.inf
                            # Find the best sequence of waypoints w_c, p, x
                            for w_c in W_calib: # Waypoint for calibration
                                nav1 = self.shortest_paths[r].get(w_r, {}).get(w_c, math.inf)
                                if nav1 == math.inf: continue
                                for p in W_visible_o: # Waypoint for taking image
                                    nav2 = self.shortest_paths[r].get(w_c, {}).get(p, math.inf)
                                    if nav2 == math.inf: continue
                                    # Cost to navigate from 'p' to comms range 'x'
                                    nav3 = self._get_min_nav_cost(r, p, self.comm_waypoints)
                                    if nav3 == math.inf: continue
                                    # Update minimum combined navigation cost
                                    min_path_cost = min(min_path_cost, nav1 + nav2 + nav3)

                            if min_path_cost != math.inf:
                                # Total cost: nav1 + calibrate + nav2 + take_image + nav3 + communicate
                                cost = min_path_cost + 1 + 1 + 1
                                current_min_cost = min(current_min_cost, cost)

                goal_cost = min(goal_cost, current_min_cost)

            # --- Accumulate cost for this goal ---
            if goal_cost == math.inf:
                # If any single goal is estimated as unreachable by this heuristic,
                # consider the entire state potentially unsolvable (or requiring
                # complex interactions not captured). Return infinity.
                return math.inf
            else:
                # Add the estimated cost for this goal to the total
                total_cost += goal_cost

        # Return the total estimated cost for all unsatisfied goals
        return total_cost
