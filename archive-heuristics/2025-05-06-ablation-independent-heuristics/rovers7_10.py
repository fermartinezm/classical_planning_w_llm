import collections
import math
# Assume Heuristic base class is available in heuristics.heuristic_base
# and Task class is available in task.
# If running standalone for testing, you might need dummy implementations.
try:
    from heuristics.heuristic_base import Heuristic
    from task import Task, Operator
except ImportError:
    # Dummy classes for development/testing outside the planner framework
    print("Warning: Using dummy Heuristic/Task classes. Run within planner for full functionality.")
    class Heuristic:
        def __init__(self, task):
            pass
        def __call__(self, node):
            # node is assumed to have a .state attribute (frozenset)
            raise NotImplementedError

    class Task:
        def __init__(self, name, facts, initial_state, goals, operators, static):
            self.name = name
            self.facts = facts # All possible facts
            self.initial_state = initial_state # frozenset
            self.goals = goals # frozenset
            self.operators = operators # list of Operator
            self.static = static # frozenset

        def goal_reached(self, state):
             return self.goals <= state

    class Operator:
         def __init__(self, name, preconditions, add_effects, del_effects):
             self.name = name
             self.preconditions = frozenset(preconditions)
             self.add_effects = frozenset(add_effects)
             self.del_effects = frozenset(del_effects)
         def applicable(self, state): return self.preconditions <= state
         def apply(self, state): return (state - self.del_effects) | self.add_effects


class roversHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Rovers domain.

    Summary:
        This heuristic estimates the cost to reach a goal state by summing the
        estimated costs for each individual unachieved goal fact. The cost for
        each goal fact is the minimum estimated number of actions required for
        any suitable rover to achieve that specific goal fact, including
        navigation, sampling/imaging, and communication steps. It aims to be
        efficiently computable and guide a greedy best-first search.

    Assumptions:
        - Navigation actions have a cost of 1.
        - Sampling (soil/rock) actions have a cost of 1.
        - Drop actions (to empty a store) have a cost of 1 and are only needed
          if a rover's store is full when attempting to sample.
        - Calibrate actions have a cost of 1.
        - Take-image actions have a cost of 1.
        - Communicate actions have a cost of 1.
        - A rover needs to be calibrated *before* each take-image action.
        - A rover needs an empty store *before* each sample action.
        - Samples (soil/rock) are static unless removed by a sample action. If
          a sample is not present in the current state and no rover has the
          corresponding data, the goal requiring that data is considered
          unachievable from this state by sampling.
        - Communication requires the rover to be at a waypoint visible from the
          lander's static location.
        - The heuristic assumes a single rover performs all steps (sampling/imaging
          and subsequent communication) for a specific goal fact.
        - Navigation costs between waypoints for each rover are precomputed
          using BFS on the `can_traverse` graph.

    Heuristic Initialization:
        In the constructor (`__init__`), the heuristic parses the static facts
        from the task description to build internal data structures:
        - Identifies the lander's location.
        - Categorizes rovers by their equipment (soil, rock, imaging).
        - Maps stores to their respective rovers.
        - Builds navigation graphs (adjacency lists) for each rover based on
          `can_traverse` facts.
        - Builds a visibility graph based on `visible` facts.
        - Maps cameras to rovers, supported modes, and calibration targets.
        - Maps objectives to waypoints from which they are visible.
        - Collects all unique waypoint names mentioned in static or initial facts.
        - Precomputes all-pairs shortest path distances for each rover on its
          navigation graph using BFS, considering all known waypoints.
        - Identifies all waypoints from which the lander is visible (communication points).
        - Stores the initial locations of soil and rock samples from the initial state.

    Step-By-Step Thinking for Computing Heuristic:
        The heuristic function (`__call__`) takes the current state (a node)
        and computes an estimated cost:
        1.  Check if the goal state is already reached (`self.task.goals <= state`). If yes, return 0.
        2.  Initialize the total heuristic value `total_h` to 0.
        3.  Iterate through each goal fact defined in the task (`self.task.goals`).
        4.  For each goal fact (`goal_fact_string`):
            a.  If the goal fact is already true in the current state, skip it.
            b.  If the goal fact is not true, calculate the estimated minimum
                cost `cost_g` to achieve this specific goal fact from the
                current state using helper methods (`_cost_to_communicate_soil`,
                `_cost_to_communicate_rock`, `_cost_to_communicate_image`).
            c.  If `cost_g` is infinity (meaning the goal fact is estimated to
                be unachievable from this state based on the heuristic's logic),
                the total heuristic for the state is infinity. Return `math.inf`
                immediately.
            d.  Otherwise, add `cost_g` to `total_h`.
        5.  After iterating through all goal facts, return the accumulated
            `total_h`.

        Calculation of `cost_g` for a specific unachieved goal fact (e.g., `(communicated_soil_data W)`):
        -   Find all rovers equipped for the required analysis (soil).
        -   If no such rover exists, cost is infinity.
        -   Otherwise, initialize `min_total_cost` to infinity.
        -   For each suitable rover R:
            -   Find R's current location (`r_loc`). If unknown or invalid, skip rover.
            -   Estimate the cost to get the data (`data_cost`) and the waypoint R is at after getting data (`current_wp_for_comm`):
                -   If `(have_soil_analysis R W)` is in the state: `data_cost` is 0, `current_wp_for_comm` is `r_loc`.
                -   Else if `(at_soil_sample W)` is in the state:
                    -   Find R's store. If none, skip rover.
                    -   `store_cost` is 1 if store is full, else 0.
                    -   Calculate navigation cost for R from `r_loc` to W (`sample_nav_cost`). If infinity, `data_cost` is infinity.
                    -   Else: `data_cost = sample_nav_cost + store_cost + 1` (sample action), `current_wp_for_comm` is W.
                -   Else (sample not available or already taken by another rover, and R doesn't have the data): `data_cost` is infinity.
            -   If `data_cost` is infinity, skip rover.
            -   Estimate the cost to communicate (`comm_cost`):
                -   Calculate navigation cost for R from `current_wp_for_comm` to the nearest communication waypoint (`comm_nav_cost`). If infinity, `comm_cost` is infinity.
                -   Else: `comm_cost = comm_nav_cost + 1` (communicate action).
            -   If `comm_cost` is infinity, skip rover.
            -   The total cost for rover R is `data_cost + comm_cost`.
            -   Update `min_total_cost = min(min_total_cost, total_cost_for_r)`.
        -   Return `min_total_cost`.

        Calculation of `cost_g` for `(communicated_image_data O M)` follows a similar structure, replacing sampling steps with calibration and image-taking steps, and finding the optimal intermediate waypoints for those actions.
    """

    def __init__(self, task):
        super().__init__(task)
        self.task = task
        self._parse_static_info(task.static)
        self._precompute_navigation_costs()
        self._precompute_communication_waypoints()
        self._parse_initial_state_samples(task.initial_state)

    def _parse_fact(self, fact_string):
        """Parses a fact string like '(pred arg1 arg2)' into (pred, [arg1, arg2])."""
        # Remove parentheses and split by whitespace
        parts = fact_string[1:-1].split()
        if not parts: # Handle empty fact string if necessary, though unlikely
            return None, []
        return parts[0], parts[1:]

    def _parse_static_info(self, static_facts):
        """Parses static facts to populate internal data structures."""
        self.lander_location = None
        self.soil_rovers = set()
        self.rock_rovers = set()
        self.imaging_rovers = set()
        self.store_to_rover = {}
        self.rover_nav_graph = collections.defaultdict(lambda: collections.defaultdict(set))
        self.waypoint_visibility = collections.defaultdict(set)
        self.camera_on_rover = {}
        self.camera_modes = collections.defaultdict(set)
        self.camera_cal_target = {}
        self.objective_visible_from = collections.defaultdict(set)
        self.all_rovers = set()
        self.all_waypoint_names = set() # Collect all unique waypoint names

        for fact_string in static_facts:
            pred, args = self._parse_fact(fact_string)
            if pred == 'at_lander' and len(args) == 2:
                self.lander_location = args[1]
                self.all_waypoint_names.add(args[1])
            elif pred == 'equipped_for_soil_analysis' and len(args) == 1:
                self.soil_rovers.add(args[0])
                self.all_rovers.add(args[0])
            elif pred == 'equipped_for_rock_analysis' and len(args) == 1:
                self.rock_rovers.add(args[0])
                self.all_rovers.add(args[0])
            elif pred == 'equipped_for_imaging' and len(args) == 1:
                self.imaging_rovers.add(args[0])
                self.all_rovers.add(args[0])
            elif pred == 'store_of' and len(args) == 2:
                self.store_to_rover[args[0]] = args[1]
                self.all_rovers.add(args[1]) # Store belongs to a rover
            elif pred == 'can_traverse' and len(args) == 3:
                rover, wp_from, wp_to = args
                self.rover_nav_graph[rover][wp_from].add(wp_to)
                self.all_rovers.add(rover)
                self.all_waypoint_names.add(wp_from)
                self.all_waypoint_names.add(wp_to)
            elif pred == 'visible' and len(args) == 2:
                wp1, wp2 = args
                self.waypoint_visibility[wp1].add(wp2)
                self.all_waypoint_names.add(wp1)
                self.all_waypoint_names.add(wp2)
            elif pred == 'on_board' and len(args) == 2:
                self.camera_on_rover[args[0]] = args[1]
                self.all_rovers.add(args[1]) # Camera is on a rover
            elif pred == 'supports' and len(args) == 2:
                self.camera_modes[args[0]].add(args[1])
            elif pred == 'calibration_target' and len(args) == 2:
                self.camera_cal_target[args[0]] = args[1]
            elif pred == 'visible_from' and len(args) == 2:
                objective, waypoint = args
                self.objective_visible_from[objective].add(waypoint)
                self.all_waypoint_names.add(waypoint)

        # Add waypoints from initial state facts if they weren't in static
        for fact_string in self.task.initial_state:
             pred, args = self._parse_fact(fact_string)
             if pred == 'at' and len(args) == 2:
                 self.all_waypoint_names.add(args[1])
                 self.all_rovers.add(args[0])
             elif (pred == 'at_soil_sample' or pred == 'at_rock_sample') and len(args) == 1:
                 self.all_waypoint_names.add(args[0])


    def _parse_initial_state_samples(self, initial_state):
        """Stores initial sample locations."""
        self.initial_soil_samples = set()
        self.initial_rock_samples = set()
        for fact_string in initial_state:
             pred, args = self._parse_fact(fact_string)
             if pred == 'at_soil_sample' and len(args) == 1:
                 self.initial_soil_samples.add(args[0])
             elif pred == 'at_rock_sample' and len(args) == 1:
                 self.initial_rock_samples.add(args[0])

    def _bfs(self, graph, start_node):
        """Performs BFS on an adjacency list graph from a start node."""
        distances = {node: math.inf for node in self.all_waypoint_names}

        if start_node not in self.all_waypoint_names:
             # Start node is not a known waypoint, cannot start BFS from here
             return distances

        distances[start_node] = 0
        queue = collections.deque([start_node])

        while queue:
            current_node = queue.popleft()

            # Check if current_node has outgoing edges in the graph
            if current_node in graph:
                for neighbor in graph[current_node]:
                    # Ensure neighbor is a known waypoint before checking distance
                    if neighbor in self.all_waypoint_names and distances[neighbor] == math.inf:
                        distances[neighbor] = distances[current_node] + 1
                        queue.append(neighbor)
        return distances

    def _precompute_navigation_costs(self):
        """Precomputes shortest path distances for each rover between all waypoints."""
        self.rover_nav_distances = {}
        for rover in self.all_rovers:
            self.rover_nav_distances[rover] = {}
            # Get the navigation graph for this rover, including all waypoints
            rover_graph_for_bfs = {wp: set() for wp in self.all_waypoint_names}
            if rover in self.rover_nav_graph:
                 for wp_from, neighbors in self.rover_nav_graph[rover].items():
                      # Only add edges if both endpoints are known waypoints
                      valid_neighbors = {n for n in neighbors if n in self.all_waypoint_names}
                      rover_graph_for_bfs[wp_from].update(valid_neighbors)

            # Run BFS from every waypoint
            for start_wp in self.all_waypoint_names:
                 self.rover_nav_distances[rover][start_wp] = self._bfs(rover_graph_for_bfs, start_wp)


    def _precompute_communication_waypoints(self):
        """Identifies waypoints visible from the lander's location."""
        self.comm_waypoints = set()
        if self.lander_location is None or self.lander_location not in self.all_waypoint_names:
             # Lander location is unknown or not a valid waypoint
             return

        # Need waypoints ?x where (visible ?x lander_location) is true.
        # Build reverse visibility graph
        reverse_visibility = collections.defaultdict(set)
        for wp1, neighbors in self.waypoint_visibility.items():
            for wp2 in neighbors:
                reverse_visibility[wp2].add(wp1)

        if self.lander_location in reverse_visibility:
             self.comm_waypoints.update(reverse_visibility[self.lander_location])

        # Note: Assumes communication requires (visible ?x ?y) where ?x != ?y.
        # If (visible W W) is possible and required for communication at lander loc,
        # this would need adjustment. Standard PDDL usually implies distinct objects.

    def _get_rover_location(self, state, rover):
        """Finds the current waypoint of a specific rover in the state."""
        for fact_string in state:
            pred, args = self._parse_fact(fact_string)
            if pred == 'at' and len(args) == 2 and args[0] == rover:
                return args[1]
        return None # Rover location not found in state (should not happen in valid states)

    def __call__(self, node):
        """Computes the heuristic value for the given state."""
        state = node.state

        # Check if goal is already reached
        if self.task.goals <= state:
            return 0

        total_h = 0
        for goal_fact_string in self.task.goals:
            if goal_fact_string in state:
                continue # Goal already achieved

            cost_g = self._cost_to_achieve(state, goal_fact_string)

            if cost_g == math.inf:
                # If any single unachieved goal is impossible, the whole state is likely a dead end
                return math.inf

            total_h += cost_g

        return total_h

    def _cost_to_achieve(self, state, goal_fact_string):
        """Estimates the minimum cost to achieve a single unachieved goal fact."""
        pred, args = self._parse_fact(goal_fact_string)

        if pred == 'communicated_soil_data' and len(args) == 1:
            waypoint = args[0]
            return self._cost_to_communicate_soil(state, waypoint)
        elif pred == 'communicated_rock_data' and len(args) == 1:
            waypoint = args[0]
            return self._cost_to_communicate_rock(state, waypoint)
        elif pred == 'communicated_image_data' and len(args) == 2:
            objective, mode = args
            return self._cost_to_communicate_image(state, objective, mode)
        else:
            # This case should ideally not be reached if goal facts are well-formed
            # and belong to the domain's communication predicates.
            # Treat as unachievable if it's an unexpected goal type.
            return math.inf

    def _cost_to_communicate_soil(self, state, waypoint):
        """Estimates the minimum cost for a rover to communicate soil data from waypoint."""
        min_total_cost = math.inf
        suitable_rovers = self.soil_rovers

        if not suitable_rovers:
            return math.inf # No rover can sample soil

        # Check if the target waypoint is a known waypoint
        if waypoint not in self.all_waypoint_names:
             return math.inf # Cannot sample at an unknown waypoint

        for rover in suitable_rovers:
            r_loc = self._get_rover_location(state, rover)
            if r_loc is None or r_loc not in self.all_waypoint_names:
                 continue # Rover location unknown or not a valid waypoint

            data_cost = math.inf
            current_wp_for_comm = None # Waypoint where rover is after getting data

            # Case 1: Rover already has the data
            if f'(have_soil_analysis {rover} {waypoint})' in state:
                data_cost = 0
                current_wp_for_comm = r_loc
            # Case 2: Need to sample
            elif f'(at_soil_sample {waypoint})' in state: # Check if sample is currently available
                # Need to sample soil
                store = None
                # Find the store belonging to this rover
                for s, r in self.store_to_rover.items():
                    if r == rover:
                        store = s
                        break
                if store is None:
                    # Rover has no store, cannot sample
                    continue

                store_full = f'(full {store})' in state
                store_cost = 1 if store_full else 0 # Cost of drop action if needed

                # Navigation cost to reach the sample waypoint
                sample_nav_cost = self.rover_nav_distances.get(rover, {}).get(r_loc, {}).get(waypoint, math.inf)

                if sample_nav_cost != math.inf:
                    # Cost = Navigate + Drop (if full) + Sample
                    data_cost = sample_nav_cost + store_cost + 1
                    current_wp_for_comm = waypoint # Rover is at waypoint after sampling
            # Case 3: Sample is gone and rover doesn't have data
            # This is implicitly handled: if sample is not in state, Case 2 is skipped, data_cost remains inf.

            if data_cost == math.inf:
                continue # Cannot get the data with this rover

            # Cost to communicate from current_wp_for_comm
            comm_nav_cost = math.inf
            if self.comm_waypoints and current_wp_for_comm is not None and current_wp_for_comm in self.all_waypoint_names:
                 # Find min nav cost from current_wp_for_comm to any comm waypoint
                 valid_comm_wps = [cw for cw in self.comm_waypoints if cw in self.all_waypoint_names]
                 if valid_comm_wps:
                    comm_nav_cost = min((self.rover_nav_distances.get(rover, {}).get(current_wp_for_comm, {}).get(cw, math.inf) for cw in valid_comm_wps), default=math.inf)


            if comm_nav_cost == math.inf:
                continue # Cannot reach a communication point

            total_cost_for_r = data_cost + comm_nav_cost + 1 # communicate action
            min_total_cost = min(min_total_cost, total_cost_for_r)

        return min_total_cost

    def _cost_to_communicate_rock(self, state, waypoint):
        """Estimates the minimum cost for a rover to communicate rock data from waypoint."""
        min_total_cost = math.inf
        suitable_rovers = self.rock_rovers

        if not suitable_rovers:
            return math.inf # No rover can sample rock

        # Check if the target waypoint is a known waypoint
        if waypoint not in self.all_waypoint_names:
             return math.inf # Cannot sample at an unknown waypoint

        for rover in suitable_rovers:
            r_loc = self._get_rover_location(state, rover)
            if r_loc is None or r_loc not in self.all_waypoint_names:
                 continue # Rover location unknown or not a valid waypoint

            data_cost = math.inf
            current_wp_for_comm = None # Waypoint where rover is after getting data

            # Case 1: Rover already has the data
            if f'(have_rock_analysis {rover} {waypoint})' in state:
                data_cost = 0
                current_wp_for_comm = r_loc
            # Case 2: Need to sample
            elif f'(at_rock_sample {waypoint})' in state: # Check if sample is currently available
                # Need to sample rock
                store = None
                # Find the store belonging to this rover
                for s, r in self.store_to_rover.items():
                    if r == rover:
                        store = s
                        break
                if store is None:
                    # Rover has no store, cannot sample
                    continue

                store_full = f'(full {store})' in state
                store_cost = 1 if store_full else 0 # Cost of drop action if needed

                # Navigation cost to reach the sample waypoint
                sample_nav_cost = self.rover_nav_distances.get(rover, {}).get(r_loc, {}).get(waypoint, math.inf)

                if sample_nav_cost != math.inf:
                    # Cost = Navigate + Drop (if full) + Sample
                    data_cost = sample_nav_cost + store_cost + 1
                    current_wp_for_comm = waypoint # Rover is at waypoint after sampling
            # Case 3: Sample is gone and rover doesn't have data
            # This is implicitly handled.

            if data_cost == math.inf:
                continue # Cannot get the data with this rover

            # Cost to communicate from current_wp_for_comm
            comm_nav_cost = math.inf
            if self.comm_waypoints and current_wp_for_comm is not None and current_wp_for_comm in self.all_waypoint_names:
                 # Find min nav cost from current_wp_for_comm to any comm waypoint
                 valid_comm_wps = [cw for cw in self.comm_waypoints if cw in self.all_waypoint_names]
                 if valid_comm_wps:
                    comm_nav_cost = min((self.rover_nav_distances.get(rover, {}).get(current_wp_for_comm, {}).get(cw, math.inf) for cw in valid_comm_wps), default=math.inf)

            if comm_nav_cost == math.inf:
                continue # Cannot reach a communication point

            total_cost_for_r = data_cost + comm_nav_cost + 1 # communicate action
            min_total_cost = min(min_total_cost, total_cost_for_r)

        return min_total_cost


    def _cost_to_communicate_image(self, state, objective, mode):
        """Estimates the minimum cost for a rover to communicate image data."""
        min_total_cost = math.inf
        suitable_rovers = self.imaging_rovers

        if not suitable_rovers:
            return math.inf # No rover can take images

        for rover in suitable_rovers:
            r_loc = self._get_rover_location(state, rover)
            if r_loc is None or r_loc not in self.all_waypoint_names:
                 continue # Rover location unknown or not a valid waypoint

            # Find a camera on this rover that supports the mode
            camera = None
            # Iterate through cameras on this rover
            for cam, r in self.camera_on_rover.items():
                if r == rover and mode in self.camera_modes.get(cam, set()):
                    camera = cam
                    break # Pick the first suitable camera found

            if camera is None:
                continue # This rover doesn't have a suitable camera

            image_cost = math.inf
            image_wp = None # Waypoint where image is taken

            # Case 1: Rover already has the image
            if f'(have_image {rover} {objective} {mode})' in state:
                image_cost = 0
                image_wp = r_loc # Rover is at its current location

            # Case 2: Need to take image
            else:
                # Calculate cost and waypoint to take the image
                (cost_to_take, wp_taken) = self._cost_and_wp_to_take_image(state, rover, objective, mode, camera)
                image_cost = cost_to_take
                image_wp = wp_taken

            if image_cost == math.inf:
                continue # Cannot get the image with this rover

            # Cost to communicate from image_wp
            comm_nav_cost = math.inf
            if self.comm_waypoints and image_wp is not None and image_wp in self.all_waypoint_names:
                 # Find min nav cost from image_wp to any comm waypoint
                 valid_comm_wps = [cw for cw in self.comm_waypoints if cw in self.all_waypoint_names]
                 if valid_comm_wps:
                    comm_nav_cost = min((self.rover_nav_distances.get(rover, {}).get(image_wp, {}).get(cw, math.inf) for cw in valid_comm_wps), default=math.inf)

            if comm_nav_cost == math.inf:
                continue # Cannot reach a communication point

            total_cost_for_r = image_cost + comm_nav_cost + 1 # communicate action
            min_total_cost = min(min_total_cost, total_cost_for_r)

        return min_total_cost


    def _cost_and_wp_to_take_image(self, state, rover, objective, mode, camera):
        """Estimates the minimum cost and the resulting waypoint to take an image."""
        r_loc = self._get_rover_location(state, rover)
        if r_loc is None or r_loc not in self.all_waypoint_names:
             return (math.inf, None)

        # Waypoints where the objective is visible from
        W_img = self.objective_visible_from.get(objective, set())
        # Filter for known waypoints
        W_img = {wp for wp in W_img if wp in self.all_waypoint_names}
        if not W_img:
            return (math.inf, None) # Objective not visible from any known waypoint

        # Calibration target for the camera
        cal_target = self.camera_cal_target.get(camera)
        if cal_target is None:
            return (math.inf, None) # Camera has no calibration target

        # Waypoints where the calibration target is visible from
        W_cal = self.objective_visible_from.get(cal_target, set())
        # Filter for known waypoints
        W_cal = {wp for wp in W_cal if wp in self.all_waypoint_names}
        if not W_cal:
            return (math.inf, None) # Calibration target not visible from any known waypoint

        min_nav_cal_img = math.inf
        best_image_wp = None

        # Find min cost over all pairs of (calibration_waypoint, image_waypoint)
        for w_cal in W_cal:
            # Cost to navigate from current location to calibration waypoint
            nav_r_to_w_cal = self.rover_nav_distances.get(rover, {}).get(r_loc, {}).get(w_cal, math.inf)
            if nav_r_to_w_cal == math.inf:
                continue # Cannot reach this calibration waypoint

            for w_img in W_img:
                # Cost to navigate from calibration waypoint to image waypoint
                nav_w_cal_to_w_img = self.rover_nav_distances.get(rover, {}).get(w_cal, {}).get(w_img, math.inf)
                if nav_w_cal_to_w_img == math.inf:
                    continue # Cannot reach this image waypoint from calibration waypoint

                # Total cost for this path:
                # Nav(r_loc -> w_cal) + Calibrate (1) + Nav(w_cal -> w_img) + TakeImage (1)
                current_nav_cal_img = nav_r_to_w_cal + 1 + nav_w_cal_to_w_img + 1

                if current_nav_cal_img < min_nav_cal_img:
                    min_nav_cal_img = current_nav_cal_img
                    best_image_wp = w_img # The waypoint where the image is taken

        return (min_nav_cal_img, best_image_wp)
