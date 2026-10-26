import math
from collections import deque, defaultdict

# Helper function to parse PDDL facts
def parse_fact(fact_str):
    """Parses a PDDL fact string into a tuple."""
    # Remove parentheses and split by space
    parts = fact_str[1:-1].split()
    return tuple(parts)

# Helper function to format PDDL facts
def format_fact(predicate, *args):
    """Formats a predicate and arguments into a PDDL fact string."""
    return f"({predicate} {' '.join(args)})"

class roversHeuristic:
    """
    Domain-dependent heuristic for the Rovers domain.

    Summary:
    The heuristic estimates the cost to reach the goal by summing up the
    estimated costs for each unachieved goal fact. For each unachieved
    goal (communicating soil data, rock data, or image data), it calculates
    the minimum number of actions required to achieve that specific goal fact,
    considering the necessary steps: navigation, sampling/imaging, and communication.
    It considers the capabilities and current location of different rovers
    and selects the minimum cost path among them. Navigation costs are
    estimated using precomputed shortest paths on the rover-specific traverse graphs.

    Assumptions:
    - The heuristic assumes that the cost of achieving multiple goals is the sum
      of the costs of achieving each goal independently. This ignores potential
      synergies (e.g., one trip fulfilling multiple navigation needs).
    - It assumes that sampling/imaging actions consume the required resource
      (soil/rock sample, camera calibration) and require an empty store for samples.
    - It assumes communication requires the rover to be at a waypoint visible
      from the lander and having the required data.
    - It assumes camera calibration is lost after taking an image.
    - It assumes static facts (like `at_soil_sample`, `at_rock_sample`,
      `can_traverse`, `visible`, `visible_from`, equipment, camera info, store info,
      lander location) do not change during planning.
    - The heuristic returns infinity (math.inf) if a goal fact is determined
      to be unreachable based on static information (e.g., no equipped rover,
      no sample at location, no communication point, disconnected graph).
    - State facts that contradict static facts (e.g., a camera calibrated on a rover it's not on board of according to static) are ignored or handled based on static capabilities.

    Heuristic Initialization:
    The constructor precomputes static information from the task's static facts:
    - Lander location(s).
    - Rover capabilities (equipped for soil, rock, imaging).
    - Store ownership.
    - Camera information (on board, supported modes, calibration target).
    - Waypoint visibility graph (used for communication points).
    - Objective visibility from waypoints (used for imaging and calibration).
    - Rover-specific traverse graphs (`can_traverse`).
    - Shortest path distances between all pairs of waypoints for each rover
      using BFS on their respective traverse graphs.
    - Identifies communication waypoints (visible from any lander location).
    - Identifies static locations of soil/rock samples.

    Step-By-Step Thinking for Computing Heuristic:
    1.  Check if the current state is a goal state. If yes, return 0.
    2.  Identify the set of goal facts that are not present in the current state.
    3.  Initialize the total heuristic value `h` to 0.
    4.  Pre-calculate state-dependent information needed for goal evaluation (rover locations, store status, rovers holding samples/images).
    5.  For each unachieved goal fact `g`:
        a.  If `g` is `(communicated_soil_data ?w)`:
            i.  Check if `(at_soil_sample ?w)` is true in the static facts. If not, this goal is impossible; add infinity to `h` and continue to the next goal.
            ii. Check if communication waypoints exist. If not, add infinity to `h` and continue.
            iii. Find rovers equipped for soil analysis. If none exist, add infinity to `h` and continue.
            iv. Calculate the minimum cost to achieve `(communicated_soil_data ?w)`. This involves two main phases: getting the sample and communicating the sample.
                - Consider two scenarios:
                    - Scenario A: A rover already has the sample (`(have_soil_analysis ?r ?w)` is in the current state for some `?r`). The cost is the minimum navigation cost for such a rover from its current location to any communication waypoint, plus 1 (for the communicate action).
                    - Scenario B: No rover has the sample yet. We need to sample. For each rover equipped for soil analysis, calculate the cost: (navigate from its current location to `?w`) + (1 if its store is full, for `drop`) + 1 (`sample_soil`) + (navigate from `?w` to a communication waypoint) + 1 (`communicate_soil_data`).
                - The minimum cost for this goal fact is the minimum cost across all applicable scenarios and rovers. Add this minimum cost to `h`. If all scenarios/rovers lead to infinite cost, add infinity.
        b.  If `g` is `(communicated_rock_data ?w)`: Follow a similar logic as for soil data, using rock-specific predicates (`at_rock_sample`, `equipped_for_rock_analysis`, `have_rock_analysis`, `communicate_rock_data`).
        c.  If `g` is `(communicated_image_data ?o ?m)`:
            i.  Check if communication waypoints exist. If not, add infinity to `h` and continue.
            ii. Find image waypoints visible from `?o` in static facts. If none exist, add infinity to `h` and continue.
            iii. Find rovers equipped for imaging that have a camera supporting mode `?m` according to static facts. If none exist, add infinity to `h` and continue.
            iv. Calculate the minimum cost to achieve `(communicated_image_data ?o ?m)`. This involves two main phases: taking the image and communicating the image.
                - Consider two scenarios:
                    - Scenario A: A rover already has the image (`(have_image ?r ?o ?m)` is in the current state for some `?r`). The cost is the minimum navigation cost for such a rover from its current location to any communication waypoint, plus 1 (for the communicate action).
                    - Scenario B: No rover has the image yet. We need to take the image. For each suitable rover/camera pair (equipped rover with camera on board supporting the mode):
                        - Find the calibration target for the camera and its visible waypoints in static facts. If none exist, this pair cannot achieve the goal.
                        - Calculate the cost: (navigate from the rover's current location to a calibration waypoint) + 1 (`calibrate`) + (navigate from a calibration waypoint to an image waypoint) + 1 (`take_image`) + (navigate from an image waypoint to a communication waypoint) + 1 (`communicate_image_data`).
                - The minimum cost for this goal fact is the minimum cost across all applicable scenarios and rover/camera pairs. Add this minimum cost to `h`. If all scenarios/pairs lead to infinite cost, add infinity.
        d.  If `g` is any other unachieved goal fact not covered above, assume it's impossible or unhandleable by this heuristic and add infinity to `h`.
    6.  Return the total heuristic value `h`.

    """
    def __init__(self, task):
        self.task = task
        self.static = task.static
        self.goals = task.goals

        # --- Precompute Static Information ---
        self.lander_locations = {} # lander -> waypoint
        self.rover_equipment = defaultdict(set) # equipment_type -> set of rovers
        self.store_owners = {} # store -> rover
        self.rover_stores = defaultdict(set) # rover -> set of stores
        self.camera_info = defaultdict(dict) # camera -> { 'on_board': rover, 'supports': set(modes), 'calibration_target': objective }
        self.objective_visible_from = defaultdict(set) # objective -> set of waypoints
        self.waypoint_visibility = defaultdict(set) # waypoint -> set of visible waypoints (undirected graph)
        self.rover_traverse_graphs = defaultdict(lambda: defaultdict(set)) # rover -> waypoint -> set of reachable waypoints (directed graph)
        self.waypoints = set()
        self.rovers = set()
        self.landers = set()
        self.objectives = set()
        self.cameras = set()
        self.modes = set()
        self.stores = set()
        self.soil_sample_locations = set() # Waypoints with static soil samples
        self.rock_sample_locations = set() # Waypoints with static rock samples


        # Parse static facts
        for fact_str in self.static:
            fact = parse_fact(fact_str)
            predicate = fact[0]
            args = fact[1:]

            if predicate == 'at_lander':
                self.landers.add(args[0])
                self.waypoints.add(args[1])
                self.lander_locations[args[0]] = args[1]
            elif predicate == 'equipped_for_soil_analysis':
                self.rovers.add(args[0])
                self.rover_equipment['soil'].add(args[0])
            elif predicate == 'equipped_for_rock_analysis':
                self.rovers.add(args[0])
                self.rover_equipment['rock'].add(args[0])
            elif predicate == 'equipped_for_imaging':
                self.rovers.add(args[0])
                self.rover_equipment['imaging'].add(args[0])
            elif predicate == 'store_of':
                self.stores.add(args[0])
                self.rovers.add(args[1])
                self.store_owners[args[0]] = args[1]
                self.rover_stores[args[1]].add(args[0])
            elif predicate == 'visible':
                self.waypoints.add(args[0])
                self.waypoints.add(args[1])
                self.waypoint_visibility[args[0]].add(args[1])
                # Assuming visible is symmetric
                self.waypoint_visibility[args[1]].add(args[0])
            elif predicate == 'can_traverse':
                self.rovers.add(args[0])
                self.waypoints.add(args[1])
                self.waypoints.add(args[2])
                self.rover_traverse_graphs[args[0]][args[1]].add(args[2])
            elif predicate == 'visible_from':
                self.objectives.add(args[0])
                self.waypoints.add(args[1])
                self.objective_visible_from[args[0]].add(args[1])
            elif predicate == 'on_board':
                self.cameras.add(args[0])
                self.rovers.add(args[1])
                if args[0] not in self.camera_info: self.camera_info[args[0]] = {}
                self.camera_info[args[0]]["on_board"] = args[1]
            elif predicate == 'supports':
                self.cameras.add(args[0])
                self.modes.add(args[1])
                if args[0] not in self.camera_info: self.camera_info[args[0]] = {}
                if 'supports' not in self.camera_info[args[0]]: self.camera_info[args[0]]['supports'] = set()
                self.camera_info[args[0]]['supports'].add(args[1])
            elif predicate == 'calibration_target':
                self.cameras.add(args[0])
                self.objectives.add(args[1])
                if args[0] not in self.camera_info: self.camera_info[args[0]] = {}
                self.camera_info[args[0]]['calibration_target'] = args[1]
            elif predicate == 'at_soil_sample':
                 self.waypoints.add(args[0])
                 self.soil_sample_locations.add(args[0])
            elif predicate == 'at_rock_sample':
                 self.waypoints.add(args[0])
                 self.rock_sample_locations.add(args[0])

            # Collect all objects mentioned
            for arg in args:
                 if arg and arg[0].isalpha():
                     if arg.startswith('rover'): self.rovers.add(arg)
                     elif arg.startswith('waypoint'): self.waypoints.add(arg)
                     elif arg.startswith('store'): self.stores.add(arg)
                     elif arg.startswith('camera'): self.cameras.add(arg)
                     elif arg.startswith('mode'): self.modes.add(arg)
                     elif arg.startswith('lander'): self.landers.add(arg)
                     elif arg.startswith('objective'): self.objectives.add(arg)


        # Precompute shortest paths for each rover
        self.dist = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: math.inf)))
        for rover in self.rovers:
            graph = self.rover_traverse_graphs[rover]
            relevant_waypoints = self.waypoints # Use all waypoints identified

            for start_wp in relevant_waypoints:
                q = deque([(start_wp, 0)])
                visited = {start_wp: 0}
                self.dist[rover][start_wp][start_wp] = 0

                while q:
                    current_wp, d = q.popleft()

                    if current_wp in graph: # Check if current_wp is a source in the graph
                        for neighbor_wp in graph[current_wp]:
                            if neighbor_wp not in visited:
                                visited[neighbor_wp] = d + 1
                                self.dist[rover][start_wp][neighbor_wp] = d + 1
                                q.append((neighbor_wp, d + 1))


        # Identify communication waypoints
        self.comm_wps = set()
        # Assuming lander location is static and unique or we need to reach any lander location
        lander_wp = next(iter(self.lander_locations.values()), None)
        if lander_wp:
             # Communication requires being at a waypoint visible from the lander's location
             if lander_wp in self.waypoint_visibility:
                self.comm_wps.update(self.waypoint_visibility[lander_wp])
             # Also check if the lander location itself is visible from itself (unlikely but possible)
             if lander_wp in self.waypoint_visibility and lander_wp in self.waypoint_visibility[lander_wp]:
                 self.comm_wps.add(lander_wp)


    def get_rover_location(self, state_facts, rover):
        """Finds the current waypoint of a rover in the given state."""
        for fact_str in state_facts:
            fact = parse_fact(fact_str)
            if fact[0] == 'at' and fact[1] == rover:
                return fact[2]
        # If rover location is not in current state, check initial state
        for fact_str in self.task.initial_state:
             fact = parse_fact(fact_str)
             if fact[0] == 'at' and fact[1] == rover:
                 return fact[2]
        return None # Should not happen for a valid problem instance with rovers

    def get_nav_cost(self, rover, start_wp, end_wps):
        """
        Gets the minimum navigation cost for a rover from start_wp to any waypoint in end_wps.
        Returns math.inf if no path exists to any target.
        """
        if not end_wps or start_wp is None: # Cannot navigate from unknown location
            return math.inf
        min_d = math.inf
        if rover in self.dist and start_wp in self.dist[rover]:
             for end_wp in end_wps:
                 min_d = min(min_d, self.dist[rover][start_wp].get(end_wp, math.inf))
        return min_d

    def get_min_nav_cost_set_to_set(self, rover, start_wps, end_wps):
        """
        Gets the minimum navigation cost for a rover from any waypoint in start_wps
        to any waypoint in end_wps.
        Returns math.inf if no path exists between any pair.
        """
        if not start_wps or not end_wps:
            return math.inf
        min_d = math.inf
        for start_wp in start_wps:
            min_d = min(min_d, self.get_nav_cost(rover, start_wp, end_wps))
        return min_d


    def __call__(self, state):
        """
        Computes the heuristic value for the given state.
        """
        # Check if it's a goal state
        if self.goals <= state:
            return 0

        h = 0
        unachieved_goals = self.goals - state

        # Pre-check for unreachable goals based on static info (e.g., no comm points)
        if not self.comm_wps and any(g.startswith('(communicated_') for g in unachieved_goals):
             return math.inf # Cannot communicate if no comm points

        # State-dependent information needed for heuristic calculation
        state_facts = set(state) # Convert frozenset to set for faster lookups
        rover_locations = {r: self.get_rover_location(state_facts, r) for r in self.rovers}
        rover_full_stores = {r: any(format_fact('full', s) in state_facts for s in self.rover_stores.get(r, set())) for r in self.rovers}
        rovers_with_soil_sample = defaultdict(set) # waypoint -> set of rovers
        rovers_with_rock_sample = defaultdict(set) # waypoint -> set of rovers
        rovers_with_image = defaultdict(set) # (objective, mode) -> set of rovers

        for fact_str in state_facts:
            fact = parse_fact(fact_str)
            if fact[0] == 'have_soil_analysis':
                if len(fact) == 3: rovers_with_soil_sample[fact[2]].add(fact[1])
            elif fact[0] == 'have_rock_analysis':
                 if len(fact) == 3: rovers_with_rock_sample[fact[2]].add(fact[1])
            elif fact[0] == 'have_image':
                 if len(fact) == 4: rovers_with_image[(fact[2], fact[3])].add(fact[1])


        # Calculate cost for each unachieved goal fact
        for goal_fact_str in unachieved_goals:
            goal_fact = parse_fact(goal_fact_str)
            predicate = goal_fact[0]

            cost_g = math.inf # Cost for this specific goal fact

            if predicate == 'communicated_soil_data':
                if len(goal_fact) != 2: continue # Malformed goal
                waypoint = goal_fact[1]

                # Check if sample exists statically
                if waypoint not in self.soil_sample_locations:
                    h += math.inf
                    continue # Goal impossible

                # Scenario A: Communicate existing sample
                rovers_with_sample = rovers_with_soil_sample.get(waypoint, set())
                min_cost_comm_existing = math.inf
                if rovers_with_sample:
                    for r in rovers_with_sample:
                        current_wp = rover_locations.get(r)
                        if current_wp:
                            nav_cost = self.get_nav_cost(r, current_wp, self.comm_wps)
                            if nav_cost != math.inf:
                                min_cost_comm_existing = min(min_cost_comm_existing, nav_cost + 1)

                # Scenario B: Sample and Communicate
                rovers_can_sample = self.rover_equipment.get('soil', set())
                min_cost_sample_and_comm = math.inf
                if rovers_can_sample:
                    for r in rovers_can_sample:
                        current_wp = rover_locations.get(r)
                        if current_wp:
                            # Cost to sample
                            nav_to_sample = self.get_nav_cost(r, current_wp, {waypoint})
                            if nav_to_sample == math.inf:
                                continue # Cannot reach sample location
                            store_full_cost = 1 if rover_full_stores.get(r, False) else 0
                            cost_get_sample = nav_to_sample + store_full_cost + 1

                            # Cost to communicate from sample location
                            # Assume rover is at 'waypoint' after sampling
                            nav_from_sample_to_comm = self.get_nav_cost(r, waypoint, self.comm_wps)
                            if nav_from_sample_to_comm == math.inf:
                                continue # Cannot reach comm location from sample location
                            cost_comm = nav_from_sample_to_comm + 1

                            min_cost_sample_and_comm = min(min_cost_sample_and_comm, cost_get_sample + cost_comm)

                cost_g = min(min_cost_comm_existing, min_cost_sample_and_comm)

            elif predicate == 'communicated_rock_data':
                if len(goal_fact) != 2: continue # Malformed goal
                waypoint = goal_fact[1]

                 # Check if sample exists statically
                if waypoint not in self.rock_sample_locations:
                    h += math.inf
                    continue # Goal impossible

                # Scenario A: Communicate existing sample
                rovers_with_sample = rovers_with_rock_sample.get(waypoint, set())
                min_cost_comm_existing = math.inf
                if rovers_with_sample:
                    for r in rovers_with_sample:
                        current_wp = rover_locations.get(r)
                        if current_wp:
                            nav_cost = self.get_nav_cost(r, current_wp, self.comm_wps)
                            if nav_cost != math.inf:
                                min_cost_comm_existing = min(min_cost_comm_existing, nav_cost + 1)

                # Scenario B: Sample and Communicate
                rovers_can_sample = self.rover_equipment.get('rock', set())
                min_cost_sample_and_comm = math.inf
                if rovers_can_sample:
                    for r in rovers_can_sample:
                        current_wp = rover_locations.get(r)
                        if current_wp:
                            # Cost to sample
                            nav_to_sample = self.get_nav_cost(r, current_wp, {waypoint})
                            if nav_to_sample == math.inf:
                                continue # Cannot reach sample location
                            store_full_cost = 1 if rover_full_stores.get(r, False) else 0
                            cost_get_sample = nav_to_sample + store_full_cost + 1

                            # Cost to communicate from sample location
                            # Assume rover is at 'waypoint' after sampling
                            nav_from_sample_to_comm = self.get_nav_cost(r, waypoint, self.comm_wps)
                            if nav_from_sample_to_comm == math.inf:
                                continue # Cannot reach comm location from sample location
                            cost_comm = nav_from_sample_to_comm + 1

                            min_cost_sample_and_comm = min(min_cost_sample_and_comm, cost_get_sample + cost_comm)

                cost_g = min(min_cost_comm_existing, min_cost_sample_and_comm)


            elif predicate == 'communicated_image_data':
                if len(goal_fact) != 3: continue # Malformed goal
                objective = goal_fact[1]
                mode = goal_fact[2]

                # Check if image waypoints exist
                img_wps = self.objective_visible_from.get(objective, set())
                if not img_wps:
                    h += math.inf
                    continue # Goal impossible

                # Scenario A: Communicate existing image
                rovers_with_image_key = (objective, mode)
                rovers_with_image_now = rovers_with_image.get(rovers_with_image_key, set())
                min_cost_comm_existing = math.inf
                if rovers_with_image_now:
                    for r in rovers_with_image_now:
                        current_wp = rover_locations.get(r)
                        if current_wp:
                            nav_cost = self.get_nav_cost(r, current_wp, self.comm_wps)
                            if nav_cost != math.inf:
                                min_cost_comm_existing = min(min_cost_comm_existing, nav_cost + 1)

                # Scenario B: Take Image and Communicate
                min_cost_take_and_comm = math.inf
                # Find rovers equipped for imaging with a camera supporting the mode
                suitable_rc_pairs = set()
                for r in self.rover_equipment.get('imaging', set()):
                    for cam, info in self.camera_info.items():
                        if info.get('on_board') == r and mode in info.get('supports', set()):
                            suitable_rc_pairs.add((r, cam))

                if suitable_rc_pairs:
                    for r, cam in suitable_rc_pairs:
                        current_wp = rover_locations.get(r)
                        if not current_wp: continue # Rover location unknown

                        cal_target = self.camera_info.get(cam, {}).get('calibration_target')
                        if not cal_target: continue # Camera has no calibration target

                        cal_wps = self.objective_visible_from.get(cal_target, set())
                        if not cal_wps: continue # Calibration target not visible from anywhere

                        # Cost to calibrate
                        nav_to_cal = self.get_nav_cost(r, current_wp, cal_wps)
                        if nav_to_cal == math.inf: continue # Cannot reach calibration waypoint
                        cost_calibrate = nav_to_cal + 1

                        # Cost to take image (from any cal_wp to any img_wp)
                        nav_cal_to_img = self.get_min_nav_cost_set_to_set(r, cal_wps, img_wps)
                        if nav_cal_to_img == math.inf: continue # Cannot reach image waypoint from any cal waypoint
                        cost_take = nav_cal_to_img + 1

                        # Cost to communicate (from any img_wp to any comm_wp)
                        nav_img_to_comm = self.get_min_nav_cost_set_to_set(r, img_wps, self.comm_wps)
                        if nav_img_to_comm == math.inf: continue # Cannot reach comm waypoint from any image waypoint
                        cost_comm = nav_img_to_comm + 1

                        min_cost_take_and_comm = min(min_cost_take_and_comm, cost_calibrate + cost_take + cost_comm)

                cost_g = min(min_cost_comm_existing, min_cost_take_and_comm)

            else:
                 # Unhandled unachieved goal fact - assume impossible
                 cost_g = math.inf

            # Add the cost for this goal fact to the total heuristic
            h += cost_g

        return h
