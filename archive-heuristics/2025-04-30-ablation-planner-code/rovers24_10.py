from fnmatch import fnmatch
from collections import defaultdict, deque
import math

# Assume Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential leading/trailing whitespace or malformed facts defensively
    fact = fact.strip()
    if not fact.startswith('(') or not fact.endswith(')'):
        # Return empty list for malformed facts
        return []
    return fact[1:-1].split()

# Helper function for BFS shortest path
def bfs(graph, start, waypoints):
    """Compute shortest paths from start node in a graph."""
    distances = {wp: math.inf for wp in waypoints}
    if start not in waypoints:
        # Start node is not a known waypoint, cannot compute paths
        return distances

    distances[start] = 0
    queue = deque([start])

    while queue:
        u = queue.popleft()
        if u in graph: # Check if u has neighbors in the graph
            for v in graph[u]:
                if distances[v] == math.inf:
                    distances[v] = distances[u] + 1
                    queue.append(v)
    return distances

# Define the heuristic class inheriting from Heuristic
# class roversHeuristic(Heuristic): # Uncomment this line in the final code
class roversHeuristic: # Use this for standalone testing without the base class
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve the goal
    conditions by summing the estimated costs for each unmet goal. The cost for
    an unmet goal is estimated based on the actions and navigation steps needed
    to acquire the required data (soil sample, rock sample, or image) and then
    communicate it. Navigation costs are estimated using precomputed shortest
    paths on the waypoint graph for each rover.

    # Assumptions
    - The heuristic assumes that for each unmet goal, the necessary data acquisition
      and communication steps are performed independently by the most suitable
      and closest available rover.
    - It estimates navigation cost as the shortest path distance for the chosen
      rover from its current location to the required site (sample location,
      image location, calibration location, or communication location).
    - Calibration is assumed to happen immediately before taking an image if needed.
    - Dropping a sample is assumed to happen immediately before sampling if the
      rover's store is full.
    - If a sample location is no longer available in the state, the corresponding
      soil/rock goal is considered unreachable.

    # Heuristic Initialization
    The initialization phase precomputes static information from the PDDL task:
    - Goal conditions.
    - Lander location and waypoints visible from the lander.
    - Rover capabilities (soil, rock, imaging).
    - Store ownership for each rover.
    - Initial locations of soil and rock samples (used to identify potential sample sites).
    - Waypoint visibility graph (for communication).
    - Rover-specific navigation graphs (`can_traverse`).
    - Shortest path distances between all pairs of waypoints for each rover's
      navigation graph using BFS.
    - Camera information (which rover, supported modes, calibration target).
    - Visibility between objectives/calibration targets and waypoints (`visible_from`).

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic calculates the total estimated cost as follows:

    1.  Identify all goal conditions that are not met in the current state.
    2.  For each unmet goal (e.g., `(communicated_soil_data w)`):
        a.  Check if the required intermediate data (`(have_soil_analysis r w)`)
            exists for any rover `r` in the current state.
        b.  If the data does *not* exist:
            i.  Estimate the cost to acquire the data (e.g., sample soil at `w`):
                -   Add 1 for the action (`sample_soil`).
                -   If the sample location `w` is no longer available in the state
                    (`(at_soil_sample w)` is false), the goal is unreachable (cost is infinity).
                -   Find the closest suitable rover `r` (soil-equipped) to `w`.
                    Add the shortest path distance from `r`'s current location to `w`.
                -   Check if `r`'s store is full. If yes, add 1 for the `drop` action.
            ii. Estimate the cost to communicate the data:
                -   Add 1 for the action (`communicate_soil_data`).
                -   Find the closest rover that will have the data (the one that sampled)
                    to any waypoint visible from the lander. Add the shortest path
                    distance from the sample location (`w`) to that lander-visible waypoint.
        c.  If the data *does* exist (for some rover `r`):
            i.  Estimate the cost to communicate the data:
                -   Add 1 for the action (`communicate_soil_data`).
                -   Find the closest rover `r` that has the data to any waypoint
                    visible from the lander. Add the shortest path distance from
                    `r`'s current location to that lander-visible waypoint.
    3.  Repeat step 2 for unmet rock data goals (`(communicated_rock_data w)`),
        following similar logic but using rock-specific predicates and capabilities.
    4.  Repeat step 2 for unmet image data goals (`(communicated_image_data o m)`):
        a.  Check if the required intermediate data (`(have_image r o m)`) exists
            for any rover `r` in the current state.
        b.  If the data does *not* exist:
            i.  Estimate the cost to acquire the data (take image of `o` in mode `m`):
                -   Add 1 for the action (`take_image`).
                -   Find imaging-equipped rovers `R_img` with cameras supporting `m`.
                -   Find the closest rover `r` in `R_img` to any waypoint `p` visible from `o`.
                    Add the shortest path distance from `r`'s current location to `p`.
                -   Estimate calibration cost: Find a suitable camera `i` on `r`.
                    If `(calibrated i r)` is not in the state:
                    -   Add 1 for the action (`calibrate`).
                    -   Find the calibration target `t` for `i`. Find the closest
                        waypoint `w` visible from `t` to the image waypoint `p`.
                        Add the shortest path distance from `p` to `w`.
            ii. Estimate the cost to communicate the data:
                -   Add 1 for the action (`communicate_image_data`).
                -   Find the closest rover that will have the image (the one that took it,
                    assumed to be at waypoint `p`) to any waypoint visible from the lander.
                    Add the shortest path distance from `p` to that lander-visible waypoint.
        c.  If the data *does* exist (for some rover `r`):
            i.  Estimate the cost to communicate the data:
                -   Add 1 for the action (`communicate_image_data`).
                -   Find the closest rover `r` that has the image to any waypoint
                    visible from the lander. Add the shortest path distance from
                    `r`'s current location to that lander-visible waypoint.
    5.  Sum up the estimated costs for all unmet goals. If any required navigation
        or action is impossible (e.g., no path exists, sample is gone), the total
        heuristic is infinity.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # --- Extract Static Information ---
        self.lander_location = None
        self.waypoint_visibility = defaultdict(set) # For communication visibility
        self.rover_navigation_graphs = defaultdict(lambda: defaultdict(set)) # For can_traverse
        self.rover_capabilities = defaultdict(set)
        self.rover_store = {} # rover -> store
        self.initial_soil_samples = set() # Waypoints with initial soil samples
        self.initial_rock_samples = set() # Waypoints with initial rock samples
        self.camera_rover = {} # camera -> rover
        self.camera_modes = defaultdict(set) # camera -> {modes}
        self.camera_calib_target = {} # camera -> objective (calibration target)
        self.objective_visibility = defaultdict(set) # objective -> {waypoints visible from}
        self.calibration_target_visibility = defaultdict(set) # objective (calib target) -> {waypoints visible from}

        all_waypoints = set()
        all_rovers = set()

        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            args = parts[1:]

            if predicate == "at_lander":
                # Assuming only one lander for simplicity based on examples
                if len(args) == 2:
                    self.lander_location = args[1]
                    all_waypoints.add(args[1])
            elif predicate == "visible":
                if len(args) == 2:
                    wp1, wp2 = args
                    self.waypoint_visibility[wp1].add(wp2)
                    all_waypoints.add(wp1)
                    all_waypoints.add(wp2)
            elif predicate == "can_traverse":
                if len(args) == 3:
                    rover, wp1, wp2 = args
                    self.rover_navigation_graphs[rover][wp1].add(wp2)
                    all_rovers.add(rover)
                    all_waypoints.add(wp1)
                    all_waypoints.add(wp2)
            elif predicate == "equipped_for_soil_analysis":
                if len(args) == 1: self.rover_capabilities[args[0]].add("soil")
                if len(args) == 1: all_rovers.add(args[0])
            elif predicate == "equipped_for_rock_analysis":
                if len(args) == 1: self.rover_capabilities[args[0]].add("rock")
                if len(args) == 1: all_rovers.add(args[0])
            elif predicate == "equipped_for_imaging":
                if len(args) == 1: self.rover_capabilities[args[0]].add("imaging")
                if len(args) == 1: all_rovers.add(args[0])
            elif predicate == "store_of":
                if len(args) == 2:
                    store, rover = args
                    self.rover_store[rover] = store
                    all_rovers.add(rover)
            elif predicate == "at_soil_sample":
                if len(args) == 1: self.initial_soil_samples.add(args[0])
                if len(args) == 1: all_waypoints.add(args[0])
            elif predicate == "at_rock_sample":
                if len(args) == 1: self.initial_rock_samples.add(args[0])
                if len(args) == 1: all_waypoints.add(args[0])
            elif predicate == "on_board":
                if len(args) == 2:
                    camera, rover = args
                    self.camera_rover[camera] = rover
                    all_rovers.add(rover)
            elif predicate == "supports":
                if len(args) == 2:
                    camera, mode = args
                    self.camera_modes[camera].add(mode)
            elif predicate == "calibration_target":
                if len(args) == 2:
                    camera, objective = args
                    self.camera_calib_target[camera] = objective
            elif predicate == "visible_from":
                if len(args) == 2:
                    objective, waypoint = args
                    self.objective_visibility[objective].add(waypoint)
                    all_waypoints.add(waypoint)

        # Ensure all waypoints from navigation graphs are included
        for rover_graph in self.rover_navigation_graphs.values():
             for wp in rover_graph:
                 all_waypoints.add(wp)
                 for neighbor in rover_graph[wp]:
                     all_waypoints.add(neighbor)

        self.all_waypoints = list(all_waypoints) # Convert to list for consistent ordering if needed, though dict keys are fine

        # Find lander-visible waypoints
        self.lander_visible_waypoints = set()
        if self.lander_location:
            if self.lander_location in self.waypoint_visibility:
                 self.lander_visible_waypoints.update(self.waypoint_visibility[self.lander_location])
            # A waypoint is visible from itself for communication
            self.lander_visible_waypoints.add(self.lander_location)


        # Precompute shortest paths for each rover
        self.rover_shortest_paths = {}
        for rover in self.rover_navigation_graphs:
            self.rover_shortest_paths[rover] = {}
            for start_wp in self.all_waypoints:
                 # Run BFS from each waypoint for this rover's graph
                 self.rover_shortest_paths[rover][start_wp] = bfs(
                     self.rover_navigation_graphs[rover], start_wp, self.all_waypoints
                 )

        # Precompute calibration target visibility for easier lookup
        for camera, target in self.camera_calib_target.items():
             if target in self.objective_visibility:
                 self.calibration_target_visibility[target].update(self.objective_visibility[target])


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        h = 0
        unreachable = False

        # --- Extract Dynamic Information from State ---
        rover_locations = {} # rover -> waypoint
        store_is_full = set() # {store}
        have_soil_analysis_facts = set() # {(rover, waypoint)}
        have_rock_analysis_facts = set() # {(rover, waypoint)}
        have_image_facts = set() # {(rover, objective, mode)}
        calibrated_facts = set() # {(camera, rover)}
        current_soil_samples = set() # {waypoint}
        current_rock_samples = set() # {waypoint}

        for fact_str in state:
            parts = get_parts(fact_str)
            if not parts: continue

            predicate = parts[0]
            args = parts[1:]

            if predicate == "at":
                if len(args) == 2:
                    obj_name, wp_name = args
                    # Check if it's a known rover based on navigation graphs
                    if obj_name in self.rover_navigation_graphs:
                         rover_locations[obj_name] = wp_name
            elif predicate == "full":
                if len(args) == 1: store_is_full.add(args[0])
            elif predicate == "have_soil_analysis":
                if len(args) == 2: have_soil_analysis_facts.add(tuple(args))
            elif predicate == "have_rock_analysis":
                 if len(args) == 2: have_rock_analysis_facts.add(tuple(args))
            elif predicate == "have_image":
                 if len(args) == 3: have_image_facts.add(tuple(args))
            elif predicate == "calibrated":
                 if len(args) == 2: calibrated_facts.add(tuple(args))
            elif predicate == "at_soil_sample":
                 if len(args) == 1: current_soil_samples.add(args[0])
            elif predicate == "at_rock_sample":
                 if len(args) == 1: current_rock_samples.add(args[0])

        # --- Calculate Heuristic ---

        # Iterate through goal conditions
        for goal_fact_str in self.goals:
            if goal_fact_str in state:
                continue # Goal already met

            parts = get_parts(goal_fact_str)
            if not parts:
                 unreachable = True # Malformed goal?
                 break

            predicate = parts[0]
            args = parts[1:]

            if predicate == "communicated_soil_data":
                if len(args) != 1: continue # Skip malformed goal
                waypoint_w = args[0]

                # Check if the sample data is already collected by any rover
                has_sample = any((r, waypoint_w) in have_soil_analysis_facts for r in self.rover_capabilities if "soil" in self.rover_capabilities.get(r, set()))

                if not has_sample:
                    # Cost to sample
                    h += 1 # sample_soil action

                    # Check if sample is still available at the waypoint
                    if waypoint_w not in current_soil_samples:
                         unreachable = True
                         break # Cannot sample if sample is gone

                    # Find closest soil-equipped rover to waypoint_w
                    soil_rovers = [r for r, caps in self.rover_capabilities.items() if "soil" in caps]
                    min_nav_to_sample = math.inf
                    best_rover_for_sample = None

                    for rover in soil_rovers:
                        if rover in rover_locations and rover in self.rover_shortest_paths:
                            dist = self.rover_shortest_paths[rover][rover_locations[rover]].get(waypoint_w, math.inf)
                            if dist < min_nav_to_sample:
                                min_nav_to_sample = dist
                                best_rover_for_sample = rover

                    if min_nav_to_sample == math.inf:
                        unreachable = True
                        break # No soil rover can reach sample waypoint

                    h += min_nav_to_sample # Navigation to sample site

                    # Check if store needs dropping for the chosen rover
                    if best_rover_for_sample and best_rover_for_sample in self.rover_store:
                        store = self.rover_store[best_rover_for_sample]
                        if store in store_is_full:
                            h += 1 # drop action

                    # After sampling, the rover is at waypoint_w
                    rover_location_after_sample = waypoint_w
                    potential_samplers = [best_rover_for_sample] if best_rover_for_sample else [] # Assume the best one does it

                else: # Sample data already exists
                    # Find rovers that have the sample
                    potential_samplers = [r for r, w in have_soil_analysis_facts if w == waypoint_w]
                     # Use their current location for communication navigation
                    rover_location_after_sample = None # Flag to use current location


                # Cost to communicate
                h += 1 # communicate_soil_data action

                min_nav_to_lander = math.inf
                for rover in potential_samplers:
                    if rover in rover_locations and rover in self.rover_shortest_paths:
                        current_loc = rover_location_after_sample if not has_sample else rover_locations[rover]
                        if current_loc in self.rover_shortest_paths[rover]:
                            min_dist_from_current_to_lander_visible = math.inf
                            if self.lander_visible_waypoints:
                                for lander_wp in self.lander_visible_waypoints:
                                    dist = self.rover_shortest_paths[rover][current_loc].get(lander_wp, math.inf)
                                    min_dist_from_current_to_lander_visible = min(min_dist_from_current_to_lander_visible, dist)
                            else: # No lander visible waypoints
                                min_dist_from_current_to_lander_visible = math.inf # Cannot communicate

                            min_nav_to_lander = min(min_nav_to_lander, min_dist_from_current_to_lander_visible)

                if min_nav_to_lander == math.inf:
                    unreachable = True
                    break # No rover with sample can reach lander visible waypoint

                h += min_nav_to_lander # Navigation to communication site


            elif predicate == "communicated_rock_data":
                if len(args) != 1: continue # Skip malformed goal
                waypoint_w = args[0]

                # Check if the sample data is already collected by any rover
                has_sample = any((r, waypoint_w) in have_rock_analysis_facts for r in self.rover_capabilities if "rock" in self.rover_capabilities.get(r, set()))

                if not has_sample:
                    # Cost to sample
                    h += 1 # sample_rock action

                    # Check if sample is still available at the waypoint
                    if waypoint_w not in current_rock_samples:
                         unreachable = True
                         break # Cannot sample if sample is gone

                    # Find closest rock-equipped rover to waypoint_w
                    rock_rovers = [r for r, caps in self.rover_capabilities.items() if "rock" in caps]
                    min_nav_to_sample = math.inf
                    best_rover_for_sample = None

                    for rover in rock_rovers:
                        if rover in rover_locations and rover in self.rover_shortest_paths:
                            dist = self.rover_shortest_paths[rover][rover_locations[rover]].get(waypoint_w, math.inf)
                            if dist < min_nav_to_sample:
                                min_nav_to_sample = dist
                                best_rover_for_sample = rover

                    if min_nav_to_sample == math.inf:
                        unreachable = True
                        break # No rock rover can reach sample waypoint

                    h += min_nav_to_sample # Navigation to sample site

                    # Check if store needs dropping for the chosen rover
                    if best_rover_for_sample and best_rover_for_sample in self.rover_store:
                        store = self.rover_store[best_rover_for_sample]
                        if store in store_is_full:
                            h += 1 # drop action

                    # After sampling, the rover is at waypoint_w
                    rover_location_after_sample = waypoint_w
                    potential_samplers = [best_rover_for_sample] if best_rover_for_sample else [] # Assume the best one does it

                else: # Sample data already exists
                    # Find rovers that have the sample
                    potential_samplers = [r for r, w in have_rock_analysis_facts if w == waypoint_w]
                     # Use their current location for communication navigation
                    rover_location_after_sample = None # Flag to use current location


                # Cost to communicate
                h += 1 # communicate_rock_data action

                min_nav_to_lander = math.inf
                for rover in potential_samplers:
                    if rover in rover_locations and rover in self.rover_shortest_paths:
                        current_loc = rover_location_after_sample if not has_sample else rover_locations[rover]
                        if current_loc in self.rover_shortest_paths[rover]:
                            min_dist_from_current_to_lander_visible = math.inf
                            if self.lander_visible_waypoints:
                                for lander_wp in self.lander_visible_waypoints:
                                    dist = self.rover_shortest_paths[rover][current_loc].get(lander_wp, math.inf)
                                    min_dist_from_current_to_lander_visible = min(min_dist_from_current_to_lander_visible, dist)
                            else: # No lander visible waypoints
                                min_dist_from_current_to_lander_visible = math.inf # Cannot communicate

                            min_nav_to_lander = min(min_nav_to_lander, min_dist_from_current_to_lander_visible)

                if min_nav_to_lander == math.inf:
                    unreachable = True
                    break # No rover with sample can reach lander visible waypoint

                h += min_nav_to_lander # Navigation to communication site


            elif predicate == "communicated_image_data":
                if len(args) != 2: continue # Skip malformed goal
                objective_o, mode_m = args

                # Check if the image data is already collected by any rover
                has_image = any((r, objective_o, mode_m) in have_image_facts for r in self.rover_capabilities if "imaging" in self.rover_capabilities.get(r, set()))

                if not has_image:
                    # Cost to take image
                    h += 1 # take_image action

                    # Find imaging-equipped rovers with cameras supporting mode_m
                    imaging_rovers = [r for r, caps in self.rover_capabilities.items() if "imaging" in caps]
                    suitable_cameras = [i for i, modes in self.camera_modes.items() if mode_m in modes]
                    suitable_rover_camera_pairs = [(self.camera_rover[i], i) for i in suitable_cameras if i in self.camera_rover and self.camera_rover[i] in imaging_rovers]

                    if not suitable_rover_camera_pairs:
                         unreachable = True
                         break # No suitable camera/rover combination

                    # Find waypoints visible from objective_o
                    image_waypoints = self.objective_visibility.get(objective_o, set())
                    if not image_waypoints:
                         unreachable = True
                         break # Cannot see objective from anywhere

                    # Find closest suitable rover to any image waypoint
                    min_nav_to_image_site = math.inf
                    best_rover_for_image = None
                    best_camera_for_image = None
                    best_image_waypoint = None

                    for rover, camera in suitable_rover_camera_pairs:
                        if rover in rover_locations and rover in self.rover_shortest_paths:
                            for img_wp in image_waypoints:
                                dist = self.rover_shortest_paths[rover][rover_locations[rover]].get(img_wp, math.inf)
                                if dist < min_nav_to_image_site:
                                    min_nav_to_image_site = dist
                                    best_rover_for_image = rover
                                    best_camera_for_image = camera
                                    best_image_waypoint = img_wp

                    if min_nav_to_image_site == math.inf:
                        unreachable = True
                        break # No suitable rover can reach an image waypoint

                    h += min_nav_to_image_site # Navigation to image site

                    # Cost for calibration (if needed)
                    # Assume calibration is needed if the specific camera on the specific rover is not calibrated
                    if (best_camera_for_image, best_rover_for_image) not in calibrated_facts:
                        h += 1 # calibrate action

                        # Find calibration target for the camera
                        calib_target = self.camera_calib_target.get(best_camera_for_image)
                        if not calib_target:
                             unreachable = True
                             break # Camera has no calibration target

                        # Find waypoints visible from calibration target
                        calib_waypoints = self.calibration_target_visibility.get(calib_target, set())
                        if not calib_waypoints:
                             unreachable = True
                             break # Calibration target not visible from anywhere

                        # Find closest calibration waypoint to the image waypoint (where rover will be after nav)
                        min_nav_to_calib = math.inf
                        if best_rover_for_image in self.rover_shortest_paths and best_image_waypoint in self.rover_shortest_paths[best_rover_for_image]:
                            for cal_wp in calib_waypoints:
                                dist = self.rover_shortest_paths[best_rover_for_image][best_image_waypoint].get(cal_wp, math.inf)
                                min_nav_to_calib = min(min_nav_to_calib, dist)

                        if min_nav_to_calib == math.inf:
                            unreachable = True
                            break # Cannot reach calibration waypoint from image waypoint

                        h += min_nav_to_calib # Navigation to calibration site

                    # After taking image, the rover is at best_image_waypoint
                    rover_location_after_image = best_image_waypoint
                    potential_imagers = [best_rover_for_image] if best_rover_for_image else [] # Assume the best one does it

                else: # Image data already exists
                    # Find rovers that have the image
                    potential_imagers = [r for r, o, m_ in have_image_facts if o == objective_o and m_ == mode_m]
                    # Use their current location for communication navigation
                    rover_location_after_image = None # Flag to use current location


                # Cost to communicate
                h += 1 # communicate_image_data action

                min_nav_to_lander = math.inf
                # Iterate over just rovers in potential_imagers
                for rover in potential_imagers:
                    if rover in rover_locations and rover in self.rover_shortest_paths:
                        current_loc = rover_location_after_image if not has_image else rover_locations[rover]
                        if current_loc in self.rover_shortest_paths[rover]:
                            min_dist_from_current_to_lander_visible = math.inf
                            if self.lander_visible_waypoints:
                                for lander_wp in self.lander_visible_waypoints:
                                    dist = self.rover_shortest_paths[rover][current_loc].get(lander_wp, math.inf)
                                    min_dist_from_current_to_lander_visible = min(min_dist_from_current_to_lander_visible, dist)
                            else: # No lander visible waypoints
                                min_dist_from_current_to_lander_visible = math.inf # Cannot communicate

                            min_nav_to_lander = min(min_nav_to_lander, min_dist_from_current_to_lander_visible)

                if min_nav_to_lander == math.inf:
                    unreachable = True
                    break # No rover with image can reach lander visible waypoint

                h += min_nav_to_lander # Navigation to communication site

            # Add other goal types if any, though rovers domain only has these three communicated types

        if unreachable:
            return math.inf
        else:
            return h
