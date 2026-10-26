import collections
from fnmatch import fnmatch

# Helper functions to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure we don't go out of bounds or have mismatching lengths
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# BFS helper functions for navigation costs
def get_rover_distance(rover_graph, start_waypoint, end_waypoint):
    """
    Performs BFS on a rover's graph to find the shortest distance.
    Returns distance or float('inf').
    """
    if start_waypoint == end_waypoint:
        return 0

    queue = collections.deque([(start_waypoint, 0)])
    visited = {start_waypoint}

    while queue:
        current_waypoint, distance = queue.popleft()

        # Check if current_waypoint exists in the graph before accessing neighbors
        if current_waypoint in rover_graph:
            for neighbor in rover_graph[current_waypoint]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    if neighbor == end_waypoint:
                        return distance + 1
                    queue.append((neighbor, distance + 1))

    return float('inf')

def get_rover_min_distance_to_set(rover_graph, start_waypoint, target_waypoints):
    """
    Performs BFS on a rover's graph to find the shortest distance to any target waypoint.
    Returns minimum distance or float('inf').
    """
    if not target_waypoints: # Cannot reach any target if set is empty
        return float('inf')
    if start_waypoint in target_waypoints:
        return 0

    queue = collections.deque([(start_waypoint, 0)])
    visited = {start_waypoint}

    while queue:
        current_waypoint, distance = queue.popleft()

        # Check if current_waypoint exists in the graph before accessing neighbors
        if current_waypoint in rover_graph:
            for neighbor in rover_graph[current_waypoint]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    if neighbor in target_waypoints:
                        return distance + 1
                    queue.append((neighbor, distance + 1))

    return float('inf')

# Define the heuristic class (assuming Heuristic base class is available in the environment)
# class roversHeuristic(Heuristic):
class roversHeuristic: # Defined without base class as per instruction
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve all
    uncommunicated goals (soil data, rock data, image data). It calculates
    the estimated cost for each unachieved goal independently and sums these costs.
    For each goal, it finds the minimum cost sequence of actions (navigation,
    sampling/imaging, communication) considering all capable rovers and relevant
    locations (sample sites, imaging/calibration waypoints, communication points).
    Navigation costs are estimated using shortest paths (BFS) on the rover's
    traverse graph.

    # Assumptions
    - The problem is solvable.
    - Navigation is possible between connected waypoints for a given rover.
    - Communication is possible from any waypoint visible from the lander's location.
    - A rover needs an empty store to sample soil or rock, or must drop its current sample first.
    - Taking an image requires a calibrated camera; the action uncalibrates it.
    - The heuristic simplifies interactions between goals (e.g., assumes sampling/imaging/communication for different goals can happen in parallel or without interfering with each other's location/resource needs beyond what's explicitly modeled in the action costs).
    - The 'visible' predicate is symmetric for communication purposes (if waypoint A is visible from B, B is visible from A). The domain only defines `(visible ?y ?z)` and `(at ?r ?x), (at_lander ?l ?y), (visible ?x ?y)` for communication. We assume `(visible ?x ?y)` means `?x` is visible *from* `?y`.
    - The 'can_traverse' predicate is symmetric based on example instances.

    # Heuristic Initialization
    The heuristic precomputes static information from the task definition:
    - The lander's location.
    - The set of waypoints visible from the lander's location (communication points).
    - The navigation graph (adjacency list) for each rover based on `can_traverse` facts.
    - Rover capabilities (equipped for soil, rock, imaging).
    - Store ownership (`store_of`).
    - Camera information (on which rover, supported modes, calibration target).
    - Waypoints visible from each objective (`visible_from`).
    - The set of goal facts.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current location of each rover, the status of samples at waypoints,
       the status of rover stores, existing analysis data, calibrated cameras,
       existing image data, and already communicated data.
    2. Identify the set of goal facts that have not yet been achieved.
    3. Initialize the total heuristic cost to 0.
    4. For each unachieved `(communicated_soil_data ?w)` goal:
       a. Check if any rover already has `(have_soil_analysis ?r' ?w)`.
       b. If yes, the cost for this goal is the minimum cost for such a rover `r'`
          to navigate from its current location to any communication waypoint plus 1 (for the communicate action).
       c. If no, the cost for this goal is the minimum cost over all rovers `?r`
          equipped for soil analysis:
          - Cost = (1 if `?r`'s store is full and needs dropping)
                 + (Cost to navigate `?r` from its current location to `?w`)
                 + 1 (for `sample_soil` action)
                 + (Cost to navigate `?r` from `?w` to any communication waypoint)
                 + 1 (for `communicate_soil_data` action).
          - This cost is only considered if `(at_soil_sample ?w)` is true in the current state.
       d. Add the minimum calculated cost for this goal (over cases b and c, and over suitable rovers) to the total heuristic cost. If the goal is unreachable, add infinity.
    5. For each unachieved `(communicated_rock_data ?w)` goal:
       a. Follow a similar process as for soil data, using rock-specific predicates and capabilities.
    6. For each unachieved `(communicated_image_data ?o ?m)` goal:
       a. Check if any rover already has `(have_image ?r' ?o ?m)`.
       b. If yes, the cost for this goal is the minimum cost for such a rover `r'`
          to navigate from its current location to any communication waypoint plus 1 (for the communicate action).
       c. If no, the cost for this goal is the minimum cost over all rovers `?r`
          equipped for imaging with a camera `?i` supporting mode `?m`:
          - Find a calibration target `?t` for camera `?i`.
          - Find waypoints `W_cal` visible from `?t` and `W_img` visible from `?o`.
          - Cost = minimum over all `w_cal` in `W_cal` and `w_img` in `W_img` of:
            (Cost to navigate `?r` from its current location to `w_cal`)
            + 1 (for `calibrate` action)
            + (Cost to navigate `?r` from `w_cal` to `w_img`)
            + 1 (for `take_image` action)
            + (Cost to navigate `?r` from `w_img` to any communication waypoint)
            + 1 (for `communicate_image_data` action).
          - This calculation assumes recalibration is needed for each image goal requiring this camera/rover, which is a simplification.
       d. Add the minimum calculated cost for this goal (over cases b and c, and over suitable rovers/cameras/waypoints) to the total heuristic cost. If the goal is unreachable, add infinity.
    7. Return the total heuristic cost. If the total cost is infinity, the problem is likely unsolvable from this state.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        # The set of facts that must hold in goal states.
        self.goals = task.goals
        # Static facts are not affected by actions.
        static_facts = task.static

        # Precompute static information

        # Lander location
        self.lander_location = None
        for fact in static_facts:
            if match(fact, "at_lander", "*", "*"):
                self.lander_location = get_parts(fact)[2]
                break # Assuming only one lander

        # Communication waypoints (visible from lander location)
        self.communication_waypoints = set()
        if self.lander_location:
            for fact in static_facts:
                # Rover at ?x, Lander at ?y, visible ?x ?y
                if match(fact, "visible", "*", self.lander_location):
                    self.communication_waypoints.add(get_parts(fact)[1])
                # Assuming visible is symmetric based on examples
                if match(fact, "visible", self.lander_location, "*"):
                     self.communication_waypoints.add(get_parts(fact)[2])


        # Rover navigation graphs
        self.rover_graphs = collections.defaultdict(lambda: collections.defaultdict(list))
        for fact in static_facts:
            if match(fact, "can_traverse", "*", "*", "*"):
                _, rover, wp_from, wp_to = get_parts(fact)
                self.rover_graphs[rover][wp_from].append(wp_to)
                # Assuming can_traverse is symmetric based on examples
                self.rover_graphs[rover][wp_to].append(wp_from)

        # Rover capabilities
        self.rover_capabilities = collections.defaultdict(lambda: {'soil': False, 'rock': False, 'imaging': False})
        for fact in static_facts:
            if match(fact, "equipped_for_soil_analysis", "*"):
                self.rover_capabilities[get_parts(fact)[1]]['soil'] = True
            elif match(fact, "equipped_for_rock_analysis", "*"):
                self.rover_capabilities[get_parts(fact)[1]]['rock'] = True
            elif match(fact, "equipped_for_imaging", "*"):
                self.rover_capabilities[get_parts(fact)[1]]['imaging'] = True

        # Store ownership
        self.store_to_rover = {}
        for fact in static_facts:
            if match(fact, "store_of", "*", "*"):
                self.store_to_rover[get_parts(fact)[1]] = get_parts(fact)[2]

        # Camera information
        self.camera_info = collections.defaultdict(lambda: {'on_board': None, 'supports': set(), 'calibration_target': None})
        for fact in static_facts:
            if match(fact, "on_board", "*", "*"):
                self.camera_info[get_parts(fact)[1]]['on_board'] = get_parts(fact)[2] # rover
            elif match(fact, "supports", "*", "*"):
                self.camera_info[get_parts(fact)[1]]['supports'].add(get_parts(fact)[2]) # mode
            elif match(fact, "calibration_target", "*", "*"):
                self.camera_info[get_parts(fact)[1]]['calibration_target'] = get_parts(fact)[2] # objective

        # Waypoints visible from objectives
        self.objective_visible_from = collections.defaultdict(set)
        for fact in static_facts:
            if match(fact, "visible_from", "*", "*"):
                self.objective_visible_from[get_parts(fact)[1]].add(get_parts(fact)[2]) # waypoint


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Parse current state information
        current_rover_locations = {}
        current_soil_samples = set()
        current_rock_samples = set()
        current_store_status = collections.defaultdict(lambda: {'full': False, 'empty': False})
        current_soil_analysis = collections.defaultdict(set) # rover -> {waypoints}
        current_rock_analysis = collections.defaultdict(set) # rover -> {waypoints}
        current_calibrated_cameras = collections.defaultdict(set) # rover -> {cameras}
        current_have_image = collections.defaultdict(lambda: collections.defaultdict(bool)) # rover -> objective -> mode -> bool
        current_communicated_soil = set() # waypoints
        current_communicated_rock = set() # waypoints
        current_communicated_image = collections.defaultdict(set) # objective -> {modes}

        for fact in state:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "at" and len(parts) == 3 and parts[1].startswith("rover"): # Check if it's a rover location
                current_rover_locations[parts[1]] = parts[2]
            elif pred == "at_soil_sample" and len(parts) == 2:
                current_soil_samples.add(parts[1])
            elif pred == "at_rock_sample" and len(parts) == 2:
                current_rock_samples.add(parts[1])
            elif pred == "full" and len(parts) == 2:
                current_store_status[parts[1]]['full'] = True
                current_store_status[parts[1]]['empty'] = False
            elif pred == "empty" and len(parts) == 2:
                current_store_status[parts[1]]['empty'] = True
                current_store_status[parts[1]]['full'] = False
            elif pred == "have_soil_analysis" and len(parts) == 3:
                current_soil_analysis[parts[1]].add(parts[2])
            elif pred == "have_rock_analysis" and len(parts) == 3:
                current_rock_analysis[parts[1]].add(parts[2])
            elif pred == "calibrated" and len(parts) == 3:
                current_calibrated_cameras[parts[2]].add(parts[1]) # rover -> camera
            elif pred == "have_image" and len(parts) == 4:
                current_have_image[parts[1]][parts[2]][parts[3]] = True # rover -> objective -> mode
            elif pred == "communicated_soil_data" and len(parts) == 2:
                current_communicated_soil.add(parts[1])
            elif pred == "communicated_rock_data" and len(parts) == 2:
                current_communicated_rock.add(parts[1])
            elif pred == "communicated_image_data" and len(parts) == 3:
                current_communicated_image[parts[1]].add(parts[2]) # objective -> mode

        total_cost = 0

        # Group goals by type
        soil_goals = set()
        rock_goals = set()
        image_goals = set()

        for goal in self.goals:
            parts = get_parts(goal)
            pred = parts[0]
            if pred == "communicated_soil_data" and len(parts) == 2:
                soil_goals.add(parts[1]) # waypoint
            elif pred == "communicated_rock_data" and len(parts) == 2:
                rock_goals.add(parts[1]) # waypoint
            elif pred == "communicated_image_data" and len(parts) == 3:
                image_goals.add((parts[1], parts[2])) # (objective, mode)

        # Process Soil Goals
        for waypoint in soil_goals:
            if waypoint in current_communicated_soil:
                continue # Already achieved

            min_goal_cost = float('inf')

            # Case 1: Analysis data already exists for some rover
            # Find min cost to communicate existing data
            for rover, waypoints_analyzed in current_soil_analysis.items():
                if waypoint in waypoints_analyzed:
                    current_loc = current_rover_locations.get(rover)
                    if current_loc is None or rover not in self.rover_graphs:
                         continue # Rover location unknown or cannot move

                    min_comm_nav_cost = get_rover_min_distance_to_set(
                        self.rover_graphs[rover],
                        current_loc,
                        self.communication_waypoints
                    )
                    if min_comm_nav_cost != float('inf'):
                         min_goal_cost = min(min_goal_cost, min_comm_nav_cost + 1) # +1 for communicate action


            # Case 2: Need to sample and communicate
            if waypoint in current_soil_samples: # Can only sample if sample is still there
                for rover, capabilities in self.rover_capabilities.items():
                    if capabilities['soil']:
                        # Find the store for this rover
                        store = None
                        for s, r in self.store_to_rover.items():
                            if r == rover:
                                store = s
                                break
                        if store is None: continue # Rover has no store? (Shouldn't happen in valid problems)

                        drop_cost = 0
                        # Check if the store is full in the current state
                        if current_store_status.get(store, {}).get('full', False):
                             drop_cost = 1 # Need to drop first

                        # Calculate navigation costs
                        current_loc = current_rover_locations.get(rover)
                        if current_loc is None or rover not in self.rover_graphs:
                             continue # Rover location unknown or cannot move

                        nav_to_sample = get_rover_distance(self.rover_graphs[rover], current_loc, waypoint)

                        if nav_to_sample == float('inf'): continue # Cannot reach sample location

                        # Cost from sample location to communication point
                        # Need to be able to navigate *from* the sample point
                        if waypoint not in self.rover_graphs.get(rover, {}): continue

                        nav_to_comm = get_rover_min_distance_to_set(
                            self.rover_graphs[rover],
                            waypoint,
                            self.communication_waypoints
                        )

                        if nav_to_comm == float('inf'): continue # Cannot reach communication point from sample location

                        # Total cost for this rover: drop (if needed) + nav to sample + sample + nav to comm + communicate
                        cost = drop_cost + nav_to_sample + 1 + nav_to_comm + 1
                        min_goal_cost = min(min_goal_cost, cost)

            if min_goal_cost != float('inf'):
                total_cost += min_goal_cost
            else:
                 total_cost = float('inf') # If any goal is unreachable, the total is unreachable.
                 break # No need to process other goals if one is impossible


        # Process Rock Goals (Similar to Soil Goals)
        if total_cost != float('inf'): # Only process if previous goals were reachable
            for waypoint in rock_goals:
                if waypoint in current_communicated_rock:
                    continue

                min_goal_cost = float('inf')

                # Case 1: Analysis data already exists
                for rover, waypoints_analyzed in current_rock_analysis.items():
                    if waypoint in waypoints_analyzed:
                        current_loc = current_rover_locations.get(rover)
                        if current_loc is None or rover not in self.rover_graphs:
                             continue

                        min_comm_nav_cost = get_rover_min_distance_to_set(
                            self.rover_graphs[rover],
                            current_loc,
                            self.communication_waypoints
                        )
                        if min_comm_nav_cost != float('inf'):
                            min_goal_cost = min(min_goal_cost, min_comm_nav_cost + 1)


                # Case 2: Need to sample and communicate
                if waypoint in current_rock_samples:
                    for rover, capabilities in self.rover_capabilities.items():
                        if capabilities['rock']:
                            store = None
                            for s, r in self.store_to_rover.items():
                                if r == rover:
                                    store = s
                                    break
                            if store is None: continue

                            drop_cost = 0
                            if current_store_status.get(store, {}).get('full', False):
                                drop_cost = 1

                            current_loc = current_rover_locations.get(rover)
                            if current_loc is None or rover not in self.rover_graphs:
                                 continue

                            nav_to_sample = get_rover_distance(self.rover_graphs[rover], current_loc, waypoint)

                            if nav_to_sample == float('inf'): continue

                            if waypoint not in self.rover_graphs.get(rover, {}): continue

                            nav_to_comm = get_rover_min_distance_to_set(
                                self.rover_graphs[rover],
                                waypoint,
                                self.communication_waypoints
                            )

                            if nav_to_comm == float('inf'): continue

                            cost = drop_cost + nav_to_sample + 1 + nav_to_comm + 1
                            min_goal_cost = min(min_goal_cost, cost)

                if min_goal_cost != float('inf'):
                    total_cost += min_goal_cost
                else:
                    total_cost = float('inf')
                    break # No need to process other goals


        # Process Image Goals
        if total_cost != float('inf'): # Only process if previous goals were reachable
            for objective, mode in image_goals:
                if mode in current_communicated_image.get(objective, set()):
                    continue # Already achieved

                min_goal_cost = float('inf')

                # Case 1: Image already exists for some rover
                # Find min cost to communicate existing image
                # Iterate through rovers that *might* have the image
                for rover in current_have_image:
                    if current_have_image[rover].get(objective, {}).get(mode, False):
                         # Found a rover with the image, calculate cost to communicate
                         current_loc = current_rover_locations.get(rover)
                         if current_loc is None or rover not in self.rover_graphs:
                              continue

                         min_comm_nav_cost = get_rover_min_distance_to_set(
                             self.rover_graphs[rover],
                             current_loc,
                             self.communication_waypoints
                         )
                         if min_comm_nav_cost != float('inf'):
                             min_goal_cost = min(min_goal_cost, min_comm_nav_cost + 1) # +1 for communicate


                # Case 2: Need to take image and communicate
                # Find suitable rovers and cameras
                suitable_rovers_cameras = [] # List of (rover, camera)
                for camera, info in self.camera_info.items():
                    rover = info['on_board']
                    if rover and self.rover_capabilities.get(rover, {}).get('imaging', False) and mode in info['supports']:
                        suitable_rovers_cameras.append((rover, camera))

                if suitable_rovers_cameras:
                    # Find potential imaging waypoints for this objective
                    imaging_waypoints = self.objective_visible_from.get(objective, set())
                    if not imaging_waypoints: continue # Cannot image this objective

                    for rover, camera in suitable_rovers_cameras:
                        cal_target = self.camera_info[camera]['calibration_target']
                        if not cal_target: continue # Camera has no calibration target

                        # Find potential calibration waypoints for this camera's target
                        calibration_waypoints = self.objective_visible_from.get(cal_target, set())
                        if not calibration_waypoints: continue # Cannot calibrate this camera

                        current_loc = current_rover_locations.get(rover)
                        if current_loc is None or rover not in self.rover_graphs:
                             continue # Rover location unknown or cannot move

                        # Find min cost path through calibration and imaging waypoints
                        min_path_cost_for_rover_camera = float('inf')

                        # Iterate through all pairs of (cal_wp, img_wp)
                        for w_cal in calibration_waypoints:
                            nav1 = get_rover_distance(self.rover_graphs[rover], current_loc, w_cal)
                            if nav1 == float('inf'): continue

                            # Need to be able to navigate *from* the calibration point
                            if w_cal not in self.rover_graphs.get(rover, {}): continue

                            for w_img in imaging_waypoints:
                                nav2 = get_rover_distance(self.rover_graphs[rover], w_cal, w_img)
                                if nav2 == float('inf'): continue

                                # Need to be able to navigate *from* the imaging point
                                if w_img not in self.rover_graphs.get(rover, {}): continue

                                nav3 = get_rover_min_distance_to_set(
                                    self.rover_graphs[rover],
                                    w_img,
                                    self.communication_waypoints
                                )
                                if nav3 == float('inf'): continue

                                # Total cost for this path: nav1 + calibrate + nav2 + take_image + nav3 + communicate
                                path_cost = nav1 + 1 + nav2 + 1 + nav3 + 1
                                min_path_cost_for_rover_camera = min(min_path_cost_for_rover_camera, path_cost)

                        if min_path_cost_for_rover_camera != float('inf'):
                             min_goal_cost = min(min_goal_cost, min_path_cost_for_rover_camera)

                if min_goal_cost != float('inf'):
                    total_cost += min_goal_cost
                else:
                    total_cost = float('inf')
                    break # No need to process other goals


        # Ensure heuristic is 0 only at the goal state
        # If total_cost is 0, it means all unachieved goals had a cost of 0,
        # which implies they were already achieved.
        # If the state is the goal state, all goals are communicated,
        # the loops add 0, and the result is 0.
        # If total_cost is inf, it means at least one goal is unreachable.

        return total_cost
