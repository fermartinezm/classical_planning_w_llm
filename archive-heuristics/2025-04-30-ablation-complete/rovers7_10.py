from fnmatch import fnmatch
from collections import deque, defaultdict
# Assuming heuristics.heuristic_base.Heuristic is available
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact."""
    # Handle potential leading/trailing whitespace and ensure it's a string
    fact_str = str(fact).strip()
    if not fact_str.startswith('(') or not fact_str.endswith(')'):
         # Handle facts that might not be in the standard (pred arg1 arg2) format
         # For this domain, they seem to be, but good practice.
         # If it's just a predicate name, return [predicate_name]
         return [fact_str]
    return fact_str[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    Wildcards `*` allowed in args.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve all goal
    conditions. It sums the estimated cost for each unachieved goal fact.
    The cost for a goal fact depends on whether necessary intermediate conditions
    (like having a sample or an image) are met, and includes estimated navigation
    costs based on precomputed shortest paths between waypoints.

    # Assumptions
    - All rovers can traverse any visible path between waypoints.
    - Navigation costs 1 action per step.
    - Sampling/Imaging/Communication actions cost 1 action.
    - Calibration costs 1 action.
    - The heuristic assumes that a suitable equipped rover/camera is available
      for each task if needed, and ignores resource contention (like store capacity
      or camera calibration state beyond the `have_image` check).
    - It assumes sample predicates (`at_soil_sample`, `at_rock_sample`) are
      only removed when a sample is successfully taken.
    - It assumes solvable instances.

    # Heuristic Initialization
    - Extracts goal conditions.
    - Extracts static facts to build:
        - Waypoint graph based on `visible` links.
        - All-pairs shortest paths between waypoints using BFS.
        - Lander location(s) and communication waypoints (visible from lander).
        - Mapping from objectives to waypoints visible from them.
        - Mapping from cameras to their calibration targets and corresponding calibration waypoints.
        - Rover capabilities, cameras on board, camera supported modes.
        - All waypoints.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize total heuristic cost to 0.
    2. Parse the current state to get the status of all relevant predicates
       (rover locations, collected samples/images, communicated data).
    3. For each goal fact in the task's goals:
        a. If the goal fact is already true in the current state, add 0 cost for this goal.
        b. If the goal is `(communicated_soil_data ?w)`:
            - Check if `(have_soil_analysis ?r ?w)` is true for any rover `?r`.
            - If yes: Find *any* such rover `?r`. Get its location `?rover_loc`. Calculate the minimum navigation distance from `?rover_loc` to any waypoint visible from the lander (`comm_waypoint`). Add this distance + 1 (for the communicate action) to the total cost.
            - If no: Find *any* soil-equipped rover `?r`. Find its location `?rover_loc`. Calculate the distance from `?rover_loc` to `?w` (`sample_waypoint`). Calculate the minimum distance from `?w` to any `comm_waypoint`. Add (distance to sample + 1 [sample action] + distance from sample to comm + 1 [communicate action]) to the total cost. Assign a high cost if no suitable rover exists or waypoints are unreachable.
        c. If the goal is `(communicated_rock_data ?w)`: Follow the same logic as soil data, checking for `have_rock_analysis` and rock-equipped rovers.
        d. If the goal is `(communicated_image_data ?o ?m)`:
            - Check if `(have_image ?r ?o ?m)` is true for any rover `?r`.
            - If yes: Find *any* such rover `?r`. Get its location `?rover_loc`. Calculate the minimum navigation distance from `?rover_loc` to any `comm_waypoint`. Add this distance + 1 (for the communicate action) to the total cost.
            - If no: Find *any* imaging-equipped rover `?r` with a camera `?i` supporting mode `?m`. Find its location `?rover_loc`. Find a calibration waypoint `?cal_wp` for camera `?i` and an image waypoint `?img_wp` for objective `?o`. Calculate the minimum navigation distance from `?rover_loc` to any `cal_wp`. Add this distance + 1 (calibrate action). Then, calculate the minimum distance from *any* `cal_wp` to *any* `img_wp`. Add this distance + 1 (take_image action). Then, calculate the minimum distance from *any* `img_wp` to *any* `comm_waypoint`. Add this distance + 1 (communicate action). Sum these costs. Assign a high cost if no suitable rover/camera exists or waypoints are unreachable.
    4. Return the total calculated cost.

    Note: When calculating minimum distances to a set of waypoints (e.g., `comm_waypoints`), we find the minimum distance from the current location to *any* waypoint in the set.
    Note: When calculating costs for goals where the intermediate product (sample/image) is missing, we assume a suitable rover/camera exists and calculate the cost for one sequence of actions (move, sample/calibrate, move, image, move, communicate), using shortest paths for navigation segments. We don't explicitly model store usage or camera calibration state beyond the `have_analysis`/`have_image` check.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # Data structures to store extracted static information
        self.lander_location = None
        self.comm_waypoints = set()
        self.objective_image_wps = defaultdict(set)
        self.camera_cal_targets = {}
        self.camera_cal_wps = defaultdict(set)
        self.rover_capabilities = defaultdict(set)
        self.rover_cameras = defaultdict(set)
        self.camera_modes = defaultdict(set)
        self.all_waypoints = set()
        self.waypoint_graph = defaultdict(set)

        # 1. Extract static information
        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue # Skip empty facts

            predicate = parts[0]

            if predicate == "at_lander":
                # Assuming only one lander
                self.lander_location = parts[2]
            elif predicate == "visible":
                wp1, wp2 = parts[1], parts[2]
                self.waypoint_graph[wp1].add(wp2)
                self.waypoint_graph[wp2].add(wp1) # Assuming symmetric visibility implies symmetric traversability
                self.all_waypoints.add(wp1)
                self.all_waypoints.add(wp2)
            elif predicate == "equipped_for_soil_analysis":
                self.rover_capabilities[parts[1]].add("soil")
            elif predicate == "equipped_for_rock_analysis":
                self.rover_capabilities[parts[1]].add("rock")
            elif predicate == "equipped_for_imaging":
                self.rover_capabilities[parts[1]].add("imaging")
            elif predicate == "on_board":
                self.rover_cameras[parts[2]].add(parts[1]) # rover -> camera
            elif predicate == "supports":
                self.camera_modes[parts[1]].add(parts[2]) # camera -> mode
            elif predicate == "calibration_target":
                self.camera_cal_targets[parts[1]] = parts[2] # camera -> objective
            elif predicate == "visible_from":
                objective, waypoint = parts[1], parts[2]
                self.objective_image_wps[objective].add(waypoint)

        # Identify communication waypoints (visible from lander)
        if self.lander_location:
             for wp in self.all_waypoints:
                 # Check if wp is visible from lander location based on the graph
                 if self.lander_location in self.waypoint_graph.get(wp, set()):
                     self.comm_waypoints.add(wp)

        # Identify calibration waypoints for each camera
        for camera, target_objective in self.camera_cal_targets.items():
            if target_objective in self.objective_image_wps:
                 self.camera_cal_wps[camera].update(self.objective_image_wps[target_objective])

        # 2. Compute all-pairs shortest paths between waypoints
        self.distances = {}
        for start_wp in self.all_waypoints:
            self.distances[start_wp] = self._bfs(start_wp)

    def _bfs(self, start_node):
        """Performs BFS from a start node to find distances to all reachable nodes."""
        distances = {node: float('inf') for node in self.all_waypoints}
        if start_node not in distances:
             # Start node might not be in all_waypoints if graph is empty or disconnected
             return distances # Return empty/inf distances

        distances[start_node] = 0
        queue = deque([start_node])

        while queue:
            current_wp = queue.popleft()

            # Ensure current_wp is in the graph keys before iterating neighbors
            if current_wp in self.waypoint_graph:
                for neighbor_wp in self.waypoint_graph[current_wp]:
                    if distances[neighbor_wp] == float('inf'):
                        distances[neighbor_wp] = distances[current_wp] + 1
                        queue.append(neighbor_wp)
        return distances

    def _get_min_dist(self, start_wp, target_wps):
        """Calculates the minimum distance from start_wp to any waypoint in target_wps."""
        if start_wp not in self.distances or not target_wps:
             return float('inf') # Cannot start or no targets

        min_d = float('inf')
        for target_wp in target_wps:
            if target_wp in self.distances[start_wp]:
                min_d = min(min_d, self.distances[start_wp][target_wp])
        return min_d # Return inf if no path to any target

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Parse current state into easily queryable structures
        rover_locations = {}
        have_soil = defaultdict(set) # waypoint -> {rover}
        have_rock = defaultdict(set) # waypoint -> {rover}
        have_image = defaultdict(lambda: defaultdict(set)) # objective -> mode -> {rover}
        communicated_soil = set() # {waypoint}
        communicated_rock = set() # {waypoint}
        communicated_image = set() # {(objective, mode)}

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]

            if predicate == "at":
                obj, loc = parts[1], parts[2]
                # Assuming only rovers are dynamic objects with 'at' predicate relevant here
                # Check if obj is a rover by seeing if it has capabilities
                if obj in self.rover_capabilities:
                    rover_locations[obj] = loc
            elif predicate == "have_soil_analysis":
                rover, wp = parts[1], parts[2]
                have_soil[wp].add(rover)
            elif predicate == "have_rock_analysis":
                rover, wp = parts[1], parts[2]
                have_rock[wp].add(rover)
            elif predicate == "have_image":
                rover, obj, mode = parts[1], parts[2], parts[3]
                have_image[obj][mode].add(rover)
            elif predicate == "communicated_soil_data":
                communicated_soil.add(parts[1])
            elif predicate == "communicated_rock_data":
                communicated_rock.add(parts[1])
            elif predicate == "communicated_image_data":
                communicated_image.add((parts[1], parts[2]))


        total_cost = 0
        UNREACHABLE_COST = 1000 # Use a large value for unreachable goal parts

        # Iterate through goals and estimate cost for unachieved ones
        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue

            predicate = parts[0]

            if predicate == "communicated_soil_data":
                wp = parts[1]
                if wp in communicated_soil:
                    continue # Goal already achieved

                # Check if soil analysis is already done for this waypoint
                if wp in have_soil and have_soil[wp]:
                    # Need to communicate the data
                    # Find a rover that has the data
                    rover_with_data = next(iter(have_soil[wp])) # Just pick one
                    current_loc = rover_locations.get(rover_with_data) # Get its location

                    if current_loc is None:
                         # Rover location not found in state - should not happen in valid states
                         total_cost += UNREACHABLE_COST
                         continue

                    # Cost is move to comm waypoint + communicate action
                    min_dist_to_comm = self._get_min_dist(current_loc, self.comm_waypoints)
                    if min_dist_to_comm == float('inf'):
                         total_cost += UNREACHABLE_COST # Cannot reach any communication waypoint
                    else:
                         total_cost += min_dist_to_comm + 1 # 1 for communicate action

                else:
                    # Need to sample soil and then communicate
                    # Find a suitable rover: soil-equipped
                    suitable_rover = None
                    for rover, capabilities in self.rover_capabilities.items():
                        if "soil" in capabilities:
                            suitable_rover = rover
                            break # Found a suitable rover

                    if suitable_rover is None:
                         total_cost += UNREACHABLE_COST # Impossible goal part
                         continue

                    current_loc = rover_locations.get(suitable_rover)
                    if current_loc is None:
                         total_cost += UNREACHABLE_COST
                         continue

                    # Cost is move to sample waypoint + sample + move to comm waypoint + communicate
                    dist_to_sample = self._get_min_dist(current_loc, {wp}) # Distance to the specific sample waypoint
                    if dist_to_sample == float('inf'):
                         total_cost += UNREACHABLE_COST # Cannot reach sample waypoint
                         continue

                    # After sampling, the rover is conceptually at wp
                    min_dist_from_sample_to_comm = self._get_min_dist(wp, self.comm_waypoints)
                    if min_dist_from_sample_to_comm == float('inf'):
                         total_cost += UNREACHABLE_COST # Cannot reach comm waypoint from sample waypoint
                         continue

                    total_cost += dist_to_sample + 1 # 1 for sample action
                    total_cost += min_dist_from_sample_to_comm + 1 # 1 for communicate action


            elif predicate == "communicated_rock_data":
                wp = parts[1]
                if wp in communicated_rock:
                    continue # Goal already achieved

                # Check if rock analysis is already done for this waypoint
                if wp in have_rock and have_rock[wp]:
                    # Need to communicate the data
                    rover_with_data = next(iter(have_rock[wp])) # Pick one
                    current_loc = rover_locations.get(rover_with_data)
                    if current_loc is None:
                         total_cost += UNREACHABLE_COST
                         continue

                    min_dist_to_comm = self._get_min_dist(current_loc, self.comm_waypoints)
                    if min_dist_to_comm == float('inf'):
                         total_cost += UNREACHABLE_COST
                    else:
                         total_cost += min_dist_to_comm + 1 # communicate action

                else:
                    # Need to sample rock and then communicate
                    # Find a suitable rover: rock-equipped
                    suitable_rover = None
                    for rover, capabilities in self.rover_capabilities.items():
                        if "rock" in capabilities:
                            suitable_rover = rover
                            break # Found a suitable rover

                    if suitable_rover is None:
                         total_cost += UNREACHABLE_COST # Impossible goal part
                         continue

                    current_loc = rover_locations.get(suitable_rover)
                    if current_loc is None:
                         total_cost += UNREACHABLE_COST
                         continue

                    dist_to_sample = self._get_min_dist(current_loc, {wp})
                    if dist_to_sample == float('inf'):
                         total_cost += UNREACHABLE_COST
                         continue

                    min_dist_from_sample_to_comm = self._get_min_dist(wp, self.comm_waypoints)
                    if min_dist_from_sample_to_comm == float('inf'):
                         total_cost += UNREACHABLE_COST
                         continue

                    total_cost += dist_to_sample + 1 # sample action
                    total_cost += min_dist_from_sample_to_comm + 1 # communicate action


            elif predicate == "communicated_image_data":
                obj, mode = parts[1], parts[2]
                if (obj, mode) in communicated_image:
                    continue # Goal already achieved

                # Check if image is already taken
                if obj in have_image and mode in have_image[obj] and have_image[obj][mode]:
                    # Need to communicate the image
                    rover_with_image = next(iter(have_image[obj][mode])) # Pick one
                    current_loc = rover_locations.get(rover_with_image)
                    if current_loc is None:
                         total_cost += UNREACHABLE_COST
                         continue

                    min_dist_to_comm = self._get_min_dist(current_loc, self.comm_waypoints)
                    if min_dist_to_comm == float('inf'):
                         total_cost += UNREACHABLE_COST
                    else:
                         total_cost += min_dist_to_comm + 1 # communicate action

                else:
                    # Need to calibrate, take image, and then communicate
                    # Find a suitable rover: imaging-equipped with camera supporting mode
                    suitable_rover = None
                    suitable_camera = None
                    for rover, capabilities in self.rover_capabilities.items():
                        if "imaging" in capabilities:
                            for camera in self.rover_cameras.get(rover, []):
                                if mode in self.camera_modes.get(camera, []):
                                    suitable_rover = rover
                                    suitable_camera = camera
                                    break # Found a suitable rover/camera
                            if suitable_rover: break # Found a suitable rover

                    if suitable_rover is None or suitable_camera is None:
                         total_cost += UNREACHABLE_COST # Impossible goal part
                         continue

                    current_loc = rover_locations.get(suitable_rover)
                    if current_loc is None:
                         total_cost += UNREACHABLE_COST
                         continue

                    # Cost is move to cal + calibrate + move to image + take_image + move to comm + communicate
                    cal_wps = self.camera_cal_wps.get(suitable_camera, set())
                    if not cal_wps:
                         total_cost += UNREACHABLE_COST # No calibration waypoints for this camera
                         continue

                    min_dist_to_cal = self._get_min_dist(current_loc, cal_wps)
                    if min_dist_to_cal == float('inf'):
                         total_cost += UNREACHABLE_COST # Cannot reach calibration waypoint
                         continue

                    # After calibrating, the rover is conceptually at a cal_wp (the closest one)
                    # Calculate distance from *any* cal_wp to *any* img_wp
                    image_wps = self.objective_image_wps.get(obj, set())
                    if not image_wps:
                         total_cost += UNREACHABLE_COST # No image waypoints for this objective
                         continue

                    min_dist_cal_to_image = float('inf')
                    for cal_wp in cal_wps:
                         min_dist_cal_to_image = min(min_dist_cal_to_image, self._get_min_dist(cal_wp, image_wps))

                    if min_dist_cal_to_image == float('inf'):
                         total_cost += UNREACHABLE_COST # Cannot reach image waypoint from any cal waypoint
                         continue

                    # Calculate distance from *any* img_wp to *any* comm_wp
                    min_dist_image_to_comm = float('inf')
                    for img_wp in image_wps:
                         min_dist_image_to_comm = min(min_dist_image_to_comm, self._get_min_dist(img_wp, self.comm_waypoints))

                    if min_dist_image_to_comm == float('inf'):
                         total_cost += UNREACHABLE_COST # Cannot reach comm waypoint from any image waypoint
                         continue

                    total_cost += min_dist_to_cal + 1 # calibrate action
                    total_cost += min_dist_cal_to_image + 1 # take_image action
                    total_cost += min_dist_image_to_comm + 1 # communicate action

        # Heuristic is 0 only if all goal facts are communicated_data and they are all in the state.
        # If total_cost is 0, it implies all relevant communicated_data goals are met.
        # Assuming the task goal is solely these communicated_data facts, h=0 means goal reached.

        return total_cost
