# Add necessary imports
from fnmatch import fnmatch
from collections import deque, defaultdict
# from heuristics.heuristic_base import Heuristic # Assuming this is available in the environment

# Define a dummy Heuristic base class if not provided, for testing purposes
# In the actual environment, this should be removed.
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    # print("Warning: heuristics.heuristic_base not found. Using dummy base class.")
    class Heuristic:
        def __init__(self, task):
            pass
        def __call__(self, node):
            raise NotImplementedError

# Helper functions
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
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# BFS function
def bfs(graph, start_node):
    """
    Performs BFS from a start node to find shortest distances to all reachable nodes.
    Returns a dictionary {node: distance}.
    """
    distances = {node: float('inf') for node in graph}
    if start_node not in graph:
        # Start node is not in the graph, no paths possible from here
        return distances

    distances[start_node] = 0
    queue = deque([start_node])

    while queue:
        current_node = queue.popleft()

        if current_node in graph:
            for neighbor in graph[current_node]:
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)
    return distances

class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve the goal
    conditions. It sums up the estimated costs for each unachieved goal,
    considering the steps needed to collect data (sample/image), calibrate
    cameras, navigate to required locations (sample sites, image viewpoints,
    calibration targets, lander communication points), and finally communicate
    the data. Navigation costs are estimated using precomputed shortest paths
    on the waypoint graph traversable by rovers.

    # Assumptions
    - The problem instance is solvable.
    - Any rover can traverse any link defined by `can_traverse` for any rover.
      (A union graph of all traversable links is used for navigation).
    - Soil/rock samples exist at the waypoints specified in the goal conditions.
    - Calibration targets are visible from waypoints as specified in static facts.
    - Store capacity is ignored for sampling (assumes a store can be made empty).
    - The lander location and visible links from it are static.
    - The heuristic counts actions required for each goal independently, which
      can lead to overestimation (non-admissible) but aims to guide GBFS.

    # Heuristic Initialization
    - Extracts goal conditions.
    - Builds a waypoint graph based on `can_traverse` facts for any rover.
    - Computes all-pairs shortest paths on this graph using BFS.
    - Identifies the lander's location and waypoints visible from it (communication points).
    - Stores static information about rover capabilities, camera properties,
      objective visibility, and calibration targets.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic calculates the sum of estimated costs
    for each goal fact that is not yet achieved:

    1.  Initialize total heuristic cost `h = 0`.
    2.  Identify current locations of all rovers.
    3.  Identify which soil/rock samples and images have already been collected/taken.
    4.  Identify current locations of soil/rock samples on the ground (though goal waypoint is assumed).
    5.  Identify calibrated cameras.
    6.  For each goal fact `G` in the task's goals:
        *   If `G` is already true in the current state, cost for this goal is 0.
        *   If `G` is `(communicated_soil_data ?w)` and not true:
            *   Add 1 for the final `communicate_soil_data` action.
            *   Check if `(have_soil_analysis ?r ?w)` is true for any rover `?r`.
            *   If not collected:
                *   Add 1 for the `sample_soil` action.
                *   Estimate navigation cost for a soil-equipped rover to reach `?w`: Find the minimum shortest path distance from any soil-equipped rover's current location to `?w`. Add this cost.
                *   Estimate navigation cost from `?w` (where sampling occurred) to a lander communication point: Find the minimum shortest path distance from `?w` to any waypoint visible from the lander. Add this cost.
            *   If collected by some rover `?r_coll`:
                *   Estimate navigation cost for `?r_coll` to reach a lander communication point: Find the minimum shortest path distance from `?r_coll`'s current location to any waypoint visible from the lander. Add this cost.
        *   If `G` is `(communicated_rock_data ?w)` and not true:
            *   Similar logic as soil data, using rock capabilities and `have_rock_analysis`.
        *   If `G` is `(communicated_image_data ?o ?m)` and not true:
            *   Add 1 for the final `communicate_image_data` action.
            *   Check if `(have_image ?r ?o ?m)` is true for any rover `?r`.
            *   If not taken:
                *   Add 1 for the `take_image` action.
                *   Find suitable rovers (equipped for imaging, with camera supporting mode `?m`).
                *   Find waypoints visible from objective `?o` (image waypoints).
                *   Find calibration target `?t` for suitable cameras and waypoints visible from `?t` (calibration waypoints).
                *   Estimate minimum navigation cost for a suitable rover to reach *both* a calibration waypoint and an image waypoint, perform calibration, and take the image:
                    *   Find min nav cost from a suitable rover's current location to *any* calibration waypoint.
                    *   Find min nav cost from *any* calibration waypoint to *any* image waypoint.
                    *   Add these two navigation costs.
                *   Check if any suitable camera on a suitable rover is currently calibrated. If not, add 1 for the `calibrate` action.
                *   Add the total estimated cost for taking the image (nav + calibrate + take_image) to `h`.
                *   Estimate navigation cost from an image waypoint (where image was taken) to a lander communication point: Find the minimum shortest path distance from any image waypoint to any waypoint visible from the lander. Add this cost.
            *   If taken by some rover `?r_img`:
                *   Estimate navigation cost for `?r_img` to reach a lander communication point: Find the minimum shortest path distance from `?r_img`'s current location to any waypoint visible from the lander. Add this cost.
    7.  Return the total heuristic cost `h`.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goal_facts = task.goals
        static_facts = task.static

        # Data structures for static information
        self.waypoint_graph = defaultdict(set)
        self.all_waypoints = set()
        self.lander_location = None
        self.lander_comm_points = set()
        self.rover_capabilities = defaultdict(lambda: defaultdict(bool)) # {rover: {soil: bool, rock: bool, imaging: bool}}
        self.camera_info = defaultdict(lambda: {'on_board': None, 'supports': set(), 'calibration_target': None}) # {camera: {on_board: rover, supports: {mode}, calibration_target: objective}}
        self.objective_visibility = defaultdict(set) # {objective: {waypoint}}
        self.calibration_targets = {} # {camera: objective}
        self.rover_list = set() # Keep track of all rovers

        # Parse static facts
        visible_links = defaultdict(set) # Need this for lander comm points

        for fact in static_facts:
            parts = get_parts(fact)
            predicate = parts[0]

            if predicate == "can_traverse":
                # Build graph based on any rover's traversal ability
                _, rover, w1, w2 = parts
                self.waypoint_graph[w1].add(w2)
                self.waypoint_graph[w2].add(w1) # Assuming traversal is bidirectional
                self.all_waypoints.add(w1)
                self.all_waypoints.add(w2)
                self.rover_list.add(rover)

            elif predicate == "at_lander":
                _, lander, w = parts
                self.lander_location = w
                self.all_waypoints.add(w)

            elif predicate == "equipped_for_soil_analysis":
                _, rover = parts
                self.rover_capabilities[rover]['soil'] = True
                self.rover_list.add(rover)
            elif predicate == "equipped_for_rock_analysis":
                _, rover = parts
                self.rover_capabilities[rover]['rock'] = True
                self.rover_list.add(rover)
            elif predicate == "equipped_for_imaging":
                _, rover = parts
                self.rover_capabilities[rover]['imaging'] = True
                self.rover_list.add(rover)

            elif predicate == "on_board":
                _, camera, rover = parts
                self.camera_info[camera]['on_board'] = rover
                self.rover_list.add(rover)
            elif predicate == "supports":
                _, camera, mode = parts
                self.camera_info[camera]['supports'].add(mode)
            elif predicate == "calibration_target":
                _, camera, objective = parts
                self.camera_info[camera]['calibration_target'] = objective
                self.calibration_targets[camera] = objective

            elif predicate == "visible_from":
                _, objective, waypoint = parts
                self.objective_visibility[objective].add(waypoint)
                self.all_waypoints.add(waypoint)

            elif predicate == "visible":
                 _, w1, w2 = get_parts(fact)
                 visible_links[w1].add(w2)
                 visible_links[w2].add(w1) # Assuming visible is symmetric


        # Process visible facts to find lander communication points
        if self.lander_location and self.lander_location in visible_links:
             self.lander_comm_points = visible_links[self.lander_location]

        # Compute all-pairs shortest paths using BFS
        self.distances = {}
        for start_wp in self.all_waypoints:
            self.distances[start_wp] = bfs(self.waypoint_graph, start_wp)

    def get_min_nav_cost(self, start_waypoints, end_waypoints):
        """
        Calculates the minimum navigation cost from any of the start_waypoints
        to any of the end_waypoints.
        start_waypoints: set of waypoints
        end_waypoints: set of waypoints
        Returns float('inf') if no path exists between any pair or inputs are empty.
        """
        min_cost = float('inf')

        if not start_waypoints or not end_waypoints:
            return min_cost # Cannot navigate if no start or end points

        # Filter for waypoints actually in our distance map (i.e., in the graph)
        # This handles cases where a waypoint might be mentioned but not traversable
        # from anywhere or to anywhere.
        reachable_start_wps = {wp for wp in start_waypoints if wp in self.distances}
        reachable_end_wps = {wp for wp in end_waypoints if wp in self.distances}

        if not reachable_start_wps or not reachable_end_wps:
             return min_cost # Cannot navigate if start/end points are not in the graph

        for start_wp in reachable_start_wps:
            for end_wp in reachable_end_wps:
                dist = self.distances[start_wp].get(end_wp, float('inf')) # Use .get for safety
                min_cost = min(min_cost, dist)

        return min_cost

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state (frozenset of facts)

        # Extract dynamic information from the current state
        rover_locations = {} # {rover: waypoint}
        collected_soil_samples = {} # {waypoint: rover}
        collected_rock_samples = {} # {waypoint: rover}
        taken_images = {} # {(objective, mode): rover}
        calibrated_cameras = set() # {(camera, rover)}

        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]

            if predicate == "at":
                obj, loc = parts[1], parts[2]
                if obj in self.rover_list: # Check if it's a rover
                    rover_locations[obj] = loc
            elif predicate == "have_soil_analysis":
                rover, waypoint = parts[1], parts[2]
                collected_soil_samples[waypoint] = rover # Store which rover has it
            elif predicate == "have_rock_analysis":
                rover, waypoint = parts[1], parts[2]
                collected_rock_samples[waypoint] = rover # Store which rover has it
            elif predicate == "have_image":
                rover, objective, mode = parts[1], parts[2], parts[3]
                taken_images[(objective, mode)] = rover # Store which rover has it
            elif predicate == "calibrated":
                calibrated_cameras.add((parts[1], parts[2]))

        total_cost = 0

        # Iterate through each goal condition
        for goal in self.goal_facts:
            if goal in state:
                continue # Goal already achieved

            parts = get_parts(goal)
            predicate = parts[0]

            if predicate == "communicated_soil_data":
                waypoint = parts[1]
                total_cost += 1 # Cost for the communicate action

                # Check if sample is collected
                sample_collected = waypoint in collected_soil_samples

                if not sample_collected:
                    total_cost += 1 # Cost for the sample action

                    # Find capable rovers
                    capable_rovers = {r for r, caps in self.rover_capabilities.items() if caps['soil']}
                    current_locs_capable = {rover_locations[r] for r in capable_rovers if r in rover_locations}

                    # Cost to navigate to sample location
                    nav_to_sample_cost = self.get_min_nav_cost(current_locs_capable, {waypoint})
                    total_cost += nav_to_sample_cost

                    # Cost to navigate from sample location to communication point
                    nav_sample_to_comm_cost = self.get_min_nav_cost({waypoint}, self.lander_comm_points)
                    total_cost += nav_sample_to_comm_cost

                else: # Sample is collected
                    rover_with_sample = collected_soil_samples[waypoint]
                    current_loc_rover = {rover_locations.get(rover_with_sample)} # Set containing one location or None

                    # Cost to navigate from rover with sample to communication point
                    nav_collected_to_comm_cost = self.get_min_nav_cost({loc for loc in current_loc_rover if loc is not None}, self.lander_comm_points)
                    total_cost += nav_collected_to_comm_cost


            elif predicate == "communicated_rock_data":
                waypoint = parts[1]
                total_cost += 1 # Cost for the communicate action

                # Check if sample is collected
                sample_collected = waypoint in collected_rock_samples

                if not sample_collected:
                    total_cost += 1 # Cost for the sample action

                    # Find capable rovers
                    capable_rovers = {r for r, caps in self.rover_capabilities.items() if caps['rock']}
                    current_locs_capable = {rover_locations[r] for r in capable_rovers if r in rover_locations}

                    # Cost to navigate to sample location
                    nav_to_sample_cost = self.get_min_nav_cost(current_locs_capable, {waypoint})
                    total_cost += nav_to_sample_cost

                    # Cost to navigate from sample location to communication point
                    nav_sample_to_comm_cost = self.get_min_nav_cost({waypoint}, self.lander_comm_points)
                    total_cost += nav_sample_to_comm_cost

                else: # Sample is collected
                    rover_with_sample = collected_rock_samples[waypoint]
                    current_loc_rover = {rover_with_sample: rover_locations.get(rover_with_sample)}

                    # Cost to navigate from rover with sample to communication point
                    nav_collected_to_comm_cost = self.get_min_nav_cost({loc for loc in current_loc_rover if loc is not None}, self.lander_comm_points)
                    total_cost += nav_collected_to_comm_cost


            elif predicate == "communicated_image_data":
                objective, mode = parts[1], parts[2]
                total_cost += 1 # Cost for the communicate action

                # Check if image is taken
                image_taken = (objective, mode) in taken_images

                if not image_taken:
                    total_cost += 1 # Cost for the take_image action

                    # Find suitable rovers (equipped for imaging, with camera supporting mode)
                    suitable_rovers = set()
                    suitable_cameras = set()
                    for camera, info in self.camera_info.items():
                        # Ensure camera is on a known rover and that rover is imaging capable
                        if mode in info['supports'] and info['on_board'] in self.rover_list and self.rover_capabilities[info['on_board']]['imaging']:
                             suitable_rovers.add(info['on_board'])
                             suitable_cameras.add(camera)

                    current_locs_suitable = {rover_locations[r] for r in suitable_rovers if r in rover_locations}

                    image_waypoints = self.objective_visibility.get(objective, set())

                    # Check if calibration is needed for a suitable camera
                    calibration_needed = True
                    for cam in suitable_cameras:
                         rover_on_board = self.camera_info[cam]['on_board']
                         if (cam, rover_on_board) in calibrated_cameras:
                              calibration_needed = False
                              break # Found a calibrated suitable camera

                    if calibration_needed:
                        total_cost += 1 # Cost for the calibrate action

                        # Find calibration waypoints for suitable cameras
                        calibration_waypoints = set()
                        for cam in suitable_cameras:
                             cal_target = self.calibration_targets.get(cam)
                             if cal_target:
                                  calibration_waypoints.update(self.objective_visibility.get(cal_target, set()))

                        # Estimate navigation cost for calibration and image taking
                        # This is an approximation: min(current->cal) + min(cal->img)
                        nav_to_cal_cost = self.get_min_nav_cost(current_locs_suitable, calibration_waypoints)
                        nav_cal_to_image_cost = self.get_min_nav_cost(calibration_waypoints, image_waypoints)
                        total_cost += nav_to_cal_cost + nav_cal_to_image_cost

                    else:
                         # If calibration is not needed, just navigate to the image waypoint
                         nav_to_image_cost = self.get_min_nav_cost(current_locs_suitable, image_waypoints)
                         total_cost += nav_to_image_cost


                    # Cost to navigate from image waypoint to communication point
                    nav_image_to_comm_cost = self.get_min_nav_cost(image_waypoints, self.lander_comm_points)
                    total_cost += nav_image_to_comm_cost


                else: # Image taken
                    rover_with_image = taken_images[(objective, mode)]
                    current_loc_rover = {rover_locations.get(rover_with_image)}

                    # Cost to navigate from rover with image to communication point
                    nav_taken_to_comm_cost = self.get_min_nav_cost({loc for loc in current_loc_rover if loc is not None}, self.lander_comm_points)
                    total_cost += nav_taken_to_comm_cost

        # Ensure heuristic is 0 only at goal.
        # The loop structure already guarantees total_cost > 0 if not at goal.
        # If total_cost is inf, it means a necessary navigation step is impossible.

        return total_cost
