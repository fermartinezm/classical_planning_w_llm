import collections
from fnmatch import fnmatch
# Assuming a base class Heuristic is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact strings or malformed facts gracefully
    if not fact or not isinstance(fact, str) or len(fact) < 2 or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at rover1 waypoint1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def bfs(graph, start_node):
    """
    Performs Breadth-First Search to find shortest distances from a start node
    in a graph represented as an adjacency list.

    - `graph`: A dictionary where keys are nodes and values are sets of neighbors.
    - `start_node`: The node to start the BFS from.
    - Returns: A dictionary mapping reachable nodes to their shortest distance from start_node.
               Nodes not reachable are not included.
    """
    distances = {start_node: 0}
    queue = collections.deque([start_node])
    while queue:
        current_node = queue.popleft()
        current_dist = distances[current_node]
        # Ensure current_node exists in the graph keys before iterating neighbors
        if current_node in graph:
            for neighbor in graph[current_node]:
                if neighbor not in distances:
                    distances[neighbor] = current_dist + 1
                    queue.append(neighbor)
    return distances

# Define the heuristic class. Inherit from Heuristic if the base class is provided.
# class roversHeuristic(Heuristic):
class roversHeuristic:
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the minimum number of actions required to satisfy all
    unachieved goal conditions. It sums the estimated cost for each individual
    unachieved goal, selecting the minimum cost plan among available rovers
    and necessary intermediate locations (sample sites, calibration targets,
    image locations, and lander communication points). Movement costs are
    estimated using shortest path distances on the waypoint graph specific
    to each rover, computed using Breadth-First Search (BFS).

    # Assumptions
    - The heuristic assumes that samples (soil/rock) required for goals are
      present initially at their respective waypoints and are consumed upon sampling.
      If a goal requires a sample not initially present, it's considered unreachable.
    - Visibility relations (`visible` and `visible_from`) are static.
    - Rover capabilities (`equipped_for_...`), camera properties (`on_board`,
      `supports`, `calibration_target`), and store ownership (`store_of`) are static.
    - The lander location is static.
    - Movement cost between adjacent waypoints is 1. Action costs (sample, drop,
      calibrate, take_image, communicate) are 1.
    - The heuristic calculates the cost for each goal independently and sums them,
      ignoring potential synergies (e.g., visiting a waypoint useful for multiple goals)
      or conflicts (e.g., multiple rovers needing the same resource). This makes it
      non-admissible but potentially informative.
    - If any required waypoint is unreachable by a capable rover, or if a required
      sample/target is unavailable, the goal (and thus the state) is considered
      unsolvable, and a large heuristic value is returned.

    # Heuristic Initialization
    The heuristic precomputes and stores static information from the task definition:
    - Lander location and waypoints visible from the lander.
    - Which rovers are equipped for soil, rock, and imaging tasks.
    - Which stores belong to which rovers.
    - The traversal graph for each rover based on `can_traverse` facts.
    - Initial locations of soil and rock samples.
    - Which objectives are visible from which waypoints (`visible_from`).
    - Calibration targets for each camera.
    - Which cameras are on board which rovers and which modes they support.
    It also initializes a cache for BFS results to avoid redundant path computations.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Parse the current state to identify:
       - Current location of each rover.
       - Which rovers have which samples (`have_soil_analysis`, `have_rock_analysis`).
       - Which rovers have which images (`have_image`).
       - Which stores are full.
       - Which cameras are calibrated on which rovers.
       - Which soil/rock samples are still at their initial locations (optional, mainly for debugging/validation, initial samples are key).
       - Which soil/rock/image data have already been communicated.
    2. Initialize the total heuristic cost to 0.
    3. Iterate through each goal condition specified in the task.
    4. For each goal:
       - If the goal is already satisfied in the current state, add 0 to the total cost and continue to the next goal.
       - If the goal is `(communicated_soil_data ?w)`:
         - If `?w` was not an initial soil sample location, the goal is impossible; return a large value.
         - Find the minimum cost among all soil-equipped rovers to achieve this goal:
           - If the rover already has `(have_soil_analysis ?r ?w)`: Cost is (distance from rover's current location to nearest lander-visible waypoint) + 1 (communicate).
           - If the rover does not have the sample: Cost is (distance from rover's current location to `?w`) + (cost to drop full stores, if any) + 1 (sample) + (distance from `?w` to nearest lander-visible waypoint) + 1 (communicate).
         - The minimum cost over all suitable rovers is the estimated cost for this goal. If no rover can achieve it (e.g., due to unreachable waypoints), return a large value.
         - Add this minimum cost to the total cost.
       - If the goal is `(communicated_rock_data ?w)`:
         - Follow the same logic as for soil data, using rock-specific predicates and facts.
       - If the goal is `(communicated_image_data ?o ?m)`:
         - Find the minimum cost among all imaging-equipped rovers with cameras supporting mode `?m` to achieve this goal:
           - If the rover already has `(have_image ?r ?o ?m)`: Cost is (distance from rover's current location to nearest lander-visible waypoint) + 1 (communicate).
           - If the rover does not have the image:
             - Find calibration target `?t` for the camera `?i` and waypoints `Calib_wps` visible from `?t`.
             - Find waypoints `Image_wps` visible from `?o`.
             - If `Calib_wps` or `Image_wps` are empty, this rover/camera cannot achieve the goal.
             - If the camera `?i` is calibrated on rover `?r`: Cost is (distance from rover's current location to nearest `Image_wp`) + 1 (take_image) + (distance from nearest `Image_wp` to nearest lander-visible waypoint) + 1 (communicate).
             - If the camera `?i` is not calibrated: Cost is (distance from rover's current location to nearest `Calib_wp`) + 1 (calibrate) + (distance from nearest `Calib_wp` to nearest `Image_wp`) + 1 (take_image) + (distance from nearest `Image_wp` to nearest lander-visible waypoint) + 1 (communicate).
           - The minimum cost over all suitable rover/camera combinations is the estimated cost for this goal. If no combination can achieve it, return a large value.
         - Add this minimum cost to the total cost.
    5. Return the total calculated cost. If at any point a goal was deemed impossible, the returned value will be the large unreachable value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # Data structures for static info
        self.lander_location = None
        self.soil_rovers = set()
        self.rock_rovers = set()
        self.imaging_rovers = set()
        self.rover_stores = collections.defaultdict(set)
        # rover_graph[rover][waypoint] = set(connected_waypoints)
        self.rover_graph = collections.defaultdict(lambda: collections.defaultdict(set))
        self.visible_waypoints = set() # Store pairs (w1, w2)
        self.lander_visible_wps = set() # Waypoints visible from lander
        self.initial_soil_samples = set()
        self.initial_rock_samples = set()
        # objective_visible_from[objective] = set(waypoints)
        self.objective_visible_from = collections.defaultdict(set)
        self.camera_calib_target = {} # camera -> objective
        # rover_cameras[rover] = set(cameras)
        self.rover_cameras = collections.defaultdict(set)
        # camera_modes[camera] = set(modes)
        self.camera_modes = collections.defaultdict(set)
        self.all_rovers = set() # Keep track of all rover names
        self.all_waypoint_names = set() # Keep track of all waypoint names mentioned

        # Parse static facts
        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == "at_lander":
                # Assuming only one lander
                self.lander_location = parts[2]
                self.all_waypoint_names.add(parts[2])
            elif predicate == "equipped_for_soil_analysis":
                self.soil_rovers.add(parts[1])
                self.all_rovers.add(parts[1])
            elif predicate == "equipped_for_rock_analysis":
                self.rock_rovers.add(parts[1])
                self.all_rovers.add(parts[1])
            elif predicate == "equipped_for_imaging":
                self.imaging_rovers.add(parts[1])
                self.all_rovers.add(parts[1])
            elif predicate == "store_of":
                store, rover = parts[1], parts[2]
                self.rover_stores[rover].add(store)
                self.all_rovers.add(rover)
            elif predicate == "can_traverse":
                rover, wp1, wp2 = parts[1], parts[2], parts[3]
                self.rover_graph[rover][wp1].add(wp2)
                self.all_rovers.add(rover)
                self.all_waypoint_names.add(wp1)
                self.all_waypoint_names.add(wp2)
            elif predicate == "visible":
                wp1, wp2 = parts[1], parts[2]
                self.visible_waypoints.add((wp1, wp2))
                self.visible_waypoints.add((wp2, wp1)) # Assume symmetric visibility
                self.all_waypoint_names.add(wp1)
                self.all_waypoint_names.add(wp2)
            elif predicate == "at_soil_sample":
                self.initial_soil_samples.add(parts[1])
                self.all_waypoint_names.add(parts[1])
            elif predicate == "at_rock_sample":
                self.initial_rock_samples.add(parts[1])
                self.all_waypoint_names.add(parts[1])
            elif predicate == "visible_from":
                objective, waypoint = parts[1], parts[2]
                self.objective_visible_from[objective].add(waypoint)
                self.all_waypoint_names.add(waypoint)
            elif predicate == "calibration_target":
                camera, objective = parts[1], parts[2]
                self.camera_calib_target[camera] = objective
            elif predicate == "on_board":
                camera, rover = parts[1], parts[2]
                self.rover_cameras[rover].add(camera)
                self.all_rovers.add(rover)
            elif predicate == "supports":
                camera, mode = parts[1], parts[2]
                self.camera_modes[camera].add(mode)

        # Determine lander visible waypoints
        if self.lander_location:
             for wp1, wp2 in self.visible_waypoints:
                 if wp1 == self.lander_location:
                     self.lander_visible_wps.add(wp2)
                 if wp2 == self.lander_location:
                     self.lander_visible_wps.add(wp1)
             # Check if lander location is visible from itself
             if (self.lander_location, self.lander_location) in self.visible_waypoints:
                  self.lander_visible_wps.add(self.lander_location)


        # BFS cache: (rover, start_wp) -> dist_dict
        self._bfs_cache = {}

    def get_distance(self, rover, start_wp, end_wp):
        """Get shortest distance between two waypoints for a specific rover using BFS cache."""
        if rover not in self.rover_graph or start_wp not in self.rover_graph[rover]:
             # Rover cannot traverse or start_wp is isolated for this rover
             return float('inf')

        if (rover, start_wp) not in self._bfs_cache:
            self._bfs_cache[(rover, start_wp)] = bfs(self.rover_graph[rover], start_wp)

        return self._bfs_cache[(rover, start_wp)].get(end_wp, float('inf'))

    def get_min_distance(self, rover, start_wp, target_wps):
        """Get minimum shortest distance from a start waypoint to any in a set of target waypoints for a rover."""
        if not target_wps:
            return float('inf') # Cannot reach any target if set is empty

        if rover not in self.rover_graph or start_wp not in self.rover_graph[rover]:
             # Rover cannot traverse or start_wp is isolated for this rover
             return float('inf')

        if (rover, start_wp) not in self._bfs_cache:
             self._bfs_cache[(rover, start_wp)] = bfs(self.rover_graph[rover], start_wp)

        dist_dict = self._bfs_cache[(rover, start_wp)]
        min_dist = float('inf')
        for target_wp in target_wps:
            if target_wp in dist_dict:
                min_dist = min(min_dist, dist_dict[target_wp])
        return min_dist

    def get_min_distance_from_set_to_set(self, rover, start_wps, target_wps):
        """Get minimum shortest distance from any waypoint in a start set to any in a target set for a rover."""
        if not start_wps or not target_wps:
            return float('inf')

        min_dist = float('inf')
        for start_wp in start_wps:
            # Only consider start_wp if it's part of the rover's graph
            if rover in self.rover_graph and start_wp in self.rover_graph[rover]:
                 min_dist_from_start = self.get_min_distance(rover, start_wp, target_wps)
                 min_dist = min(min_dist, min_dist_from_start)
            # else: start_wp is isolated for this rover, cannot start from here

        return min_dist # Will be inf if no path from any start_wp to any target_wp

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Data structures for state info
        rover_locations = {} # rover -> waypoint
        has_soil = collections.defaultdict(set) # rover -> set(waypoints)
        has_rock = collections.defaultdict(set) # rover -> set(waypoints)
        has_image = collections.defaultdict(set) # rover -> set((objective, mode))
        store_full = set() # set(stores)
        camera_calibrated = set() # set((camera, rover))
        # current_soil_samples = set() # Not strictly needed for heuristic logic based on initial samples
        # current_rock_samples = set() # Not strictly needed for heuristic logic based on initial samples
        soil_communicated = set() # set(waypoints)
        rock_communicated = set() # set(waypoints)
        image_communicated = set() # set((objective, mode))

        # Parse state facts
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == "at":
                obj, loc = parts[1], parts[2]
                if obj in self.all_rovers: # Check if it's a rover
                    rover_locations[obj] = loc
            elif predicate == "have_soil_analysis":
                rover, wp = parts[1], parts[2]
                has_soil[rover].add(wp)
            elif predicate == "have_rock_analysis":
                rover, wp = parts[1], parts[2]
                has_rock[rover].add(wp)
            elif predicate == "have_image":
                rover, obj, mode = parts[1], parts[2], parts[3]
                has_image[rover].add((obj, mode))
            elif predicate == "full":
                store_full.add(parts[1])
            elif predicate == "calibrated":
                camera, rover = parts[1], parts[2]
                camera_calibrated.add((camera, rover))
            # elif predicate == "at_soil_sample": # Not needed for heuristic calculation
            #     current_soil_samples.add(parts[1])
            # elif predicate == "at_rock_sample": # Not needed for heuristic calculation
            #     current_rock_samples.add(parts[1])
            elif predicate == "communicated_soil_data":
                soil_communicated.add(parts[1])
            elif predicate == "communicated_rock_data":
                rock_communicated.add(parts[1])
            elif predicate == "communicated_image_data":
                obj, mode = parts[1], parts[2]
                image_communicated.add((obj, mode))

        total_cost = 0
        UNREACHABLE = 1000000 # Large value for unreachable goals

        # Process goals
        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue # Skip malformed goals

            predicate = parts[0]

            if predicate == "communicated_soil_data":
                wp = parts[1]
                if wp in soil_communicated:
                    continue # Goal already achieved

                goal_cost = float('inf')

                # If the sample wasn't initially available, this goal is impossible
                if wp not in self.initial_soil_samples:
                     total_cost += UNREACHABLE
                     return total_cost # Unsolvable

                # Find best rover/plan for this soil goal
                for rover in self.soil_rovers:
                    if rover not in rover_locations: continue # Should not happen in valid states

                    rover_at = rover_locations[rover]
                    # Cost to drop full stores for this rover
                    drop_cost = sum(1 for s in self.rover_stores.get(rover, set()) if s in store_full)

                    if wp in has_soil[rover]: # Rover has the sample
                        # Need to move to lander-visible and communicate
                        dist_to_lander = self.get_min_distance(rover, rover_at, self.lander_visible_wps)
                        if dist_to_lander != float('inf'):
                            cost = dist_to_lander + 1 # communicate action
                            goal_cost = min(goal_cost, cost)
                    else: # Rover needs to sample and communicate
                        # Need to move to sample location, sample, move to lander-visible, communicate
                        dist_to_sample = self.get_distance(rover, rover_at, wp)
                        dist_sample_to_lander = self.get_min_distance(rover, wp, self.lander_visible_wps)

                        if dist_to_sample != float('inf') and dist_sample_to_lander != float('inf'):
                            cost = dist_to_sample + drop_cost + 1 + dist_sample_to_lander + 1
                            goal_cost = min(goal_cost, cost)

                if goal_cost == float('inf'):
                    total_cost += UNREACHABLE # Unreachable goal
                    return total_cost # Unsolvable

                total_cost += goal_cost

            elif predicate == "communicated_rock_data":
                wp = parts[1]
                if wp in rock_communicated:
                    continue # Goal already achieved

                goal_cost = float('inf')

                # If the sample wasn't initially available, this goal is impossible
                if wp not in self.initial_rock_samples:
                     total_cost += UNREACHABLE
                     return total_cost # Unsolvable

                # Find best rover/plan for this rock goal
                for rover in self.rock_rovers:
                    if rover not in rover_locations: continue

                    rover_at = rover_locations[rover]
                    drop_cost = sum(1 for s in self.rover_stores.get(rover, set()) if s in store_full)

                    if wp in has_rock[rover]: # Rover has the sample
                        # Need to move to lander-visible and communicate
                        dist_to_lander = self.get_min_distance(rover, rover_at, self.lander_visible_wps)
                        if dist_to_lander != float('inf'):
                            cost = dist_to_lander + 1 # communicate action
                            goal_cost = min(goal_cost, cost)
                    else: # Rover needs to sample and communicate
                        # Need to move to sample location, sample, move to lander-visible, communicate
                        dist_to_sample = self.get_distance(rover, rover_at, wp)
                        dist_sample_to_lander = self.get_min_distance(rover, wp, self.lander_visible_wps)

                        if dist_to_sample != float('inf') and dist_sample_to_lander != float('inf'):
                            cost = dist_to_sample + drop_cost + 1 + dist_sample_to_lander + 1
                            goal_cost = min(goal_cost, cost)

                if goal_cost == float('inf'):
                    total_cost += UNREACHABLE # Unreachable goal
                    return total_cost # Unsolvable

                total_cost += goal_cost

            elif predicate == "communicated_image_data":
                obj, mode = parts[1], parts[2]
                if (obj, mode) in image_communicated:
                    continue # Goal already achieved

                goal_cost = float('inf')

                # Find best rover/camera/plan for this image goal
                for rover in self.imaging_rovers:
                    if rover not in rover_locations: continue
                    rover_at = rover_locations[rover]

                    for camera in self.rover_cameras.get(rover, set()):
                        if mode in self.camera_modes.get(camera, set()):
                            # Found a suitable rover and camera
                            if (obj, mode) in has_image[rover]: # Rover has the image
                                # Need to move to lander-visible and communicate
                                dist_to_lander = self.get_min_distance(rover, rover_at, self.lander_visible_wps)
                                if dist_to_lander != float('inf'):
                                    cost = dist_to_lander + 1 # communicate action
                                    goal_cost = min(goal_cost, cost)
                            else: # Rover needs to take image and communicate
                                calib_target = self.camera_calib_target.get(camera)
                                if calib_target is None: continue # Camera has no calib target

                                calib_wps = self.objective_visible_from.get(calib_target, set())
                                image_wps = self.objective_visible_from.get(obj, set())

                                if not calib_wps or not image_wps: continue # Cannot calibrate or take image

                                if (camera, rover) in camera_calibrated: # Camera is calibrated
                                    # Need to move to image location, take image, move to lander-visible, communicate
                                    dist_to_image = self.get_min_distance(rover, rover_at, image_wps)
                                    dist_image_to_lander = self.get_min_distance_from_set_to_set(rover, image_wps, self.lander_visible_wps)

                                    if dist_to_image != float('inf') and dist_image_to_lander != float('inf'):
                                        cost = dist_to_image + 1 + dist_image_to_lander + 1
                                        goal_cost = min(goal_cost, cost)
                                else: # Camera is not calibrated
                                    # Need to move to calib, calibrate, move to image, take image, move to lander-visible, communicate
                                    dist_to_calib = self.get_min_distance(rover, rover_at, calib_wps)
                                    dist_calib_to_image = self.get_min_distance_from_set_to_set(rover, calib_wps, image_wps)
                                    dist_image_to_lander = self.get_min_distance_from_set_to_set(rover, image_wps, self.lander_visible_wps)

                                    if dist_to_calib != float('inf') and dist_calib_to_image != float('inf') and dist_image_to_lander != float('inf'):
                                        cost = dist_to_calib + 1 + dist_calib_to_image + 1 + dist_image_to_lander + 1
                                        goal_cost = min(goal_cost, cost)

                if goal_cost == float('inf'):
                    total_cost += UNREACHABLE # Unreachable goal
                    return total_cost # Unsolvable

                total_cost += goal_cost

            # Add other goal types if any

        return total_cost

