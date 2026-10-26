import math
from collections import deque, defaultdict
from fnmatch import fnmatch
# Assume Heuristic base class is available in the environment
# from heuristics.heuristic_base import Heuristic

# Dummy Heuristic base class for standalone testing
# In a real planning environment, this would be provided.
class Heuristic:
    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static
        # In a real scenario, task might also have initial_state, operators, etc.
        # For this heuristic, we only need goals and static facts from task.

    def __call__(self, node):
        # node.state is the frozenset of facts
        raise NotImplementedError("Subclass must implement abstract method")

# Helper functions (can be inside as static methods if preferred)
def get_parts(fact):
    """Extract the components of a PDDL fact."""
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
    Performs BFS on a graph to find shortest distances from a start node.
    Graph is an adjacency dictionary: {node: [neighbor1, neighbor2, ...]}
    Returns a dictionary {node: distance}. Unreachable nodes have distance infinity.
    """
    distances = {node: math.inf for node in graph}
    if start_node not in graph:
         # Start node is not in the graph nodes (e.g., waypoint not in domain)
         # or graph is empty. All nodes are unreachable from here.
         return distances

    distances[start_node] = 0
    queue = deque([start_node])

    while queue:
        current_node = queue.popleft()

        # current_node should always be in graph keys if it came from the queue
        # but check defensively
        if current_node in graph:
            for neighbor in graph[current_node]:
                if distances[neighbor] == math.inf:
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)
    return distances


class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    Estimates the cost to reach the goal by summing the estimated costs
    for each unachieved goal fact independently.

    For sample goals (soil/rock):
    Cost = (Nav to sample location) + (Drop if store full) + (Sample action) +
           (Nav to communication location) + (Communicate action)
    Minimize over suitable rovers. If sample already collected, skip sample steps.

    For image goals:
    Cost = (Nav to calibration location) + (Calibrate action) +
           (Nav to image location) + (Take image action) +
           (Nav to communication location) + (Communicate action)
    Minimize over suitable rovers/cameras/waypoints. If camera is calibrated,
    skip initial calibration nav/action (but calibration is consumed by take_image).
    If image already collected, skip image/calibration steps.

    Navigation costs are precomputed shortest paths using BFS.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and static facts,
        and precomputing navigation shortest paths.
        """
        super().__init__(task)

        # --- Precompute Static Information ---

        self.rover_capabilities = defaultdict(set) # rover -> {soil, rock, imaging}
        self.rover_stores = {} # rover -> store
        self.rover_cameras = defaultdict(list) # rover -> [camera1, camera2, ...]
        self.camera_modes = defaultdict(set) # camera -> {mode1, mode2, ...}
        self.camera_calib_target = {} # camera -> objective
        self.objective_visible_from = defaultdict(set) # objective -> {waypoint1, waypoint2, ...}
        self.lander_location = None
        self.lander_visible_waypoints = set()
        self.all_waypoints = set() # Collect all waypoints for graph
        self.all_rovers = set() # Collect all rovers

        # Navigation graph: rover -> {waypoint: [neighbor1, ...]}
        self.rover_nav_graphs = defaultdict(lambda: defaultdict(list))

        # Collect all visible pairs first
        all_visible_pairs = set()
        for fact in self.static:
             parts = get_parts(fact)
             if parts and parts[0] == "visible":
                  all_visible_pairs.add((parts[1], parts[2]))
                  self.all_waypoints.add(parts[1])
                  self.all_waypoints.add(parts[2])

        # Build navigation graphs based on can_traverse and visible
        for fact in self.static:
             parts = get_parts(fact)
             if not parts: continue

             predicate = parts[0]
             if predicate == "at_lander":
                 self.lander_location = parts[2]
             elif predicate == "equipped_for_soil_analysis":
                 self.rover_capabilities[parts[1]].add("soil")
             elif predicate == "equipped_for_rock_analysis":
                 self.rover_capabilities[parts[1]].add("rock")
             elif predicate == "equipped_for_imaging":
                 self.rover_capabilities[parts[1]].add("imaging")
             elif predicate == "store_of":
                 self.rover_stores[parts[2]] = parts[1] # store -> rover
             elif predicate == "on_board":
                 self.rover_cameras[parts[2]].append(parts[1]) # rover -> camera
             elif predicate == "supports":
                 self.camera_modes[parts[1]].add(parts[2]) # camera -> mode
             elif predicate == "calibration_target":
                 self.camera_calib_target[parts[1]] = parts[2] # camera -> objective
             elif predicate == "visible_from":
                 self.objective_visible_from[parts[1]].add(parts[2]) # objective -> waypoint
             elif predicate == "can_traverse":
                 rover, wp_from, wp_to = parts[1], parts[2], parts[3]
                 self.all_rovers.add(rover)
                 self.all_waypoints.add(wp_from)
                 self.all_waypoints.add(wp_to)
                 if (wp_from, wp_to) in all_visible_pairs:
                       self.rover_nav_graphs[rover][wp_from].append(wp_to)

        # Ensure all waypoints are in the graph keys even if they have no edges
        for rover in self.all_rovers:
             for wp in self.all_waypoints:
                  if wp not in self.rover_nav_graphs[rover]:
                       self.rover_nav_graphs[rover][wp] = []

        # Precompute shortest paths for each rover from every waypoint
        self.rover_shortest_paths = {} # rover -> {start_wp: {end_wp: distance}}
        for rover in self.all_rovers:
            self.rover_shortest_paths[rover] = {}
            for start_wp in self.all_waypoints:
                self.rover_shortest_paths[rover][start_wp] = bfs(self.rover_nav_graphs[rover], start_wp)

        # Find lander visible waypoints (only need to do this once)
        if self.lander_location:
             for wp in self.all_waypoints:
                  if f"(visible {wp} {self.lander_location})" in self.static:
                       self.lander_visible_waypoints.add(wp)


    def get_distance(self, rover, start_wp, end_wp):
        """Lookup precomputed shortest distance."""
        if rover not in self.rover_shortest_paths or \
           start_wp not in self.rover_shortest_paths[rover] or \
           end_wp not in self.rover_shortest_paths[rover][start_wp]:
             # This should not happen if precomputation is correct and covers all waypoints
             # but as a safeguard, return infinity
             # print(f"Warning: Distance lookup failed for {rover} from {start_wp} to {end_wp}")
             return math.inf
        return self.rover_shortest_paths[rover][start_wp][end_wp]

    def get_state_info(self, state):
        """Parse dynamic facts from the current state."""
        rover_at = {} # rover -> waypoint
        store_full = set() # {store1, store2, ...}
        camera_calibrated = set() # {(camera, rover), ...}
        rover_has_soil = defaultdict(set) # rover -> {waypoint1, ...}
        rover_has_rock = defaultdict(set) # rover -> {waypoint1, ...}
        rover_has_image = defaultdict(set) # rover -> {(objective, mode), ...}
        soil_samples_at = set() # {waypoint1, ...}
        rock_samples_at = set() # {waypoint1, ...}
        communicated_soil = set() # {waypoint1, ...}
        communicated_rock = set() # {waypoint1, ...}
        communicated_image = set() # {(objective, mode), ...}

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]
            if predicate == "at":
                # Assuming (at ?x ?y) where ?x is a rover
                if parts[1] in self.all_rovers: # Check if it's a known rover
                     rover_at[parts[1]] = parts[2]
            elif predicate == "full":
                store_full.add(parts[1])
            elif predicate == "calibrated":
                 # Check if the camera is actually on the rover according to static facts
                 camera, rover = parts[1], parts[2]
                 if camera in self.rover_cameras.get(rover, []):
                    camera_calibrated.add((camera, rover)) # (camera, rover)
                 # else: ignore inconsistent state fact
            elif predicate == "have_soil_analysis":
                rover_has_soil[parts[1]].add(parts[2]) # rover -> waypoint
            elif predicate == "have_rock_analysis":
                rover_has_rock[parts[1]].add(parts[2]) # rover -> waypoint
            elif predicate == "have_image":
                rover_has_image[parts[1]].add((parts[2], parts[3])) # rover -> (objective, mode)
            elif predicate == "at_soil_sample":
                soil_samples_at.add(parts[1])
            elif predicate == "at_rock_sample":
                rock_samples_at.add(parts[1])
            elif predicate == "communicated_soil_data":
                communicated_soil.add(parts[1])
            elif predicate == "communicated_rock_data":
                communicated_rock.add(parts[1])
            elif predicate == "communicated_image_data":
                communicated_image.add((parts[1], parts[2])) # (objective, mode)

        return {
            "rover_at": rover_at,
            "store_full": store_full,
            "camera_calibrated": camera_calibrated,
            "rover_has_soil": rover_has_soil,
            "rover_has_rock": rover_has_rock,
            "rover_has_image": rover_has_image,
            "soil_samples_at": soil_samples_at,
            "rock_samples_at": rock_samples_at,
            "communicated_soil": communicated_soil,
            "communicated_rock": communicated_rock,
            "communicated_image": communicated_image,
        }


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        state_info = self.get_state_info(state)

        h = 0

        # Iterate through goal facts
        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue

            predicate = parts[0]

            if predicate == "communicated_soil_data":
                waypoint = parts[1]
                if waypoint in state_info["communicated_soil"]:
                    continue # Goal already achieved

                min_goal_cost = math.inf

                # Find a rover that can achieve this goal
                for rover in self.all_rovers:
                    if "soil" not in self.rover_capabilities[rover]:
                        continue # Rover cannot sample soil

                    current_rover_wp = state_info["rover_at"].get(rover)
                    if not current_rover_wp:
                         continue # Rover location unknown

                    rover_store = self.rover_stores.get(rover)
                    if not rover_store:
                         # Rover has no store (shouldn't happen based on domain structure)
                         # Treat as unable to sample
                         continue

                    cost = 0

                    # Part 1: Get the sample (if not already held)
                    has_sample = waypoint in state_info["rover_has_soil"].get(rover, set())
                    if not has_sample:
                        # Need to sample
                        if waypoint not in state_info["soil_samples_at"]:
                            # Sample is gone and no rover has it - unreachable for this goal
                            continue # Try next rover/goal

                        # Navigate to sample location
                        nav_cost_to_sample = self.get_distance(rover, current_rover_wp, waypoint)
                        if nav_cost_to_sample == math.inf:
                            continue # Cannot reach sample location

                        cost += nav_cost_to_sample

                        # Drop if store is full
                        if rover_store in state_info["store_full"]:
                            cost += 1 # Drop action

                        cost += 1 # Sample action

                        # After sampling, rover is at 'waypoint'
                        wp_after_sample = waypoint
                    else:
                        # Rover already has the sample
                        # Rover is wherever it was when we checked state_info
                        wp_after_sample = current_rover_wp

                    # Part 2: Communicate the sample
                    min_comm_nav_cost = math.inf
                    if not self.lander_visible_waypoints:
                         # Cannot communicate if no waypoints visible from lander
                         continue # Try next rover/goal

                    for land_wp in self.lander_visible_waypoints:
                         nav_cost = self.get_distance(rover, wp_after_sample, land_wp)
                         min_comm_nav_cost = min(min_comm_nav_cost, nav_cost)

                    if min_comm_nav_cost == math.inf:
                         continue # Cannot reach any communication waypoint

                    cost += min_comm_nav_cost
                    cost += 1 # Communicate action

                    min_goal_cost = min(min_goal_cost, cost)

                if min_goal_cost == math.inf:
                     # If no suitable rover with known location could achieve this goal, return infinity
                     return math.inf
                h += min_goal_cost

            elif predicate == "communicated_rock_data":
                waypoint = parts[1]
                if waypoint in state_info["communicated_rock"]:
                    continue # Goal already achieved

                min_goal_cost = math.inf

                # Find a rover that can achieve this goal
                for rover in self.all_rovers:
                    if "rock" not in self.rover_capabilities[rover]:
                        continue # Rover cannot sample rock

                    current_rover_wp = state_info["rover_at"].get(rover)
                    if not current_rover_wp:
                         continue # Rover location unknown

                    rover_store = self.rover_stores.get(rover)
                    if not rover_store:
                         # Rover has no store
                         continue

                    cost = 0

                    # Part 1: Get the sample (if not already held)
                    has_sample = waypoint in state_info["rover_has_rock"].get(rover, set())
                    if not has_sample:
                        # Need to sample
                        if waypoint not in state_info["rock_samples_at"]:
                            # Sample is gone and no rover has it - unreachable for this goal
                            continue # Try next rover/goal

                        # Navigate to sample location
                        nav_cost_to_sample = self.get_distance(rover, current_rover_wp, waypoint)
                        if nav_cost_to_sample == math.inf:
                            continue # Cannot reach sample location

                        cost += nav_cost_to_sample

                        # Drop if store is full
                        if rover_store in state_info["store_full"]:
                            cost += 1 # Drop action

                        cost += 1 # Sample action

                        # After sampling, rover is at 'waypoint'
                        wp_after_sample = waypoint
                    else:
                        # Rover already has the sample
                        wp_after_sample = current_rover_wp # Rover is wherever it was

                    # Part 2: Communicate the sample
                    min_comm_nav_cost = math.inf
                    if not self.lander_visible_waypoints:
                         # Cannot communicate
                         continue # Try next rover/goal

                    for land_wp in self.lander_visible_waypoints:
                         nav_cost = self.get_distance(rover, wp_after_sample, land_wp)
                         min_comm_nav_cost = min(min_comm_nav_cost, nav_cost)

                    if min_comm_nav_cost == math.inf:
                         continue # Cannot reach any communication waypoint

                    cost += min_comm_nav_cost
                    cost += 1 # Communicate action

                    min_goal_cost = min(min_goal_cost, cost)

                if min_goal_cost == math.inf:
                     return math.inf
                h += min_goal_cost

            elif predicate == "communicated_image_data":
                objective, mode = parts[1], parts[2]
                if (objective, mode) in state_info["communicated_image"]:
                    continue # Goal already achieved

                min_goal_cost = math.inf

                # Find a rover/camera that can achieve this goal
                for rover in self.all_rovers:
                    if "imaging" not in self.rover_capabilities[rover]:
                        continue # Rover cannot image

                    current_rover_wp = state_info["rover_at"].get(rover)
                    if not current_rover_wp:
                         continue # Rover location unknown

                    for camera in self.rover_cameras.get(rover, []):
                        if mode not in self.camera_modes.get(camera, set()):
                            continue # Camera does not support mode

                        cost = 0

                        # Part 1: Get the image (if not already held)
                        has_image = (objective, mode) in state_info["rover_has_image"].get(rover, set())
                        if not has_image:
                            # Need to take image

                            # Calibration step (if needed)
                            is_calibrated = (camera, rover) in state_info["camera_calibrated"]
                            wp_after_calib_nav = current_rover_wp # Default start point if already calibrated

                            if not is_calibrated:
                                calib_target = self.camera_calib_target.get(camera)
                                if not calib_target:
                                     # Camera has no calibration target - cannot calibrate
                                     continue # Try next camera/rover

                                calib_wps = self.objective_visible_from.get(calib_target, set())
                                if not calib_wps:
                                     # Calibration target not visible from anywhere - cannot calibrate
                                     continue # Try next camera/rover

                                min_calib_nav_cost = math.inf
                                best_calib_wp = None
                                for calib_wp in calib_wps:
                                     nav_cost = self.get_distance(rover, current_rover_wp, calib_wp)
                                     if nav_cost < min_calib_nav_cost:
                                          min_calib_nav_cost = nav_cost
                                          best_calib_wp = calib_wp

                                if min_calib_nav_cost == math.inf:
                                     continue # Cannot reach any calibration waypoint

                                cost += min_calib_nav_cost
                                cost += 1 # Calibrate action
                                wp_after_calib_nav = best_calib_wp # Rover is now at the calibration waypoint

                            # Take image step
                            image_wps = self.objective_visible_from.get(objective, set())
                            if not image_wps:
                                 # Objective not visible from anywhere - cannot take image
                                 continue # Try next camera/rover

                            min_image_nav_cost = math.inf
                            best_image_wp = None
                            for image_wp in image_wps:
                                 nav_cost = self.get_distance(rover, wp_after_calib_nav, image_wp)
                                 if nav_cost < min_image_nav_cost:
                                      min_image_nav_cost = nav_cost
                                      best_image_wp = image_wp

                            if min_image_nav_cost == math.inf:
                                 continue # Cannot reach any image waypoint

                            cost += min_image_nav_cost
                            cost += 1 # Take image action

                            # After taking image, rover is at 'best_image_wp'
                            wp_after_image = best_image_wp
                        else:
                            # Rover already has the image
                            wp_after_image = current_rover_wp # Rover is wherever it was

                        # Part 2: Communicate the image
                        min_comm_nav_cost = math.inf
                        if not self.lander_visible_waypoints:
                             # Cannot communicate
                             continue # Try next camera/rover

                        for land_wp in self.lander_visible_waypoints:
                             nav_cost = self.get_distance(rover, wp_after_image, land_wp)
                             min_comm_nav_cost = min(min_comm_nav_cost, nav_cost)

                        if min_comm_nav_cost == math.inf:
                             continue # Cannot reach any communication waypoint

                        cost += min_comm_nav_cost
                        cost += 1 # Communicate action

                        min_goal_cost = min(min_goal_cost, cost)

                if min_goal_cost == math.inf:
                     return math.inf
                h += min_goal_cost

        return h

