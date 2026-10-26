from fnmatch import fnmatch
from collections import defaultdict, deque
import math

# Assume Heuristic base class is available
# from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential leading/trailing whitespace and empty facts
    fact = fact.strip()
    if not fact or fact[0] != '(' or fact[-1] != ')':
        return [] # Invalid fact format
    return fact[1:-1].split()

# Helper function to match PDDL facts (optional, but useful)
def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    Wildcards `*` are allowed in args.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class roversHeuristic: # Inherit from Heuristic in the actual environment
    """
    A domain-dependent heuristic for the Rovers domain.

    Estimates the number of actions required to achieve each uncommunicated goal.
    The total heuristic is the sum of the estimated costs for each goal.

    For each goal (communicating soil, rock, or image data):
    - If the data is already communicated, cost is 0.
    - If the data exists on a rover but is not communicated, cost is estimated
      as navigation to a communication point + 1 (communicate action). The minimum
      cost over all rovers holding the data and all communication points is used.
    - If the data does not exist, cost is estimated as:
      - Navigation to sample/image location
      - (Optional: drop action if store full for samples)
      - Sample/Image action
      - (Optional: calibration steps for images, including navigation to calibration waypoint)
      - Navigation from sample/image location to a communication point
      - Communicate action
      The minimum cost over all suitable rovers, cameras (for imaging), and relevant
      waypoints (sample, image, calibration, communication) is used.

    Navigation costs are estimated using precomputed shortest paths on the
    waypoint graph (based on 'visible' predicates).
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and static facts.
        Precomputes navigation shortest paths.
        """
        self.goals = task.goals
        static_facts = task.static

        # --- Precompute static information ---

        self.lander_location = None
        self.waypoint_graph = defaultdict(set) # Adj list: waypoint -> set of connected waypoints
        self.waypoints = set()
        self.rovers = set()
        self.stores = set()
        self.cameras = set()
        self.modes = set()
        self.objectives = set()
        self.landers = set()

        self.rover_capabilities = defaultdict(set) # rover -> {soil, rock, imaging}
        self.rover_stores = {} # rover -> store
        self.rover_cameras = defaultdict(set) # rover -> {camera}
        self.camera_modes = defaultdict(set) # camera -> {mode}
        self.camera_calibration_target = {} # camera -> objective
        self.objective_visible_from = defaultdict(set) # objective -> {waypoint}

        # Parse static facts
        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue # Skip invalid facts

            pred = parts[0]
            if pred == 'at_lander':
                self.landers.add(parts[1])
                self.lander_location = parts[2] # Assuming only one lander
                self.waypoints.add(parts[2])
            elif pred == 'visible':
                w1, w2 = parts[1], parts[2]
                self.waypoint_graph[w1].add(w2)
                self.waypoint_graph[w2].add(w1) # Assuming visibility is symmetric
                self.waypoints.add(w1)
                self.waypoints.add(w2)
            elif pred == 'equipped_for_soil_analysis':
                self.rover_capabilities[parts[1]].add('soil')
                self.rovers.add(parts[1])
            elif pred == 'equipped_for_rock_analysis':
                self.rover_capabilities[parts[1]].add('rock')
                self.rovers.add(parts[1])
            elif pred == 'equipped_for_imaging':
                self.rover_capabilities[parts[1]].add('imaging')
                self.rovers.add(parts[1])
            elif pred == 'store_of':
                self.rover_stores[parts[2]] = parts[1] # rover -> store
                self.rovers.add(parts[2])
                self.stores.add(parts[1])
            elif pred == 'on_board':
                self.rover_cameras[parts[2]].add(parts[1]) # rover -> camera
                self.rovers.add(parts[2])
                self.cameras.add(parts[1])
            elif pred == 'supports':
                self.camera_modes[parts[1]].add(parts[2]) # camera -> mode
                self.cameras.add(parts[1])
                self.modes.add(parts[2])
            elif pred == 'calibration_target':
                self.camera_calibration_target[parts[1]] = parts[2] # camera -> objective
                self.cameras.add(parts[1])
                self.objectives.add(parts[2])
            elif pred == 'visible_from':
                self.objective_visible_from[parts[1]].add(parts[2]) # objective -> waypoint
                self.objectives.add(parts[1])
                self.waypoints.add(parts[2])
            # Add other types to sets if needed for completeness, though not strictly used in heuristic logic yet
            elif pred == 'rover': self.rovers.add(parts[1])
            elif pred == 'waypoint': self.waypoints.add(parts[1])
            elif pred == 'store': self.stores.add(parts[1])
            elif pred == 'camera': self.cameras.add(parts[1])
            elif pred == 'mode': self.modes.add(parts[1])
            elif pred == 'lander': self.landers.add(parts[1])
            elif pred == 'objective': self.objectives.add(parts[1])


        # Precompute shortest paths between all waypoints using BFS
        self.shortest_paths = {} # (w1, w2) -> distance
        for start_waypoint in self.waypoints:
            self._bfs(start_waypoint)

        # Identify communication points (waypoints visible from lander)
        self.communication_points = set()
        if self.lander_location:
             for fact in static_facts:
                 parts = get_parts(fact)
                 if parts[0] == 'visible' and parts[2] == self.lander_location:
                     self.communication_points.add(parts[1])
                 elif parts[0] == 'visible' and parts[1] == self.lander_location:
                     self.communication_points.add(parts[2])


    def _bfs(self, start_waypoint):
        """Performs BFS from a start waypoint to find distances to all others."""
        distances = {wp: math.inf for wp in self.waypoints}
        if start_waypoint in distances: # Ensure start_waypoint is valid
            distances[start_waypoint] = 0
            queue = deque([start_waypoint])

            while queue:
                current_wp = queue.popleft()

                for neighbor in self.waypoint_graph.get(current_wp, []):
                    if distances[neighbor] == math.inf:
                        distances[neighbor] = distances[current_wp] + 1
                        queue.append(neighbor)

        # Store results
        for end_waypoint, dist in distances.items():
            self.shortest_paths[(start_waypoint, end_waypoint)] = dist

    def get_shortest_path_cost(self, w1, w2):
        """Returns the precomputed shortest path cost between two waypoints."""
        if w1 == w2:
            return 0
        # Return infinity if either waypoint is not in our graph (shouldn't happen in valid problems)
        if w1 not in self.waypoints or w2 not in self.waypoints:
             return math.inf
        return self.shortest_paths.get((w1, w2), math.inf)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # If goal is reached, heuristic is 0
        if self.goals <= state:
            return 0

        # --- Extract dynamic information from the current state ---
        current_rover_locations = {} # rover -> waypoint
        rover_has_soil = defaultdict(set) # rover -> {waypoint}
        rover_has_rock = defaultdict(set) # rover -> {waypoint}
        rover_has_image = defaultdict(set) # rover -> {(objective, mode)}
        rover_store_status = {} # rover -> 'empty' or 'full'
        camera_calibrated = defaultdict(set) # rover -> {camera}
        soil_samples_at = set() # {waypoint}
        rock_samples_at = set() # {waypoint}
        communicated_soil = set() # {waypoint}
        communicated_rock = set() # {waypoint}
        communicated_image = set() # {(objective, mode)}

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            pred = parts[0]
            if pred == 'at':
                if parts[1] in self.rovers: # Ensure it's a rover
                    current_rover_locations[parts[1]] = parts[2]
            elif pred == 'have_soil_analysis':
                rover_has_soil[parts[1]].add(parts[2])
            elif pred == 'have_rock_analysis':
                rover_has_rock[parts[1]].add(parts[2])
            elif pred == 'have_image':
                if len(parts) > 3: # Ensure mode is present
                    rover_has_image[parts[1]].add((parts[2], parts[3]))
            elif pred == 'empty':
                 # Find which rover this store belongs to
                 for r, s in self.rover_stores.items():
                     if s == parts[1]:
                         rover_store_status[r] = 'empty'
                         break
            elif pred == 'full':
                 # Find which rover this store belongs to
                 for r, s in self.rover_stores.items():
                     if s == parts[1]:
                         rover_store_status[r] = 'full'
                         break
            elif pred == 'calibrated':
                if len(parts) > 2: # Ensure rover is present
                    camera_calibrated[parts[2]].add(parts[1]) # rover -> camera
            elif pred == 'at_soil_sample':
                soil_samples_at.add(parts[1])
            elif pred == 'at_rock_sample':
                rock_samples_at.add(parts[1])
            elif pred == 'communicated_soil_data':
                communicated_soil.add(parts[1])
            elif pred == 'communicated_rock_data':
                communicated_rock.add(parts[1])
            elif pred == 'communicated_image_data':
                if len(parts) > 2: # Ensure mode is present
                    communicated_image.add((parts[1], parts[2]))

        total_heuristic_cost = 0

        # --- Calculate cost for each unachieved goal ---
        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue

            pred = parts[0]

            if pred == 'communicated_soil_data':
                waypoint = parts[1]
                if waypoint in communicated_soil:
                    continue # Goal already achieved

                min_goal_cost = math.inf

                # Find all rovers that currently have the soil data for this waypoint
                rovers_with_data = [r for r, waypoints_with_data in rover_has_soil.items() if waypoint in waypoints_with_data]

                if rovers_with_data:
                    # Data exists, find the best rover to communicate
                    for rover_with_data in rovers_with_data:
                        current_rover_loc = current_rover_locations.get(rover_with_data)
                        if current_rover_loc is None: continue # Rover location unknown

                        min_comm_nav_cost = math.inf
                        for comm_point in self.communication_points:
                             nav_cost = self.get_shortest_path_cost(current_rover_loc, comm_point)
                             min_comm_nav_cost = min(min_comm_nav_cost, nav_cost)

                        if min_comm_nav_cost != math.inf:
                            cost = min_comm_nav_cost + 1 # +1 for communicate action
                            min_goal_cost = min(min_goal_cost, cost)

                else:
                    # Data does not exist, need to sample and communicate
                    # Assume sample exists if needed and not already collected by *any* rover for this goal
                    if waypoint in soil_samples_at:
                         for rover in self.rovers:
                             if 'soil' in self.rover_capabilities.get(rover, set()):
                                 current_rover_loc = current_rover_locations.get(rover)
                                 if current_rover_loc is None: continue

                                 # Cost to sample
                                 cost_to_sample_waypoint = self.get_shortest_path_cost(current_rover_loc, waypoint)
                                 if cost_to_sample_waypoint == math.inf: continue # Cannot reach sample waypoint

                                 sampling_prep_cost = 0
                                 # Check store status - need empty store
                                 store = self.rover_stores.get(rover)
                                 if store and rover_store_status.get(rover) == 'full':
                                     sampling_prep_cost = 1 # Cost for drop action

                                 sampling_cost = cost_to_sample_waypoint + sampling_prep_cost + 1 # +1 for sample_soil action
                                 location_after_sampling = waypoint

                                 # Cost to communicate
                                 min_comm_nav_cost = math.inf
                                 for comm_point in self.communication_points:
                                     nav_cost = self.get_shortest_path_cost(location_after_sampling, comm_point)
                                     min_comm_nav_cost = min(min_comm_nav_cost, nav_cost)

                                 if min_comm_nav_cost != math.inf:
                                     communication_cost = min_comm_nav_cost + 1 # +1 for communicate action
                                     total_rover_cost = sampling_cost + communication_cost
                                     min_goal_cost = min(min_goal_cost, total_rover_cost)

                if min_goal_cost == math.inf:
                     # If the goal is unachieved and we couldn't find a finite path
                     # to achieve it, it's likely unreachable.
                     return math.inf
                else:
                     total_heuristic_cost += min_goal_cost


            elif pred == 'communicated_rock_data':
                waypoint = parts[1]
                if waypoint in communicated_rock:
                    continue # Goal already achieved

                min_goal_cost = math.inf

                # Find all rovers that currently have the rock data for this waypoint
                rovers_with_data = [r for r, waypoints_with_data in rover_has_rock.items() if waypoint in waypoints_with_data]

                if rovers_with_data:
                    # Data exists, find the best rover to communicate
                    for rover_with_data in rovers_with_data:
                        current_rover_loc = current_rover_locations.get(rover_with_data)
                        if current_rover_loc is None: continue

                        min_comm_nav_cost = math.inf
                        for comm_point in self.communication_points:
                             nav_cost = self.get_shortest_path_cost(current_rover_loc, comm_point)
                             min_comm_nav_cost = min(min_comm_nav_cost, nav_cost)

                        if min_comm_nav_cost != math.inf:
                            cost = min_comm_nav_cost + 1 # +1 for communicate action
                            min_goal_cost = min(min_goal_cost, cost)

                else:
                    # Data does not exist, need to sample and communicate
                    # Assume sample exists if needed
                    if waypoint in rock_samples_at:
                         for rover in self.rovers:
                             if 'rock' in self.rover_capabilities.get(rover, set()):
                                 current_rover_loc = current_rover_locations.get(rover)
                                 if current_rover_loc is None: continue

                                 # Cost to sample
                                 cost_to_sample_waypoint = self.get_shortest_path_cost(current_rover_loc, waypoint)
                                 if cost_to_sample_waypoint == math.inf: continue

                                 sampling_prep_cost = 0
                                 # Check store status - need empty store
                                 store = self.rover_stores.get(rover)
                                 if store and rover_store_status.get(rover) == 'full':
                                     sampling_prep_cost = 1 # Cost for drop action

                                 sampling_cost = cost_to_sample_waypoint + sampling_prep_cost + 1 # +1 for sample_rock action
                                 location_after_sampling = waypoint

                                 # Cost to communicate
                                 min_comm_nav_cost = math.inf
                                 for comm_point in self.communication_points:
                                     nav_cost = self.get_shortest_path_cost(location_after_sampling, comm_point)
                                     min_comm_nav_cost = min(min_comm_nav_cost, nav_cost)

                                 if min_comm_nav_cost != math.inf:
                                     communication_cost = min_comm_nav_cost + 1 # +1 for communicate action
                                     total_rover_cost = sampling_cost + communication_cost
                                     min_goal_cost = min(min_goal_cost, total_rover_cost)

                if min_goal_cost == math.inf:
                     return math.inf
                else:
                     total_heuristic_cost += min_goal_cost


            elif pred == 'communicated_image_data':
                objective, mode = parts[1], parts[2]
                if (objective, mode) in communicated_image:
                    continue # Goal already achieved

                min_goal_cost = math.inf

                # Find all rovers that currently have the image for this objective and mode
                rovers_with_image = [r for r, images_held in rover_has_image.items() if (objective, mode) in images_held]

                if rovers_with_image:
                    # Image exists, find the best rover to communicate
                    for rover_with_image in rovers_with_image:
                        current_rover_loc = current_rover_locations.get(rover_with_image)
                        if current_rover_loc is None: continue

                        min_comm_nav_cost = math.inf
                        for comm_point in self.communication_points:
                             nav_cost = self.get_shortest_path_cost(current_rover_loc, comm_point)
                             min_comm_nav_cost = min(min_comm_nav_cost, nav_cost)

                        if min_comm_nav_cost != math.inf:
                            cost = min_comm_nav_cost + 1 # +1 for communicate action
                            min_goal_cost = min(min_goal_cost, cost)

                else:
                    # Image does not exist, need to acquire and communicate
                    for rover in self.rovers:
                        if 'imaging' in self.rover_capabilities.get(rover, set()):
                            current_rover_loc = current_rover_locations.get(rover)
                            if current_rover_loc is None: continue

                            for camera in self.rover_cameras.get(rover, set()):
                                if mode in self.camera_modes.get(camera, set()):

                                    # Cost to acquire image
                                    image_acquisition_cost = math.inf

                                    # Find best waypoint to take image from
                                    possible_image_waypoints = self.objective_visible_from.get(objective, set())
                                    if not possible_image_waypoints: continue # Cannot see objective

                                    calibration_needed = (camera not in camera_calibrated.get(rover, set()))

                                    if calibration_needed:
                                        cal_target = self.camera_calibration_target.get(camera)
                                        if cal_target is None: continue # Camera has no calibration target

                                        possible_cal_waypoints = self.objective_visible_from.get(cal_target, set()) # Calibration target is an objective
                                        if not possible_cal_waypoints: continue # Cannot see calibration target

                                        min_cost_to_cal_waypoint = math.inf
                                        best_cal_waypoint = None
                                        for w_cal in possible_cal_waypoints:
                                            nav_cost = self.get_shortest_path_cost(current_rover_loc, w_cal)
                                            if nav_cost != math.inf:
                                                if nav_cost < min_cost_to_cal_waypoint:
                                                    min_cost_to_cal_waypoint = nav_cost
                                                    best_cal_waypoint = w_cal

                                        if best_cal_waypoint is None: continue # Cannot reach any calibration waypoint

                                        cost_calibration_step = min_cost_to_cal_waypoint + 1 # +1 for calibrate action
                                        location_after_calibration = best_cal_waypoint

                                        # Now move from calibration location to image location
                                        min_cost_from_cal_to_image = math.inf
                                        best_image_waypoint_after_cal = None
                                        for p in possible_image_waypoints:
                                            nav_cost = self.get_shortest_path_cost(location_after_calibration, p)
                                            if nav_cost != math.inf:
                                                if nav_cost < min_cost_from_cal_to_image:
                                                    min_cost_from_cal_to_image = nav_cost
                                                    best_image_waypoint_after_cal = p

                                        if best_image_waypoint_after_cal is None: continue # Cannot reach image waypoint from cal waypoint

                                        image_acquisition_cost = cost_calibration_step + min_cost_from_cal_to_image + 1 # +1 for take_image action
                                        location_after_acquisition = best_image_waypoint_after_cal

                                    else: # Calibration not needed
                                        # Move directly to image location
                                        min_cost_to_image_waypoint_direct = math.inf
                                        best_image_waypoint_direct = None
                                        for p in possible_image_waypoints:
                                            nav_cost = self.get_shortest_path_cost(current_rover_loc, p)
                                            if nav_cost != math.inf:
                                                if nav_cost < min_cost_to_image_waypoint_direct:
                                                    min_cost_to_image_waypoint_direct = nav_cost
                                                    best_image_waypoint_direct = p

                                        if best_image_waypoint_direct is None: continue # Cannot reach image waypoint

                                        image_acquisition_cost = min_cost_to_image_waypoint_direct + 1 # +1 for take_image action
                                        location_after_acquisition = best_image_waypoint_direct

                                    # Cost to communicate
                                    min_comm_nav_cost = math.inf
                                    for comm_point in self.communication_points:
                                        nav_cost = self.get_shortest_path_cost(location_after_acquisition, comm_point)
                                        min_comm_nav_cost = min(min_comm_nav_cost, nav_cost)

                                    if min_comm_nav_cost != math.inf:
                                        communication_cost = min_comm_nav_cost + 1 # +1 for communicate action
                                        total_rover_camera_cost = image_acquisition_cost + communication_cost
                                        min_goal_cost = min(min_goal_cost, total_rover_camera_cost)

                if min_goal_cost == math.inf:
                     return math.inf
                else:
                     total_heuristic_cost += min_goal_cost

        # If we reached here, it means all unachieved goals had a finite estimated cost.
        # The total heuristic is the sum of these finite costs.
        # If total_heuristic_cost is 0, it means all goals were already achieved (handled at the start).
        return total_heuristic_cost

