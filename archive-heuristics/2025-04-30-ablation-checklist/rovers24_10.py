from fnmatch import fnmatch
from collections import deque, defaultdict
# Assume Heuristic base class is available
# from heuristics.heuristic_base import Heuristic

# Define a dummy Heuristic base class if the actual one is not provided in the execution environment
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    class Heuristic:
        def __init__(self, task):
            pass
        def __call__(self, node):
            pass

# Utility functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    if not fact or not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    Estimates the cost to achieve uncommunicated goals by summing up
    estimated costs for each goal independently.

    For each goal (soil, rock, or image data communicated):
    - If the data is already acquired, estimate cost to navigate to a communication
      waypoint and communicate.
    - If the data needs acquisition:
        - For soil/rock: Estimate cost to navigate to sample, sample, navigate
          to communication waypoint, and communicate.
        - For image: Estimate cost to navigate to calibration waypoint, calibrate,
          navigate to image waypoint, take image, navigate to communication
          waypoint, and communicate.
    - Choose the minimum estimated cost over suitable rovers.

    Navigation costs are precomputed using BFS on the traversable graph for each rover.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, static facts,
        and precomputing navigation distances.
        """
        self.goal_facts = task.goals
        static_facts = task.static
        initial_state = task.initial_state

        # --- Parse Static Facts ---
        self.lander_location = None
        self.comm_waypoints = set() # Waypoints visible from lander location
        self.rover_equipment = defaultdict(set) # rover -> {soil, rock, imaging}
        self.rover_store = {} # rover -> store
        self.rover_cameras = defaultdict(set) # rover -> {camera1, ...}
        self.camera_modes = defaultdict(set) # camera -> {mode1, ...}
        self.camera_calibration_target = {} # camera -> objective
        self.objective_image_waypoints = defaultdict(set) # objective -> {waypoint1, ...}
        self.objective_calibration_waypoints = defaultdict(set) # objective -> {waypoint1, ...}
        self.rover_navigation_graphs = {} # rover -> {wp: {neighbor_wp, ...}}
        self.rover_distances = {} # rover -> {start_wp: {end_wp: distance}}
        self.waypoints = set() # Set of all waypoints

        # Collect all waypoints mentioned in static facts
        for fact in static_facts:
             parts = get_parts(fact)
             if not parts: continue
             for part in parts[1:]:
                  # Simple check: does the part look like a waypoint?
                  # Assumes waypoint names are distinct and identifiable
                  if 'waypoint' in part.lower():
                       self.waypoints.add(part)

        # Build combined navigation graph (can_traverse AND visible)
        rover_combined_nav_edges = defaultdict(lambda: defaultdict(set))
        visible_pairs = set()
        for fact in static_facts:
             if match(fact, 'visible', '*', '*'):
                  wp1, wp2 = get_parts(fact)[1], get_parts(fact)[2]
                  visible_pairs.add((wp1, wp2))

        for fact in static_facts:
             parts = get_parts(fact)
             if not parts: continue
             pred = parts[0]

             if pred == 'at_lander':
                 self.lander_location = parts[2]
             elif pred == 'can_traverse':
                  rover, wp1, wp2 = parts[1], parts[2], parts[3]
                  if (wp1, wp2) in visible_pairs:
                       rover_combined_nav_edges[rover][wp1].add(wp2)
             elif pred == 'equipped_for_soil_analysis':
                 self.rover_equipment[parts[1]].add('soil')
             elif pred == 'equipped_for_rock_analysis':
                 self.rover_equipment[parts[1]].add('rock')
             elif pred == 'equipped_for_imaging':
                 self.rover_equipment[parts[1]].add('imaging')
             elif pred == 'store_of':
                 self.rover_store[parts[2]] = parts[1] # rover -> store
             elif pred == 'on_board':
                 self.rover_cameras[parts[2]].add(parts[1]) # rover -> camera
             elif pred == 'supports':
                 self.camera_modes[parts[1]].add(parts[2]) # camera -> mode
             elif pred == 'calibration_target':
                 self.camera_calibration_target[parts[1]] = parts[2] # camera -> objective
             elif pred == 'visible_from':
                 obj, wp = parts[1], parts[2]
                 self.objective_image_waypoints[obj].add(wp)
                 # Calibration waypoints are visible_from locations for the calibration target objective
                 # This mapping is done after parsing calibration_target

        # Populate calibration waypoints based on calibration_target and visible_from
        for camera, cal_obj in self.camera_calibration_target.items():
             if cal_obj in self.objective_image_waypoints: # visible_from facts for the objective
                  self.objective_calibration_waypoints[cal_obj].update(self.objective_image_waypoints[cal_obj])


        self.rover_navigation_graphs = rover_combined_nav_edges

        # Precompute distances for each rover
        for rover in self.rover_navigation_graphs.keys(): # Only build for rovers that can traverse
            self.rover_distances[rover] = self.precompute_distances(self.rover_navigation_graphs[rover], self.waypoints)

        # Identify communication waypoints (visible from lander)
        if self.lander_location:
             for fact in static_facts:
                  if match(fact, 'visible', '*', self.lander_location):
                       self.comm_waypoints.add(get_parts(fact)[1])
                  if match(fact, 'visible', self.lander_location, '*'): # Visible is often symmetric
                       self.comm_waypoints.add(get_parts(fact)[2])

        # Store initial sample locations
        self.initial_soil_samples = {get_parts(fact)[1] for fact in initial_state if match(fact, 'at_soil_sample', '*')}
        self.initial_rock_samples = {get_parts(fact)[1] for fact in initial_state if match(fact, 'at_rock_sample', '*')}


    def precompute_distances(self, graph, all_waypoints):
        """
        Computes shortest path distances from all waypoints to all other waypoints
        in the given graph using BFS.
        """
        distances = {start_wp: {end_wp: float('inf') for end_wp in all_waypoints} for start_wp in all_waypoints}

        # Ensure distance to self is 0 for all waypoints
        for wp in all_waypoints:
             distances[wp][wp] = 0

        # Run BFS from every waypoint
        for start_wp in all_waypoints:
            q = deque([(start_wp, 0)])
            # distances[start_wp][start_wp] is already 0
            visited = {start_wp}

            while q:
                current_wp, dist = q.popleft()

                # Only explore neighbors if the current waypoint is a source in the graph
                if current_wp in graph:
                    for neighbor_wp in graph[current_wp]:
                        # Check if neighbor_wp is a valid waypoint in our set
                        if neighbor_wp in all_waypoints and neighbor_wp not in visited:
                            visited.add(neighbor_wp)
                            distances[start_wp][neighbor_wp] = dist + 1
                            q.append((neighbor_wp, dist + 1))
        return distances

    def get_navigation_cost(self, rover, start_wp, end_wp):
        """Looks up the precomputed distance between two waypoints for a rover."""
        if rover not in self.rover_distances:
             return float('inf') # Rover cannot navigate
        # Check if start_wp and end_wp are valid waypoints in the precomputed distances
        if start_wp not in self.rover_distances[rover] or end_wp not in self.rover_distances[rover][start_wp]:
             # This can happen if a waypoint exists but is not part of the rover's graph
             return float('inf')
        return self.rover_distances[rover][start_wp][end_wp]

    def get_min_navigation_cost(self, rover, start_wp, target_wps):
        """Min distance from start_wp to any waypoint in target_wps."""
        min_dist = float('inf')
        if not target_wps: return float('inf')

        for target_wp in target_wps:
            dist = self.get_navigation_cost(rover, start_wp, target_wp)
            min_dist = min(min_dist, dist)
        return min_dist

    def get_min_navigation_cost_set_to_set(self, rover, start_wps, target_wps):
        """Min distance from any waypoint in start_wps to any waypoint in target_wps."""
        min_dist = float('inf')
        if not start_wps or not target_wps: return float('inf')

        for start_wp in start_wps:
            min_dist_from_start = self.get_min_navigation_cost(rover, start_wp, target_wps)
            min_dist = min(min_dist, min_dist_from_start)
        return min_dist


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # --- Extract Relevant State Information ---
        current_rover_locations = {}
        rover_has_soil_data = defaultdict(set)
        rover_has_rock_data = defaultdict(set)
        rover_has_image_data = defaultdict(set) # rover -> {(obj, mode), ...}
        rover_store_full = {} # rover -> bool
        rover_calibrated_cameras = defaultdict(set) # rover -> {camera, ...}
        at_soil_samples_in_state = set()
        at_rock_samples_in_state = set()


        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            pred = parts[0]
            if pred == 'at':
                obj, loc = parts[1], parts[2]
                # Assuming objects with 'at' predicate that are not samples are rovers or landers
                # Lander location is static, so this must be a rover
                if obj.startswith('rover'):
                     current_rover_locations[obj] = loc
            elif pred == 'have_soil_analysis':
                rover, wp = parts[1], parts[2]
                rover_has_soil_data[rover].add(wp)
            elif pred == 'have_rock_analysis':
                rover, wp = parts[1], parts[2]
                rover_has_rock_data[rover].add(wp)
            elif pred == 'have_image':
                rover, obj, mode = parts[1], parts[2], parts[3]
                rover_has_image_data[rover].add((obj, mode))
            elif pred == 'full':
                 # Find which rover this store belongs to
                 store = parts[1]
                 # Look up in the precomputed static mapping
                 for r, s in self.rover_store.items():
                      if s == store:
                           rover_store_full[r] = True
                           break
            elif pred == 'calibrated':
                 camera, rover = parts[1], parts[2]
                 rover_calibrated_cameras[rover].add(camera)
            elif pred == 'at_soil_sample':
                 at_soil_samples_in_state.add(parts[1])
            elif pred == 'at_rock_sample':
                 at_rock_samples_in_state.add(parts[1])


        total_cost = 0

        # --- Estimate Cost for Unachieved Goals ---
        for goal in self.goal_facts:
            if goal in state:
                continue # Goal already achieved

            parts = get_parts(goal)
            if not parts: continue

            pred = parts[0]

            if pred == 'communicated_soil_data':
                w = parts[1]
                min_cost_for_goal = float('inf')

                # Find rovers that are equipped for soil analysis
                equipped_rovers = {r for r, equip in self.rover_equipment.items() if 'soil' in equip}

                if not equipped_rovers:
                     return float('inf') # Cannot achieve this goal

                for rover in equipped_rovers:
                    current_wp = current_rover_locations.get(rover)
                    if current_wp is None: continue # Rover location unknown

                    acquisition_cost = float('inf')
                    communication_cost = float('inf')

                    # Case 1: Rover already has the data
                    if w in rover_has_soil_data.get(rover, set()):
                        acquisition_cost = 0
                        # Cost to communicate from current location
                        communication_cost = self.get_min_navigation_cost(rover, current_wp, self.comm_waypoints) + 1
                        if communication_cost == float('inf'): continue # Cannot reach comm spot

                    # Case 2: Sample is still at the waypoint and rover can sample it
                    elif w in at_soil_samples_in_state:
                        # Cost to navigate to sample + (drop if needed) + sample
                        nav_to_sample = self.get_navigation_cost(rover, current_wp, w)
                        if nav_to_sample == float('inf'): continue # Cannot reach sample spot

                        # Check if rover has a store and if it's full
                        store = self.rover_store.get(rover)
                        drop_cost = 1 if store and rover_store_full.get(rover, False) else 0
                        sample_cost = 1
                        acquisition_cost = nav_to_sample + drop_cost + sample_cost

                        # Cost to communicate *after* sampling (rover is at w)
                        communication_cost = self.get_min_navigation_cost(rover, w, self.comm_waypoints) + 1
                        if communication_cost == float('inf'): continue # Cannot reach comm spot

                    # Case 3: Sample is gone but was initially present (assume data exists on *some* equipped rover)
                    # If the sample is gone, and no rover currently holds the data, it implies the data was dropped.
                    # In this domain, dropped data is lost. So the goal is impossible.
                    # However, the simpler heuristic approach is to assume the data *is* held by *some* equipped rover
                    # if it was initially present and is now gone. Let's use this simpler (non-admissible) version.
                    elif w in self.initial_soil_samples:
                         # Assume data exists on this equipped rover
                         acquisition_cost = 0
                         communication_cost = self.get_min_navigation_cost(rover, current_wp, self.comm_waypoints) + 1
                         if communication_cost == float('inf'): continue # Cannot reach comm spot
                    # Else: Sample was never there or is gone and wasn't initial -> impossible for this rover

                    # Combine acquisition and communication costs for this rover
                    if acquisition_cost != float('inf') and communication_cost != float('inf'):
                         min_cost_for_goal = min(min_cost_for_goal, acquisition_cost + communication_cost)


                if min_cost_for_goal == float('inf'):
                    # If no equipped rover can achieve this goal (either acquire or communicate)
                    return float('inf') # Goal is unreachable

                total_cost += min_cost_for_goal

            elif pred == 'communicated_rock_data':
                w = parts[1]
                min_cost_for_goal = float('inf')

                # Find rovers that are equipped for rock analysis
                equipped_rovers = {r for r, equip in self.rover_equipment.items() if 'rock' in equip}

                if not equipped_rovers:
                     return float('inf') # Cannot achieve this goal

                for rover in equipped_rovers:
                    current_wp = current_rover_locations.get(rover)
                    if current_wp is None: continue # Rover location unknown

                    acquisition_cost = float('inf')
                    communication_cost = float('inf')

                    # Case 1: Rover already has the data
                    if w in rover_has_rock_data.get(rover, set()):
                        acquisition_cost = 0
                        # Cost to communicate from current location
                        communication_cost = self.get_min_navigation_cost(rover, current_wp, self.comm_waypoints) + 1
                        if communication_cost == float('inf'): continue # Cannot reach comm spot

                    # Case 2: Sample is still at the waypoint and rover can sample it
                    elif w in at_rock_samples_in_state:
                        # Cost to navigate to sample + (drop if needed) + sample
                        nav_to_sample = self.get_navigation_cost(rover, current_wp, w)
                        if nav_to_sample == float('inf'): continue # Cannot reach sample spot

                        # Check if rover has a store and if it's full
                        store = self.rover_store.get(rover)
                        drop_cost = 1 if store and rover_store_full.get(rover, False) else 0
                        sample_cost = 1
                        acquisition_cost = nav_to_sample + drop_cost + sample_cost

                        # Cost to communicate *after* sampling (rover is at w)
                        communication_cost = self.get_min_navigation_cost(rover, w, self.comm_waypoints) + 1
                        if communication_cost == float('inf'): continue # Cannot reach comm spot

                    # Case 3: Sample is gone but was initially present (assume data exists on *some* equipped rover)
                    elif w in self.initial_rock_samples:
                         # Assume data exists on this equipped rover
                         acquisition_cost = 0
                         communication_cost = self.get_min_navigation_cost(rover, current_wp, self.comm_waypoints) + 1
                         if communication_cost == float('inf'): continue # Cannot reach comm spot
                    # Else: Sample was never there or is gone and wasn't initial -> impossible for this rover

                    # Combine acquisition and communication costs for this rover
                    if acquisition_cost != float('inf') and communication_cost != float('inf'):
                         min_cost_for_goal = min(min_cost_for_goal, acquisition_cost + communication_cost)

                if min_cost_for_goal == float('inf'):
                    return float('inf') # Goal is unreachable

                total_cost += min_cost_for_goal

            elif pred == 'communicated_image_data':
                o, m = parts[1], parts[2]
                min_cost_for_goal = float('inf')

                # Find rovers that are equipped for imaging and have a camera supporting the mode
                suitable_rovers_cameras = [] # List of (rover, camera) tuples
                for rover in self.rover_equipment:
                    if 'imaging' in self.rover_equipment[rover]:
                        for camera in self.rover_cameras.get(rover, set()):
                            if m in self.camera_modes.get(camera, set()):
                                suitable_rovers_cameras.append((rover, camera))
                                # Assume one suitable camera per rover/mode is enough for the heuristic
                                break

                if not suitable_rovers_cameras:
                     return float('inf') # Cannot achieve this goal

                # Find image waypoints for objective o
                image_wps = self.objective_image_waypoints.get(o, set())
                if not image_wps:
                     return float('inf') # No waypoint to view objective o from

                for rover, camera in suitable_rovers_cameras:
                    current_wp = current_rover_locations.get(rover)
                    if current_wp is None: continue # Rover location unknown

                    acquisition_cost = float('inf')
                    communication_cost = float('inf')

                    # Case 1: Rover already has the image
                    if (o, m) in rover_has_image_data.get(rover, set()):
                        acquisition_cost = 0
                        # Cost to communicate from current location
                        communication_cost = self.get_min_navigation_cost(rover, current_wp, self.comm_waypoints) + 1
                        if communication_cost == float('inf'): continue # Cannot reach comm spot

                    # Case 2: Image needs to be acquired
                    else:
                        # Find calibration waypoints for this camera
                        cal_target = self.camera_calibration_target.get(camera)
                        cal_wps = self.objective_calibration_waypoints.get(cal_target, set()) if cal_target else set()

                        if not cal_wps:
                            # No waypoint to calibrate this camera -> impossible for this rover/camera
                            continue

                        # Estimate cost assuming sequence: current -> cal_wp -> image_wp -> comm_wp
                        # Nav curr to cal + calibrate + Nav cal to image + take_image + Nav image to comm + communicate

                        # Nav cost from current_wp to closest calibration waypoint
                        nav_curr_to_cal = self.get_min_navigation_cost(rover, current_wp, cal_wps)
                        if nav_curr_to_cal == float('inf'): continue

                        # Nav cost from closest calibration waypoint to closest image waypoint
                        # Use min distance from *any* cal_wp to *any* image_wp as approximation
                        nav_cal_to_image = self.get_min_navigation_cost_set_to_set(rover, cal_wps, image_wps)
                        if nav_cal_to_image == float('inf'): continue

                        # Nav cost from closest image waypoint to closest comm waypoint
                        # Use min distance from *any* image_wp to *any* comm_wp as approximation
                        nav_image_to_comm = self.get_min_navigation_cost_set_to_set(rover, image_wps, self.comm_waypoints)
                        if nav_image_to_comm == float('inf'): continue

                        # Total acquisition + communication cost for this rover/camera sequence
                        # Nav curr to cal + calibrate + Nav cal to image + take_image + Nav image to comm + communicate
                        acquisition_cost = nav_curr_to_cal + 1 + nav_cal_to_image + 1
                        communication_cost = nav_image_to_comm + 1

                        cost_this_rover = acquisition_cost + communication_cost
                        min_cost_for_goal = min(min_cost_for_goal, cost_this_rover)


                if min_cost_for_goal == float('inf'):
                    return float('inf') # Goal is unreachable

                total_cost += min_cost_for_goal

            # Add other goal types if any (only soil, rock, image in this domain)

        return total_cost

