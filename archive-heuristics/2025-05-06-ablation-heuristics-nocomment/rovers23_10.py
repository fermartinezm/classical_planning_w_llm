import sys
from collections import deque
from fnmatch import fnmatch
# Assuming heuristic_base is available in the parent directory or PYTHONPATH
from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL fact strings
def get_parts(fact):
    """Removes parentheses and splits the fact string into parts."""
    # Handle potential empty fact string or malformed fact
    if not fact or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()

# Helper function for pattern matching facts
def match(fact, *args):
    """Checks if a fact matches a pattern."""
    parts = get_parts(fact)
    # Ensure we don't go out of bounds if fact has fewer parts than args
    if len(parts) < len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# Function to compute shortest paths using BFS
def bfs(graph, start_node, all_nodes):
    """Computes shortest path distances from a start_node in a graph."""
    distances = {node: float('inf') for node in all_nodes}

    if start_node in distances:
        distances[start_node] = 0
        queue = deque([start_node])
    else:
        # Start node is not a known waypoint
        return distances # All distances remain inf

    while queue:
        current_node = queue.popleft()

        # Check if current_node has neighbors in the graph
        if current_node in graph:
            for neighbor in graph[current_node]:
                if distances.get(neighbor, float('inf')) == float('inf'): # Use .get for safety
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)
    return distances

class roversHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Rovers domain.

    Summary:
    This heuristic estimates the cost to reach the goal by summing up the
    estimated costs for each unachieved goal fact. For each unachieved
    communication goal (soil, rock, or image), it calculates the minimum
    cost required to first obtain the necessary data (sampling or imaging)
    and then communicate it. The cost calculation considers navigation
    (using precomputed shortest paths), sampling/dropping, imaging/calibration,
    and communication actions. It selects the most efficient rover and
    waypoints (sample/image location, calibration location, communication
    location) for each individual goal independently.

    Assumptions:
    - The PDDL task is well-formed and follows the rovers domain structure.
    - Static facts correctly define rover capabilities, camera properties,
      waypoint visibility, and traversal paths.
    - Goals are communication goals.
    - If a sample is no longer at its initial location (`at_soil_sample`/`at_rock_sample` is false)
      and no rover currently holds the corresponding analysis data (`have_soil_analysis`/`have_rock_analysis`),
      that specific soil/rock data goal is considered unachievable from this state
      and contributes 0 to the heuristic.
    - Calibration is only needed if the camera is not currently calibrated.
      The cost includes navigation to a calibration target waypoint, calibration,
      and navigation back to the image waypoint.
    - The heuristic sums costs for individual goals, ignoring potential
      synergies or conflicts (e.g., one rover collecting multiple samples,
      multiple rovers needing the same waypoint). This makes it non-admissible
      but potentially effective for greedy search.
    - Unreachable waypoints via navigation result in infinite distance,
      which correctly makes a goal requiring that navigation step have
      infinite cost unless an alternative path/rover exists. If a goal
      is determined to be truly impossible (e.g., sample gone), its cost is 0.
    - Rover identification in the state is done by checking against the set
      of rovers identified in the static facts during initialization.

    Heuristic Initialization:
    The constructor processes the static facts from the task description.
    - It identifies the lander's location.
    - It maps rovers to their capabilities (soil, rock, imaging) and stores.
    - It maps cameras to the rovers they are on, the modes they support,
      and their calibration targets.
    - It maps objectives/targets to the waypoints from which they are visible.
    - It builds a navigation graph for each rover based on `can_traverse` facts.
    - It collects all unique objects by type (rover, waypoint, camera, etc.)
      mentioned in static facts.
    - For each rover, it precomputes all-pairs shortest path distances using
      Breadth-First Search (BFS) on their respective navigation graphs,
      considering all known waypoints.
    - It identifies communication waypoints (those visible from the lander's location).

    Step-By-Step Thinking for Computing Heuristic:
    1. Initialize total heuristic cost to 0.
    2. Parse the current state to quickly access facts like rover locations,
       held data (`have_soil_analysis`, `have_rock_analysis`, `have_image`),
       camera calibration status (`calibrated`), store status (`full`),
       and remaining sample locations (`at_soil_sample`, `at_rock_sample`).
       Rover locations are identified by checking if the object is in the set
       of rovers collected during initialization.
    3. Iterate through each goal fact defined in the task.
    4. For an unachieved goal fact `g`:
        a. Initialize the cost for this specific goal (`goal_cost`) to infinity.
        b. If `g` is `(communicated_soil_data ?w)`:
            i. Check if any rover `?r` currently has `(have_soil_analysis ?r ?w)`.
               If yes, calculate the minimum cost for such a rover to navigate
               from its current location to any communication waypoint and communicate (navigate_cost + 1).
               Update `goal_cost` with the minimum found.
            ii. If no rover has the data (and `goal_cost` is still infinity),
                check if `(at_soil_sample ?w)` is true in the state.
                If yes, find soil-equipped rovers `?r`. For each such rover,
                calculate the cost to navigate to `?w`, sample (potentially dropping first),
                navigate from `?w` to a communication waypoint, and communicate.
                Cost = `dist(?r_loc, ?w) + (1 if store full) + 1 (sample) + dist(?w, comm_wp) + 1 (communicate)`.
                Update `goal_cost` with the minimum cost over all suitable rovers and communication waypoints.
            iii. If `(at_soil_sample ?w)` is false (and no rover has data), the goal is
                considered unachievable from this state; set `goal_cost` to 0.
        c. If `g` is `(communicated_rock_data ?w)`: Similar logic as soil data,
           using rock-specific predicates and equipped rovers.
        d. If `g` is `(communicated_image_data ?o ?m)`:
            i. Check if any rover `?r` currently has `(have_image ?r ?o ?m)`.
               If yes, calculate the minimum cost for such a rover to navigate
               from its current location to any communication waypoint and communicate (navigate_cost + 1).
               Update `goal_cost` with the minimum found.
            ii. If no rover has the image (and `goal_cost` is still infinity),
                find imaging-equipped rovers `?r` with cameras `?i` supporting `?m`.
                Find waypoints `?p` where `(visible_from ?o ?p)` is true.
                For each suitable rover `?r` and image waypoint `?p`:
                    Calculate the cost to navigate to `?p` (`dist(?r_loc, ?p)`).
                    If camera `?i` is not calibrated for `?r`, calculate the minimum
                    calibration cost: navigate from `?p` to a calibration waypoint `?w_cal`
                    for `?i`'s target, calibrate (1), navigate back to `?p`.
                    Calibration cost = `dist(?p, ?w_cal) + 1 + dist(?w_cal, ?p)`.
                    Add this calibration cost (if needed) and the take_image cost (1).
                    Calculate the minimum cost to navigate from `?p` to any communication
                    waypoint and communicate (`dist(?p, comm_wp) + 1`).
                    Total cost for this (`?r`, `?p`) combination = `dist(?r_loc, ?p) + (calibration_cost if needed) + 1 (take_image) + dist(?p, comm_wp) + 1 (communicate)`.
                    Update `goal_cost` with the minimum cost over all suitable rovers,
                    image waypoints, and calibration waypoints.
            iii. If no suitable rover/camera/image waypoint exists (and `goal_cost` is still infinity),
                 the goal is considered unachievable from this state; set `goal_cost` to 0.
        e. Add the calculated `goal_cost` to the `total_heuristic`.
    5. Return the `total_heuristic`. If the total heuristic is 0 but the state is not a goal state, return 1 to ensure the heuristic is 0 only at the goal.
    """
    def __init__(self, task):
        self.goals = task.goals
        static_facts = task.static

        # Data structures for static information
        self.lander_location = {} # {lander: waypoint}
        self.rover_capabilities = {} # {rover: set(capabilities)} e.g., {'rover1': {'soil', 'imaging'}}
        self.rover_stores = {} # {rover: store} e.g., {'rover1': 'rover1store'}
        self.rover_traversal_graphs = {} # {rover: {waypoint: set(reachable_waypoints)}}
        self.camera_on_rover = {} # {camera: rover}
        self.camera_modes = {} # {camera: set(modes)}
        self.camera_calibration_target = {} # {camera: target_objective}
        self.objective_visibility = {} # {objective: set(visible_waypoints)}
        self.target_visibility = {} # {target_objective: set(visible_waypoints)} # Targets are objectives used for calibration
        self.waypoint_visibility = {} # {waypoint: set(visible_waypoints)} # Used for communication waypoints

        # Collect all objects by type (approximation based on usage in static facts)
        self.all_rovers = set()
        self.all_waypoints = set()
        self.all_cameras = set()
        self.all_objectives = set()
        self.all_modes = set()
        self.all_landers = set()
        self.all_stores = set()

        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts
            predicate = parts[0]

            if predicate == 'at_lander':
                lander, waypoint = parts[1], parts[2]
                self.lander_location[lander] = waypoint
                self.all_landers.add(lander)
                self.all_waypoints.add(waypoint)
            elif predicate.startswith('equipped_for_'):
                rover = parts[1]
                capability = predicate.split('_')[-1] # soil, rock, imaging
                self.rover_capabilities.setdefault(rover, set()).add(capability)
                self.all_rovers.add(rover)
            elif predicate == 'store_of':
                store, rover = parts[1], parts[2]
                self.rover_stores[rover] = store
                self.all_stores.add(store)
                self.all_rovers.add(rover)
            elif predicate == 'visible':
                w1, w2 = parts[1], parts[2]
                self.waypoint_visibility.setdefault(w1, set()).add(w2)
                self.waypoint_visibility.setdefault(w2, set()).add(w1) # Visibility is symmetric
                self.all_waypoints.add(w1)
                self.all_waypoints.add(w2)
            elif predicate == 'can_traverse':
                rover, w1, w2 = parts[1], parts[2], parts[3]
                self.rover_traversal_graphs.setdefault(rover, {}).setdefault(w1, set()).add(w2)
                self.all_rovers.add(rover)
                self.all_waypoints.add(w1)
                self.all_waypoints.add(w2)
            elif predicate == 'on_board':
                camera, rover = parts[1], parts[2]
                self.camera_on_rover[camera] = rover
                self.all_cameras.add(camera)
                self.all_rovers.add(rover)
            elif predicate == 'supports':
                camera, mode = parts[1], parts[2]
                self.camera_modes.setdefault(camera, set()).add(mode)
                self.all_cameras.add(camera)
                self.all_modes.add(mode)
            elif predicate == 'calibration_target':
                camera, objective = parts[1], parts[2]
                self.camera_calibration_target[camera] = objective # objective acts as target
                self.all_cameras.add(camera)
                self.all_objectives.add(objective)
            elif predicate == 'visible_from':
                objective, waypoint = parts[1], parts[2]
                self.objective_visibility.setdefault(objective, set()).add(waypoint)
                self.target_visibility.setdefault(objective, set()).add(waypoint) # Assume any objective can be a target if visible_from is listed
                self.all_objectives.add(objective)
                self.all_waypoints.add(waypoint)

        # Ensure all waypoints mentioned in traversal graphs or visibility are included
        for rover_graph in self.rover_traversal_graphs.values():
             for wp in rover_graph:
                 self.all_waypoints.add(wp)
                 for neighbor in rover_graph[wp]:
                     self.all_waypoints.add(neighbor)
        for wp in self.waypoint_visibility:
            self.all_waypoints.add(wp)
            for neighbor in self.waypoint_visibility[wp]:
                self.all_waypoints.add(neighbor)


        # Compute shortest paths for each rover
        self.rover_distances = {} # {rover: {from_wp: {to_wp: distance}}}
        for rover in self.all_rovers:
            self.rover_distances[rover] = {}
            # Get the traversal graph for this rover, or an empty graph if none defined
            rover_graph = self.rover_traversal_graphs.get(rover, {})
            # Ensure all known waypoints are in the graph dictionary keys for BFS, even if they have no edges
            graph_with_all_wps = {wp: set() for wp in self.all_waypoints}
            for wp, neighbors in rover_graph.items():
                 graph_with_all_wps[wp] = neighbors

            for start_wp in self.all_waypoints:
                 self.rover_distances[rover][start_wp] = bfs(graph_with_all_wps, start_wp, self.all_waypoints)

        # Identify communication waypoints
        self.communication_waypoints = set()
        # Assuming there's at least one lander
        if self.lander_location:
            lander_wp = list(self.lander_location.values())[0]
            self.communication_waypoints = self.waypoint_visibility.get(lander_wp, set())

        # Map goals for easier lookup
        self.soil_goals = set() # {waypoint}
        self.rock_goals = set() # {waypoint}
        self.image_goals = set() # {(objective, mode)}
        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue
            predicate = parts[0]
            if predicate == 'communicated_soil_data':
                if len(parts) > 1: self.soil_goals.add(parts[1])
            elif predicate == 'communicated_rock_data':
                if len(parts) > 1: self.rock_goals.add(parts[1])
            elif predicate == 'communicated_image_data':
                if len(parts) > 2: self.image_goals.add((parts[1], parts[2]))


    def __call__(self, node):
        state = node.state

        # Parse current state for quick access
        rover_locations = {} # {rover: waypoint}
        have_soil = {} # {rover: set(waypoints)}
        have_rock = {} # {rover: set(waypoints)}
        have_image = {} # {rover: set((objective, mode))}
        calibrated_cameras = set() # {(camera, rover)}
        full_stores = set() # {store}
        at_soil_sample = set() # {waypoint}
        at_rock_sample = set() # {waypoint}

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue
            predicate = parts[0]

            if predicate == 'at':
                obj, loc = parts[1], parts[2]
                # Identify rovers based on the list collected in __init__
                if obj in self.all_rovers:
                     rover_locations[obj] = loc
            elif predicate == 'have_soil_analysis':
                rover, waypoint = parts[1], parts[2]
                have_soil.setdefault(rover, set()).add(waypoint)
            elif predicate == 'have_rock_analysis':
                rover, waypoint = parts[1], parts[2]
                have_rock.setdefault(rover, set()).add(waypoint)
            elif predicate == 'have_image':
                rover, objective, mode = parts[1], parts[2], parts[3]
                have_image.setdefault(rover, set()).add((objective, mode))
            elif predicate == 'calibrated':
                camera, rover = parts[1], parts[2]
                calibrated_cameras.add((camera, rover))
            elif predicate == 'full':
                store = parts[1]
                full_stores.add(store)
            elif predicate == 'at_soil_sample':
                waypoint = parts[1]
                at_soil_sample.add(waypoint)
            elif predicate == 'at_rock_sample':
                waypoint = parts[1]
                at_rock_sample.add(waypoint)

        total_heuristic = 0

        # Evaluate Soil Goals
        for waypoint in self.soil_goals:
            goal_fact = f'(communicated_soil_data {waypoint})'
            if goal_fact in state:
                continue # Goal already achieved

            goal_cost = float('inf')

            # Case 1: Data is already collected by a rover
            rover_with_data = None
            for rover, waypoints_held in have_soil.items():
                if waypoint in waypoints_held:
                    rover_with_data = rover
                    break

            if rover_with_data:
                current_wp = rover_locations.get(rover_with_data)
                if current_wp and rover_with_data in self.rover_distances and current_wp in self.rover_distances[rover_with_data]:
                    # Find min dist to a communication waypoint
                    min_dist_to_comm = float('inf')
                    for comm_wp in self.communication_waypoints:
                         dist = self.rover_distances[rover_with_data][current_wp].get(comm_wp, float('inf'))
                         min_dist_to_comm = min(min_dist_to_comm, dist)

                    if min_dist_to_comm != float('inf'):
                         goal_cost = min(goal_cost, min_dist_to_comm + 1) # +1 for communicate action

            # Case 2: Sample is still at the waypoint
            if goal_cost == float('inf') and waypoint in at_soil_sample:
                 # Find best soil-equipped rover
                 best_rover_cost = float('inf')
                 for rover in self.all_rovers: # Iterate through all rovers
                      if 'soil' in self.rover_capabilities.get(rover, set()): # Check if equipped
                           current_wp = rover_locations.get(rover)
                           if current_wp and rover in self.rover_distances and current_wp in self.rover_distances[rover]:
                                # Cost to navigate to sample
                                dist_to_sample = self.rover_distances[rover][current_wp].get(waypoint, float('inf'))

                                if dist_to_sample != float('inf'):
                                     # Cost for sampling
                                     sample_cost = dist_to_sample + 1 # +1 for sample_soil action
                                     # Check if drop is needed
                                     rover_store = self.rover_stores.get(rover)
                                     if rover_store and rover_store in full_stores:
                                          sample_cost += 1 # +1 for drop action

                                     # Cost to navigate from sample location to comm waypoint
                                     min_dist_from_sample_to_comm = float('inf')
                                     if waypoint in self.rover_distances.get(rover, {}): # Ensure sample waypoint is traversable by this rover
                                         for comm_wp in self.communication_waypoints:
                                              dist = self.rover_distances[rover][waypoint].get(comm_wp, float('inf'))
                                              min_dist_from_sample_to_comm = min(min_dist_from_sample_to_comm, dist)

                                     if min_dist_from_sample_to_comm != float('inf'):
                                          # Total cost for this rover
                                          rover_total_cost = sample_cost + min_dist_from_sample_to_comm + 1 # +1 for communicate
                                          best_rover_cost = min(best_rover_cost, rover_total_cost)

                 if best_rover_cost != float('inf'):
                      goal_cost = min(goal_cost, best_rover_cost)

            # If goal is still unreachable (sample gone, or no equipped rover, or no path)
            if goal_cost == float('inf'):
                 goal_cost = 0 # Treat as unachievable from this state

            total_heuristic += goal_cost

        # Evaluate Rock Goals (similar to Soil Goals)
        for waypoint in self.rock_goals:
            goal_fact = f'(communicated_rock_data {waypoint})'
            if goal_fact in state:
                continue # Goal already achieved

            goal_cost = float('inf')

            # Case 1: Data is already collected by a rover
            rover_with_data = None
            for rover, waypoints_held in have_rock.items():
                if waypoint in waypoints_held:
                    rover_with_data = rover
                    break

            if rover_with_data:
                current_wp = rover_locations.get(rover_with_data)
                if current_wp and rover_with_data in self.rover_distances and current_wp in self.rover_distances[rover_with_data]:
                    # Find min dist to a communication waypoint
                    min_dist_to_comm = float('inf')
                    for comm_wp in self.communication_waypoints:
                         dist = self.rover_distances[rover_with_data][current_wp].get(comm_wp, float('inf'))
                         min_dist_to_comm = min(min_dist_to_comm, dist)

                    if min_dist_to_comm != float('inf'):
                         goal_cost = min(goal_cost, min_dist_to_comm + 1) # +1 for communicate action

            # Case 2: Sample is still at the waypoint
            if goal_cost == float('inf') and waypoint in at_rock_sample:
                 # Find best rock-equipped rover
                 best_rover_cost = float('inf')
                 for rover in self.all_rovers: # Iterate through all rovers
                      if 'rock' in self.rover_capabilities.get(rover, set()): # Check if equipped
                           current_wp = rover_locations.get(rover)
                           if current_wp and rover in self.rover_distances and current_wp in self.rover_distances[rover]:
                                # Cost to navigate to sample
                                dist_to_sample = self.rover_distances[rover][current_wp].get(waypoint, float('inf'))

                                if dist_to_sample != float('inf'):
                                     # Cost for sampling
                                     sample_cost = dist_to_sample + 1 # +1 for sample_rock action
                                     # Check if drop is needed
                                     rover_store = self.rover_stores.get(rover)
                                     if rover_store and rover_store in full_stores:
                                          sample_cost += 1 # +1 for drop action

                                     # Cost to navigate from sample location to comm waypoint
                                     min_dist_from_sample_to_comm = float('inf')
                                     if waypoint in self.rover_distances.get(rover, {}): # Ensure sample waypoint is traversable by this rover
                                         for comm_wp in self.communication_waypoints:
                                              dist = self.rover_distances[rover][waypoint].get(comm_wp, float('inf'))
                                              min_dist_from_sample_to_comm = min(min_dist_from_sample_to_comm, dist)

                                     if min_dist_from_sample_to_comm != float('inf'):
                                          # Total cost for this rover
                                          rover_total_cost = sample_cost + min_dist_from_sample_to_comm + 1 # +1 for communicate
                                          best_rover_cost = min(best_rover_cost, rover_total_cost)

                 if best_rover_cost != float('inf'):
                      goal_cost = min(goal_cost, best_rover_cost)

            # If goal is still unreachable
            if goal_cost == float('inf'):
                 goal_cost = 0 # Treat as unachievable from this state

            total_heuristic += goal_cost


        # Evaluate Image Goals
        for objective, mode in self.image_goals:
            goal_fact = f'(communicated_image_data {objective} {mode})'
            if goal_fact in state:
                continue # Goal already achieved

            goal_cost = float('inf')

            # Case 1: Image is already taken by a rover
            rover_with_image = None
            for rover, images_held in have_image.items():
                if (objective, mode) in images_held:
                    rover_with_image = rover
                    break

            if rover_with_image:
                current_wp = rover_locations.get(rover_with_image)
                if current_wp and rover_with_image in self.rover_distances and current_wp in self.rover_distances[rover_with_image]:
                    # Find min dist to a communication waypoint
                    min_dist_to_comm = float('inf')
                    for comm_wp in self.communication_waypoints:
                         dist = self.rover_distances[rover_with_image][current_wp].get(comm_wp, float('inf'))
                         min_dist_to_comm = min(min_dist_to_comm, dist)

                    if min_dist_to_comm != float('inf'):
                         goal_cost = min(goal_cost, min_dist_to_comm + 1) # +1 for communicate action

            # Case 2: Image needs to be taken
            if goal_cost == float('inf'):
                 # Find best imaging-equipped rover with suitable camera
                 best_rover_cost = float('inf')
                 for rover in self.all_rovers: # Iterate through all rovers
                      if 'imaging' in self.rover_capabilities.get(rover, set()): # Check if equipped
                           # Find cameras on this rover supporting the mode
                           suitable_cameras = [
                                cam for cam, r in self.camera_on_rover.items()
                                if r == rover and mode in self.camera_modes.get(cam, set())
                           ]

                           if suitable_cameras:
                                current_wp = rover_locations.get(rover)
                                if current_wp and rover in self.rover_distances and current_wp in self.rover_distances[rover]:
                                     # Find suitable image waypoints for this objective
                                     image_wps = self.objective_visibility.get(objective, set())

                                     if image_wps:
                                          # Iterate through suitable cameras and image waypoints
                                          for camera in suitable_cameras:
                                               calibration_target = self.camera_calibration_target.get(camera)
                                               # Calibration waypoints are those visible_from the calibration target
                                               calibration_wps = self.target_visibility.get(calibration_target, set()) if calibration_target else set()

                                               for img_wp in image_wps:
                                                    # Cost to navigate to image waypoint
                                                    dist_to_img_wp = self.rover_distances[rover][current_wp].get(img_wp, float('inf'))

                                                    if dist_to_img_wp != float('inf'):
                                                         # Cost for imaging
                                                         imaging_prep_cost = dist_to_img_wp # Navigation to image waypoint

                                                         # Check if calibration is needed
                                                         if (camera, rover) not in calibrated_cameras:
                                                              # Cost for calibration
                                                              min_calibration_nav_cost = float('inf')
                                                              if calibration_wps:
                                                                   if img_wp in self.rover_distances.get(rover, {}): # Ensure image waypoint is traversable
                                                                        for cal_wp in calibration_wps:
                                                                             if cal_wp in self.rover_distances.get(rover, {}): # Ensure calibration waypoint is traversable
                                                                                  dist_img_to_cal = self.rover_distances[rover][img_wp].get(cal_wp, float('inf'))
                                                                                  dist_cal_to_img = self.rover_distances[rover][cal_wp].get(img_wp, float('inf'))
                                                                                  if dist_img_to_cal != float('inf') and dist_cal_to_img != float('inf'):
                                                                                       min_calibration_nav_cost = min(min_calibration_nav_cost, dist_img_to_cal + 1 + dist_cal_to_img) # +1 for calibrate

                                                              if min_calibration_nav_cost != float('inf'):
                                                                   imaging_prep_cost += min_calibration_nav_cost
                                                              else:
                                                                   # Cannot calibrate from image waypoint, this path is impossible for this goal
                                                                   continue # Skip this img_wp/camera combination

                                                         imaging_prep_cost += 1 # +1 for take_image action

                                                         # Cost to navigate from image location to comm waypoint
                                                         min_dist_from_img_to_comm = float('inf')
                                                         if img_wp in self.rover_distances.get(rover, {}): # Ensure image waypoint is traversable by this rover
                                                             for comm_wp in self.communication_waypoints:
                                                                  dist = self.rover_distances[rover][img_wp].get(comm_wp, float('inf'))
                                                                  min_dist_from_img_to_comm = min(min_dist_from_img_to_comm, dist)

                                                         if min_dist_from_img_to_comm != float('inf'):
                                                              # Total cost for this rover/camera/img_wp combination
                                                              combination_total_cost = imaging_prep_cost + min_dist_from_img_to_comm + 1 # +1 for communicate
                                                              best_rover_cost = min(best_rover_cost, combination_total_cost)


                 if best_rover_cost != float('inf'):
                      goal_cost = min(goal_cost, best_rover_cost)

            # If goal is still unreachable
            if goal_cost == float('inf'):
                 goal_cost = 0 # Treat as unachievable from this state

            total_heuristic += goal_cost


        # Ensure heuristic is 0 only at goal state
        # Check if all goals are met
        all_goals_met = True
        for goal in self.goals:
            if goal not in state:
                all_goals_met = False
                break

        if total_heuristic == 0 and not all_goals_met:
             # This can happen if all unachieved goals were deemed unreachable (cost 0)
             # Return 1 to distinguish from a true goal state
             return 1

        return total_heuristic
