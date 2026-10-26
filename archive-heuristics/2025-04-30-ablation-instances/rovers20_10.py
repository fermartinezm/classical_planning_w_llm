from fnmatch import fnmatch
import collections
from heuristics.heuristic_base import Heuristic

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
    # Ensure we don't compare more args than parts or vice versa
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def bfs(graph, start_node):
    """
    Performs Breadth-First Search on a graph to find shortest distances
    from a start node to all other nodes.
    """
    distances = {node: float('inf') for node in graph}
    if start_node not in graph:
        # Start node is not in the graph (e.g., waypoint not identified)
        return distances # All distances remain inf

    distances[start_node] = 0
    queue = collections.deque([start_node])
    while queue:
        current_node = queue.popleft()
        # Handle potential missing neighbors key gracefully
        for neighbor in graph.get(current_node, set()):
            if distances[neighbor] == float('inf'):
                distances[neighbor] = distances[current_node] + 1
                queue.append(neighbor)
    return distances


class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve all goal
    conditions, which are typically communicating collected data (soil, rock,
    images) to the lander. It sums the estimated costs for each unachieved
    communication goal independently, ignoring potential synergies or conflicts
    between goals or rovers.

    # Assumptions
    - Each unachieved communication goal is pursued by a single, suitable rover.
      If the required data already exists with *any* rover, the cost is just
      communication from that rover's current location. Otherwise, a suitable
      rover must first collect the data (sample or image) and then communicate it.
    - The cost of achieving a data piece (sample or image) and then communicating
      it is estimated sequentially.
    - Navigation cost is estimated using shortest path distance in the waypoint
      graph specific to the assigned rover.
    - Resource constraints (like limited store capacity) are partially considered
      for sampling (need to drop if store is full).
    - Camera calibration state is considered for imaging.
    - The heuristic assumes the problem is solvable and required objects (rovers,
      cameras, waypoints, etc.) exist and are configured appropriately based on
      static facts.
    - Assumes a single lander named 'general' (or picks the first one found).

    # Heuristic Initialization
    The constructor precomputes static information from the task:
    - Identifies all objects (rovers, waypoints, stores, cameras, modes, objectives, landers)
      from initial state and static facts.
    - Stores lander location(s).
    - Stores capabilities of each rover (soil, rock, imaging).
    - Maps store to its rover.
    - Maps camera to its rover, supported modes, and calibration target.
    - Maps objective to waypoints visible from it.
    - Stores waypoint visibility relationships (symmetric).
    - Builds the waypoint graph for each rover based on traversable and visible links.
    - Computes all-pairs shortest path distances for each rover in their respective graphs using BFS.
    - Identifies waypoints visible from the lander.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic is computed as follows:

    1. Initialize total heuristic value `h = 0`.
    2. Extract the current state of all relevant predicates (rover locations,
       data collected, camera calibration, store status).
    3. Iterate through each goal fact specified in the task.
    4. If a goal fact is already true in the current state, contribute 0 to `h`.
    5. If a goal fact is not true, estimate the cost to achieve it and add to `h`.
       The estimation depends on the type of communication goal:

       a.  **Communicating Soil Data `(communicated_soil_data ?w)`:**
           - Check if `(have_soil_analysis ?r_any ?w)` is true for *any* rover `?r_any`.
           - If data exists: Find the first rover `?r_any` that has it. Get its current location `?r_loc`. Find the closest waypoint `?comm_wp` visible from the lander, reachable by `?r_any`. Add navigation cost from `?r_loc` to `?comm_wp` + 1 (communicate).
           - If data does not exist: Find the first rover `?r` equipped for soil. Get its current location `?r_loc`. Calculate cost to sample: navigation from `?r_loc` to `?w` + (1 if `?r`'s store is full) + 1 (sample). Update `?r_loc` to `?w`. Calculate cost to communicate: navigation from new `?r_loc` to closest lander-visible waypoint + 1 (communicate). Sum these costs.

       b.  **Communicating Rock Data `(communicated_rock_data ?w)`:**
           - Similar logic to soil data, using rock-specific predicates and a rock-equipped rover.

       c.  **Communicating Image Data `(communicated_image_data ?o ?m)`:**
           - Check if `(have_image ?r_any ?o ?m)` is true for *any* rover `?r_any`.
           - If data exists: Find the first rover `?r_any` that has it. Get its current location `?r_loc`. Find the closest waypoint `?comm_wp` visible from the lander, reachable by `?r_any`. Add navigation cost from `?r_loc` to `?comm_wp` + 1 (communicate).
           - If data does not exist: Find the first rover `?r` equipped for imaging. Find the first camera `?i` on `?r` supporting `?m`. Get `?r_loc`. Calculate cost to image:
               - If `?i` is not calibrated: Find calibration target `?t` for `?i`. Find closest waypoint `?cal_wp` visible from `?t`, reachable by `?r`. Add navigation cost from `?r_loc` to `?cal_wp` + 1 (calibrate). Update `?r_loc` to `?cal_wp`.
               - Find closest waypoint `?img_wp` visible from `?o`, reachable by `?r`. Add navigation cost from current `?r_loc` to `?img_wp` + 1 (take_image). Update `?r_loc` to `?img_wp`.
           - Calculate cost to communicate: navigation from new `?r_loc` to closest lander-visible waypoint + 1 (communicate). Sum these costs.

    6. The total `h` is the sum of costs for all unachieved goals.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information and
        precomputing navigation distances.
        """
        self.goals = task.goals
        static_facts = task.static
        initial_state = task.initial_state # Need initial state to find all objects

        # --- Extract all objects ---
        self.rovers = set()
        self.waypoints = set()
        self.stores = set()
        self.cameras = set()
        self.modes = set()
        self.objectives = set()
        self.landers = set()

        # Collect all terms that appear in facts to find potential objects
        all_terms = set()
        for fact in static_facts | initial_state:
            all_terms.update(get_parts(fact))

        # Classify terms based on predicate positions and common naming conventions
        # This is heuristic-specific and might need adjustment based on problem file specifics
        for term in all_terms:
            if term.startswith('rover'): self.rovers.add(term)
            elif term.startswith('waypoint'): self.waypoints.add(term)
            elif term.startswith('store'): self.stores.add(term)
            elif term.startswith('camera'): self.cameras.add(term)
            elif term.startswith('mode'): self.modes.add(term) # Modes often don't have type predicates
            elif term.startswith('objective'): self.objectives.add(term)
            elif term.startswith('lander'): self.landers.add(term)
            # Add other potential object types if needed

        # Ensure all objects mentioned in static facts are captured
        for fact in static_facts:
             parts = get_parts(fact)
             if parts[0] == 'at_lander' and len(parts) == 3: self.landers.add(parts[1]); self.waypoints.add(parts[2])
             elif parts[0] == 'can_traverse' and len(parts) == 4: self.rovers.add(parts[1]); self.waypoints.add(parts[2]); self.waypoints.add(parts[3])
             elif parts[0] == 'equipped_for_soil_analysis' and len(parts) == 2: self.rovers.add(parts[1])
             elif parts[0] == 'equipped_for_rock_analysis' and len(parts) == 2: self.rovers.add(parts[1])
             elif parts[0] == 'equipped_for_imaging' and len(parts) == 2: self.rovers.add(parts[1])
             elif parts[0] == 'store_of' and len(parts) == 3: self.stores.add(parts[1]); self.rovers.add(parts[2])
             elif parts[0] == 'calibrated' and len(parts) == 3: self.cameras.add(parts[1]); self.rovers.add(parts[2])
             elif parts[0] == 'supports' and len(parts) == 3: self.cameras.add(parts[1]); self.modes.add(parts[2])
             elif parts[0] == 'visible' and len(parts) == 3: self.waypoints.add(parts[1]); self.waypoints.add(parts[2])
             elif parts[0] == 'visible_from' and len(parts) == 3: self.objectives.add(parts[1]); self.waypoints.add(parts[2])
             elif parts[0] == 'calibration_target' and len(parts) == 3: self.cameras.add(parts[1]); self.objectives.add(parts[2])
             elif parts[0] == 'on_board' and len(parts) == 3: self.cameras.add(parts[1]); self.rovers.add(parts[2])

        # --- Extract static information ---
        self.lander_location = {} # lander -> waypoint
        self.rover_capabilities = {r: {'soil': False, 'rock': False, 'imaging': False} for r in self.rovers}
        self.rover_store = {} # rover -> store
        self.camera_on_rover = {} # camera -> rover
        self.camera_supports = {c: set() for c in self.cameras} # camera -> set of modes
        self.camera_calibration_target = {} # camera -> objective
        self.objective_visible_wps = {o: set() for o in self.objectives} # objective -> set of waypoints
        self.waypoint_visibility = {w: set() for w in self.waypoints} # waypoint -> set of visible waypoints (symmetric)
        self.rover_traversable_edges = {r: set() for r in self.rovers} # rover -> set of (w1, w2) tuples (undirected)

        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'at_lander':
                self.lander_location[parts[1]] = parts[2]
            elif parts[0] == 'equipped_for_soil_analysis':
                if parts[1] in self.rover_capabilities: self.rover_capabilities[parts[1]]['soil'] = True
            elif parts[0] == 'equipped_for_rock_analysis':
                if parts[1] in self.rover_capabilities: self.rover_capabilities[parts[1]]['rock'] = True
            elif parts[0] == 'equipped_for_imaging':
                if parts[1] in self.rover_capabilities: self.rover_capabilities[parts[1]]['imaging'] = True
            elif parts[0] == 'store_of':
                if parts[2] in self.rovers and parts[1] in self.stores: self.rover_store[parts[2]] = parts[1] # rover -> store
            elif parts[0] == 'on_board':
                if parts[1] in self.cameras and parts[2] in self.rovers: self.camera_on_rover[parts[1]] = parts[2] # camera -> rover
            elif parts[0] == 'supports':
                if parts[1] in self.camera_supports and parts[2] in self.modes: # Ensure camera and mode were identified
                    self.camera_supports[parts[1]].add(parts[2])
            elif parts[0] == 'calibration_target':
                 if parts[1] in self.camera_calibration_target and parts[2] in self.objectives: # Ensure camera and objective were identified
                    self.camera_calibration_target[parts[1]] = parts[2]
            elif parts[0] == 'visible':
                w1, w2 = parts[1], parts[2]
                if w1 in self.waypoint_visibility and w2 in self.waypoint_visibility: # Ensure waypoints were identified
                    self.waypoint_visibility[w1].add(w2)
                    self.waypoint_visibility[w2].add(w1) # Visibility is symmetric
            elif parts[0] == 'can_traverse':
                 r, w1, w2 = parts[1], parts[2], parts[3]
                 if r in self.rover_traversable_edges and w1 in self.waypoints and w2 in self.waypoints: # Ensure objects were identified
                     # An edge exists for rover r between w1 and w2 if both can_traverse and visible hold
                     # Check visibility explicitly here
                     if w2 in self.waypoint_visibility.get(w1, set()):
                         self.rover_traversable_edges[r].add(tuple(sorted((w1, w2)))) # Store as undirected edge

        # Re-iterate for visible_from after objective_visible_wps is initialized
        for fact in static_facts:
             parts = get_parts(fact)
             if parts[0] == 'visible_from':
                 obj, wp = parts[1], parts[2]
                 if obj in self.objectives and wp in self.waypoints: # Ensure objects were identified
                    self.objective_visible_wps[obj].add(wp)


        # --- Precompute shortest paths for each rover ---
        self.rover_shortest_paths = {}
        for rover in self.rovers:
            graph = {wp: set() for wp in self.waypoints}
            # Build undirected graph based on traversable edges
            for w1, w2 in self.rover_traversable_edges.get(rover, set()):
                 if w1 in graph and w2 in graph: # Ensure waypoints exist
                    graph[w1].add(w2)
                    graph[w2].add(w1)

            self.rover_shortest_paths[rover] = {}
            for start_wp in self.waypoints:
                self.rover_shortest_paths[rover][start_wp] = bfs(graph, start_wp)

        # --- Identify waypoints visible from the lander ---
        self.lander_visible_waypoints = set()
        # Assuming there's at least one lander and its location is known
        if self.lander_location:
            lander_loc = list(self.lander_location.values())[0] # Assuming one lander
            for wp in self.waypoints:
                if lander_loc in self.waypoint_visibility.get(wp, set()):
                    self.lander_visible_waypoints.add(wp)


    def get_distance(self, rover, w1, w2):
        """Get the precomputed shortest distance for a rover between two waypoints."""
        # Handle cases where rover, w1, or w2 might not be valid or reachable
        if rover not in self.rovers or w1 not in self.waypoints or w2 not in self.waypoints:
             return 1000 # Large cost for invalid inputs

        dist = self.rover_shortest_paths.get(rover, {}).get(w1, {}).get(w2, float('inf'))

        # If distance is inf, it means unreachable. Return a large cost.
        return dist if dist != float('inf') else 1000 # Use a large number for unreachable

    def find_closest_waypoint(self, start_wp, rover, target_wps):
        """Find the waypoint in target_wps closest to start_wp for the given rover."""
        min_dist = float('inf')
        closest_wp = None
        if not target_wps:
            return None, float('inf') # No target waypoints

        for target_wp in target_wps:
            dist = self.get_distance(rover, start_wp, target_wp)
            if dist < min_dist:
                min_dist = dist
                closest_wp = target_wp
        return closest_wp, min_dist # Return both the waypoint and the distance

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        h = 0

        # --- Extract current state information ---
        at_rover = {} # rover -> waypoint
        have_soil = {} # rover -> set of waypoints
        have_rock = {} # rover -> set of waypoints
        have_image = {} # rover -> objective -> set of modes
        calibrated = {} # rover -> set of cameras (calibrated on this rover)
        full_store = set() # set of stores
        # at_soil_sample and at_rock_sample are not needed in state as we check have_...

        for r in self.rovers:
            have_soil[r] = set()
            have_rock[r] = set()
            have_image[r] = {o: set() for o in self.objectives}
            calibrated[r] = set() # Store calibrated cameras per rover

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at' and parts[1] in self.rovers:
                at_rover[parts[1]] = parts[2]
            elif parts[0] == 'have_soil_analysis' and parts[1] in self.rovers:
                have_soil[parts[1]].add(parts[2])
            elif parts[0] == 'have_rock_analysis' and parts[1] in self.rovers:
                have_rock[parts[1]].add(parts[2])
            elif parts[0] == 'have_image' and parts[1] in self.rovers:
                 # Ensure objective and mode exist in our precomputed sets
                 if parts[2] in self.objectives and parts[3] in self.modes:
                    have_image[parts[1]][parts[2]].add(parts[3])
            elif parts[0] == 'calibrated' and parts[1] in self.cameras:
                 # Ensure rover exists in our precomputed sets
                 if parts[2] in self.rovers:
                    calibrated[parts[2]].add(parts[1]) # Store calibrated cameras per rover
            elif parts[0] == 'full' and parts[1] in self.stores:
                full_store.add(parts[1])

        # --- Estimate cost for unachieved goals ---
        for goal in self.goals:
            if goal in state:
                continue # Goal already achieved

            parts = get_parts(goal)
            goal_type = parts[0]

            if goal_type == 'communicated_soil_data':
                soil_wp = parts[1]
                cost_for_goal = 0
                data_exists = False
                rover_with_data = None

                # Check if soil data for this waypoint exists with ANY rover
                for r in self.rovers:
                    if soil_wp in have_soil.get(r, set()):
                        data_exists = True
                        rover_with_data = r # Found a rover that has the data
                        break # Found one, can communicate

                if data_exists:
                    # Data exists, just need to communicate it
                    rover_to_use = rover_with_data # Use the rover that has the data
                    current_r_loc = at_rover.get(rover_to_use)
                    if current_r_loc is None:
                         # This rover exists but its location is unknown? Problematic state.
                         # Assign a large penalty.
                         h += 1000
                         continue

                    # Find closest communication waypoint
                    comm_wps = self.lander_visible_waypoints
                    comm_wp, dist_to_comm = self.find_closest_waypoint(current_r_loc, rover_to_use, comm_wps)
                    if comm_wp is None:
                         # No reachable communication waypoint - should not happen in solvable problems
                         h += 1000
                         continue

                    cost_for_goal += dist_to_comm
                    cost_for_goal += 1 # communicate_soil_data action

                else:
                    # Data does not exist, need to sample and then communicate
                    # Find a suitable rover (equipped for soil) to do the sampling
                    suitable_rovers = [r for r, cap in self.rover_capabilities.items() if cap['soil']]
                    if not suitable_rovers:
                         # No rover can sample soil - should not happen in solvable problems
                         h += 1000
                         continue
                    rover_to_use = suitable_rovers[0] # Pick the first suitable rover

                    current_r_loc = at_rover.get(rover_to_use)
                    if current_r_loc is None:
                         h += 1000
                         continue

                    temp_r_loc = current_r_loc # Track rover location for sequential steps

                    # Cost to get soil sample
                    dist_to_sample = self.get_distance(rover_to_use, temp_r_loc, soil_wp)
                    cost_for_goal += dist_to_sample
                    temp_r_loc = soil_wp # Rover is now at sample location

                    # Check store status
                    store = self.rover_store.get(rover_to_use)
                    if store and store in full_store:
                        cost_for_goal += 1 # drop action

                    cost_for_goal += 1 # sample_soil action

                    # Find closest communication waypoint from the sample location
                    comm_wps = self.lander_visible_waypoints
                    comm_wp, dist_to_comm = self.find_closest_waypoint(temp_r_loc, rover_to_use, comm_wps)
                    if comm_wp is None:
                         h += 1000
                         continue

                    # Cost to communicate
                    cost_for_goal += dist_to_comm
                    cost_for_goal += 1 # communicate_soil_data action

                h += cost_for_goal

            elif goal_type == 'communicated_rock_data':
                 # Similar refined logic for rock data
                 rock_wp = parts[1]
                 cost_for_goal = 0
                 data_exists = False
                 rover_with_data = None

                 # Check if rock data for this waypoint exists with ANY rover
                 for r in self.rovers:
                     if rock_wp in have_rock.get(r, set()):
                         data_exists = True
                         rover_with_data = r # Found a rover that has the data
                         break # Found one, can communicate

                 if data_exists:
                     # Data exists, just need to communicate it
                     rover_to_use = rover_with_data # Use the rover that has the data
                     current_r_loc = at_rover.get(rover_to_use)
                     if current_r_loc is None:
                         h += 1000
                         continue

                     # Find closest communication waypoint
                     comm_wps = self.lander_visible_waypoints
                     comm_wp, dist_to_comm = self.find_closest_waypoint(current_r_loc, rover_to_use, comm_wps)
                     if comm_wp is None:
                         h += 1000
                         continue

                     cost_for_goal += dist_to_comm
                     cost_for_goal += 1 # communicate_rock_data action

                 else:
                     # Data does not exist, need to sample and then communicate
                     # Find a suitable rover (equipped for rock) to do the sampling
                     suitable_rovers = [r for r, cap in self.rover_capabilities.items() if cap['rock']]
                     if not suitable_rovers:
                         h += 1000
                         continue
                     rover_to_use = suitable_rovers[0] # Pick the first suitable rover

                     current_r_loc = at_rover.get(rover_to_use)
                     if current_r_loc is None:
                         h += 1000
                         continue

                     temp_r_loc = current_r_loc # Track rover location for sequential steps

                     # Cost to get rock sample
                     dist_to_sample = self.get_distance(rover_to_use, temp_r_loc, rock_wp)
                     cost_for_goal += dist_to_sample
                     temp_r_loc = rock_wp # Rover is now at sample location

                     # Check store status
                     store = self.rover_store.get(rover_to_use)
                     if store and store in full_store:
                         cost_for_goal += 1 # drop action

                     cost_for_goal += 1 # sample_rock action

                     # Find closest communication waypoint from the sample location
                     comm_wps = self.lander_visible_waypoints
                     comm_wp, dist_to_comm = self.find_closest_waypoint(temp_r_loc, rover_to_use, comm_wps)
                     if comm_wp is None:
                         h += 1000
                         continue

                     # Cost to communicate
                     cost_for_goal += dist_to_comm
                     cost_for_goal += 1 # communicate_rock_data action

                 h += cost_for_goal

            elif goal_type == 'communicated_image_data':
                 # Similar refined logic for image data
                 objective = parts[1]
                 mode = parts[2]

                 cost_for_goal = 0
                 data_exists = False
                 rover_with_data = None

                 # Check if image data for this objective/mode exists with ANY rover
                 for r in self.rovers:
                     if mode in have_image.get(r, {}).get(objective, set()):
                         data_exists = True
                         rover_with_data = r # Found a rover that has the data
                         break # Found one, can communicate

                 if data_exists:
                     # Data exists, just need to communicate it
                     rover_to_use = rover_with_data # Use the rover that has the data
                     current_r_loc = at_rover.get(rover_to_use)
                     if current_r_loc is None:
                         h += 1000
                         continue

                     # Find closest communication waypoint
                     comm_wps = self.lander_visible_waypoints
                     comm_wp, dist_to_comm = self.find_closest_waypoint(current_r_loc, rover_to_use, comm_wps)
                     if comm_wp is None:
                         h += 1000
                         continue

                     cost_for_goal += dist_to_comm
                     cost_for_goal += 1 # communicate_image_data action

                 else:
                     # Data does not exist, need to take image and then communicate
                     # Find a suitable rover (equipped for imaging) to take the image
                     suitable_rovers = [r for r, cap in self.rover_capabilities.items() if cap['imaging']]
                     if not suitable_rovers:
                         h += 1000
                         continue
                     rover_to_use = suitable_rovers[0] # Pick the first suitable rover

                     # Find a suitable camera on this rover supporting the mode
                     suitable_cameras = [c for c, r in self.camera_on_rover.items() if r == rover_to_use and mode in self.camera_supports.get(c, set())]
                     if not suitable_cameras:
                         # No camera on this rover supports the mode - should not happen for a suitable rover
                         h += 1000
                         continue
                     camera_to_use = suitable_cameras[0] # Pick the first suitable camera

                     current_r_loc = at_rover.get(rover_to_use)
                     if current_r_loc is None:
                         h += 1000
                         continue

                     temp_r_loc = current_r_loc # Track rover location for sequential steps

                     # Cost to get image
                     # Cost to calibrate if needed
                     if camera_to_use not in calibrated.get(rover_to_use, set()):
                         # Need to calibrate camera
                         cal_target = self.camera_calibration_target.get(camera_to_use)
                         if cal_target is None:
                             # Camera has no calibration target - should not happen
                             h += 1000
                             continue
                         cal_wps = self.objective_visible_wps.get(cal_target, set())
                         if not cal_wps:
                             # Cannot calibrate this camera - should not happen
                             h += 1000
                             continue
                         cal_wp, dist_to_cal = self.find_closest_waypoint(temp_r_loc, rover_to_use, cal_wps)
                         if cal_wp is None:
                             h += 1000
                             continue

                         cost_for_goal += dist_to_cal
                         temp_r_loc = cal_wp # Rover is now at calibration location
                         cost_for_goal += 1 # calibrate action

                     # Cost to take image
                     img_wps = self.objective_visible_wps.get(objective, set())
                     if not img_wps:
                         # Cannot image this objective - should not happen
                         h += 1000
                         continue
                     img_wp, dist_to_img = self.find_closest_waypoint(temp_r_loc, rover_to_use, img_wps)
                     if img_wp is None:
                         h += 1000
                         continue

                     dist_to_img_from_current = self.get_distance(rover_to_use, temp_r_loc, img_wp)
                     cost_for_goal += dist_to_img_from_current
                     temp_r_loc = img_wp # Rover is now at image location
                     cost_for_goal += 1 # take_image action

                     # Find closest communication waypoint from the image location
                     comm_wps = self.lander_visible_waypoints
                     comm_wp, dist_to_comm = self.find_closest_waypoint(temp_r_loc, rover_to_use, comm_wps)
                     if comm_wp is None:
                         h += 1000
                         continue

                     # Cost to communicate
                     cost_for_goal += dist_to_comm
                     cost_for_goal += 1 # communicate_image_data action

                 h += cost_for_goal

        return h
