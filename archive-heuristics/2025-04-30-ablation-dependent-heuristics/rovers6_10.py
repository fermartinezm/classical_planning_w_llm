import math
from collections import deque

class roversHeuristic:
    """
    Domain-dependent heuristic for the Rovers domain.

    Summary:
    This heuristic estimates the cost to reach a goal state from a given state
    by summing up the estimated costs for each unmet goal fact. For each unmet
    goal (communicating soil data, rock data, or image data), it calculates
    the minimum number of actions required by any suitable rover, considering
    navigation, sampling/imaging, and communication steps. Navigation costs
    are estimated using precomputed shortest path distances on the waypoint graph.
    The heuristic is non-admissible and designed to guide a greedy best-first search.

    Assumptions:
    - The navigation graph defined by 'visible' facts is symmetric and the same
      for all rovers (i.e., 'can_traverse' links are redundant or identical
      to 'visible' links).
    - Calibration targets for cameras are visible from at least one waypoint
      where the corresponding objective is visible, if calibration is required.
    - Problem instances are solvable, meaning necessary resources (equipped rovers,
      samples, objectives visible from waypoints, communication waypoints) exist
      and are reachable. A penalty is added for goals that appear unreachable
      in the current state based on distance calculations.
    - The number of waypoints, rovers, etc., is small enough for BFS precomputation.

    Heuristic Initialization:
    The constructor `__init__` performs the following steps:
    1. Stores the planning task object.
    2. Parses all static facts from `task.static`. It extracts information about:
       - Lander locations.
       - Rover capabilities (equipped for soil, rock, imaging).
       - Store ownership by rovers.
       - Camera information (on board, supported modes, calibration target).
       - Objective visibility from waypoints.
       - Waypoints and the navigation graph based on 'visible' links.
    3. Identifies communication waypoints (waypoints visible from any lander location).
    4. Precomputes shortest path distances between all pairs of waypoints using BFS
       on the navigation graph. This information is stored in `self.distances`.

    Step-By-Step Thinking for Computing Heuristic:
    The `__call__` method computes the heuristic value for the given state:
    1. Checks if the state is a goal state. If yes, returns 0.
    2. Parses the current state facts into structured dictionaries/sets for easy lookup
       (e.g., rover locations, samples held, images held, calibration status, store status,
       sample locations, communicated data).
    3. Identifies the set of goal facts that are not yet true in the current state (`unmet_goals`).
    4. Initializes the total heuristic value `h` to 0.
    5. Iterates through each unmet goal fact:
       - For a `(communicated_soil_data ?w)` goal:
         - Finds suitable rovers (equipped for soil analysis).
         - For each suitable rover, calculates the cost to achieve this specific goal:
           - If the rover already has `(have_soil_analysis ?r ?w)`: Cost is navigation from current location to the closest communication waypoint + 1 (communicate action).
           - If the rover needs to sample: Cost is navigation from current location to `?w` + 1 (sample action) + [1 (drop action) if store is full] + navigation from `?w` to the closest communication waypoint + 1 (communicate action). This is only considered if a soil sample exists at `?w`.
         - Takes the minimum cost over all suitable rovers.
         - Adds this minimum cost to the total heuristic `h`. If no path is found (minimum cost remains infinity), adds a large penalty (100) instead.
       - For a `(communicated_rock_data ?w)` goal: Follows similar logic as soil goals.
       - For a `(communicated_image_data ?o ?m)` goal:
         - Finds suitable rovers (equipped for imaging) with a camera supporting mode `?m`.
         - For each suitable rover/camera combination, calculates the cost:
           - If the rover already has `(have_image ?r ?o ?m)`: Cost is navigation from current location to the closest communication waypoint + 1 (communicate action).
           - If the rover needs to take the image: Finds waypoints `?img_w` where `?o` is visible. For each such waypoint, calculates the cost: navigation from current location to `?img_w` + 1 (take_image action) + [1 (calibrate action) if camera is not calibrated and a calibration target is visible from `?img_w`] + navigation from `?img_w` to the closest communication waypoint + 1 (communicate action).
         - Takes the minimum cost over all suitable rovers/cameras and imaging waypoints.
         - Adds this minimum cost to the total heuristic `h`. If no path is found, adds a large penalty (100).
    6. Returns the total heuristic value `h`.
    """
    def __init__(self, task):
        self.task = task
        self._process_static(task.static)
        self._precompute_distances()

    def _parse_fact(self, fact_str):
        """Helper to parse a PDDL fact string into predicate and arguments."""
        # Remove parentheses and split by space
        parts = fact_str[1:-1].split()
        predicate = parts[0]
        args = parts[1:]
        return predicate, args

    def _process_static(self, static_facts):
        """Processes static facts to build internal data structures."""
        self.static_data = {}
        self.waypoints = set()
        self.rovers = set()
        self.landers = set()
        self.stores = set()
        self.cameras = set()
        self.modes = set()
        self.objectives = set()

        # First pass to collect all objects by type (approximate) and store facts
        for fact_str in static_facts:
            pred, args = self._parse_fact(fact_str)
            self.static_data.setdefault(pred, []).append(tuple(args))
            # Collect objects - simple approach based on typical naming conventions
            for arg in args:
                 if arg.startswith('rover'): self.rovers.add(arg)
                 elif arg.startswith('waypoint'): self.waypoints.add(arg)
                 elif arg.startswith('lander'): self.landers.add(arg)
                 elif arg.startswith('store'): self.stores.add(arg)
                 elif arg.startswith('camera'): self.cameras.add(arg)
                 elif arg.startswith('objective'): self.objectives.add(arg)
                 elif arg in ('colour', 'high_res', 'low_res'): self.modes.add(arg)
                 # Add other types if necessary

        # Build navigation graph using 'visible' facts (assuming symmetric and same for all rovers)
        self.navigation_graph = {}
        for w1, w2 in self.static_data.get('visible', []):
            self.navigation_graph.setdefault(w1, set()).add(w2)
            self.navigation_graph.setdefault(w2, set()).add(w1)

        # Determine communication waypoints
        self.comm_waypoints = set()
        lander_locations = {l: w for l, w in self.static_data.get('at_lander', [])}
        for lander_w in lander_locations.values():
            # A waypoint x is a communication waypoint if (visible x lander_w) is true
            # (visible w1 w2) means w1 is visible FROM w2
            for w1, w2 in self.static_data.get('visible', []):
                if w2 == lander_w:
                    self.comm_waypoints.add(w1)


        # Store other static info for easy lookup
        self._rover_equipped = {r: {'soil': False, 'rock': False, 'imaging': False} for r in self.rovers}
        for r, in self.static_data.get('equipped_for_soil_analysis', []): self._rover_equipped[r]['soil'] = True
        for r, in self.static_data.get('equipped_for_rock_analysis', []): self._rover_equipped[r]['rock'] = True
        for r, in self.static_data.get('equipped_for_imaging', []): self._rover_equipped[r]['imaging'] = True

        self._store_owner = {s: r for s, r in self.static_data.get('store_of', [])}

        self._camera_info = {c: {'rover': None, 'modes': set(), 'calibration_target': None} for c in self.cameras}
        for c, r in self.static_data.get('on_board', []): self._camera_info[c]['rover'] = r
        for c, m in self.static_data.get('supports', []): self._camera_info[c]['modes'].add(m)
        for c, t in self.static_data.get('calibration_target', []): self._camera_info[c]['calibration_target'] = t

        self._objective_visibility = {o: set() for o in self.objectives}
        for o, w in self.static_data.get('visible_from', []): self._objective_visibility[o].add(w)

    def _precompute_distances(self):
        """Precomputes shortest path distances between all waypoints using BFS."""
        self.distances = {} # waypoint -> waypoint -> distance
        waypoints_list = list(self.waypoints)
        for start_node in waypoints_list:
            self.distances[start_node] = {}
            q = deque([(start_node, 0)]) # Use deque for efficient BFS
            visited = {start_node}
            while q:
                (curr, dist) = q.popleft()
                self.distances[start_node][curr] = dist
                if curr in self.navigation_graph:
                    for neighbor in self.navigation_graph[curr]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            q.append((neighbor, dist + 1))

    def _get_distance(self, w1, w2):
        """Gets the precomputed shortest distance between two waypoints."""
        if w1 is None or w2 is None: return float('inf') # Cannot navigate from/to unknown location
        return self.distances.get(w1, {}).get(w2, float('inf'))

    def __call__(self, state):
        """Computes the heuristic value for the given state."""
        # Check if goal is reached
        if self.task.goal_reached(state):
            return 0

        # Parse current state
        state_data = {}
        for fact_str in state:
            pred, args = self._parse_fact(fact_str)
            state_data.setdefault(pred, []).append(tuple(args))

        at_rover = {r: w for r, w in state_data.get('at', [])}
        have_soil = {r: {w for r_, w in state_data.get('have_soil_analysis', []) if r_ == r} for r in self.rovers}
        have_rock = {r: {w for r_, w in state_data.get('have_rock_analysis', []) if r_ == r} for r in self.rovers}
        have_image = {r: {tuple(img) for r_, *img in state_data.get('have_image', []) if r_ == r} for r in self.rovers}
        calibrated = {c: True for c, in state_data.get('calibrated', [])}
        store_full = {s: True for s, in state_data.get('full', [])}
        soil_samples_at = {w for w, in state_data.get('at_soil_sample', [])}
        rock_samples_at = {w for w, in state_data.get('at_rock_sample', [])}
        communicated_soil = {w for w, in state_data.get('communicated_soil_data', [])}
        communicated_rock = {w for w, in state_data.get('communicated_rock_data', [])}
        communicated_image = {tuple(img) for *img, in state_data.get('communicated_image_data', [])}


        h = 0
        unmet_goals = self.task.goals - set(state) # Compare goal facts directly with state facts

        # Heuristic for soil goals
        for goal_str in unmet_goals:
             pred, args = self._parse_fact(goal_str)
             if pred == 'communicated_soil_data':
                 soil_w = args[0]
                 # Check if already communicated (should be caught by unmet_goals, but double check)
                 if soil_w in communicated_soil: continue

                 min_soil_goal_cost = float('inf')

                 # Find a rover that can potentially satisfy this goal
                 for rover in self.rovers:
                     if not self._rover_equipped.get(rover, {}).get('soil', False):
                         continue # Rover not equipped

                     current_rover_loc = at_rover.get(rover)
                     if current_rover_loc is None: continue # Rover location unknown (should not happen in valid states)

                     rover_has_sample = soil_w in have_soil.get(rover, set())

                     # Option 1: Rover already has the sample
                     if rover_has_sample:
                         # Need to communicate
                         min_comm_nav_cost = float('inf')
                         for comm_w in self.comm_waypoints:
                             dist = self._get_distance(current_rover_loc, comm_w)
                             if dist != float('inf'):
                                  min_comm_nav_cost = min(min_comm_nav_cost, dist)

                         if min_comm_nav_cost != float('inf'):
                             cost = min_comm_nav_cost + 1 # navigate + communicate
                             min_soil_goal_cost = min(min_soil_goal_cost, cost)

                     # Option 2: Rover needs to sample and then communicate
                     elif soil_w in soil_samples_at: # Check if sample exists at the waypoint
                         cost = 0
                         # Navigate to sample location
                         nav_to_sample_cost = self._get_distance(current_rover_loc, soil_w)
                         if nav_to_sample_cost == float('inf'): continue # Cannot reach sample

                         cost += nav_to_sample_cost
                         cost += 1 # sample action

                         # Check store status
                         rover_store = None
                         for s, r in self._store_owner.items():
                             if r == rover:
                                 rover_store = s
                                 break
                         if rover_store and store_full.get(rover_store, False):
                              cost += 1 # drop action

                         # After sampling, rover is at soil_w
                         current_rover_loc_after_sample = soil_w

                         # Navigate to communication waypoint and communicate
                         min_comm_nav_cost = float('inf')
                         for comm_w in self.comm_waypoints:
                             dist = self._get_distance(current_rover_loc_after_sample, comm_w)
                             if dist != float('inf'):
                                  min_comm_nav_cost = min(min_comm_nav_cost, dist)

                         if min_comm_nav_cost != float('inf'):
                             cost += min_comm_nav_cost + 1 # navigate + communicate
                             min_soil_goal_cost = min(min_soil_goal_cost, cost)

                 # Add the minimum cost for this specific soil goal to the total heuristic
                 if min_soil_goal_cost != float('inf'):
                      h += min_soil_goal_cost
                 else:
                      # If a soil goal is unreachable by any equipped rover, add a large penalty
                      # This indicates the state is likely bad or unsolvable.
                      h += 100 # Penalty for unreachable soil goal


        # Heuristic for rock goals (similar to soil)
        for goal_str in unmet_goals:
             pred, args = self._parse_fact(goal_str)
             if pred == 'communicated_rock_data':
                 rock_w = args[0]
                 if rock_w in communicated_rock: continue

                 min_rock_goal_cost = float('inf')

                 for rover in self.rovers:
                     if not self._rover_equipped.get(rover, {}).get('rock', False):
                         continue

                     current_rover_loc = at_rover.get(rover)
                     if current_rover_loc is None: continue

                     rover_has_sample = rock_w in have_rock.get(rover, set())

                     # Option 1: Already has sample
                     if rover_has_sample:
                         min_comm_nav_cost = float('inf')
                         for comm_w in self.comm_waypoints:
                             dist = self._get_distance(current_rover_loc, comm_w)
                             if dist != float('inf'):
                                  min_comm_nav_cost = min(min_comm_nav_cost, dist)
                         if min_comm_nav_cost != float('inf'):
                             cost = min_comm_nav_cost + 1
                             min_rock_goal_cost = min(min_rock_goal_cost, cost)

                     # Option 2: Needs to sample
                     elif rock_w in rock_samples_at:
                         cost = 0
                         nav_to_sample_cost = self._get_distance(current_rover_loc, rock_w)
                         if nav_to_sample_cost == float('inf'): continue

                         cost += nav_to_sample_cost
                         cost += 1 # sample action

                         rover_store = None
                         for s, r in self._store_owner.items():
                             if r == rover:
                                 rover_store = s
                                 break
                         if rover_store and store_full.get(rover_store, False):
                              cost += 1 # drop action

                         current_rover_loc_after_sample = rock_w

                         min_comm_nav_cost = float('inf')
                         for comm_w in self.comm_waypoints:
                             dist = self._get_distance(current_rover_loc_after_sample, comm_w)
                             if dist != float('inf'):
                                  min_comm_nav_cost = min(min_comm_nav_cost, dist)

                         if min_comm_nav_cost != float('inf'):
                             cost += min_comm_nav_cost + 1
                             min_rock_goal_cost = min(min_rock_goal_cost, cost)

                 if min_rock_goal_cost != float('inf'):
                      h += min_rock_goal_cost
                 else:
                      h += 100 # Penalty


        # Heuristic for image goals
        for goal_str in unmet_goals:
             pred, args = self._parse_fact(goal_str)
             if pred == 'communicated_image_data':
                 obj, mode = args
                 img_goal_tuple = (obj, mode)
                 if img_goal_tuple in communicated_image: continue

                 min_image_goal_cost = float('inf')

                 # Find a rover that can potentially satisfy this goal
                 for rover in self.rovers:
                     if not self._rover_equipped.get(rover, {}).get('imaging', False):
                         continue # Rover not equipped

                     # Find a camera on this rover that supports the mode
                     suitable_camera = None
                     for cam, info in self._camera_info.items():
                         if info.get('rover') == rover and mode in info.get('modes', set()):
                             suitable_camera = cam
                             break
                     if suitable_camera is None:
                         continue # No suitable camera on this rover

                     current_rover_loc = at_rover.get(rover)
                     if current_rover_loc is None: continue

                     rover_has_image = img_goal_tuple in have_image.get(rover, set())

                     # Option 1: Already has the image
                     if rover_has_image:
                         min_comm_nav_cost = float('inf')
                         for comm_w in self.comm_waypoints:
                             dist = self._get_distance(current_rover_loc, comm_w)
                             if dist != float('inf'):
                                  min_comm_nav_cost = min(min_comm_nav_cost, dist)
                         if min_comm_nav_cost != float('inf'):
                             cost = min_comm_nav_cost + 1
                             min_image_goal_cost = min(min_image_goal_cost, cost)

                     # Option 2: Needs to take image
                     else:
                         # Find a waypoint where the objective is visible
                         possible_img_waypoints = self._objective_visibility.get(obj, set())
                         if not possible_img_waypoints: continue # Objective not visible from anywhere

                         min_imaging_path_cost = float('inf')

                         for img_w in possible_img_waypoints:
                             cost = 0
                             # Navigate to imaging location
                             nav_to_image_cost = self._get_distance(current_rover_loc, img_w)
                             if nav_to_image_cost == float('inf'): continue

                             cost += nav_to_image_cost
                             cost += 1 # take_image action

                             # Check calibration status of the suitable camera
                             # Calibration requires a target visible from img_w
                             cal_target = self._camera_info[suitable_camera].get('calibration_target')
                             needs_calibration = not calibrated.get(suitable_camera, False)

                             if needs_calibration:
                                 # Must be able to calibrate at img_w
                                 if cal_target and img_w in self._objective_visibility.get(cal_target, set()):
                                     cost += 1 # calibrate action
                                 else:
                                     # Cannot calibrate this camera at this imaging waypoint
                                     continue # Try next imaging waypoint
                             # If already calibrated, no calibration action needed

                             # After imaging, rover is at img_w
                             current_rover_loc_after_image = img_w

                             # Navigate to communication waypoint and communicate
                             min_comm_nav_cost = float('inf')
                             for comm_w in self.comm_waypoints:
                                 dist = self._get_distance(current_rover_loc_after_image, comm_w)
                                 if dist != float('inf'):
                                      min_comm_nav_cost = min(min_comm_nav_cost, dist)

                             if min_comm_nav_cost != float('inf'):
                                 cost += min_comm_nav_cost + 1 # navigate + communicate
                                 min_imaging_path_cost = min(min_imaging_path_cost, cost)

                         if min_imaging_path_cost != float('inf'):
                              min_image_goal_cost = min(min_image_goal_cost, min_imaging_path_cost)


                 if min_image_goal_cost != float('inf'):
                      h += min_image_goal_cost
                 else:
                      h += 100 # Penalty


        return h
