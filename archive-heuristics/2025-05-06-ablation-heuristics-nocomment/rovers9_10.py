from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque, defaultdict

def get_parts(fact):
    """Removes leading/trailing parentheses and splits by space."""
    return fact[1:-1].split()

def match(fact, *args):
    """Checks if fact parts match the given args (using fnmatch for wildcards)."""
    parts = get_parts(fact)
    return len(parts) == len(args) and all(fnmatch(part, arg) for part, arg in zip(parts, args))

def bfs(graph, start_node):
    """Performs BFS to find shortest paths from start_node."""
    distances = {node: float('inf') for node in graph}
    if start_node not in graph:
         # Start node is not a known waypoint in this graph (shouldn't happen if graph nodes are all waypoints)
         return distances
    distances[start_node] = 0
    queue = deque([start_node])
    while queue:
        current_node = queue.popleft()
        if current_node not in graph: continue # Should not happen if graph is built correctly
        for neighbor in graph.get(current_node, []):
            if distances[neighbor] == float('inf'):
                distances[neighbor] = distances[current_node] + 1
                queue.append(neighbor)
    return distances

class roversHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Rovers domain.

    Summary:
    This heuristic estimates the remaining cost to reach the goal state
    by summing the estimated costs for each unachieved goal predicate.
    For each unachieved goal (communicating soil data, rock data, or image data),
    it calculates the minimum cost required to achieve that specific goal
    considering the current state and static domain information. The minimum
    cost for a goal is found by considering all capable rovers and calculating
    the cost of the necessary sequence of actions (navigation, sampling/imaging,
    calibration if needed, and communication). Navigation costs are precomputed
    using BFS on the rover-specific traversability graphs. Resource constraints
    like store capacity are largely ignored for simplicity and efficiency.

    Assumptions:
    - Action costs are uniform (cost 1 per action).
    - The heuristic ignores resource constraints (e.g., store capacity for samples).
    - The heuristic ignores potential negative interactions or synergies between
      achieving different goals simultaneously with multiple rovers. It sums
      the minimum cost for each goal independently.
    - The heuristic assumes that if a sample exists at a waypoint or an objective
      is visible from a waypoint, the corresponding sample/image action is possible
      if a capable rover is present and other preconditions (like calibration or empty store)
      are met (though some preconditions are relaxed).
    - Unreachable goals or required locations result in a large heuristic value
      to guide the search away from unproductive paths.

    Heuristic Initialization:
    The constructor parses the static facts from the task description to build
    data structures used for efficient heuristic computation:
    - `lander_waypoint`: The location of the lander.
    - `comm_waypoints`: Set of waypoints visible from the lander waypoint (where communication is possible).
    - `rovers`: Set of all rover names.
    - `rover_capabilities`: Maps each rover to the set of analysis types ('soil', 'rock', 'imaging') it is equipped for.
    - `rover_stores`: Maps each rover to its store object.
    - `rover_cameras`: Maps each rover to the list of cameras on board.
    - `camera_modes`: Maps each camera to the set of modes it supports.
    - `camera_cal_target`: Maps each camera to its calibration target objective.
    - `objective_vis_waypoints`: Maps each objective to the set of waypoints it is visible from.
    - `waypoint_set`: Set of all known waypoints.
    - `rover_shortest_paths`: Maps each rover to a dictionary containing shortest path distances
      from every waypoint to every other waypoint, computed using BFS based on the rover's
      `can_traverse` facts.
    - `rover_min_dist_to_comm`: Maps each rover to a dictionary containing the minimum
      shortest path distance from every waypoint to any communication waypoint.

    Step-By-Step Thinking for Computing Heuristic:
    1. Check if the current state is a goal state. If yes, return 0.
    2. Extract relevant dynamic information from the current state: rover locations,
       collected samples (`have_soil_analysis`, `have_rock_analysis`), taken images
       (`have_image`), store statuses (`empty`, `full`), camera calibration statuses
       (`calibrated`), and remaining samples at waypoints (`at_soil_sample`, `at_rock_sample`).
    3. Initialize `total_cost` to 0.
    4. Identify the set of goal predicates that are not yet satisfied in the current state.
    5. Iterate through each unachieved goal predicate:
        a. If the goal is `(communicated_soil_data ?w)`:
           i. Check if `(have_soil_analysis ?r ?w)` is true for any rover `r`.
           ii. If not collected: Find the minimum cost among all soil-equipped rovers.
               The cost for a rover is estimated as:
               Navigation cost from current location to `?w` + 1 (sample_soil) +
               Navigation cost from `?w` to nearest communication waypoint + 1 (communicate_soil_data).
               Use precomputed shortest paths for navigation costs. If any step is impossible (e.g., cannot reach waypoint), this rover cannot achieve the goal.
           iii. If collected: Find the minimum cost among all rovers that have `(have_soil_analysis ?r ?w)`.
               The cost for a rover is estimated as:
               Navigation cost from current location to nearest communication waypoint + 1 (communicate_soil_data).
               Use precomputed shortest paths.
           iv. Add the minimum cost found (over all relevant rovers) for this goal to `total_cost`. If no rover can achieve the goal, return a large value (1000000).
        b. If the goal is `(communicated_rock_data ?w)`: Follow the same logic as soil data, using rock-specific predicates and capabilities.
        c. If the goal is `(communicated_image_data ?o ?m)`:
           i. Check if `(have_image ?r ?o ?m)` is true for any rover `r`.
           ii. If not taken: Find the minimum cost among all imaging-equipped rovers with a camera supporting mode `?m`.
               The cost for a (rover, camera) pair is estimated based on the sequence:
               Current location -> Calibration waypoint -> Image waypoint -> Communication waypoint.
               Cost = Navigation(curr, cal_wp) + 1 (calibrate) + Navigation(cal_wp, img_wp) + 1 (take_image) + Navigation(img_wp, comm_wp) + 1 (communicate_image_data).
               Minimize this cost over all possible calibration waypoints (`cal_wp` visible from camera's target) and image waypoints (`img_wp` visible from objective `?o`). This minimization is calculated efficiently using precomputed shortest paths and minimum distances to communication waypoints.
               If any required waypoint set is empty or navigation is impossible, this rover/camera pair cannot achieve the goal.
           iii. If taken: Find the minimum cost among all rovers that have `(have_image ?r ?o ?m)`.
               The cost for a rover is estimated as:
               Navigation cost from current location to nearest communication waypoint + 1 (communicate_image_data).
               Use precomputed shortest paths.
           iv. Add the minimum cost found (over all relevant rovers/cameras) for this goal to `total_cost`. If no rover/camera can achieve the goal, return a large value (1000000).
    6. Return the final `total_cost`.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by parsing static facts and precomputing
        navigation costs and relevant waypoint sets.

        Args:
            task: The planning task object containing initial state, goals, static facts, etc.
        """
        self.goals = task.goals
        static_facts = task.static

        # --- Precompute Static Information ---

        self.lander_waypoint = None
        for fact in static_facts:
            if match(fact, "at_lander", "*", "*"):
                self.lander_waypoint = get_parts(fact)[2]
                break

        self.comm_waypoints = set()
        if self.lander_waypoint:
            for fact in static_facts:
                # Communication requires (visible ?x ?y) where rover is at ?x and lander at ?y
                if match(fact, "visible", "*", self.lander_waypoint):
                     self.comm_waypoints.add(get_parts(fact)[1])

        self.rover_capabilities = defaultdict(set)
        self.rovers = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue
            if parts[0].startswith('rover'): # Heuristic: Assume any object starting with 'rover' is a rover
                 self.rovers.add(parts[0])
            elif parts[0] == 'equipped_for_soil_analysis':
                if len(parts) == 2: self.rover_capabilities[parts[1]].add('soil')
                if len(parts) > 1: self.rovers.add(parts[1])
            elif parts[0] == 'equipped_for_rock_analysis':
                if len(parts) == 2: self.rover_capabilities[parts[1]].add('rock')
                if len(parts) > 1: self.rovers.add(parts[1])
            elif parts[0] == 'equipped_for_imaging':
                if len(parts) == 2: self.rover_capabilities[parts[1]].add('imaging')
                if len(parts) > 1: self.rovers.add(parts[1])

        self.rover_stores = {}
        for fact in static_facts:
             if match(fact, "store_of", "*", "*"):
                  parts = get_parts(fact)
                  if len(parts) == 3:
                    store, rover = parts[1:3]
                    self.rover_stores[rover] = store

        self.rover_cameras = defaultdict(list)
        self.camera_modes = defaultdict(set)
        self.camera_cal_target = {}
        self.cameras = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue
            if parts[0] == 'on_board':
                if len(parts) == 3:
                    camera, rover = parts[1:3]
                    self.rover_cameras[rover].append(camera)
                    self.cameras.add(camera)
            elif parts[0] == 'supports':
                if len(parts) == 3:
                    camera, mode = parts[1:3]
                    self.camera_modes[camera].add(mode)
                    self.cameras.add(camera)
            elif parts[0] == 'calibration_target':
                if len(parts) == 3:
                    camera, objective = parts[1:3]
                    self.camera_cal_target[camera] = objective
                    self.cameras.add(camera)

        self.objective_vis_waypoints = defaultdict(set)
        self.objectives = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue
            if parts[0] == 'visible_from':
                if len(parts) == 3:
                    objective, waypoint = parts[1:3]
                    self.objective_vis_waypoints[objective].add(waypoint)
                    self.objectives.add(objective)

        # Collect all waypoints
        self.waypoint_set = set()
        for fact in static_facts:
            parts = get_parts(fact)
            for part in parts:
                if part.startswith('waypoint'): # Simple heuristic check for waypoint names
                    self.waypoint_set.add(part)
        # Add waypoints from goals if any
        for goal in self.goals:
             parts = get_parts(goal)
             for part in parts:
                 if part.startswith('waypoint'):
                      self.waypoint_set.add(part)


        # Build navigation graphs and compute shortest paths per rover
        self.rover_shortest_paths = {}
        for rover in self.rovers:
            graph = {wp: [] for wp in self.waypoint_set}
            for fact in static_facts:
                if match(fact, "can_traverse", rover, "*", "*"):
                    _, r_name, w_from, w_to = get_parts(fact)
                    if r_name == rover:
                         graph[w_from].append(w_to)
            # Ensure all waypoints are in the graph even if no edges
            for wp in self.waypoint_set:
                 if wp not in graph:
                      graph[wp] = []

            shortest_paths = {}
            for start_wp in self.waypoint_set:
                shortest_paths[start_wp] = bfs(graph, start_wp)
            self.rover_shortest_paths[rover] = shortest_paths

        # Precompute min distance from any waypoint to any communication waypoint for each rover
        self.rover_min_dist_to_comm = {}
        if self.comm_waypoints: # Only precompute if there are comm waypoints
            for rover in self.rovers:
                min_dist_map = {}
                if rover in self.rover_shortest_paths:
                    for start_wp in self.waypoint_set:
                        min_dist = float('inf')
                        if start_wp in self.rover_shortest_paths[rover]:
                            for comm_wp in self.comm_waypoints:
                                dist = self.rover_shortest_paths[rover][start_wp].get(comm_wp, float('inf'))
                                min_dist = min(min_dist, dist)
                        min_dist_map[start_wp] = min_dist
                self.rover_min_dist_to_comm[rover] = min_dist_map


    def get_navigation_cost(self, rover, start_wp, end_wp):
        """Gets shortest path distance for a specific rover."""
        if rover not in self.rover_shortest_paths or start_wp not in self.rover_shortest_paths[rover] or end_wp not in self.rover_shortest_paths[rover][start_wp]:
            return float('inf') # Cannot navigate
        return self.rover_shortest_paths[rover][start_wp][end_wp]

    def get_min_navigation_cost_to_comm(self, rover, start_wp):
        """Gets minimum shortest path distance for a rover from start_wp to any comm waypoint."""
        if not self.comm_waypoints: return float('inf') # No communication possible
        if rover not in self.rover_min_dist_to_comm or start_wp not in self.rover_min_dist_to_comm[rover]:
             return float('inf')
        return self.rover_min_dist_to_comm[rover][start_wp]


    def __call__(self, node):
        """
        Computes the heuristic value for the given state.

        Args:
            node: The search node containing the current state.

        Returns:
            An estimated cost (integer) to reach the goal state.
        """
        state = node.state
        goals = self.goals

        # Goal state check
        if goals <= state:
            return 0

        # Extract current state information
        rover_locations = {}
        have_soil = set() # {(rover, waypoint)}
        have_rock = set() # {(rover, waypoint)}
        have_image = set() # {(rover, objective, mode)}
        # stores_empty = set() # {store} # Not used in this heuristic relaxation
        # stores_full = set() # {store} # Not used in this heuristic relaxation
        # cameras_calibrated = set() # {(camera, rover)} # Not used in this heuristic relaxation
        # soil_samples_at_wp = set() # {waypoint} # Not used in this heuristic relaxation
        # rock_samples_at_wp = set() # {waypoint} # Not used in this heuristic relaxation

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip empty facts
            pred = parts[0]
            if pred == 'at' and len(parts) == 3 and parts[1].startswith('rover'):
                rover_locations[parts[1]] = parts[2]
            elif pred == 'have_soil_analysis':
                if len(parts) == 3: have_soil.add((parts[1], parts[2]))
            elif pred == 'have_rock_analysis':
                if len(parts) == 3: have_rock.add((parts[1], parts[2]))
            elif pred == 'have_image':
                 if len(parts) == 4: have_image.add((parts[1], parts[2], parts[3]))
            # elif pred == 'empty':
            #      if len(parts) == 2: stores_empty.add(parts[1])
            # elif pred == 'full':
            #      if len(parts) == 2: stores_full.add(parts[1])
            # elif pred == 'calibrated':
            #      if len(parts) == 3: cameras_calibrated.add((parts[1], parts[2]))
            # elif pred == 'at_soil_sample':
            #      if len(parts) == 2: soil_samples_at_wp.add(parts[1])
            # elif pred == 'at_rock_sample':
            #      if len(parts) == 2: rock_samples_at_wp.add(parts[1])


        total_cost = 0
        unachieved_goals = goals - state

        # Process unachieved goals
        for goal in unachieved_goals:
            parts = get_parts(goal)
            if not parts: continue
            pred = parts[0]

            if pred == 'communicated_soil_data':
                if len(parts) != 2: continue # Malformed goal
                waypoint = parts[1]

                # Check if data is already collected by any rover
                collected = any((r, waypoint) in have_soil for r in self.rovers)

                min_goal_cost = float('inf')

                if not collected:
                    # Need to collect and then communicate
                    # Find capable rovers
                    capable_rovers = [r for r in self.rovers if 'soil' in self.rover_capabilities.get(r, set())]
                    if not capable_rovers:
                         min_goal_cost = float('inf') # Cannot collect if no rover is equipped
                    else:
                        for rover in capable_rovers:
                             curr_wp = rover_locations.get(rover)
                             if curr_wp is None or curr_wp not in self.waypoint_set: continue # Rover location unknown or invalid

                             # Cost to collect: navigate to waypoint + sample (1 action)
                             nav_to_sample_cost = self.get_navigation_cost(rover, curr_wp, waypoint)
                             if nav_to_sample_cost == float('inf'): continue # Cannot reach sample location

                             collect_cost = nav_to_sample_cost + 1 # sample_soil action

                             # Cost to communicate: navigate from sample location to comm location + communicate (1 action)
                             # Assume after sampling, rover is at 'waypoint'
                             nav_to_comm_cost = self.get_min_navigation_cost_to_comm(rover, waypoint)
                             if nav_to_comm_cost == float('inf'): continue # Cannot reach any comm location

                             comm_cost = nav_to_comm_cost + 1 # communicate_soil_data action

                             min_goal_cost = min(min_goal_cost, collect_cost + comm_cost)

                else:
                    # Data is collected, just need to communicate
                    # Find which rover(s) have the data
                    rovers_with_data = [r for r, w in have_soil if w == waypoint]
                    if not rovers_with_data:
                         min_goal_cost = float('inf') # Should not happen if collected is True
                    else:
                        for rover in rovers_with_data:
                             curr_wp = rover_locations.get(rover)
                             if curr_wp is None or curr_wp not in self.waypoint_set: continue

                             nav_to_comm_cost = self.get_min_navigation_cost_to_comm(rover, curr_wp)
                             if nav_to_comm_cost == float('inf'): continue

                             comm_cost = nav_to_comm_cost + 1 # communicate_soil_data action
                             min_goal_cost = min(min_goal_cost, comm_cost)

                if min_goal_cost != float('inf'):
                     total_cost += min_goal_cost
                else:
                     # If goal is unreachable by any means, assign a large cost
                     return 1000000


            elif pred == 'communicated_rock_data':
                if len(parts) != 2: continue # Malformed goal
                waypoint = parts[1]

                # Check if data is already collected by any rover
                collected = any((r, waypoint) in have_rock for r in self.rovers)

                min_goal_cost = float('inf')

                if not collected:
                    # Need to collect and then communicate
                    # Find capable rovers
                    capable_rovers = [r for r in self.rovers if 'rock' in self.rover_capabilities.get(r, set())]
                    if not capable_rovers:
                         min_goal_cost = float('inf')
                    else:
                        for rover in capable_rovers:
                             curr_wp = rover_locations.get(rover)
                             if curr_wp is None or curr_wp not in self.waypoint_set: continue

                             nav_to_sample_cost = self.get_navigation_cost(rover, curr_wp, waypoint)
                             if nav_to_sample_cost == float('inf'): continue

                             collect_cost = nav_to_sample_cost + 1 # sample_rock action

                             nav_to_comm_cost = self.get_min_navigation_cost_to_comm(rover, waypoint)
                             if nav_to_comm_cost == float('inf'): continue

                             comm_cost = nav_to_comm_cost + 1 # communicate_rock_data action

                             min_goal_cost = min(min_goal_cost, collect_cost + comm_cost)

                else:
                    # Data is collected, just need to communicate
                    # Find which rover(s) have the data
                    rovers_with_data = [r for r, w in have_rock if w == waypoint]
                    if not rovers_with_data:
                         min_goal_cost = float('inf')
                    else:
                        for rover in rovers_with_data:
                             curr_wp = rover_locations.get(rover)
                             if curr_wp is None or curr_wp not in self.waypoint_set: continue

                             nav_to_comm_cost = self.get_min_navigation_cost_to_comm(rover, curr_wp)
                             if nav_to_comm_cost == float('inf'): continue

                             comm_cost = nav_to_comm_cost + 1 # communicate_rock_data action
                             min_goal_cost = min(min_goal_cost, comm_cost)

                if min_goal_cost != float('inf'):
                     total_cost += min_goal_cost
                else:
                     return 1000000


            elif pred == 'communicated_image_data':
                if len(parts) != 3: continue # Malformed goal
                objective, mode = parts[1], parts[2]

                # Check if image is already taken by any rover
                taken = any((r, objective, mode) in have_image for r in self.rovers)

                min_goal_cost = float('inf')

                if not taken:
                    # Need to take image and then communicate
                    # Find rovers capable of imaging with a camera supporting the mode
                    capable_rover_camera_pairs = []
                    for rover in self.rovers:
                        if 'imaging' in self.rover_capabilities.get(rover, set()):
                            for camera in self.rover_cameras.get(rover, []):
                                if mode in self.camera_modes.get(camera, set()):
                                    capable_rover_camera_pairs.append((rover, camera))
                                    break # Found a suitable camera on this rover, move to next rover

                    if not capable_rover_camera_pairs:
                         min_goal_cost = float('inf') # Cannot take image if no rover/camera is equipped/supports mode
                    else:
                        for rover, camera in capable_rover_camera_pairs:
                             curr_wp = rover_locations.get(rover)
                             if curr_wp is None or curr_wp not in self.waypoint_set: continue

                             # Find suitable image waypoints
                             img_wps = self.objective_vis_waypoints.get(objective, set())
                             if not img_wps: continue # Cannot see objective from anywhere

                             # Find calibration target and suitable calibration waypoints
                             cal_target = self.camera_cal_target.get(camera)
                             if not cal_target: continue # Camera has no calibration target
                             cal_wps = self.objective_vis_waypoints.get(cal_target, set())
                             if not cal_wps: continue # Calibration target not visible from anywhere

                             # Calculate min cost for sequence: curr -> cal -> img -> comm
                             # min_{cal in cal_wps, img in img_wps} (dist(curr, cal) + 1 + dist(cal, img) + 1 + min_dist(img, comm_wps) + 1)

                             min_nav_cal_img_comm = float('inf')

                             # Calculate min_dist(img_wp, comm_wps) for all relevant img_wps
                             min_dist_img_to_comm_map = {}
                             for img_wp in img_wps:
                                 min_dist_img_to_comm_map[img_wp] = self.get_min_navigation_cost_to_comm(rover, img_wp)

                             # Calculate min_cost_from_cal[cal_wp] = min_{img in img_wps} (dist(cal_wp, img) + min_dist_img_to_comm_map[img])
                             min_cost_from_cal_map = {}
                             for cal_wp in cal_wps:
                                 min_cost_from_cal = float('inf')
                                 for img_wp in img_wps:
                                     dist_cal_to_img = self.get_navigation_cost(rover, cal_wp, img_wp)
                                     if dist_cal_to_img != float('inf') and min_dist_img_to_comm_map.get(img_wp, float('inf')) != float('inf'):
                                         min_cost_from_cal = min(min_cost_from_cal, dist_cal_to_img + min_dist_img_to_comm_map[img_wp])
                                 min_cost_from_cal_map[cal_wp] = min_cost_from_cal

                             # Calculate min_nav_cost = min_{cal in cal_wps} (dist(curr, cal) + min_cost_from_cal_map[cal])
                             min_nav_cost = float('inf')
                             for cal_wp in cal_wps:
                                 dist_curr_to_cal = self.get_navigation_cost(rover, curr_wp, cal_wp)
                                 if dist_curr_to_cal != float('inf') and min_cost_from_cal_map.get(cal_wp, float('inf')) != float('inf'):
                                     min_nav_cost = min(min_nav_cost, dist_curr_to_cal + min_cost_from_cal_map[cal_wp])

                             if min_nav_cost != float('inf'):
                                 # Add action costs: calibrate (1), take_image (1), communicate (1)
                                 min_take_comm_cost = min_nav_cost + 3
                                 min_goal_cost = min(min_goal_cost, min_take_comm_cost)


                else:
                    # Image is taken, just need to communicate
                    # Find which rover(s) have the image
                    rovers_with_image = [r for r, o, m in have_image if o == objective and m == mode]
                    if not rovers_with_image:
                         min_goal_cost = float('inf') # Should not happen if taken is True
                    else:
                        for rover in rovers_with_image:
                             curr_wp = rover_locations.get(rover)
                             if curr_wp is None or curr_wp not in self.waypoint_set: continue

                             nav_to_comm_cost = self.get_min_navigation_cost_to_comm(rover, curr_wp)
                             if nav_to_comm_cost == float('inf'): continue

                             comm_cost = nav_to_comm_cost + 1 # communicate_image_data action
                             min_goal_cost = min(min_goal_cost, comm_cost)

                if min_goal_cost != float('inf'):
                     total_cost += min_goal_cost
                else:
                     return 1000000 # Indicate very high cost / likely unreachable


        # If we reached here, there are unachieved goals, and total_cost is the sum
        # of minimum estimated costs for each. If total_cost is 0, it implies
        # all unachieved goals had an estimated cost of 0, which shouldn't happen
        # unless the goal is actually achieved (handled at the start).
        # Return total_cost.
        return total_cost

