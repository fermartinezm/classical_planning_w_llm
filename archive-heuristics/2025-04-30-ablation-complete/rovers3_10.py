from collections import deque, defaultdict
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
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
    if not parts or not args:
        return False

    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# BFS function
def bfs(graph, start_node):
    """Computes shortest path distances from start_node to all reachable nodes."""
    distances = {node: float('inf') for node in graph}
    if start_node not in graph:
        # Start node is not in the graph (e.g., isolated waypoint)
        return distances

    distances[start_node] = 0
    queue = deque([start_node])
    while queue:
        current_node = queue.popleft()
        # Check if current_node is still valid in graph (should be if added from graph keys)
        if current_node not in graph: continue

        for neighbor in graph.get(current_node, []):
            if distances[neighbor] == float('inf'):
                distances[neighbor] = distances[current_node] + 1
                queue.append(neighbor)
    return distances


class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve all goal
    conditions. It sums the estimated costs for each unachieved goal fact.
    The cost for a goal fact (communicating data) is estimated based on whether
    the data has already been collected or not, and includes navigation costs
    (estimated using precomputed shortest paths) and action costs (sampling,
    imaging, calibrating, dropping, communicating).

    # Assumptions
    - The navigation graph defined by `can_traverse` is static and symmetric
      for all rovers.
    - Action costs (sample, image, calibrate, drop, communicate) are 1.
    - The heuristic estimates the minimum cost over available rovers and
      suitable waypoints.
    - Unreachable goals (e.g., no equipped rover, no sample at location)
      result in an infinite heuristic value.
    - Calibration cost is simplified: if a camera is not calibrated and needed
      for an image goal, we add the cost of navigating from the rover's current
      location to *any* suitable calibration waypoint and performing the
      calibrate action (cost 1). This cost is added *in addition* to the
      navigation cost to the imaging waypoint.

    # Heuristic Initialization
    - Parse static facts to build data structures:
        - Sets of equipped rovers (soil, rock, imaging).
        - Mapping of stores to rovers.
        - Sets of waypoints with soil/rock samples.
        - Mapping of cameras to their properties (rover, supported modes, calibration target).
        - Mapping of objectives/calibration targets to waypoints visible from them.
        - Lander location and waypoints visible from the lander.
        - Navigation graph from `can_traverse` facts.
    - Precompute all-pairs shortest paths on the navigation graph using BFS.
    - Store goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize total heuristic cost to 0.
    2. Extract current state information: rover locations, store status, collected data (soil, rock, image), camera calibration status.
    3. Iterate through each goal fact in `self.goals`.
    4. If the goal fact is already true in the current state, add 0 cost for this goal and continue.
    5. If the goal is `(communicated_soil_data ?w)`:
        - Check if `(have_soil_analysis ?r ?w)` is true for any rover `?r`.
        - If data is collected: Find the rover `?r` that has it. Get its location `current_wp`. Find the minimum navigation cost from `current_wp` to any waypoint visible from the lander (`comm_wp`). Add `min_nav_cost + 1` (communicate) to total cost. If no such rover or no reachable comm_wp, return infinity.
        - If data is not collected: Find an equipped rover `?r`. Check if `?w` has a soil sample. If not, return infinity. Get rover's location `current_wp`. Calculate cost: `dist(current_wp, ?w)` (nav to sample) + `(1 if rover's store is full else 0)` (drop) + `1` (sample) + `min_dist(?w, comm_wp)` (nav from sample to comm) + `1` (communicate). Minimize this cost over all equipped rovers and all suitable communication waypoints. Add the minimum cost to total cost. If no equipped rover or no reachable waypoints, return infinity.
    6. If the goal is `(communicated_rock_data ?w)`: Follow similar logic as soil data, using rock predicates and equipped_for_rock_analysis.
    7. If the goal is `(communicated_image_data ?o ?m)`:
        - Check if `(have_image ?r ?o ?m)` is true for any rover `?r`.
        - If data is collected: Find the rover `?r` that has it. Get its location `current_wp`. Find the minimum navigation cost from `current_wp` to any waypoint visible from the lander (`comm_wp`). Add `min_nav_cost + 1` (communicate) to total cost. If no such rover or no reachable comm_wp, return infinity.
        - If data is not collected: Find an equipped rover `?r` with a camera `?i` that supports mode `?m`. Find a waypoint `?p` visible from objective `?o`. If no such rover/camera/waypoint, return infinity. Get rover's location `current_wp`.
          Calculate base cost: `dist(current_wp, ?p)` (nav to imaging spot) + `1` (take image) + `min_dist(?p, comm_wp)` (nav from imaging to comm) + `1` (communicate).
          If camera `?i` is not calibrated: Find its calibration target `?t` and waypoints `?w` visible from `?t`. If none, return infinity. Add `min_dist(current_wp, ?w)` (nav to cal spot) + `1` (calibrate) to the base cost.
          Minimize this total cost over all suitable rover/camera pairs, imaging waypoints `?p`, and communication waypoints `comm_wp`. Add the minimum cost to total cost. If no suitable options, return infinity.
    8. Return the total heuristic cost. Handle infinity propagation.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # Extract all objects by type from initial state, static facts, and goals
        self.waypoints = set()
        self.rovers = set()
        self.stores = set()
        self.cameras = set()
        self.modes = set()
        self.landers = set()
        self.objectives = set()

        all_relevant_facts = task.initial_state | static_facts | self.goals
        for fact in all_relevant_facts:
             parts = get_parts(fact)
             if not parts: continue
             predicate = parts[0]
             if predicate == 'at' and len(parts) == 3:
                 obj, wp = parts[1], parts[2]
                 if obj.startswith('rover'): self.rovers.add(obj)
                 if wp.startswith('waypoint'): self.waypoints.add(wp)
             elif predicate == 'at_lander' and len(parts) == 3:
                 lander, wp = parts[1], parts[2]
                 self.landers.add(lander)
                 self.waypoints.add(wp)
             elif predicate == 'can_traverse' and len(parts) == 4:
                 rover, wp1, wp2 = parts[1], parts[2], parts[3]
                 self.rovers.add(rover)
                 self.waypoints.add(wp1)
                 self.waypoints.add(wp2)
             elif predicate in ['equipped_for_soil_analysis', 'equipped_for_rock_analysis', 'equipped_for_imaging'] and len(parts) == 2:
                 self.rovers.add(parts[1])
             elif predicate in ['empty', 'full'] and len(parts) == 2:
                 self.stores.add(parts[1])
             elif predicate in ['have_rock_analysis', 'have_soil_analysis'] and len(parts) == 3:
                 self.rovers.add(parts[1])
                 self.waypoints.add(parts[2])
             elif predicate == 'calibrated' and len(parts) == 3:
                 self.cameras.add(parts[1])
                 self.rovers.add(parts[2])
             elif predicate == 'supports' and len(parts) == 3:
                 self.cameras.add(parts[1])
                 self.modes.add(parts[2])
             elif predicate == 'have_image' and len(parts) == 4:
                 self.rovers.add(parts[1])
                 self.objectives.add(parts[2])
                 self.modes.add(parts[3])
             elif predicate in ['communicated_soil_data', 'communicated_rock_data'] and len(parts) == 2:
                 self.waypoints.add(parts[1])
             elif predicate == 'communicated_image_data' and len(parts) == 3:
                 self.objectives.add(parts[1])
                 self.modes.add(parts[2])
             elif predicate in ['at_soil_sample', 'at_rock_sample'] and len(parts) == 2:
                 self.waypoints.add(parts[1])
             elif predicate == 'visible_from' and len(parts) == 3:
                 # Add both objective and waypoint
                 self.objectives.add(parts[1])
                 self.waypoints.add(parts[2])
             elif predicate == 'store_of' and len(parts) == 3:
                 self.stores.add(parts[1])
                 self.rovers.add(parts[2])
             elif predicate == 'calibration_target' and len(parts) == 3:
                 self.cameras.add(parts[1])
                 self.objectives.add(parts[2]) # Calibration target is an objective
             elif predicate == 'on_board' and len(parts) == 3:
                 self.cameras.add(parts[1])
                 self.rovers.add(parts[2])
             elif predicate == 'visible' and len(parts) == 3:
                 self.waypoints.add(parts[1])
                 self.waypoints.add(parts[2])


        # Extract static information into suitable data structures
        self.equipped_for_soil = {get_parts(f)[1] for f in static_facts if match(f, "equipped_for_soil_analysis", "*")}
        self.equipped_for_rock = {get_parts(f)[1] for f in static_facts if match(f, "equipped_for_rock_analysis", "*")}
        self.equipped_for_imaging = {get_parts(f)[1] for f in static_facts if match(f, "equipped_for_imaging", "*")}
        self.store_to_rover = {get_parts(f)[1]: get_parts(f)[2] for f in static_facts if match(f, "store_of", "*", "*")}
        self.at_soil_sample_wps = {get_parts(f)[1] for f in static_facts if match(f, "at_soil_sample", "*")}
        self.at_rock_sample_wps = {get_parts(f)[1] for f in static_facts if match(f, "at_rock_sample", "*")}

        self.camera_info = {}
        for cam in self.cameras:
            info = {'rover': None, 'supports': set(), 'calibration_target': None}
            for fact in static_facts:
                if match(fact, "on_board", cam, "*"):
                    info['rover'] = get_parts(fact)[2]
                elif match(fact, "supports", cam, "*"):
                    info['supports'].add(get_parts(fact)[2])
                elif match(fact, "calibration_target", cam, "*"):
                    info['calibration_target'] = get_parts(fact)[2]
            if info['rover']: # Only store info for cameras on board a rover
                 self.camera_info[cam] = info

        self.objective_imaging_wps = defaultdict(set)
        self.calibration_target_wps = defaultdict(set)
        for fact in static_facts:
            if match(fact, "visible_from", "*", "*"):
                obj, wp = get_parts(fact)[1], get_parts(fact)[2]
                # Check if this objective is a calibration target for any camera
                is_cal_target = any(info.get('calibration_target') == obj for info in self.camera_info.values())
                if is_cal_target:
                     self.calibration_target_wps[obj].add(wp)
                # Assume all visible_from facts are relevant for imaging objectives
                self.objective_imaging_wps[obj].add(wp)


        self.lander_wp = None
        lander_facts = [f for f in static_facts if match(f, "at_lander", "*", "*")]
        if lander_facts:
             self.lander_wp = get_parts(lander_facts[0])[2] # Assuming only one lander

        self.visible_communication_wps = set()
        if self.lander_wp:
            for fact in static_facts:
                if match(fact, "visible", "*", self.lander_wp):
                    self.visible_communication_wps.add(get_parts(fact)[1])
                elif match(fact, "visible", self.lander_wp, "*"): # Assuming visible is symmetric
                     self.visible_communication_wps.add(get_parts(fact)[2])


        # Build navigation graph
        self.navigation_graph = defaultdict(set)
        for fact in static_facts:
            if match(fact, "can_traverse", "*", "*", "*"):
                _, _, wp1, wp2 = get_parts(fact)
                self.navigation_graph[wp1].add(wp2)
                self.navigation_graph[wp2].add(wp1) # Assuming symmetric traversal

        # Ensure all waypoints are keys in the graph, even if isolated
        for wp in self.waypoints:
             if wp not in self.navigation_graph:
                  self.navigation_graph[wp] = set()


        # Precompute all-pairs shortest paths
        self.dist = {}
        for wp in self.waypoints:
            self.dist[wp] = bfs(self.navigation_graph, wp)

        # Ensure all-pairs distances are stored, using infinity for unreachable
        for wp1 in self.waypoints:
            for wp2 in self.waypoints:
                if wp2 not in self.dist[wp1]:
                    self.dist[wp1][wp2] = float('inf')


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        total_cost = 0

        # Extract current state information
        rover_locations = {}
        for fact in state:
            if match(fact, "at", "rover*", "waypoint*"):
                rover, wp = get_parts(fact)[1], get_parts(fact)[2]
                rover_locations[rover] = wp

        store_status = {}
        for store, rover in self.store_to_rover.items():
             store_status[store] = 'full' if '(full '+store+')' in state else 'empty'

        have_soil = {(get_parts(f)[1], get_parts(f)[2]) for f in state if match(f, "have_soil_analysis", "rover*", "waypoint*")}
        have_rock = {(get_parts(f)[1], get_parts(f)[2]) for f in state if match(f, "have_rock_analysis", "rover*", "waypoint*")}
        have_image = {(get_parts(f)[1], get_parts(f)[2], get_parts(f)[3]) for f in state if match(f, "have_image", "rover*", "objective*", "mode*")}
        calibrated_cams = {(get_parts(f)[1], get_parts(f)[2]) for f in state if match(f, "calibrated", "camera*", "rover*")}

        # Check each goal fact
        for goal in self.goals:
            if goal in state:
                continue # Goal already achieved

            parts = get_parts(goal)
            if not parts or parts[0] not in ["communicated_soil_data", "communicated_rock_data", "communicated_image_data"]:
                 # Ignore non-communication goals or malformed goals for this heuristic
                 continue

            goal_type = parts[0]

            if goal_type == "communicated_soil_data":
                w = parts[1]
                # Check if data is already collected by any rover
                data_collected = any((r, w) in have_soil for r in self.rovers)

                if data_collected:
                    # Find the rover that has the data
                    rover_with_data = next((r for r in self.rovers if (r, w) in have_soil), None)
                    if rover_with_data is None: return float('inf') # Should not happen if data_collected is True

                    current_wp = rover_locations.get(rover_with_data)
                    if current_wp is None: return float('inf') # Rover location unknown

                    # Estimate communication cost
                    min_nav_to_comm = float('inf')
                    for comm_wp in self.visible_communication_wps:
                        min_nav_to_comm = min(min_nav_to_comm, self.dist[current_wp].get(comm_wp, float('inf')))

                    if min_nav_to_comm == float('inf'): return float('inf') # Cannot reach communication point

                    total_cost += min_nav_to_comm + 1 # navigate + communicate

                else: # Data not collected
                    # Check if sampling is possible at all
                    if w not in self.at_soil_sample_wps: return float('inf') # No soil sample here

                    min_collect_and_comm_cost = float('inf')

                    # Find suitable rovers (equipped for soil)
                    suitable_rovers = [r for r in self.rovers if r in self.equipped_for_soil]
                    if not suitable_rovers: return float('inf') # No equipped rover

                    for r in suitable_rovers:
                        current_wp = rover_locations.get(r)
                        if current_wp is None: continue # Rover location unknown

                        # Cost to navigate to sample location
                        nav_to_sample_cost = self.dist[current_wp].get(w, float('inf'))
                        if nav_to_sample_cost == float('inf'): continue # Cannot reach sample location

                        # Cost to drop if store is full
                        rover_store = next((s for s, rv in self.store_to_rover.items() if rv == r), None)
                        drop_cost = 0
                        if rover_store and store_status.get(rover_store) == 'full':
                            drop_cost = 1

                        # Cost to sample
                        sample_cost = 1

                        # Cost to navigate from sample location to communication location
                        min_nav_from_sample_to_comm = float('inf')
                        for comm_wp in self.visible_communication_wps:
                             min_nav_from_sample_to_comm = min(min_nav_from_sample_to_comm, self.dist[w].get(comm_wp, float('inf')))

                        if min_nav_from_sample_to_comm == float('inf'): continue # Cannot reach communication point from sample location

                        # Cost to communicate
                        communicate_cost = 1

                        cost_for_this_rover = nav_to_sample_cost + drop_cost + sample_cost + min_nav_from_sample_to_comm + communicate_cost
                        min_collect_and_comm_cost = min(min_collect_and_comm_cost, cost_for_this_rover)

                    if min_collect_and_comm_cost == float('inf'): return float('inf') # No suitable rover could achieve this

                    total_cost += min_collect_and_comm_cost

            elif goal_type == "communicated_rock_data":
                w = parts[1]
                # Check if data is already collected by any rover
                data_collected = any((r, w) in have_rock for r in self.rovers)

                if data_collected:
                    # Find the rover that has the data
                    rover_with_data = next((r for r in self.rovers if (r, w) in have_rock), None)
                    if rover_with_data is None: return float('inf') # Should not happen

                    current_wp = rover_locations.get(rover_with_data)
                    if current_wp is None: return float('inf') # Rover location unknown

                    # Estimate communication cost
                    min_nav_to_comm = float('inf')
                    for comm_wp in self.visible_communication_wps:
                        min_nav_to_comm = min(min_nav_to_comm, self.dist[current_wp].get(comm_wp, float('inf')))

                    if min_nav_to_comm == float('inf'): return float('inf') # Cannot reach communication point

                    total_cost += min_nav_to_comm + 1 # navigate + communicate

                else: # Data not collected
                    # Check if sampling is possible at all
                    if w not in self.at_rock_sample_wps: return float('inf') # No rock sample here

                    min_collect_and_comm_cost = float('inf')

                    # Find suitable rovers (equipped for rock)
                    suitable_rovers = [r for r in self.rovers if r in self.equipped_for_rock]
                    if not suitable_rovers: return float('inf') # No equipped rover

                    for r in suitable_rovers:
                        current_wp = rover_locations.get(r)
                        if current_wp is None: continue # Rover location unknown

                        # Cost to navigate to sample location
                        nav_to_sample_cost = self.dist[current_wp].get(w, float('inf'))
                        if nav_to_sample_cost == float('inf'): continue # Cannot reach sample location

                        # Cost to drop if store is full
                        rover_store = next((s for s, rv in self.store_to_rover.items() if rv == r), None)
                        drop_cost = 0
                        if rover_store and store_status.get(rover_store) == 'full':
                            drop_cost = 1

                        # Cost to sample
                        sample_cost = 1

                        # Cost to navigate from sample location to communication location
                        min_nav_from_sample_to_comm = float('inf')
                        for comm_wp in self.visible_communication_wps:
                             min_nav_from_sample_to_comm = min(min_nav_from_sample_to_comm, self.dist[w].get(comm_wp, float('inf')))

                        if min_nav_from_sample_to_comm == float('inf'): continue # Cannot reach communication point from sample location

                        # Cost to communicate
                        communicate_cost = 1

                        cost_for_this_rover = nav_to_sample_cost + drop_cost + sample_cost + min_nav_from_sample_to_comm + communicate_cost
                        min_collect_and_comm_cost = min(min_collect_and_comm_cost, cost_for_this_rover)

                    if min_collect_and_comm_cost == float('inf'): return float('inf') # No suitable rover could achieve this

                    total_cost += min_collect_and_comm_cost

            elif goal_type == "communicated_image_data":
                o, m = parts[1], parts[2]
                # Check if data is already collected by any rover
                data_collected = any((r, o, m) in have_image for r in self.rovers)

                if data_collected:
                    # Find the rover that has the data
                    rover_with_data = next((r for r in self.rovers if (r, o, m) in have_image), None)
                    if rover_with_data is None: return float('inf') # Should not happen

                    current_wp = rover_locations.get(rover_with_data)
                    if current_wp is None: return float('inf') # Rover location unknown

                    # Estimate communication cost
                    min_nav_to_comm = float('inf')
                    for comm_wp in self.visible_communication_wps:
                        min_nav_to_comm = min(min_nav_to_comm, self.dist[current_wp].get(comm_wp, float('inf')))

                    if min_nav_to_comm == float('inf'): return float('inf') # Cannot reach communication point

                    total_cost += min_nav_to_comm + 1 # navigate + communicate

                else: # Data not collected
                    # Check if imaging is possible at all for this objective
                    if o not in self.objective_imaging_wps or not self.objective_imaging_wps[o]:
                         return float('inf') # No waypoint to view this objective from

                    min_collect_and_comm_cost = float('inf')

                    # Find suitable rovers/cameras
                    suitable_rover_camera_pairs = []
                    for r in self.equipped_for_imaging:
                        for cam, info in self.camera_info.items():
                            if info['rover'] == r and m in info['supports']:
                                suitable_rover_camera_pairs.append((r, cam))

                    if not suitable_rover_camera_pairs: return float('inf') # No equipped rover with supporting camera

                    for r, cam in suitable_rover_camera_pairs:
                        current_wp = rover_locations.get(r)
                        if current_wp is None: continue # Rover location unknown

                        # Find best imaging waypoint
                        imaging_wps = self.objective_imaging_wps[o]
                        min_nav_to_img = float('inf')
                        best_img_wp = None
                        for img_wp in imaging_wps:
                            dist = self.dist[current_wp].get(img_wp, float('inf'))
                            if dist < min_nav_to_img:
                                min_nav_to_img = dist
                                best_img_wp = img_wp

                        if min_nav_to_img == float('inf'): continue # Cannot reach any imaging waypoint

                        # Estimate calibration cost if needed
                        calibration_cost = 0
                        if (cam, r) not in calibrated_cams:
                            cal_target = self.camera_info[cam].get('calibration_target')
                            if cal_target is None: continue # Camera has no calibration target
                            cal_wps = self.calibration_target_wps.get(cal_target)
                            if not cal_wps: continue # No waypoint to view calibration target from

                            min_nav_to_cal = float('inf')
                            for cal_wp in cal_wps:
                                min_nav_to_cal = min(min_nav_to_cal, self.dist[current_wp].get(cal_wp, float('inf')))

                            if min_nav_to_cal == float('inf'): continue # Cannot reach any calibration waypoint

                            # Simplified calibration cost: nav from current location to cal spot + calibrate action
                            calibration_cost = min_nav_to_cal + 1

                        # Cost to take image
                        take_image_cost = 1

                        # Cost to navigate from imaging location to communication location
                        min_nav_from_img_to_comm = float('inf')
                        if best_img_wp:
                            for comm_wp in self.visible_communication_wps:
                                min_nav_from_img_to_comm = min(min_nav_from_img_to_comm, self.dist[best_img_wp].get(comm_wp, float('inf')))

                        if min_nav_from_img_to_comm == float('inf'): continue # Cannot reach communication point from imaging location

                        # Cost to communicate
                        communicate_cost = 1

                        # Total cost for this rover/camera pair
                        cost_for_this_pair = min_nav_to_img + calibration_cost + take_image_cost + min_nav_from_img_to_comm + communicate_cost
                        min_collect_and_comm_cost = min(min_collect_and_comm_cost, cost_for_this_pair)

                    if min_collect_and_comm_cost == float('inf'): return float('inf') # No suitable rover/camera could achieve this

                    total_cost += min_collect_and_comm_cost

        return total_cost
