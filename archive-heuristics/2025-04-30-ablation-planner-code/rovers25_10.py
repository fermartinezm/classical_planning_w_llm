from collections import deque
# from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

class roversHeuristic: # Inherit from Heuristic if available
# class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the remaining actions to satisfy all goal conditions.
    It sums up the estimated costs for each unsatisfied goal fact independently.
    The cost for each goal fact includes the necessary actions (sample, calibrate,
    take_image, communicate, drop) and estimated navigation costs.

    # Assumptions
    - Each unsatisfied goal fact (communicated_soil_data, communicated_rock_data,
      communicated_image_data) is treated independently.
    - Navigation costs are estimated using shortest paths on the rover's
      can_traverse graph.
    - For image goals, calibration is assumed to happen before taking the image,
      and navigation costs reflect this sequence (curr -> cal_wp -> img_wp).
    - Drop action is only considered if a store is full and a sample is needed.
    - Assumes solvable problems where necessary equipment, samples, objectives,
      calibration targets, and paths exist. Unreachable goals are assigned a large cost (1000).

    # Heuristic Initialization
    - Parses static facts to build navigation graphs for each rover,
      precompute all-pairs shortest paths (or single-source from all waypoints).
    - Extracts static relationships: rover equipment, camera info (on_board, supports,
      calibration_target), objective visibility, lander locations, store ownership,
      and potential communication waypoints (visible from lander).
    - Stores goal facts.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify all goal facts that are not yet true in the current state.
    2. Initialize total heuristic cost to 0.
    3. For each unsatisfied goal fact:
        a. Estimate the cost to achieve this specific goal fact. This cost is added
           to the total heuristic.
        b. Cost for `(communicated_soil_data ?w)`:
           - Add 1 for the `communicate_soil_data` action.
           - If no rover currently has `(have_soil_analysis ?r ?w)`:
             - Add 1 for the `sample_soil` action.
             - Find the best soil-equipped rover `r` (minimizing nav + drop cost).
             - Add 1 if `r`'s store is full (for a `drop` action).
             - Add the minimum navigation cost for `r` from its current location to `w`.
             - The rover is now conceptually at `w`.
             - Add the minimum navigation cost for `r` from `w` to any communication waypoint.
           - If a rover `r` already has `(have_soil_analysis ?r ?w)`:
             - Add the minimum navigation cost for `r` from its current location to any communication waypoint.
        c. Cost for `(communicated_rock_data ?w)`: Similar logic to soil data.
        d. Cost for `(communicated_image_data ?o ?m)`:
           - Add 1 for the `communicate_image_data` action.
           - If no rover currently has `(have_image ?r ?o ?m)`:
             - Add 1 for the `take_image` action.
             - Find the best imaging-equipped rover `r` with a camera `i` supporting mode `m` (minimizing total image task cost).
             - Find the best image waypoint `p` where `o` is visible.
             - Find the best calibration waypoint `w` where `i`'s target is visible.
             - If camera `i` is not calibrated on rover `r`:
               - Add 1 for the `calibrate` action.
               - Add navigation cost: `min_nav_cost(r_curr, w)` + `min_nav_cost(w, p)`.
               - The rover is now conceptually at `p`.
             - If camera `i` is calibrated:
               - Add navigation cost: `min_nav_cost(r_curr, p)`.
               - The rover is now conceptually at `p`.
             - Add the minimum navigation cost for `r` from `p` to any communication waypoint.
           - If a rover `r` already has `(have_image ?r ?o ?m)`:
             - Add the minimum navigation cost for `r` from its current location to any communication waypoint.
    4. Return the total calculated cost. If any necessary navigation is impossible, return a large value (1000).
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = set(task.goals) # Use a set for faster lookup
        static_facts = set(task.static) # Use a set for faster lookup

        # Data structures to store static information
        self.rovers = set()
        self.waypoints = set()
        self.lander_locations = set() # Can be multiple landers at different spots
        self.calibration_targets = {} # camera -> objective
        self.camera_on_board = {} # camera -> rover
        self.camera_supports = {} # camera -> [mode, ...]
        self.objective_visible_from = {} # objective -> [waypoint, ...]
        self.lander_visible_from = {} # lander_wp -> [waypoint, ...] (waypoints from which lander is visible)
        self.rover_stores = {} # rover -> store
        self.rover_equipment = {} # rover -> {soil: bool, rock: bool, imaging: bool}
        self.rover_nav_graphs = {} # rover -> {wp_from -> [wp_to, ...]}

        # Parse static facts
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'rover': self.rovers.add(parts[1])
            elif parts[0] == 'waypoint': self.waypoints.add(parts[1])
            elif parts[0] == 'lander': pass # Type declaration
            elif parts[0] == 'store': pass # Type declaration
            elif parts[0] == 'camera': pass # Type declaration
            elif parts[0] == 'mode': pass # Type declaration
            elif parts[0] == 'objective': pass # Type declaration

            elif parts[0] == 'at_lander': self.lander_locations.add(parts[2])
            elif parts[0] == 'can_traverse':
                r, w1, w2 = parts[1:]
                if r not in self.rover_nav_graphs: self.rover_nav_graphs[r] = {}
                if w1 not in self.rover_nav_graphs[r]: self.rover_nav_graphs[r][w1] = []
                self.rover_nav_graphs[r][w1].append(w2)
            elif parts[0] == 'equipped_for_soil_analysis':
                if parts[1] not in self.rover_equipment: self.rover_equipment[parts[1]] = {}
                self.rover_equipment[parts[1]]['soil'] = True
            elif parts[0] == 'equipped_for_rock_analysis':
                if parts[1] not in self.rover_equipment: self.rover_equipment[parts[1]] = {}
                self.rover_equipment[parts[1]]['rock'] = True
            elif parts[0] == 'equipped_for_imaging':
                if parts[1] not in self.rover_equipment: self.rover_equipment[parts[1]] = {}
                self.rover_equipment[parts[1]]['imaging'] = True
            elif parts[0] == 'store_of': self.rover_stores[parts[2]] = parts[1] # rover -> store
            elif parts[0] == 'calibration_target': self.calibration_targets[parts[1]] = parts[2]
            elif parts[0] == 'on_board': self.camera_on_board[parts[1]] = parts[2]
            elif parts[0] == 'supports':
                c, m = parts[1:]
                if c not in self.camera_supports: self.camera_supports[c] = []
                self.camera_supports[c].append(m)
            elif parts[0] == 'visible':
                w1, w2 = parts[1:]
                # We need waypoints visible *from* the lander location(s)
                if w2 in self.lander_locations:
                     if w2 not in self.lander_visible_from: self.lander_visible_from[w2] = []
                     self.lander_visible_from[w2].append(w1) # w1 is visible from lander_wp
            elif parts[0] == 'visible_from':
                o, w = parts[1:]
                if o not in self.objective_visible_from: self.objective_visible_from[o] = []
                self.objective_visible_from[o].append(w)
            # at_soil_sample and at_rock_sample are dynamic, but initial ones are in static
            # We will check state directly for these.

        # Precompute distances for each rover
        self.rover_distances = {} # rover -> {wp_from -> {wp_to -> distance}}
        for rover in self.rovers:
            self.rover_distances[rover] = {}
            graph = self.rover_nav_graphs.get(rover, {})
            # Consider all waypoints in the domain as potential start/end points for BFS
            all_waypoints_in_domain = list(self.waypoints) # Use the set of all waypoints declared in objects

            for start_wp in all_waypoints_in_domain:
                 self.rover_distances[rover][start_wp] = self._bfs(start_wp, graph)

        # Find potential communication waypoints (waypoints visible from any lander location)
        self.communication_waypoints = set()
        for lander_wp in self.lander_locations:
            self.communication_waypoints.update(self.lander_visible_from.get(lander_wp, []))

    def _bfs(self, start_waypoint, graph):
        """Computes shortest path distances from start_waypoint to all reachable waypoints."""
        distances = {start_waypoint: 0}
        queue = deque([start_waypoint])
        visited = {start_waypoint}

        while queue:
            current_wp = queue.popleft()
            # Neighbors are waypoints reachable from current_wp
            neighbors = graph.get(current_wp, [])
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    distances[neighbor] = distances[current_wp] + 1
                    queue.append(neighbor)
        return distances

    def _min_nav_cost_and_wp(self, rover, start_wp, target_wps):
        """
        Finds the minimum navigation cost for 'rover' from 'start_wp' to any waypoint in 'target_wps'
        and returns the minimum cost and the target waypoint that achieves it.
        Returns (1000, None) if unreachable or no target_wps.
        """
        if not target_wps:
             return (0, None)

        if rover not in self.rover_distances or start_wp not in self.rover_distances[rover]:
             return (1000, None) # Large penalty

        distances_from_start = self.rover_distances[rover].get(start_wp, {}) # Handle start_wp not in precomputed table
        if not distances_from_start:
             return (1000, None) # Start waypoint is isolated or invalid

        min_dist = float('inf')
        best_wp = None
        for target_wp in target_wps:
            if target_wp in distances_from_start:
                if distances_from_start[target_wp] < min_dist:
                    min_dist = distances_from_start[target_wp]
                    best_wp = target_wp

        return (min_dist if min_dist != float('inf') else 1000, best_wp)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        state_facts = set(state) # Convert frozenset to set for faster lookups

        # Extract current state info
        rover_locations = {} # rover -> waypoint
        rover_full_stores = set() # store
        rover_has_soil = {} # (rover, waypoint) -> bool
        rover_has_rock = {} # (rover, waypoint) -> bool
        rover_has_image = {} # (rover, objective, mode) -> bool
        camera_calibrated = set() # (camera, rover)
        # soil_samples_at_wp = set() # waypoint # Not needed for heuristic calculation based on goals
        # rock_samples_at_wp = set() # waypoint # Not needed for heuristic calculation based on goals

        for fact in state_facts:
            parts = get_parts(fact)
            if parts[0] == 'at' and parts[1] in self.rovers:
                rover_locations[parts[1]] = parts[2]
            elif parts[0] == 'full':
                rover_full_stores.add(parts[1])
            elif parts[0] == 'have_soil_analysis':
                rover_has_soil[(parts[1], parts[2])] = True
            elif parts[0] == 'have_rock_analysis':
                rover_has_rock[(parts[1], parts[2])] = True
            elif parts[0] == 'have_image':
                rover_has_image[(parts[1], parts[2], parts[3])] = True
            elif parts[0] == 'calibrated':
                camera_calibrated.add((parts[1], parts[2]))
            # elif parts[0] == 'at_soil_sample': soil_samples_at_wp.add(parts[1])
            # elif parts[0] == 'at_rock_sample': rock_samples_at_wp.add(parts[1])


        total_cost = 0

        # Check each goal fact
        for goal_fact_str in self.goals:
            if goal_fact_str in state_facts:
                continue # Goal already achieved

            parts = get_parts(goal_fact_str)
            predicate = parts[0]

            if predicate == 'communicated_soil_data':
                waypoint = parts[1]
                cost_for_goal = 0

                # 1. Communication action
                cost_for_goal += 1

                # 2. Have sample
                needs_sample = True
                rover_with_sample = None
                for rover in self.rovers:
                    if rover_has_soil.get((rover, waypoint), False):
                        needs_sample = False
                        rover_with_sample = rover
                        break

                if needs_sample:
                    # Need to sample
                    cost_for_goal += 1 # sample_soil action
                    min_sample_nav_cost = float('inf')
                    min_sample_drop_cost = float('inf')
                    best_sample_rover = None

                    # Find best rover to sample
                    for rover in self.rovers:
                         if self.rover_equipment.get(rover, {}).get('soil', False):
                            curr_wp = rover_locations.get(rover)
                            if curr_wp:
                                nav_cost, _ = self._min_nav_cost_and_wp(rover, curr_wp, [waypoint])
                                drop_cost = 1 if self.rover_stores.get(rover) in rover_full_stores else 0

                                if nav_cost + drop_cost < min_sample_nav_cost + min_sample_drop_cost:
                                     min_sample_nav_cost = nav_cost
                                     min_sample_drop_cost = drop_cost
                                     best_sample_rover = rover

                    if best_sample_rover and min_sample_nav_cost != 1000:
                         cost_for_goal += min_sample_nav_cost + min_sample_drop_cost
                         rover_for_comm = best_sample_rover # This rover will do the communication
                         # Navigation for communication starts from sample waypoint
                         nav_start_wp_for_comm = waypoint
                    else:
                         # No soil-equipped rover or no path to sample location
                         cost_for_goal += 1000 # Large penalty
                         rover_for_comm = None # Cannot communicate if no rover can sample
                         nav_start_wp_for_comm = None

                else: # Already has sample
                    rover_for_comm = rover_with_sample
                    # Navigation for communication starts from current location of the rover with sample
                    nav_start_wp_for_comm = rover_locations.get(rover_for_comm)


                # 3. Navigation to communication waypoint
                if rover_for_comm and nav_start_wp_for_comm:
                     min_comm_nav_cost, _ = self._min_nav_cost_and_wp(rover_for_comm, nav_start_wp_for_comm, list(self.communication_waypoints))
                     cost_for_goal += min_comm_nav_cost
                elif rover_for_comm: # Rover exists but start_wp is None (shouldn't happen)
                     cost_for_goal += 1000
                else: # No rover can perform the task
                     cost_for_goal += 1000


                total_cost += cost_for_goal

            elif predicate == 'communicated_rock_data':
                waypoint = parts[1]
                cost_for_goal = 0

                # 1. Communication action
                cost_for_goal += 1

                # 2. Have sample
                needs_sample = True
                rover_with_sample = None
                for rover in self.rovers:
                    if rover_has_rock.get((rover, waypoint), False):
                        needs_sample = False
                        rover_with_sample = rover
                        break

                if needs_sample:
                    # Need to sample
                    cost_for_goal += 1 # sample_rock action
                    min_sample_nav_cost = float('inf')
                    min_sample_drop_cost = float('inf')
                    best_sample_rover = None

                    # Find best rover to sample
                    for rover in self.rovers:
                         if self.rover_equipment.get(rover, {}).get('rock', False):
                            curr_wp = rover_locations.get(rover)
                            if curr_wp:
                                nav_cost, _ = self._min_nav_cost_and_wp(rover, curr_wp, [waypoint])
                                drop_cost = 1 if self.rover_stores.get(rover) in rover_full_stores else 0

                                if nav_cost + drop_cost < min_sample_nav_cost + min_sample_drop_cost:
                                     min_sample_nav_cost = nav_cost
                                     min_sample_drop_cost = drop_cost
                                     best_sample_rover = rover

                    if best_sample_rover and min_sample_nav_cost != 1000:
                         cost_for_goal += min_sample_nav_cost + min_sample_drop_cost
                         rover_for_comm = best_sample_rover # This rover will do the communication
                         nav_start_wp_for_comm = waypoint
                    else:
                         cost_for_goal += 1000
                         rover_for_comm = None
                         nav_start_wp_for_comm = None

                else: # Already has sample
                    rover_for_comm = rover_with_sample
                    nav_start_wp_for_comm = rover_locations.get(rover_for_comm)

                # 3. Navigation to communication waypoint
                if rover_for_comm and nav_start_wp_for_comm:
                     min_comm_nav_cost, _ = self._min_nav_cost_and_wp(rover_for_comm, nav_start_wp_for_comm, list(self.communication_waypoints))
                     cost_for_goal += min_comm_nav_cost
                elif rover_for_comm:
                     cost_for_goal += 1000
                else:
                     cost_for_goal += 1000

                total_cost += cost_for_goal

            elif predicate == 'communicated_image_data':
                objective, mode = parts[1:]
                cost_for_goal = 0

                # 1. Communication action
                cost_for_goal += 1

                # 2. Have image
                needs_image = True
                rover_with_image = None
                for rover in self.rovers:
                    if rover_has_image.get((rover, objective, mode), False):
                        needs_image = False
                        rover_with_image = rover
                        break

                if needs_image:
                    # Need to take image
                    cost_for_goal += 1 # take_image action
                    min_image_task_cost = float('inf') # Cost includes nav to cal, cal, nav to image
                    best_image_rover = None
                    best_image_wp_after_task = None # Where the rover ends up after taking image

                    # Find best rover/camera combination
                    for rover in self.rovers:
                         if self.rover_equipment.get(rover, {}).get('imaging', False):
                            curr_wp = rover_locations.get(rover)
                            if not curr_wp: continue

                            for camera in self.camera_on_board:
                                if self.camera_on_board[camera] == rover and mode in self.camera_supports.get(camera, []):
                                    # Found a suitable rover/camera

                                    # Find best image waypoint
                                    image_wps = self.objective_visible_from.get(objective, [])
                                    if not image_wps: continue # Cannot take image if no visible waypoint

                                    # Find best calibration waypoint
                                    cal_target = self.calibration_targets.get(camera)
                                    cal_wps = self.objective_visible_from.get(cal_target, []) if cal_target else []
                                    needs_calibration = (camera, rover) not in camera_calibrated

                                    if needs_calibration and not cal_wps:
                                         # Needs calibration but no cal waypoint
                                         continue # Cannot perform task with this camera

                                    # Calculate cost to get image (nav to cal + cal + nav to image)
                                    current_image_task_cost = 0
                                    nav_start_wp = curr_wp

                                    if needs_calibration:
                                        current_image_task_cost += 1 # calibrate action
                                        # Find best cal_wp and nav cost to it
                                        nav_to_cal, chosen_cal_wp = self._min_nav_cost_and_wp(rover, nav_start_wp, cal_wps)
                                        if chosen_cal_wp is None or nav_to_cal == 1000: continue # Cannot reach any cal waypoint

                                        current_image_task_cost += nav_to_cal
                                        nav_start_wp = chosen_cal_wp # Rover is now conceptually at cal_wp

                                    # Find best image_wp and nav cost from current nav_start_wp
                                    nav_to_image, chosen_image_wp = self._min_nav_cost_and_wp(rover, nav_start_wp, image_wps)
                                    if chosen_image_wp is None or nav_to_image == 1000: continue # Cannot reach any image waypoint

                                    current_image_task_cost += nav_to_image
                                    # Rover is now conceptually at chosen_image_wp after taking image
                                    wp_after_task = chosen_image_wp

                                    # Update minimum cost found so far
                                    if current_image_task_cost < min_image_task_cost:
                                         min_image_task_cost = current_image_task_cost
                                         best_image_rover = rover
                                         best_image_wp_after_task = wp_after_task


                    if best_image_rover and min_image_task_cost != float('inf'):
                         cost_for_goal += min_image_task_cost
                         rover_for_comm = best_image_rover
                         nav_start_wp_for_comm = best_image_wp_after_task
                    else:
                         cost_for_goal += 1000
                         rover_for_comm = None
                         nav_start_wp_for_comm = None


                else: # Already has image
                    rover_for_comm = rover_with_image
                    nav_start_wp_for_comm = rover_locations.get(rover_for_comm)


                # 3. Navigation to communication waypoint
                if rover_for_comm and nav_start_wp_for_comm:
                     min_comm_nav_cost, _ = self._min_nav_cost_and_wp(rover_for_comm, nav_start_wp_for_comm, list(self.communication_waypoints))
                     cost_for_goal += min_comm_nav_cost
                elif rover_for_comm: # Rover exists but start_wp is None (shouldn't happen)
                     cost_for_goal += 1000
                else: # No rover can perform the task
                     cost_for_goal += 1000

                total_cost += cost_for_goal

        # Ensure heuristic is 0 only at goal
        # The loop structure guarantees total_cost > 0 if any goal is not in state_facts.
        # If self.goals is empty, total_cost is 0. If self.goals is not empty but all are in state_facts, total_cost is 0.
        # This is correct.

        return total_cost
