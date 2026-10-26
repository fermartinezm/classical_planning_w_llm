import itertools
import heapq
from collections import defaultdict
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
# Make sure the base class `Heuristic` is available in the environment.
# If not, define a placeholder:
# class Heuristic:
#     def __init__(self, task): pass
#     def __call__(self, node): raise NotImplementedError

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extracts components from a PDDL fact string by removing parentheses and splitting."""
    return fact[1:-1].split()

def match(fact, *args):
    """Checks if a PDDL fact matches a pattern (supports '*' wildcards)."""
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class RoversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the PDDL Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal state
    by summing the estimated costs for achieving each unsatisfied goal predicate independently.
    It considers navigation, sampling, calibration, imaging, and communication actions.
    The cost for each goal is estimated by finding the cheapest sequence of actions
    using any capable rover, including pre-calculated shortest path navigation costs.
    This approach is similar to the additive heuristic (h_add).

    # Assumptions
    - Navigation costs are based on the shortest path in terms of 'navigate' actions (unit cost).
    - The heuristic assumes rovers can perform actions sequentially for a single goal
      without interference from other rovers or goals (relaxed assumption). Resource contention
      like store state or camera calibration is considered for the *individual* rover's plan
      for that goal, but not across goals or rovers simultaneously.
    - If multiple waypoints satisfy a visibility requirement (for imaging, calibration,
      communication), the one closest to the rover's relevant location (in terms of navigation steps)
      is chosen to estimate the cost.
    - Infinite cost (float('inf')) is used if a required path or action sequence is impossible
      from the current state for all capable rovers, indicating a potential dead end.

    # Heuristic Initialization
    - Parses static facts from the task to build efficient lookups for:
        - Rover equipment and capabilities (soil, rock, imaging).
        - Camera properties (which rover it's on, supported modes, calibration targets).
        - Visibility information (waypoint-to-waypoint, objective-to-waypoint).
        - Lander location.
        - Rover-specific store mapping.
        - Rover-specific traversability (`can_traverse`).
    - Pre-computes All-Pairs Shortest Paths (APSP) for each rover based on
      'can_traverse' and 'visible' predicates using Breadth-First Search (BFS).
      Distances (number of navigate actions) are stored in `self.rover_apsp[rover][from_wp][to_wp]`.
    - Stores the goal predicates in a parsed format.
    - Identifies waypoints from which the lander is visible for communication.

    # Step-By-Step Thinking for Computing Heuristic
    1. Check if the current state satisfies all goal predicates. If yes, return 0.
    2. Parse the current state (`node.state`) to determine dynamic information:
        - Current location of each rover (`at`).
        - State of each rover's store (`empty` or `full`).
        - Which cameras are currently calibrated (`calibrated`).
        - What soil/rock analyses are held by which rovers (`have_soil_analysis`, `have_rock_analysis`).
        - What images are held by which rovers (`have_image`).
    3. Identify all goal predicates defined in `task.goals` that are NOT satisfied in the current state.
    4. Initialize total heuristic cost `h = 0`.
    5. For each unsatisfied goal `g`:
        a. Determine the type of goal (`communicated_soil_data`, `communicated_rock_data`, `communicated_image_data`).
        b. Find all rovers potentially capable of achieving this goal based on their static equipment.
        c. Initialize `min_goal_cost = float('inf')`.
        d. For each capable rover `r`:
            i.   Estimate the minimum action cost `cost_r(g)` for rover `r` to achieve goal `g`, starting from the current state. This cost calculation involves two main phases: Acquisition and Communication.
            ii.  **Acquisition Cost:** Calculate the cost to get the necessary prerequisite fact (`have_soil_analysis`, `have_rock_analysis`, or `have_image`) if the rover doesn't already have it. This includes:
                 - Cost of `drop` (1 action) if the store is full and sampling is needed.
                 - Cost of `navigate` actions (using precomputed APSP) to reach the sampling location, calibration location, or imaging location. The closest valid waypoint is chosen if multiple options exist.
                 - Cost of `sample_soil`/`sample_rock` (1 action) or `calibrate` (1 action) or `take_image` (1 action).
                 - Track the rover's estimated location after the acquisition phase.
            iii. **Communication Cost:** Calculate the cost to communicate the acquired data/image. This includes:
                 - Cost of `navigate` actions (using APSP) from the rover's location *after acquisition* to the closest waypoint `x` that is visible from the lander's location.
                 - Cost of the communication action (`communicate_soil_data`, etc. - 1 action).
            iv.  The total cost for the rover is `cost_r(g) = cost_acquisition + cost_communication`.
            v.   Update `min_goal_cost = min(min_goal_cost, cost_r(g))`.
        e. If `min_goal_cost` remains `float('inf')` after checking all rovers (meaning no rover can achieve this goal from the current state), return `float('inf')` for the entire state heuristic, as it's likely a dead end.
        f. Add the calculated `min_goal_cost` for goal `g` to the total heuristic cost `h`.
    6. Return the total estimated cost `h`.
    """

    def __init__(self, task):
        self.goals = task.goals
        static_facts = task.static

        # Extract object types from the full fact list
        self.objects = defaultdict(set)
        # Assuming task.facts contains all possible ground atoms, which might not be standard.
        # A safer way is to parse task.objects if available, or infer from static/init/goal facts.
        # Let's infer from static facts and goals for robustness.
        all_facts_to_parse = static_facts.union(task.initial_state).union(task.goals)
        for fact in all_facts_to_parse:
             parts = get_parts(fact)
             # Simple type inference based on predicate signatures (requires knowledge of domain structure)
             # This part might need adjustment based on how types are represented or if task.objects is available
             if parts[0] in ['at', 'can_traverse', 'equipped_for_soil_analysis', 'equipped_for_rock_analysis', 'equipped_for_imaging', 'have_rock_analysis', 'have_soil_analysis', 'calibrated', 'have_image', 'communicate_soil_data', 'communicate_rock_data', 'communicate_image_data']:
                 self.objects['rover'].add(parts[1])
             if parts[0] in ['at', 'at_lander', 'can_traverse', 'have_rock_analysis', 'have_soil_analysis', 'visible', 'communicated_soil_data', 'communicated_rock_data', 'at_soil_sample', 'at_rock_sample', 'visible_from']:
                 # Waypoints appear in various positions
                 for arg in parts[1:]:
                     if arg.startswith('waypoint'): self.objects['waypoint'].add(arg)
             # Add similar logic for other types (camera, objective, store, lander, mode) based on predicates they appear in.
             # Example:
             if parts[0] in ['calibrated', 'supports', 'calibration_target', 'on_board']:
                 if parts[1].startswith('camera'): self.objects['camera'].add(parts[1])
             if parts[0] in ['have_image', 'communicated_image_data', 'visible_from', 'calibration_target']:
                 if parts[1].startswith('objective'): self.objects['objective'].add(parts[1])
                 if len(parts) > 2 and parts[2].startswith('objective'): self.objects['objective'].add(parts[2])
             if parts[0] in ['empty', 'full', 'store_of']:
                 if parts[1].startswith('store'): self.objects['store'].add(parts[1]) # Assuming store names start with 'store'
                 elif len(parts) > 1 and parts[0].startswith('store'): self.objects['store'].add(parts[0]) # store_of case
             if parts[0] == 'at_lander':
                 self.objects['lander'].add(parts[1])
             if parts[0] in ['supports', 'have_image', 'communicated_image_data']:
                 # Modes can be in different positions
                 for arg in parts[1:]:
                     # Add check if arg is a known mode type if possible
                     if arg in ['colour', 'high_res', 'low_res']: # Example modes
                          self.objects['mode'].add(arg)


        # Use inferred objects
        self.all_waypoints = self.objects.get('waypoint', set())
        self.all_rovers = self.objects.get('rover', set())
        self.all_cameras = self.objects.get('camera', set())
        self.all_objectives = self.objects.get('objective', set())
        self.all_stores = self.objects.get('store', set())

        # --- Static Information Extraction ---
        self.rover_equipment = defaultdict(set)
        self.rover_cameras = defaultdict(set) # Map: rover -> set of cameras
        self.camera_on_rover = {} # Map: camera -> rover
        self.camera_supports = defaultdict(set) # Map: camera -> set of modes
        self.camera_calibration_target = {} # Map: camera -> calibration objective
        self.objective_visible_from = defaultdict(set) # Map: objective -> set of waypoints
        self.lander_location = None
        self.waypoints_visible_from = defaultdict(set) # Map: wp -> set of visible wps
        self.rover_stores = {} # Map: rover -> store name
        self.can_traverse = defaultdict(set) # Map: rover -> set of (from_wp, to_wp) tuples
        self.calibration_target_visible_from = defaultdict(set) # Map: objective -> set of waypoints

        for fact in static_facts:
            parts = get_parts(fact)
            pred = parts[0]
            args = parts[1:]

            try:
                if pred == 'equipped_for_soil_analysis': self.rover_equipment[args[0]].add('soil')
                elif pred == 'equipped_for_rock_analysis': self.rover_equipment[args[0]].add('rock')
                elif pred == 'equipped_for_imaging': self.rover_equipment[args[0]].add('imaging')
                elif pred == 'on_board':
                    self.rover_cameras[args[1]].add(args[0])
                    self.camera_on_rover[args[0]] = args[1]
                elif pred == 'supports': self.camera_supports[args[0]].add(args[1])
                elif pred == 'calibration_target': self.camera_calibration_target[args[0]] = args[1]
                elif pred == 'visible_from':
                    self.objective_visible_from[args[0]].add(args[1])
                    # Check if this objective is a calibration target and update visibility
                    if args[0] in self.camera_calibration_target.values():
                         self.calibration_target_visible_from[args[0]].add(args[1])
                elif pred == 'at_lander': self.lander_location = args[1]
                elif pred == 'visible': self.waypoints_visible_from[args[0]].add(args[1])
                elif pred == 'store_of': self.rover_stores[args[1]] = args[0]
                elif pred == 'can_traverse': self.can_traverse[args[0]].add((args[1], args[2]))
            except IndexError:
                print(f"Warning: Skipping malformed static fact: {fact}")


        # --- Precompute APSP for each rover ---
        self.rover_apsp = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: float('inf'))))
        self.waypoints_visible_by_lander = set()
        if self.lander_location:
             # Check which waypoints can see the lander's waypoint
             for wp_from, visible_wps in self.waypoints_visible_from.items():
                 if self.lander_location in visible_wps:
                     self.waypoints_visible_by_lander.add(wp_from)

        for rover in self.all_rovers:
            # Build adjacency list for this rover's navigation graph
            adj = defaultdict(list)
            for wp1 in self.all_waypoints:
                 self.rover_apsp[rover][wp1][wp1] = 0
                 # Check outgoing traversable and visible edges
                 # An edge exists from wp1 to wp2 if rover can traverse AND wp1 can see wp2
                 for wp2 in self.waypoints_visible_from.get(wp1, set()):
                     if (wp1, wp2) in self.can_traverse.get(rover, set()):
                         adj[wp1].append(wp2)

            # Run BFS from each waypoint for this rover to compute shortest paths
            for start_node in self.all_waypoints:
                if not start_node: continue # Skip if waypoint name is empty/invalid
                queue = [(0, start_node)] # (distance, node)
                # No need for visited set if we check distance before adding
                # self.rover_apsp[rover][start_node][start_node] = 0 # Already set

                processed_nodes = {start_node} # Keep track of nodes added to queue to avoid cycles in BFS path length

                head = 0
                while head < len(queue):
                    dist, curr_node = queue[head]
                    head += 1

                    # Check neighbors
                    for neighbor in adj.get(curr_node, []):
                        # If we found a shorter path (should not happen in unweighted BFS if processed correctly)
                        # or if the neighbor hasn't been reached yet
                        if self.rover_apsp[rover][start_node][neighbor] > dist + 1:
                             self.rover_apsp[rover][start_node][neighbor] = dist + 1
                             # Add to queue only if not already processed in this BFS run from start_node
                             if neighbor not in processed_nodes:
                                 queue.append((dist + 1, neighbor))
                                 processed_nodes.add(neighbor)


        # Store goal predicates parsed into tuples for easy checking
        self.parsed_goals = set()
        for goal_fact in self.goals:
            try:
                self.parsed_goals.add(tuple(get_parts(goal_fact)))
            except:
                 print(f"Warning: Could not parse goal fact: {goal_fact}")


    def _get_min_dist(self, rover, start_wp, target_wps):
        """Finds minimum distance and the corresponding waypoint from start_wp to any waypoint in target_wps."""
        min_dist = float('inf')
        best_wp = None
        if not target_wps or not start_wp:
            return float('inf'), None # No target waypoints or invalid start

        for target_wp in target_wps:
            if not target_wp: continue # Skip invalid target
            try:
                dist = self.rover_apsp[rover][start_wp][target_wp]
                if dist < min_dist:
                    min_dist = dist
                    best_wp = target_wp
            except KeyError: # Handle cases where start_wp or target_wp might not be in the APSP keys
                continue
        return min_dist, best_wp

    def __call__(self, node):
        state = node.state
        # Check if goal is already reached
        if self.goals <= state:
            return 0

        # --- Parse Current State ---
        rover_locations = {}
        store_states = {} # map store -> 'empty' or 'full'
        calibrated_cameras = set() # set of (camera, rover) tuples
        have_soil = set() # set of (rover, waypoint) tuples
        have_rock = set() # set of (rover, waypoint) tuples
        have_image = set() # set of (rover, objective, mode) tuples
        current_facts_parsed = set() # Store parsed facts for quick lookup

        for fact in state:
            try:
                parts = get_parts(fact)
                pred = parts[0]
                args = parts[1:]
                current_facts_parsed.add(tuple(parts))

                if pred == 'at': rover_locations[args[0]] = args[1]
                elif pred == 'empty': store_states[args[0]] = 'empty'
                elif pred == 'full': store_states[args[0]] = 'full'
                elif pred == 'calibrated': calibrated_cameras.add((args[0], args[1])) # (camera, rover)
                elif pred == 'have_soil_analysis': have_soil.add((args[0], args[1])) # (rover, wp)
                elif pred == 'have_rock_analysis': have_rock.add((args[0], args[1])) # (rover, wp)
                elif pred == 'have_image': have_image.add(tuple(args)) # (rover, obj, mode)
            except IndexError:
                 print(f"Warning: Skipping malformed state fact: {fact}")


        total_heuristic_cost = 0
        infinity = float('inf')

        # --- Calculate Cost for Unsatisfied Goals ---
        for goal_tuple in self.parsed_goals:
            if goal_tuple in current_facts_parsed:
                continue # Goal already satisfied

            goal_pred = goal_tuple[0]
            goal_args = goal_tuple[1:]
            min_goal_cost = infinity

            # --- Case 1: communicated_soil_data(w) ---
            if goal_pred == 'communicated_soil_data':
                target_wp = goal_args[0]
                for rover in self.all_rovers:
                    if 'soil' not in self.rover_equipment.get(rover, set()):
                        continue # Rover not equipped

                    rover_loc = rover_locations.get(rover)
                    if not rover_loc: continue # Rover location unknown

                    cost_r = 0
                    rover_store = self.rover_stores.get(rover)
                    store_state = store_states.get(rover_store, 'empty') # Assume empty if not specified

                    # 1. Acquisition Cost
                    if (rover, target_wp) in have_soil:
                        cost_acquire = 0
                        loc_after_acquire = rover_loc
                    else:
                        # Need to sample
                        cost_drop = 0
                        if rover_store and store_state == 'full':
                            cost_drop = 1 # Drop action cost

                        cost_nav_to_sample, _ = self._get_min_dist(rover, rover_loc, {target_wp})
                        if cost_nav_to_sample == infinity: continue # Cannot reach sample location

                        cost_sample = 1 # Sample action cost
                        cost_acquire = cost_drop + cost_nav_to_sample + cost_sample
                        loc_after_acquire = target_wp # Rover is now at the sample location

                    # 2. Communication Cost
                    cost_nav_to_comm, _ = self._get_min_dist(rover, loc_after_acquire, self.waypoints_visible_by_lander)
                    if cost_nav_to_comm == infinity: continue # Cannot reach a communication spot

                    cost_comm_action = 1 # Communicate action cost
                    cost_r = cost_acquire + cost_nav_to_comm + cost_comm_action
                    min_goal_cost = min(min_goal_cost, cost_r)

            # --- Case 2: communicated_rock_data(w) ---
            elif goal_pred == 'communicated_rock_data':
                target_wp = goal_args[0]
                for rover in self.all_rovers:
                    if 'rock' not in self.rover_equipment.get(rover, set()):
                        continue

                    rover_loc = rover_locations.get(rover)
                    if not rover_loc: continue

                    cost_r = 0
                    rover_store = self.rover_stores.get(rover)
                    store_state = store_states.get(rover_store, 'empty')

                    # 1. Acquisition Cost
                    if (rover, target_wp) in have_rock:
                        cost_acquire = 0
                        loc_after_acquire = rover_loc
                    else:
                        cost_drop = 0
                        if rover_store and store_state == 'full':
                            cost_drop = 1

                        cost_nav_to_sample, _ = self._get_min_dist(rover, rover_loc, {target_wp})
                        if cost_nav_to_sample == infinity: continue

                        cost_sample = 1
                        cost_acquire = cost_drop + cost_nav_to_sample + cost_sample
                        loc_after_acquire = target_wp

                    # 2. Communication Cost
                    cost_nav_to_comm, _ = self._get_min_dist(rover, loc_after_acquire, self.waypoints_visible_by_lander)
                    if cost_nav_to_comm == infinity: continue

                    cost_comm_action = 1
                    cost_r = cost_acquire + cost_nav_to_comm + cost_comm_action
                    min_goal_cost = min(min_goal_cost, cost_r)

            # --- Case 3: communicated_image_data(o, m) ---
            elif goal_pred == 'communicated_image_data':
                target_obj, target_mode = goal_args[0], goal_args[1]
                for rover in self.all_rovers:
                    if 'imaging' not in self.rover_equipment.get(rover, set()):
                        continue

                    rover_loc = rover_locations.get(rover)
                    if not rover_loc: continue

                    # Find a suitable camera on this rover that supports the mode
                    suitable_camera = None
                    for cam in self.rover_cameras.get(rover, set()):
                        if target_mode in self.camera_supports.get(cam, set()):
                            suitable_camera = cam
                            break
                    if not suitable_camera: continue # No suitable camera on this rover for this mode

                    cost_r = 0

                    # 1. Acquisition Cost
                    if (rover, target_obj, target_mode) in have_image:
                        cost_acquire = 0
                        loc_after_acquire = rover_loc
                    else:
                        # Need to take image
                        cost_calibration_steps = 0
                        loc_after_calib = rover_loc # Location after calibration (or current if no calibration needed)

                        # a) Calibration if needed
                        if (suitable_camera, rover) not in calibrated_cameras:
                            calib_target = self.camera_calibration_target.get(suitable_camera)
                            if not calib_target: continue # Camera needs calibration but has no target defined

                            # Find waypoints from which the calibration target is visible
                            visible_calib_wps = self.calibration_target_visible_from.get(calib_target, set())
                            if not visible_calib_wps: continue # Calibration target not visible from anywhere

                            # Find closest calibration waypoint
                            cost_nav_to_calib, best_calib_wp = self._get_min_dist(rover, rover_loc, visible_calib_wps)
                            if cost_nav_to_calib == infinity: continue # Cannot reach any calibration spot

                            cost_calibrate_action = 1
                            cost_calibration_steps = cost_nav_to_calib + cost_calibrate_action
                            loc_after_calib = best_calib_wp # Rover is now at the calibration waypoint

                        # b) Navigate to imaging spot from location after calibration
                        visible_image_wps = self.objective_visible_from.get(target_obj, set())
                        if not visible_image_wps: continue # Target objective not visible from anywhere

                        # Find closest imaging waypoint from loc_after_calib
                        cost_nav_to_image, best_image_wp = self._get_min_dist(rover, loc_after_calib, visible_image_wps)
                        if cost_nav_to_image == infinity: continue # Cannot reach any imaging spot

                        cost_take_image_action = 1
                        cost_acquire = cost_calibration_steps + cost_nav_to_image + cost_take_image_action
                        loc_after_acquire = best_image_wp # Rover is now at the imaging waypoint

                    # 2. Communication Cost
                    cost_nav_to_comm, _ = self._get_min_dist(rover, loc_after_acquire, self.waypoints_visible_by_lander)
                    if cost_nav_to_comm == infinity: continue # Cannot reach a communication spot

                    cost_comm_action = 1
                    cost_r = cost_acquire + cost_nav_to_comm + cost_comm_action
                    min_goal_cost = min(min_goal_cost, cost_r)

            # --- Accumulate cost for this goal ---
            if min_goal_cost == infinity:
                 # If any goal is deemed unreachable by any capable rover from this state,
                 # this state is likely a dead end for achieving that goal.
                 # Return infinity to prune this branch in greedy search.
                 return infinity
            total_heuristic_cost += min_goal_cost

        return total_heuristic_cost

