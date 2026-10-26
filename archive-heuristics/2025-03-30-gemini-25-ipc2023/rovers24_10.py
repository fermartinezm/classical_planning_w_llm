import heapq
from collections import deque
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import itertools
import math # Import math for rounding

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact string by removing parentheses and splitting."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern. Wildcards (*) are allowed.

    - `fact`: The complete fact as a string, e.g., "(at rover1 waypoint1)".
    - `args`: The expected pattern elements (e.g., "at", "rover*", "*").
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure the number of parts matches the pattern length
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class RoversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the PDDL Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal state
    by summing the estimated costs for achieving each individual unsatisfied goal predicate.
    It considers navigation costs (precomputed shortest paths), sampling, calibration,
    imaging, and communication actions. It tries to assign each goal to the rover
    that can achieve it with the minimum estimated cost from the current state.

    # Assumptions
    - The heuristic assumes that actions required for one goal do not significantly
      interfere with or contribute to other goals (potential overestimation or
      underestimation, but simplifies calculation). This is a common relaxation.
    - Navigation costs are based on the shortest path considering rover-specific
      traversal capabilities (`can_traverse`) and waypoint visibility (`visible`)
      as required by the `navigate` action.
    - Rovers can drop samples anywhere when their store is full before sampling,
      costing 1 action (`drop`).
    - Calibration is required before taking an image unless the camera is already calibrated.
      Taking an image makes the camera uncalibrated. The cost calculation assumes
      each imaging goal might require its own calibration sequence if the camera
      is not already calibrated, potentially overestimating if multiple images
      are taken sequentially by the same rover/camera using the same calibration.
    - The heuristic assumes the problem instance is solvable and static facts correctly
      define capabilities and the environment.

    # Heuristic Initialization
    - Extracts static information from `task.static` and `task.initial_state`:
        - Rover types, waypoint types, camera types, objectives, modes, stores, lander.
        - Rover equipment (soil, rock, imaging capabilities).
        - Rover-store mapping.
        - Camera-rover mapping (`on_board`).
        - Camera capabilities (`supports`, `calibration_target`).
        - Visibility between waypoints (`visible`).
        - Rover-specific traversal permissions (`can_traverse`).
        - Objective visibility from waypoints (`visible_from`).
        - Lander location (`at_lander`).
        - Initial sample locations (`at_soil_sample`, `at_rock_sample`).
    - Precomputes all-pairs shortest paths for each rover between all waypoints using BFS.
      The graph edges for navigation from waypoint `Y` to `Z` for rover `X` exist only if
      `(can_traverse X Y Z)` and `(visible Y Z)` are both true. Distances are stored.
    - Identifies waypoints suitable for communication (those visible from the lander's location).

    # Step-By-Step Thinking for Computing Heuristic
    1.  **Identify Unsatisfied Goals**: Compare the current state `node.state` with the task goals `self.goals`.
    2.  **Pre-parse State**: Extract current rover locations, store states (empty/full), and camera calibration states for efficient lookup. Also identify which data/images have already been collected (`have_...`) and which samples are currently available (`at_..._sample`).
    3.  **Iterate Through Unsatisfied Goals**: For each goal predicate `g` not met in the current state:
        a.  **Determine Goal Type**: Check if `g` is `communicated_soil_data`, `communicated_rock_data`, or `communicated_image_data`.
        b.  **Estimate Minimum Cost for Goal `g`**:
            i.   **Data Goals (`communicated_soil_data(w)` or `communicated_rock_data(w)`)**:
                 - **Check if data already held**: If `(have_soil/rock_analysis r w)` is true for some rover `r`:
                     - Find rover `r`'s current location `curr_w`.
                     - Find the closest communication waypoint `comm_w` reachable by `r` from `curr_w`.
                     - Cost = `distance(r, curr_w, comm_w) + 1` (for `communicate_...`).
                     - Take the minimum cost over all rovers `r` holding the data.
                 - **Check if sample exists**: If `(at_soil/rock_sample w)` is true:
                     - Find all rovers `r` equipped for this analysis type.
                     - For each capable rover `r`:
                         - Get current location `curr_w` and store state.
                         - Calculate cost to sample: `distance(r, curr_w, w) + (1 if store full else 0) + 1` (navigate + drop? + sample).
                         - Calculate cost to communicate from sample location: `distance(r, w, comm_w) + 1` (navigate + communicate), where `comm_w` is the closest communication point from `w`.
                         - Total cost for rover `r` = cost to sample + cost to communicate.
                     - Take the minimum total cost over all capable rovers.
                 - If neither data is held nor sample exists, the goal is currently unachievable (cost = infinity, handled by large penalty).
            ii.  **Image Goals (`communicated_image_data(o, m)`)**:
                 - **Check if image already held**: If `(have_image r o m)` is true for some rover `r`:
                     - Calculate cost similarly to held data: `distance(r, curr_w, comm_w) + 1`.
                     - Take the minimum cost over all rovers `r` holding the image.
                 - **If image needs taking**:
                     - Find all rovers `r` equipped for imaging with a camera `c` supporting mode `m`.
                     - For each capable rover `r` and its suitable camera `c`:
                         - Get current location `curr_w`. Check if `(calibrated c r)` is true.
                         - **If calibrated**:
                             - Find closest waypoint `wp_image` visible to objective `o` reachable from `curr_w`.
                             - Cost to acquire = `distance(r, curr_w, wp_image) + 1` (navigate + take_image).
                             - `last_wp = wp_image`.
                         - **If not calibrated**:
                             - Find calibration target `t` for camera `c`. Find waypoints `calib_waypoints` where `t` is visible.
                             - Find closest `wp_calib` in `calib_waypoints` reachable from `curr_w`.
                             - Cost to calibrate = `distance(r, curr_w, wp_calib) + 1`.
                             - Find closest `wp_image` visible to `o` reachable from `wp_calib`.
                             - Cost to acquire = `cost_calibrate + distance(r, wp_calib, wp_image) + 1` (calibrate + navigate + take_image).
                             - `last_wp = wp_image`.
                         - Calculate cost to communicate: `distance(r, last_wp, comm_w) + 1`, where `comm_w` is the closest communication point from `last_wp`.
                         - Total cost for rover `r` with camera `c` = cost to acquire + cost to communicate.
                     - Take the minimum total cost over all capable rovers and their suitable cameras.
        c.  **Add Minimum Cost**: Add the minimum estimated cost found for achieving goal `g` to the total heuristic value. If a goal's minimum cost remains infinity, add a large penalty (e.g., 1000) to signify potential unreachability or high difficulty.
    4.  **Return Total Cost**: The sum of minimum costs for all unsatisfied goals. If the state is a goal state, the cost is 0. If the calculated sum is 0 but the state is not a goal state (e.g., due to estimation inaccuracies), return 1 to ensure non-zero cost for non-goal states.
    """

    def __init__(self, task):
        super().__init__(task)
        self.goals = task.goals
        static_facts = task.static

        # --- Extract Static Information ---
        self.rovers = set()
        self.waypoints = set()
        self.cameras = set()
        self.objectives = set()
        self.modes = set()
        self.stores = set()
        self.lander = None
        self.lander_location = None

        # Use dictionaries for efficient lookups
        self.rover_equipment = {} # rover -> set(capabilities: 'soil', 'rock', 'imaging')
        self.rover_store = {} # rover -> store
        self.store_rover = {} # store -> rover
        self.rover_cameras = {} # rover -> set(camera)
        self.camera_supports = {} # camera -> set(mode)
        self.camera_calibration_target = {} # camera -> objective
        self.objective_visible_from = {} # objective -> set(waypoint)
        self.calibration_target_visible_from = {} # objective (used as target) -> set(waypoint)
        self.visibility = {} # waypoint -> set(visible waypoint)
        self.can_traverse = {} # rover -> dict(from_wp -> set(to_wp))

        # Parse static facts and infer types/relationships
        all_objects_in_static = set()
        for fact in static_facts:
             parts = get_parts(fact)
             all_objects_in_static.update(parts[1:])
             pred = parts[0]
             args = parts[1:]

             # Infer types and relationships from predicates
             if pred == 'at_lander':
                 self.lander = args[0]
                 self.lander_location = args[1]
                 self.waypoints.add(args[1])
             elif pred == 'equipped_for_soil_analysis':
                 r = args[0]; self.rovers.add(r)
                 self.rover_equipment.setdefault(r, set()).add('soil')
             elif pred == 'equipped_for_rock_analysis':
                 r = args[0]; self.rovers.add(r)
                 self.rover_equipment.setdefault(r, set()).add('rock')
             elif pred == 'equipped_for_imaging':
                 r = args[0]; self.rovers.add(r)
                 self.rover_equipment.setdefault(r, set()).add('imaging')
             elif pred == 'store_of':
                 s, r = args; self.stores.add(s); self.rovers.add(r)
                 self.rover_store[r] = s
                 self.store_rover[s] = r
             elif pred == 'on_board':
                 c, r = args; self.cameras.add(c); self.rovers.add(r)
                 self.rover_cameras.setdefault(r, set()).add(c)
             elif pred == 'supports':
                 c, m = args; self.cameras.add(c); self.modes.add(m)
                 self.camera_supports.setdefault(c, set()).add(m)
             elif pred == 'calibration_target':
                 c, o = args; self.cameras.add(c); self.objectives.add(o)
                 self.camera_calibration_target[c] = o
             elif pred == 'visible_from':
                 o, w = args; self.objectives.add(o); self.waypoints.add(w)
                 self.objective_visible_from.setdefault(o, set()).add(w)
             elif pred == 'visible':
                 w1, w2 = args; self.waypoints.add(w1); self.waypoints.add(w2)
                 self.visibility.setdefault(w1, set()).add(w2)
             elif pred == 'can_traverse':
                 r, w1, w2 = args; self.rovers.add(r); self.waypoints.add(w1); self.waypoints.add(w2)
                 self.can_traverse.setdefault(r, {}).setdefault(w1, set()).add(w2)

        # Update calibration target visibility based on objective visibility
        for cam, target_obj in self.camera_calibration_target.items():
            if target_obj in self.objective_visible_from:
                self.calibration_target_visible_from[target_obj] = self.objective_visible_from[target_obj]

        # Ensure all waypoints mentioned in traversal/visibility are captured
        for r_data in self.can_traverse.values():
            for w1, w2_set in r_data.items():
                self.waypoints.add(w1)
                self.waypoints.update(w2_set)
        for w1, w2_set in self.visibility.items():
            self.waypoints.add(w1)
            self.waypoints.update(w2_set)

        # Find lander location from initial state if not in static facts
        if not self.lander_location:
             for fact in task.initial_state:
                 if match(fact, "at_lander", "*", "*"):
                     self.lander = get_parts(fact)[1]
                     self.lander_location = get_parts(fact)[2]
                     self.waypoints.add(self.lander_location)
                     break
        assert self.lander_location is not None, "Lander location could not be determined"

        # --- Precompute Shortest Paths using BFS ---
        # Communication waypoints are those from which the lander location is visible
        self.comm_waypoints = {w for w, visible_wps in self.visibility.items() if self.lander_location in visible_wps}
        # Ensure comm waypoints are valid waypoints
        self.comm_waypoints = {wp for wp in self.comm_waypoints if wp in self.waypoints}

        self.distances = {} # rover -> from_wp -> to_wp -> distance
        for r in self.rovers:
            self.distances[r] = {}
            # Build rover-specific navigation graph considering 'can_traverse' and 'visible'
            rover_graph = {wp: set() for wp in self.waypoints}
            if r in self.can_traverse:
                for w1, destinations in self.can_traverse[r].items():
                    # Check visibility constraint for navigate action (visible ?y ?z)
                    if w1 in self.visibility:
                        visible_from_w1 = self.visibility[w1]
                        for w2 in destinations:
                            if w2 in visible_from_w1:
                                rover_graph[w1].add(w2)

            # Run BFS from each waypoint for this rover
            for start_node in self.waypoints:
                self.distances[r][start_node] = {wp: float('inf') for wp in self.waypoints}
                if start_node not in rover_graph: continue # Should be present if waypoint exists

                self.distances[r][start_node][start_node] = 0
                queue = deque([start_node])
                visited = {start_node} # Keep track of visited nodes per BFS run

                while queue:
                    current_wp = queue.popleft()
                    current_dist = self.distances[r][start_node][current_wp]

                    # Explore neighbors based on the rover-specific graph
                    for neighbor_wp in rover_graph.get(current_wp, set()):
                        if neighbor_wp not in visited:
                            visited.add(neighbor_wp)
                            self.distances[r][start_node][neighbor_wp] = current_dist + 1
                            queue.append(neighbor_wp)

    def _get_rover_location(self, rover, state_facts):
        """Find the current waypoint location of a rover from parsed state facts."""
        return state_facts.get("at", {}).get(rover)

    def _get_rover_store_state(self, rover, state_facts):
        """Find if the rover's store is empty or full from parsed state facts."""
        store = self.rover_store.get(rover)
        if not store: return None
        if store in state_facts.get("empty", set()):
            return "empty"
        if store in state_facts.get("full", set()):
            return "full"
        return None # Should be either empty or full

    def _get_camera_calibration_state(self, camera, rover, state_facts):
        """Check if a camera is calibrated from parsed state facts."""
        return (camera, rover) in state_facts.get("calibrated", set())

    def _find_closest_waypoint(self, rover, current_wp, target_waypoints):
        """Find the closest waypoint in target_waypoints from current_wp for a rover."""
        min_dist = float('inf')
        closest_wp = None

        if not target_waypoints or current_wp not in self.distances[rover]:
             return None, float('inf')

        # Ensure target_waypoints is iterable
        if not isinstance(target_waypoints, (set, list, tuple)):
            target_waypoints = {target_waypoints} # Make it a set if single item

        for target_wp in target_waypoints:
             # Check if target_wp is a valid key in the precomputed distances
             if target_wp in self.distances[rover][current_wp]:
                 dist = self.distances[rover][current_wp][target_wp]
                 if dist < min_dist:
                     min_dist = dist
                     closest_wp = target_wp
             # else: print(f"Warning: Target waypoint {target_wp} not found in distance map for rover {rover} from {current_wp}")

        return closest_wp, min_dist

    def __call__(self, node):
        state = node.state
        if self.task.goal_reached(state):
            return 0

        total_cost = 0
        unsatisfied_goals = self.goals - state

        # --- Pre-parse state for faster lookups ---
        state_facts = {
            "at": {}, "empty": set(), "full": set(), "calibrated": set(),
            "have_soil": {}, "have_rock": {}, "have_image": {},
            "at_soil_sample": set(), "at_rock_sample": set()
        }
        for fact in state:
            parts = get_parts(fact)
            pred = parts[0]
            args = parts[1:]
            if pred == 'at' and len(args) == 2: state_facts["at"][args[0]] = args[1] # rover -> waypoint
            elif pred == 'empty' and len(args) == 1: state_facts["empty"].add(args[0]) # store
            elif pred == 'full' and len(args) == 1: state_facts["full"].add(args[0]) # store
            elif pred == 'calibrated' and len(args) == 2: state_facts["calibrated"].add(tuple(args)) # (camera, rover)
            elif pred == 'have_soil_analysis' and len(args) == 2: state_facts["have_soil"].setdefault(args[1], set()).add(args[0]) # waypoint -> set(rover)
            elif pred == 'have_rock_analysis' and len(args) == 2: state_facts["have_rock"].setdefault(args[1], set()).add(args[0]) # waypoint -> set(rover)
            elif pred == 'have_image' and len(args) == 3: state_facts["have_image"].setdefault((args[1], args[2]), set()).add(args[0]) # (obj, mode) -> set(rover)
            elif pred == 'at_soil_sample' and len(args) == 1: state_facts["at_soil_sample"].add(args[0]) # waypoint
            elif pred == 'at_rock_sample' and len(args) == 1: state_facts["at_rock_sample"].add(args[0]) # waypoint

        # --- Estimate cost for each unsatisfied goal ---
        for goal in unsatisfied_goals:
            parts = get_parts(goal)
            pred = parts[0]
            args = parts[1:]
            min_goal_cost = float('inf')

            # --- Communicated Soil/Rock Data Goal ---
            if pred == 'communicated_soil_data' or pred == 'communicated_rock_data':
                w = args[0]
                is_soil = (pred == 'communicated_soil_data')
                analysis_type = 'soil' if is_soil else 'rock'
                have_analysis_dict = state_facts["have_soil"] if is_soil else state_facts["have_rock"]
                at_sample_set = state_facts["at_soil_sample"] if is_soil else state_facts["at_rock_sample"]

                # Case 1: A rover already has the analysis
                if w in have_analysis_dict:
                    for r in have_analysis_dict[w]:
                        curr_w = self._get_rover_location(r, state_facts)
                        if curr_w is None: continue
                        comm_wp, dist_to_comm = self._find_closest_waypoint(r, curr_w, self.comm_waypoints)
                        if comm_wp is not None:
                            cost = dist_to_comm + 1 # navigate + communicate
                            min_goal_cost = min(min_goal_cost, cost)

                # Case 2: Sample exists at waypoint w, needs collection and communication
                elif w in at_sample_set:
                    possible_rovers = [r for r in self.rovers if analysis_type in self.rover_equipment.get(r, set())]
                    for r in possible_rovers:
                        curr_w = self._get_rover_location(r, state_facts)
                        if curr_w is None: continue
                        store_state = self._get_rover_store_state(r, state_facts)
                        drop_cost = 1 if store_state == 'full' else 0

                        dist_to_sample = self.distances[r][curr_w].get(w, float('inf'))
                        if dist_to_sample == float('inf'): continue # Cannot reach sample

                        cost_to_acquire = dist_to_sample + drop_cost + 1 # navigate + drop? + sample

                        comm_wp, dist_sample_to_comm = self._find_closest_waypoint(r, w, self.comm_waypoints)
                        if comm_wp is None: continue # Cannot reach comm point from sample location

                        cost_to_comm = dist_sample_to_comm + 1 # navigate + communicate
                        min_goal_cost = min(min_goal_cost, cost_to_acquire + cost_to_comm)

            # --- Communicated Image Data Goal ---
            elif pred == 'communicated_image_data':
                o, m = args
                goal_key = (o, m)

                # Case 1: A rover already has the image
                if goal_key in state_facts["have_image"]:
                    for r in state_facts["have_image"][goal_key]:
                        curr_w = self._get_rover_location(r, state_facts)
                        if curr_w is None: continue
                        comm_wp, dist_to_comm = self._find_closest_waypoint(r, curr_w, self.comm_waypoints)
                        if comm_wp is not None:
                            cost = dist_to_comm + 1 # navigate + communicate
                            min_goal_cost = min(min_goal_cost, cost)

                # Case 2: Image needs to be taken
                else:
                    possible_rovers = [
                        r for r in self.rovers
                        if 'imaging' in self.rover_equipment.get(r, set())
                        and r in self.rover_cameras # Rover has cameras
                        and any(m in self.camera_supports.get(c, set()) for c in self.rover_cameras[r]) # A camera supports the mode
                    ]

                    for r in possible_rovers:
                        curr_w = self._get_rover_location(r, state_facts)
                        if curr_w is None: continue

                        # Find suitable cameras on this rover that support mode m
                        suitable_cameras = [c for c in self.rover_cameras.get(r, set()) if m in self.camera_supports.get(c, set())]

                        min_cost_for_rover = float('inf')
                        for c in suitable_cameras:
                            is_calibrated = self._get_camera_calibration_state(c, r, state_facts)
                            calib_target = self.camera_calibration_target.get(c)
                            # Use .get(calib_target, set()) for safety if target has no visible points
                            calib_waypoints = self.calibration_target_visible_from.get(calib_target, set())
                            image_waypoints = self.objective_visible_from.get(o, set())

                            if not image_waypoints: continue # Cannot image this objective from anywhere

                            cost_acquire_image = 0
                            last_wp_before_comm = None # Waypoint where rover is after taking image

                            if is_calibrated:
                                wp_image, dist_to_image = self._find_closest_waypoint(r, curr_w, image_waypoints)
                                if wp_image is None: continue # Cannot reach any imaging spot

                                cost_acquire_image = dist_to_image + 1 # navigate + take_image
                                last_wp_before_comm = wp_image
                            else:
                                # Needs calibration first
                                if not calib_target or not calib_waypoints: continue # Cannot calibrate this camera

                                wp_calib, dist_to_calib = self._find_closest_waypoint(r, curr_w, calib_waypoints)
                                if wp_calib is None: continue # Cannot reach calibration spot

                                cost_calibrate = dist_to_calib + 1 # navigate + calibrate

                                wp_image, dist_calib_to_image = self._find_closest_waypoint(r, wp_calib, image_waypoints)
                                if wp_image is None: continue # Cannot reach imaging spot from calib spot

                                cost_acquire_image = cost_calibrate + dist_calib_to_image + 1 # ... + navigate + take_image
                                last_wp_before_comm = wp_image

                            # Calculate cost to communicate from the location after taking the image
                            if last_wp_before_comm is None: continue # Should have failed earlier if None

                            comm_wp, dist_image_to_comm = self._find_closest_waypoint(r, last_wp_before_comm, self.comm_waypoints)
                            if comm_wp is None: continue # Cannot reach comm point from imaging spot

                            cost_to_comm = dist_image_to_comm + 1 # navigate + communicate
                            total_camera_cost = cost_acquire_image + cost_to_comm
                            min_cost_for_rover = min(min_cost_for_rover, total_camera_cost)

                        min_goal_cost = min(min_goal_cost, min_cost_for_rover)

            # Add the minimum cost found for this goal
            if min_goal_cost == float('inf'):
                 # If a goal seems impossible from this state, assign a large penalty.
                 # This helps guide the search away from states where goals are hard/impossible.
                 total_cost += 1000
            else:
                 total_cost += min_goal_cost

        # Ensure heuristic is 0 iff state is goal. If cost is 0 but not goal, return 1.
        if total_cost == 0 and unsatisfied_goals:
            return 1
        # Return integer cost
        return int(math.ceil(total_cost)) # Use ceil to avoid rounding down to 0 for small costs
