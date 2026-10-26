import heapq
from collections import defaultdict
from fnmatch import fnmatch
# The base class Heuristic is expected to be in the heuristics package.
from heuristics.heuristic_base import Heuristic
import math # For infinity

# Helper function to parse facts: remove parentheses and split
def get_parts(fact):
    """Extracts predicate and arguments from a PDDL fact string."""
    return fact.strip()[1:-1].split()

# Helper function to match facts with potential wildcards
def match(fact_parts, *pattern):
    """
    Checks if the parts of a fact match a given pattern.
    - fact_parts: A list of strings (predicate and arguments).
    - pattern: A sequence of strings, potentially including '*' for wildcards.
    Returns True if it matches, False otherwise.
    """
    if len(fact_parts) != len(pattern):
        return False
    return all(fnmatch(part, pat) for part, pat in zip(fact_parts, pattern))


class RoversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the PDDL Rovers domain.

    # Summary
    This heuristic estimates the total number of actions required to achieve all goals
    from a given state. It calculates the estimated cost for each unsatisfied goal
    independently and sums these costs. The cost estimation for each goal considers
    the necessary steps: navigation, potential calibration, sampling or imaging,
    and communication. It aims to provide good guidance for a greedy best-first search
    by factoring in rover capabilities, locations, equipment status (store, calibration),
    and precomputed shortest navigation paths.

    # Assumptions
    - Each PDDL action has a uniform cost of 1.
    - The heuristic sums the estimated costs for achieving each goal individually.
      This ignores potential positive interactions (e.g., one trip achieving multiple
      goals) and negative interactions (resource contention), but aims for better
      guidance than simple goal count. It is not admissible.
    - Rovers must travel to specific waypoints for sampling, calibration, imaging,
      and communication based on visibility and task requirements.
    - Shortest path distances between waypoints for each rover are precomputed based
      on `can_traverse` and `visible` predicates defining the navigation graph.
    - Communication actions require the rover to be at a waypoint `x` and the lander
      at a waypoint `y` such that `(visible x y)` holds.
    - Taking an image requires the camera to be calibrated. If not, a calibration
      step (navigate + calibrate action) is added to the cost. Taking an image
      uncalibrates the camera.
    - Sampling requires an empty store. If a rover needs to sample but its store is
      full, an additional cost of 1 (for the `drop` action) is added. The heuristic
      doesn't model the navigation cost potentially associated with dropping.

    # Heuristic Initialization (`__init__`)
    - Parses the task's static facts to build internal data structures representing:
        - Rover equipment (soil, rock, imaging capabilities).
        - Camera properties (modes supported, calibration targets).
        - Mappings between rovers, stores, and cameras.
        - Visibility information between waypoints (`visible`) and for objectives/targets
          from waypoints (`visible_from`).
        - The lander's fixed location.
        - Sets of all relevant objects (waypoints, rovers, cameras, etc.).
    - Constructs a navigation graph for each rover. An edge exists from waypoint `w1`
      to `w2` for rover `r` if `(can_traverse r w1 w2)` and `(visible w1 w2)` are
      both true in the static facts.
    - Precomputes all-pairs shortest path distances for each rover on its specific
      navigation graph using Breadth-First Search (BFS). Distances are stored.
    - Identifies all potential communication waypoints (those visible from the
      lander's location, based on the `communicate_*` action preconditions).
    - Stores the set of goal predicates.

    # Step-By-Step Thinking for Computing Heuristic (`__call__`)
    1.  Parse the current state to determine dynamic information: rover locations,
        store states (empty/full), calibrated cameras, data currently held by rovers
        (soil, rock, images), and which goals are already achieved.
    2.  Initialize `total_cost = 0`.
    3.  Iterate through each goal predicate `g` defined in the task's goals.
    4.  If `g` is already satisfied in the current state, continue to the next goal.
    5.  Estimate the minimum cost `goal_cost` to achieve the unsatisfied goal `g`:
        a.  **If `g` is `(communicated_soil_data w)`:**
            i.   Check if any rover `r` currently holds `(have_soil_analysis r w)`.
            ii.  If yes (data held): Find the rover `r` among those holding the data
                 that minimizes the cost to reach the *closest* communication waypoint
                 (`comm_wp`) from its current location.
                 Cost = `min_dist(r, current_loc(r), comm_wp) + 1` (navigate + communicate).
            iii. If no (data not held) or to find a potentially better path: Find the best rover `r` (equipped for soil)
                 to perform the entire sequence: navigate to `w`, sample, navigate
                 to a communication waypoint, communicate.
                 - For each capable rover `r`:
                     - Cost = `dist(r, current_loc(r), w)` (navigate to sample)
                            + `1` (sample action)
                            + `1` (if store is full, for drop action)
                            + `dist(r, w, closest_comm_wp)` (navigate from `w` to communicate)
                            + `1` (communicate action).
                 - Choose the rover `r` that minimizes this total sequence cost.
            iv.  Set `goal_cost` to the minimum cost found across all valid options (holding data vs. sampling now).
        b.  **If `g` is `(communicated_rock_data w)`:** Follow logic similar to soil data (5a).
        c.  **If `g` is `(communicated_image_data o m)`:**
            i.   Check if any rover `r` currently holds `(have_image r o m)`.
            ii.  If yes (image held): Find the rover `r` among those holding the image
                 that minimizes the cost to reach the *closest* communication waypoint.
                 Cost = `min_dist(r, current_loc(r), comm_wp) + 1`.
            iii. If no (image not held) or to find a potentially better path: Find the best rover `r` (equipped for imaging,
                 with a suitable camera `i` supporting mode `m`) to perform the sequence:
                 potentially calibrate, navigate to image location, take image, navigate
                 to communication waypoint, communicate.
                 - For each capable rover `r` and suitable camera `i`:
                     - `current_step_cost = 0`
                     - `location_after_step = current_loc(r)`
                     - **Calibration Check:** If `(calibrated i r)` is not true:
                         - Find calibration target `t` for camera `i`.
                         - Find the closest waypoint `w_cal` (visible from `t`) reachable by `r`.
                         - `current_step_cost += dist(r, location_after_step, w_cal) + 1` (navigate + calibrate).
                         - `location_after_step = w_cal`.
                     - **Take Image:**
                         - Find the closest waypoint `p` (from which objective `o` is visible) reachable by `r` from `location_after_step`.
                         - `current_step_cost += dist(r, location_after_step, p) + 1` (navigate + take_image).
                         - `location_after_step = p`.
                     - **Communicate:**
                         - Find the closest communication waypoint `comm_wp` reachable by `r` from `location_after_step`.
                         - `current_step_cost += dist(r, location_after_step, comm_wp) + 1` (navigate + communicate).
                     - Keep track of the minimum `current_step_cost` found across all suitable rovers/cameras.
            iv.  Set `goal_cost` to the minimum cost found across all valid options (holding image vs. taking image now).
        d. If, for any step (finding closest waypoint for calibration, imaging, or communication),
           no reachable waypoint is found, that specific rover/camera combination is considered
           incapable of achieving the goal via that path. If no rover/path works, `goal_cost`
           remains infinity.
    6.  If `goal_cost` is finite, add it to `total_cost`. If `goal_cost` is infinite
        (meaning the heuristic estimates the goal is unreachable from the current state
        by any means considered), add a large penalty value (e.g., 1000) to `total_cost`.
    7.  After checking all goals, if `total_cost` is 0 but the current state is not
        actually a goal state (i.e., `task.goal_reached(state)` is false), return 1
        (or the number of unsatisfied goals) to ensure non-goal states have a non-zero
        heuristic value. Otherwise, return `total_cost`.
    """

    def __init__(self, task):
        super().__init__(task)
        self.goals = task.goals
        static_facts = task.static

        # --- Object Sets ---
        self.waypoints = set()
        self.rovers = set()
        self.cameras = set()
        self.objectives = set()
        self.modes = set()
        self.stores = set()
        self.lander = None # Assuming one lander based on examples

        # --- Static Fact Derived Data Structures ---
        self.rover_equipment = defaultdict(set) # rover -> {soil, rock, imaging}
        self.rover_stores = {} # rover -> store
        self.store_rover = {} # store -> rover
        self.rover_cameras = defaultdict(set) # rover -> {camera1, camera2}
        self.camera_supports = defaultdict(set) # camera -> {mode1, mode2}
        self.camera_calibration_target = {} # camera -> objective
        # objective -> {wp1, wp2} where obj is visible from wp
        self.objective_visibility = defaultdict(set)
        # objective -> {wp1, wp2} (subset of objective_visibility for calib targets)
        self.calibration_target_visibility = defaultdict(set)
        # wp_src -> {wp_dest} where wp_dest is visible from wp_src
        self.waypoint_visibility = defaultdict(set)
        self.lander_location = None
        # rover -> {wp_from: [wp_to1, wp_to2]} adjacency list
        self.rover_graphs = defaultdict(lambda: defaultdict(list))
        # rover -> {wp_from: {wp_to: dist}}
        self.shortest_paths = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: float('inf'))))

        # --- Parse Static Facts ---
        temp_can_traverse = defaultdict(list) # Store temporarily before checking visibility

        for fact in static_facts:
            parts = get_parts(fact)
            pred = parts[0]
            args = parts[1:]

            # Dynamically identify objects and types based on predicates
            if pred == 'at_lander':
                self.lander = args[0]
                self.lander_location = args[1]
                self.waypoints.add(args[1])
            elif pred == 'equipped_for_soil_analysis':
                self.rovers.add(args[0])
                self.rover_equipment[args[0]].add('soil')
            elif pred == 'equipped_for_rock_analysis':
                self.rovers.add(args[0])
                self.rover_equipment[args[0]].add('rock')
            elif pred == 'equipped_for_imaging':
                self.rovers.add(args[0])
                self.rover_equipment[args[0]].add('imaging')
            elif pred == 'store_of':
                self.stores.add(args[0])
                self.rovers.add(args[1])
                self.rover_stores[args[1]] = args[0]
                self.store_rover[args[0]] = args[1]
            elif pred == 'visible':
                self.waypoints.add(args[0])
                self.waypoints.add(args[1])
                self.waypoint_visibility[args[0]].add(args[1])
            elif pred == 'can_traverse':
                self.rovers.add(args[0])
                self.waypoints.add(args[1])
                self.waypoints.add(args[2])
                # Store (rover, w1, w2)
                temp_can_traverse[args[0]].append((args[1], args[2]))
            elif pred == 'calibration_target':
                self.cameras.add(args[0])
                self.objectives.add(args[1])
                self.camera_calibration_target[args[0]] = args[1]
            elif pred == 'on_board':
                self.cameras.add(args[0])
                self.rovers.add(args[1])
                self.rover_cameras[args[1]].add(args[0])
            elif pred == 'supports':
                self.cameras.add(args[0])
                self.modes.add(args[1])
                self.camera_supports[args[0]].add(args[1])
            elif pred == 'visible_from':
                self.objectives.add(args[0])
                self.waypoints.add(args[1])
                self.objective_visibility[args[0]].add(args[1])

        # Populate calibration target visibility using objective visibility
        for cam, target_obj in self.camera_calibration_target.items():
            if target_obj in self.objective_visibility:
                 self.calibration_target_visibility[target_obj].update(
                     self.objective_visibility[target_obj]
                 )

        # Build rover graphs using can_traverse AND visible
        # The navigate action requires (visible ?y ?z) for move from y to z.
        # Let's assume can_traverse(r, y, z) implies ability, and visible(y, z) enables it.
        for rover, traversals in temp_can_traverse.items():
            for w1, w2 in traversals:
                 # Check if w2 is visible from w1 according to static facts
                 if w2 in self.waypoint_visibility.get(w1, set()):
                     self.rover_graphs[rover][w1].append(w2)

        # --- Precompute Shortest Paths using BFS ---
        for rover in self.rovers:
            rover_graph = self.rover_graphs[rover]
            # Check all waypoints, even those not directly in graph keys/values initially
            for start_node in self.waypoints:
                # Check if start_node is relevant for this rover's graph
                # A node is relevant if it's a source or destination in the graph
                is_relevant_node = start_node in rover_graph or \
                                   any(start_node in destinations for destinations in rover_graph.values())
                if not is_relevant_node:
                     # If a waypoint is totally isolated for this rover, its distance to/from others is inf
                     continue

                # Initialize BFS from start_node
                self.shortest_paths[rover][start_node][start_node] = 0
                # Use deque for efficiency if available, otherwise list works
                queue = [(0, start_node)] # (distance, node)
                visited_dist = {start_node: 0} # Store distances during BFS

                processed_nodes = 0 # Safety break for large graphs / potential issues
                max_processed = len(self.waypoints) * 2 # Heuristic limit

                while queue and processed_nodes < max_processed:
                    processed_nodes += 1
                    dist, current_node = heapq.heappop(queue) # Use heapq for priority queue behavior (though BFS is unweighted)

                    # If we found a shorter path already, skip
                    if dist > visited_dist[current_node]:
                        continue

                    # Check neighbors in the specific rover's graph
                    if current_node in rover_graph:
                        for neighbor in rover_graph[current_node]:
                            new_dist = dist + 1
                            if new_dist < visited_dist.get(neighbor, float('inf')):
                                visited_dist[neighbor] = new_dist
                                self.shortest_paths[rover][start_node][neighbor] = new_dist
                                heapq.heappush(queue, (new_dist, neighbor))

        # --- Identify Communication Waypoints ---
        self.communication_waypoints = set()
        if self.lander_location:
            # Communicate action precondition is (visible ?x ?y) where rover is at x, lander at y.
            # So, we need waypoints 'x' such that the lander location 'y' is visible FROM 'x'.
            self.communication_waypoints = {
                wp_x for wp_x in self.waypoints
                if self.lander_location in self.waypoint_visibility.get(wp_x, set())
            }


    def _get_shortest_path(self, rover, start_wp, end_wp):
        """ Safely retrieves the precomputed shortest path distance. """
        # Return 0 if start and end are the same, even if not in graph (e.g., isolated waypoint)
        if start_wp == end_wp:
            return 0
        dist = self.shortest_paths.get(rover, {}).get(start_wp, {}).get(end_wp, float('inf'))
        return dist

    def _find_closest_waypoint(self, rover, current_wp, target_waypoints):
        """
        Finds the waypoint in target_waypoints that is closest to current_wp for the given rover.
        Returns the closest waypoint and the minimum distance.
        Returns (None, float('inf')) if no target waypoint is reachable or target_waypoints is empty.
        """
        min_dist = float('inf')
        closest_wp = None

        if not target_waypoints:
            return None, float('inf')

        # Ensure current_wp is valid before proceeding
        if current_wp not in self.waypoints:
             return None, float('inf') # Cannot calculate distance from invalid waypoint

        for target_wp in target_waypoints:
             # Ensure target_wp is valid
             if target_wp not in self.waypoints:
                 continue
             dist = self._get_shortest_path(rover, current_wp, target_wp)
             if dist < min_dist:
                 min_dist = dist
                 closest_wp = target_wp

        # If min_dist is still infinity, no target waypoint was reachable
        if min_dist == float('inf'):
            return None, float('inf')

        return closest_wp, min_dist


    def __call__(self, node):
        state = node.state
        total_cost = 0

        # --- Parse Current State ---
        rover_locations = {}
        store_states = {} # store -> 'empty' or 'full'
        calibrated_cameras = set() # (camera, rover) tuple
        held_soil_data = set() # (rover, waypoint) tuple
        held_rock_data = set() # (rover, waypoint) tuple
        held_image_data = set() # (rover, objective, mode) tuple
        achieved_goals = set() # Predicates in state that are also goals

        for fact in state:
            # Use try-except for robustness against malformed facts
            try:
                parts = get_parts(fact)
                if not parts: continue # Skip empty facts
                pred = parts[0]
                args = parts[1:]

                if pred == 'at' and args and args[0] in self.rovers: # Check if first arg is a rover
                    rover_locations[args[0]] = args[1]
                elif pred == 'empty' and args:
                    store_states[args[0]] = 'empty'
                elif pred == 'full' and args:
                    store_states[args[0]] = 'full'
                elif pred == 'calibrated' and len(args) == 2:
                    calibrated_cameras.add((args[0], args[1])) # (camera, rover)
                elif pred == 'have_soil_analysis' and len(args) == 2:
                    held_soil_data.add((args[0], args[1]))
                elif pred == 'have_rock_analysis' and len(args) == 2:
                    held_rock_data.add((args[0], args[1]))
                elif pred == 'have_image' and len(args) == 3:
                    held_image_data.add((args[0], args[1], args[2]))

                if fact in self.goals:
                    achieved_goals.add(fact)
            except Exception:
                # Handle potential errors during parsing if needed, e.g., log a warning
                # print(f"Warning: Could not parse fact: {fact}")
                pass


        # --- Calculate Cost for Unsatisfied Goals ---
        num_unsatisfied = 0
        for goal in self.goals:
            if goal in achieved_goals:
                continue

            num_unsatisfied += 1
            goal_parts = get_parts(goal)
            goal_pred = goal_parts[0]
            goal_args = goal_parts[1:]
            goal_cost = float('inf') # Min cost estimated for THIS goal

            # --- Case 1: communicated_soil_data ---
            if goal_pred == 'communicated_soil_data' and goal_args:
                wp_soil = goal_args[0]
                min_rover_cost = float('inf')

                # Option A: Data already held by a rover
                rovers_with_data = {r for r, w in held_soil_data if w == wp_soil}
                for rover in rovers_with_data:
                    if rover not in rover_locations: continue
                    current_loc = rover_locations[rover]
                    # Find closest comm waypoint reachable from current location
                    closest_comm_wp, dist_to_comm = self._find_closest_waypoint(rover, current_loc, self.communication_waypoints)
                    if closest_comm_wp is not None:
                        cost = dist_to_comm + 1 # navigate + communicate
                        min_rover_cost = min(min_rover_cost, cost)

                # Option B: Need to sample and communicate
                possible_rovers = {r for r in self.rovers if 'soil' in self.rover_equipment.get(r, set())}
                for rover in possible_rovers:
                    if rover not in rover_locations: continue
                    current_loc = rover_locations[rover]
                    store = self.rover_stores.get(rover)
                    # Check store state, default to 'empty' if not specified? Assume empty if not full.
                    store_is_full = store_states.get(store) == 'full' if store else False

                    # Cost to get to sample location
                    dist_to_sample = self._get_shortest_path(rover, current_loc, wp_soil)
                    if dist_to_sample == float('inf'): continue # Cannot reach sample location

                    cost_sequence = dist_to_sample + 1 # navigate + sample
                    if store_is_full:
                        cost_sequence += 1 # drop action cost

                    # Cost to get from sample location to communication
                    closest_comm_wp, dist_sample_to_comm = self._find_closest_waypoint(rover, wp_soil, self.communication_waypoints)
                    if closest_comm_wp is not None:
                        cost_sequence += dist_sample_to_comm + 1 # navigate + communicate
                        min_rover_cost = min(min_rover_cost, cost_sequence)
                    # else: cannot communicate from sample location - this rover path is invalid for Option B

                goal_cost = min_rover_cost


            # --- Case 2: communicated_rock_data ---
            elif goal_pred == 'communicated_rock_data' and goal_args:
                wp_rock = goal_args[0]
                min_rover_cost = float('inf')

                # Option A: Data already held
                rovers_with_data = {r for r, w in held_rock_data if w == wp_rock}
                for rover in rovers_with_data:
                    if rover not in rover_locations: continue
                    current_loc = rover_locations[rover]
                    closest_comm_wp, dist_to_comm = self._find_closest_waypoint(rover, current_loc, self.communication_waypoints)
                    if closest_comm_wp is not None:
                        cost = dist_to_comm + 1
                        min_rover_cost = min(min_rover_cost, cost)

                # Option B: Need to sample and communicate
                possible_rovers = {r for r in self.rovers if 'rock' in self.rover_equipment.get(r, set())}
                for rover in possible_rovers:
                    if rover not in rover_locations: continue
                    current_loc = rover_locations[rover]
                    store = self.rover_stores.get(rover)
                    store_is_full = store_states.get(store) == 'full' if store else False

                    dist_to_sample = self._get_shortest_path(rover, current_loc, wp_rock)
                    if dist_to_sample == float('inf'): continue

                    cost_sequence = dist_to_sample + 1 # navigate + sample
                    if store_is_full:
                        cost_sequence += 1 # drop

                    closest_comm_wp, dist_sample_to_comm = self._find_closest_waypoint(rover, wp_rock, self.communication_waypoints)
                    if closest_comm_wp is not None:
                        cost_sequence += dist_sample_to_comm + 1 # navigate + communicate
                        min_rover_cost = min(min_rover_cost, cost_sequence)

                goal_cost = min_rover_cost


            # --- Case 3: communicated_image_data ---
            elif goal_pred == 'communicated_image_data' and len(goal_args) == 2:
                obj, mode = goal_args
                min_rover_cost = float('inf')

                # Option A: Image already held
                rovers_with_data = {r for r, o, m in held_image_data if o == obj and m == mode}
                for rover in rovers_with_data:
                    if rover not in rover_locations: continue
                    current_loc = rover_locations[rover]
                    closest_comm_wp, dist_to_comm = self._find_closest_waypoint(rover, current_loc, self.communication_waypoints)
                    if closest_comm_wp is not None:
                        cost = dist_to_comm + 1
                        min_rover_cost = min(min_rover_cost, cost)

                # Option B: Need to take image and communicate
                possible_rovers = {r for r in self.rovers if 'imaging' in self.rover_equipment.get(r, set())}
                for rover in possible_rovers:
                    if rover not in rover_locations: continue
                    initial_rover_loc = rover_locations[rover] # Location at the start of considering this rover

                    # Find cameras on this rover supporting the mode
                    suitable_cameras = {
                        cam for cam in self.rover_cameras.get(rover, set())
                        if mode in self.camera_supports.get(cam, set())
                    }

                    for camera in suitable_cameras:
                        cost_sequence = 0
                        loc_after_calib = initial_rover_loc # Location after potential calibration step

                        # Step B.1: Calibration (if needed)
                        if (camera, rover) not in calibrated_cameras:
                            calib_target = self.camera_calibration_target.get(camera)
                            if not calib_target: continue # Camera has no calibration target defined

                            visible_calib_wps = self.calibration_target_visibility.get(calib_target, set())
                            if not visible_calib_wps: continue # Target not visible anywhere

                            # Find closest calibration spot from initial location
                            closest_calib_wp, dist_to_calib = self._find_closest_waypoint(rover, initial_rover_loc, visible_calib_wps)

                            if closest_calib_wp is None: continue # Cannot reach calibration spot

                            cost_sequence += dist_to_calib + 1 # navigate + calibrate
                            loc_after_calib = closest_calib_wp # Update location for next step
                        # else: already calibrated, loc_after_calib remains initial_rover_loc

                        # Step B.2: Take Image
                        visible_image_wps = self.objective_visibility.get(obj, set())
                        if not visible_image_wps: continue # Objective not visible anywhere

                        # Find closest imaging spot from location *after calibration*
                        closest_image_wp, dist_to_image = self._find_closest_waypoint(rover, loc_after_calib, visible_image_wps)

                        if closest_image_wp is None: continue # Cannot reach imaging spot

                        cost_sequence += dist_to_image + 1 # navigate + take_image
                        loc_after_image = closest_image_wp # Location after taking image

                        # Step B.3: Communicate
                        # Find closest comm spot from location *after taking image*
                        closest_comm_wp, dist_to_comm = self._find_closest_waypoint(rover, loc_after_image, self.communication_waypoints)

                        if closest_comm_wp is None: continue # Cannot reach communication spot

                        cost_sequence += dist_to_comm + 1 # navigate + communicate

                        min_rover_cost = min(min_rover_cost, cost_sequence) # Update minimum cost found

                goal_cost = min_rover_cost

            # --- Accumulate cost for this goal ---
            if goal_cost != float('inf'):
                total_cost += goal_cost
            else:
                # If heuristic estimates goal is unreachable, add a large penalty.
                # Using a value larger than any realistic plan length.
                total_cost += 1000

        # --- Final Adjustments ---
        # Check if the state actually satisfies all goals
        is_goal_state = self.goals <= state if self.goals else True # Handle case with no goals

        if is_goal_state:
            return 0
        elif total_cost == 0:
            # If calculated cost is 0 but it's not a goal state, return a small positive value.
            # Return number of unsatisfied goals found earlier.
            return num_unsatisfied if num_unsatisfied > 0 else 1
        else:
            # Ensure the cost is an integer if it's not infinity
            return int(total_cost) if total_cost != float('inf') else 1000 * len(self.goals) # Return a very large number if inf

