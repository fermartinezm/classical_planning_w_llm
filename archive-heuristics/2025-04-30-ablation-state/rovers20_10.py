from fnmatch import fnmatch
from collections import deque, defaultdict
# Assuming heuristics.heuristic_base exists and provides a Heuristic base class
# from heuristics.heuristic_base import Heuristic

# Define the base class if it's not provided externally for standalone testing
# In the actual planning environment, the import should be used.
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    class Heuristic:
        def __init__(self, task):
            pass
        def __call__(self, node):
            raise NotImplementedError


# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty facts or malformed strings gracefully
    if not fact or not isinstance(fact, str) or len(fact) < 2 or fact[0] != '(' or fact[-1] != ')':
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

def bfs(graph, start_node):
    """Computes shortest path distances from start_node in an unweighted graph."""
    distances = {node: float('inf') for node in graph}
    if start_node not in graph:
         return distances # Start node not in graph or graph is empty
    distances[start_node] = 0
    queue = deque([start_node])

    while queue:
        current_node = queue.popleft()

        # Check if current_node is a valid key in the graph
        if current_node not in graph:
             continue # Should not happen if start_node was in graph and graph structure is consistent

        # Use graph.get() with a default empty set to handle nodes with no outgoing edges
        for neighbor in graph.get(current_node, set()):
            if distances[neighbor] == float('inf'):
                distances[neighbor] = distances[current_node] + 1
                queue.append(neighbor)
    return distances

# The main heuristic class
class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing the estimated
    minimum costs for each unachieved goal predicate independently. The cost for
    each goal is estimated by finding the minimum navigation cost for a suitable
    rover to visit necessary locations (sample/image/calibration waypoints) and
    then a communication waypoint, plus the costs of the required actions
    (sample, drop, calibrate, take_image, communicate). Navigation costs are
    computed using shortest paths on the rover-specific traverse graphs.

    # Assumptions:
    - The set of waypoints, rovers, cameras, objectives, etc., is static.
    - The `can_traverse` relationships are static and define an unweighted graph.
    - Lander locations are static.
    - Initial soil/rock sample locations are static.
    - Objective/calibration target visibility from waypoints is static.
    - Rover capabilities (equipped_for_...) are static.
    - Camera properties (on_board, supports, calibration_target) are static.
    - Each goal predicate (communicated_...) is treated independently.
    - The heuristic ignores negative interactions like resource contention (e.g.,
      a rover's store being full preventing sampling for another goal, or a
      camera being uncalibrated by taking an image for a different goal).
    - The heuristic assumes solvable goals (e.g., required samples exist,
      visible_from facts exist for imaging). Unsolvable goals contribute infinity.

    # Heuristic Initialization
    - Extracts all objects and their types.
    - Extracts static facts like lander locations, rover capabilities, store mappings,
      camera info, visibility facts (for imaging/calibration/communication),
      and initial sample locations.
    - Builds the navigation graph for each rover based on `can_traverse` facts.
    - Precomputes all-pairs shortest paths for each rover on its navigation graph
      using BFS.
    - Identifies communication waypoints (visible from lander locations).

    # Step-By-Step Thinking for Computing Heuristic
    1.  Identify all goal predicates that are not yet satisfied in the current state.
    2.  Initialize the total heuristic cost to 0.
    3.  For each unachieved goal predicate:
        a.  Determine the type of goal (soil, rock, image).
        b.  Find all rovers/cameras capable of achieving this goal based on static facts.
        c.  If no capable rover/camera exists, this goal is likely impossible; skip or assign a very high cost.
        d.  Estimate the minimum cost to achieve *only this specific goal* from the current state, considering only the state of the chosen rover/camera (location, store status, calibration).
            -   **For Soil/Rock Goals (communicated_soil/rock_data ?w):**
                -   If the sample `(have_soil/rock_analysis ?r ?w)` is not yet collected by any suitable rover `?r`:
                    -   Cost includes: Navigation from `?r`'s current location to `?w`, plus 1 for `sample_soil/rock`, plus 1 for `drop` if `?r`'s store is full.
                    -   After sampling, the rover is at `?w`.
                    -   Cost includes: Navigation from `?w` to the nearest communication waypoint, plus 1 for `communicate_soil/rock_data`.
                -   If the sample `(have_soil/rock_analysis ?r ?w)` is already collected by a suitable rover `?r`:
                    -   Cost includes: Navigation from `?r`'s current location to the nearest communication waypoint, plus 1 for `communicate_soil/rock_data`.
                -   Take the minimum cost over all suitable rovers.
            -   **For Image Goals (communicated_image_data ?o ?m):**
                -   If the image `(have_image ?r ?o ?m)` is not yet taken by any suitable rover `?r` with camera `?i`:
                    -   Cost includes: Navigation from `?r`'s current location to a calibration waypoint for `?i`, plus 1 for `calibrate`, plus navigation from the calibration waypoint to an image waypoint for `?o`, plus 1 for `take_image`. The navigation costs are minimized over all possible calibration and image waypoints.
                    -   After taking the image, the rover is at the chosen image waypoint.
                    -   Cost includes: Navigation from the image waypoint to the nearest communication waypoint, plus 1 for `communicate_image_data`.
                -   If the image `(have_image ?r ?o ?m)` is already taken by a suitable rover `?r` with camera `?i`:
                    -   Cost includes: Navigation from `?r`'s current location to the nearest communication waypoint, plus 1 for `communicate_image_data`.
                -   Take the minimum cost over all suitable rover/camera pairs.
        e.  Add the estimated minimum cost for this goal to the total heuristic cost.
    4.  Return the total heuristic cost. If any necessary goal was impossible to achieve based on static facts (e.g., no equipped rover, no visible_from waypoint), the total cost might be infinity; handle this by returning a large finite number.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals

        # --- Extract Objects and Types ---
        # Initialize sets for all object types
        self.rovers = set()
        self.waypoints = set()
        self.stores = set()
        self.cameras = set()
        self.modes = set()
        self.landers = set()
        self.objectives = set()
        self.cal_targets = set() # Calibration targets are objectives

        # Extract objects and types from initial state and static facts
        # Iterate through all facts in the initial state and static facts
        all_facts = set(task.initial_state) | set(task.static)
        for fact in all_facts:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts
            pred = parts[0]

            # Identify objects based on predicates they appear in
            if pred == 'at' and len(parts) == 3:
                obj, loc = parts[1:]
                if obj.startswith('rover'): self.rovers.add(obj)
                if loc.startswith('waypoint'): self.waypoints.add(loc)
            elif pred == 'at_lander' and len(parts) == 3:
                 lander, loc = parts[1:]
                 self.landers.add(lander)
                 self.waypoints.add(loc)
            elif pred == 'store_of' and len(parts) == 3:
                 store, rover = parts[1:]
                 self.stores.add(store)
                 self.rovers.add(rover)
            elif pred == 'on_board' and len(parts) == 3:
                 camera, rover = parts[1:]
                 self.cameras.add(camera)
                 self.rovers.add(rover)
            elif pred == 'supports' and len(parts) == 3:
                 camera, mode = parts[1:]
                 self.cameras.add(camera)
                 self.modes.add(mode)
            elif pred == 'calibration_target' and len(parts) == 3:
                 camera, objective = parts[1:]
                 self.cameras.add(camera)
                 self.objectives.add(objective)
                 self.cal_targets.add(objective) # Calibration targets are a subset of objectives
            elif pred == 'visible_from' and len(parts) == 3:
                 objective, waypoint = parts[1:]
                 self.objectives.add(objective)
                 self.waypoints.add(waypoint)
            elif pred in ['at_soil_sample', 'at_rock_sample'] and len(parts) == 2:
                 waypoint = parts[1]
                 self.waypoints.add(waypoint)
            elif pred == 'can_traverse' and len(parts) == 4:
                 rover, wp1, wp2 = parts[1:]
                 self.rovers.add(rover)
                 self.waypoints.add(wp1)
                 self.waypoints.add(wp2)
            elif pred == 'visible' and len(parts) == 3:
                 wp1, wp2 = parts[1:]
                 self.waypoints.add(wp1)
                 self.waypoints.add(wp2)
            # Add other predicates if they might introduce new objects not covered above
            # e.g., 'equipped_for_soil_analysis' might introduce a rover if not in 'at'
            # but the above predicates should cover all typed objects in typical instances.


        # --- Extract Static Facts into Data Structures ---
        self.lander_locations = {parts[2] for fact in task.static if match(fact, "at_lander", "*", "*")}

        # Map rover capabilities
        self.rover_capabilities = defaultdict(lambda: {'soil': False, 'rock': False, 'imaging': False})
        for fact in task.static:
            if match(fact, "equipped_for_soil_analysis", "*"):
                self.rover_capabilities[get_parts(fact)[1]]['soil'] = True
            elif match(fact, "equipped_for_rock_analysis", "*"):
                self.rover_capabilities[get_parts(fact)[1]]['rock'] = True
            elif match(fact, "equipped_for_imaging", "*"):
                self.rover_capabilities[get_parts(fact)[1]]['imaging'] = True

        # Map store to rover and vice versa
        self.store_to_rover = {s: r for fact in task.static if match(fact, "store_of", "*", "*") for s, r in [get_parts(fact)[1:]]}
        self.rover_to_store = {r: s for s, r in self.store_to_rover.items()}


        # Map camera info
        self.camera_info = defaultdict(lambda: {'supports': set(), 'on_board': None, 'calibration_target': None})
        for fact in task.static:
            if match(fact, "supports", "*", "*"):
                c, m = get_parts(fact)[1:]
                self.camera_info[c]['supports'].add(m)
            elif match(fact, "on_board", "*", "*"):
                c, r = get_parts(fact)[1:]
                self.camera_info[c]['on_board'] = r
            elif match(fact, "calibration_target", "*", "*"):
                c, t = get_parts(fact)[1:]
                self.camera_info[c]['calibration_target'] = t

        # Map objective/calibration target visibility to waypoints
        self.objective_wps = defaultdict(set)
        self.caltarget_wps = defaultdict(set)
        for fact in task.static:
            if match(fact, "visible_from", "*", "*"):
                obj, wp = get_parts(fact)[1:]
                self.objective_wps[obj].add(wp)
                # Check if this objective is also a calibration target
                if obj in self.cal_targets:
                     self.caltarget_wps[obj].add(wp)


        # Initial sample locations (only relevant if goal requires sampling)
        # The goal predicates `communicated_soil_data` etc. refer to the *waypoint*
        # where the sample *was* taken. So we need the initial sample locations.
        self.initial_soil_samples = {parts[1] for fact in task.initial_state if match(fact, "at_soil_sample", "*")}
        self.initial_rock_samples = {parts[1] for fact in task.initial_state if match(fact, "at_rock_sample", "*")}


        # Communication waypoints (visible from any lander location)
        self.comm_wps = set()
        lander_visible_wps = defaultdict(set)
        for fact in task.static:
            if match(fact, "visible", "*", "*"):
                wp1, wp2 = get_parts(fact)[1:]
                lander_visible_wps[wp1].add(wp2)
                lander_visible_wps[wp2].add(wp1) # Assuming visible is symmetric

        for lander_loc in self.lander_locations:
             self.comm_wps.update(lander_visible_wps.get(lander_loc, set()))
             # A rover can also communicate if it is AT the lander location itself,
             # and the lander is there. The action precondition is (at ?r ?x) (at_lander ?l ?y) (visible ?x ?y).
             # If ?x == ?y, then (visible ?x ?y) is usually not in PDDL.
             # However, the examples show visible between distinct waypoints.
             # Let's assume communication is only possible from waypoints visible *from* the lander location,
             # not necessarily including the lander location itself unless it's visible from itself (unlikely).
             # Re-reading the action: (visible ?x ?y) where ?x is rover loc, ?y is lander loc.
             # So we need waypoints ?x such that (visible ?x ?y) is true for some lander loc ?y.
             # My current logic `if wp1 in self.lander_locations: self.comm_wps.add(wp2)` and vice versa is correct.

        # Build rover navigation graphs
        self.rover_graphs = {r: {wp: set() for wp in self.waypoints} for r in self.rovers}
        for fact in task.static:
            if match(fact, "can_traverse", "*", "*", "*"):
                r, wp1, wp2 = get_parts(fact)[1:]
                # Ensure waypoints are known before adding edges
                if r in self.rover_graphs and wp1 in self.waypoints and wp2 in self.waypoints:
                    self.rover_graphs[r][wp1].add(wp2)
                    # Assuming can_traverse is symmetric based on example instances
                    self.rover_graphs[r][wp2].add(wp1)


        # Precompute all-pairs shortest paths for each rover
        self.all_paths = {}
        for rover in self.rovers:
            self.all_paths[rover] = {}
            # Only compute paths if the rover graph is not empty and has waypoints
            if self.rover_graphs[rover] and self.waypoints:
                for start_wp in self.waypoints:
                     self.all_paths[rover][start_wp] = bfs(self.rover_graphs[rover], start_wp)

        # Use float('inf') for unreachable distances
        self._infinity = float('inf')
        self._large_finite = 1000000 # Value to return for impossible goals

    def get_distance(self, rover, wp1, wp2):
        """Get shortest distance for a rover between two waypoints."""
        # Check if rover exists and waypoints are in its graph's distance map
        if rover not in self.all_paths or wp1 not in self.all_paths[rover] or wp2 not in self.all_paths[rover].get(wp1, {}):
            return self._infinity
        return self.all_paths[rover][wp1][wp2]

    def get_min_distance_to_comm(self, rover, start_wp):
        """Get shortest distance for a rover from a waypoint to any communication waypoint."""
        if not self.comm_wps: return self._infinity # No communication points defined
        if rover not in self.all_paths or start_wp not in self.all_paths[rover]:
            return self._infinity # Rover or start_wp not in graph's distance map

        min_dist = self._infinity
        paths_from_start = self.all_paths[rover][start_wp]

        for comm_wp in self.comm_wps:
            if comm_wp in paths_from_start:
                 min_dist = min(min_dist, paths_from_start[comm_wp])

        return min_dist


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Extract relevant state facts efficiently
        rover_locations = {}
        store_full = set() # Set of stores that are full
        calibrated_cams = set() # Set of (camera, rover) tuples
        have_soil = set() # Set of (rover, waypoint) tuples
        have_rock = set() # Set of (rover, waypoint) tuples
        have_image = set() # Set of (rover, objective, mode) tuples
        comm_soil = set() # Set of waypoints
        comm_rock = set() # Set of waypoints
        comm_image = set() # Set of (objective, mode) tuples

        # Using a single pass through the state facts
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue
            pred = parts[0]

            if pred == 'at' and len(parts) == 3 and parts[1] in self.rovers and parts[2] in self.waypoints:
                rover_locations[parts[1]] = parts[2]
            elif pred == 'full' and len(parts) == 2 and parts[1] in self.stores:
                store_full.add(parts[1])
            elif pred == 'calibrated' and len(parts) == 3 and parts[1] in self.cameras and parts[2] in self.rovers:
                calibrated_cams.add((parts[1], parts[2]))
            elif pred == 'have_soil_analysis' and len(parts) == 3 and parts[1] in self.rovers and parts[2] in self.waypoints:
                have_soil.add((parts[1], parts[2]))
            elif pred == 'have_rock_analysis' and len(parts) == 3 and parts[1] in self.rovers and parts[2] in self.waypoints:
                have_rock.add((parts[1], parts[2]))
            elif pred == 'have_image' and len(parts) == 4 and parts[1] in self.rovers and parts[2] in self.objectives and parts[3] in self.modes:
                have_image.add((parts[1], parts[2], parts[3]))
            elif pred == 'communicated_soil_data' and len(parts) == 2 and parts[1] in self.waypoints:
                comm_soil.add(parts[1])
            elif pred == 'communicated_rock_data' and len(parts) == 2 and parts[1] in self.waypoints:
                comm_rock.add(parts[1])
            elif pred == 'communicated_image_data' and len(parts) == 3 and parts[1] in self.objectives and parts[2] in self.modes:
                comm_image.add((parts[1], parts[2]))

        total_cost = 0
        infinity = self._infinity
        large_finite = self._large_finite

        # Process unachieved goals
        for goal in self.goals:
            # Check if goal is already achieved in the current state
            if goal in state:
                continue

            parts = get_parts(goal)
            if not parts: continue # Should not happen for valid goals
            pred = parts[0]

            min_goal_cost = infinity

            if pred == "communicated_soil_data" and len(parts) == 2:
                w = parts[1]
                # Check if the goal waypoint initially had a soil sample
                if w not in self.initial_soil_samples:
                    # This goal is likely impossible in this domain based on initial state
                    continue # Skip this goal

                # Find suitable rovers equipped for soil analysis
                R_soil = {r for r, cap in self.rover_capabilities.items() if cap['soil']}
                if not R_soil:
                    # No rover can perform soil analysis
                    continue # Skip this goal

                for r in R_soil:
                    current_loc = rover_locations.get(r)
                    # If rover location is not in state, it's an invalid state or setup
                    if current_loc is None: continue
                    if current_loc not in self.waypoints: continue # Rover at unknown location

                    cost_r = 0
                    r_has_sample = (r, w) in have_soil
                    r_store = self.rover_to_store.get(r)
                    r_store_full = r_store is not None and r_store in store_full

                    if not r_has_sample:
                        # Cost to get sample: navigate to w, maybe drop, sample.
                        nav_cost_to_sample = self.get_distance(r, current_loc, w)
                        if nav_cost_to_sample == infinity: continue # Rover cannot reach sample location

                        cost_r += nav_cost_to_sample
                        if r_store_full: cost_r += 1 # drop action cost
                        cost_r += 1 # sample_soil action cost

                        # After sampling, rover is at w
                        location_after_sample = w
                    else:
                        # Rover already has the sample, starts communication path from current_loc
                        location_after_sample = current_loc

                    # Cost to communicate: navigate to comm point, communicate.
                    nav_cost_to_comm = self.get_min_distance_to_comm(r, location_after_sample)
                    if nav_cost_to_comm == infinity: continue # Cannot reach any communication point

                    cost_r += nav_cost_to_comm
                    cost_r += 1 # communicate_soil_data action cost

                    min_goal_cost = min(min_goal_cost, cost_r)

            elif pred == "communicated_rock_data" and len(parts) == 2:
                w = parts[1]
                # Check if the goal waypoint initially had a rock sample
                if w not in self.initial_rock_samples:
                    continue # Skip this goal

                # Find suitable rovers equipped for rock analysis
                R_rock = {r for r, cap in self.rover_capabilities.items() if cap['rock']}
                if not R_rock:
                    continue # Skip this goal

                for r in R_rock:
                    current_loc = rover_locations.get(r)
                    if current_loc is None: continue
                    if current_loc not in self.waypoints: continue # Rover at unknown location

                    cost_r = 0
                    r_has_sample = (r, w) in have_rock
                    r_store = self.rover_to_store.get(r)
                    r_store_full = r_store is not None and r_store in store_full

                    if not r_has_sample:
                        # Cost to get sample: navigate to w, maybe drop, sample.
                        nav_cost_to_sample = self.get_distance(r, current_loc, w)
                        if nav_cost_to_sample == infinity: continue

                        cost_r += nav_cost_to_sample
                        if r_store_full: cost_r += 1 # drop action cost
                        cost_r += 1 # sample_rock action cost

                        # After sampling, rover is at w
                        location_after_sample = w
                    else:
                        # Rover already has the sample, starts communication path from current_loc
                        location_after_sample = current_loc

                    # Cost to communicate: navigate to comm point, communicate.
                    nav_cost_to_comm = self.get_min_distance_to_comm(r, location_after_sample)
                    if nav_cost_to_comm == infinity: continue

                    cost_r += nav_cost_to_comm
                    cost_r += 1 # communicate_rock_data action cost

                    min_goal_cost = min(min_goal_cost, cost_r)

            elif pred == "communicated_image_data" and len(parts) == 3:
                o, m = parts[1:]

                # Find suitable rover/camera pairs
                RC_imaging = set()
                for r, cap in self.rover_capabilities.items():
                    if cap['imaging']:
                        for c, info in self.camera_info.items():
                            # Check if camera is on board this rover and supports the mode
                            if info['on_board'] == r and m in info['supports']:
                                RC_imaging.add((r, c))
                if not RC_imaging:
                    continue # No suitable rover/camera pair

                # Find relevant waypoints
                W_img = self.objective_wps.get(o, set())
                if not W_img:
                    # No waypoint from which the objective is visible
                    continue # Skip this goal

                min_goal_cost = infinity

                for r, i in RC_imaging:
                    current_loc = rover_locations.get(r)
                    if current_loc is None: continue
                    if current_loc not in self.waypoints: continue # Rover at unknown location

                    cost_r_i = 0
                    r_i_has_image = (r, o, m) in have_image

                    if not r_i_has_image:
                        # Cost to get image: navigate to cal_wp, calibrate, navigate to img_wp, take_image.
                        cal_target = self.camera_info[i]['calibration_target']
                        if cal_target is None:
                            # Camera has no calibration target defined
                            continue # Cannot calibrate this camera

                        W_cal = self.caltarget_wps.get(cal_target, set())
                        if not W_cal:
                            # No waypoint from which the calibration target is visible
                            continue # Cannot calibrate this camera

                        min_nav_cost_to_image_loc = infinity
                        best_wp_img = None # Track the image waypoint for communication path

                        # Find minimum path cost from current_loc -> wp_cal -> wp_img
                        # This involves iterating through all combinations of calibration and image waypoints
                        for wp_cal in W_cal:
                            nav_cost_curr_to_cal = self.get_distance(r, current_loc, wp_cal)
                            if nav_cost_curr_to_cal == infinity: continue # Cannot reach this calibration waypoint

                            for wp_img in W_img:
                                nav_cost_cal_to_img = self.get_distance(r, wp_cal, wp_img)
                                if nav_cost_cal_to_img == infinity: continue # Cannot reach this image waypoint from calibration waypoint

                                total_nav = nav_cost_curr_to_cal + nav_cost_cal_to_img
                                if total_nav < min_nav_cost_to_image_loc:
                                    min_nav_cost_to_image_loc = total_nav
                                    best_wp_img = wp_img

                        if min_nav_cost_to_image_loc == infinity:
                            # Cannot find a path through calibration and image waypoints
                            continue # Skip this rover/camera pair for this goal

                        cost_r_i += min_nav_cost_to_image_loc
                        cost_r_i += 1 # calibrate action cost
                        cost_r_i += 1 # take_image action cost

                        # After taking image, rover is at best_wp_img
                        location_after_image = best_wp_img
                    else:
                        # Rover already has the image, starts communication path from current_loc
                        location_after_image = current_loc

                    # Cost to communicate: navigate to comm point, communicate.
                    nav_cost_to_comm = self.get_min_distance_to_comm(r, location_after_image)
                    if nav_cost_to_comm == infinity:
                        # Cannot reach any communication point from the location after getting the image
                        continue # Skip this rover/camera pair for this goal

                    cost_r_i += nav_cost_to_comm
                    cost_r_i += 1 # communicate_image_data action cost

                    min_goal_cost = min(min_goal_cost, cost_r_i)

            # Add the minimum cost found for this goal to the total heuristic cost
            # If min_goal_cost is still infinity, it means this specific goal is impossible
            # for any suitable rover/camera given the static facts and initial state.
            # In a solvable problem, this shouldn't happen for necessary goals.
            # If it does, we treat the whole state as very costly.
            if min_goal_cost != infinity:
                total_cost += min_goal_cost
            else:
                 # If a necessary goal is impossible to achieve, return a large finite value
                 # to indicate a high cost state, but avoid true infinity issues.
                 # This state might be unsolvable.
                 return large_finite

        return total_cost
