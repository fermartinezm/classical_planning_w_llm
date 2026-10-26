from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at rover1 waypoint1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure the fact has at least as many parts as the pattern
    if len(parts) < len(args):
        return False
    # Check if each part matches the corresponding pattern argument
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve all goal
    conditions. It breaks down each goal (communicating data) into a sequence
    of necessary steps (collect/calibrate, analyse/take_image, communicate)
    and counts the number of unachieved steps. It also adds an estimated
    movement cost if the necessary item (sample, analysis, image, calibrated
    camera) is not already at the location required for the next step in the
    sequence. Movement cost is estimated as the shortest path distance from
    the nearest *relevant* rover to the target location.

    # Assumptions
    - The heuristic assumes a simplified view of movement: the cost between
      waypoints is the shortest path distance in the 'visible' graph, and
      any rover relevant to a task stage can potentially move along these paths.
    - It assumes that if a required object (like an equipped rover, camera,
      sample at a waypoint, or objective visibility) does not exist based on
      static facts, the goal is unreachable (heuristic returns infinity).
    - It assumes that freeing up a full store takes one action (analyse or drop).
    - It assumes that 'visible' predicate implies bidirectional traversal for
      shortest path calculation.

    # Heuristic Initialization
    The constructor precomputes static information from the task:
    - All-pairs shortest paths between waypoints based on the 'visible' predicate.
    - The location of the lander.
    - Locations of soil, rock, and mineral samples.
    - Objective visibility from waypoints.
    - Camera capabilities (which rover they are on, modes supported, calibration target).
    - Which rovers are equipped for soil, rock, mineral, and imaging analysis.
    - Which store belongs to which rover.
    - A list of all rovers mentioned in static facts.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic is computed as follows:

    1. Initialize total_cost to 0.
    2. Precompute shortest path distances between all waypoints using BFS on the
       'visible' graph during initialization. Extract other static information.
    3. For each goal fact specified in the task:
        a. If the goal fact is already true in the current state, continue to the next goal.
        b. If the goal is `(communicated_soil_data ?w)` (or rock/mineral):
            i. Add 1 to total_cost for the 'communicate' action.
            ii. Check if `(have_soil_analysis ?r ?w)` (or rock/mineral) is true for any rover `?r` in the state. If not:
                - Add 1 for the 'analyse' action.
                - Check if `(have_soil_sample ?r ?w)` (or rock/mineral) is true for any rover `?r` in the state. If not:
                    - Add 1 for the 'collect' action.
                    - Identify rovers equipped for this data type. Check if any of them is currently at waypoint `?w`. If not, add the minimum distance from any of these equipped rovers' current locations to `?w` (using precomputed distances).
                    - Check if any equipped rover for this data type has an empty store. If not, add 1 (cost to free a store).
                - Identify rovers that currently have the sample. Check if any of them is currently at waypoint `?w` (needed for analyse). If not, add the minimum distance from any of these rovers' current locations to `?w`.
            iii. Identify rovers that currently have the analysis. Check if any of them is currently at the lander location (needed for communicate). If not, add the minimum distance from any of these rovers' current locations to the lander location.
        c. If the goal is `(communicated_image_data ?o ?m)`:
            i. Add 1 to total_cost for the 'communicate' action.
            ii. Check if `(have_image ?r ?o ?m)` is true for any rover `?r` in the state. If not:
                - Add 1 for the 'take_image' action.
                - Check if there exists a camera `?c` (supporting `?m`, targeting `?o`, on an imaging rover `?r`) that is currently calibrated (`(calibrated ?c ?r)` in state) AND that rover `?r` is currently at a waypoint `?wp` where `(visible_from ?o ?wp)` is true. If not:
                    - Add 1 for the 'calibrate' action.
                    - Identify rovers that statically have a suitable camera. Check if any of them is currently at any suitable waypoint (static visibility check). If not, add the minimum distance from any of these rovers' current locations to the set of suitable waypoints.
                - Identify rovers that currently have a suitable *calibrated* camera. Check if any of them is currently at any suitable waypoint (static visibility check) (needed for take_image). If not, add the minimum distance from any of these rovers' current locations to the set of suitable waypoints.
            iii. Identify rovers that currently have the image. Check if any of them is currently at the lander location (needed for communicate). If not, add the minimum distance from any of these rovers' current locations to the lander location.
    4. Return the total_cost. If any required static condition (like existence of equipped rover or visible objective) is missing or a required movement is impossible, return infinity.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information and
        precomputing waypoint distances.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # --- Precompute Waypoint Distances (BFS on visible graph) ---
        waypoints = set()
        visible_graph = {} # Adjacency list

        for fact in static_facts:
            if match(fact, "visible", "*", "*"):
                wp1, wp2 = get_parts(fact)[1], get_parts(fact)[2]
                waypoints.add(wp1)
                waypoints.add(wp2)
                visible_graph.setdefault(wp1, set()).add(wp2)
                visible_graph.setdefault(wp2, set()).add(wp1) # Assuming visible is symmetric

        self.distances = {}
        for start_wp in waypoints:
            self.distances[start_wp] = {}
            queue = deque([(start_wp, 0)])
            visited = {start_wp}
            while queue:
                current_wp, dist = queue.popleft()
                self.distances[start_wp][current_wp] = dist
                for neighbor in visible_graph.get(current_wp, set()):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, dist + 1))

        # --- Extract Other Static Information ---
        self.lander_loc = None
        self.soil_samples = set()
        self.rock_samples = set()
        self.mineral_samples = set()
        self.objective_visibility = {} # objective -> {waypoint, ...}
        self.camera_info = {} # camera -> {'rover': r, 'modes': {m, ...}, 'target': o}
        self.equipped_rovers = {'soil': set(), 'rock': set(), 'mineral': set(), 'imaging': set()}
        self.store_of = {} # store -> rover
        self.all_rovers = set() # Collect all rovers mentioned in static facts

        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "at_lander":
                self.lander_loc = parts[2]
            elif parts[0] == "at_soil_sample":
                self.soil_samples.add(parts[1])
            elif parts[0] == "at_rock_sample":
                self.rock_samples.add(parts[1])
            elif parts[0] == "at_mineral_sample":
                self.mineral_samples.add(parts[1])
            elif parts[0] == "visible_from":
                obj, wp = parts[1], parts[2]
                self.objective_visibility.setdefault(obj, set()).add(wp)
            elif parts[0] == "on_board":
                cam, r = parts[1], parts[2]
                self.camera_info.setdefault(cam, {})['rover'] = r
                self.camera_info[cam].setdefault('modes', set())
                self.all_rovers.add(r)
            elif parts[0] == "supports":
                cam, mode = parts[1], parts[2]
                self.camera_info.setdefault(cam, {}).setdefault('modes', set()).add(mode)
            elif parts[0] == "calibration_target":
                cam, obj = parts[1], parts[2]
                self.camera_info.setdefault(cam, {})['target'] = obj
            elif parts[0] == "equipped_for_soil_analysis":
                self.equipped_rovers['soil'].add(parts[1])
                self.all_rovers.add(parts[1])
            elif parts[0] == "equipped_for_rock_analysis":
                self.equipped_rovers['rock'].add(parts[1])
                self.all_rovers.add(parts[1])
            elif parts[0] == "equipped_for_mineral_analysis":
                self.equipped_rovers['mineral'].add(parts[1])
                self.all_rovers.add(parts[1])
            elif parts[0] == "equipped_for_imaging":
                self.equipped_rovers['imaging'].add(parts[1])
                self.all_rovers.add(parts[1])
            elif parts[0] == "store_of":
                s, r = parts[1], parts[2]
                self.store_of[s] = r
                self.all_rovers.add(r)

        # Ensure lander location was found (critical for communication goals)
        if self.lander_loc is None:
             # If lander location is not in static facts, goals requiring communication are impossible.
             # We can signal this by setting distances to lander as infinity for all waypoints.
             # Or simply return infinity if lander_loc is None when needed.
             # Let's assume lander_loc is always present in solvable problems.
             pass


    def get_location(self, state, obj):
        """Finds the current waypoint location of an object (rover)."""
        for fact in state:
            if match(fact, "at", obj, "*"):
                return get_parts(fact)[2]
        # Lander location is static, handled separately if needed.
        return None # Object not found at any waypoint in the current state


    def min_dist_from_rovers_in_set(self, state, rovers_set, target_wps):
        """
        Finds the minimum distance from any rover in rovers_set's current location
        to any waypoint in the target_wps set.
        target_wps can be a single waypoint string or a set of waypoint strings.
        Returns float('inf') if rovers_set is empty or no path exists.
        """
        if not rovers_set:
            return float('inf')

        if isinstance(target_wps, str):
            target_wps = {target_wps}

        min_d = float('inf')
        for rover in rovers_set:
            rover_loc = self.get_location(state, rover)
            if rover_loc and rover_loc in self.distances:
                for target_wp in target_wps:
                    if target_wp in self.distances.get(rover_loc, {}):
                         min_d = min(min_d, self.distances[rover_loc][target_wp])
        return min_d


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.
        total_cost = 0  # Initialize action cost counter.

        # Check if goal is already reached
        if self.goals <= state:
            return 0

        # Handle Soil/Rock/Mineral Goals
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] in ["communicated_soil_data", "communicated_rock_data", "communicated_mineral_data"]:
                data_type = parts[0].split('_')[1] # soil, rock, mineral
                waypoint = parts[1]
                have_sample_pred = f"have_{data_type}_sample"
                have_analysis_pred = f"have_{data_type}_analysis"
                equipped_type = data_type # Matches key in self.equipped_rovers

                # Check if sample exists at waypoint (static check)
                if data_type == 'soil' and waypoint not in self.soil_samples: return float('inf') # Goal impossible
                if data_type == 'rock' and waypoint not in self.rock_samples: return float('inf') # Goal impossible
                if data_type == 'mineral' and waypoint not in self.mineral_samples: return float('inf') # Goal impossible

                if goal not in state:
                    total_cost += 1 # Communicate

                    # Check if analysis exists
                    has_analysis = any(match(fact, have_analysis_pred, "*", waypoint) for fact in state)
                    if not has_analysis:
                        total_cost += 1 # Analyse

                        # Check if sample exists
                        has_sample = any(match(fact, have_sample_pred, "*", waypoint) for fact in state)
                        if not has_sample:
                            total_cost += 1 # Collect

                            # Need equipped rover at sample waypoint for collect
                            equipped_rovers = self.equipped_rovers.get(equipped_type, set())
                            if not equipped_rovers: return float('inf') # Cannot collect
                            equipped_at_wp = any(self.get_location(state, r) == waypoint for r in equipped_rovers)
                            if not equipped_at_wp:
                                dist = self.min_dist_from_rovers_in_set(state, equipped_rovers, waypoint)
                                if dist == float('inf'): return float('inf')
                                total_cost += dist

                            # Need empty store on equipped rover for collect
                            has_empty_store = any(match(fact, "empty", s) for fact in state for s, r_s in self.store_of.items() if r_s in equipped_rovers)
                            if not has_empty_store:
                                total_cost += 1 # Assume 1 action to free store

                        # Need rover with sample at sample waypoint for analyse
                        # Find rovers that currently have the sample
                        rovers_with_sample = {get_parts(fact)[2] for fact in state if match(fact, have_sample_pred, "*", waypoint)}
                        rover_with_sample_at_wp = any(self.get_location(state, r) == waypoint for r in rovers_with_sample)

                        if not rover_with_sample_at_wp:
                             dist = self.min_dist_from_rovers_in_set(state, rovers_with_sample, waypoint)
                             if dist == float('inf'): return float('inf')
                             total_cost += dist

                    # Need rover with analysis at lander location for communicate
                    # Find rovers that currently have the analysis
                    rovers_with_analysis = {get_parts(fact)[2] for fact in state if match(fact, have_analysis_pred, "*", waypoint)}
                    rover_with_analysis_at_lander = any(self.get_location(state, r) == self.lander_loc for r in rovers_with_analysis)

                    if not rover_with_analysis_at_lander:
                         dist = self.min_dist_from_rovers_in_set(state, rovers_with_analysis, self.lander_loc)
                         if dist == float('inf'): return float('inf')
                         total_cost += dist


        # Handle Image Goals
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "communicated_image_data":
                objective, mode = parts[1], parts[2]

                # Check if objective is visible from any waypoint (static check)
                suitable_wps = self.objective_visibility.get(objective, set())
                if not suitable_wps: return float('inf') # Cannot see objective

                # Check if any camera supports this mode and targets this objective on an imaging rover (static check)
                imaging_rovers = self.equipped_rovers.get('imaging', set())
                rovers_with_suitable_camera_static = set() # Rovers that *can* take the picture
                suitable_camera_exists = False
                for cam, info in self.camera_info.items():
                    if info.get('target') == objective and mode in info.get('modes', set()) and info.get('rover') in imaging_rovers:
                        suitable_camera_exists = True
                        rovers_with_suitable_camera_static.add(info['rover'])
                if not suitable_camera_exists: return float('inf') # No suitable camera setup

                if goal not in state:
                    total_cost += 1 # Communicate

                    # Check if image exists
                    has_image = any(match(fact, "have_image", "*", objective, mode) for fact in state)
                    if not has_image:
                        total_cost += 1 # Take Image

                        # Check if suitable camera is calibrated on a rover at a suitable waypoint
                        is_calibrated_at_suitable_wp = False
                        rovers_with_suitable_calibrated_camera_state = set() # Rovers that *currently* have a suitable calibrated camera

                        for cam, info in self.camera_info.items():
                            r = info.get('rover')
                            if r and info.get('target') == objective and mode in info.get('modes', set()) and r in imaging_rovers:
                                if f"(calibrated {cam} {r})" in state:
                                    rovers_with_suitable_calibrated_camera_state.add(r)
                                    rover_loc = self.get_location(state, r)
                                    if rover_loc and rover_loc in suitable_wps:
                                        is_calibrated_at_suitable_wp = True
                                        # No break here, need to find all rovers with calibrated cameras

                        if not is_calibrated_at_suitable_wp:
                            total_cost += 1 # Calibrate

                            # Need rover with suitable camera at suitable waypoint for calibration
                            # Check if any rover with a suitable camera (static set) is currently at any suitable waypoint.
                            rover_with_suitable_camera_at_suitable_wp = any(self.get_location(state, r) in suitable_wps for r in rovers_with_suitable_camera_static)

                            if not rover_with_suitable_camera_at_suitable_wp:
                                 dist = self.min_dist_from_rovers_in_set(state, rovers_with_suitable_camera_static, suitable_wps)
                                 if dist == float('inf'): return float('inf')
                                 total_cost += dist


                        # Need rover with calibrated camera at suitable waypoint for take_image
                        # This movement is needed if `is_calibrated_at_suitable_wp` is false.
                        if not is_calibrated_at_suitable_wp:
                             if rovers_with_suitable_calibrated_camera_state:
                                 dist = self.min_dist_from_rovers_in_set(state, rovers_with_suitable_calibrated_camera_state, suitable_wps)
                                 if dist == float('inf'): return float('inf')
                                 total_cost += dist
                             # else: No suitable calibrated camera exists *yet*. Movement covered by calibration step.


                    # Need rover with image at lander_loc for communicate
                    # Find rovers that currently have the image
                    rovers_with_image = {get_parts(fact)[2] for fact in state if match(fact, "have_image", "*", objective, mode)}
                    rover_with_image_at_lander = any(self.get_location(state, r) == self.lander_loc for r in rovers_with_image)

                    if not rover_with_image_at_lander:
                         dist = self.min_dist_from_rovers_in_set(state, rovers_with_image, self.lander_loc)
                         if dist == float('inf'): return float('inf')
                         total_cost += dist

        return total_cost
