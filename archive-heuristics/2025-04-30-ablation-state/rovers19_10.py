from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque
import math

# Helper functions to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and starts/ends with parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
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
    # Ensure the number of parts matches the number of arguments in the pattern
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the minimum number of actions required to achieve
    each unachieved goal fact independently and sums these minimum costs.
    It considers the steps needed for sampling (soil/rock), imaging (calibration,
    taking picture), and communication, including necessary navigation between
    relevant waypoints. Navigation cost is estimated using shortest path on
    the traverse graph.

    # Assumptions
    - Shortest path distance represents the minimum number of `navigate` actions.
    - Each action (sample, drop, calibrate, take_image, communicate) costs 1.
    - If a sample location (`at_soil_sample`, `at_rock_sample`) is not present
      in the initial state, the corresponding communication goal is considered
      impossible to achieve by sampling.
    - If required resources (equipped rover, camera, visible objective/target)
      are not available based on static/initial facts, the corresponding goal
      is considered impossible.
    - The heuristic calculates the cost for each goal independently and sums them,
      ignoring potential synergies (e.g., one navigation serving multiple goals)
      or conflicts (e.g., multiple rovers needing the same resource). This makes
      it non-admissible but potentially effective for greedy search.

    # Heuristic Initialization
    - Parses static facts and initial state to identify objects (rovers, waypoints, etc.)
      based on predicate arguments. Note: This object inference method is an approximation
      and might not be robust for all PDDL structures.
    - Extracts static relationships: lander locations, rover capabilities, store-rover mapping,
      camera information (on-board, supported modes, calibration target), objective/target
      visibility from waypoints, and the navigation graph (`can_traverse`).
    - Identifies initial sample locations (`at_soil_sample`, `at_rock_sample`).
    - Identifies communication waypoints (visible from lander locations).
    - Precomputes shortest path distances between all pairs of waypoints for each rover
      using Breadth-First Search (BFS) on the navigation graph.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic `h(state)` is calculated as follows:

    1.  Initialize total heuristic cost `total_h = 0`.
    2.  Extract dynamic information from the current state: current rover locations,
        calibrated cameras, collected soil/rock data, taken images, and full stores.
    3.  For each goal fact `G` in the task's goals:
        a.  If `G` is already true in the current state, continue to the next goal (cost is 0).
        b.  If `G` is `(communicated_soil_data ?w)`:
            i.  Initialize minimum cost for this goal `min_goal_cost = infinity`.
            ii. If `(at_soil_sample ?w)` was not in the initial state, this goal is impossible to achieve by sampling; `min_goal_cost` remains infinity.
            iii. Check if `(have_soil_analysis ?r ?w)` is true for any rover `?r` in the current state. If yes:
                -   Find the current location of `?r`.
                -   Calculate the cost to navigate `?r` to the closest communication waypoint (`dist + 1` for communicate). Update `min_goal_cost` with the minimum such cost.
            iv. If the data is not yet collected (`have_soil_analysis` is false for all rovers) AND `(at_soil_sample ?w)` is true in the current state:
                -   For each rover `?r` equipped for soil analysis:
                    -   Calculate the cost: navigate to `?w` (`dist`), plus 1 for `drop` if its store is full, plus 1 for `sample_soil`, plus navigate from `?w` to the closest communication waypoint (`dist`), plus 1 for `communicate_soil_data`.
                    -   Update `min_goal_cost` with the minimum cost over all suitable rovers.
            v.  Add `min_goal_cost` to `total_h`.
        c.  If `G` is `(communicated_rock_data ?w)`: Follow the same logic as for soil data, substituting 'soil' with 'rock' and checking `at_rock_sample`.
        d.  If `G` is `(communicated_image_data ?o ?m)`:
            i.  Initialize minimum cost for this goal `min_goal_cost = infinity`.
            ii. Check if `(have_image ?r ?o ?m)` is true for any rover `?r` in the current state. If yes:
                -   Find the current location of `?r`.
                -   Calculate the cost to navigate `?r` to the closest communication waypoint (`dist + 1` for communicate). Update `min_goal_cost` with the minimum such cost.
            iii. If the image is not yet taken (`have_image` is false for all rovers):
                -   For each rover `?r` equipped for imaging, with a camera `?i` supporting mode `?m`, where `?o` is visible from some waypoint `?p`, and `?i`'s calibration target `?t` is visible from some waypoint `?w`:
                    -   Calculate the cost for this combination (`?r`, `?i`, `?p`, `?w`):
                        -   Start with the cost to navigate `?r` from its current location.
                        -   If camera `?i` is not calibrated for `?r` in the current state: add cost to navigate to `?w` (`dist`) + 1 (`calibrate`). Current location becomes `?w`.
                        -   Add cost to navigate from the current location to `?p` (`dist`) + 1 (`take_image`). Current location becomes `?p`.
                        -   Add cost to navigate from `?p` to the closest communication waypoint (`dist`) + 1 (`communicate_image_data`).
                    -   Update `min_goal_cost` with the minimum cost over all suitable combinations.
            iv. Add `min_goal_cost` to `total_h`.
    4.  Return `total_h`.

    Note: If at any point a required navigation is impossible (distance is infinity), that specific path/option is discarded. If all options for a goal lead to infinity, the goal's minimum cost remains infinity, correctly contributing to an infinite total heuristic for an unsolvable state.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static
        initial_state = task.initial_state # Need initial state for sample locations

        # --- Precomputation ---

        # Infer objects and types (approximation based on predicate arguments)
        self.all_rovers = set()
        self.all_waypoints = set()
        self.all_landers = set()
        self.all_stores = set()
        self.all_cameras = set()
        self.all_modes = set()
        self.all_objectives = set()

        all_facts = set(initial_state) | set(static_facts) | set(self.goals) # Include goals to find all objects
        for fact in all_facts:
            parts = get_parts(fact)
            if not parts: continue # Skip empty facts

            pred = parts[0]
            args = parts[1:]

            if pred in ['at', 'at_lander'] and len(args) == 2:
                if args[0].startswith('rover'): self.all_rovers.add(args[0])
                elif args[0].startswith('lander') or args[0] == 'general': self.all_landers.add(args[0])
                if args[1].startswith('waypoint'): self.all_waypoints.add(args[1])
            elif pred == 'can_traverse' and len(args) == 3:
                 if args[0].startswith('rover'): self.all_rovers.add(args[0])
                 if args[1].startswith('waypoint'): self.all_waypoints.add(args[1])
                 if args[2].startswith('waypoint'): self.all_waypoints.add(args[2])
            elif pred in ['equipped_for_soil_analysis', 'equipped_for_rock_analysis', 'equipped_for_imaging'] and len(args) == 1:
                 if args[0].startswith('rover'): self.all_rovers.add(args[0])
            elif pred in ['empty', 'full'] and len(args) == 1:
                 if args[0].startswith('store'): self.all_stores.add(args[0])
            elif pred in ['have_rock_analysis', 'have_soil_analysis'] and len(args) == 2:
                 if args[0].startswith('rover'): self.all_rovers.add(args[0])
                 if args[1].startswith('waypoint'): self.all_waypoints.add(args[1])
            elif pred == 'calibrated' and len(args) == 2:
                 if args[0].startswith('camera'): self.all_cameras.add(args[0])
                 if args[1].startswith('rover'): self.all_rovers.add(args[1])
            elif pred == 'supports' and len(args) == 2:
                 if args[0].startswith('camera'): self.all_cameras.add(args[0])
                 self.all_modes.add(args[1])
            elif pred == 'visible' and len(args) == 2:
                 if args[0].startswith('waypoint'): self.all_waypoints.add(args[0])
                 if args[1].startswith('waypoint'): self.all_waypoints.add(args[1])
            elif pred == 'have_image' and len(args) == 3:
                 if args[0].startswith('rover'): self.all_rovers.add(args[0])
                 if args[1].startswith('objective'): self.all_objectives.add(args[1])
                 self.all_modes.add(args[2])
            elif pred in ['communicated_soil_data', 'communicated_rock_data'] and len(args) == 1:
                 if args[0].startswith('waypoint'): self.all_waypoints.add(args[0])
            elif pred == 'communicated_image_data' and len(args) == 2:
                 if args[0].startswith('objective'): self.all_objectives.add(args[0])
                 self.all_modes.add(args[1])
            elif pred in ['at_soil_sample', 'at_rock_sample'] and len(args) == 1:
                 if args[0].startswith('waypoint'): self.all_waypoints.add(args[0])
            elif pred == 'visible_from' and len(args) == 2:
                 if args[0].startswith('objective'): self.all_objectives.add(args[0])
                 if args[1].startswith('waypoint'): self.all_waypoints.add(args[1])
            elif pred == 'store_of' and len(args) == 2:
                 if args[0].startswith('store'): self.all_stores.add(args[0])
                 if args[1].startswith('rover'): self.all_rovers.add(args[1])
            elif pred == 'calibration_target' and len(args) == 2:
                 if args[0].startswith('camera'): self.all_cameras.add(args[0])
                 if args[1].startswith('objective'): self.all_objectives.add(args[1])
            elif pred == 'on_board' and len(args) == 2:
                 if args[0].startswith('camera'): self.all_cameras.add(args[0])
                 if args[1].startswith('rover'): self.all_rovers.add(args[1])


        # Static info extraction
        self.lander_loc = {} # {lander: waypoint}
        self.rover_caps = {r: set() for r in self.all_rovers} # {rover: set of capabilities}
        self.store_of_rover = {} # {rover: store} - assuming one store per rover
        self.camera_info = {c: {'rover': None, 'modes': set(), 'cal_target': None} for c in self.all_cameras} # {camera: info}
        self.obj_visible_from = {o: set() for o in self.all_objectives} # {objective: set of waypoints}
        self.cal_target_visible_from = {} # {objective: set of waypoints} - derived later
        self.nav_graph = {r: {wp: set() for wp in self.all_waypoints} for r in self.all_rovers} # {rover: {wp: set of wps}}
        self.comm_wps = set() # set of waypoints visible from any lander
        self.initial_soil_samples = set() # set of waypoints with soil samples initially
        self.initial_rock_samples = set() # set of waypoints with rock samples initially

        for fact in static_facts:
            parts = get_parts(fact)
            if match(fact, "at_lander", "*", "*"):
                self.lander_loc[parts[1]] = parts[2]
            elif match(fact, "equipped_for_soil_analysis", "*"):
                if parts[1] in self.rover_caps: self.rover_caps[parts[1]].add('soil')
            elif match(fact, "equipped_for_rock_analysis", "*"):
                if parts[1] in self.rover_caps: self.rover_caps[parts[1]].add('rock')
            elif match(fact, "equipped_for_imaging", "*"):
                if parts[1] in self.rover_caps: self.rover_caps[parts[1]].add('imaging')
            elif match(fact, "store_of", "*", "*"):
                if parts[2] in self.all_rovers and parts[1] in self.all_stores:
                    self.store_of_rover[parts[2]] = parts[1]
            elif match(fact, "on_board", "*", "*"):
                if parts[1] in self.camera_info and parts[2] in self.all_rovers:
                    self.camera_info[parts[1]]['rover'] = parts[2]
            elif match(fact, "supports", "*", "*"):
                 if parts[1] in self.camera_info and parts[2] in self.all_modes:
                    self.camera_info[parts[1]]['modes'].add(parts[2])
            elif match(fact, "calibration_target", "*", "*"):
                 if parts[1] in self.camera_info and parts[2] in self.all_objectives:
                    self.camera_info[parts[1]]['cal_target'] = parts[2]
            elif match(fact, "visible_from", "*", "*"):
                 if parts[1] in self.obj_visible_from and parts[2] in self.all_waypoints:
                    self.obj_visible_from[parts[1]].add(parts[2])
            elif match(fact, "can_traverse", "*", "*", "*"):
                r, w1, w2 = parts[1], parts[2], parts[3]
                if r in self.nav_graph and w1 in self.nav_graph[r] and w2 in self.all_waypoints:
                    self.nav_graph[r][w1].add(w2)
            # visible facts are processed after lander_loc is populated

        # Derive cal_target_visible_from
        for cam, info in self.camera_info.items():
            target = info.get('cal_target')
            if target and target in self.obj_visible_from:
                 self.cal_target_visible_from[target] = self.obj_visible_from[target]

        # Derive comm_wps
        lander_waypoints = set(self.lander_loc.values())
        for fact in static_facts:
             if match(fact, "visible", "*", "*"):
                 wp1, wp2 = get_parts(fact)[1], get_parts(fact)[2]
                 if wp1 in self.all_waypoints and wp2 in self.all_waypoints:
                     if wp2 in lander_waypoints:
                         self.comm_wps.add(wp1)
                     if wp1 in lander_waypoints:
                         self.comm_wps.add(wp2)

        # Initial state facts needed for samples
        for fact in initial_state:
            if match(fact, "at_soil_sample", "*"):
                if get_parts(fact)[1] in self.all_waypoints:
                    self.initial_soil_samples.add(get_parts(fact)[1])
            elif match(fact, "at_rock_sample", "*"):
                if get_parts(fact)[1] in self.all_waypoints:
                    self.initial_rock_samples.add(get_parts(fact)[1])

        # Compute shortest paths for each rover
        self.dist = {r: {} for r in self.all_rovers}
        for rover in self.all_rovers:
            for start_wp in self.all_waypoints:
                self.dist[rover][start_wp] = {wp: math.inf for wp in self.all_waypoints}
                if start_wp in self.dist[rover]: # Ensure start_wp is a valid key
                    self.dist[rover][start_wp][start_wp] = 0
                    q = deque([start_wp])
                    while q:
                        u = q.popleft()
                        if u in self.nav_graph.get(rover, {}): # Check if rover and waypoint exist in graph
                            for v in self.nav_graph[rover].get(u, set()):
                                if v in self.dist[rover][start_wp] and self.dist[rover][start_wp][v] == math.inf:
                                    self.dist[rover][start_wp][v] = self.dist[rover][start_wp][u] + 1
                                    q.append(v)


    def __call__(self, node):
        """Estimate the minimum cost to reach a goal state."""
        state = node.state
        total_h = 0

        # Get dynamic state info
        rover_locations = {} # {rover: waypoint}
        calibrated_cams = set() # {(camera, rover)}
        have_soil = set() # {(rover, waypoint)}
        have_rock = set() # {(rover, waypoint)}
        have_image = set() # {(rover, objective, mode)}
        full_stores = set() # {store}

        for fact in state:
            parts = get_parts(fact)
            if match(fact, "at", "*", "*"):
                if parts[1] in self.all_rovers and parts[2] in self.all_waypoints:
                    rover_locations[parts[1]] = parts[2]
            elif match(fact, "calibrated", "*", "*"):
                 if parts[1] in self.all_cameras and parts[2] in self.all_rovers:
                    calibrated_cams.add((parts[1], parts[2]))
            elif match(fact, "have_soil_analysis", "*", "*"):
                 if parts[1] in self.all_rovers and parts[2] in self.all_waypoints:
                    have_soil.add((parts[1], parts[2]))
            elif match(fact, "have_rock_analysis", "*", "*"):
                 if parts[1] in self.all_rovers and parts[2] in self.all_waypoints:
                    have_rock.add((parts[1], parts[2]))
            elif match(fact, "have_image", "*", "*", "*"):
                 if parts[1] in self.all_rovers and parts[2] in self.all_objectives and parts[3] in self.all_modes:
                    have_image.add((parts[1], parts[2], parts[3]))
            elif match(fact, "full", "*"):
                 if parts[1] in self.all_stores:
                    full_stores.add(parts[1])

        # Calculate cost for each unachieved goal
        for goal_fact in self.goals:
            if goal_fact in state:
                continue # Goal already achieved

            parts = get_parts(goal_fact)
            predicate = parts[0]

            min_goal_cost = math.inf

            if predicate == "communicated_soil_data":
                w = parts[1]
                # Check if sample existed initially (cannot sample if not there at start)
                if w not in self.initial_soil_samples:
                     min_goal_cost = math.inf
                else:
                    # Option 1: Data already collected by some rover
                    for r in self.all_rovers:
                        if (r, w) in have_soil:
                            r_loc = rover_locations.get(r)
                            if r_loc is None: continue # Should not happen in a valid state

                            # Find closest comm waypoint
                            min_comm_dist = math.inf
                            for x in self.comm_wps:
                                if r_loc in self.dist.get(r, {}) and x in self.dist[r].get(r_loc, {}):
                                     min_comm_dist = min(min_comm_dist, self.dist[r][r_loc][x])

                            if min_comm_dist != math.inf:
                                cost = min_comm_dist + 1 # Navigate + Communicate
                                min_goal_cost = min(min_goal_cost, cost)

                    # Option 2: Need to sample and then communicate
                    # Check if sample is still there to be sampled in the current state
                    at_soil_sample_w = f"(at_soil_sample {w})" in state
                    if at_soil_sample_w:
                        for r in self.all_rovers:
                            if 'soil' in self.rover_caps.get(r, set()):
                                r_loc = rover_locations.get(r)
                                if r_loc is None: continue

                                # Cost to sample
                                cost_to_sample_nav = self.dist[r].get(r_loc, {}).get(w, math.inf)
                                if cost_to_sample_nav == math.inf: continue # Cannot reach sample

                                current_cost = cost_to_sample_nav # Navigate to sample
                                # Check store
                                store = self.store_of_rover.get(r)
                                if store and store in full_stores:
                                    current_cost += 1 # Drop

                                current_cost += 1 # Sample

                                # Cost to communicate
                                min_comm_dist = math.inf
                                for x in self.comm_wps:
                                     if w in self.dist.get(r, {}) and x in self.dist[r].get(w, {}):
                                        min_comm_dist = min(min_comm_dist, self.dist[r][w][x])

                                if min_comm_dist != math.inf:
                                    current_cost += min_comm_dist # Navigate to comm
                                    current_cost += 1 # Communicate
                                    min_goal_cost = min(min_goal_cost, current_cost)


            elif predicate == "communicated_rock_data":
                 w = parts[1]
                 # Check if sample existed initially
                 if w not in self.initial_rock_samples:
                     min_goal_cost = math.inf
                 else:
                    # Option 1: Data already collected by some rover
                    for r in self.all_rovers:
                        if (r, w) in have_rock:
                            r_loc = rover_locations.get(r)
                            if r_loc is None: continue
                            min_comm_dist = math.inf
                            for x in self.comm_wps:
                                if r_loc in self.dist.get(r, {}) and x in self.dist[r].get(r_loc, {}):
                                     min_comm_dist = min(min_comm_dist, self.dist[r][r_loc][x])
                            if min_comm_dist != math.inf:
                                cost = min_comm_dist + 1
                                min_goal_cost = min(min_goal_cost, cost)

                    # Option 2: Need to sample and then communicate
                    at_rock_sample_w = f"(at_rock_sample {w})" in state # Check dynamic state
                    if at_rock_sample_w:
                        for r in self.all_rovers:
                            if 'rock' in self.rover_caps.get(r, set()):
                                r_loc = rover_locations.get(r)
                                if r_loc is None: continue

                                cost_to_sample_nav = self.dist[r].get(r_loc, {}).get(w, math.inf)
                                if cost_to_sample_nav == math.inf: continue

                                current_cost = cost_to_sample_nav # Navigate to sample
                                store = self.store_of_rover.get(r)
                                if store and store in full_stores:
                                    current_cost += 1 # Drop

                                current_cost += 1 # Sample

                                min_comm_dist = math.inf
                                for x in self.comm_wps:
                                     if w in self.dist.get(r, {}) and x in self.dist[r].get(w, {}):
                                        min_comm_dist = min(min_comm_dist, self.dist[r][w][x])

                                if min_comm_dist != math.inf:
                                    current_cost += min_comm_dist # Navigate to comm
                                    current_cost += 1 # Communicate
                                    min_goal_cost = min(min_goal_cost, current_cost)


            elif predicate == "communicated_image_data":
                o, m = parts[1], parts[2]

                # Option 1: Image already taken by some rover
                for r in self.all_rovers:
                    if (r, o, m) in have_image:
                        r_loc = rover_locations.get(r)
                        if r_loc is None: continue
                        min_comm_dist = math.inf
                        for x in self.comm_wps:
                            if r_loc in self.dist.get(r, {}) and x in self.dist[r].get(r_loc, {}):
                                 min_comm_dist = min(min_comm_dist, self.dist[r][r_loc][x])
                        if min_comm_dist != math.inf:
                            cost = min_comm_dist + 1
                            min_goal_cost = min(min_goal_cost, cost)

                # Option 2: Need to take image and then communicate
                for r in self.all_rovers:
                    if 'imaging' in self.rover_caps.get(r, set()):
                        r_loc = rover_locations.get(r)
                        if r_loc is None: continue

                        for cam, info in self.camera_info.items():
                            if info.get('rover') == r and m in info.get('modes', set()):
                                cal_target = info.get('cal_target')
                                if not cal_target: continue # Camera has no cal target

                                cal_wps = self.cal_target_visible_from.get(cal_target, set())
                                if not cal_wps: continue # Cal target not visible from anywhere

                                img_wps = self.obj_visible_from.get(o, set())
                                if not img_wps: continue # Objective not visible from anywhere

                                # Iterate through all combinations of cal waypoint and image waypoint
                                for w in cal_wps:
                                    for p in img_wps:
                                        # Cost calculation for this path (r, cam, w, p)
                                        current_cost = 0
                                        current_loc = r_loc

                                        # Calibrate if needed
                                        if (cam, r) not in calibrated_cams:
                                            dist_to_cal = self.dist[r].get(current_loc, {}).get(w, math.inf)
                                            if dist_to_cal == math.inf: continue
                                            current_cost += dist_to_cal + 1 # Navigate + Calibrate
                                            current_loc = w

                                        # Take image
                                        dist_to_img = self.dist[r].get(current_loc, {}).get(p, math.inf)
                                        if dist_to_img == math.inf: continue
                                        current_cost += dist_to_img + 1 # Navigate + Take image
                                        current_loc = p

                                        # Communicate
                                        min_comm_dist = math.inf
                                        for x in self.comm_wps:
                                            if current_loc in self.dist.get(r, {}) and x in self.dist[r].get(current_loc, {}):
                                                min_comm_dist = min(min_comm_dist, self.dist[r][current_loc][x])

                                        if min_comm_dist != math.inf:
                                            current_cost += min_comm_dist + 1 # Navigate + Communicate
                                            min_goal_cost = min(min_goal_cost, current_cost)


            # Add the minimum cost for this goal to the total
            total_h += min_goal_cost

        return total_h
