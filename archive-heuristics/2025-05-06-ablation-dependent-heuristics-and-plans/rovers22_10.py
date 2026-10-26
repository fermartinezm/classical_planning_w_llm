from collections import deque
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper functions (outside the class)
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure the number of parts matches the number of args unless args contains wildcards appropriately
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# The heuristic class
class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve all goal
    conditions. It sums the estimated minimum cost for each unachieved goal.
    The cost for a goal (communicating data) includes the cost to acquire the
    data (sampling or imaging) and the cost to navigate to a communication
    location and communicate. Navigation costs are estimated using shortest
    paths on the traversable graph for each rover.

    # Assumptions
    - Navigation cost between adjacent visible and traversable waypoints is 1.
    - Sampling, dropping, calibrating, taking image, and communicating each cost 1 action.
    - The cost for an unachieved goal is estimated independently of other goals,
      except for reusing data if it already exists.
    - For image goals, calibration is assumed to happen before taking the image
      if needed, and the rover moves sequentially through calibration, image,
      and communication waypoints.
    - The heuristic assumes the existence of necessary objects (equipped rovers,
      cameras, calibration targets, visible waypoints, landers) to achieve goals.
      If a goal is impossible due to missing static requirements, the heuristic
      returns a large value.

    # Heuristic Initialization
    - Extracts static facts about rover equipment, camera properties, waypoint
      visibility, calibration targets, and lander locations.
    - Builds a navigation graph for each rover based on `can_traverse` and
      `visible` static facts.
    - Computes all-pairs shortest paths between waypoints for each rover using BFS.
    - Stores goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize total heuristic cost `h = 0`.
    2. Identify all unachieved goal conditions from `task.goals`.
    3. Precompute navigation distances: In `__init__`, for each rover, build a graph
       where edges are pairs of waypoints (w1, w2) such that `(can_traverse rover w1 w2)`
       and `(visible w1 w2)` are static facts. Use BFS from each waypoint to find
       shortest paths to all other waypoints for that rover. Store these distances.
    4. In `__call__`, for the current state:
       - Determine the current location of each rover.
       - Identify which stores are full.
       - Identify which cameras are calibrated on which rovers.
       - Identify which soil/rock samples exist (`at_soil_sample`, `at_rock_sample`).
       - Identify which soil/rock/image data samples are held by rovers (`have_soil_analysis`, `have_rock_analysis`, `have_image`).
       - Identify which data has been communicated (`communicated_soil_data`, etc.).
       - Find lander location(s) and communication waypoints (visible from lander).
    5. For each unachieved goal `G`:
       - If `G` is `(communicated_soil_data ?w)`:
         - If no rover has `(have_soil_analysis ?r ?w)`:
           - Find the minimum cost over all equipped soil rovers `r`:
             Nav(at `r` -> `?w`) + (1 if `r`'s store is full) + 1 (sample) + Nav(`?w` -> closest comm waypoint for `r`) + 1 (communicate).
           - Add this minimum cost to `h`. Handle unreachable cases (add large value).
         - Else (some rover `r_have` has `(have_soil_analysis ?r_have ?w)`):
           - Find the minimum cost over all such rovers `r_have`:
             Nav(at `r_have` -> closest comm waypoint for `r_have`) + 1 (communicate).
           - Add this minimum cost to `h`. Handle unreachable cases.
       - If `G` is `(communicated_rock_data ?w)`: Similar logic as soil data.
       - If `G` is `(communicated_image_data ?o ?m)`:
         - If no rover has `(have_image ?r ?o ?m)`:
           - Find the minimum cost over all equipped imaging rovers `r` with camera `i` supporting `m`:
             - Find image waypoints for `o` and calibration waypoints for `i`.
             - Cost to get image:
               - If camera `i` on `r` is not calibrated: Min(Nav(at `r` -> cal waypoint) + 1 (calibrate) + Nav(cal waypoint -> image waypoint) + 1 (take_image)) over suitable cal and image waypoints.
               - If camera `i` on `r` is calibrated: Min(Nav(at `r` -> image waypoint) + 1 (take_image)) over suitable image waypoints.
             - Cost to communicate: Min(Nav(image waypoint -> comm waypoint) + 1 (communicate)) over suitable image and comm waypoints.
             - Total cost = Cost to get image + Cost to communicate.
           - Add this minimum total cost to `h`. Handle unreachable cases.
         - Else (some rover `r_have` has `(have_image ?r_have ?o ?m)`):
           - Find the minimum cost over all such rovers `r_have`:
             Nav(at `r_have` -> closest comm waypoint for `r_have`) + 1 (communicate).
           - Add this minimum cost to `h`. Handle unreachable cases.
    6. Return the total heuristic cost `h`.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # Extract static information and infer objects
        self.lander_locations = set()
        self.equipped_soil = set()
        self.equipped_rock = set()
        self.equipped_imaging = set()
        self.store_of = {} # store -> rover
        self.supports = set() # (camera, mode)
        self.visible_wps = set() # (wp1, wp2)
        self.visible_from_obj = {} # objective -> set of waypoints
        self.calibration_target = {} # camera -> objective
        self.on_board = {} # camera -> rover
        self.can_traverse = set() # (rover, wp1, wp2)

        all_rovers = set()
        all_waypoints = set()
        all_cameras = set()
        all_objectives = set()
        all_modes = set()
        all_stores = set()
        all_landers = set()

        for fact in static_facts:
             parts = get_parts(fact)
             if parts[0] == 'at_lander' and len(parts) == 3: all_landers.add(parts[1]); all_waypoints.add(parts[2]); self.lander_locations.add(parts[2])
             elif parts[0] == 'equipped_for_soil_analysis' and len(parts) == 2: all_rovers.add(parts[1]); self.equipped_soil.add(parts[1])
             elif parts[0] == 'equipped_for_rock_analysis' and len(parts) == 2: all_rovers.add(parts[1]); self.equipped_rock.add(parts[1])
             elif parts[0] == 'equipped_for_imaging' and len(parts) == 2: all_rovers.add(parts[1]); self.equipped_imaging.add(parts[1])
             elif parts[0] == 'store_of' and len(parts) == 3: all_stores.add(parts[1]); all_rovers.add(parts[2]); self.store_of[parts[1]] = parts[2]
             elif parts[0] == 'supports' and len(parts) == 3: all_cameras.add(parts[1]); all_modes.add(parts[2]); self.supports.add(tuple(parts[1:]))
             elif parts[0] == 'visible' and len(parts) == 3: all_waypoints.update(parts[1:]); self.visible_wps.add(tuple(parts[1:]))
             elif parts[0] == 'visible_from' and len(parts) == 3: all_objectives.add(parts[1]); all_waypoints.add(parts[2]); self.visible_from_obj.setdefault(parts[1], set()).add(parts[2])
             elif parts[0] == 'calibration_target' and len(parts) == 3: all_cameras.add(parts[1]); all_objectives.add(parts[2]); self.calibration_target[parts[1]] = parts[2]
             elif parts[0] == 'on_board' and len(parts) == 3: all_cameras.add(parts[1]); all_rovers.add(parts[2]); self.on_board[parts[1]] = parts[2]
             elif parts[0] == 'can_traverse' and len(parts) == 4: all_rovers.add(parts[1]); all_waypoints.update(parts[2:]); self.can_traverse.add(tuple(parts[1:]))

        # Add objects from goals if not in static
        for goal in self.goals:
             parts = get_parts(goal)
             if parts[0] == 'communicated_soil_data' and len(parts) == 2: all_waypoints.add(parts[1])
             elif parts[0] == 'communicated_rock_data' and len(parts) == 2: all_waypoints.add(parts[1])
             elif parts[0] == 'communicated_image_data' and len(parts) == 3: all_objectives.add(parts[1]); all_modes.add(parts[2])

        self.all_rovers = list(all_rovers)
        self.all_waypoints = list(all_waypoints)
        self.all_cameras = list(all_cameras)
        self.all_objectives = list(all_objectives)
        self.all_modes = list(all_modes)
        self.all_stores = list(all_stores)
        self.all_landers = list(all_landers)


        # Precompute navigation distances for each rover
        self.dist = {}
        for rover in self.all_rovers:
            self.dist[rover] = self._compute_shortest_paths(rover)

        # Find communication waypoints (visible from any lander)
        self.comm_waypoints = set()
        for lander_pos in self.lander_locations:
             for wp1, wp2 in self.visible_wps:
                 if wp2 == lander_pos:
                     self.comm_waypoints.add(wp1)
                 if wp1 == lander_pos: # visible is often symmetric
                     self.comm_waypoints.add(wp2)


    def _compute_shortest_paths(self, rover):
        """
        Computes shortest path distances from every waypoint to every other
        waypoint for a specific rover using BFS.
        """
        distances = {wp: {other_wp: float('inf') for other_wp in self.all_waypoints} for wp in self.all_waypoints}

        # Build adjacency list for this rover based on can_traverse and visible
        adj = {wp: [] for wp in self.all_waypoints}
        for r, w1, w2 in self.can_traverse:
            if r == rover and (w1, w2) in self.visible_wps:
                 adj[w1].append(w2)

        for start_wp in self.all_waypoints:
            distances[start_wp][start_wp] = 0
            queue = deque([start_wp])
            while queue:
                curr_wp = queue.popleft()
                if curr_wp not in adj: continue # Handle waypoints with no outgoing edges
                for next_wp in adj[curr_wp]:
                    if distances[start_wp][next_wp] == float('inf'):
                        distances[start_wp][next_wp] = distances[start_wp][curr_wp] + 1
                        queue.append(next_wp)
        return distances

    def _get_nav_cost(self, rover, start_wp, end_wp):
        """Helper to get precomputed navigation cost, handling unreachable."""
        if rover not in self.dist or start_wp not in self.dist[rover] or end_wp not in self.dist[rover][start_wp]:
             return float('inf')
        return self.dist[rover][start_wp][end_wp]

    def _get_min_nav_cost_to_set(self, rover, start_wp, target_wps):
        """Helper to get min nav cost from start_wp to any waypoint in target_wps."""
        min_cost = float('inf')
        if rover not in self.dist or start_wp not in self.dist[rover]: return min_cost
        for target_wp in target_wps:
            min_cost = min(min_cost, self.dist[rover][start_wp].get(target_wp, float('inf')))
        return min_cost

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        h = 0
        LARGE_VALUE = 1000 # Use a large value for impossible goals

        # Extract state information
        rover_at = {}
        have_soil = set() # (rover, waypoint)
        have_rock = set() # (rover, waypoint)
        have_image = set() # (rover, objective, mode)
        full_stores = set() # store
        calibrated_cams = set() # (camera, rover)
        communicated_soil = set() # waypoint
        communicated_rock = set() # waypoint
        communicated_image = set() # (objective, mode)
        # at_soil_sample and at_rock_sample are not needed from state for this heuristic logic

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at' and len(parts) == 3 and parts[1] in self.all_rovers: rover_at[parts[1]] = parts[2]
            elif parts[0] == 'have_soil_analysis' and len(parts) == 3: have_soil.add(tuple(parts[1:]))
            elif parts[0] == 'have_rock_analysis' and len(parts) == 3: have_rock.add(tuple(parts[1:]))
            elif parts[0] == 'have_image' and len(parts) == 4: have_image.add(tuple(parts[1:]))
            elif parts[0] == 'full' and len(parts) == 2: full_stores.add(parts[1])
            elif parts[0] == 'calibrated' and len(parts) == 3: calibrated_cams.add(tuple(parts[1:]))
            elif parts[0] == 'communicated_soil_data' and len(parts) == 2: communicated_soil.add(parts[1])
            elif parts[0] == 'communicated_rock_data' and len(parts) == 2: communicated_rock.add(parts[1])
            elif parts[0] == 'communicated_image_data' and len(parts) == 3: communicated_image.add(tuple(parts[1:]))
            # Ignore at_soil_sample/at_rock_sample from state as we only care if data is collected

        # Ensure all rovers have a known location in the state (they should in valid states)
        # If not, the heuristic might be inaccurate for that state, but we proceed.
        # If a rover is missing, nav costs involving it will be infinity.

        # Process each goal
        for goal in self.goals:
            parts = get_parts(goal)
            pred = parts[0]

            if pred == 'communicated_soil_data' and len(parts) == 2:
                wp = parts[1]
                if wp in communicated_soil:
                    continue # Goal already achieved

                min_goal_cost = float('inf')
                rovers_with_data = {r for r, w in have_soil if w == wp}

                if rovers_with_data:
                    # Data exists, just need to communicate
                    for r in rovers_with_data:
                        curr = rover_at.get(r)
                        if curr is None: continue # Rover location unknown
                        nav_cost = self._get_min_nav_cost_to_set(r, curr, self.comm_waypoints)
                        if nav_cost is not float('inf'):
                            min_goal_cost = min(min_goal_cost, nav_cost + 1) # +1 for communicate

                else:
                    # Need to sample and communicate
                    equipped_rovers = self.equipped_soil
                    if not equipped_rovers:
                         min_goal_cost = LARGE_VALUE # Cannot sample if no rover is equipped

                    if min_goal_cost != LARGE_VALUE: # Check if already impossible
                        for r in equipped_rovers:
                            curr = rover_at.get(r)
                            if curr is None: continue

                            # Cost to sample
                            nav_to_sample = self._get_nav_cost(r, curr, wp)
                            if nav_to_sample is float('inf'): continue # Cannot reach sample location

                            store_is_full = any(s in full_stores for s in self.store_of if self.store_of.get(s) == r)
                            drop_cost = 1 if store_is_full else 0

                            cost_get_sample = nav_to_sample + drop_cost + 1 # +1 for sample action

                            # Cost to communicate from sample location
                            nav_sample_to_comm = self._get_min_nav_cost_to_set(r, wp, self.comm_waypoints)
                            if nav_sample_to_comm is float('inf'): continue # Cannot reach comm location from sample location

                            cost_comm = nav_sample_to_comm + 1 # +1 for communicate action

                            min_goal_cost = min(min_goal_cost, cost_get_sample + cost_comm)

                if min_goal_cost is float('inf'):
                    h += LARGE_VALUE # Impossible goal
                else:
                    h += min_goal_cost

            elif pred == 'communicated_rock_data' and len(parts) == 2:
                wp = parts[1]
                if wp in communicated_rock:
                    continue # Goal already achieved

                min_goal_cost = float('inf')
                rovers_with_data = {r for r, w in have_rock if w == wp}

                if rovers_with_data:
                    # Data exists, just need to communicate
                    for r in rovers_with_data:
                        curr = rover_at.get(r)
                        if curr is None: continue
                        nav_cost = self._get_min_nav_cost_to_set(r, curr, self.comm_waypoints)
                        if nav_cost is not float('inf'):
                            min_goal_cost = min(min_goal_cost, nav_cost + 1) # +1 for communicate
                else:
                    # Need to sample and communicate
                    equipped_rovers = self.equipped_rock
                    if not equipped_rovers:
                         min_goal_cost = LARGE_VALUE # Cannot sample if no rover is equipped

                    if min_goal_cost != LARGE_VALUE: # Check if already impossible
                        for r in equipped_rovers:
                            curr = rover_at.get(r)
                            if curr is None: continue

                            # Cost to sample
                            nav_to_sample = self._get_nav_cost(r, curr, wp)
                            if nav_to_sample is float('inf'): continue

                            store_is_full = any(s in full_stores for s in self.store_of if self.store_of.get(s) == r)
                            drop_cost = 1 if store_is_full else 0

                            cost_get_sample = nav_to_sample + drop_cost + 1 # +1 for sample action

                            # Cost to communicate from sample location
                            nav_sample_to_comm = self._get_min_nav_cost_to_set(r, wp, self.comm_waypoints)
                            if nav_sample_to_comm is float('inf'): continue

                            cost_comm = nav_sample_to_comm + 1 # +1 for communicate action

                            min_goal_cost = min(min_goal_cost, cost_get_sample + cost_comm)

                if min_goal_cost is float('inf'):
                    h += LARGE_VALUE # Impossible goal
                else:
                    h += min_goal_cost

            elif pred == 'communicated_image_data' and len(parts) == 3:
                obj, mode = parts[1:]
                if (obj, mode) in communicated_image:
                    continue # Goal already achieved

                min_goal_cost = float('inf')
                rovers_with_image = {r for r, o, m in have_image if o == obj and m == mode}

                if rovers_with_image:
                    # Image exists, just need to communicate
                    for r in rovers_with_image:
                        curr = rover_at.get(r)
                        if curr is None: continue
                        nav_cost = self._get_min_nav_cost_to_set(r, curr, self.comm_waypoints)
                        if nav_cost is not float('inf'):
                            min_goal_cost = min(min_goal_cost, nav_cost + 1) # +1 for communicate
                else:
                    # Need to take image and communicate
                    # Find suitable rovers and cameras
                    suitable_rovers_cams = [] # List of (rover, camera)
                    for r in self.equipped_imaging:
                        for cam, rover_on_board in self.on_board.items():
                            if rover_on_board == r and (cam, mode) in self.supports:
                                suitable_rovers_cams.append((r, cam))

                    if not suitable_rovers_cams:
                        min_goal_cost = LARGE_VALUE # Cannot take image if no suitable rover/camera

                    image_wps = self.visible_from_obj.get(obj, set())
                    if not image_wps:
                        min_goal_cost = LARGE_VALUE # Cannot take image if objective not visible from anywhere

                    if min_goal_cost != LARGE_VALUE: # Check if already impossible
                        for r, cam in suitable_rovers_cams:
                            curr = rover_at.get(r)
                            if curr is None: continue

                            cal_target = self.calibration_target.get(cam)
                            if cal_target is None: continue # Camera has no calibration target
                            cal_wps = self.visible_from_obj.get(cal_target, set())
                            if not cal_wps: continue # Calibration target not visible from anywhere

                            calibrated = (cam, r) in calibrated_cams

                            # Cost to get image
                            cost_get_image = float('inf')
                            if not calibrated:
                                # Need to calibrate first
                                min_nav_cal_then_image = float('inf')
                                for cal_w in cal_wps:
                                    nav_curr_to_cal = self._get_nav_cost(r, curr, cal_w)
                                    if nav_curr_to_cal is float('inf'): continue
                                    min_nav_cal_then_image_from_cal_w = self._get_min_nav_cost_to_set(r, cal_w, image_wps)
                                    if min_nav_cal_then_image_from_cal_w is float('inf'): continue
                                    min_nav_cal_then_image = min(min_nav_cal_then_image, nav_curr_to_cal + min_nav_cal_then_image_from_cal_w)

                                if min_nav_cal_then_image is not float('inf'):
                                     cost_get_image = min(cost_get_image, min_nav_cal_then_image + 1 + 1) # +1 calibrate, +1 take_image
                            else:
                                # Already calibrated, just need to go to image location
                                min_nav_image_only = self._get_min_nav_cost_to_set(r, curr, image_wps)
                                if min_nav_image_only is not float('inf'):
                                    cost_get_image = min(cost_get_image, min_nav_image_only + 1) # +1 take_image

                            if cost_get_image is float('inf'): continue # This rover/camera cannot get the image

                            # Cost to communicate from image location
                            min_nav_image_to_comm = float('inf')
                            for image_w in image_wps:
                                min_nav_image_to_comm_from_image_w = self._get_min_nav_cost_to_set(r, image_w, self.comm_waypoints)
                                if min_nav_image_to_comm_from_image_w is not float('inf'):
                                    min_nav_image_to_comm = min(min_nav_image_to_comm, min_nav_image_to_comm_from_image_w)

                            if min_nav_image_to_comm is float('inf'): continue # Cannot communicate from image locations

                            cost_comm = min_nav_image_to_comm + 1 # +1 communicate action

                            min_goal_cost = min(min_goal_cost, cost_get_image + cost_comm)


                if min_goal_cost is float('inf'):
                    h += LARGE_VALUE # Impossible goal
                else:
                    h += min_goal_cost
            # Ignore other goal types if any

        return h
