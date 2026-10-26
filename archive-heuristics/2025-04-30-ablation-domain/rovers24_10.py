from fnmatch import fnmatch
from collections import deque
import math

# Assume Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

# Dummy Heuristic base class (will be replaced by the actual one in the environment)
class Heuristic:
    def __init__(self, task):
        pass
    def __call__(self, node):
        pass

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
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to satisfy all
    goal conditions. It considers the steps needed to acquire data (soil, rock,
    or image) and then communicate it from a lander location. Movement costs
    are estimated using precomputed shortest path distances on the waypoint graph.
    The heuristic sums the minimum estimated cost for each unachieved goal
    independently.

    # Assumptions
    - Each uncommunicated data goal (soil, rock, image) must be achieved.
    - To communicate data, a rover must have the data and be at the lander's location.
    - To get soil/rock data, a rover must be equipped, be at the sample location, and take the sample. Store capacity is ignored.
    - To get image data, a rover must be equipped for imaging, have a camera supporting the mode, be at a waypoint visible from the objective, calibrate the camera, and take the image. Calibration is a separate action needed before taking an image if the camera is not calibrated.
    - Movement costs are shortest path distances between waypoints, assuming any rover can traverse any edge if any rover can.
    - When multiple rovers/waypoints can achieve a sub-goal, the minimum cost path is considered.
    - The heuristic is non-admissible and aims to guide greedy best-first search by prioritizing states closer to satisfying multiple goals or goals requiring less movement/fewer actions.

    # Heuristic Initialization
    - Parse static facts to identify objects (waypoints, rovers, cameras, etc.) and relationships.
    - Build the waypoint graph based on `can_traverse` facts (edges exist if *any* rover can traverse).
    - Compute all-pairs shortest path distances between waypoints using BFS.
    - Identify the lander's static location.
    - Store static capabilities and locations: which rovers are equipped, which cameras are on which rovers, supported modes, calibration targets, objective visibility, and sample locations.
    - Store the set of goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Initialize total heuristic cost `h = 0`.
    2. Parse the current state to find dynamic facts:
       - Current location of each rover.
       - Which rovers have which soil/rock analyses.
       - Which rovers have which images.
       - Which cameras on which rovers are calibrated.
    3. Identify the lander's location (precomputed in `__init__`).
    4. For each goal predicate in the task's goals:
       - If the goal predicate is already true in the current state, add 0 to `h` and continue to the next goal.
       - If the goal is `(communicated_soil_data ?w_sample)`:
         - Calculate the minimum cost to achieve this goal.
         - `min_goal_cost = infinity`.
         - Option A (Already have data): Find rovers `r_have` with `(have_soil_analysis ?r_have ?w_sample)` in the state. For each such rover, calculate cost: `distance(current_location(r_have), lander_location) + 1 (communicate)`. Update `min_goal_cost`.
         - Option B (Need to get data): Find rovers `r_equip` equipped for soil analysis. If no such rover exists and Option A is not possible, the goal is impossible (return infinity). For each `r_equip`, calculate cost: `distance(current_location(r_equip), w_sample) + 1 (take_sample) + distance(w_sample, lander_location) + 1 (communicate)`. Update `min_goal_cost`. This option is only considered if no rover has the data (Option A was not possible or resulted in infinity).
         - Add `min_goal_cost` to `h`. If `min_goal_cost` is still infinity, return infinity.
       - If the goal is `(communicated_rock_data ?w_sample)`:
         - Similar calculation as for soil data, using rock analysis facts and rock-equipped rovers and rock sample locations.
       - If the goal is `(communicated_image_data ?o ?m)`:
         - Calculate the minimum cost to achieve this goal.
         - `min_goal_cost = infinity`.
         - Option A (Already have data): Find rovers `r_have` with `(have_image ?r_have ?o ?m)` in the state. For each such rover, calculate cost: `distance(current_location(r_have), lander_location) + 1 (communicate)`. Update `min_goal_cost`.
         - Option B (Need to get data): Find imaging-equipped rovers `r_equip` with cameras `c` supporting mode `m`. If no such pair exists and Option A is not possible, the goal is impossible (return infinity). For each such `(r_equip, c)` pair, and for each waypoint `w_img` visible from `o`:
           - Calculate cost for this specific path: `distance(current_location(r_equip), w_img) + 1 (take_image) + distance(w_img, lander_location) + 1 (communicate)`.
           - Add calibration cost: If camera `c` on rover `r_equip` is not calibrated in the current state, add the cost of calibrating. This involves finding a calibration waypoint `w_cal` for camera `c` visible from some target, moving from `w_img` to `w_cal`, calibrating (1 action), and moving back to `w_img`. Find the minimum such detour cost. Add this minimum detour cost to the path cost. If calibration is needed but impossible (no cal target/waypoint), this specific `(r_equip, c, w_img)` path is impossible.
           - Update `min_goal_cost` with the minimum path cost found over all suitable `(r_equip, c, w_img)` combinations. This option is only considered if no rover has the image data.
         - Add `min_goal_cost` to `h`. If `min_goal_cost` is still infinity, return infinity.
    5. Return the total heuristic cost `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information and
        precomputing distances.
        """
        self.goals = task.goals
        static_facts = task.static

        self.waypoints = set()
        self.rovers = set()
        self.cameras = set()
        self.objectives = set()
        self.modes = set()
        self.stores = set()
        self.landers = set()

        # Extract objects and static relationships
        self.can_traverse_graph = {} # w1 -> set of w2
        self.lander_location = None
        self.equipped_rovers = {'soil': set(), 'rock': set(), 'imaging': set()}
        self.rover_stores = {} # rover -> store
        self.rover_cameras = {} # rover -> set of cameras
        self.camera_modes = {} # camera -> set of modes
        self.camera_cal_targets = {} # camera -> set of objectives
        self.objective_visible_wps = {} # objective -> set of waypoints
        self.soil_sample_wps = set()
        self.rock_sample_wps = set()

        for fact in static_facts:
            parts = get_parts(fact)
            pred = parts[0]

            if pred == 'at_lander':
                self.landers.add(parts[1])
                self.waypoints.add(parts[2])
                self.lander_location = parts[2] # Assuming only one lander and it's static
            elif pred == 'at': # Initial rover locations are dynamic, but objects are static
                 self.rovers.add(parts[1])
                 self.waypoints.add(parts[2])
            elif pred == 'equipped_for_soil_analysis':
                self.rovers.add(parts[1])
                self.equipped_rovers['soil'].add(parts[1])
            elif pred == 'equipped_for_rock_analysis':
                self.rovers.add(parts[1])
                self.equipped_rovers['rock'].add(parts[1])
            elif pred == 'equipped_for_imaging':
                self.rovers.add(parts[1])
                self.equipped_rovers['imaging'].add(parts[1])
            elif pred == 'store_of':
                self.stores.add(parts[1])
                self.rovers.add(parts[2])
                self.rover_stores[parts[2]] = parts[1] # Assuming one store per rover
            elif pred == 'at_rock_sample':
                self.waypoints.add(parts[1])
                self.rock_sample_wps.add(parts[1])
            elif pred == 'at_soil_sample':
                self.waypoints.add(parts[1])
                self.soil_sample_wps.add(parts[1])
            elif pred == 'visible': # visible implies can_traverse for some rovers? Or use can_traverse directly?
                 self.waypoints.add(parts[1])
                 self.waypoints.add(parts[2])
                 # We will build the graph from can_traverse
            elif pred == 'can_traverse':
                 rover, w1, w2 = parts[1:]
                 self.rovers.add(rover)
                 self.waypoints.add(w1)
                 self.waypoints.add(w2)
                 if w1 not in self.can_traverse_graph:
                     self.can_traverse_graph[w1] = set()
                 if w2 not in self.can_traverse_graph:
                     self.can_traverse_graph[w2] = set()
                 # Add edge if any rover can traverse it
                 self.can_traverse_graph[w1].add(w2)
                 # Assuming can_traverse facts are given for all traversable directions.
            elif pred == 'calibration_target':
                self.cameras.add(parts[1])
                self.objectives.add(parts[2])
                if parts[1] not in self.camera_cal_targets:
                    self.camera_cal_targets[parts[1]] = set()
                self.camera_cal_targets[parts[1]].add(parts[2])
            elif pred == 'on_board':
                self.cameras.add(parts[1])
                self.rovers.add(parts[2])
                if parts[2] not in self.rover_cameras:
                    self.rover_cameras[parts[2]] = set()
                self.rover_cameras[parts[2]].add(parts[1])
            elif pred == 'supports':
                self.cameras.add(parts[1])
                self.modes.add(parts[2])
                if parts[1] not in self.camera_modes:
                    self.camera_modes[parts[1]] = set()
                self.camera_modes[parts[1]].add(parts[2])
            elif pred == 'visible_from':
                self.objectives.add(parts[1])
                self.waypoints.add(parts[2])
                if parts[1] not in self.objective_visible_wps:
                    self.objective_visible_wps[parts[1]] = set()
                self.objective_visible_wps[parts[1]].add(parts[2])
            # Add other object types if needed for parsing
            elif pred in ['colour', 'high_res', 'low_res']:
                 self.modes.add(parts[0])
            elif pred == 'general':
                 self.landers.add(parts[0])
            elif pred.endswith('_res'): # Catch modes like high_res
                 self.modes.add(parts[0])
            elif pred == 'rover':
                 self.rovers.add(parts[0])
            elif pred == 'store':
                 self.stores.add(parts[0])
            elif pred == 'waypoint':
                 self.waypoints.add(parts[0])
            elif pred == 'camera':
                 self.cameras.add(parts[0])
            elif pred == 'objective':
                 self.objectives.add(parts[0])


        # Ensure all waypoints from objects list are in the graph, even if isolated
        for w in self.waypoints:
             if w not in self.can_traverse_graph:
                 self.can_traverse_graph[w] = set()

        # Compute all-pairs shortest paths using BFS from each node
        self.shortest_distances = {}
        for start_wp in self.waypoints:
            q = deque([(start_wp, 0)])
            visited = {start_wp}
            self.shortest_distances[(start_wp, start_wp)] = 0

            while q:
                curr_wp, d = q.popleft()

                for next_wp in self.can_traverse_graph.get(curr_wp, []):
                    if next_wp not in visited:
                        visited.add(next_wp)
                        self.shortest_distances[(start_wp, next_wp)] = d + 1
                        q.append((next_wp, d + 1))

    def get_distance(self, w1, w2):
        """Returns shortest distance between w1 and w2, or infinity if no path."""
        if w1 is None or w2 is None: return math.inf # Handle cases where location is unknown
        return self.shortest_distances.get((w1, w2), math.inf)

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        h = 0
        lander_loc = self.lander_location # Lander location is static

        # Parse dynamic facts from the current state
        rover_locations = {}
        have_soil = set() # (rover, waypoint)
        have_rock = set() # (rover, waypoint)
        have_image = set() # (rover, objective, mode)
        calibrated_cameras = set() # (camera, rover)
        # empty_stores = set() # store # Not needed for this heuristic simplification

        for fact in state:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == 'at':
                rover_locations[parts[1]] = parts[2]
            elif pred == 'have_soil_analysis':
                have_soil.add((parts[1], parts[2]))
            elif pred == 'have_rock_analysis':
                have_rock.add((parts[1], parts[2]))
            elif pred == 'have_image':
                have_image.add((parts[1], parts[2], parts[3]))
            elif pred == 'calibrated':
                calibrated_cameras.add((parts[1], parts[2]))
            # elif pred == 'empty':
            #      empty_stores.add(parts[1])


        # Check each goal
        for goal in self.goals:
            if goal in state:
                continue # Goal already achieved

            parts = get_parts(goal)
            pred = parts[0]

            if pred == 'communicated_soil_data':
                w_sample = parts[1]
                min_goal_cost = math.inf

                # Option A: Find a rover that already has the soil data
                rovers_with_data = {r for (r, w) in have_soil if w == w_sample}
                for r_have in rovers_with_data:
                    if r_have in rover_locations: # Ensure rover location is known
                        cost = self.get_distance(rover_locations[r_have], lander_loc) + 1 # move + communicate
                        min_goal_cost = min(min_goal_cost, cost)

                # Option B: Find an equipped rover to get the data and communicate
                # Only consider Option B if no rover currently has the data
                if not rovers_with_data:
                    equipped_soil_rovers = self.equipped_rovers.get('soil', set())
                    if not equipped_soil_rovers:
                         # No equipped rover and no rover has the data -> Impossible
                         return math.inf

                    # Check if sample exists at the waypoint (static fact)
                    if w_sample not in self.soil_sample_wps:
                         # Sample doesn't exist -> Impossible
                         return math.inf

                    for r_equip in equipped_soil_rovers:
                        if r_equip in rover_locations: # Ensure rover location is known
                            # Add cost for take sample (1) and communicate (1)
                            # Add movement cost: current -> sample -> lander
                            cost = self.get_distance(rover_locations[r_equip], w_sample) + 1 + \
                                   self.get_distance(w_sample, lander_loc) + 1
                            min_goal_cost = min(min_goal_cost, cost)

                if min_goal_cost == math.inf:
                     return math.inf # Still impossible after checking all options

                h += min_goal_cost

            elif pred == 'communicated_rock_data':
                w_sample = parts[1]
                min_goal_cost = math.inf

                # Option A: Find a rover that already has the rock data
                rovers_with_data = {r for (r, w) in have_rock if w == w_sample}
                for r_have in rovers_with_data:
                    if r_have in rover_locations:
                        cost = self.get_distance(rover_locations[r_have], lander_loc) + 1 # move + communicate
                        min_goal_cost = min(min_goal_cost, cost)

                # Option B: Find an equipped rover to get the data and communicate
                # Only consider Option B if no rover currently has the data
                if not rovers_with_data:
                    equipped_rock_rovers = self.equipped_rovers.get('rock', set())
                    if not equipped_rock_rovers:
                         # No equipped rover and no rover has the data -> Impossible
                         return math.inf

                    # Check if sample exists at the waypoint (static fact)
                    if w_sample not in self.rock_sample_wps:
                         # Sample doesn't exist -> Impossible
                         return math.inf

                    for r_equip in equipped_rock_rovers:
                        if r_equip in rover_locations:
                            # Add cost for take sample (1) and communicate (1)
                            # Add movement cost: current -> sample -> lander
                            cost = self.get_distance(rover_locations[r_equip], w_sample) + 1 + \
                                   self.get_distance(w_sample, lander_loc) + 1
                            min_goal_cost = min(min_goal_cost, cost)

                if min_goal_cost == math.inf:
                     return math.inf # Still impossible

                h += min_goal_cost

            elif pred == 'communicated_image_data':
                objective, mode = parts[1:]
                min_goal_cost = math.inf

                # Option A: Find a rover that already has the image
                rovers_with_data = {r for (r, o, m) in have_image if o == objective and m == mode}
                for r_have in rovers_with_data:
                    if r_have in rover_locations:
                        cost = self.get_distance(rover_locations[r_have], lander_loc) + 1 # move + communicate
                        min_goal_cost = min(min_goal_cost, cost)

                # Option B: Find an equipped rover/camera to get the image and communicate
                # Only consider Option B if no rover currently has the image
                if not rovers_with_data:
                    equipped_imaging_rovers = self.equipped_rovers.get('imaging', set())
                    suitable_rover_camera_pairs = []
                    for r_equip in equipped_imaging_rovers:
                        for c in self.rover_cameras.get(r_equip, set()):
                            if mode in self.camera_modes.get(c, set()):
                                suitable_rover_camera_pairs.append((r_equip, c))

                    if not suitable_rover_camera_pairs:
                         # No suitable rover/camera and no rover has the data -> Impossible
                         return math.inf

                    # Find waypoints visible from the objective (static fact)
                    image_wps = self.objective_visible_wps.get(objective, set())
                    if not image_wps:
                         # Cannot take image if objective not visible from anywhere -> Impossible
                         return math.inf

                    min_acquisition_path_cost = math.inf # Min cost to get the image and be at lander loc

                    for r_equip, c in suitable_rover_camera_pairs:
                        if r_equip not in rover_locations: continue # Ensure rover location is known

                        # Find calibration waypoints for this camera (static fact)
                        cal_targets = self.camera_cal_targets.get(c, set())
                        possible_cal_wps = set()
                        for o_cal in cal_targets:
                            possible_cal_wps.update(self.objective_visible_wps.get(o_cal, set()))

                        # If calibration is needed but impossible for this camera, skip this pair
                        needs_calibration = (c, r_equip) not in calibrated_cameras
                        if needs_calibration and not possible_cal_wps:
                             continue

                        for w_img in image_wps:
                            # Cost = move to image_wp + calibrate (if needed) + take image + move to lander + communicate
                            cost_to_img_wp = self.get_distance(rover_locations[r_equip], w_img)
                            if cost_to_img_wp == math.inf: continue # Cannot reach image waypoint

                            cost_from_img_wp_to_lander = self.get_distance(w_img, lander_loc)
                            if cost_from_img_wp_to_lander == math.inf: continue # Cannot reach lander from image waypoint

                            # Actions: take image (1), communicate (1)
                            action_cost = 1 + 1

                            # Calibration cost: If camera is not calibrated
                            calibration_detour_cost = 0
                            if needs_calibration:
                                # Need to move to a calibration waypoint, calibrate, move back to image waypoint
                                min_dist_to_cal_from_img_wp = math.inf
                                best_w_cal = None

                                for w_cal in possible_cal_wps:
                                    dist_to_cal = self.get_distance(w_img, w_cal)
                                    if dist_to_cal < min_dist_to_cal_from_img_wp:
                                        min_dist_to_cal_from_img_wp = dist_to_cal
                                        best_w_cal = w_cal

                                if min_dist_to_cal_from_img_wp == math.inf:
                                     # Cannot reach any calibration waypoint from the image waypoint
                                     continue # Skip this r_equip/c/w_img combination

                                # Calibration requires moving from w_img to best_w_cal, calibrate (1), moving back to w_img
                                calibration_detour_cost = min_dist_to_cal_from_img_wp + 1 + self.get_distance(best_w_cal, w_img)

                            total_path_cost = cost_to_img_wp + calibration_detour_cost + action_cost + cost_from_img_wp_to_lander
                            min_acquisition_path_cost = min(min_acquisition_path_cost, total_path_cost)

                    # Update min_goal_cost with the best acquisition path cost found
                    min_goal_cost = min(min_goal_cost, min_acquisition_path_cost)


                if min_goal_cost == math.inf:
                     return math.inf # Still impossible

                h += min_goal_cost

        # If h is 0, it means all goals are achieved.
        # If the state is a goal state, h must be 0.
        # If the state is not a goal state, h must be > 0 (unless it's unsolvable, then infinity).
        # This logic sums costs for each unachieved goal independently, ensuring h > 0 for non-goal states
        # that are solvable.

        return h
