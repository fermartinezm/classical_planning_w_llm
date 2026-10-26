from fnmatch import fnmatch
from collections import deque
import sys

# Define a large number to represent infinity for unreachable goals
INF = sys.maxsize

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact strings or malformed facts defensively
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
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
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# Assume Heuristic base class is defined elsewhere, e.g.:
# from heuristics.heuristic_base import Heuristic
# If no base class is provided, the class definition would just be:
# class roversHeuristic:
#     def __init__(self, task):
#         self.goals = task.goals
#         self.static = task.static
#     def __call__(self, node):
#         pass # Implementation below

# Assuming Heuristic base class exists as per example
class Heuristic:
    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static
    def __call__(self, node):
        raise NotImplementedError

class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve all
    goal conditions. It sums the estimated costs for each unachieved goal
    fact. The cost for a goal fact is estimated by finding the minimum
    actions required for any suitable rover to collect the necessary data
    and communicate it to the lander. Movement costs are estimated using
    shortest path distances on the waypoint graph.

    # Assumptions
    - The waypoint graph defined by `can_traverse` predicates is undirected
      and the same for all rovers (simplification).
    - Soil/rock sample collection requires being at the sample location,
      having the equipment, and an empty store. The heuristic simplifies
      store management, primarily counting the 'take' action if the data
      is not already 'have_...'.
    - Image collection requires calibration and taking the image. Calibration
      requires being at a waypoint visible from the camera's calibration target.
      Taking an image requires being at a waypoint visible from the objective.
      The heuristic estimates movement to *any* suitable waypoint for each step.
    - A rover can carry multiple images but only one soil/rock sample at a time
      (implied by store usage). The heuristic simplifies this by just counting
      the 'take' action if the data isn't 'have_...'.
    - Resource contention (e.g., multiple rovers needing the same waypoint or lander)
      is ignored.
    - The cost of each relevant action (move, take_soil_sample, take_rock_sample,
      calibrate_camera, take_image, communicate_data) is assumed to be 1.

    # Heuristic Initialization
    - Parses static facts to build:
        - Waypoint graph based on `can_traverse`.
        - Shortest path distances between all waypoints using BFS.
        - Lander location.
        - Rover capabilities (`equipped_for_...`).
        - Camera information (`on_board`, `supports`, `calibration_target`).
        - Objective visibility (`visible_from`).
    - Stores the goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize total heuristic cost to 0.
    2. Identify the lander's location from static facts.
    3. For each goal fact in the task's goals:
        a. If the goal fact is already true in the current state, continue to the next goal.
        b. If the goal fact is `(communicated_soil_data ?w)`:
            i. Find all rovers equipped for soil analysis.
            ii. If no such rover exists, add a large penalty (infinity) and continue.
            iii. For each capable rover, calculate the cost:
                - Get the rover's current location from the state.
                - If the rover does *not* have `(have_soil_analysis ?r ?w)` in the state:
                    - Cost includes moving from the rover's current location to `?w` (distance lookup) + 1 (take sample action).
                    - The rover is now conceptually at `?w` with the data.
                - If the rover *does* have `(have_soil_analysis ?r ?w)` in the state:
                    - Cost to get data is 0.
                    - The rover is at its current location with the data.
                - Cost includes moving from the data location (or current location if already have data) to the lander location (distance lookup) + 1 (communicate action).
            iv. Find the minimum cost among all capable rovers for this goal.
            v. Add this minimum cost to the total heuristic.
        c. If the goal fact is `(communicated_rock_data ?w)`:
            i. Follow the same logic as for soil data, using rock-specific predicates (`equipped_for_rock_analysis`, `have_rock_analysis`).
        d. If the goal fact is `(communicated_image_data ?o ?m)`:
            i. Find all rovers equipped for imaging that have a camera supporting mode `?m` and calibrating for objective `?o`.
            ii. If no such rover/camera exists, add a large penalty (infinity) and continue.
            iii. For each suitable rover/camera pair, calculate the cost:
                - Get the rover's current location from the state.
                - If the rover does *not* have `(have_image ?r ?o ?m)` in the state:
                    - Initialize current conceptual location to the rover's current location.
                    - If the camera is *not* calibrated for the rover `(calibrated ?c ?r)` in the state:
                        - Find waypoints visible from the camera's calibration target for `?o`.
                        - If none exist, this pair cannot achieve the goal; skip this pair.
                        - Find the nearest such waypoint from the current conceptual location.
                        - Cost includes moving to the nearest waypoint (distance lookup) + 1 (calibrate action).
                        - Update current conceptual location to that nearest waypoint.
                    - Find waypoints visible from objective `?o` (for taking the image).
                    - If none exist, this pair cannot achieve the goal; skip this pair.
                    - Find the nearest such waypoint from the current conceptual location.
                    - Cost includes moving to the nearest waypoint (distance lookup) + 1 (take image action).
                    - Update current conceptual location to that nearest waypoint with the data.
                - If the rover *does* have `(have_image ?r ?o ?m)` in the state:
                    - Cost to get data is 0.
                    - Current conceptual location is the rover's current location.
                - Cost includes moving from the data location (or current location if already have data) to the lander location (distance lookup) + 1 (communicate action).
            iv. Find the minimum cost among all suitable rover/camera pairs for this goal.
            v. Add this minimum cost to the total heuristic.
    4. Return the total heuristic cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information and
        precomputing shortest path distances.
        """
        # Call base class constructor if inheriting
        # super().__init__(task)
        self.goals = task.goals
        static_facts = task.static

        self.lander_location = None
        self.rover_capabilities = {} # rover -> set of {'soil', 'rock', 'imaging'}
        self.camera_info = {} # camera -> {'on_board': rover, 'supports': set(modes), 'calibrates_for': objective}
        self.objective_visibility = {} # objective -> set(waypoints visible from)
        self.waypoint_graph = {} # waypoint -> set(connected_waypoints)
        self.shortest_paths = {} # (w1, w2) -> distance

        all_waypoints = set()
        rovers = set()
        cameras = set()
        objectives = set()
        modes = set()

        # Parse static facts
        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            pred = parts[0]

            if pred == 'at_lander':
                if len(parts) == 3:
                    self.lander_location = parts[2]
                    all_waypoints.add(self.lander_location)
            elif pred == 'equipped_for_soil_analysis':
                if len(parts) == 2:
                    rover = parts[1]
                    rovers.add(rover)
                    self.rover_capabilities.setdefault(rover, set()).add('soil')
            elif pred == 'equipped_for_rock_analysis':
                if len(parts) == 2:
                    rover = parts[1]
                    rovers.add(rover)
                    self.rover_capabilities.setdefault(rover, set()).add('rock')
            elif pred == 'equipped_for_imaging':
                if len(parts) == 2:
                    rover = parts[1]
                    rovers.add(rover)
                    self.rover_capabilities.setdefault(rover, set()).add('imaging')
            elif pred == 'can_traverse':
                if len(parts) == 4:
                    r, w1, w2 = parts[1], parts[2], parts[3]
                    all_waypoints.add(w1)
                    all_waypoints.add(w2)
                    # Build undirected graph assuming symmetry and common paths
                    self.waypoint_graph.setdefault(w1, set()).add(w2)
                    self.waypoint_graph.setdefault(w2, set()).add(w1)
            elif pred == 'on_board':
                if len(parts) == 3:
                    camera, rover = parts[1], parts[2]
                    cameras.add(camera)
                    rovers.add(rover)
                    self.camera_info.setdefault(camera, {})['on_board'] = rover
            elif pred == 'supports':
                if len(parts) == 3:
                    camera, mode = parts[1], parts[2]
                    cameras.add(camera)
                    modes.add(mode)
                    self.camera_info.setdefault(camera, {}).setdefault('supports', set()).add(mode)
            elif pred == 'calibration_target':
                if len(parts) == 3:
                    camera, objective = parts[1], parts[2]
                    cameras.add(camera)
                    objectives.add(objective)
                    self.camera_info.setdefault(camera, {})['calibrates_for'] = objective
            elif pred == 'visible_from':
                if len(parts) == 3:
                    objective, waypoint = parts[1], parts[2]
                    objectives.add(objective)
                    all_waypoints.add(waypoint)
                    self.objective_visibility.setdefault(objective, set()).add(waypoint)
            elif pred in ['at_rock_sample', 'at_soil_sample']:
                 if len(parts) == 2:
                     all_waypoints.add(parts[1])
            # Add other predicates that might introduce waypoints if necessary

        # Ensure all collected waypoints are in the graph dictionary keys
        for w in list(all_waypoints): # Use list to allow modification during iteration if needed, though not here
             self.waypoint_graph.setdefault(w, set())

        # Compute shortest paths using BFS from each waypoint
        for start_w in self.waypoint_graph:
            self.shortest_paths[(start_w, start_w)] = 0
            queue = deque([(start_w, 0)])
            visited = {start_w}

            while queue:
                current_w, dist = queue.popleft()

                for neighbor_w in self.waypoint_graph.get(current_w, set()):
                    if neighbor_w not in visited:
                        visited.add(neighbor_w)
                        self.shortest_paths[(start_w, neighbor_w)] = dist + 1
                        queue.append((neighbor_w, dist + 1))

    def get_distance(self, w1, w2):
        """Lookup precomputed shortest distance between two waypoints."""
        # Return INF if path doesn't exist or waypoints are invalid
        if w1 is None or w2 is None:
            return INF
        return self.shortest_paths.get((w1, w2), INF)

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        total_cost = 0

        # Map current rover locations from the state
        rover_locations = {}
        for fact in state:
            if match(fact, 'at', '*', '*'):
                parts = get_parts(fact)
                if len(parts) == 3 and parts[1].startswith('rover'):
                     rover_locations[parts[1]] = parts[2]

        # Iterate through goals
        for goal in self.goals:
            if goal in state:
                continue # Goal already achieved

            parts = get_parts(goal)
            if not parts: continue # Skip malformed goals

            predicate = parts[0]

            if predicate == 'communicated_soil_data':
                if len(parts) != 2: continue # Malformed goal
                waypoint = parts[1]
                cost = self._cost_for_soil_goal(state, rover_locations, waypoint)
                total_cost += cost

            elif predicate == 'communicated_rock_data':
                if len(parts) != 2: continue # Malformed goal
                waypoint = parts[1]
                cost = self._cost_for_rock_goal(state, rover_locations, waypoint)
                total_cost += cost

            elif predicate == 'communicated_image_data':
                if len(parts) != 3: continue # Malformed goal
                objective, mode = parts[1], parts[2]
                cost = self._cost_for_image_goal(state, rover_locations, objective, mode)
                total_cost += cost
            # Note: Based on examples, only communicated_..._data are goals.
            # If other predicates like have_... could be goals, they would need handling here.

        return total_cost

    def _cost_for_soil_goal(self, state, rover_locations, waypoint):
        """Estimate cost for a single communicated_soil_data goal."""
        lander_w = self.lander_location
        if lander_w is None: return INF # Lander location unknown

        capable_rovers = [r for r, caps in self.rover_capabilities.items() if 'soil' in caps]
        if not capable_rovers: return INF # No rover can do soil analysis

        min_rover_cost = INF

        for rover in capable_rovers:
            r_w = rover_locations.get(rover)
            if r_w is None: continue # Rover location unknown (shouldn't happen in valid state)

            rover_cost = 0
            current_w = r_w # Rover's current location

            have_pred = f'(have_soil_analysis {rover} {waypoint})'

            # Cost to get data if not already have it
            if have_pred not in state:
                dist_to_sample = self.get_distance(current_w, waypoint)
                if dist_to_sample == INF: continue # Cannot reach sample location

                rover_cost += dist_to_sample + 1 # Move to sample + Take sample
                current_w = waypoint # Rover is now conceptually at the sample location

            # Cost to communicate data
            dist_to_lander = self.get_distance(current_w, lander_w)
            if dist_to_lander == INF: continue # Cannot reach lander

            rover_cost += dist_to_lander + 1 # Move to lander + Communicate

            min_rover_cost = min(min_rover_cost, rover_cost)

        return min_rover_cost if min_rover_cost != INF else INF

    def _cost_for_rock_goal(self, state, rover_locations, waypoint):
        """Estimate cost for a single communicated_rock_data goal."""
        lander_w = self.lander_location
        if lander_w is None: return INF # Lander location unknown

        capable_rovers = [r for r, caps in self.rover_capabilities.items() if 'rock' in caps]
        if not capable_rovers: return INF # No rover can do rock analysis

        min_rover_cost = INF

        for rover in capable_rovers:
            r_w = rover_locations.get(rover)
            if r_w is None: continue # Rover location unknown

            rover_cost = 0
            current_w = r_w # Rover's current location

            have_pred = f'(have_rock_analysis {rover} {waypoint})'

            # Cost to get data if not already have it
            if have_pred not in state:
                dist_to_sample = self.get_distance(current_w, waypoint)
                if dist_to_sample == INF: continue # Cannot reach sample location

                rover_cost += dist_to_sample + 1 # Move to sample + Take sample
                current_w = waypoint # Rover is now conceptually at the sample location

            # Cost to communicate data
            dist_to_lander = self.get_distance(current_w, lander_w)
            if dist_to_lander == INF: continue # Cannot reach lander

            rover_cost += dist_to_lander + 1 # Move to lander + Communicate

            min_rover_cost = min(min_rover_cost, rover_cost)

        return min_rover_cost if min_rover_cost != INF else INF

    def _cost_for_image_goal(self, state, rover_locations, objective, mode):
        """Estimate cost for a single communicated_image_data goal."""
        lander_w = self.lander_location
        if lander_w is None: return INF # Lander location unknown

        # Find suitable rover/camera pairs
        suitable_pairs = [] # List of (rover, camera, cal_target) tuples
        for camera, info in self.camera_info.items():
            rover = info.get('on_board')
            supports_mode = mode in info.get('supports', set())
            cal_target = info.get('calibrates_for')
            rover_has_imaging = 'imaging' in self.rover_capabilities.get(rover, set())

            # Need a camera on board an imaging rover, supporting the mode, AND having a calibration target
            if rover and supports_mode and cal_target and rover_has_imaging:
                 suitable_pairs.append((rover, camera, cal_target))

        if not suitable_pairs: return INF # No suitable rover/camera combination

        min_pair_cost = INF

        for rover, camera, cal_target in suitable_pairs:
            r_w = rover_locations.get(rover)
            if r_w is None: continue # Rover location unknown

            pair_cost = 0
            current_w = r_w # Rover's current location

            have_pred = f'(have_image {rover} {objective} {mode})'

            # Cost to get data if not already have it
            if have_pred not in state:
                calibrated_pred = f'(calibrated {camera} {rover})'

                # Cost to calibrate if needed
                if calibrated_pred not in state:
                    # Find waypoints visible from the calibration target
                    cal_target_ws = self.objective_visibility.get(cal_target, set())
                    if not cal_target_ws: continue # No waypoint to calibrate from

                    # Find nearest calibration waypoint
                    min_dist_to_cal = INF
                    best_cal_w = None
                    for w_cal in cal_target_ws:
                        dist = self.get_distance(current_w, w_cal)
                        if dist < min_dist_to_cal:
                            min_dist_to_cal = dist
                            best_cal_w = w_cal

                    if best_cal_w is None or min_dist_to_cal == INF: continue # Cannot reach any calibration waypoint

                    pair_cost += min_dist_to_cal + 1 # Move to cal_w + Calibrate
                    current_w = best_cal_w # Rover is now conceptually at the chosen calibration waypoint
                # Else (already calibrated), current_w remains the same, cost_calibrate is 0.

                # Cost to take image
                # Find waypoints visible from the objective (for taking the image)
                img_ws = self.objective_visibility.get(objective, set())
                if not img_ws: continue # No waypoint to image from

                # Find nearest image waypoint from current location (after potential calibration move)
                min_dist_to_img = INF
                best_img_w = None
                for w_img in img_ws:
                    dist = self.get_distance(current_w, w_img)
                    if dist < min_dist_to_img:
                        min_dist_to_img = dist
                        best_img_w = w_img

                if best_img_w is None or min_dist_to_img == INF: continue # Cannot reach any image waypoint

                pair_cost += min_dist_to_img + 1 # Move to img_w + Take Image
                current_w = best_img_w # Rover is now conceptually at the chosen image waypoint

            # Cost to communicate data
            dist_to_lander = self.get_distance(current_w, lander_w)
            if dist_to_lander == INF: continue # Cannot reach lander

            pair_cost += dist_to_lander + 1 # Move to lander + Communicate

            min_pair_cost = min(min_pair_cost, pair_cost)

        return min_pair_cost if min_pair_cost != INF else INF

