from fnmatch import fnmatch
from collections import deque
import math

# Assume Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential whitespace issues
    return fact.strip()[1:-1].split()

# Helper function to match PDDL facts with patterns
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

# BFS function to compute shortest paths for a specific rover
def bfs_shortest_paths(rover, start_wp, waypoints, visible_map, can_traverse_map):
    """
    Computes shortest path distances from a start waypoint for a specific rover.

    Args:
        rover (str): The name of the rover.
        start_wp (str): The starting waypoint.
        waypoints (set): Set of all waypoint names.
        visible_map (dict): Map from waypoint to set of visible waypoints.
        can_traverse_map (dict): Map from rover to set of traversable (from, to) waypoint pairs.

    Returns:
        dict: A dictionary mapping reachable waypoints to their distance from start_wp.
              Returns float('inf') for unreachable waypoints.
    """
    distances = {wp: math.inf for wp in waypoints}
    if start_wp not in waypoints:
        # Start waypoint might not be in the list if it's a lander location etc.
        # Or if the waypoint list parsing was incomplete.
        # Assuming all locations in the problem are waypoints or lander locations at waypoints.
        # If start_wp is not a known waypoint, no paths are possible.
        return distances

    distances[start_wp] = 0
    queue = deque([start_wp])

    rover_can_traverse = can_traverse_map.get(rover, set())

    while queue:
        current_wp = queue.popleft()

        if current_wp in visible_map:
            for neighbor_wp in visible_map[current_wp]:
                # Check if the rover can traverse this specific edge
                if (current_wp, neighbor_wp) in rover_can_traverse:
                    if distances[neighbor_wp] == math.inf:
                        distances[neighbor_wp] = distances[current_wp] + 1
                        queue.append(neighbor_wp)
    return distances


class roversHeuristic: # Inherit from Heuristic if available
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the minimum number of actions required to achieve
    each unachieved goal fact independently, summing these minimum costs.
    It considers navigation costs (shortest paths), sampling/imaging costs,
    calibration costs, and communication costs.

    # Assumptions
    - The cost of each action (navigate, sample, drop, calibrate, take_image, communicate) is 1.
    - The heuristic calculates the cost for each unachieved goal fact independently,
      ignoring potential negative interactions (like resource contention, store capacity
      beyond needing one drop action, or calibration state changes for multiple images
      with the same camera beyond the first).
    - Navigation costs are precomputed shortest paths for each rover considering its
      traverse capabilities.
    - If a soil/rock sample is needed for a goal, it is assumed to be initially present
      at the waypoint unless already sampled (i.e., the `at_soil_sample` or `at_rock_sample`
      fact is in the initial state or a previous state).
    - If a store is full and a sample is needed, one 'drop' action is assumed to free it.

    # Heuristic Initialization
    - Parses static facts to build:
        - A graph of waypoints and visibility connections.
        - A map of which rovers can traverse which connections.
        - Shortest path distances between all pairs of waypoints for each rover.
        - Lander location and communication waypoints (visible from lander).
        - Rover capabilities (soil, rock, imaging).
        - Rover store mapping.
        - Camera information (on-board rover, supported modes, calibration target).
        - Objective visibility from waypoints.
    - Stores the set of goal facts.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify all goal facts that are not yet true in the current state.
    2. Initialize total heuristic cost `h = 0`.
    3. For each unachieved goal fact:
        a. Estimate the minimum cost to achieve *only* this goal fact from the current state,
           assuming unlimited resources (except for those explicitly checked, like rover equipment,
           camera support, or a single store slot).
        b. Add this minimum estimated cost to `h`.
        c. If any unachieved goal is determined to be impossible based on static facts
           (e.g., no equipped rover, no visible_from waypoint), the heuristic is infinity.

    Detailed cost estimation for each goal type (minimum over applicable rovers/cameras/waypoint combinations):

    - `(communicated_soil_data ?w)`:
        - If `(have_soil_analysis ?r ?w)` is true for some rover `?r`:
            - Cost = `dist(?r_at, comm_wp)` + 1 (communicate). Minimize over `comm_wp`.
        - If `(have_soil_analysis ?r ?w)` is false for all rovers:
            - Find an `equipped_for_soil_analysis` rover `?r`.
            - Check if `(at_soil_sample ?w)` is true in the state. If not, impossible (cost infinity).
            - Cost = `dist(?r_at, ?w)` + (1 if rover's store is full) + 1 (sample) + `dist(?w, comm_wp)` + 1 (communicate). Minimize over `?r` and `comm_wp`.

    - `(communicated_rock_data ?w)`:
        - Similar logic to soil data, using rock-specific predicates and equipment.

    - `(communicated_image_data ?o ?m)`:
        - If `(have_image ?r ?o ?m)` is true for some rover `?r`:
            - Cost = `dist(?r_at, comm_wp)` + 1 (communicate). Minimize over `comm_wp`.
        - If `(have_image ?r ?o ?m)` is false for all rovers:
            - Find an `equipped_for_imaging` rover `?r` with camera `?i` supporting mode `?m`.
            - Find a calibration waypoint `cal_wp` for `?i`'s target `?t` (`visible_from ?t cal_wp`). If none, impossible (cost infinity).
            - Find an image waypoint `img_wp` for `?o` (`visible_from ?o img_wp`). If none, impossible (cost infinity).
            - Check if `(calibrated ?i ?r)` is true in the current state.
            - Calculate cost if camera is currently calibrated: `dist(?r_at, img_wp)` + 1 (take_image) + `dist(img_wp, comm_wp)` + 1 (communicate). Minimize over `img_wp`, `comm_wp`.
            - Calculate cost if camera needs calibration: `dist(?r_at, cal_wp)` + 1 (calibrate) + `dist(cal_wp, img_wp)` + 1 (take_image) + `dist(img_wp, comm_wp)` + 1 (communicate). Minimize over `cal_wp`, `img_wp`, `comm_wp`.
            - The cost for this rover/camera is the minimum of the above two costs.
            - Minimize this cost over all suitable `?r` and `?i`.

    4. Return the total cost `h`. If `h` is infinity, the goal is unreachable in this relaxation.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = set(task.goals)  # Use a set for efficient lookup
        static_facts = set(task.static) # Use a set for efficient lookup

        # --- Precompute Navigation Info ---
        waypoints = set()
        visible_map = {} # {wp: {neighbor_wp, ...}}
        can_traverse_map = {} # {rover: {(from_wp, to_wp), ...}}
        rovers = set()

        # First pass to identify objects and basic relations
        for fact in static_facts:
            parts = get_parts(fact)
            if len(parts) > 1: # Ensure fact has at least predicate and one argument
                if parts[0] == 'waypoint':
                    waypoints.add(parts[1])
                elif parts[0] == 'rover':
                    rovers.add(parts[1])
                elif parts[0] == 'visible' and len(parts) == 3:
                    wp1, wp2 = parts[1], parts[2]
                    visible_map.setdefault(wp1, set()).add(wp2)
                    visible_map.setdefault(wp2, set()).add(wp1) # Visibility is symmetric
                elif parts[0] == 'can_traverse' and len(parts) == 4:
                    r, wp1, wp2 = parts[1], parts[2], parts[3]
                    can_traverse_map.setdefault(r, set()).add((wp1, wp2))

        self.waypoints = waypoints
        self.rovers = rovers
        self.rover_shortest_paths = {}
        for rover in self.rovers:
             self.rover_shortest_paths[rover] = {}
             for start_wp in self.waypoints:
                 self.rover_shortest_paths[rover][start_wp] = bfs_shortest_paths(
                     rover, start_wp, self.waypoints, visible_map, can_traverse_map
                 )

        # --- Precompute Goal-Related Static Info ---
        self.lander_location = None
        self.comm_waypoints = set() # Waypoints visible from lander

        self.rover_capabilities = {} # {rover: {soil: bool, rock: bool, imaging: bool}}
        self.store_to_rover = {} # {store: rover}
        self.camera_info = {} # {camera: {rover: rover, modes: set, cal_target: objective}}
        self.objective_visibility = {} # {objective: {waypoint, ...}}

        for rover in self.rovers:
             self.rover_capabilities[rover] = {'soil': False, 'rock': False, 'imaging': False}

        for fact in static_facts:
            parts = get_parts(fact)
            if len(parts) > 1:
                if parts[0] == 'at_lander' and len(parts) == 3:
                    self.lander_location = parts[2] # (at_lander lander waypoint)
                elif parts[0] == 'equipped_for_soil_analysis' and len(parts) == 2:
                    self.rover_capabilities[parts[1]]['soil'] = True
                elif parts[0] == 'equipped_for_rock_analysis' and len(parts) == 2:
                    self.rover_capabilities[parts[1]]['rock'] = True
                elif parts[0] == 'equipped_for_imaging' and len(parts) == 2:
                    self.rover_capabilities[parts[1]]['imaging'] = True
                elif parts[0] == 'store_of' and len(parts) == 3:
                    self.store_to_rover[parts[1]] = parts[2] # {store: rover}
                elif parts[0] == 'on_board' and len(parts) == 3:
                    camera, rover = parts[1], parts[2]
                    self.camera_info.setdefault(camera, {}).update({'rover': rover, 'modes': set()})
                elif parts[0] == 'supports' and len(parts) == 3:
                    camera, mode = parts[1], parts[2]
                    self.camera_info.setdefault(camera, {}).setdefault('modes', set()).add(mode)
                elif parts[0] == 'calibration_target' and len(parts) == 3:
                    camera, objective = parts[1], parts[2]
                    self.camera_info.setdefault(camera, {}).update({'cal_target': objective})
                elif parts[0] == 'visible_from' and len(parts) == 3:
                    objective, waypoint = parts[1], parts[2]
                    self.objective_visibility.setdefault(objective, set()).add(waypoint)

        # Find communication waypoints
        if self.lander_location and self.lander_location in visible_map:
             # Comm waypoints are those visible *from* the lander's location
             # The predicate is (visible ?w ?p) meaning ?w is visible from ?p
             # So we need waypoints ?x such that (visible ?x lander_location)
             # The visible_map is built symmetrically, so we can check neighbors of lander_location
             self.comm_waypoints = visible_map.get(self.lander_location, set())


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state (frozenset of facts)

        # Check if goal is already reached
        if self.goals.issubset(state):
            return 0

        # Extract relevant dynamic state information
        rover_at = {} # {rover: waypoint}
        have_soil = set() # {(rover, waypoint)}
        have_rock = set() # {(rover, waypoint)}
        have_image = set() # {(rover, objective, mode)}
        store_full = set() # {rover}
        calibrated = set() # {(camera, rover)}
        soil_at = set() # {waypoint}
        rock_at = set() # {waypoint}
        communicated_soil = set() # {waypoint}
        communicated_rock = set() # {waypoint}
        communicated_image = set() # {(objective, mode)}


        for fact in state:
            parts = get_parts(fact)
            if len(parts) > 1:
                if parts[0] == 'at' and len(parts) == 3 and parts[1] in self.rovers:
                    rover_at[parts[1]] = parts[2]
                elif parts[0] == 'have_soil_analysis' and len(parts) == 3:
                    have_soil.add((parts[1], parts[2]))
                elif parts[0] == 'have_rock_analysis' and len(parts) == 3:
                    have_rock.add((parts[1], parts[2]))
                elif parts[0] == 'have_image' and len(parts) == 4:
                    have_image.add((parts[1], parts[2], parts[3]))
                elif parts[0] == 'full' and len(parts) == 2:
                     store = parts[1]
                     if store in self.store_to_rover:
                         rover = self.store_to_rover[store]
                         store_full.add(rover)
                elif parts[0] == 'calibrated' and len(parts) == 3:
                    calibrated.add((parts[1], parts[2]))
                elif parts[0] == 'at_soil_sample' and len(parts) == 2:
                    soil_at.add(parts[1])
                elif parts[0] == 'at_rock_sample' and len(parts) == 2:
                    rock_at.add(parts[1])
                elif parts[0] == 'communicated_soil_data' and len(parts) == 2:
                    communicated_soil.add(parts[1])
                elif parts[0] == 'communicated_rock_data' and len(parts) == 2:
                    communicated_rock.add(parts[1])
                elif parts[0] == 'communicated_image_data' and len(parts) == 3:
                    communicated_image.add((parts[1], parts[2]))


        total_cost = 0

        # Iterate through goals and sum costs for unachieved ones
        for goal in self.goals:
            parts = get_parts(goal)
            goal_type = parts[0]

            if goal_type == 'communicated_soil_data' and len(parts) == 2:
                w = parts[1]
                if w in communicated_soil:
                    continue # Goal already achieved

                best_goal_cost = math.inf

                # Option 1: Find a rover that already has the sample
                for r, soil_w in have_soil:
                    if soil_w == w:
                        current_rover_loc = rover_at.get(r)
                        if current_rover_loc is None or current_rover_loc not in self.waypoints: continue # Rover location unknown or invalid

                        # Cost to communicate = navigate to comm_wp + communicate
                        min_cost_to_comm = math.inf
                        for comm_wp in self.comm_waypoints:
                            dist = self.rover_shortest_paths[r][current_rover_loc].get(comm_wp, math.inf)
                            if dist != math.inf:
                                min_cost_to_comm = min(min_cost_to_comm, dist + 1)
                        best_goal_cost = min(best_goal_cost, min_cost_to_comm)

                # Option 2: Find a rover to sample and communicate
                if best_goal_cost == math.inf: # Only consider sampling if no rover has it yet
                    if w not in soil_at:
                         # Soil sample is gone, goal might be impossible unless already sampled by a rover not in state?
                         # Assuming samples only disappear when sampled by a rover in the state.
                         # If the sample is not at the waypoint and no rover has it, it's impossible in this relaxation.
                         total_cost = math.inf
                         break # Cannot achieve this goal

                    for r in self.rovers:
                        if self.rover_capabilities.get(r, {}).get('soil', False):
                            current_rover_loc = rover_at.get(r)
                            if current_rover_loc is None or current_rover_loc not in self.waypoints: continue # Rover location unknown or invalid

                            dist_to_sample_wp = self.rover_shortest_paths[r][current_rover_loc].get(w, math.inf)

                            if dist_to_sample_wp != math.inf:
                                store_is_full = r in store_full
                                cost_drop = 1 if store_is_full else 0 # Cost to drop if store is full

                                min_cost_from_sample_to_comm = math.inf
                                # Need to navigate from sample waypoint (w) to a communication waypoint
                                if w in self.rover_shortest_paths[r]: # Check if w is a valid start for pathfinding for this rover
                                    for comm_wp in self.comm_waypoints:
                                         dist_to_comm_wp = self.rover_shortest_paths[r][w].get(comm_wp, math.inf)
                                         if dist_to_comm_wp != math.inf:
                                             min_cost_from_sample_to_comm = min(min_cost_from_sample_to_comm, dist_to_comm_wp + 1) # +1 for communicate

                                if min_cost_from_sample_to_comm != math.inf:
                                    cost_to_sample_and_comm = dist_to_sample_wp + cost_drop + 1 + min_cost_from_sample_to_comm # +1 for sample
                                    best_goal_cost = min(best_goal_cost, cost_to_sample_and_comm)

                if best_goal_cost == math.inf:
                    # Cannot find a way to achieve this soil goal
                    total_cost = math.inf
                    break # Cannot achieve this goal
                else:
                    total_cost += best_goal_cost

            elif goal_type == 'communicated_rock_data' and len(parts) == 2:
                w = parts[1]
                if w in communicated_rock:
                    continue # Goal already achieved

                best_goal_cost = math.inf

                # Option 1: Find a rover that already has the sample
                for r, rock_w in have_rock:
                    if rock_w == w:
                        current_rover_loc = rover_at.get(r)
                        if current_rover_loc is None or current_rover_loc not in self.waypoints: continue

                        min_cost_to_comm = math.inf
                        for comm_wp in self.comm_waypoints:
                            dist = self.rover_shortest_paths[r][current_rover_loc].get(comm_wp, math.inf)
                            if dist != math.inf:
                                min_cost_to_comm = min(min_cost_to_comm, dist + 1)
                        best_goal_cost = min(best_goal_cost, min_cost_to_comm)

                # Option 2: Find a rover to sample and communicate
                if best_goal_cost == math.inf: # Only consider sampling if no rover has it yet
                    if w not in rock_at:
                         total_cost = math.inf
                         break # Cannot achieve this goal

                    for r in self.rovers:
                        if self.rover_capabilities.get(r, {}).get('rock', False):
                            current_rover_loc = rover_at.get(r)
                            if current_rover_loc is None or current_rover_loc not in self.waypoints: continue

                            dist_to_sample_wp = self.rover_shortest_paths[r][current_rover_loc].get(w, math.inf)

                            if dist_to_sample_wp != math.inf:
                                store_is_full = r in store_full
                                cost_drop = 1 if store_is_full else 0

                                min_cost_from_sample_to_comm = math.inf
                                if w in self.rover_shortest_paths[r]: # Check if w is a valid start for pathfinding for this rover
                                    for comm_wp in self.comm_waypoints:
                                         dist_to_comm_wp = self.rover_shortest_paths[r][w].get(comm_wp, math.inf)
                                         if dist_to_comm_wp != math.inf:
                                             min_cost_from_sample_to_comm = min(min_cost_from_sample_to_comm, dist_to_comm_wp + 1)

                                if min_cost_from_sample_to_comm != math.inf:
                                    cost_to_sample_and_comm = dist_to_sample_wp + cost_drop + 1 + min_cost_from_sample_to_comm
                                    best_goal_cost = min(best_goal_cost, cost_to_sample_and_comm)

                if best_goal_cost == math.inf:
                    total_cost = math.inf
                    break # Cannot achieve this goal
                else:
                    total_cost += best_goal_cost

            elif goal_type == 'communicated_image_data' and len(parts) == 3:
                o, m = parts[1], parts[2]
                if (o, m) in communicated_image:
                    continue # Goal already achieved

                best_goal_cost = math.inf

                # Option 1: Find a rover that already has the image
                for r, img_o, img_m in have_image:
                    if img_o == o and img_m == m:
                        current_rover_loc = rover_at.get(r)
                        if current_rover_loc is None or current_rover_loc not in self.waypoints: continue

                        min_cost_to_comm = math.inf
                        for comm_wp in self.comm_waypoints:
                            dist = self.rover_shortest_paths[r][current_rover_loc].get(comm_wp, math.inf)
                            if dist != math.inf:
                                min_cost_to_comm = min(min_cost_to_comm, dist + 1)
                        best_goal_cost = min(best_goal_cost, min_cost_to_comm)

                # Option 2: Find a rover/camera to take the image and communicate
                # Only consider taking image if no rover has it yet (best_goal_cost is still inf)
                if best_goal_cost == math.inf:
                    for r in self.rovers:
                        if self.rover_capabilities.get(r, {}).get('imaging', False):
                            current_rover_loc = rover_at.get(r)
                            if current_rover_loc is None or current_rover_loc not in self.waypoints: continue

                            for c, cam_info in self.camera_info.items():
                                if cam_info.get('rover') == r and m in cam_info.get('modes', set()):
                                    cal_target = cam_info.get('cal_target')
                                    if cal_target is None: continue # Camera has no calibration target

                                    cal_wps = self.objective_visibility.get(cal_target, set())
                                    img_wps = self.objective_visibility.get(o, set())

                                    if not cal_wps or not img_wps:
                                        continue # Cannot calibrate or take image for this objective/camera/mode

                                    is_calibrated_now = (c, r) in calibrated

                                    # Calculate cost assuming camera is calibrated now
                                    cost_if_calibrated = math.inf
                                    if is_calibrated_now:
                                        # Cost = navigate_img + take_image + navigate_comm + communicate
                                        min_nav_img = math.inf
                                        for img_wp in img_wps:
                                            min_nav_img = min(min_nav_img, self.rover_shortest_paths[r][current_rover_loc].get(img_wp, math.inf))

                                        min_nav_comm_from_img = math.inf
                                        for img_wp in img_wps:
                                            if img_wp in self.rover_shortest_paths[r]: # Check if img_wp is valid start
                                                for comm_wp in self.comm_waypoints:
                                                     dist = self.rover_shortest_paths[r][img_wp].get(comm_wp, math.inf)
                                                     if dist != math.inf:
                                                         min_nav_comm_from_img = min(min_nav_comm_from_img, dist)

                                        if min_nav_img != math.inf and min_nav_comm_from_img != math.inf:
                                            cost_if_calibrated = min_nav_img + 1 + min_nav_comm_from_img + 1

                                    # Calculate cost assuming camera needs calibration
                                    # Cost = navigate_cal + calibrate + navigate_img + take_image + navigate_comm + communicate
                                    min_nav_cal = math.inf
                                    for cal_wp in cal_wps:
                                        min_nav_cal = min(min_nav_cal, self.rover_shortest_paths[r][current_rover_loc].get(cal_wp, math.inf))

                                    min_nav_img_from_cal = math.inf
                                    for cal_wp in cal_wps:
                                        if cal_wp in self.rover_shortest_paths[r]: # Check if cal_wp is valid start
                                            for img_wp in img_wps:
                                                dist = self.rover_shortest_paths[r][cal_wp].get(img_wp, math.inf)
                                                if dist != math.inf:
                                                    min_nav_img_from_cal = min(min_nav_img_from_cal, dist)

                                    min_nav_comm_from_img_after_cal = math.inf # This is the same as min_nav_comm_from_img above
                                    for img_wp in img_wps:
                                        if img_wp in self.rover_shortest_paths[r]: # Check if img_wp is valid start
                                            for comm_wp in self.comm_waypoints:
                                                dist = self.rover_shortest_paths[r][img_wp].get(comm_wp, math.inf)
                                                if dist != math.inf:
                                                    min_nav_comm_from_img_after_cal = min(min_nav_comm_from_img_after_cal, dist)


                                    cost_if_needs_cal = math.inf
                                    if min_nav_cal != math.inf and min_nav_img_from_cal != math.inf and min_nav_comm_from_img_after_cal != math.inf:
                                         cost_if_needs_cal = min_nav_cal + 1 + min_nav_img_from_cal + 1 + min_nav_comm_from_img_after_cal + 1

                                    # The cost for this rover/camera is the minimum of the two scenarios
                                    min_cost_for_this_rover_camera = min(cost_if_calibrated, cost_if_needs_cal)

                                    best_goal_cost = min(best_goal_cost, min_cost_for_this_rover_camera)

                if best_goal_cost == math.inf:
                    total_cost = math.inf
                    break # Cannot achieve this goal
                else:
                    total_cost += best_goal_cost

            # Add other goal types if any (currently only communication goals)
            # ...

        return total_cost
