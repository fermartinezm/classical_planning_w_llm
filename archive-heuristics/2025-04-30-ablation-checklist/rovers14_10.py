from fnmatch import fnmatch
from collections import deque
import math # Use math.inf for unreachable distances

from heuristics.heuristic_base import Heuristic

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
    # Ensure the number of parts matches the number of args, unless args has a wildcard at the end
    if len(parts) != len(args) and args[-1] != '*':
         return False
    # Check if each part matches the corresponding arg pattern
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def build_waypoint_graph(static_facts):
    """Builds an adjacency list representation of the waypoint graph based on visible predicates."""
    graph = {}
    waypoints = set()
    for fact in static_facts:
        if match(fact, "visible", "*", "*"):
            _, wp1, wp2 = get_parts(fact)
            waypoints.add(wp1)
            waypoints.add(wp2)
            if wp1 not in graph:
                graph[wp1] = set()
            if wp2 not in graph:
                graph[wp2] = set()
            graph[wp1].add(wp2)
            # Assuming visible is symmetric unless specified otherwise, but PDDL usually lists both ways.
            # If not symmetric, remove the line below. The domain file shows symmetric visible.
            graph[wp2].add(wp1)
    return graph, list(waypoints)

def bfs_shortest_paths(graph, start_node, all_nodes):
    """Computes shortest path distances from a start_node to all other nodes using BFS."""
    distances = {node: math.inf for node in all_nodes}
    if start_node not in all_nodes:
        return distances # Start node not in graph

    distances[start_node] = 0
    queue = deque([start_node])

    while queue:
        current_node = queue.popleft()

        if current_node in graph:
            for neighbor in graph[current_node]:
                if distances[neighbor] == math.inf:
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)
    return distances

def precompute_all_pairs_shortest_paths(graph, waypoints):
    """Computes shortest path distances between all pairs of waypoints."""
    all_pairs_dist = {}
    for start_wp in waypoints:
        all_pairs_dist[start_wp] = bfs_shortest_paths(graph, start_wp, waypoints)
    return all_pairs_dist

class RoversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal state
    by summing the estimated costs for each unachieved goal fact. It considers
    action costs (sample, image, calibrate, communicate, drop) and estimated
    navigation costs based on shortest paths between required locations.

    # Heuristic Initialization
    - Precomputes the waypoint graph and all-pairs shortest paths.
    - Extracts static information about rovers, cameras, landers, objectives,
      and their relationships (equipment, stores, visibility, calibration targets).

    # Step-By-Step Thinking for Computing Heuristic
    For each goal fact that is not yet true in the current state:
    1. Add 1 for the final 'communicate' action.
    2. Determine if the required sample or image data is already held by a rover.
    3. If the data is NOT held:
       - Add 1 for the 'sample' or 'take_image' action.
       - If taking an image requires calibration and the camera is not calibrated, add 1 for 'calibrate'.
       - Estimate navigation cost: Find the closest capable rover. Calculate the shortest path distance from its current location to the *first* required waypoint (sample location, calibration target visible waypoint, or objective visible waypoint). Then add the shortest path distance from that first required waypoint (or set of waypoints) to the *next* required waypoint (objective visible waypoint if calibration was first, otherwise lander visible waypoint). Finally, add the shortest path distance from the objective visible waypoint (if applicable) to the lander visible waypoint. Sum these navigation segments.
       - If sampling, add 1 for a 'drop' action if any equipped rover's store is full (a simplification).
       - If any required location is unreachable, the cost for this goal is infinite.
    4. If the data IS held by a rover:
       - Estimate navigation cost: Find the closest rover holding the data. Calculate the shortest path distance from its current location to any lander visible waypoint.
       - If the lander visible waypoint is unreachable, the cost for this goal is infinite.
    5. Sum the costs for all unachieved goals. If any goal has infinite cost, the total heuristic is infinite.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # --- Precompute Static Information ---

        # Waypoint graph and distances
        self.waypoint_graph, self.waypoints = build_waypoint_graph(static_facts)
        self.dist = precompute_all_pairs_shortest_paths(self.waypoint_graph, self.waypoints)

        # Lander locations and lander-visible waypoints
        self.lander_wps = {get_parts(fact)[2] for fact in static_facts if match(fact, "at_lander", "*", "*")}
        self.lander_visible_wps = set()
        for lander_wp in self.lander_wps:
             if lander_wp in self.waypoint_graph: # Ensure lander_wp is a known waypoint
                self.lander_visible_wps.add(lander_wp) # Lander location itself is visible
                if lander_wp in self.waypoint_graph:
                    self.lander_visible_wps.update(self.waypoint_graph[lander_wp]) # Waypoints visible from lander location

        # Objective visible waypoints
        self.obj_visible_wps = {}
        for fact in static_facts:
            if match(fact, "visible_from", "*", "*"):
                _, obj, wp = get_parts(fact)
                if obj not in self.obj_visible_wps:
                    self.obj_visible_wps[obj] = set()
                self.obj_visible_wps[obj].add(wp)

        # Calibration target visible waypoints
        self.cal_visible_wps = {}
        self.camera_cal_target = {}
        for fact in static_facts:
             if match(fact, "calibration_target", "*", "*"):
                 _, camera, target_obj = get_parts(fact)
                 self.camera_cal_target[camera] = target_obj
                 if target_obj in self.obj_visible_wps:
                     self.cal_visible_wps[target_obj] = self.obj_visible_wps[target_obj]
                 else:
                     self.cal_visible_wps[target_obj] = set() # Should not happen in valid problems?

        # Rover capabilities and equipment
        self.equipped_rovers = {'soil': set(), 'rock': set(), 'imaging': set()}
        for fact in static_facts:
            if match(fact, "equipped_for_soil_analysis", "*"):
                self.equipped_rovers['soil'].add(get_parts(fact)[1])
            elif match(fact, "equipped_for_rock_analysis", "*"):
                self.equipped_rovers['rock'].add(get_parts(fact)[1])
            elif match(fact, "equipped_for_imaging", "*"):
                self.equipped_rovers['imaging'].add(get_parts(fact)[1])

        self.rover_stores = {}
        for fact in static_facts:
            if match(fact, "store_of", "*", "*"):
                _, store, rover = get_parts(fact)
                self.rover_stores[rover] = store

        self.rover_cameras = {}
        for fact in static_facts:
            if match(fact, "on_board", "*", "*"):
                _, camera, rover = get_parts(fact)
                if rover not in self.rover_cameras:
                    self.rover_cameras[rover] = set()
                self.rover_cameras[rover].add(camera)

        self.camera_modes = {}
        for fact in static_facts:
            if match(fact, "supports", "*", "*"):
                _, camera, mode = get_parts(fact)
                if camera not in self.camera_modes:
                    self.camera_modes[camera] = set()
                self.camera_modes[camera].add(mode)

        # Initial sample locations (needed to check if a sample still exists)
        self.initial_soil_samples = {get_parts(fact)[1] for fact in static_facts if match(fact, "at_soil_sample", "*")}
        self.initial_rock_samples = {get_parts(fact)[1] for fact in static_facts if match(fact, "at_rock_sample", "*")}


    def min_dist_to_set(self, start_wp, target_wp_set):
        """Calculates minimum distance from start_wp to any waypoint in target_wp_set."""
        if not target_wp_set or start_wp not in self.dist:
            return math.inf # Cannot reach any target if set is empty or start is invalid

        min_d = math.inf
        for target_wp in target_wp_set:
            if target_wp in self.dist[start_wp]:
                 min_d = min(min_d, self.dist[start_wp][target_wp])
        return min_d

    def min_dist_set_to_set(self, start_wp_set, target_wp_set):
        """Calculates minimum distance from any waypoint in start_wp_set to any waypoint in target_wp_set."""
        if not start_wp_set or not target_wp_set:
            return math.inf # Cannot connect if either set is empty

        min_d = math.inf
        for start_wp in start_wp_set:
            min_d = min(min_d, self.min_dist_to_set(start_wp, target_wp_set))
        return min_d

    def closest_wp_in_set(self, start_wp, target_wp_set):
        """Finds the waypoint in target_wp_set closest to start_wp."""
        if not target_wp_set or start_wp not in self.dist:
            return None # Cannot find closest if set is empty or start is invalid

        closest_wp = None
        min_d = math.inf
        for target_wp in target_wp_set:
            if target_wp in self.dist[start_wp]:
                 d = self.dist[start_wp][target_wp]
                 if d < min_d:
                     min_d = d
                     closest_wp = target_wp
        return closest_wp


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # --- Parse Current State ---
        rover_locations = {}
        rover_full_stores = set()
        rover_have_soil = set() # Store as (rover, waypoint)
        rover_have_rock = set() # Store as (rover, waypoint)
        rover_have_image = set() # Store as (rover, objective, mode)
        rover_calibrated_cameras = set() # Store as (rover, camera)
        current_soil_samples = set()
        current_rock_samples = set()

        for fact in state:
            if match(fact, "at", "rover*", "*"):
                _, rover, wp = get_parts(fact)
                rover_locations[rover] = wp
            elif match(fact, "full", "rover*store"):
                 _, store = get_parts(fact)
                 # Find which rover this store belongs to
                 for r, s in self.rover_stores.items():
                     if s == store:
                         rover_full_stores.add(r)
                         break
            elif match(fact, "have_soil_analysis", "rover*", "*"):
                _, rover, wp = get_parts(fact)
                rover_have_soil.add((rover, wp))
            elif match(fact, "have_rock_analysis", "rover*", "*"):
                _, rover, wp = get_parts(fact)
                rover_have_rock.add((rover, wp))
            elif match(fact, "have_image", "rover*", "*", "*"):
                _, rover, obj, mode = get_parts(fact)
                rover_have_image.add((rover, obj, mode))
            elif match(fact, "calibrated", "*", "rover*"):
                _, camera, rover = get_parts(fact)
                rover_calibrated_cameras.add((rover, camera))
            elif match(fact, "at_soil_sample", "*"):
                _, wp = get_parts(fact)
                current_soil_samples.add(wp)
            elif match(fact, "at_rock_sample", "*"):
                _, wp = get_parts(fact)
                current_rock_samples.add(wp)

        total_heuristic_cost = 0

        # --- Estimate Cost for Each Unachieved Goal ---
        for goal in self.goals:
            if goal in state:
                continue # Goal already achieved

            goal_cost = 0
            g_parts = get_parts(goal)
            goal_type = g_parts[0]

            if goal_type == "communicated_soil_data":
                w = g_parts[1]
                goal_cost += 1 # Cost for communicate action

                # Check if sample is already collected by any equipped rover
                have_sample = any((r, w) in rover_have_soil for r in self.equipped_rovers.get('soil', set()))

                if not have_sample:
                    # Need to sample
                    if w not in current_soil_samples:
                        # Sample is gone and no one has it - impossible goal
                        return math.inf
                    goal_cost += 1 # Cost for sample action
                    # Add drop cost if any equipped rover has a full store (simplification)
                    if any(r in rover_full_stores for r in self.equipped_rovers.get('soil', set())):
                         goal_cost += 1 # Cost for drop action

                    # Navigation cost: current -> sample_wp -> lander_wp
                    min_nav_cost = math.inf
                    for r in self.equipped_rovers.get('soil', set()):
                        if r in rover_locations:
                            current_wp = rover_locations[r]
                            cost_to_sample = self.dist[current_wp].get(w, math.inf)
                            if cost_to_sample != math.inf:
                                cost_sample_to_lander = self.min_dist_to_set(w, self.lander_visible_wps)
                                if cost_sample_to_lander != math.inf:
                                    min_nav_cost = min(min_nav_cost, cost_to_sample + cost_sample_to_lander)

                    if min_nav_cost == math.inf:
                        return math.inf # Unreachable sample or lander spot
                    goal_cost += min_nav_cost

                else: # Have sample, need to communicate
                    # Navigation cost: current -> lander_wp
                    min_nav_cost = math.inf
                    for r in self.equipped_rovers.get('soil', set()):
                        if (r, w) in rover_have_soil and r in rover_locations:
                            current_wp = rover_locations[r]
                            cost_to_lander = self.min_dist_to_set(current_wp, self.lander_visible_wps)
                            min_nav_cost = min(min_nav_cost, cost_to_lander)

                    if min_nav_cost == math.inf:
                        return math.inf # Unreachable lander spot
                    goal_cost += min_nav_cost

            elif goal_type == "communicated_rock_data":
                w = g_parts[1]
                goal_cost += 1 # Cost for communicate action

                # Check if sample is already collected by any equipped rover
                have_sample = any((r, w) in rover_have_rock for r in self.equipped_rovers.get('rock', set()))

                if not have_sample:
                    # Need to sample
                    if w not in current_rock_samples:
                         # Sample is gone and no one has it - impossible goal
                         return math.inf
                    goal_cost += 1 # Cost for sample action
                    # Add drop cost if any equipped rover has a full store (simplification)
                    if any(r in rover_full_stores for r in self.equipped_rovers.get('rock', set())):
                         goal_cost += 1 # Cost for drop action

                    # Navigation cost: current -> sample_wp -> lander_wp
                    min_nav_cost = math.inf
                    for r in self.equipped_rovers.get('rock', set()):
                        if r in rover_locations:
                            current_wp = rover_locations[r]
                            cost_to_sample = self.dist[current_wp].get(w, math.inf)
                            if cost_to_sample != math.inf:
                                cost_sample_to_lander = self.min_dist_to_set(w, self.lander_visible_wps)
                                if cost_sample_to_lander != math.inf:
                                    min_nav_cost = min(min_nav_cost, cost_to_sample + cost_sample_to_lander)

                    if min_nav_cost == math.inf:
                        return math.inf # Unreachable sample or lander spot
                    goal_cost += min_nav_cost

                else: # Have sample, need to communicate
                    # Navigation cost: current -> lander_wp
                    min_nav_cost = math.inf
                    for r in self.equipped_rovers.get('rock', set()):
                        if (r, w) in rover_have_rock and r in rover_locations:
                            current_wp = rover_locations[r]
                            cost_to_lander = self.min_dist_to_set(current_wp, self.lander_visible_wps)
                            min_nav_cost = min(min_nav_cost, cost_to_lander)

                    if min_nav_cost == math.inf:
                        return math.inf # Unreachable lander spot
                    goal_cost += min_nav_cost

            elif goal_type == "communicated_image_data":
                o, m = g_parts[1], g_parts[2]
                goal_cost += 1 # Cost for communicate action

                # Check if image is already taken by any suitable rover/camera
                have_image = any((r, o, m) in rover_have_image for r in self.equipped_rovers.get('imaging', set()))

                if not have_image:
                    # Need to take image
                    goal_cost += 1 # Cost for take_image action

                    suitable_pairs = [(r, c) for r in self.equipped_rovers.get('imaging', set())
                                      for c in self.rover_cameras.get(r, set())
                                      if m in self.camera_modes.get(c, set())]

                    if not suitable_pairs:
                        # No equipment for this image goal - impossible
                        return math.inf

                    # Check if calibration is needed for any suitable camera
                    calibration_needed = any((r, c) not in rover_calibrated_cameras for (r, c) in suitable_pairs)

                    min_nav_cost = math.inf

                    for (r, c) in suitable_pairs:
                        if r in rover_locations:
                            current_wp = rover_locations[r]
                            cost = 0
                            path_wps = [current_wp] # Track sequence of locations

                            cal_target = self.camera_cal_target.get(c)
                            cal_wps = self.cal_visible_wps.get(cal_target, set()) if cal_target else set()
                            obj_wps = self.obj_visible_wps.get(o, set())

                            if calibration_needed and cal_target and cal_wps:
                                goal_cost_for_cal = 1 # Cost for calibrate action
                                # Nav cost: current -> cal_wp
                                cost_to_cal = self.min_dist_to_set(path_wps[-1], cal_wps)
                                if cost_to_cal == math.inf:
                                    cost = math.inf # Cannot reach calibration spot
                                else:
                                    cost += cost_to_cal
                                    # Find the specific cal_wp reached
                                    reached_cal_wp = self.closest_wp_in_set(path_wps[-1], cal_wps)
                                    if reached_cal_wp:
                                        path_wps.append(reached_cal_wp)
                                    else: # Should not happen if cost_to_cal is not inf
                                         cost = math.inf

                                if cost != math.inf:
                                    # Nav cost: cal_wp -> obj_wp
                                    cost_cal_to_obj = self.min_dist_to_set(path_wps[-1], obj_wps)
                                    if cost_cal_to_obj == math.inf:
                                        cost = math.inf # Cannot reach objective spot from calibration spot
                                    else:
                                        cost += cost_cal_to_obj
                                        # Find the specific obj_wp reached
                                        reached_obj_wp = self.closest_wp_in_set(path_wps[-1], obj_wps)
                                        if reached_obj_wp:
                                            path_wps.append(reached_obj_wp)
                                        else: # Should not happen if cost_cal_to_obj is not inf
                                            cost = math.inf
                                else:
                                     # If cost is already inf, skip further nav calculation for this pair
                                     pass

                            elif obj_wps: # Calibration not needed or impossible, go straight to objective
                                # Nav cost: current -> obj_wp
                                cost_to_obj = self.min_dist_to_set(path_wps[-1], obj_wps)
                                if cost_to_obj == math.inf:
                                    cost = math.inf # Cannot reach objective spot
                                else:
                                    cost += cost_to_obj
                                    # Find the specific obj_wp reached
                                    reached_obj_wp = self.closest_wp_in_set(path_wps[-1], obj_wps)
                                    if reached_obj_wp:
                                        path_wps.append(reached_obj_wp)
                                    else: # Should not happen if cost_to_obj is not inf
                                        cost = math.inf
                            else:
                                cost = math.inf # Cannot take image without obj_wp

                            if cost != math.inf:
                                # Nav cost: obj_wp -> lander_wp (communication spot)
                                cost_obj_to_lander = self.min_dist_to_set(path_wps[-1], self.lander_visible_wps)
                                if cost_obj_to_lander == math.inf:
                                    cost = math.inf # Cannot reach lander spot
                                else:
                                    cost += cost_obj_to_lander

                            min_nav_cost = min(min_nav_cost, cost)

                    if min_nav_cost == math.inf:
                        return math.inf # Unreachable calibration, objective, or lander spot
                    goal_cost += min_nav_cost

                else: # Have image, need to communicate
                    # Navigation cost: current -> lander_wp
                    min_nav_cost = math.inf
                    for r in self.equipped_rovers.get('imaging', set()):
                        if (r, o, m) in rover_have_image and r in rover_locations:
                            current_wp = rover_locations[r]
                            cost_to_lander = self.min_dist_to_set(current_wp, self.lander_visible_wps)
                            min_nav_cost = min(min_nav_cost, cost_to_lander)

                    if min_nav_cost == math.inf:
                        return math.inf # Unreachable lander spot
                    goal_cost += min_nav_cost

            else:
                # Unknown goal type - should not happen in this domain
                continue

            # If goal_cost is still infinity, it means this goal is impossible from this state
            if goal_cost == math.inf:
                 return math.inf

            total_heuristic_cost += goal_cost

        return total_heuristic_cost

