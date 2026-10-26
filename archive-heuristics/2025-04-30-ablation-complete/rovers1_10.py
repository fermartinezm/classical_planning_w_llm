from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque
import sys

# Helper functions (adapted from provided examples)
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and starts/ends with parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
         # Handle potential non-string or malformed inputs gracefully,
         # though planner states are expected to be strings.
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

# BFS helper for shortest path calculation
def bfs(graph, start_node):
    """Compute shortest path distances from start_node in a graph."""
    distances = {node: float('inf') for node in graph}
    if start_node not in graph:
        # Start node is not a valid waypoint in the graph
        return distances

    distances[start_node] = 0
    queue = deque([start_node])
    while queue:
        current_node = queue.popleft()
        # Handle cases where current_node might not have neighbors in the graph dict
        if current_node in graph:
            for neighbor in graph[current_node]:
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)
    return distances

# Helper to compute all-pairs shortest paths for a specific rover
def compute_rover_distances(rover, waypoints, can_traverse_facts):
    """Compute all-pairs shortest path distances for a specific rover."""
    graph = {wp: [] for wp in waypoints}
    for fact in can_traverse_facts:
        parts = get_parts(fact)
        if len(parts) == 4 and parts[0] == "can_traverse":
             r_obj, from_wp, to_wp = parts[1:]
             if r_obj == rover and from_wp in graph and to_wp in graph: # Ensure waypoints are valid
                 graph[from_wp].append(to_wp)

    all_distances = {}
    for start_wp in waypoints:
        all_distances[start_wp] = bfs(graph, start_wp)
    return all_distances

class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve
    each uncommunicated goal (soil data, rock data, image data).
    The total heuristic value is the sum of the minimum estimated costs
    for each individual unachieved goal.

    # Assumptions
    - Navigation cost is the shortest path distance between waypoints.
    - Sampling (soil/rock) requires being at the sample location with an empty store,
      then moving to a communication location.
    - Imaging requires being at a calibration waypoint, calibrating, moving to an
      image location, taking the image, then moving to a communication location.
    - Communication requires being at a waypoint visible from the lander.
    - Costs for actions (sample, drop, calibrate, take_image, communicate) are 1.
    - The heuristic is non-admissible.

    # Heuristic Initialization
    - Extracts static information: lander location, communication waypoints,
      rover capabilities (soil, rock, imaging), rover stores, camera details
      (on-board, supported modes, calibration target), objective visibility,
      and calibration target visibility.
    - Precomputes all-pairs shortest path distances for each rover based on
      `can_traverse` facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize total heuristic cost to 0.
    2. Identify the current location of each rover.
    3. For each goal fact in the task's goals:
       - If the goal fact is already true in the current state, continue to the next goal.
       - If the goal is `(communicated_soil_data ?w)`:
         - Find the minimum cost to achieve this goal. Consider all rovers equipped for soil analysis.
         - If any such rover already has `(have_soil_analysis ?r ?w)`, the cost is navigation from current location to a communication waypoint + 1 (communicate). Minimize over rovers and communication waypoints.
         - Otherwise, the cost involves navigation from current location to `?w`, potentially dropping a sample (cost 1 if store is full), sampling (cost 1), navigation from `?w` to a communication waypoint, and communicating (cost 1). Minimize over rovers and communication waypoints.
         - Add the minimum cost found for this goal to the total heuristic cost. If no path is possible (e.g., no capable rover, sample doesn't exist), add infinity.
       - If the goal is `(communicated_rock_data ?w)`:
         - Similar logic as for soil data, using rock analysis capabilities and rock samples.
         - Add the minimum cost found for this goal to the total heuristic cost. If no path is possible, add infinity.
       - If the goal is `(communicated_image_data ?o ?m)`:
         - Find the minimum cost to achieve this goal. Consider all rovers equipped for imaging, cameras on board supporting the mode `?m`, waypoints from which `?o` is visible, and waypoints from which the camera's calibration target is visible.
         - If any such rover already has `(have_image ?r ?o ?m)`, the cost is navigation from current location to a communication waypoint + 1 (communicate). Minimize over rovers and communication waypoints.
         - Otherwise, the cost involves navigation from current location to a calibration waypoint `?w_cal`, calibrating (cost 1), navigation from `?w_cal` to an image waypoint `?p` where `?o` is visible, taking the image (cost 1), navigation from `?p` to a communication waypoint, and communicating (cost 1). Minimize over rovers, cameras, calibration waypoints, image waypoints, and communication waypoints.
         - Add the minimum cost found for this goal to the total heuristic cost. If no path is possible (e.g., no capable rover/camera, no visible image/calibration locations), add infinity.
    4. Return the total heuristic cost. If the total cost is infinity (because some goal was unreachable), return a large finite number (sys.maxsize) to avoid issues with planner implementations that might not handle float('inf') correctly in priority queues.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static
        all_facts = task.facts # Includes all possible facts, useful for getting all objects

        # Extract all objects of relevant types from all possible facts
        self.rovers = {get_parts(f)[1] for f in all_facts if match(f, "rover", "*")}
        self.waypoints = {get_parts(f)[1] for f in all_facts if match(f, "waypoint", "*")}
        self.landers = {get_parts(f)[1] for f in all_facts if match(f, "lander", "*")}
        self.stores = {get_parts(f)[1] for f in all_facts if match(f, "store", "*")}
        self.cameras = {get_parts(f)[1] for f in all_facts if match(f, "camera", "*")}
        self.modes = {get_parts(f)[1] for f in all_facts if match(f, "mode", "*")}
        self.objectives = {get_parts(f)[1] for f in all_facts if match(f, "objective", "*")}

        # Extract static information
        self.lander_at = None
        for fact in static_facts:
            if match(fact, "at_lander", "*", "*"):
                parts = get_parts(fact)
                if len(parts) == 3:
                    self.lander_at = parts[2]
                    break # Assuming only one lander

        self.visible_wps = {} # waypoint -> set of visible waypoints
        for fact in static_facts:
             if match(fact, "visible", "*", "*"):
                 parts = get_parts(fact)
                 if len(parts) == 3:
                     wp1, wp2 = parts[1:]
                     self.visible_wps.setdefault(wp1, set()).add(wp2)

        self.comm_wps = self.visible_wps.get(self.lander_at, set()) # Waypoints visible from lander

        self.rover_capabilities = {r: set() for r in self.rovers}
        for fact in static_facts:
            if match(fact, "equipped_for_soil_analysis", "*"):
                parts = get_parts(fact)
                if len(parts) == 2: self.rover_capabilities[parts[1]].add("soil")
            elif match(fact, "equipped_for_rock_analysis", "*"):
                parts = get_parts(fact)
                if len(parts) == 2: self.rover_capabilities[parts[1]].add("rock")
            elif match(fact, "equipped_for_imaging", "*"):
                parts = get_parts(fact)
                if len(parts) == 2: self.rover_capabilities[parts[1]].add("imaging")

        self.rover_stores = {} # rover -> store
        for fact in static_facts:
            if match(fact, "store_of", "*", "*"):
                parts = get_parts(fact)
                if len(parts) == 3:
                    store, rover = parts[1:]
                    self.rover_stores[rover] = store

        self.camera_info = {} # camera -> {'rover': rover, 'modes': set(), 'cal_target': target}
        for cam in self.cameras:
            self.camera_info[cam] = {'rover': None, 'modes': set(), 'cal_target': None}

        for fact in static_facts:
            if match(fact, "on_board", "*", "*"):
                parts = get_parts(fact)
                if len(parts) == 3:
                    cam, rover = parts[1:]
                    if cam in self.camera_info: self.camera_info[cam]['rover'] = rover
            elif match(fact, "supports", "*", "*"):
                parts = get_parts(fact)
                if len(parts) == 3:
                    cam, mode = parts[1:]
                    if cam in self.camera_info: self.camera_info[cam]['modes'].add(mode)
            elif match(fact, "calibration_target", "*", "*"):
                parts = get_parts(fact)
                if len(parts) == 3:
                    cam, target = parts[1:]
                    if cam in self.camera_info: self.camera_info[cam]['cal_target'] = target

        self.objective_visibility = {} # objective -> set of visible waypoints
        for fact in static_facts:
            if match(fact, "visible_from", "*", "*"):
                parts = get_parts(fact)
                if len(parts) == 3:
                    obj, wp = parts[1:]
                    self.objective_visibility.setdefault(obj, set()).add(wp)

        self.cal_target_visibility = {} # cal_target -> set of visible waypoints
        # Calibration targets are objectives, so reuse objective_visibility
        # We need to map cal_target object names to the objective names
        cal_targets_map = {info['cal_target']: cam for cam, info in self.camera_info.items() if info['cal_target']}
        for target_obj in cal_targets_map:
             self.cal_target_visibility[target_obj] = self.objective_visibility.get(target_obj, set())


        # Precompute rover distances
        can_traverse_facts = [f for f in static_facts if match(f, "can_traverse", "*", "*", "*")]
        self.rover_distances = {} # rover -> from_wp -> to_wp -> distance
        for rover in self.rovers:
            self.rover_distances[rover] = compute_rover_distances(rover, self.waypoints, can_traverse_facts)

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        total_cost = 0

        # Get current rover locations
        rover_locations = {}
        for fact in state:
            if match(fact, "at", "*", "*"):
                parts = get_parts(fact)
                if len(parts) == 3:
                    obj, loc = parts[1:]
                    if obj in self.rovers:
                        rover_locations[obj] = loc

        # Get current soil/rock analysis and image data
        have_soil = set() # (rover, waypoint)
        have_rock = set() # (rover, waypoint)
        have_image = set() # (rover, objective, mode)
        calibrated_cams = set() # (camera, rover) # Not strictly needed for this heuristic logic, but good to track
        store_full = set() # store
        soil_samples_at = set() # waypoint
        rock_samples_at = set() # waypoint

        for fact in state:
            if match(fact, "have_soil_analysis", "*", "*"):
                parts = get_parts(fact)
                if len(parts) == 3: have_soil.add(tuple(parts[1:]))
            elif match(fact, "have_rock_analysis", "*", "*"):
                parts = get_parts(fact)
                if len(parts) == 3: have_rock.add(tuple(parts[1:]))
            elif match(fact, "have_image", "*", "*", "*"):
                parts = get_parts(fact)
                if len(parts) == 4: have_image.add(tuple(parts[1:]))
            elif match(fact, "calibrated", "*", "*"):
                 parts = get_parts(fact)
                 if len(parts) == 3: calibrated_cams.add(tuple(parts[1:]))
            elif match(fact, "full", "*"):
                 parts = get_parts(fact)
                 if len(parts) == 2: store_full.add(parts[1])
            elif match(fact, "at_soil_sample", "*"):
                 parts = get_parts(fact)
                 if len(parts) == 2: soil_samples_at.add(parts[1])
            elif match(fact, "at_rock_sample", "*"):
                 parts = get_parts(fact)
                 if len(parts) == 2: rock_samples_at.add(parts[1])


        # Track communicated goals from the goal set
        communicated_soil_goals = {get_parts(g)[1] for g in self.goals if match(g, "communicated_soil_data", "*")}
        communicated_rock_goals = {get_parts(g)[1] for g in self.goals if match(g, "communicated_rock_data", "*")}
        communicated_image_goals = {tuple(get_parts(g)[1:]) for g in self.goals if match(g, "communicated_image_data", "*", "*")}


        # Process each goal type
        # Soil Data Goals
        for goal_w in communicated_soil_goals:
            if f"(communicated_soil_data {goal_w})" not in state:
                min_goal_cost = float('inf')
                # Find a rover that can achieve this goal
                for rover in self.rovers:
                    if "soil" in self.rover_capabilities.get(rover, set()):
                        rover_at = rover_locations.get(rover)
                        if not rover_at: continue # Rover location unknown

                        current_soil_analysis = (rover, goal_w) in have_soil

                        if current_soil_analysis:
                            # Already have the analysis, just need to communicate
                            for comm_wp in self.comm_wps:
                                dist = self.rover_distances[rover].get(rover_at, {}).get(comm_wp, float('inf'))
                                if dist != float('inf'):
                                    cost = dist + 1 # move + communicate
                                    min_goal_cost = min(min_goal_cost, cost)
                        else:
                            # Need to sample and communicate
                            store = self.rover_stores.get(rover)
                            if not store: continue # Rover has no store?

                            store_needs_drop = store in store_full
                            drop_cost = 1 if store_needs_drop else 0

                            # Check if sample exists at waypoint
                            if goal_w in soil_samples_at:
                                for comm_wp in self.comm_wps:
                                    dist1 = self.rover_distances[rover].get(rover_at, {}).get(goal_w, float('inf'))
                                    dist2 = self.rover_distances[rover].get(goal_w, {}).get(comm_wp, float('inf'))
                                    if dist1 != float('inf') and dist2 != float('inf'):
                                        # move_to_sample + drop(if needed) + sample + move_to_comm + communicate
                                        cost = dist1 + drop_cost + 1 + dist2 + 1
                                        min_goal_cost = min(min_goal_cost, cost)

                if min_goal_cost != float('inf'):
                    total_cost += min_goal_cost
                else:
                    # Goal is unreachable by any capable rover
                    return sys.maxsize # Return a large number

        # Rock Data Goals
        for goal_w in communicated_rock_goals:
             if f"(communicated_rock_data {goal_w})" not in state:
                min_goal_cost = float('inf')
                # Find a rover that can achieve this goal
                for rover in self.rovers:
                    if "rock" in self.rover_capabilities.get(rover, set()):
                        rover_at = rover_locations.get(rover)
                        if not rover_at: continue

                        current_rock_analysis = (rover, goal_w) in have_rock

                        if current_rock_analysis:
                            # Already have the analysis, just need to communicate
                            for comm_wp in self.comm_wps:
                                dist = self.rover_distances[rover].get(rover_at, {}).get(comm_wp, float('inf'))
                                if dist != float('inf'):
                                    cost = dist + 1 # move + communicate
                                    min_goal_cost = min(min_goal_cost, cost)
                        else:
                            # Need to sample and communicate
                            store = self.rover_stores.get(rover)
                            if not store: continue

                            store_needs_drop = store in store_full
                            drop_cost = 1 if store_needs_drop else 0

                            # Check if sample exists at waypoint
                            if goal_w in rock_samples_at:
                                for comm_wp in self.comm_wps:
                                    dist1 = self.rover_distances[rover].get(rover_at, {}).get(goal_w, float('inf'))
                                    dist2 = self.rover_distances[rover].get(goal_w, {}).get(comm_wp, float('inf'))
                                    if dist1 != float('inf') and dist2 != float('inf'):
                                        # move_to_sample + drop(if needed) + sample + move_to_comm + communicate
                                        cost = dist1 + drop_cost + 1 + dist2 + 1
                                        min_goal_cost = min(min_goal_cost, cost)

                if min_goal_cost != float('inf'):
                    total_cost += min_goal_cost
                else:
                    # Goal is unreachable
                    return sys.maxsize

        # Image Data Goals
        for goal_o, goal_m in communicated_image_goals:
            if f"(communicated_image_data {goal_o} {goal_m})" not in state:
                min_goal_cost = float('inf')
                # Find a rover/camera/waypoint combination that can achieve this goal
                for rover in self.rovers:
                    if "imaging" in self.rover_capabilities.get(rover, set()):
                        rover_at = rover_locations.get(rover)
                        if not rover_at: continue

                        # Check if rover already has the image
                        current_image = (rover, goal_o, goal_m) in have_image

                        if current_image:
                             # Already have the image, just need to communicate
                            for comm_wp in self.comm_wps:
                                dist = self.rover_distances[rover].get(rover_at, {}).get(comm_wp, float('inf'))
                                if dist != float('inf'):
                                    cost = dist + 1 # move + communicate
                                    min_goal_cost = min(min_goal_cost, cost)
                        else:
                            # Need to take image and communicate
                            # Find cameras on this rover supporting the mode
                            possible_cams = [cam for cam, info in self.camera_info.items()
                                             if info['rover'] == rover and goal_m in info['modes']]

                            for camera in possible_cams:
                                cal_target = self.camera_info[camera]['cal_target']
                                if not cal_target: continue # Camera has no calibration target

                                # Find waypoints visible from the calibration target
                                cal_wps = self.cal_target_visibility.get(cal_target, set())
                                if not cal_wps: continue # Calibration target not visible from anywhere

                                # Find waypoints visible from the objective
                                img_wps = self.objective_visibility.get(goal_o, set())
                                if not img_wps: continue # Objective not visible from anywhere

                                # Find the best sequence: move_to_cal -> calibrate -> move_to_img -> take_image -> move_to_comm -> communicate
                                for cal_wp in cal_wps:
                                    dist_to_cal = self.rover_distances[rover].get(rover_at, {}).get(cal_wp, float('inf'))
                                    if dist_to_cal == float('inf'): continue

                                    for img_wp in img_wps:
                                        dist_cal_to_img = self.rover_distances[rover].get(cal_wp, {}).get(img_wp, float('inf'))
                                        if dist_cal_to_img == float('inf'): continue

                                        for comm_wp in self.comm_wps:
                                            dist_img_to_comm = self.rover_distances[rover].get(img_wp, {}).get(comm_wp, float('inf'))
                                            if dist_img_to_comm == float('inf'): continue

                                            # Cost: move_to_cal + calibrate + move_to_img + take_image + move_to_comm + communicate
                                            cost = dist_to_cal + 1 + dist_cal_to_img + 1 + dist_img_to_comm + 1
                                            min_goal_cost = min(min_goal_cost, cost)

                if min_goal_cost != float('inf'):
                    total_cost += min_goal_cost
                else:
                    # Goal is unreachable
                    return sys.maxsize

        # If all goals are achieved, total_cost will be 0.
        # If any goal was unreachable, we returned sys.maxsize earlier.
        # Otherwise, it's the sum of minimum costs for each unachieved goal.
        return total_cost
