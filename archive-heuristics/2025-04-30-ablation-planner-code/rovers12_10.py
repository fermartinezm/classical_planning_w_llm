from fnmatch import fnmatch
from collections import deque
import math

# Assume Heuristic base class is provided elsewhere and imported as 'Heuristic'
# from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential leading/trailing whitespace or multiple spaces
    return fact.strip()[1:-1].split()

# Helper function to match PDDL facts (optional, but useful)
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

# BFS function to find shortest path distances
def bfs(graph, start_node):
    """
    Performs BFS on a graph to find shortest distances from a start node.
    graph: adjacency dictionary {node: set(neighbor_nodes)}
    start_node: the node to start BFS from
    Returns: dictionary {node: distance}, where distance is math.inf if unreachable
    """
    distances = {node: math.inf for node in graph}
    if start_node not in graph:
         # Start node is not in the graph, no paths possible
         return distances

    distances[start_node] = 0
    queue = deque([start_node])

    while queue:
        current_node = queue.popleft()

        # Ensure current_node is a valid key in the graph
        if current_node in graph:
            for neighbor in graph[current_node]:
                # Ensure neighbor is a valid key in the graph before accessing distances
                if neighbor in graph and distances[neighbor] == math.inf:
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)
    return distances


class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve all unmet goals.
    It considers the core actions (sample, calibrate, take_image, communicate) and
    adds the minimum navigation steps needed to move between the relevant locations
    (current location, sample/calibration/image location, communication location).
    The heuristic sums the estimated costs for each unmet goal independently.

    # Assumptions
    - Rovers have the necessary equipment (soil, rock, imaging) as specified in static facts.
    - Cameras are on board the rovers as specified in static facts.
    - Calibration targets and objectives are visible from waypoints as specified in static facts.
    - Soil and rock samples exist initially at the specified waypoints (if they are goal waypoints).
    - The `can_traverse` predicates define the movement graph for each rover.
    - The heuristic assumes a sequence of actions and navigations for each goal:
        - Soil/Rock: Current -> Sample Loc -> Comm Loc
        - Image (needs calibration): Current -> Cal Loc -> Image Loc -> Comm Loc
        - Image (calibrated): Current -> Image Loc -> Comm Loc
    - The cost of navigation is the shortest path distance (number of `navigate` actions)
      between waypoints for the specific rover.
    - The heuristic sums costs per goal, which may overestimate if actions/navigation
      can contribute to multiple goals.
    - If a required location (sample, image, cal, comm) is unreachable for the relevant rover,
      the heuristic returns infinity for that goal, making the total heuristic infinity.
    - When multiple rovers or cameras can achieve a goal, the heuristic picks the first one found
      for estimating costs.

    # Heuristic Initialization
    - Extracts goal predicates (`communicated_soil_data`, `communicated_rock_data`, `communicated_image_data`).
    - Extracts static facts: lander location, rover capabilities, store info, camera info (on_board, supports, calibration_target), objective visibility, calibration target visibility, waypoint visibility graph, and rover traversal graphs.
    - Precomputes the set of communication waypoints (visible from the lander).
    - Precomputes shortest path distances between all pairs of waypoints for each rover based on its traversal graph using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Initialize total heuristic `h = 0`.
    2. Identify the current location of each rover.
    3. For each goal predicate `g` in the task's goals:
        a. If `g` is already present in the current state, continue to the next goal.
        b. If `g` is `(communicated_soil_data ?w)`:
            i. Add 1 for the `communicate_soil_data` action.
            ii. Check if `(have_soil_analysis ?r ?w)` exists for any rover `?r`.
            iii. If not (sample needed):
                - Add 1 for the `sample_soil` action.
                - Find a rover `?r_cap` equipped for soil analysis (pick first).
                - Find `?r_cap`'s current location `?current_loc`.
                - Calculate navigation cost: `dist(?r_cap, ?current_loc, ?w)` + `min(dist(?r_cap, ?w, ?comm_loc) for ?comm_loc in comm_waypoints)`.
                - Add 1 for `drop` action if `?r_cap`'s store is full in the current state.
                - Add navigation cost to `h`.
            iv. If yes (sample exists for rover `?r_sample`):
                - Find `?r_sample`'s current location `?current_loc`.
                - Calculate navigation cost: `min(dist(?r_sample, ?current_loc, ?comm_loc) for ?comm_loc in comm_waypoints)`.
                - Add navigation cost to `h`.
        c. If `g` is `(communicated_rock_data ?w)`:
            i. Similar logic as soil data, using rock-specific predicates and capabilities.
        d. If `g` is `(communicated_image_data ?o ?m)`:
            i. Add 1 for the `communicate_image_data` action.
            ii. Check if `(have_image ?r ?o ?m)` exists for any rover `?r`.
            iii. If not (take image needed):
                - Add 1 for the `take_image` action.
                - Find a rover `?r_cap` equipped for imaging, with a camera `?i` supporting `?m` (pick first).
                - Find `?r_cap`'s current location `?current_loc`.
                - Check if `(calibrated ?i ?r_cap)` is in the state.
                - If not (calibration needed):
                    - Add 1 for the `calibrate` action.
                    - Find calibration locations `CalLocs` for `?i` and image locations `ImgLocs` for `?o`.
                    - Calculate navigation cost: `min(dist(?r_cap, ?current_loc, ?cal_loc) + dist(?r_cap, ?cal_loc, ?img_loc) + dist(?r_cap, ?img_loc, ?comm_loc))` over all `?cal_loc` in `CalLocs`, `?img_loc` in `ImgLocs`, `?comm_loc` in `comm_waypoints`.
                    - Add navigation cost to `h`.
                iv. If yes (camera calibrated):
                    - Find image locations `ImgLocs` for `?o`.
                    - Calculate navigation cost: `min(dist(?r_cap, ?current_loc, ?img_loc) + dist(?r_cap, ?img_loc, ?comm_loc))` over all `?img_loc` in `ImgLocs`, `?comm_loc` in `comm_waypoints`.
                    - Add navigation cost to `h`.
            v. If yes (image exists for rover `?r_image`):
                - Find `?r_image`'s current location `?current_loc`.
                - Calculate navigation cost: `min(dist(?r_image, ?current_loc, ?comm_loc) for ?comm_loc in comm_waypoints)`.
                - Add navigation cost to `h`.
        e. If any calculated navigation distance is infinity, return infinity immediately (goal unreachable).
    4. Return the total heuristic value `h`.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        # Access problem_pddl from task to parse objects
        problem_pddl_content = task.problem_pddl

        self.goals = task.goals
        static_facts = task.static

        # --- Precompute static information ---

        self.lander_location = None
        self.rover_info = {} # {rover_name: {capabilities: set(), store: store_name, cameras: {cam_name: {supports: set(), calibration_targets: set()}}}}
        self.objective_visibility = {} # {objective_name: set(waypoint_names)}
        self.caltarget_visibility = {} # {caltarget_name: set(waypoint_names)}
        self.waypoint_graph = {} # {waypoint_name: set(neighbor_waypoint_names)}
        self.rover_traversal_graphs = {} # {rover_name: {waypoint_name: set(neighbor_waypoint_names)}}
        self.waypoints = set() # Set of all waypoints

        # Parse objects section to find waypoints and rovers
        objects_section_start = problem_pddl_content.find('(:objects')
        objects_section_end = problem_pddl_content.find(')', objects_section_start)
        objects_content = problem_pddl_content[objects_section_start + len('(:objects'):objects_section_end].strip()

        if objects_content:
            # Split by type separator '-'
            typed_objects = objects_content.split(' - ')
            current_objects = []
            for i, part in enumerate(typed_objects):
                if i == 0:
                    # The first part is just objects before the first type
                    current_objects.extend(part.split())
                else:
                    # The part contains type at the end and objects before it
                    obj_names_str, obj_type = part.rsplit(' ', 1)
                    obj_names = obj_names_str.split()
                    if obj_type == 'waypoint':
                        self.waypoints.update(obj_names)
                    elif obj_type == 'rover':
                         for r_name in obj_names:
                             self.rover_info[r_name] = {'capabilities': set(), 'store': None, 'cameras': {}}
                    # Add other types if needed, but waypoints and rovers are key for heuristic

        # Initialize waypoint graph and traversal graphs with all known waypoints
        for wp in self.waypoints:
            self.waypoint_graph[wp] = set()
            for rover_name in self.rover_info:
                 if rover_name not in self.rover_traversal_graphs:
                      self.rover_traversal_graphs[rover_name] = {}
                 self.rover_traversal_graphs[rover_name][wp] = set()


        # Process static facts
        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]
            args = parts[1:]

            if predicate == 'at_lander':
                # Ensure lander location is a known waypoint
                if len(args) == 2 and args[1] in self.waypoints:
                    self.lander_location = args[1]
            elif predicate == 'equipped_for_soil_analysis':
                if len(args) == 1 and args[0] in self.rover_info:
                    self.rover_info[args[0]]['capabilities'].add('soil')
            elif predicate == 'equipped_for_rock_analysis':
                 if len(args) == 1 and args[0] in self.rover_info:
                    self.rover_info[args[0]]['capabilities'].add('rock')
            elif predicate == 'equipped_for_imaging':
                 if len(args) == 1 and args[0] in self.rover_info:
                    self.rover_info[args[0]]['capabilities'].add('imaging')
            elif predicate == 'store_of':
                 if len(args) == 2 and args[1] in self.rover_info:
                    store_name, rover_name = args
                    self.rover_info[rover_name]['store'] = store_name
            elif predicate == 'on_board':
                 if len(args) == 2 and args[1] in self.rover_info:
                    camera_name, rover_name = args
                    if camera_name not in self.rover_info[rover_name]['cameras']:
                        self.rover_info[rover_name]['cameras'][camera_name] = {'supports': set(), 'calibration_targets': set()}
            elif predicate == 'supports':
                 if len(args) == 2:
                    camera_name, mode_name = args
                    # Find which rover this camera is on
                    for r_name, r_info in self.rover_info.items():
                        if camera_name in r_info['cameras']:
                            r_info['cameras'][camera_name]['supports'].add(mode_name)
                            break
            elif predicate == 'calibration_target':
                 if len(args) == 2:
                    camera_name, target_name = args
                    # Find which rover this camera is on
                    for r_name, r_info in self.rover_info.items():
                        if camera_name in r_info['cameras']:
                            r_info['cameras'][camera_name]['calibration_targets'].add(target_name)
                            break
            elif predicate == 'visible':
                 if len(args) == 2 and args[0] in self.waypoints and args[1] in self.waypoints:
                    wp1, wp2 = args
                    self.waypoint_graph[wp1].add(wp2)
                    self.waypoint_graph[wp2].add(wp1) # Assuming visibility is symmetric
            elif predicate == 'can_traverse':
                 if len(args) == 3 and args[0] in self.rover_info and args[1] in self.waypoints and args[2] in self.waypoints:
                    rover_name, wp1, wp2 = args
                    self.rover_traversal_graphs[rover_name][wp1].add(wp2)
                    self.rover_traversal_graphs[rover_name][wp2].add(wp1) # Assuming traversal is symmetric
            elif predicate == 'visible_from':
                 if len(args) == 2 and args[1] in self.waypoints:
                    obj_name, wp_name = args
                    # Could be objective or calibration target
                    # Check if it's a calibration target for any camera
                    is_cal_target = False
                    for r_name, r_info in self.rover_info.items():
                        for cam_name, cam_info in r_info['cameras'].items():
                            if obj_name in cam_info['calibration_targets']:
                                if obj_name not in self.caltarget_visibility:
                                    self.caltarget_visibility[obj_name] = set()
                                self.caltarget_visibility[obj_name].add(wp_name)
                                is_cal_target = True
                                break
                        if is_cal_target: break

                    # If not a calibration target, assume it's an objective
                    if not is_cal_target:
                        if obj_name not in self.objective_visibility:
                            self.objective_visibility[obj_name] = set()
                        self.objective_visibility[obj_name].add(wp_name)

        # Precompute communication waypoints (visible from lander)
        self.comm_waypoints = set()
        if self.lander_location and self.lander_location in self.waypoint_graph:
             self.comm_waypoints = self.waypoint_graph[self.lander_location]
        # If lander location is not in graph or has no visible neighbors, comm is impossible?
        # Assume lander location is a waypoint and is in the graph if visible facts exist.
        # If no waypoints are visible from lander, comm_waypoints will be empty.

        # Precompute rover distances
        self.rover_distances = {} # {rover_name: {start_wp: {end_wp: distance}}}
        for rover_name, graph in self.rover_traversal_graphs.items():
            self.rover_distances[rover_name] = {}
            for start_wp in self.waypoints:
                 self.rover_distances[rover_name][start_wp] = bfs(graph, start_wp)

        # Store goal predicates for easier access
        self.soil_goals = {get_parts(g)[1] for g in self.goals if match(g, "communicated_soil_data", "*")}
        self.rock_goals = {get_parts(g)[1] for g in self.goals if match(g, "communicated_rock_data", "*")}
        self.image_goals = {(get_parts(g)[1], get_parts(g)[2]) for g in self.goals if match(g, "communicated_image_data", "*", "*")}


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        h = 0

        # Get current rover locations
        current_rover_locations = {}
        for fact in state:
            if fact.startswith('(at '):
                parts = get_parts(fact)
                if len(parts) == 3 and parts[1] in self.rover_info and parts[2] in self.waypoints:
                    current_rover_locations[parts[1]] = parts[2]

        # Handle Soil Goals
        for w in self.soil_goals:
            goal_achieved = f'(communicated_soil_data {w})' in state
            if goal_achieved:
                continue

            # Goal not achieved, estimate cost
            goal_cost = 0
            goal_cost += 1 # communicate_soil_data action

            has_sample = False
            rover_with_sample = None
            for r_name in self.rover_info:
                if f'(have_soil_analysis {r_name} {w})' in state:
                    has_sample = True
                    rover_with_sample = r_name
                    break

            if not has_sample:
                # Need to sample and communicate
                goal_cost += 1 # sample_soil action

                # Find a capable rover
                capable_rovers = [r for r, info in self.rover_info.items() if 'soil' in info['capabilities']]
                if not capable_rovers: return math.inf # Goal impossible

                # Pick the first capable rover for estimation
                rover_to_use = capable_rovers[0]
                current_loc = current_rover_locations.get(rover_to_use)
                if not current_loc: return math.inf # Rover location unknown or not at a waypoint

                # Navigation: Current -> Sample Loc -> Comm Loc
                # Nav 1: Current -> Sample Loc (?w)
                dist_curr_to_sample = self.rover_distances[rover_to_use][current_loc].get(w, math.inf)
                if dist_curr_to_sample == math.inf: return math.inf # Unreachable

                # Nav 2: Sample Loc (?w) -> Comm Loc
                min_dist_sample_to_comm = math.inf
                if not self.comm_waypoints: return math.inf # No communication points
                for comm_loc in self.comm_waypoints:
                    min_dist_sample_to_comm = min(min_dist_sample_to_comm, self.rover_distances[rover_to_use][w].get(comm_loc, math.inf))
                if min_dist_sample_to_comm == math.inf: return math.inf # Comm point unreachable

                goal_cost += dist_curr_to_sample + min_dist_sample_to_comm

                # Check if store is full for the chosen rover
                store_name = self.rover_info[rover_to_use]['store']
                if store_name and f'(full {store_name})' in state:
                    goal_cost += 1 # drop action

            else: # has_sample is True
                # Need to communicate
                current_loc = current_rover_locations.get(rover_with_sample)
                if not current_loc: return math.inf # Rover location unknown or not at a waypoint

                # Navigation: Current -> Comm Loc
                min_dist_curr_to_comm = math.inf
                if not self.comm_waypoints: return math.inf # No communication points
                for comm_loc in self.comm_waypoints:
                    min_dist_curr_to_comm = min(min_dist_curr_to_comm, self.rover_distances[rover_with_sample][current_loc].get(comm_loc, math.inf))
                if min_dist_curr_to_comm == math.inf: return math.inf # Comm point unreachable

                goal_cost += min_dist_curr_to_comm

            h += goal_cost
            if h == math.inf: return math.inf # Propagate infinity

        # Handle Rock Goals
        for w in self.rock_goals:
            goal_achieved = f'(communicated_rock_data {w})' in state
            if goal_achieved:
                continue

            # Goal not achieved, estimate cost
            goal_cost = 0
            goal_cost += 1 # communicate_rock_data action

            has_sample = False
            rover_with_sample = None
            for r_name in self.rover_info:
                if f'(have_rock_analysis {r_name} {w})' in state:
                    has_sample = True
                    rover_with_sample = r_name
                    break

            if not has_sample:
                # Need to sample and communicate
                goal_cost += 1 # sample_rock action

                # Find a capable rover
                capable_rovers = [r for r, info in self.rover_info.items() if 'rock' in info['capabilities']]
                if not capable_rovers: return math.inf # Goal impossible

                # Pick the first capable rover for estimation
                rover_to_use = capable_rovers[0]
                current_loc = current_rover_locations.get(rover_to_use)
                if not current_loc: return math.inf

                # Navigation: Current -> Sample Loc -> Comm Loc
                # Nav 1: Current -> Sample Loc (?w)
                dist_curr_to_sample = self.rover_distances[rover_to_use][current_loc].get(w, math.inf)
                if dist_curr_to_sample == math.inf: return math.inf

                # Nav 2: Sample Loc (?w) -> Comm Loc
                min_dist_sample_to_comm = math.inf
                if not self.comm_waypoints: return math.inf
                for comm_loc in self.comm_waypoints:
                    min_dist_sample_to_comm = min(min_dist_sample_to_comm, self.rover_distances[rover_to_use][w].get(comm_loc, math.inf))
                if min_dist_sample_to_comm == math.inf: return math.inf

                goal_cost += dist_curr_to_sample + min_dist_sample_to_comm

                # Check if store is full for the chosen rover
                store_name = self.rover_info[rover_to_use]['store']
                if store_name and f'(full {store_name})' in state:
                    goal_cost += 1 # drop action

            else: # has_sample is True
                # Need to communicate
                current_loc = current_rover_locations.get(rover_with_sample)
                if not current_loc: return math.inf

                # Navigation: Current -> Comm Loc
                min_dist_curr_to_comm = math.inf
                if not self.comm_waypoints: return math.inf
                for comm_loc in self.comm_waypoints:
                    min_dist_curr_to_comm = min(min_dist_curr_to_comm, self.rover_distances[rover_with_sample][current_loc].get(comm_loc, math.inf))
                if min_dist_curr_to_comm == math.inf: return math.inf

                goal_cost += min_dist_curr_to_comm

            h += goal_cost
            if h == math.inf: return math.inf # Propagate infinity


        # Handle Image Goals
        for o, m in self.image_goals:
            goal_achieved = f'(communicated_image_data {o} {m})' in state
            if goal_achieved:
                continue

            # Goal not achieved, estimate cost
            goal_cost = 0
            goal_cost += 1 # communicate_image_data action

            has_image = False
            rover_with_image = None
            for r_name in self.rover_info:
                 for cam_name, cam_info in self.rover_info[r_name]['cameras'].items():
                      if m in cam_info['supports'] and f'(have_image {r_name} {o} {m})' in state:
                           has_image = True
                           rover_with_image = r_name
                           break
                 if has_image: break


            if not has_image:
                # Need to take image and communicate
                goal_cost += 1 # take_image action

                # Find a capable rover+camera
                capable_rover_cam = None
                for r_name, r_info in self.rover_info.items():
                    if 'imaging' in r_info['capabilities']:
                        for cam_name, cam_info in r_info['cameras'].items():
                            if m in cam_info['supports']:
                                capable_rover_cam = (r_name, cam_name)
                                break
                    if capable_rover_cam: break
                if not capable_rover_cam: return math.inf # Goal impossible

                rover_to_use, camera_to_use = capable_rover_cam
                current_loc = current_rover_locations.get(rover_to_use)
                if not current_loc: return math.inf

                needs_calibration = f'(calibrated {camera_to_use} {rover_to_use})' not in state

                # Find relevant locations
                img_locs = self.objective_visibility.get(o, set())
                if not img_locs: return math.inf # Objective not visible from anywhere

                comm_locs = self.comm_waypoints
                if not comm_locs: return math.inf # No communication points

                if needs_calibration:
                    goal_cost += 1 # calibrate action

                    cal_targets = self.rover_info[rover_to_use]['cameras'][camera_to_use]['calibration_targets']
                    cal_locs = set()
                    for target in cal_targets:
                        cal_locs.update(self.caltarget_visibility.get(target, set()))
                    if not cal_locs: return math.inf # No visible calibration target

                    # Navigation: Current -> Cal Loc -> Image Loc -> Comm Loc
                    min_total_nav = math.inf
                    for cal_loc in cal_locs:
                        dist_curr_to_cal = self.rover_distances[rover_to_use][current_loc].get(cal_loc, math.inf)
                        if dist_curr_to_cal == math.inf: continue

                        for img_loc in img_locs:
                            dist_cal_to_img = self.rover_distances[rover_to_use][cal_loc].get(img_loc, math.inf)
                            if dist_cal_to_img == math.inf: continue

                            for comm_loc in comm_locs:
                                dist_img_to_comm = self.rover_distances[rover_to_use][img_loc].get(comm_loc, math.inf)
                                if dist_img_to_comm == math.inf: continue

                                min_total_nav = min(min_total_nav, dist_curr_to_cal + dist_cal_to_img + dist_img_to_comm)

                    if min_total_nav == math.inf: return math.inf # Path unreachable
                    goal_cost += min_total_nav

                else: # Camera is calibrated
                    # Navigation: Current -> Image Loc -> Comm Loc
                    min_total_nav = math.inf
                    for img_loc in img_locs:
                        dist_curr_to_img = self.rover_distances[rover_to_use][current_loc].get(img_loc, math.inf)
                        if dist_curr_to_img == math.inf: continue

                        for comm_loc in comm_locs:
                            dist_img_to_comm = self.rover_distances[rover_to_use][img_loc].get(comm_loc, math.inf)
                            if dist_img_to_comm == math.inf: continue

                            min_total_nav = min(min_total_nav, dist_curr_to_img + dist_img_to_comm)

                    if min_total_nav == math.inf: return math.inf # Path unreachable
                    goal_cost += min_total_nav

            else: # has_image is True
                # Need to communicate
                current_loc = current_rover_locations.get(rover_with_image)
                if not current_loc: return math.inf

                # Navigation: Current -> Comm Loc
                min_dist_curr_to_comm = math.inf
                if not self.comm_waypoints: return math.inf
                for comm_loc in self.comm_waypoints:
                    min_dist_curr_to_comm = min(min_dist_curr_to_comm, self.rover_distances[rover_with_image][current_loc].get(comm_loc, math.inf))
                if min_dist_curr_to_comm == math.inf: return math.inf

                goal_cost += min_dist_curr_to_comm

            h += goal_cost
            if h == math.inf: return math.inf # Propagate infinity


        # The heuristic is 0 if and only if all goals are in the state.
        # If h is finite and non-zero, it means there are unmet goals.
        # If h is infinity, it means at least one unmet goal is unreachable.
        # This check is implicitly handled by the loop structure and returning math.inf early.
        # If the loop finishes and h is 0, all goals were in the state.

        return h
