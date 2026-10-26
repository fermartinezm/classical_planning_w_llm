from fnmatch import fnmatch
from collections import deque
import sys

# Assume Heuristic base class is available in the environment, e.g.:
# class Heuristic:
#     def __init__(self, task):
#         pass
#     def __call__(self, node):
#         raise NotImplementedError


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact string or malformed fact
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
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


def bfs(graph, start):
    """Compute shortest distances from start to all reachable nodes in a graph."""
    distances = {start: 0}
    queue = deque([start])
    while queue:
        current = queue.popleft()
        # Use graph.get(current, []) to handle nodes with no outgoing edges
        for neighbor in graph.get(current, []):
            if neighbor not in distances:
                distances[neighbor] = distances[current] + 1
                queue.append(neighbor)
    return distances


class roversHeuristic: # Inherit from Heuristic if available in the environment
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve all
    unmet goal conditions related to communicating data (soil, rock, images).
    It sums the estimated costs for each individual unachieved goal,
    considering the steps needed to collect the data (navigate, take sample/image,
    calibrate camera if needed) and communicate it (navigate to lander, communicate).
    Navigation costs are estimated using shortest paths on the waypoint graph.

    # Assumptions
    - Each sample goal requires navigating to the sample location, taking the sample,
      navigating to the lander, and communicating. A rover can only hold one sample
      at a time; if its store is full with a different sample, it must communicate
      that sample first (penalty involves navigation to lander and communication).
    - Each image goal requires navigating to a calibration waypoint (if not calibrated),
      calibrating, navigating to an imaging waypoint, taking the image, navigating
      to the lander, and communicating. Images do not fill the sample store.
    - Navigation cost between adjacent waypoints is 1. Shortest paths are precomputed.
    - Necessary resources (equipped rovers, cameras, visible waypoints) exist for
      solvable problems.
    - The heuristic sums costs for individual goals, ignoring potential synergies
      (e.g., collecting multiple samples/images on one trip, or one trip to lander
       for multiple communication tasks). This might overestimate but is fast.
    - Assumes a single lander location.
    - Assumes 'visible' or 'can_traverse' predicates define the traversable graph
      for navigation cost estimation. The heuristic uses both, treating the graph
      as undirected based on examples.

    # Heuristic Initialization
    - Parses static facts to identify:
        - Lander location.
        - Rover capabilities (soil, rock, imaging).
        - Rover store mapping.
        - Camera details (on-board rover, supported modes, calibration target).
        - Objective visibility from waypoints.
        - Waypoint graph connectivity (using 'visible' and 'can_traverse' predicates).
    - Precomputes all-pairs shortest paths between waypoints using BFS.
    - Stores goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Initialize total_cost to 0.
    2. Get the current state and the lander's location (precomputed during initialization).
    3. Check if the current state is a goal state. If yes, return 0.
    4. For each goal predicate in the task's goals:
        a. If the goal predicate is already true in the current state, continue to the next goal.
        b. If the goal is `(communicated_soil_data ?w_sample)`:
            i. Find a rover `?r` equipped for soil analysis. Use the first one found for simplicity.
            ii. If no equipped rover is found, this goal is impossible (in solvable problems, this shouldn't happen). Add infinity cost.
            iii. Get `?r`'s current location `?w_current`. If location is unknown, add infinity cost.
            iv. Check if `?r` currently holds `(have_soil_sample ?r ?w_sample)`.
            v. If yes (rover has the sample):
                - Estimate cost: 1 (communicate) + shortest_path_distance(`?w_current`, lander_loc).
                - Add this cost to total_cost.
            vi. If no rover has the sample:
                - Calculate base cost: shortest_path_distance(`?w_current`, `?w_sample`) + 1 (take sample) + shortest_path_distance(`?w_sample`, lander_loc) + 1 (communicate).
                - Check if `?r`'s store is full with a *different* sample. If `(full ?s)` is true for `?r`'s store `?s`, and `(have_soil_sample ?r ?w')` or `(have_rock_sample ?r ?w')` for any `?w'` is true in the state (and it's not the sample `?w_sample`), add a penalty to the base cost. The penalty is estimated as the cost to clear the store: shortest_path_distance(`?w_current`, lander_loc) + 2 (communicate old sample + empty store - approximation).
                - Add the calculated cost (base + penalty) to total_cost.
        c. If the goal is `(communicated_rock_data ?w_sample)`:
            i. Follow similar logic as for soil data, using rock-specific predicates and capabilities.
        d. If the goal is `(communicated_image_data ?o ?m)`:
            i. Find a rover `?r` equipped for imaging with a camera `?c` supporting mode `?m` and calibrated for objective `?o`. Use the first suitable one found.
            ii. If no suitable rover/camera is found, this goal is impossible. Add infinity cost.
            iii. Get `?r`'s current location `?w_current`. If location is unknown, add infinity cost.
            iv. Check if `?r` currently holds `(have_image ?r ?o ?m)`.
            v. If yes (rover has the image):
                - Estimate cost: 1 (communicate) + shortest_path_distance(`?w_current`, lander_loc).
                - Add this cost to total_cost.
            vi. If no rover has the image:
                - Initialize step_cost = 0.
                - Initialize current_nav_loc = `?w_current`.
                - If camera `?c` is not `(calibrated ?c ?r)` in the state:
                    - Find the closest waypoint `?w_cal` from `current_nav_loc` where `?o` is visible.
                    - If no such waypoint exists or is reachable, this path is impossible. Add infinity cost.
                    - Otherwise, add shortest_path_distance(`current_nav_loc`, `?w_cal`) + 1 (calibrate) to step_cost.
                    - Update `current_nav_loc` = `?w_cal`.
                - Find the closest waypoint `?w_img` from `current_nav_loc` where `?o` is visible.
                - If no such waypoint exists or is reachable, this path is impossible. Add infinity cost.
                - Otherwise, add shortest_path_distance(`current_nav_loc`, `?w_img`) + 1 (take image) to step_cost.
                - Update `current_nav_loc` = `?w_img`.
                - Add shortest_path_distance(`current_nav_loc`, lander_loc) + 1 (communicate) to step_cost.
                - Add step_cost to total_cost.
    5. Return total_cost. If total_cost includes infinity from any goal, the result will be infinity.

    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        self.lander_location = None
        self.rover_capabilities = {} # rover -> set of capabilities ('soil', 'rock', 'imaging')
        self.rover_stores = {} # rover -> store
        self.camera_info = {} # camera -> {'rover': rover, 'modes': set(modes), 'cal_target': objective}
        self.objective_visibility = {} # objective -> set(waypoints)
        self.waypoint_graph = {} # waypoint -> set(neighbor_waypoints)
        self.shortest_paths = {} # (w1, w2) -> distance

        waypoints = set()
        rovers = set()
        cameras = set()
        objectives = set()
        modes = set()

        # --- Parse Static Facts ---
        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]

            if predicate == 'at_lander':
                # Assuming only one lander and one location
                if len(parts) == 3:
                    self.lander_location = parts[2]
            elif predicate == 'equipped_for_soil_analysis':
                if len(parts) == 2:
                    rover = parts[1]
                    self.rover_capabilities.setdefault(rover, set()).add('soil')
                    rovers.add(rover)
            elif predicate == 'equipped_for_rock_analysis':
                if len(parts) == 2:
                    rover = parts[1]
                    self.rover_capabilities.setdefault(rover, set()).add('rock')
                    rovers.add(rover)
            elif predicate == 'equipped_for_imaging':
                if len(parts) == 2:
                    rover = parts[1]
                    self.rover_capabilities.setdefault(rover, set()).add('imaging')
                    rovers.add(rover)
            elif predicate == 'store_of':
                if len(parts) == 3:
                    store, rover = parts[1], parts[2]
                    self.rover_stores[rover] = store
                    rovers.add(rover)
            elif predicate == 'visible':
                if len(parts) == 3:
                    w1, w2 = parts[1], parts[2]
                    self.waypoint_graph.setdefault(w1, set()).add(w2)
                    self.waypoint_graph.setdefault(w2, set()).add(w1) # Assuming symmetric visibility
                    waypoints.add(w1)
                    waypoints.add(w2)
            elif predicate == 'can_traverse':
                 # Use can_traverse to build the graph as it's more explicit about traversability
                 # Assume can_traverse is symmetric if visible is symmetric, or build directed graph if needed.
                 # Examples show symmetric can_traverse matching visible, so stick to undirected.
                 if len(parts) == 4:
                     # parts[1] is rover, parts[2] is from, parts[3] is to
                     w1, w2 = parts[2], parts[3]
                     self.waypoint_graph.setdefault(w1, set()).add(w2)
                     self.waypoint_graph.setdefault(w2, set()).add(w1) # Assuming symmetric traversability
                     waypoints.add(w1)
                     waypoints.add(w2)
            elif predicate == 'on_board':
                if len(parts) == 3:
                    camera, rover = parts[1], parts[2]
                    self.camera_info.setdefault(camera, {})['rover'] = rover
                    cameras.add(camera)
                    rovers.add(rover)
            elif predicate == 'supports':
                if len(parts) == 3:
                    camera, mode = parts[1], parts[2]
                    self.camera_info.setdefault(camera, {}).setdefault('modes', set()).add(mode)
                    cameras.add(camera)
                    modes.add(mode)
            elif predicate == 'calibration_target':
                if len(parts) == 3:
                    camera, objective = parts[1], parts[2]
                    self.camera_info.setdefault(camera, {})['cal_target'] = objective
                    cameras.add(camera)
                    objectives.add(objective)
            elif predicate == 'visible_from':
                if len(parts) == 3:
                    objective, waypoint = parts[1], parts[2]
                    self.objective_visibility.setdefault(objective, set()).add(waypoint)
                    objectives.add(objective)
                    waypoints.add(waypoint)
            # Add other static predicates if needed, but these seem sufficient for goals

        # Ensure all waypoints mentioned are in the graph keys, even if isolated
        for w in waypoints:
             self.waypoint_graph.setdefault(w, set())

        # --- Precompute Shortest Paths ---
        # Run BFS from each waypoint
        all_waypoints = list(self.waypoint_graph.keys())
        for start_node in all_waypoints:
            distances = bfs(self.waypoint_graph, start_node)
            for end_node, dist in distances.items():
                self.shortest_paths[(start_node, end_node)] = dist
            # Add unreachable nodes with maxsize distance
            for end_node in all_waypoints:
                 if (start_node, end_node) not in self.shortest_paths:
                      self.shortest_paths[(start_node, end_node)] = sys.maxsize


    def get_distance(self, w1, w2):
        """Get shortest distance between two waypoints, returns sys.maxsize if unreachable."""
        if w1 is None or w2 is None: return sys.maxsize # Cannot navigate from/to nowhere
        return self.shortest_paths.get((w1, w2), sys.maxsize)

    def find_rover_location(self, state, rover):
        """Find the current location of a specific rover in the state."""
        for fact in state:
            if match(fact, "at", rover, "*"):
                return get_parts(fact)[2]
        return None # Rover location not found in state

    def find_sample_in_store(self, state, rover):
        """Check if a rover's store is full and what sample it contains."""
        store = self.rover_stores.get(rover)
        if not store or f'(full {store})' not in state:
            return None # Store not full or rover has no store

        # Check for soil sample
        for fact in state:
            if match(fact, "have_soil_sample", rover, "*"):
                 if len(get_parts(fact)) == 4:
                    return ('soil', get_parts(fact)[3])

        # Check for rock sample
        for fact in state:
            if match(fact, "have_rock_sample", rover, "*"):
                 if len(get_parts(fact)) == 4:
                    return ('rock', get_parts(fact)[3])

        return ('unknown', None) # Store is full, but contains an unknown sample type? Or empty store fact is missing? Treat as full with wrong sample.


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        total_cost = 0

        # Check if goal is already reached
        if self.goals <= state:
            return 0

        lander_loc = self.lander_location
        if lander_loc is None:
             return sys.maxsize

        # Iterate through each goal predicate
        for goal in self.goals:
            if goal in state:
                continue # Goal already achieved

            parts = get_parts(goal)
            if not parts: continue

            predicate = parts[0]

            if predicate == 'communicated_soil_data':
                if len(parts) != 2: continue

                w_sample = parts[1]
                goal_cost = sys.maxsize # Initialize cost for this goal

                # Find an equipped rover
                equipped_rover = None
                for rover, caps in self.rover_capabilities.items():
                    if 'soil' in caps:
                        equipped_rover = rover
                        break # Use the first equipped rover

                if equipped_rover is not None:
                    current_rover_loc = self.find_rover_location(state, equipped_rover)
                    if current_rover_loc is not None:
                        # Check if sample is already collected by this rover
                        if f'(have_soil_sample {equipped_rover} {w_sample})' in state:
                            # Sample collected, need to communicate
                            cost = 1 # communicate
                            cost += self.get_distance(current_rover_loc, lander_loc)
                            goal_cost = cost
                        else:
                            # Sample not collected, need to collect and communicate
                            cost = self.get_distance(current_rover_loc, w_sample) + 1 # nav to sample + take
                            cost += self.get_distance(w_sample, lander_loc) + 1 # nav to lander + communicate

                            # Add penalty if store is full with a different sample
                            sample_in_store = self.find_sample_in_store(state, equipped_rover)
                            if sample_in_store is not None:
                                sample_type, sample_waypoint = sample_in_store
                                # Penalty if it's a different sample type or same type from different waypoint
                                if not (sample_type == 'soil' and sample_waypoint == w_sample):
                                    # Cost to clear store: nav to lander + communicate old + empty store effect
                                    # Estimate penalty as nav to lander + 2 actions
                                    penalty = self.get_distance(current_rover_loc, lander_loc) + 2
                                    cost += penalty

                            goal_cost = cost

                total_cost += goal_cost

            elif predicate == 'communicated_rock_data':
                if len(parts) != 2: continue

                w_sample = parts[1]
                goal_cost = sys.maxsize # Initialize cost for this goal

                # Find an equipped rover
                equipped_rover = None
                for rover, caps in self.rover_capabilities.items():
                    if 'rock' in caps:
                        equipped_rover = rover
                        break # Use the first equipped rover

                if equipped_rover is not None:
                    current_rover_loc = self.find_rover_location(state, equipped_rover)
                    if current_rover_loc is not None:
                        # Check if sample is already collected by this rover
                        if f'(have_rock_sample {equipped_rover} {w_sample})' in state:
                            # Sample collected, need to communicate
                            cost = 1 # communicate
                            cost += self.get_distance(current_rover_loc, lander_loc)
                            goal_cost = cost
                        else:
                            # Sample not collected, need to collect and communicate
                            cost = self.get_distance(current_rover_loc, w_sample) + 1 # nav to sample + take
                            cost += self.get_distance(w_sample, lander_loc) + 1 # nav to lander + communicate

                            # Add penalty if store is full with a different sample
                            sample_in_store = self.find_sample_in_store(state, equipped_rover)
                            if sample_in_store is not None:
                                sample_type, sample_waypoint = sample_in_store
                                if not (sample_type == 'rock' and sample_waypoint == w_sample):
                                    # Cost to clear store: nav to lander + communicate old + empty store effect
                                    # Estimate penalty as nav to lander + 2 actions
                                    penalty = self.get_distance(current_rover_loc, lander_loc) + 2
                                    cost += penalty

                            goal_cost = cost

                total_cost += goal_cost

            elif predicate == 'communicated_image_data':
                if len(parts) != 3: continue

                objective = parts[1]
                mode = parts[2]
                goal_cost = sys.maxsize # Initialize cost for this goal

                # Find a suitable rover/camera
                suitable_rover = None
                suitable_camera = None
                for camera, info in self.camera_info.items():
                    if 'rover' in info and 'modes' in info and 'cal_target' in info:
                        if mode in info['modes'] and info['cal_target'] == objective:
                            rover = info['rover']
                            if 'imaging' in self.rover_capabilities.get(rover, set()):
                                suitable_rover = rover
                                suitable_camera = camera
                                break # Use the first suitable pair

                if suitable_rover is not None:
                    current_rover_loc = self.find_rover_location(state, suitable_rover)
                    if current_rover_loc is not None:
                        # Check if image is already taken by this rover
                        if f'(have_image {suitable_rover} {objective} {mode})' in state:
                            # Image taken, need to communicate
                            cost = 1 # communicate
                            cost += self.get_distance(current_rover_loc, lander_loc)
                            goal_cost = cost
                        else:
                            # Image not taken, need to take and communicate
                            step_cost = 0
                            current_nav_loc = current_rover_loc

                            # Check if calibration is needed
                            if f'(calibrated {suitable_camera} {suitable_rover})' not in state:
                                # Find closest calibration waypoint (visible_from objective)
                                cal_waypoints = self.objective_visibility.get(objective, set())
                                best_cal_w = None
                                min_cal_dist = sys.maxsize
                                for w_cal in cal_waypoints:
                                     dist = self.get_distance(current_nav_loc, w_cal)
                                     if dist < min_cal_dist:
                                         min_cal_dist = dist
                                         best_cal_w = w_cal

                                if best_cal_w is None or min_cal_dist == sys.maxsize:
                                     # Cannot calibrate, goal impossible via this path
                                     pass # Keep goal_cost as sys.maxsize
                                else:
                                    step_cost += min_cal_dist + 1 # nav + calibrate
                                    current_nav_loc = best_cal_w

                                    # Find closest imaging waypoint *after* calibration
                                    img_waypoints = self.objective_visibility.get(objective, set())
                                    best_img_w = None
                                    min_img_dist = sys.maxsize
                                    for w_img in img_waypoints:
                                        dist = self.get_distance(current_nav_loc, w_img)
                                        if dist < min_img_dist:
                                            min_img_dist = dist
                                            best_img_w = w_img

                                    if best_img_w is None or min_img_dist == sys.maxsize:
                                         # Cannot image, goal impossible via this path
                                         pass # Keep goal_cost as sys.maxsize
                                    else:
                                        step_cost += min_img_dist + 1 # nav + take image
                                        current_nav_loc = best_img_w

                                        # Navigate to lander and communicate
                                        step_cost += self.get_distance(current_nav_loc, lander_loc) + 1 # nav + communicate
                                        goal_cost = step_cost # Valid path found

                            else: # Camera is already calibrated
                                # Find closest imaging waypoint directly from current location
                                img_waypoints = self.objective_visibility.get(objective, set())
                                best_img_w = None
                                min_img_dist = sys.maxsize
                                for w_img in img_waypoints:
                                    dist = self.get_distance(current_nav_loc, w_img)
                                    if dist < min_img_dist:
                                        min_img_dist = dist
                                        best_img_w = w_img

                                if best_img_w is None or min_img_dist == sys.maxsize:
                                     # Cannot image, goal impossible via this path
                                     pass # Keep goal_cost as sys.maxsize
                                else:
                                    step_cost += min_img_dist + 1 # nav + take image
                                    current_nav_loc = best_img_w

                                    # Navigate to lander and communicate
                                    step_cost += self.get_distance(current_nav_loc, lander_loc) + 1 # nav + communicate
                                    goal_cost = step_cost # Valid path found

                total_cost += goal_cost

        return total_cost
