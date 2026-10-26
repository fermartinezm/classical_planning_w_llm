from fnmatch import fnmatch
from collections import deque
import math # Use math.inf for infinity

# Assume Heuristic base class is available from the planning framework
# from heuristics.heuristic_base import Heuristic

# Define a dummy Heuristic base class for standalone testing if needed
# In a real planning framework, this would be provided.
class Heuristic:
    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static
        # Add any necessary preprocessing here
        pass

    def __call__(self, node):
        # Compute heuristic value
        pass

# Helper functions to parse PDDL facts
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
    # Check if each part matches the corresponding argument pattern
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# BFS for shortest path on waypoint graph
def bfs_shortest_path(graph, start):
    """Computes shortest path distances from start to all reachable nodes using BFS."""
    distances = {node: math.inf for node in graph}
    if start in graph: # Ensure start node is in the graph
        distances[start] = 0
        queue = deque([start])
        while queue:
            current_wp = queue.popleft()
            if current_wp in graph: # Ensure current_wp is a valid key
                for neighbor in graph[current_wp]:
                    if distances[neighbor] == math.inf:
                        distances[neighbor] = distances[current_wp] + 1
                        queue.append(neighbor)
    return distances


class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to satisfy all
    unmet goal conditions. It calculates the minimum cost for each unmet goal
    independently, considering the necessary steps (movement, sampling/imaging,
    analysis/calibration, communication) and sums these minimum costs. It
    ignores resource conflicts (multiple goals needing the same rover) and
    simplifies some dependencies (like store state).

    # Assumptions
    - Samples (rock/soil) and objective visibility are static.
    - Lander location is static.
    - Rover capabilities and camera configurations are static.
    - `can_traverse` links are symmetric and the same for all rovers (or we build a symmetric graph).
    - The cost of any atomic action (move between adjacent waypoints, sample, analyze, calibrate, image, communicate) is 1.
    - The heuristic calculates the minimum cost for each goal over all suitable rovers/cameras.

    # Heuristic Initialization
    - Parses static facts to build data structures:
        - Set of all rovers, waypoints, objectives, modes, cameras, landers, stores.
        - Waypoint graph based on `can_traverse` facts (symmetric).
        - All-pairs shortest path distances on the waypoint graph using BFS.
        - Lander location waypoint.
        - Set of waypoints visible from the lander (communication points).
        - Set of waypoints with soil samples.
        - Set of waypoints with rock samples.
        - Mapping from objectives to the set of waypoints from which they are visible (for imaging and calibration).
        - Mapping from rovers to their set of capabilities (soil, rock, imaging).
        - Mapping from cameras to their information (on-board rover, supported modes, calibration target objective).
        - Mapping from rovers to their store object (not strictly used in this simplified heuristic, but extracted).

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize total heuristic cost to 0.
    2. For each goal condition specified in the task:
        a. Check if the goal condition is already true in the current state. If yes, continue to the next goal.
        b. If the goal is not met, calculate the minimum estimated cost to achieve *only this goal* from the current state. This involves:
            i. Identifying the type of goal (communicated soil, rock, or image data).
            ii. Finding suitable agents (rovers for soil/rock, rover-camera pairs for image) based on capabilities and equipment. If no suitable agent exists or required static locations (samples, visible points) are missing, the goal is impossible; return infinity for the total heuristic.
            iii. Estimating the steps required for each suitable agent and taking the minimum:
                - **For Soil/Rock Goals (communicated_soil/rock_data ?w):**
                    - Cost starts at 0.
                    - If `(have_soil/rock_analysis r w)` is not true for this specific rover `r`:
                        - Add 1 (for analysis).
                        - If `(have_soil/rock_sample r w)` is not true for this specific rover `r`:
                            - Add 1 (for taking sample).
                            - Add distance from the rover's current location (`r_loc`) to waypoint `?w`.
                            - Set the rover's location *after* data collection to `?w`.
                        else: # Have sample, need analysis
                            # Add distance from r_loc to w (to analyze)
                            Add distance from `r_loc` to `?w`.
                            Set the rover's location *after* data collection to `?w`.
                    else: # Already have analysis
                        # Rover's location after data collection is its current location (`r_loc`).
                        pass # location is already r_loc

                    # Steps to communicate
                    Add 1 (for communication).
                    Add distance from the rover's location *after* data collection to the nearest waypoint visible from the lander.
                    Calculate this cost for each suitable rover and take the minimum.
                - **For Image Goals (communicated_image_data ?o ?m):**
                    - Cost starts at 0.
                    - Find a waypoint `w_img` visible from objective `?o`.
                    - Find a waypoint `w_cal` visible from the camera's calibration target objective.
                    - If `(have_image r o m)` is not true for this specific rover `r` and camera `c`:
                        - Add 1 (for taking image).
                        - If `(calibrated c r)` is not true for this specific camera `c` on rover `r`:
                            - Add 1 (for calibration).
                            - Add distance from `r_loc` to `w_cal`.
                            - Add distance from `w_cal` to `w_img`.
                            - Set the rover's location *after* data collection to `w_img`.
                        else: # Already calibrated
                            # Add distance from r_loc to w_img.
                            Add distance from `r_loc` to `w_img`.
                            Set the rover's location *after* data collection to `w_img`.
                    else: # Already have image
                        # Rover's location after data collection is its current location (`r_loc`).
                        pass # location is already r_loc

                    # Steps to communicate
                    Add 1 (for communication).
                    Add distance from the rover's location *after* data collection to the nearest waypoint visible from the lander.
                    Calculate this cost for each suitable rover/camera pair and take the minimum.
            iv. If the minimum cost for this goal is infinity (e.g., no path, no suitable agent), the total heuristic is infinity.
        c. Add the minimum estimated cost for this unmet goal to the total heuristic cost.
    3. Return the total heuristic cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information from the task.
        """
        super().__init__(task)

        # --- Extract Static Information ---

        # Collect all objects by type by parsing initial state and static facts
        # This is a simplification; a proper parser would get this from the domain/instance file.
        self.rovers = set()
        self.waypoints = set()
        self.objectives = set()
        self.modes = set()
        self.cameras = set()
        self.landers = set()
        self.stores = set()

        all_facts = task.initial_state | task.static
        # Also include goal facts as they mention objects
        if task.goals:
             all_facts |= task.goals

        for fact in all_facts:
             parts = get_parts(fact)
             if not parts: continue # Skip empty facts

             pred = parts[0]
             if pred == 'at_lander' and len(parts) == 3: self.landers.add(parts[1]); self.waypoints.add(parts[2])
             elif pred == 'at' and len(parts) == 3: self.rovers.add(parts[1]); self.waypoints.add(parts[2])
             elif pred in ['equipped_for_soil_analysis', 'equipped_for_rock_analysis', 'equipped_for_imaging'] and len(parts) == 2: self.rovers.add(parts[1])
             elif pred == 'store_of' and len(parts) == 3: self.stores.add(parts[1]); self.rovers.add(parts[2])
             elif pred == 'visible' and len(parts) == 3: self.waypoints.add(parts[1]); self.waypoints.add(parts[2])
             elif pred == 'can_traverse' and len(parts) == 4: self.rovers.add(parts[1]); self.waypoints.add(parts[2]); self.waypoints.add(parts[3])
             elif pred == 'calibration_target' and len(parts) == 3: self.cameras.add(parts[1]); self.objectives.add(parts[2])
             elif pred == 'on_board' and len(parts) == 3: self.cameras.add(parts[1]); self.rovers.add(parts[2])
             elif pred == 'supports' and len(parts) == 3: self.cameras.add(parts[1]); self.modes.add(parts[2])
             elif pred == 'visible_from' and len(parts) == 3: self.objectives.add(parts[1]); self.waypoints.add(parts[2])
             elif pred in ['at_rock_sample', 'at_soil_sample'] and len(parts) == 2: self.waypoints.add(parts[1])
             elif pred in ['have_soil_analysis', 'have_rock_analysis'] and len(parts) == 3: self.rovers.add(parts[1]); self.waypoints.add(parts[2])
             elif pred == 'have_image' and len(parts) == 4: self.rovers.add(parts[1]); self.objectives.add(parts[2]); self.modes.add(parts[3])
             elif pred == 'calibrated' and len(parts) == 3: self.cameras.add(parts[1]); self.rovers.add(parts[2])
             elif pred in ['communicated_soil_data', 'communicated_rock_data'] and len(parts) == 2: self.waypoints.add(parts[1])
             elif pred == 'communicated_image_data' and len(parts) == 3: self.objectives.add(parts[1]); self.modes.add(parts[2])
             elif pred in ['empty', 'full'] and len(parts) == 2: self.stores.add(parts[1])


        self.lander_waypoint = None
        for fact in self.static:
            if match(fact, "at_lander", "*", "*"):
                self.lander_waypoint = get_parts(fact)[2]
                break # Assuming only one lander

        self.comm_waypoint_set = set()
        if self.lander_waypoint:
             for fact in self.static:
                 if match(fact, "visible", "*", self.lander_waypoint):
                     self.comm_waypoint_set.add(get_parts(fact)[1])
                 elif match(fact, "visible", self.lander_waypoint, "*"): # visible is symmetric
                      self.comm_waypoint_set.add(get_parts(fact)[2])


        self.soil_sample_wps = {get_parts(fact)[1] for fact in self.static if match(fact, "at_soil_sample", "*")}
        self.rock_sample_wps = {get_parts(fact)[1] for fact in self.static if match(fact, "at_rock_sample", "*")}

        self.objective_visible_wps = {}
        for fact in self.static:
            if match(fact, "visible_from", "*", "*"):
                o, w = get_parts(fact)[1:]
                self.objective_visible_wps.setdefault(o, set()).add(w)

        self.rover_capabilities = {}
        for rover in self.rovers:
            caps = set()
            if f"(equipped_for_soil_analysis {rover})" in self.static: caps.add('soil')
            if f"(equipped_for_rock_analysis {rover})" in self.static: caps.add('rock')
            if f"(equipped_for_imaging {rover})" in self.static: caps.add('imaging')
            self.rover_capabilities[rover] = caps

        self.camera_info = {}
        for camera in self.cameras:
            info = {'rover': None, 'modes': set(), 'cal_target': None}
            for fact in self.static:
                if match(fact, "on_board", camera, "*"):
                    info['rover'] = get_parts(fact)[2]
                elif match(fact, "supports", camera, "*"):
                    info['modes'].add(get_parts(fact)[2])
                elif match(fact, "calibration_target", camera, "*"):
                    info['cal_target'] = get_parts(fact)[2]
            self.camera_info[camera] = info

        self.rover_stores = {}
        for fact in self.static:
            if match(fact, "store_of", "*", "*"):
                s, r = get_parts(fact)[1:]
                self.rover_stores[r] = s

        # Build waypoint graph
        self.waypoint_graph = {}
        all_waypoints = list(self.waypoints) # Use the set collected earlier
        for wp in all_waypoints:
            self.waypoint_graph[wp] = set()

        for fact in self.static:
            if match(fact, "can_traverse", "*", "*", "*"):
                 # Assuming can_traverse is symmetric or adding both directions
                 # Assuming rover doesn't matter for graph structure
                 _, r, w1, w2 = get_parts(fact)
                 if w1 in self.waypoint_graph and w2 in self.waypoint_graph:
                     self.waypoint_graph[w1].add(w2)
                     self.waypoint_graph[w2].add(w1) # Add reverse edge for symmetry


        # Compute all-pairs shortest paths
        self.distance_map = {}
        for start_wp in self.waypoint_graph:
            distances = bfs_shortest_path(self.waypoint_graph, start_wp)
            for end_wp, dist in distances.items():
                 self.distance_map[(start_wp, end_wp)] = dist

    def get_rover_location(self, state, rover):
        """Find the current waypoint of a given rover in the state."""
        for fact in state:
            if match(fact, "at", rover, "*"):
                return get_parts(fact)[2]
        return None # Rover location not found (shouldn't happen in valid states)

    def find_nearest_waypoint(self, from_wp, target_wps, distance_map):
        """Find the minimum distance from from_wp to any waypoint in target_wps."""
        if from_wp is None or not target_wps:
            return math.inf
        min_dist = math.inf
        for target_wp in target_wps:
            dist = distance_map.get((from_wp, target_wp), math.inf)
            min_dist = min(min_dist, dist)
        return min_dist

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        total_cost = 0

        # Check if goal is already reached
        if self.goals <= state:
            return 0

        for goal_fact in self.goals:
            # If goal is already satisfied, skip it
            if goal_fact in state:
                continue

            predicate, *args = get_parts(goal_fact)
            goal_cost = math.inf # Minimum cost for this specific goal over all agents

            if predicate == 'communicated_soil_data':
                w = args[0]
                # Check if sample exists (static)
                if w not in self.soil_sample_wps:
                    return math.inf # Impossible goal

                # Find suitable rovers
                suitable_rovers = [r for r in self.rovers if 'soil' in self.rover_capabilities.get(r, set())]
                if not suitable_rovers:
                    return math.inf # Impossible goal

                # Calculate cost for each suitable rover and take the minimum
                for r in suitable_rovers:
                    r_loc = self.get_rover_location(state, r)
                    if r_loc is None: continue # Should not happen

                    current_rover_goal_cost = 0
                    rover_location_after_data_collection = r_loc # Default: rover stays put

                    # Steps to get analysis
                    if f"(have_soil_analysis {r} {w})" not in state:
                        current_rover_goal_cost += 1 # analyze action
                        if f"(have_soil_sample {r} {w})" not in state:
                            current_rover_goal_cost += 1 # take_sample action
                            # Need to move to sample spot first
                            move_to_sample_dist = self.distance_map.get((r_loc, w), math.inf)
                            if move_to_sample_dist == math.inf: continue # Cannot reach sample spot
                            current_rover_goal_cost += move_to_sample_dist # move to sample spot
                            rover_location_after_data_collection = w # Rover is at sample waypoint after taking/analyzing sample
                        else: # Have sample, need analysis
                            # Need to move to sample spot if not already there
                            move_to_sample_dist = self.distance_map.get((r_loc, w), math.inf)
                            if move_to_sample_dist == math.inf: continue # Cannot reach sample spot
                            current_rover_goal_cost += move_to_sample_dist # move to sample spot
                            rover_location_after_data_collection = w # Rover is at sample waypoint after analysis

                    # Steps to communicate
                    current_rover_goal_cost += 1 # communicate action
                    # Need to move from rover's location (where it has the analysis) to a communication waypoint
                    w_comm_dist = self.find_nearest_waypoint(rover_location_after_data_collection, self.comm_waypoint_set, self.distance_map)
                    if w_comm_dist == math.inf: continue # Cannot communicate from rover's location
                    current_rover_goal_cost += w_comm_dist # move from rover's location to comm spot

                    goal_cost = min(goal_cost, current_rover_goal_cost)

            elif predicate == 'communicated_rock_data':
                w = args[0]
                 # Check if sample exists (static)
                if w not in self.rock_sample_wps:
                    return math.inf # Impossible goal

                # Find suitable rovers
                suitable_rovers = [r for r in self.rovers if 'rock' in self.rover_capabilities.get(r, set())]
                if not suitable_rovers:
                    return math.inf # Impossible goal

                # Calculate cost for each suitable rover and take the minimum
                for r in suitable_rovers:
                    r_loc = self.get_rover_location(state, r)
                    if r_loc is None: continue # Should not happen

                    current_rover_goal_cost = 0
                    rover_location_after_data_collection = r_loc # Default: rover stays put

                    # Steps to get analysis
                    if f"(have_rock_analysis {r} {w})" not in state:
                        current_rover_goal_cost += 1 # analyze action
                        if f"(have_rock_sample {r} {w})" not in state:
                            current_rover_goal_cost += 1 # take_sample action
                            # Need to move to sample spot first
                            move_to_sample_dist = self.distance_map.get((r_loc, w), math.inf)
                            if move_to_sample_dist == math.inf: continue # Cannot reach sample spot
                            current_rover_goal_cost += move_to_sample_dist # move to sample spot
                            rover_location_after_data_collection = w # Rover is at sample waypoint after taking/analyzing sample
                        else: # Have sample, need analysis
                            # Need to move to sample spot if not already there
                            move_to_sample_dist = self.distance_map.get((r_loc, w), math.inf)
                            if move_to_sample_dist == math.inf: continue # Cannot reach sample spot
                            current_rover_goal_cost += move_to_sample_dist # move to sample spot
                            rover_location_after_data_collection = w # Rover is at sample waypoint after analysis

                    # Steps to communicate
                    current_rover_goal_cost += 1 # communicate action
                    # Need to move from rover's location (where it has the analysis) to a communication waypoint
                    w_comm_dist = self.find_nearest_waypoint(rover_location_after_data_collection, self.comm_waypoint_set, self.distance_map)
                    if w_comm_dist == math.inf: continue # Cannot communicate from rover's location
                    current_rover_goal_cost += w_comm_dist # move from rover's location to comm spot

                    goal_cost = min(goal_cost, current_rover_goal_cost)


            elif predicate == 'communicated_image_data':
                o, m = args

                # Find waypoints visible from the objective (for taking image)
                image_wps = self.objective_visible_wps.get(o, set())
                if not image_wps:
                    return math.inf # Impossible goal

                # Find suitable rover/camera pairs
                suitable_pairs = []
                for r in self.rovers:
                    if 'imaging' in self.rover_capabilities.get(r, set()):
                        for c, info in self.camera_info.items():
                            if info['rover'] == r and m in info['modes'] and info['cal_target'] is not None:
                                 # Check if calibration target is visible from anywhere
                                 o_cal = info['cal_target']
                                 cal_wps = self.objective_visible_wps.get(o_cal, set())
                                 if cal_wps: # Only add pair if calibration is possible
                                    suitable_pairs.append((r, c))

                if not suitable_pairs:
                    return math.inf # Impossible goal

                # Calculate cost for each suitable pair and take the minimum
                for r, c in suitable_pairs:
                    r_loc = self.get_rover_location(state, r)
                    if r_loc is None: continue # Should not happen

                    current_pair_goal_cost = 0
                    o_cal = self.camera_info[c]['cal_target']
                    cal_wps = self.objective_visible_wps.get(o_cal, set()) # We already checked cal_wps is not empty
                    rover_location_after_data_collection = r_loc # Default: rover stays put

                    # Steps to get image
                    if f"(have_image {r} {o} {m})" not in state:
                        current_pair_goal_cost += 1 # take_image action

                        # Find nearest image waypoint from current rover location
                        nearest_image_wp = self.find_nearest_waypoint(r_loc, image_wps, self.distance_map)
                        if nearest_image_wp == math.inf: continue # Cannot reach any image spot

                        if f"(calibrated {c} {r})" not in state:
                            current_pair_goal_cost += 1 # calibrate action
                            # Find nearest calibration waypoint from current rover location
                            nearest_cal_wp = self.find_nearest_waypoint(r_loc, cal_wps, self.distance_map)
                            if nearest_cal_wp == math.inf: continue # Cannot reach any cal spot

                            # Movement: r_loc -> cal_wp -> image_wp
                            move_to_cal_dist = self.distance_map.get((r_loc, nearest_cal_wp), math.inf)
                            move_cal_to_image_dist = self.distance_map.get((nearest_cal_wp, nearest_image_wp), math.inf)
                            if move_to_cal_dist == math.inf or move_cal_to_image_dist == math.inf: continue # Cannot complete movement
                            current_pair_goal_cost += move_to_cal_dist + move_cal_to_image_dist
                            rover_location_after_data_collection = nearest_image_wp # Rover is at image spot after taking image
                        else: # Already calibrated
                            # Movement: r_loc -> image_wp
                            move_to_image_dist = self.distance_map.get((r_loc, nearest_image_wp), math.inf)
                            if move_to_image_dist == math.inf: continue # Cannot reach image spot
                            current_pair_goal_cost += move_to_image_dist
                            rover_location_after_data_collection = nearest_image_wp # Rover is at image spot after taking image

                    # Steps to communicate
                    current_pair_goal_cost += 1 # communicate action
                    # Need to move from rover's location (where it has the image) to a communication waypoint
                    w_comm_dist = self.find_nearest_waypoint(rover_location_after_data_collection, self.comm_waypoint_set, self.distance_map)
                    if w_comm_dist == math.inf: continue # Cannot communicate from rover's location
                    current_pair_goal_cost += w_comm_dist # move from rover's location to comm spot

                    goal_cost = min(goal_cost, current_pair_goal_cost)

            # If after checking all options, the goal cost is still infinity,
            # it means this goal is unreachable from the current state.
            if goal_cost == math.inf:
                 return math.inf # Problem is unsolvable from this state

            total_cost += goal_cost

        return total_cost
