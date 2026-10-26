from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic # Assuming this base class exists

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    if not fact or fact[0] != '(' or fact[-1] != ')':
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

def bfs(graph, start, goal):
    """
    Performs BFS to find the shortest path distance in an unweighted graph.
    graph: adjacency dictionary {waypoint: {neighbor1, neighbor2, ...}}
    start: starting waypoint
    goal: target waypoint
    Returns: shortest distance (number of edges) or float('inf') if unreachable.
    """
    if start == goal:
        return 0
    if start not in graph:
         return float('inf')

    queue = deque([(start, 0)])
    visited = {start}

    while queue:
        current_waypoint, distance = queue.popleft()

        if current_waypoint == goal:
            return distance

        if current_waypoint in graph:
            for neighbor in graph[current_waypoint]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, distance + 1))

    return float('inf') # Goal not reachable


class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve all
    unmet goal conditions (communicating image, soil, or rock data). It sums
    the estimated costs for each individual unmet goal, ignoring potential
    interactions (positive or negative) between goals. The cost for each goal
    is estimated by finding the minimum number of actions (moves, samples,
    analyses, calibrations, communications) required for a suitable rover
    to collect the necessary data and communicate it to the lander.

    # Assumptions
    - Action costs are uniform (each action costs 1).
    - Waypoint traversal costs are uniform (each move between visible waypoints costs 1).
    - The heuristic ignores resource constraints beyond basic capabilities (e.g., assumes a rover with a full store can eventually drop it to take a new sample, adding a small penalty).
    - It assumes solvable instances, meaning necessary objects (rovers, cameras, samples, objectives, lander) and paths exist. Unreachable goals contribute infinity.
    - Calibration is needed once per camera-rover pair for a specific target objective before taking images using that camera.

    # Heuristic Initialization
    The heuristic extracts static information from the task:
    - `rover_graph`: Adjacency list representation of the waypoint graph for each rover based on `can_traverse` facts.
    - `camera_on_board`: Mapping from camera to the rover it's on.
    - `camera_supports`: Mapping from camera to the set of modes it supports.
    - `camera_calibration_target`: Mapping from camera to its calibration target objective.
    - `visible_from`: Mapping from objectives/targets to the set of waypoints they are visible from.
    - `lander_location`: The waypoint where the lander is located.
    - `rover_capabilities`: Mapping from rover to the set of capabilities it has (imaging, soil, rock).
    - `soil_samples_at`: Set of waypoints with soil samples.
    - `rock_samples_at`: Set of waypoints with rock samples.
    - `store_of_rover`: Mapping from store object to the rover it belongs to.
    - `rover_store_empty_static`: Initial empty status of each rover's store based on static facts.
    - `goal_info`: Set of goal facts represented as tuples.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Initialize `total_cost = 0`.
    2. Extract dynamic information from the state:
       - Current location of each rover.
       - Data already collected (`have_image`, `have_soil_analysis`, `have_rock_analysis`).
       - Cameras already calibrated on rovers (`calibrated`).
       - Current empty status of each rover's store.
    3. Convert state facts into a set of tuples for efficient lookup.
    4. Iterate through each goal fact (`goal_tuple`) stored in `self.goal_info`:
       - If `goal_tuple` is already present in the current state (checked against the set of state tuples), continue to the next goal (cost is 0 for this goal).
       - If the goal is an image communication `('communicated_image_data', obj, mode)`:
         - Check if `('have_image', r, obj, mode)` exists for any rover `r` in the state.
         - If yes: Find the minimum cost for a rover `r` that has the image to move to the lander and communicate: `min(bfs(rover_graph[r], current_rover_location[r], lander_location) + 1)`. Add this minimum cost to `total_cost`.
         - If no: Find the minimum cost over all suitable rovers `r` (imaging capable), cameras `c` (on `r`, supports `mode`), calibration targets `t` (for `c`), calibration waypoints `w_cal` (visible from `t`), and image waypoints `w_img` (visible from `obj`). The cost for each combination is estimated as:
           `bfs(r_loc, w_cal)` (move to cal) + `(1 if (c, r) not in calibrated_cameras else 0)` (calibrate action cost, only if needed) + `bfs(w_cal, w_img)` (move to img) + `1` (take image action cost) + `bfs(w_img, lander_location)` (move to lander) + `1` (communicate action cost).
           Find the minimum of this cost over all valid combinations. Add this minimum cost to `total_cost`. If no valid combination exists (e.g., no path, no suitable rover/camera/waypoint), the cost is infinity.
       - If the goal is soil communication `('communicated_soil_data', w_sample)`:
         - Check if `('have_soil_analysis', r, w_sample)` exists for any rover `r` in the state.
         - If yes: Find the minimum cost for a rover `r` that has the analysis to move to the lander and communicate: `min(bfs(rover_graph[r], current_rover_location[r], lander_location) + 1)`. Add this minimum cost to `total_cost`.
         - If no: Find the minimum cost over all suitable rovers `r` (soil analysis capable). The cost for each rover is estimated as:
           `(1 if not rover_store_empty_current.get(r, True) else 0)` (penalty if store is not empty) + `bfs(rover_graph[r], current_rover_location[r], w_sample)` (move to sample) + `1` (take sample action cost) + `1` (analyse action cost) + `bfs(rover_graph[r], w_sample, lander_location)` (move to lander) + `1` (communicate action cost).
           Find the minimum of this cost over all valid rovers. Add this minimum cost to `total_cost`. If no valid rover exists or no path, the cost is infinity.
       - If the goal is rock communication `('communicated_rock_data', w_sample)`:
         - Similar calculation as for soil communication, using rock capabilities and rock sample locations.
    5. Return the `total_cost`.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # Extract static information
        self.rover_graph = {} # {rover_name: {waypoint: {neighbor1, neighbor2, ...}}}
        self.camera_on_board = {} # {camera: rover}
        self.camera_supports = {} # {camera: {mode1, mode2, ...}}
        self.camera_calibration_target = {} # {camera: target}
        self.visible_from = {} # {objective/target: {waypoint1, waypoint2, ...}}
        self.lander_location = None
        self.rover_capabilities = {} # {rover: {soil, rock, imaging}}
        self.soil_samples_at = set() # {waypoint1, waypoint2, ...}
        self.rock_samples_at = set() # {waypoint1, waypoint2, ...}
        self.store_of_rover = {} # {store: rover}
        self.rover_store_empty_static = {} # {rover: bool}

        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts
            predicate = parts[0]

            if predicate == "can_traverse":
                if len(parts) == 4:
                    rover, w1, w2 = parts[1:]
                    if rover not in self.rover_graph:
                        self.rover_graph[rover] = {}
                    if w1 not in self.rover_graph[rover]:
                        self.rover_graph[rover][w1] = set()
                    self.rover_graph[rover][w1].add(w2)
            elif predicate == "on_board":
                 if len(parts) == 3:
                    camera, rover = parts[1:]
                    self.camera_on_board[camera] = rover
            elif predicate == "supports":
                 if len(parts) == 3:
                    camera, mode = parts[1:]
                    if camera not in self.camera_supports:
                        self.camera_supports[camera] = set()
                    self.camera_supports[camera].add(mode)
            elif predicate == "calibration_target":
                 if len(parts) == 3:
                    camera, target = parts[1:]
                    self.camera_calibration_target[camera] = target
            elif predicate == "visible_from":
                 if len(parts) == 3:
                    obj_or_target, waypoint = parts[1:]
                    if obj_or_target not in self.visible_from:
                        self.visible_from[obj_or_target] = set()
                    self.visible_from[obj_or_target].add(waypoint)
            elif predicate == "at_lander":
                 if len(parts) == 3:
                    lander, waypoint = parts[1:]
                    self.lander_location = waypoint # Assuming only one lander
            elif predicate.startswith("equipped_for_"):
                 if len(parts) == 2:
                    capability = predicate.split("_")[2] # e.g., "imaging", "soil", "rock"
                    rover = parts[1]
                    if rover not in self.rover_capabilities:
                        self.rover_capabilities[rover] = set()
                    self.rover_capabilities[rover].add(capability)
            elif predicate == "at_soil_sample":
                 if len(parts) == 2:
                    waypoint = parts[1]
                    self.soil_samples_at.add(waypoint)
            elif predicate == "at_rock_sample":
                 if len(parts) == 2:
                    waypoint = parts[1]
                    self.rock_samples_at.add(waypoint)
            elif predicate == "store_of":
                 if len(parts) == 3:
                    store, rover = parts[1:]
                    self.store_of_rover[store] = rover
                    # Initialize store status based on static facts later
                    self.rover_store_empty_static[rover] = False # Default assumption: not empty unless explicitly stated

        # Now initialize rover_store_empty_static based on static 'empty' facts
        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue
            if parts[0] == "empty" and len(parts) == 2:
                store = parts[1]
                rover = self.store_of_rover.get(store)
                if rover:
                    self.rover_store_empty_static[rover] = True
            # Note: 'full' is not a static predicate in the examples, only dynamic.

        # Store goal facts as tuples for easier lookup
        self.goal_info = set()
        for goal_fact_str in self.goals:
            parts = get_parts(goal_fact_str)
            if parts:
                self.goal_info.add(tuple(parts))


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        state_tuples = {tuple(get_parts(fact)) for fact in state}

        # Extract dynamic information from the current state
        current_rover_location = {} # {rover: waypoint}
        have_image_data = set() # {(rover, objective, mode), ...}
        have_soil_data = set() # {(rover, waypoint), ...}
        have_rock_data = set() # {(rover, waypoint), ...}
        calibrated_cameras = set() # {(camera, rover), ...}
        rover_store_empty_current = self.rover_store_empty_static.copy() # Start with static state

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue
            predicate = parts[0]

            if predicate == "at":
                if len(parts) == 3:
                    obj, waypoint = parts[1:]
                    # Assuming only rovers and lander have 'at' predicate
                    if obj in self.rover_graph: # Check if obj is a rover
                         current_rover_location[obj] = waypoint
            elif predicate == "have_image":
                 if len(parts) == 4:
                    rover, obj, mode = parts[1:]
                    have_image_data.add((rover, obj, mode))
            elif predicate == "have_soil_analysis":
                 if len(parts) == 3:
                    rover, waypoint = parts[1:]
                    have_soil_data.add((rover, waypoint))
            elif predicate == "have_rock_analysis":
                 if len(parts) == 3:
                    rover, waypoint = parts[1:]
                    have_rock_data.add((rover, waypoint))
            elif predicate == "calibrated":
                 if len(parts) == 3:
                    camera, rover = parts[1:]
                    calibrated_cameras.add((camera, rover))
            elif predicate == "empty":
                 if len(parts) == 2:
                    store = parts[1]
                    rover = self.store_of_rover.get(store)
                    if rover:
                        rover_store_empty_current[rover] = True
            elif predicate == "full":
                 if len(parts) == 2:
                    store = parts[1]
                    rover = self.store_of_rover.get(store)
                    if rover:
                        rover_store_empty_current[rover] = False # Explicitly full means not empty


        total_cost = 0

        # Iterate through each goal fact
        for goal_tuple in self.goal_info:
            # If goal is already achieved, cost is 0 for this goal
            if goal_tuple in state_tuples:
                continue

            predicate = goal_tuple[0]

            # Handle Image Communication Goals
            if predicate == "communicated_image_data" and len(goal_tuple) == 3:
                obj, mode = goal_tuple[1:]
                lander_loc = self.lander_location
                if not lander_loc: # Lander location must be known
                    total_cost += float('inf')
                    continue

                # Check if image is already collected by any rover
                rovers_with_image = [r for (r, o, m) in have_image_data if o == obj and m == mode]

                if rovers_with_image:
                    # Image collected, need to communicate
                    min_comm_cost = float('inf')
                    for rover in rovers_with_image:
                        if rover in current_rover_location and rover in self.rover_graph:
                            r_loc = current_rover_location[rover]
                            dist_to_lander = bfs(self.rover_graph[rover], r_loc, lander_loc)
                            if dist_to_lander != float('inf'):
                                min_comm_cost = min(min_comm_cost, dist_to_lander + 1) # move + communicate
                    total_cost += min_comm_cost

                else:
                    # Image not collected, need to collect and communicate
                    min_collect_comm_cost = float('inf')

                    # Find suitable rovers (imaging capable)
                    suitable_rovers = [r for r, caps in self.rover_capabilities.items() if 'imaging' in caps]

                    for rover in suitable_rovers:
                        if rover not in current_rover_location or rover not in self.rover_graph:
                            continue # Rover location unknown or cannot traverse

                        r_loc = current_rover_location[rover]

                        # Find suitable cameras on this rover that support the mode
                        suitable_cameras = [c for c, r in self.camera_on_board.items()
                                            if r == rover and mode in self.camera_supports.get(c, set())]

                        for camera in suitable_cameras:
                            cal_target = self.camera_calibration_target.get(camera)
                            if not cal_target: continue # Camera has no calibration target

                            # Find waypoints visible from the calibration target
                            cal_wps = self.visible_from.get(cal_target, set())
                            if not cal_wps: continue

                            # Find waypoints visible from the objective
                            img_wps = self.visible_from.get(obj, set())
                            if not img_wps: continue

                            is_calibrated = (camera, rover) in calibrated_cameras

                            # Find minimum cost path through cal_wp and img_wp
                            for w_cal in cal_wps:
                                cost_to_cal = bfs(self.rover_graph[rover], r_loc, w_cal)
                                if cost_to_cal == float('inf'): continue

                                cost_cal_action = 0 if is_calibrated else 1 # Cost of calibrate action

                                for w_img in img_wps:
                                    cost_to_img = bfs(self.rover_graph[rover], w_cal, w_img)
                                    if cost_to_img == float('inf'): continue

                                    cost_img_action = 1 # Cost of take_image action

                                    cost_to_lander = bfs(self.rover_graph[rover], w_img, lander_loc)
                                    if cost_to_lander == float('inf'): continue

                                    cost_comm_action = 1 # Cost of communicate_data action

                                    current_path_cost = cost_to_cal + cost_cal_action + cost_to_img + cost_img_action + cost_to_lander + cost_comm_action

                                    min_collect_comm_cost = min(min_collect_comm_cost, current_path_cost)

                    total_cost += min_collect_comm_cost


            # Handle Soil Communication Goals
            elif predicate == "communicated_soil_data" and len(goal_tuple) == 2:
                w_sample = goal_tuple[1]
                lander_loc = self.lander_location
                if not lander_loc:
                    total_cost += float('inf')
                    continue

                # Check if analysis is already collected by any rover
                rovers_with_analysis = [r for (r, w) in have_soil_data if w == w_sample]

                if rovers_with_analysis:
                    # Analysis collected, need to communicate
                    min_comm_cost = float('inf')
                    for rover in rovers_with_analysis:
                         if rover in current_rover_location and rover in self.rover_graph:
                            r_loc = current_rover_location[rover]
                            dist_to_lander = bfs(self.rover_graph[rover], r_loc, lander_loc)
                            if dist_to_lander != float('inf'):
                                min_comm_cost = min(min_comm_cost, dist_to_lander + 1) # move + communicate
                    total_cost += min_comm_cost
                else:
                    # Analysis not collected, need to collect and communicate
                    min_collect_comm_cost = float('inf')

                    # Find suitable rovers (soil analysis capable)
                    suitable_rovers = [r for r, caps in self.rover_capabilities.items() if 'soil' in caps]

                    if w_sample not in self.soil_samples_at:
                         # Sample not available at this waypoint, goal is impossible
                         total_cost += float('inf')
                         continue

                    for rover in suitable_rovers:
                        if rover not in current_rover_location or rover not in self.rover_graph:
                            continue # Rover location unknown or cannot traverse

                        r_loc = current_rover_location[rover]

                        cost_to_sample = bfs(self.rover_graph[rover], r_loc, w_sample)
                        if cost_to_sample == float('inf'): continue

                        cost_take_action = 1 # Cost of take_soil_sample
                        cost_analyse_action = 1 # Cost of analyse_soil

                        cost_to_lander = bfs(self.rover_graph[rover], w_sample, lander_loc)
                        if cost_to_lander == float('inf'): continue

                        cost_comm_action = 1 # Cost of communicate_data

                        # Penalty if store is not empty when needing to take a sample
                        store_penalty = 1 if not rover_store_empty_current.get(rover, True) else 0

                        current_path_cost = store_penalty + cost_to_sample + cost_take_action + cost_analyse_action + cost_to_lander + cost_comm_action

                        min_collect_comm_cost = min(min_collect_comm_cost, current_path_cost)

                    total_cost += min_collect_comm_cost


            # Handle Rock Communication Goals
            elif predicate == "communicated_rock_data" and len(goal_tuple) == 2:
                w_sample = goal_tuple[1]
                lander_loc = self.lander_location
                if not lander_loc:
                    total_cost += float('inf')
                    continue

                # Check if analysis is already collected by any rover
                rovers_with_analysis = [r for (r, w) in have_rock_data if w == w_sample]

                if rovers_with_analysis:
                    # Analysis collected, need to communicate
                    min_comm_cost = float('inf')
                    for rover in rovers_with_analysis:
                         if rover in current_rover_location and rover in self.rover_graph:
                            r_loc = current_rover_location[rover]
                            dist_to_lander = bfs(self.rover_graph[rover], r_loc, lander_loc)
                            if dist_to_lander != float('inf'):
                                min_comm_cost = min(min_comm_cost, dist_to_lander + 1) # move + communicate
                    total_cost += min_comm_cost
                else:
                    # Analysis not collected, need to collect and communicate
                    min_collect_comm_cost = float('inf')

                    # Find suitable rovers (rock analysis capable)
                    suitable_rovers = [r for r, caps in self.rover_capabilities.items() if 'rock' in caps]

                    if w_sample not in self.rock_samples_at:
                         # Sample not available at this waypoint, goal is impossible
                         total_cost += float('inf')
                         continue

                    for rover in suitable_rovers:
                        if rover not in current_rover_location or rover not in self.rover_graph:
                            continue # Rover location unknown or cannot traverse

                        r_loc = current_rover_location[rover]

                        cost_to_sample = bfs(self.rover_graph[rover], r_loc, w_sample)
                        if cost_to_sample == float('inf'): continue

                        cost_take_action = 1 # Cost of take_rock_sample
                        cost_analyse_action = 1 # Cost of analyse_rock

                        cost_to_lander = bfs(self.rover_graph[rover], w_sample, lander_loc)
                        if cost_to_lander == float('inf'): continue

                        cost_comm_action = 1 # Cost of communicate_data

                        # Penalty if store is not empty when needing to take a sample
                        store_penalty = 1 if not rover_store_empty_current.get(rover, True) else 0

                        current_path_cost = store_penalty + cost_to_sample + cost_take_action + cost_analyse_action + cost_to_lander + cost_comm_action

                        min_collect_comm_cost = min(min_collect_comm_cost, current_path_cost)

                    total_cost += min_collect_comm_cost

        return total_cost
