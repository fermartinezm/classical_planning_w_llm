from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque, defaultdict

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential leading/trailing whitespace and ensure it's a string
    fact_str = str(fact).strip()
    if not fact_str.startswith('(') or not fact_str.endswith(')'):
         # This should not happen with valid PDDL facts from the parser
         return []
    return fact_str[1:-1].split()

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

def bfs(graph, start_node):
    """Compute shortest paths from start_node in a graph."""
    distances = {node: float('inf') for node in graph}
    if start_node not in graph: # Handle cases where start_node is isolated or doesn't exist in the graph keys
         return distances
    distances[start_node] = 0
    queue = deque([start_node])
    while queue:
        current_node = queue.popleft()
        # Check if current_node is in graph before accessing neighbors
        if current_node in graph:
            for neighbor in graph.get(current_node, []):
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)
    return distances

class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the minimum number of actions required to achieve all uncommunicated goals (soil data, rock data, image data). It calculates the cost for each unachieved goal independently and sums these costs. For each goal, it considers two main ways to achieve it: communicating data that a rover already possesses, or collecting the data (sampling or imaging/calibrating) and then communicating it. It estimates the cost of each option by summing action costs (1 per action) and estimated navigation costs (shortest path distance between waypoints). The minimum cost option for each goal is chosen.

    # Assumptions
    - The cost of each action (navigate, sample, drop, calibrate, take_image, communicate) is 1.
    - Navigation cost between two waypoints for a specific rover is the shortest path distance in the graph defined by `can_traverse` and `visible` predicates for that rover. Shortest paths are precomputed using BFS.
    - The heuristic sums the costs for achieving each unachieved goal independently, ignoring potential negative interactions (e.g., multiple rovers needing the same resource or path simultaneously) or positive interactions (e.g., one move satisfying needs for multiple goals). This makes the heuristic non-admissible but potentially good for greedy search.
    - If a rover needs to sample and its store is full, one `drop` action is added to the cost. This assumes the rover has a store and can perform the drop.
    - If a goal requires reaching a waypoint or set of waypoints unreachable by any suitable rover, the heuristic for that goal (and thus the total heuristic) is infinite.
    - Object types (rovers, waypoints, etc.) are inferred from initial state and static facts.

    # Heuristic Initialization
    The heuristic precomputes and stores the following information from the task definition (`task.initial_state` and `task.static`):
    - Rover capabilities (equipped for soil, rock, imaging).
    - Mapping of stores to rovers.
    - Information about cameras (which rover they are on, modes supported, calibration target).
    - Mapping of objectives to waypoints from which they are visible.
    - The lander's location.
    - The set of communication waypoints (visible from the lander's location).
    - The initial locations of soil and rock samples.
    - Shortest path distances between all pairs of waypoints for each rover, based on their specific `can_traverse` and `visible` links. This is done using BFS on the directed graph for each rover.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current location of each rover, the status of stores (full/empty), which rovers have collected which data/images, and which cameras are calibrated.
    2. Initialize the total heuristic cost to 0.
    3. Iterate through each goal fact defined in the task.
    4. If a goal fact is already true in the current state, skip it.
    5. If a goal fact is not true, calculate the estimated minimum cost to achieve *only* this goal fact, and add it to the total cost.
       - For a `(communicated_soil_data W)` goal:
         - Check if any rover currently has `(have_soil_analysis R W)`.
         - Option A (Communicate existing data): If data exists, find the minimum cost among all rovers having the data to move to a communication waypoint and communicate (shortest_path_cost + 1).
         - Option B (Sample and Communicate): If data doesn't exist, find the minimum cost among all suitable rovers (equipped for soil, has store) to move to W, sample (add 1 for sample action, plus 1 for drop if the rover's store is full), move from W to a communication waypoint, and communicate (add 1 for communicate action). The move costs are shortest paths.
         - The cost for this goal is the minimum of Option A and Option B.
       - For a `(communicated_rock_data W)` goal: Similar logic to soil data.
       - For a `(communicated_image_data O M)` goal:
         - Check if any rover currently has `(have_image R O M)`.
         - Option A (Communicate existing data): If image exists, find the minimum cost among all rovers having the image to move to a communication waypoint and communicate (shortest_path_cost + 1).
         - Option B (Image and Communicate): If image doesn't exist, find the minimum cost among all suitable rover/camera pairs (rover equipped for imaging, camera on board, camera supports mode) to move to a calibration waypoint, calibrate (add 1), move from the calibration waypoint to an imaging waypoint for O, take the image (add 1), move from the imaging waypoint to a communication waypoint, and communicate (add 1). Move costs are shortest paths between the specific waypoints chosen in the sequence.
         - The cost for this goal is the minimum of Option A and Option B.
    6. If the calculated cost for any unachieved goal is infinite (meaning it's impossible to reach a required location), the total heuristic is infinite.
    7. Return the total accumulated cost.
    """
    def __init__(self, task):
        self.goals = task.goals
        self.initial_state = task.initial_state # Need initial state for static samples

        # Extract objects and static info
        self.rovers = set()
        self.waypoints = set()
        self.stores = set()
        self.cameras = set()
        self.modes = set()
        self.landers = set()
        self.objectives = set()

        # Attempt to extract objects from initial state facts (might not be exhaustive)
        for fact in task.initial_state:
             parts = get_parts(fact)
             if not parts: continue
             pred = parts[0]
             if pred == "at": self.rovers.add(parts[1]); self.waypoints.add(parts[2])
             elif pred == "at_lander": self.landers.add(parts[1]); self.waypoints.add(parts[2])
             elif pred == "can_traverse": self.rovers.add(parts[1]); self.waypoints.add(parts[2]); self.waypoints.add(parts[3])
             elif pred == "equipped_for_soil_analysis": self.rovers.add(parts[1])
             elif pred == "equipped_for_rock_analysis": self.rovers.add(parts[1])
             elif pred == "equipped_for_imaging": self.rovers.add(parts[1])
             elif pred == "empty" or pred == "full": self.stores.add(parts[1])
             elif pred == "have_rock_analysis": self.rovers.add(parts[1]); self.waypoints.add(parts[2])
             elif pred == "have_soil_analysis": self.rovers.add(parts[1]); self.waypoints.add(parts[2])
             elif pred == "calibrated": self.cameras.add(parts[1]); self.rovers.add(parts[2])
             elif pred == "supports": self.cameras.add(parts[1]); self.modes.add(parts[2])
             elif pred == "visible": self.waypoints.add(parts[1]); self.waypoints.add(parts[2])
             elif pred == "have_image": self.rovers.add(parts[1]); self.objectives.add(parts[2]); self.modes.add(parts[3])
             elif pred == "communicated_soil_data": self.waypoints.add(parts[1])
             elif pred == "communicated_rock_data": self.waypoints.add(parts[1])
             elif pred == "communicated_image_data": self.objectives.add(parts[1]); self.modes.add(parts[2])
             elif pred == "at_soil_sample": self.waypoints.add(parts[1])
             elif pred == "at_rock_sample": self.waypoints.add(parts[1])
             elif pred == "visible_from": self.objectives.add(parts[1]); self.waypoints.add(parts[2])
             elif pred == "store_of": self.stores.add(parts[1]); self.rovers.add(parts[2])
             elif pred == "calibration_target": self.cameras.add(parts[1]); self.objectives.add(parts[2])
             elif pred == "on_board": self.cameras.add(parts[1]); self.rovers.add(parts[2])

        # Use all objects mentioned in goals as well
        for goal in self.goals:
             parts = get_parts(goal)
             if not parts: continue
             pred = parts[0]
             if pred == "communicated_soil_data": self.waypoints.add(parts[1])
             elif pred == "communicated_rock_data": self.waypoints.add(parts[1])
             elif pred == "communicated_image_data": self.objectives.add(parts[1]); self.modes.add(parts[2])

        # Static info maps
        self.rover_capabilities = defaultdict(set)
        self.store_map = {} # store -> rover
        self.camera_map = {} # camera -> {rover, modes, target}
        self.objective_visible_from = defaultdict(set) # objective -> {waypoint}
        self.lander_location = None
        self.initial_soil_samples = set()
        self.initial_rock_samples = set()

        visible_graph = defaultdict(set) # Undirected
        rover_traverse_graph = defaultdict(lambda: defaultdict(set)) # rover -> waypoint -> {neighbor} (Directed)

        for fact in task.static:
            parts = get_parts(fact)
            if not parts: continue
            pred = parts[0]
            if pred == "equipped_for_soil_analysis": self.rover_capabilities[parts[1]].add("soil")
            elif pred == "equipped_for_rock_analysis": self.rover_capabilities[parts[1]].add("rock")
            elif pred == "equipped_for_imaging": self.rover_capabilities[parts[1]].add("imaging")
            elif pred == "store_of": self.store_map[parts[1]] = parts[2]
            elif pred == "on_board":
                camera, rover = parts[1], parts[2]
                if camera not in self.camera_map: self.camera_map[camera] = {'rover': None, 'modes': set(), 'target': None}
                self.camera_map[camera]['rover'] = rover
            elif pred == "supports":
                camera, mode = parts[1], parts[2]
                if camera not in self.camera_map: self.camera_map[camera] = {'rover': None, 'modes': set(), 'target': None}
                self.camera_map[camera]['modes'].add(mode)
            elif pred == "calibration_target":
                camera, target = parts[1], parts[2]
                if camera not in self.camera_map: self.camera_map[camera] = {'rover': None, 'modes': set(), 'target': None}
                self.camera_map[camera]['target'] = target
            elif pred == "visible":
                w1, w2 = parts[1], parts[2]
                visible_graph[w1].add(w2)
                visible_graph[w2].add(w1) # Assuming visible is symmetric
            elif pred == "can_traverse":
                 rover, w1, w2 = parts[1], parts[2], parts[3]
                 # Add edge only if visible as well
                 if w2 in visible_graph.get(w1, set()):
                     rover_traverse_graph[rover][w1].add(w2)
            elif pred == "visible_from":
                objective, waypoint = parts[1], parts[2]
                self.objective_visible_from[objective].add(waypoint)
            elif pred == "at_lander":
                self.lander_location = parts[2]

        # Get initial sample locations from initial state (they are consumed)
        for fact in task.initial_state:
             if match(fact, "at_soil_sample", "*"):
                 self.initial_soil_samples.add(get_parts(fact)[1])
             elif match(fact, "at_rock_sample", "*"):
                 self.initial_rock_samples.add(get_parts(fact)[1])

        # Identify communication waypoints (visible from lander location)
        self.communication_waypoints = visible_graph.get(self.lander_location, set())

        # Precompute shortest paths for each rover
        all_waypoints_list = list(self.waypoints) # Use all identified waypoints
        self.rover_shortest_paths = {}
        for rover in self.rovers:
             self.rover_shortest_paths[rover] = {}
             graph_for_bfs = rover_traverse_graph.get(rover, {})
             # Ensure all waypoints are in the graph keys for BFS
             full_graph_keys = {w: set() for w in all_waypoints_list}
             full_graph_keys.update(graph_for_bfs)

             for start_w in all_waypoints_list:
                  self.rover_shortest_paths[rover][start_w] = bfs(full_graph_keys, start_w)

        # Map rover to its store
        self.rover_to_store = {v: k for k, v in self.store_map.items()}


    def get_rover_location(self, state_facts, rover):
        """Find the current location of a rover in the state."""
        for fact in state_facts:
            if match(fact, "at", rover, "*"):
                return get_parts(fact)[2]
        return None # Should not happen in a valid state

    def get_min_dist(self, rover, start_w, target_waypoints):
        """Find the minimum shortest path distance from start_w to any target_waypoint for the given rover."""
        if start_w is None or rover not in self.rover_shortest_paths or start_w not in self.rover_shortest_paths[rover]:
             return float('inf') # Rover location unknown or not in graph

        min_d = float('inf')
        distances_from_start = self.rover_shortest_paths[rover].get(start_w, {}) # Use .get for safety

        for target_w in target_waypoints:
            if target_w in distances_from_start:
                 min_d = min(min_d, distances_from_start[target_w])
        return min_d

    def find_nearest_waypoint(self, rover, start_w, target_waypoints):
        """Find the specific waypoint in a target set that is nearest to the start for the given rover."""
        min_d = float('inf')
        nearest_w = None
        if start_w is None or rover not in self.rover_shortest_paths or start_w not in self.rover_shortest_paths[rover]:
             return float('inf'), None

        distances_from_start = self.rover_shortest_paths[rover].get(start_w, {}) # Use .get for safety

        for target_w in target_waypoints:
            if target_w in distances_from_start and distances_from_start[target_w] < min_d:
                 min_d = distances_from_start[target_w]
                 nearest_w = target_w
        return min_d, nearest_w


    def __call__(self, node):
        state = node.state
        total_cost = 0

        # Pre-process state for quick lookups
        state_facts = set(state)
        rover_locations = {r: self.get_rover_location(state_facts, r) for r in self.rovers}
        rover_has_soil = defaultdict(set) # rover -> {waypoint}
        rover_has_rock = defaultdict(set) # rover -> {waypoint}
        rover_has_image = defaultdict(set) # rover -> {(objective, mode)} pairs
        store_is_full = {} # store -> bool
        camera_is_calibrated = {} # (camera, rover) -> bool

        for fact in state_facts:
            parts = get_parts(fact)
            if not parts: continue

            pred = parts[0]
            if pred == "have_soil_analysis":
                rover_has_soil[parts[1]].add(parts[2])
            elif pred == "have_rock_analysis":
                rover_has_rock[parts[1]].add(parts[2])
            elif pred == "have_image":
                rover_has_image[parts[1]].add((parts[2], parts[3]))
            elif pred == "full":
                store_is_full[parts[1]] = True
            elif pred == "empty":
                 store_is_full[parts[1]] = False # Explicitly mark empty
            elif pred == "calibrated":
                camera_is_calibrated[(parts[1], parts[2])] = True # (camera, rover) -> bool

        # Calculate cost for each unachieved goal
        for goal in self.goals:
            if goal in state_facts:
                continue # Goal already achieved

            parts = get_parts(goal)
            if not parts: continue

            pred = parts[0]

            if pred == "communicated_soil_data":
                waypoint = parts[1]

                # Option 1: Communicate existing data
                min_comm_existing_cost = float('inf')
                rovers_with_data = [r for r in self.rovers if waypoint in rover_has_soil[r]]

                if rovers_with_data:
                    for r in rovers_with_data:
                        current_w = rover_locations[r]
                        move_cost = self.get_min_dist(r, current_w, self.communication_waypoints)
                        if move_cost != float('inf'):
                             min_comm_existing_cost = min(min_comm_existing_cost, move_cost + 1) # move + communicate

                # Option 2: Sample and communicate
                min_sample_and_comm_cost = float('inf')

                # Find suitable rovers: equipped for soil, has a store
                suitable_rovers = [r for r in self.rovers if "soil" in self.rover_capabilities.get(r, set()) and self.rover_to_store.get(r) is not None]

                if suitable_rovers:
                    for r in suitable_rovers:
                         current_w = rover_locations[r]
                         rover_store = self.rover_to_store.get(r)

                         # Cost to sample: move to waypoint + potential drop + sample action
                         sample_move_cost, sample_w = self.find_nearest_waypoint(r, current_w, {waypoint})
                         if sample_move_cost == float('inf'): continue # Cannot reach sample location

                         drop_cost = 1 if rover_store and store_is_full.get(rover_store, False) else 0
                         sample_action_cost = 1 # sample_soil action

                         # Cost to communicate (after sampling at 'waypoint'): move from 'waypoint' + communicate action
                         # Assume rover is at 'sample_w' after sampling
                         comm_move_cost = self.get_min_dist(r, sample_w, self.communication_waypoints)
                         if comm_move_cost == float('inf'): continue # Cannot reach communication waypoint

                         communicate_action_cost = 1 # communicate_soil_data action

                         task_cost = sample_move_cost + drop_cost + sample_action_cost + comm_move_cost + communicate_action_cost
                         min_sample_and_comm_cost = min(min_sample_and_comm_cost, task_cost)

                # The cost for this goal is the minimum of the two options
                goal_cost = min(min_comm_existing_cost, min_sample_and_comm_cost)

                if goal_cost == float('inf'): return float('inf') # Impossible goal
                total_cost += goal_cost


            elif pred == "communicated_rock_data":
                waypoint = parts[1]

                # Option 1: Communicate existing data
                min_comm_existing_cost = float('inf')
                rovers_with_data = [r for r in self.rovers if waypoint in rover_has_rock[r]]

                if rovers_with_data:
                    for r in rovers_with_data:
                        current_w = rover_locations[r]
                        move_cost = self.get_min_dist(r, current_w, self.communication_waypoints)
                        if move_cost != float('inf'):
                             min_comm_existing_cost = min(min_comm_existing_cost, move_cost + 1) # move + communicate

                # Option 2: Sample and communicate
                min_sample_and_comm_cost = float('inf')

                # Find suitable rovers: equipped for rock, has a store
                suitable_rovers = [r for r in self.rovers if "rock" in self.rover_capabilities.get(r, set()) and self.rover_to_store.get(r) is not None]

                if suitable_rovers:
                    for r in suitable_rovers:
                         current_w = rover_locations[r]
                         rover_store = self.rover_to_store.get(r)

                         # Cost to sample: move to waypoint + potential drop + sample action
                         sample_move_cost, sample_w = self.find_nearest_waypoint(r, current_w, {waypoint})
                         if sample_move_cost == float('inf'): continue # Cannot reach sample location

                         drop_cost = 1 if rover_store and store_is_full.get(rover_store, False) else 0
                         sample_action_cost = 1 # sample_rock action

                         # Cost to communicate (after sampling at 'waypoint'): move from 'waypoint' + communicate action
                         # Assume rover is at 'sample_w' after sampling
                         comm_move_cost = self.get_min_dist(r, sample_w, self.communication_waypoints)
                         if comm_move_cost == float('inf'): continue # Cannot reach communication waypoint

                         communicate_action_cost = 1 # communicate_rock_data action

                         task_cost = sample_move_cost + drop_cost + sample_action_cost + comm_move_cost + communicate_action_cost
                         min_sample_and_comm_cost = min(min_sample_and_comm_cost, task_cost)

                # The cost for this goal is the minimum of the two options
                goal_cost = min(min_comm_existing_cost, min_sample_and_comm_cost)

                if goal_cost == float('inf'): return float('inf') # Impossible goal
                total_cost += goal_cost


            elif pred == "communicated_image_data":
                objective, mode = parts[1], parts[2]

                # Option 1: Communicate existing data
                min_comm_existing_cost = float('inf')
                rovers_with_data = [r for r in self.rovers if (objective, mode) in rover_has_image[r]]

                if rovers_with_data:
                    for r in rovers_with_data:
                        current_w = rover_locations[r]
                        move_cost = self.get_min_dist(r, current_w, self.communication_waypoints)
                        if move_cost != float('inf'):
                             min_comm_existing_cost = min(min_comm_existing_cost, move_cost + 1) # move + communicate

                # Option 2: Take image and communicate
                min_image_and_comm_cost = float('inf')

                # Find suitable rovers/cameras: rover equipped for imaging, camera on board rover, camera supports mode
                suitable_rover_camera_pairs = []
                for r in self.rovers:
                     if "imaging" in self.rover_capabilities.get(r, set()):
                          for cam_name, cam_info in self.camera_map.items():
                               if cam_info['rover'] == r and mode in cam_info['modes']:
                                    suitable_rover_camera_pairs.append((r, cam_name))

                if suitable_rover_camera_pairs:
                    for r, cam_name in suitable_rover_camera_pairs:
                         current_w = rover_locations[r]
                         camera_info = self.camera_map[cam_name]
                         calibration_target = camera_info['target']

                         # Cost to calibrate: move to calibration target waypoint + calibrate action
                         calibration_waypoints = self.objective_visible_from.get(calibration_target, set())
                         if not calibration_waypoints: continue # Cannot calibrate this camera

                         cal_move_cost, nearest_cal_w = self.find_nearest_waypoint(r, current_w, calibration_waypoints)
                         if cal_move_cost == float('inf'): continue # Cannot reach calibration waypoint

                         calibrate_action_cost = 1 # calibrate action

                         # Cost to take image: move from calibration waypoint to imaging waypoint + take_image action
                         imaging_waypoints = self.objective_visible_from.get(objective, set())
                         if not imaging_waypoints: continue # Cannot image this objective

                         # Assume rover is at 'nearest_cal_w' after calibrating
                         img_move_cost, nearest_img_w = self.find_nearest_waypoint(r, nearest_cal_w, imaging_waypoints)
                         if img_move_cost == float('inf'): continue # Cannot reach imaging waypoint from calibration waypoint

                         take_image_action_cost = 1 # take_image action

                         # Cost to communicate (after taking image at imaging waypoint): move from imaging waypoint + communicate action
                         # Assume rover is at 'nearest_img_w' after taking image
                         comm_move_cost = self.get_min_dist(r, nearest_img_w, self.communication_waypoints)
                         if comm_move_cost == float('inf'): continue # Cannot reach communication waypoint from imaging waypoint

                         communicate_action_cost = 1 # communicate_image_data action

                         task_cost = cal_move_cost + calibrate_action_cost + img_move_cost + take_image_action_cost + comm_move_cost + communicate_action_cost
                         min_image_and_comm_cost = min(min_image_and_comm_cost, task_cost)

                # The cost for this goal is the minimum of the two options
                goal_cost = min(min_comm_existing_cost, min_image_and_comm_cost)

                if goal_cost == float('inf'): return float('inf') # Impossible goal
                total_cost += goal_cost

        # Return infinity if any goal was impossible
        if total_cost == float('inf'):
             return float('inf')

        return total_cost
