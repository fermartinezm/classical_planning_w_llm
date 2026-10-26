from collections import deque, defaultdict
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential leading/trailing whitespace or malformed facts defensively
    fact = fact.strip()
    if not fact.startswith('(') or not fact.endswith(')'):
         return [] # Indicate invalid fact
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if not parts or len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def shortest_path(graph, start_node, target_nodes):
    """
    Find the shortest path distance from a start_node to any node in target_nodes
    using BFS.

    Args:
        graph: A dictionary representing the graph {node: set(neighbors)}.
        start_node: The starting node.
        target_nodes: A set of target nodes.

    Returns:
        The shortest distance, or float('inf') if no target is reachable.
    """
    if start_node in target_nodes:
        return 0

    queue = deque([(start_node, 0)])
    visited = {start_node}

    while queue:
        current_node, dist = queue.popleft()

        # Check neighbors
        if current_node in graph:
            for neighbor in graph[current_node]:
                if neighbor in target_nodes:
                    return dist + 1
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))

    return float('inf') # Not reachable

def shortest_path_set(graph, start_nodes, target_nodes):
    """
    Find the minimum shortest path distance from any node in start_nodes
    to any node in target_nodes.

    Args:
        graph: A dictionary representing the graph {node: set(neighbors)}.
        start_nodes: A set of starting nodes.
        target_nodes: A set of target nodes.

    Returns:
        The minimum shortest distance, or float('inf') if no target is reachable
        from any start node.
    """
    if not start_nodes or not target_nodes:
        return float('inf')

    # Optimize: If any start node is a target node, distance is 0
    if any(start in target_nodes for start in start_nodes):
        return 0

    # Use a multi-source BFS
    queue = deque([(node, 0) for node in start_nodes if node in graph]) # Start only from nodes present in graph
    visited = set(start_nodes)

    while queue:
        current_node, dist = queue.popleft()

        # Check neighbors
        if current_node in graph: # Ensure current_node is still valid in graph
            for neighbor in graph[current_node]:
                if neighbor in target_nodes:
                    return dist + 1 # Found the shortest path from any source to any target
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))

    return float('inf') # No target reachable from any start node


class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic function for the PDDL rovers domain.

    # Summary
    This heuristic estimates the cost to reach the goal state by summing the
    estimated costs for each uncommunicated goal fact. The cost for each goal
    is estimated based on the minimum number of actions and movements required
    to achieve that specific goal, considering the current state of rovers,
    samples, images, and cameras. Movement costs are estimated using shortest
    paths on the traversability graph for each rover.

    # Assumptions
    - The heuristic assumes that each uncommunicated goal can be pursued
      independently by the most suitable available rover.
    - It estimates the cost for a goal by finding the minimum cost path
      through necessary stages:
        - For soil/rock: Go to sample location -> Sample -> Go to communication location -> Communicate.
        - For image: Go to calibration location -> Calibrate -> Go to image location -> Take Image -> Go to communication location -> Communicate.
    - It uses shortest path distances (number of navigation actions) for movement costs.
    - It adds a fixed cost (1) for each necessary action (sample, drop, calibrate, take_image, communicate).
    - It assumes that if a sample is no longer at its initial location, it must have been picked up by a rover (and is held as a `have_soil_analysis` or `have_rock_analysis` fact).
    - It assumes recalibration is needed before taking an image if the camera is not currently calibrated, unless the rover is already at a suitable image waypoint. Taking an image always uncalibrates the camera.
    - If a goal is determined to be unreachable by any suitable rover, a large penalty is added.

    # Heuristic Initialization
    The heuristic initializes by processing the static facts from the task:
    - Identifies the lander's location and the waypoints visible from it (communication points).
    - Records which rovers are equipped for soil, rock, and imaging analysis.
    - Maps rovers to their respective store objects.
    - Stores information about cameras: which rover they are on, their calibration target, and supported modes.
    - Maps objectives to waypoints from which they are visible (image points).
    - Maps calibration targets to waypoints from which they are visible (calibration points).
    - Builds a traversability graph for each rover based on `can_traverse` and `visible` facts.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic calculates the total cost as follows:
    1. Identify all goal facts that are not currently true in the state.
    2. For each uncommunicated goal fact:
        a. Initialize the minimum cost for this goal to infinity.
        b. If the goal is `(communicated_soil_data ?w)`:
            - Find all rovers equipped for soil analysis.
            - For each equipped rover `?r`:
                - Get the rover's current location `?r_loc`.
                - If `?r` currently has `(have_soil_analysis ?r ?w)`:
                    - Calculate cost: `shortest_path_set(rover_graph[?r], {?r_loc}, comm_wps) + 1` (communicate).
                    - Update minimum cost for this goal.
                - Else (`?r` does not have the sample):
                    - If `(at_soil_sample ?w)` is true in the state:
                        - Get `?r`'s store `?s`. Check if `(full ?s)` is true (cost 1 for drop, else 0).
                        - Calculate cost: `shortest_path_set(rover_graph[?r], {?r_loc}, {?w})` (to sample) + 1 (sample) + (drop cost) + `shortest_path_set(rover_graph[?r], {?w}, comm_wps)` (to communicate) + 1 (communicate).
                        - Update minimum cost for this goal.
        c. If the goal is `(communicated_rock_data ?w)`:
            - Follow the same logic as for soil data, using rock-specific predicates and equipment.
        d. If the goal is `(communicated_image_data ?o ?m)`:
            - Find all rovers equipped for imaging that have a camera supporting mode `?m` with a calibration target `?t`.
            - For each such rover `?r` with camera `?i` and target `?t`:
                - Get the rover's current location `?r_loc`.
                - If `?r` currently has `(have_image ?r ?o ?m)`:
                    - Calculate cost: `shortest_path_set(rover_graph[?r], {?r_loc}, comm_wps) + 1` (communicate).
                    - Update minimum cost for this goal.
                - Else (`?r` does not have the image):
                    - Get calibration waypoints `?cal_wps` for target `?t`.
                    - Get image waypoints `?img_wps` for objective `?o`.

                    - Calculate cost:
                        cost_to_get_image_and_comm = float('inf')

                        # Cost = move_to_cal + calibrate + move_to_img + take_image + move_to_comm + communicate
                        cost_to_cal = shortest_path_set(rover_graph[?r], {r_loc}, cal_wps)
                        cost_cal_to_img = shortest_path_set(rover_graph[?r], cal_wps, img_wps)
                        cost_img_to_comm = shortest_path_set(rover_graph[?r], img_wps, self.comm_wps)

                        if cost_to_cal != float('inf') and cost_cal_to_img != float('inf') and cost_img_to_comm != float('inf'):
                            cost_to_get_image_and_comm = cost_to_cal + 1 + cost_cal_to_img + 1 + cost_img_to_comm + 1

                        min_goal_cost = min(min_goal_cost, cost_to_get_image_and_comm)

        e. If the minimum cost for a goal remains infinity, it is unreachable. Add a large penalty (e.g., 1000) to the total cost.
    3. The total heuristic value is the sum of the minimum costs for all uncommunicated goals.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.task = task # Store task for goal_reached check if needed
        self.goals = task.goals
        static_facts = task.static

        # Process static facts
        self.lander_location = None
        self.comm_wps = set()
        self.rover_equipment = defaultdict(set)
        self.rover_stores = {}
        self.camera_info = {} # camera -> (rover, target, modes)
        self.visible_from_objective = defaultdict(set) # objective -> set(waypoints)
        self.visible_from_calibration_target = defaultdict(set) # target -> set(waypoints)
        self.rover_graph = defaultdict(lambda: defaultdict(set)) # rover -> {wp -> set(neighbors)}
        self.initial_soil_samples = set() # Waypoints with soil samples initially
        self.initial_rock_samples = set() # Waypoints with rock samples initially

        # Temporary storage for camera info before combining
        camera_on_board = {} # camera -> rover
        camera_target = {} # camera -> target
        camera_modes = defaultdict(set) # camera -> set(modes)

        # First pass to get basic info and visible pairs
        visible_pairs = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue # Skip invalid facts

            predicate = parts[0]

            if predicate == "at_lander":
                self.lander_location = parts[2]
            elif predicate == "visible":
                wp1, wp2 = parts[1], parts[2]
                visible_pairs.add((wp1, wp2))
            elif predicate == "equipped_for_soil_analysis":
                self.rover_equipment[parts[1]].add("soil")
            elif predicate == "equipped_for_rock_analysis":
                self.rover_equipment[parts[1]].add("rock")
            elif predicate == "equipped_for_imaging":
                self.rover_equipment[parts[1]].add("imaging")
            elif predicate == "store_of":
                self.rover_stores[parts[2]] = parts[1] # rover -> store
            elif predicate == "on_board":
                camera_on_board[parts[1]] = parts[2] # camera -> rover
            elif predicate == "calibration_target":
                camera_target[parts[1]] = parts[2] # camera -> target
            elif predicate == "supports":
                camera_modes[parts[1]].add(parts[2]) # camera -> mode
            elif predicate == "visible_from":
                # This predicate is used for both objectives and calibration targets
                obj_or_target, wp = parts[1], parts[2]
                self.visible_from_objective[obj_or_target].add(wp) # Store for objectives
                # We'll populate visible_from_calibration_target later based on known targets
            elif predicate == "at_soil_sample":
                 self.initial_soil_samples.add(parts[1])
            elif predicate == "at_rock_sample":
                 self.initial_rock_samples.add(parts[1])

        # Build the actual traversability graph considering both can_traverse and visible
        self.rover_graph = defaultdict(lambda: defaultdict(set))
        for fact in static_facts:
            if match(fact, "can_traverse", "*", "*", "*"):
                rover, wp1, wp2 = get_parts(fact)[1], get_parts(fact)[2], get_parts(fact)[3]
                if (wp1, wp2) in visible_pairs:
                     self.rover_graph[rover][wp1].add(wp2)

        # Determine communication waypoints based on lander location and visibility
        if self.lander_location:
             for wp1, wp2 in visible_pairs:
                 if wp2 == self.lander_location:
                     self.comm_wps.add(wp1)
                 if wp1 == self.lander_location: # Visibility is symmetric
                     self.comm_wps.add(wp2)

        # Combine camera info and populate visible_from_calibration_target
        calibration_targets = set(camera_target.values())
        for camera, rover in camera_on_board.items():
            target = camera_target.get(camera)
            modes = camera_modes.get(camera, set())
            if target is not None: # Camera must have a calibration target
                 self.camera_info[camera] = (rover, target, modes)

        # Now populate visible_from_calibration_target using the already collected visible_from facts
        for target in calibration_targets:
             # visible_from_objective already holds facts like (visible_from target_obj wp)
             if target in self.visible_from_objective:
                 self.visible_from_calibration_target[target] = self.visible_from_objective[target]


    def __call__(self, node):
        """Estimate the minimum cost to reach the goal state."""
        state = node.state

        # Identify current state facts
        current_locations = {} # rover -> waypoint
        full_stores = set() # store objects that are full
        have_soil = defaultdict(set) # rover -> set(waypoints)
        have_rock = defaultdict(set) # rover -> set(waypoints)
        have_image = defaultdict(set) # rover -> set((objective, mode))
        calibrated_cameras = set() # set((camera, rover))
        current_soil_samples = set() # Waypoints with soil samples
        current_rock_samples = set() # Waypoints with rock samples

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]

            if predicate == "at":
                current_locations[parts[1]] = parts[2] # rover -> waypoint
            elif predicate == "full":
                full_stores.add(parts[1]) # store
            elif predicate == "have_soil_analysis":
                have_soil[parts[1]].add(parts[2]) # rover -> waypoint
            elif predicate == "have_rock_analysis":
                have_rock[parts[1]].add(parts[2]) # rover -> waypoint
            elif predicate == "have_image":
                if len(parts) == 4: # (have_image rover objective mode)
                    have_image[parts[1]].add((parts[2], parts[3])) # rover -> (objective, mode)
            elif predicate == "calibrated":
                if len(parts) == 3: # (calibrated camera rover)
                    calibrated_cameras.add((parts[1], parts[2])) # (camera, rover)
            elif predicate == "at_soil_sample":
                 current_soil_samples.add(parts[1])
            elif predicate == "at_rock_sample":
                 current_rock_samples.add(parts[1])


        total_cost = 0
        large_penalty = 1000 # Penalty for seemingly unreachable goals

        # Iterate through goal facts
        for goal in self.goals:
            if goal in state:
                continue # Goal already achieved

            parts = get_parts(goal)
            if not parts: continue

            predicate = parts[0]

            if predicate == "communicated_soil_data":
                waypoint = parts[1]
                min_goal_cost = float('inf')

                # Find equipped rovers
                equipped_rovers = [r for r, equip in self.rover_equipment.items() if "soil" in equip]

                if not equipped_rovers:
                    total_cost += large_penalty # No rover can ever sample soil
                    continue

                for rover in equipped_rovers:
                    r_loc = current_locations.get(rover)
                    if r_loc is None: continue # Rover location unknown (shouldn't happen in valid states)

                    rover_graph = self.rover_graph.get(rover)
                    if not rover_graph: continue # Rover cannot traverse

                    # Case 1: Rover already has the sample analysis
                    if waypoint in have_soil.get(rover, set()):
                        cost_to_comm = shortest_path_set(rover_graph, {r_loc}, self.comm_wps)
                        if cost_to_comm != float('inf'):
                            min_goal_cost = min(min_goal_cost, cost_to_comm + 1) # +1 for communicate

                    # Case 2: Rover needs to sample and then communicate
                    elif waypoint in current_soil_samples:
                        store = self.rover_stores.get(rover)
                        if store is None: continue # Rover has no store

                        drop_cost = 1 if store in full_stores else 0

                        cost_to_sample = shortest_path_set(rover_graph, {r_loc}, {waypoint})
                        cost_sample_to_comm = shortest_path_set(rover_graph, {waypoint}, self.comm_wps)

                        if cost_to_sample != float('inf') and cost_sample_to_comm != float('inf'):
                             # Cost = move_to_sample + sample + (drop if needed) + move_to_comm + communicate
                            cost = cost_to_sample + 1 + drop_cost + cost_sample_to_comm + 1
                            min_goal_cost = min(min_goal_cost, cost)

                if min_goal_cost == float('inf'):
                    total_cost += large_penalty # Goal seems unreachable in this state
                else:
                    total_cost += min_goal_cost

            elif predicate == "communicated_rock_data":
                waypoint = parts[1]
                min_goal_cost = float('inf')

                # Find equipped rovers
                equipped_rovers = [r for r, equip in self.rover_equipment.items() if "rock" in equip]

                if not equipped_rovers:
                    total_cost += large_penalty # No rover can ever sample rock
                    continue

                for rover in equipped_rovers:
                    r_loc = current_locations.get(rover)
                    if r_loc is None: continue

                    rover_graph = self.rover_graph.get(rover)
                    if not rover_graph: continue

                    # Case 1: Rover already has the sample analysis
                    if waypoint in have_rock.get(rover, set()):
                        cost_to_comm = shortest_path_set(rover_graph, {r_loc}, self.comm_wps)
                        if cost_to_comm != float('inf'):
                            min_goal_cost = min(min_goal_cost, cost_to_comm + 1) # +1 for communicate

                    # Case 2: Rover needs to sample and then communicate
                    elif waypoint in current_rock_samples:
                        store = self.rover_stores.get(rover)
                        if store is None: continue # Rover has no store

                        drop_cost = 1 if store in full_stores else 0

                        cost_to_sample = shortest_path_set(rover_graph, {r_loc}, {waypoint})
                        cost_sample_to_comm = shortest_path_set(rover_graph, {waypoint}, self.comm_wps)

                        if cost_to_sample != float('inf') and cost_sample_to_comm != float('inf'):
                             # Cost = move_to_sample + sample + (drop if needed) + move_to_comm + communicate
                            cost = cost_to_sample + 1 + drop_cost + cost_sample_to_comm + 1
                            min_goal_cost = min(min_goal_cost, cost)

                if min_goal_cost == float('inf'):
                    total_cost += large_penalty # Goal seems unreachable in this state
                else:
                    total_cost += min_goal_cost

            elif predicate == "communicated_image_data":
                objective, mode = parts[1], parts[2]
                min_goal_cost = float('inf')

                # Find suitable rovers (equipped for imaging, with camera supporting mode and target)
                suitable_rovers = [] # List of (rover, camera, target) tuples
                for camera, (rover, target, modes) in self.camera_info.items():
                    if mode in modes and "imaging" in self.rover_equipment.get(rover, set()):
                         suitable_rovers.append((rover, camera, target))

                if not suitable_rovers:
                    total_cost += large_penalty # No rover can ever take this image
                    continue

                for rover, camera, target in suitable_rovers:
                    r_loc = current_locations.get(rover)
                    if r_loc is None: continue

                    rover_graph = self.rover_graph.get(rover)
                    if not rover_graph: continue

                    # Case 1: Rover already has the image
                    if (objective, mode) in have_image.get(rover, set()):
                        cost_to_comm = shortest_path_set(rover_graph, {r_loc}, self.comm_wps)
                        if cost_to_comm != float('inf'):
                            min_goal_cost = min(min_goal_cost, cost_to_comm + 1) # +1 for communicate

                    # Case 2: Rover needs to take the image and then communicate
                    else:
                        cal_wps = self.visible_from_calibration_target.get(target, set())
                        img_wps = self.visible_from_objective.get(objective, set())

                        if not cal_wps or not img_wps:
                             # Cannot calibrate or take image of this objective/target
                             continue

                        # Cost = move_to_cal + calibrate + move_to_img + take_image + move_to_comm + communicate
                        cost_to_cal = shortest_path_set(rover_graph, {r_loc}, cal_wps)
                        cost_cal_to_img = shortest_path_set(rover_graph, cal_wps, img_wps)
                        cost_img_to_comm = shortest_path_set(rover_graph, img_wps, self.comm_wps)

                        if cost_to_cal != float('inf') and cost_cal_to_img != float('inf') and cost_img_to_comm != float('inf'):
                            cost = cost_to_cal + 1 + cost_cal_to_img + 1 + cost_img_to_comm + 1
                            min_goal_cost = min(min_goal_cost, cost)

                if min_goal_cost == float('inf'):
                    total_cost += large_penalty # Goal seems unreachable in this state
                else:
                    total_cost += min_goal_cost

            # Add other goal types if necessary (though rovers only has communicated_X_data)
            # else:
            #     # Unknown goal type, maybe add a penalty?
            #     total_cost += large_penalty

        # Ensure heuristic is 0 only at goal state.
        # For this domain, if total_cost is 0, it means all communicated_X_data goals
        # are in the state, which is the definition of the goal state.
        # So, total_cost == 0 implies self.task.goal_reached(state).
        # No adjustment needed.

        return total_cost
