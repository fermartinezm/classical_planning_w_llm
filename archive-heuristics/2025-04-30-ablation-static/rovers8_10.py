import collections
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL fact strings
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

# Helper function to match PDDL fact strings with patterns
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

# Helper function for graph traversal (BFS)
def bfs_all_pairs_shortest_path(graph, nodes):
    """
    Computes shortest path distances from all nodes to all other nodes in the graph.
    Returns a dictionary dist[start_node][end_node] = distance.
    Returns float('inf') for unreachable pairs.
    """
    dist = {node: {other_node: float('inf') for other_node in nodes} for node in nodes}

    for start_node in nodes:
        dist[start_node][start_node] = 0
        queue = collections.deque([(start_node, 0)])
        visited = {start_node}

        while queue:
            current_node, d = queue.popleft()

            if current_node in graph:
                for neighbor in graph[current_node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        dist[start_node][neighbor] = d + 1
                        queue.append((neighbor, d + 1))
    return dist

# Helper function to find minimum distance from a node to a set of nodes
def min_dist_to_set(dist_map, start_node, end_nodes_set):
    """
    Finds the minimum distance from start_node to any node in end_nodes_set
    using the precomputed distance map.
    Returns float('inf') if end_nodes_set is empty or no path exists.
    """
    if not end_nodes_set:
        return float('inf')
    # Ensure start_node is in the dist_map keys before accessing
    if start_node not in dist_map:
         return float('inf')
    return min(dist_map[start_node].get(end_node, float('inf')) for end_node in end_nodes_set)


class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal
    by summing the estimated costs for each unachieved goal fact. The cost
    for each goal fact is estimated based on the minimum number of actions
    (sample/image/calibrate/communicate/drop) and necessary navigation steps
    required to achieve it from the current state, considering available
    rovers and equipment. It uses precomputed shortest path distances between
    waypoints for each rover.

    # Assumptions
    - The heuristic assumes that all necessary static conditions (like equipment,
      sample locations, visible_from, calibration_target, etc.) are present
      in the initial state if required by a goal.
    - It assumes solvable instances, meaning paths exist between necessary waypoints.
    - It considers the 'drop' action cost (1) if a rover's store is full and
      sampling is required for a goal.
    - It simplifies the image goal process by not explicitly tracking camera
      calibration state changes caused by the 'take_image' action. It assumes
      a camera, once calibrated for a goal, remains calibrated for the purpose
      of estimating the cost for *that specific goal fact*. This is a
      simplification for efficiency and potential non-admissibility.

    # Heuristic Initialization
    - Extracts goal conditions.
    - Extracts static facts: rover equipment, store ownership, camera properties,
      waypoint graph connectivity (`can_traverse`), visibility (`visible`, `visible_from`),
      sample locations (`at_soil_sample`, `at_rock_sample`), lander location (`at_lander`).
    - Builds a waypoint graph for each rover based on `can_traverse`.
    - Collects all unique waypoints mentioned in static facts.
    - Computes all-pairs shortest path distances for each rover's graph using BFS.
    - Identifies communication points (waypoints visible from the lander).
    - Stores relevant static information in dictionaries/sets for quick lookup.

    # Step-By-Step Thinking for Computing Heuristic
    The heuristic value for a state is the sum of estimated costs for each goal
    fact that is not yet satisfied in the state.

    For each unachieved goal `G`:

    1.  **Identify Goal Type**: Determine if `G` is `communicated_soil_data`,
        `communicated_rock_data`, or `communicated_image_data`.

    2.  **Estimate Cost for Soil/Rock Goals (`communicated_soil_data W` or `communicated_rock_data W`)**:
        -   Find the lander's location (`LanderLoc`) and the set of communication waypoints (`CommPoints`) visible from `LanderLoc`.
        -   Find all rovers equipped for the required analysis type (soil or rock).
        -   Calculate the minimum cost over all suitable rovers `R`:
            -   Find the current location (`CurrentLoc`) of rover `R`.
            -   If rover `R` already has the required analysis data (`have_soil_analysis R W` or `have_rock_analysis R W` is in the state):
                -   Cost for this rover = 1 (communicate) + shortest_distance(CurrentLoc, CommPoints) for rover `R`.
            -   If rover `R` does NOT have the required analysis data:
                -   Cost for this rover = 1 (sample) + 1 (communicate).
                -   Check if rover R's store is full (`(full S)` where `(store_of S R)`). If yes, add 1 (drop).
                -   Cost += shortest_distance(CurrentLoc, W) for rover `R`.
                -   Cost += shortest_distance(W, CommPoints) for rover `R`.
        -   The estimated cost for goal `G` is the minimum cost found among all suitable rovers. If no suitable rover can complete the necessary navigation, the cost is infinite.

    3.  **Estimate Cost for Image Goals (`communicated_image_data O M`)**:
        -   Find the lander's location (`LanderLoc`) and the set of communication waypoints (`CommPoints`) visible from `LanderLoc`.
        -   Find the set of imaging waypoints (`ImagingPoints`) where objective `O` is visible (`visible_from O P`).
        -   Find all rovers equipped for imaging.
        -   Calculate the minimum cost over all suitable rovers `R`:
            -   Find the current location (`CurrentLoc`) of rover `R`.
            -   Find cameras `I_R` on rover `R` that support mode `M`. If none, this rover is not suitable for this goal.
            -   If rover `R` already has the required image data (`have_image R O M` is in the state):
                -   Cost for this rover = 1 (communicate) + shortest_distance(CurrentLoc, CommPoints) for rover `R`.
            -   If rover `R` does NOT have the required image data:
                -   Cost = 1 (communicate) + 1 (take image).
                -   Calculate the minimum cost over all suitable cameras `I` on rover `R`:
                    -   If camera `I` is calibrated on rover `R` (`calibrated I R` is in the state):
                        -   Cost_I = 1 + 1 + min_{P in ImagingPoints} (shortest_distance(CurrentLoc, P) + shortest_distance(P, CommPoints)) for rover `R`.
                    -   If camera `I` is NOT calibrated on rover `R`:
                        -   Cost_I = 1 + 1 + 1 (calibrate).
                        -   Find the calibration target `T` for camera `I`. Find the set of calibration waypoints (`CalPoints`) where target `T` is visible (`visible_from T W`).
                        -   Cost_I += min_{W in CalPoints, P in ImagingPoints} (shortest_distance(CurrentLoc, W) + shortest_distance(W, P) + shortest_distance(P, CommPoints)) for rover `R`.
                -   The cost for rover `R` is the minimum `Cost_I` found among its suitable cameras. If no suitable camera path exists, the cost for this rover is infinite.
        -   The estimated cost for goal `G` is the minimum cost found among all suitable rovers. If no suitable rover can complete the necessary navigation, the cost is infinite.

    4.  **Sum Costs**: The total heuristic value is the sum of the estimated costs for all unachieved goal facts. If the cost for any unachieved goal is infinite, the total heuristic is infinite.

    Note on shortest_distance(Loc, SetOfLocs): This is computed as min_{TargetLoc in SetOfLocs} shortest_distance(Loc, TargetLoc).
    Note on shortest_distance(SetOfLocs1, SetOfLocs2): This is computed as min_{Loc1 in SetOfLocs1, Loc2 in SetOfLocs2} shortest_distance(Loc1, Loc2).
    The heuristic uses the precomputed all-pairs shortest paths for efficiency.
    If any required waypoint set (CommPoints, ImagingPoints, CalPoints) is empty or unreachable, the cost is considered infinite.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # --- Extract Static Information ---
        self.lander_location = None
        self.rover_equipment = collections.defaultdict(set) # rover -> {soil, rock, imaging}
        self.store_of_rover = {} # store -> rover
        self.rover_of_store = {} # rover -> store
        self.camera_on_board = {} # camera -> rover
        self.camera_supports = collections.defaultdict(set) # camera -> {mode1, mode2, ...}
        self.camera_calibration_target = {} # camera -> objective
        self.waypoint_graph = collections.defaultdict(lambda: collections.defaultdict(list)) # rover -> {from_wp: [to_wp, ...]}
        self.visible_waypoints = collections.defaultdict(set) # wp1 -> {wp2, wp3, ...} (wp2 is visible from wp1)
        self.objective_visible_from = collections.defaultdict(set) # objective -> {wp1, wp2, ...} (objective is visible from wp)
        self.soil_sample_locations = set()
        self.rock_sample_locations = set()
        self.all_waypoints = set() # Collect all waypoints mentioned in static facts

        # Collect all objects of type waypoint from the task definition (more robust)
        # Assuming task object has an 'objects' attribute or similar structure
        # If not, rely on parsing facts, but be aware it might miss isolated waypoints.
        # For this implementation, we rely on waypoints mentioned in static facts.
        # A better approach would parse the full PDDL problem file objects section.
        # Let's stick to parsing facts for simplicity as per example structure.

        for fact in static_facts:
            parts = get_parts(fact)
            predicate = parts[0]

            if predicate == "at_lander":
                self.lander_location = parts[2]
                self.all_waypoints.add(parts[2])
            elif predicate == "equipped_for_soil_analysis":
                self.rover_equipment[parts[1]].add("soil")
            elif predicate == "equipped_for_rock_analysis":
                self.rover_equipment[parts[1]].add("rock")
            elif predicate == "equipped_for_imaging":
                self.rover_equipment[parts[1]].add("imaging")
            elif predicate == "store_of":
                store, rover = parts[1], parts[2]
                self.store_of_rover[store] = rover
                self.rover_of_store[rover] = store
            elif predicate == "on_board":
                camera, rover = parts[1], parts[2]
                self.camera_on_board[camera] = rover
            elif predicate == "supports":
                camera, mode = parts[1], parts[2]
                self.camera_supports[camera].add(mode)
            elif predicate == "calibration_target":
                camera, objective = parts[1], parts[2]
                self.camera_calibration_target[camera] = objective
            elif predicate == "can_traverse":
                rover, wp_from, wp_to = parts[1], parts[2], parts[3]
                self.waypoint_graph[rover][wp_from].append(wp_to)
                self.all_waypoints.add(wp_from)
                self.all_waypoints.add(wp_to)
            elif predicate == "visible":
                wp1, wp2 = parts[1], parts[2]
                self.visible_waypoints[wp1].add(wp2)
                self.all_waypoints.add(wp1)
                self.all_waypoints.add(wp2)
            elif predicate == "visible_from":
                objective, wp = parts[1], parts[2]
                self.objective_visible_from[objective].add(wp)
                # self.all_waypoints.add(objective) # Objectives are not waypoints, ignore
                self.all_waypoints.add(wp)
            elif predicate == "at_soil_sample":
                self.soil_sample_locations.add(parts[1])
                self.all_waypoints.add(parts[1])
            elif predicate == "at_rock_sample":
                self.rock_sample_locations.add(parts[1])
                self.all_waypoints.add(parts[1])
            # elif predicate == "at": # Initial rover locations are not static, but waypoints are
            #      self.all_waypoints.add(parts[2]) # This fact is not static

        # Ensure all waypoints collected are included in the graph nodes for BFS
        # Even if a waypoint has no 'can_traverse' facts, it might be a start/end point.
        # The BFS function handles nodes with no outgoing edges.
        all_waypoint_list = list(self.all_waypoints)

        # Compute all-pairs shortest paths for each rover
        self.rover_dist_maps = {}
        for rover in self.waypoint_graph: # Only build graph for rovers that have can_traverse facts
             self.rover_dist_maps[rover] = bfs_all_pairs_shortest_path(self.waypoint_graph[rover], all_waypoint_list)

        # Identify communication points (waypoints visible from the lander)
        self.communication_points = self.visible_waypoints.get(self.lander_location, set())


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # If goal is reached, heuristic is 0
        if self.goals <= state:
            return 0

        # Track current locations of rovers
        rover_locations = {}
        for fact in state:
            if match(fact, "at", "*", "*"):
                # Assuming the first argument after 'at' is the object (rover)
                # and the second is the location (waypoint)
                parts = get_parts(fact)
                if len(parts) == 3: # Ensure it's (at obj loc)
                     obj, loc = parts[1], parts[2]
                     # Need to check if obj is a rover type.
                     # A simple check is if it's a key in self.waypoint_graph (has can_traverse)
                     # or self.rover_equipment (has equipment).
                     # Assuming 'at' facts in state only apply to rovers for movement purposes.
                     if obj in self.waypoint_graph or obj in self.rover_equipment:
                         rover_locations[obj] = loc


        total_heuristic_cost = 0

        # Iterate through each goal fact
        for goal in self.goals:
            # If the goal is already achieved, skip it
            if goal in state:
                continue

            parts = get_parts(goal)
            predicate = parts[0]

            if predicate == "communicated_soil_data":
                waypoint_w = parts[1]
                goal_data_available = False
                # Check if any rover has the soil analysis for this waypoint
                for fact in state:
                    if match(fact, "have_soil_analysis", "*", waypoint_w):
                        goal_data_available = True
                        break

                min_rover_cost = float('inf')
                # Find suitable rovers (equipped for soil)
                suitable_rovers = [r for r, equip in self.rover_equipment.items() if "soil" in equip]

                for rover in suitable_rovers:
                    current_loc = rover_locations.get(rover)
                    if current_loc is None or rover not in self.rover_dist_maps: continue # Rover not located or no graph info

                    if goal_data_available:
                        # Cost: communicate (1) + move from current to comm point
                        cost = 1
                        dist_to_comm = min_dist_to_set(self.rover_dist_maps[rover], current_loc, self.communication_points)
                        if dist_to_comm != float('inf'):
                             cost += dist_to_comm
                             min_rover_cost = min(min_rover_cost, cost)
                    else:
                        # Need to sample and communicate
                        # Cost: sample (1) + communicate (1)
                        cost = 1 + 1

                        # Check if store is full, requiring a drop action
                        rover_store = self.rover_of_store.get(rover)
                        if rover_store and f"(full {rover_store})" in state:
                             cost += 1 # Add cost for drop action

                        dist_to_sample_wp = self.rover_dist_maps[rover].get(current_loc, {}).get(waypoint_w, float('inf'))
                        dist_sample_to_comm = min_dist_to_set(self.rover_dist_maps[rover], waypoint_w, self.communication_points)

                        if dist_to_sample_wp != float('inf') and dist_sample_to_comm != float('inf'):
                             cost += dist_to_sample_wp + dist_sample_to_comm
                             min_rover_cost = min(min_rover_cost, cost)

                # Add the minimum cost for this goal to the total heuristic
                total_heuristic_cost += min_rover_cost


            elif predicate == "communicated_rock_data":
                waypoint_w = parts[1]
                goal_data_available = False
                # Check if any rover has the rock analysis for this waypoint
                for fact in state:
                    if match(fact, "have_rock_analysis", "*", waypoint_w):
                        goal_data_available = True
                        break

                min_rover_cost = float('inf')
                # Find suitable rovers (equipped for rock)
                suitable_rovers = [r for r, equip in self.rover_equipment.items() if "rock" in equip]

                for rover in suitable_rovers:
                    current_loc = rover_locations.get(rover)
                    if current_loc is None or rover not in self.rover_dist_maps: continue

                    if goal_data_available:
                        # Cost: communicate (1) + move from current to comm point
                        cost = 1
                        dist_to_comm = min_dist_to_set(self.rover_dist_maps[rover], current_loc, self.communication_points)
                        if dist_to_comm != float('inf'):
                             cost += dist_to_comm
                             min_rover_cost = min(min_rover_cost, cost)
                    else:
                        # Need to sample and communicate
                        # Cost: sample (1) + communicate (1)
                        cost = 1 + 1

                        # Check if store is full, requiring a drop action
                        rover_store = self.rover_of_store.get(rover)
                        if rover_store and f"(full {rover_store})" in state:
                             cost += 1 # Add cost for drop action

                        dist_to_sample_wp = self.rover_dist_maps[rover].get(current_loc, {}).get(waypoint_w, float('inf'))
                        dist_sample_to_comm = min_dist_to_set(self.rover_dist_maps[rover], waypoint_w, self.communication_points)

                        if dist_to_sample_wp != float('inf') and dist_sample_to_comm != float('inf'):
                             cost += dist_to_sample_wp + dist_sample_to_comm
                             min_rover_cost = min(min_rover_cost, cost)

                # Add the minimum cost for this goal to the total heuristic
                total_heuristic_cost += min_rover_cost


            elif predicate == "communicated_image_data":
                objective_o = parts[1]
                mode_m = parts[2]
                goal_data_available = False
                # Check if any rover has the image for this objective and mode
                for fact in state:
                    if match(fact, "have_image", "*", objective_o, mode_m):
                        goal_data_available = True
                        break

                min_rover_camera_cost = float('inf')
                # Find suitable rovers (equipped for imaging)
                suitable_rovers = [r for r, equip in self.rover_equipment.items() if "imaging" in equip]

                # Find imaging waypoints for this objective
                imaging_points = self.objective_visible_from.get(objective_o, set())

                for rover in suitable_rovers:
                    current_loc = rover_locations.get(rover)
                    if current_loc is None or rover not in self.rover_dist_maps: continue

                    # Find cameras on this rover that support this mode
                    suitable_cameras = [cam for cam, r in self.camera_on_board.items()
                                        if r == rover and mode_m in self.camera_supports.get(cam, set())]

                    for camera in suitable_cameras:
                        if goal_data_available:
                            # Cost: communicate (1) + move from current to comm point
                            cost_rc = 1
                            dist_to_comm = min_dist_to_set(self.rover_dist_maps[rover], current_loc, self.communication_points)
                            if dist_to_comm != float('inf'):
                                 cost_rc += dist_to_comm
                                 min_rover_camera_cost = min(min_rover_camera_cost, cost_rc)
                        else:
                            # Need to take image and communicate
                            # Cost: communicate (1) + take_image (1)
                            cost_rc = 1 + 1

                            camera_calibrated = False
                            if f"(calibrated {camera} {rover})" in state:
                                camera_calibrated = True

                            if camera_calibrated:
                                # Cost: move to imaging point + move from imaging point to comm point
                                # Find min cost over all pairs of (imaging_point, comm_point)
                                min_nav_cost = float('inf')
                                for img_wp in imaging_points:
                                    dist_curr_to_img = self.rover_dist_maps[rover].get(current_loc, {}).get(img_wp, float('inf'))
                                    dist_img_to_comm = min_dist_to_set(self.rover_dist_maps[rover], img_wp, self.communication_points)
                                    if dist_curr_to_img != float('inf') and dist_img_to_comm != float('inf'):
                                         min_nav_cost = min(min_nav_cost, dist_curr_to_img + dist_img_to_comm)

                                if min_nav_cost != float('inf'):
                                     cost_rc += min_nav_cost
                                     min_rover_camera_cost = min(min_rover_camera_cost, cost_rc)

                            else:
                                # Need to calibrate first
                                # Cost: calibrate (1) + move to cal point + move from cal point to imaging point + move from imaging point to comm point
                                cost_rc += 1 # Add calibrate cost

                                # Find calibration target for this camera
                                cal_target = self.camera_calibration_target.get(camera)
                                if cal_target is None: continue # Cannot calibrate this camera

                                # Find calibration waypoints for this target
                                cal_points = self.objective_visible_from.get(cal_target, set())

                                # Find min cost over all triplets (cal_point, imaging_point, comm_point)
                                min_nav_cost = float('inf')
                                for cal_wp in cal_points:
                                    for img_wp in imaging_points:
                                        dist_curr_to_cal = self.rover_dist_maps[rover].get(current_loc, {}).get(cal_wp, float('inf'))
                                        dist_cal_to_img = self.rover_dist_maps[rover].get(cal_wp, {}).get(img_wp, float('inf'))
                                        dist_img_to_comm = min_dist_to_set(self.rover_dist_maps[rover], img_wp, self.communication_points)

                                        if dist_curr_to_cal != float('inf') and dist_cal_to_img != float('inf') and dist_img_to_comm != float('inf'):
                                             min_nav_cost = min(min_nav_cost, dist_curr_to_cal + dist_cal_to_img + dist_img_to_comm)

                                if min_nav_cost != float('inf'):
                                     cost_rc += min_nav_cost
                                     min_rover_camera_cost = min(min_rover_camera_cost, cost_rc)

                # Add the minimum cost for this goal to the total heuristic
                total_heuristic_cost += min_rover_camera_cost

        # Return infinity if any goal is unreachable, otherwise return the sum
        if total_heuristic_cost == float('inf'):
             return float('inf')
        return total_heuristic_cost

