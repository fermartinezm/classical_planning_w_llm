import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic # Assuming heuristic_base is available

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty string or non-string input gracefully
    if not isinstance(fact, str) or len(fact) < 2 or fact[0] != '(' or fact[-1] != ')':
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
    return all(fnmatch.fnmatch(part, arg) for part, arg in zip(parts, args))

def bfs(graph, start_node, all_nodes):
    """
    Performs Breadth-First Search on a graph to find shortest distances
    from a start node to all reachable nodes.

    Args:
        graph (dict): Adjacency list representation {node: set(neighbors)}.
        start_node (str): The starting node for the BFS.
        all_nodes (set): A set of all possible nodes in the graph.

    Returns:
        dict: A dictionary mapping each reachable node to its distance from start_node.
              Nodes not reachable have distance float('inf').
    """
    distances = {node: float('inf') for node in all_nodes}
    if start_node not in all_nodes:
        return distances # Start node is not a valid waypoint

    distances[start_node] = 0
    queue = deque([(start_node, 0)])

    while queue:
        current_node, current_dist = queue.popleft()

        # Ensure current_node is a key in the graph before accessing neighbors
        if current_node in graph:
            for neighbor in graph[current_node]:
                if distances[neighbor] == float('inf'): # Check if visited
                    distances[neighbor] = current_dist + 1
                    queue.append((neighbor, current_dist + 1))
    return distances


class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve all
    unmet communication goals. It calculates the minimum cost for each goal
    independently by finding the best-suited rover and summing the estimated
    costs for:
    1. Acquiring the necessary data (sample analysis or image).
    2. Moving to a location visible from the lander.
    3. Communicating the data.
    Movement costs are estimated using precomputed shortest paths on the
    traversal graph for each rover.

    # Assumptions
    - Each communication goal (soil, rock, mineral, image) is independent.
    - The cost of acquiring data (take sample/image, analyze/calibrate) is fixed
      per type (e.g., 2 actions for sample, 2 actions for image + calibration).
    - Store capacity is simplified: assume an empty store is available when
      needed for taking a sample.
    - Camera calibration is simplified: assume calibration is possible if the
      camera *can* be calibrated anywhere, and costs 1 action just before
      taking an image. Calibration state is not tracked precisely.
    - Movement cost is the shortest path distance in the traversal graph.
    - The lander's location is static.

    # Heuristic Initialization
    The constructor extracts static information from the task:
    - Lander location.
    - Rover capabilities (equipped for soil, rock, mineral, imaging).
    - Cameras on board each rover and modes they support.
    - Calibration targets for each camera.
    - Visibility between objectives and waypoints.
    - Locations of soil, rock, and mineral samples.
    - Rover traversal graphs (can_traverse).
    - All waypoints and rovers.
    - Waypoints visible from the lander.
    - Precomputes shortest path distances between all pairs of waypoints
      for each rover using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic is computed as follows:
    1. Identify all goals that have not yet been achieved in the current state.
    2. Initialize the total heuristic cost to 0.
    3. For each unachieved goal (e.g., `(communicated_type_data P)`):
        a. Initialize the minimum cost for this goal (`min_goal_cost`) to infinity.
        b. Iterate through each rover:
            i. Check if the rover is equipped to handle the goal type (soil, rock, mineral, image). If not, skip this rover for this goal.
            ii. Estimate the cost for this rover to achieve the goal:
                - **Cost to get data:**
                    - If the rover already `have_` the required data (e.g., `(have_soil_analysis ?r ?w)` or `(have_image ?r ?o ?m)`), the cost is 0, and the data location is the rover's current location.
                    - If the rover does *not* have the data:
                        - For sample data (`?w`): Check if the sample exists at `?w`. If yes, cost is movement from rover's current location to `?w` + 2 actions (take + analyze). Data location is `?w`. If sample doesn't exist, this path is impossible for this rover/waypoint.
                        - For image data (`?o ?m`): Find a camera on the rover supporting mode `?m`. Check if this camera can be calibrated at all (i.e., has a calibration target visible from *any* waypoint). If yes, find waypoints `?w_img` where `?o` is visible. For each `?w_img`, cost is movement from rover's current location to `?w_img` + 1 action (calibrate) + 1 action (take_image). Find the minimum cost over all suitable cameras and image waypoints. Data location is the best `?w_img`. If no suitable camera or image waypoint, this path is impossible.
                - **Cost to communicate data:**
                    - If the cost to get data is finite, find the minimum movement cost from the data location (or rover's location if data was already had) to any waypoint visible from the lander.
                    - Cost is minimum movement cost + 1 action (communicate). If no communication waypoint is reachable, this path is impossible.
            iii. The total cost for this rover for this goal is the sum of the cost to get data and the cost to communicate data.
            iv. Update `min_goal_cost` with the minimum cost found across all rovers.
        c. If `min_goal_cost` is still infinity after checking all rovers, the goal is unreachable; add a large penalty to the total heuristic.
        d. Otherwise, add `min_goal_cost` to the total heuristic.
    4. Return the total heuristic cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information and
        precomputing shortest paths.
        """
        self.goals = task.goals
        static_facts = task.static

        # Data structures to store static information
        self.lander_location = None
        self.capable_rovers = {'soil': set(), 'rock': set(), 'mineral': set(), 'imaging': set()}
        self.rover_cameras = {} # rover -> set(camera)
        self.camera_modes = {} # camera -> set(mode)
        self.calibratable_objectives = {} # camera -> set(objective)
        self.objective_visibility = {} # objective -> set(waypoint)
        self.sample_locations = {'soil': set(), 'rock': set(), 'mineral': set()}
        self.rover_traversal_graph = {} # rover -> {waypoint -> set(neighbor_waypoint)}
        self.waypoints = set()
        self.rovers = set()

        # Parse static facts
        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue # Skip invalid facts

            predicate = parts[0]

            if predicate == 'at_lander':
                # (at_lander ?l ?w)
                self.lander_location = parts[2]
                self.waypoints.add(parts[2])
            elif predicate.startswith('equipped_for_'):
                # (equipped_for_soil_analysis ?r) etc.
                capability = predicate.split('_')[2] # soil, rock, imaging
                rover = parts[1]
                self.capable_rovers[capability].add(rover)
                self.rovers.add(rover)
            elif predicate == 'on_board':
                # (on_board ?c ?r)
                camera, rover = parts[1], parts[2]
                self.rover_cameras.setdefault(rover, set()).add(camera)
                self.rovers.add(rover)
            elif predicate == 'supports':
                # (supports ?c ?m)
                camera, mode = parts[1], parts[2]
                self.camera_modes.setdefault(camera, set()).add(mode)
            elif predicate == 'calibration_target':
                # (calibration_target ?c ?o)
                camera, objective = parts[1], parts[2]
                self.calibratable_objectives.setdefault(camera, set()).add(objective)
            elif predicate == 'visible_from':
                # (visible_from ?o ?w)
                objective, waypoint = parts[1], parts[2]
                self.objective_visibility.setdefault(objective, set()).add(waypoint)
                self.waypoints.add(waypoint)
            elif predicate.startswith('at_') and predicate.endswith('_sample'):
                # (at_soil_sample ?w) etc.
                sample_type = predicate.split('_')[1] # soil, rock, mineral
                waypoint = parts[1]
                self.sample_locations[sample_type].add(waypoint)
                self.waypoints.add(waypoint)
            elif predicate == 'can_traverse':
                # (can_traverse ?r ?w1 ?w2)
                rover, w1, w2 = parts[1], parts[2], parts[3]
                self.rover_traversal_graph.setdefault(rover, {}).setdefault(w1, set()).add(w2)
                self.rovers.add(rover)
                self.waypoints.add(w1)
                self.waypoints.add(w2)
            elif predicate == 'at':
                 # (at ?r ?w) - Initial rover locations are static in init
                 # We don't need initial location here, it's in the state.
                 # But we can collect rovers and waypoints from here too.
                 obj_type = parts[1] # Could be rover or package etc.
                 location = parts[2]
                 if obj_type.startswith('rover'):
                     self.rovers.add(obj_type)
                 self.waypoints.add(location)
            elif predicate == 'visible':
                 # (visible ?w1 ?w2) - Used for communication locations
                 w1, w2 = parts[1], parts[2]
                 self.waypoints.add(w1)
                 self.waypoints.add(w2)
            # Add other static predicates if needed, though the above cover goal dependencies

        # Determine waypoints visible from the lander
        self.waypoints_visible_from_lander = set()
        if self.lander_location:
             for fact in static_facts:
                 if match(fact, 'visible', '*', self.lander_location):
                     self.waypoints_visible_from_lander.add(get_parts(fact)[1])
                 if match(fact, 'visible', self.lander_location, '*'):
                      # visible is symmetric, but let's be explicit
                      self.waypoints_visible_from_lander.add(get_parts(fact)[2])


        # Precompute shortest paths for each rover
        self.shortest_paths = {}
        for rover in self.rovers:
            graph = self.rover_traversal_graph.get(rover, {})
            self.shortest_paths[rover] = {}
            # Run BFS from every waypoint that is a key in this rover's graph
            # or is a waypoint where a rover could potentially start or need to go.
            # Running from all known waypoints is safest.
            for start_wp in self.waypoints:
                 self.shortest_paths[rover][start_wp] = bfs(graph, start_wp, self.waypoints)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Data structures to store current state information
        rover_location = {} # rover -> waypoint
        rover_has_data = {} # rover -> set(data_fact_string) e.g. '(have_soil_analysis rover1 waypoint4)'
        camera_calibrated_state = {} # camera -> bool
        achieved_goals = set() # set(communicated_fact_string)

        # Parse current state facts
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]

            if predicate == 'at':
                # (at ?r ?w)
                obj, waypoint = parts[1], parts[2]
                if obj in self.rovers: # Only track rover locations
                    rover_location[obj] = waypoint
            elif predicate.startswith('have_') and (predicate.endswith('_analysis') or predicate == 'have_image'):
                 # (have_soil_analysis ?r ?w), (have_image ?r ?o ?m) etc.
                 rover = parts[1]
                 rover_has_data.setdefault(rover, set()).add(fact)
            elif predicate == 'calibrated':
                 # (calibrated ?c ?r)
                 camera = parts[1]
                 camera_calibrated_state[camera] = True
            elif predicate.startswith('communicated_'):
                 # (communicated_soil_data ?w), (communicated_image_data ?o ?m) etc.
                 achieved_goals.add(fact)

        total_heuristic = 0
        UNREACHABLE_PENALTY = 1000 # Penalty for goals deemed unreachable

        # Iterate through each goal
        for goal in self.goals:
            if goal in achieved_goals:
                continue # Goal already achieved

            parts = get_parts(goal)
            if not parts: continue

            predicate = parts[0]
            goal_type = None
            goal_param = None # ?w or (?o, ?m)

            if predicate == 'communicated_soil_data':
                goal_type = 'soil'
                goal_param = parts[1] # waypoint
            elif predicate == 'communicated_rock_data':
                goal_type = 'rock'
                goal_param = parts[1] # waypoint
            elif predicate == 'communicated_mineral_data':
                goal_type = 'mineral'
                goal_param = parts[1] # waypoint
            elif predicate == 'communicated_image_data':
                goal_type = 'image'
                goal_param = (parts[1], parts[2]) # (objective, mode)
            else:
                # Unknown goal type, ignore or penalize? Ignore for now.
                continue

            min_goal_cost = float('inf')

            # Iterate through each rover to find the minimum cost path for this goal
            for rover in self.rovers:
                # Check if this rover is capable of handling the goal type
                if rover not in self.capable_rovers.get(goal_type, set()):
                    continue # Rover cannot perform this task type

                cost_to_get_data_part = float('inf')
                data_location_for_this_rover = None # Waypoint where data is acquired

                # --- Part 1: Cost to get the data (have_...) ---
                data_fact_needed = None
                if goal_type in ['soil', 'rock', 'mineral']:
                    waypoint = goal_param
                    data_fact_needed = f'(have_{goal_type}_analysis {rover} {waypoint})'
                    sample_type = goal_type
                    sample_waypoint = waypoint

                    # Check if sample exists at the waypoint (static)
                    if sample_waypoint in self.sample_locations.get(sample_type, set()):
                        # Sample exists, calculate cost to get it
                        current_rover_wp = rover_location.get(rover)
                        if current_rover_wp and current_rover_wp in self.shortest_paths.get(rover, {}):
                            movement_cost = self.shortest_paths[rover][current_rover_wp].get(sample_waypoint, float('inf'))
                            if movement_cost != float('inf'):
                                # Cost: move to sample + take + analyze
                                cost_to_get_data_part = movement_cost + 1 + 1
                                data_location_for_this_rover = sample_waypoint
                        # else: cost_to_get_data_part remains infinity (rover location unknown or no path)
                    # else: cost_to_get_data_part remains infinity (sample not at location)

                elif goal_type == 'image':
                    objective, mode = goal_param
                    data_fact_needed = f'(have_image {rover} {objective} {mode})'
                    image_objective = objective
                    image_mode = mode

                    min_image_part_cost = float('inf')
                    best_image_location = None

                    # Find a camera on this rover that supports the mode
                    suitable_cameras = [c for c in self.rover_cameras.get(rover, set()) if image_mode in self.camera_modes.get(c, set())]

                    for camera in suitable_cameras:
                         # Check if this camera can be calibrated at all (i.e., has a target visible from *any* waypoint)
                         can_calibrate_camera = False
                         for cal_target in self.calibratable_objectives.get(camera, set()):
                             if self.objective_visibility.get(cal_target, set()): # Check if target is visible from *any* waypoint
                                 can_calibrate_camera = True
                                 break
                         if not can_calibrate_camera:
                             continue # This camera cannot be calibrated

                         # Find waypoints where the objective is visible (static)
                         image_waypoints = self.objective_visibility.get(image_objective, set())

                         for img_wp in image_waypoints:
                             current_rover_wp = rover_location.get(rover)
                             if current_rover_wp and current_rover_wp in self.shortest_paths.get(rover, {}):
                                 movement_cost = self.shortest_paths[rover][current_rover_wp].get(img_wp, float('inf'))
                                 if movement_cost != float('inf'):
                                     # Simplified image cost: move to image location + calibrate + take_image
                                     # Assumes calibration is needed and possible at the image location or nearby
                                     current_image_part_cost = movement_cost + 1 + 1
                                     if current_image_part_cost < min_image_part_cost:
                                         min_image_part_cost = current_image_part_cost
                                         best_image_location = img_wp
                             # else: skip this img_wp (rover location unknown or no path)

                    cost_to_get_data_part = min_image_part_cost
                    data_location_for_this_rover = best_image_location

                # Check if data is already present on this specific rover
                if data_fact_needed in rover_has_data.get(rover, set()):
                     cost_to_get_data_part = 0
                     data_location_for_this_rover = rover_location.get(rover) # Data is at rover's current location

                # If data acquisition is impossible for this rover, skip communication part
                if cost_to_get_data_part == float('inf') or data_location_for_this_rover is None:
                    continue # Cannot get data with this rover

                # --- Part 2: Cost to communicate the data ---
                cost_to_communicate_part = float('inf')
                min_comm_movement_cost = float('inf')

                # Need to move from data_location_for_this_rover to a communication waypoint
                start_comm_location = data_location_for_this_rover

                if start_comm_location in self.shortest_paths.get(rover, {}):
                    for comm_wp in self.waypoints_visible_from_lander:
                        movement_cost = self.shortest_paths[rover][start_comm_location].get(comm_wp, float('inf'))
                        min_comm_movement_cost = min(min_comm_movement_cost, movement_cost)

                if min_comm_movement_cost != float('inf'):
                    # Cost: move to comm waypoint + communicate
                    cost_to_communicate_part = min_comm_movement_cost + 1
                # else: cost_to_communicate_part remains infinity (no path to any comm waypoint)


                # --- Total cost for this rover for this goal ---
                if cost_to_communicate_part != float('inf'):
                     cost_for_this_rover = cost_to_get_data_part + cost_to_communicate_part
                     min_goal_cost = min(min_goal_cost, cost_for_this_rover)

            # After checking all rovers for goal G:
            if min_goal_cost == float('inf'):
                # Goal is unreachable by any capable rover
                total_heuristic += UNREACHABLE_PENALTY
            else:
                total_heuristic += min_goal_cost

        return total_heuristic

