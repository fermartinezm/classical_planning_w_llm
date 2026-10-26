from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure the fact is a string and has parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        # Handle unexpected input, though PDDL facts should be strings
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

def build_graph(static_facts):
    """Build a waypoint graph from can_traverse facts."""
    graph = {}
    waypoints = set()

    # Collect all waypoints mentioned in can_traverse facts
    for fact in static_facts:
        if match(fact, "can_traverse", "*", "*", "*"):
            _, rover, w1, w2 = get_parts(fact)
            waypoints.add(w1)
            waypoints.add(w2)

    # Add all collected waypoints to the graph dictionary
    for wp in waypoints:
        if wp not in graph:
            graph[wp] = set()

    # Add edges based on can_traverse facts (assuming symmetry)
    for fact in static_facts:
        if match(fact, "can_traverse", "*", "*", "*"):
            _, rover, w1, w2 = get_parts(fact)
            # Ensure waypoints exist in the graph before adding edges
            if w1 in graph and w2 in graph:
                 graph[w1].add(w2)
                 graph[w2].add(w1) # Assuming can_traverse is symmetric

    # Also include waypoints mentioned in other static facts (like at_lander, at_soil_sample, visible_from)
    # even if they have no traverse edges, to ensure they are in the graph keys for BFS.
    for fact in static_facts:
         parts = get_parts(fact)
         for part in parts:
             # Simple check: if it looks like a waypoint name
             if part.startswith('waypoint'):
                 if part not in graph:
                     graph[part] = set()
                 waypoints.add(part)


    return graph, list(waypoints) # Return graph and list of all waypoints

def bfs(start_node, graph):
    """Compute shortest distances from start_node to all reachable nodes in graph."""
    distances = {node: float('inf') for node in graph}
    if start_node in graph: # Ensure start_node is in the graph
        distances[start_node] = 0
        queue = deque([start_node])

        while queue:
            current_node = queue.popleft()
            current_dist = distances[current_node]

            if current_node in graph: # Should always be true here, but defensive
                for neighbor in graph[current_node]:
                    if distances[neighbor] == float('inf'):
                        distances[neighbor] = current_dist + 1
                        queue.append(neighbor)
    return distances


class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to satisfy all
    unmet goal conditions. It considers the steps needed to acquire data
    (move to sample/image location, take sample/image, calibrate camera if needed)
    and communicate data (move to communication point, communicate). It sums
    the minimum costs for each unmet goal independently.

    # Assumptions
    - Rovers can traverse the graph defined by `can_traverse` facts.
    - `can_traverse` is symmetric.
    - Lander location is static.
    - Camera calibration is specific to a rover and persists.
    - Dropping a sample is possible anywhere and costs 1 action if the store is not empty when a new sample is needed.
    - Any equipped and available rover/camera can be used for a task.
    - The cost of moving between waypoints is 1 per step. Other actions (take sample, take image, calibrate, communicate, drop sample) cost 1.
    - If a goal requires data from a sample/objective at a waypoint, that sample/objective is indeed located there (static facts guarantee this).
    - If a goal requires an image of an objective, that objective is visible from at least one waypoint (static facts guarantee this).
    - If a camera needs calibration, it has a calibration target visible from at least one waypoint (static facts guarantee this).
    - The lander is at a waypoint visible from at least one other waypoint (static facts guarantee this for communication).

    # Heuristic Initialization
    - Parses static facts to build the waypoint graph and precompute all-pairs
      shortest path distances.
    - Extracts lander location, rover capabilities, camera information,
      objective visibility, sample locations, and store ownership.
    - Identifies the set of waypoints suitable for communication (visible from lander).
    - Extracts the specific soil, rock, and image data goals.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize total heuristic cost to 0.
    2. Identify all goal conditions that are *not* met in the current state.
    3. For each unmet goal condition:
       - Calculate the minimum estimated cost to achieve *only* this goal, considering all available rovers and cameras.
       - If the goal is `(communicated_soil_data ?w)`:
         - Find rovers equipped for soil analysis.
         - For each equipped rover:
           - Calculate cost:
             - Get rover's current location.
             - If `(have_soil_analysis ?r ?w)` is false:
               - Add distance from rover's current location to `?w`.
               - Add 1 for `take_soil_sample`.
               - If rover's store is not empty, add 1 for `drop_sample`.
               - Update rover's conceptual location to `?w`.
             - Add minimum distance from rover's current conceptual location to any communication waypoint.
             - Add 1 for `communicate_soil_data`.
           - Take the minimum cost over all equipped rovers. Add this minimum to the total heuristic. If no equipped rover or no path exists, add a large value (infinity).
       - If the goal is `(communicated_rock_data ?w)`:
         - Similar logic to soil data, using rock-specific predicates and equipment.
       - If the goal is `(communicated_image_data ?o ?m)`:
         - Find rovers with cameras supporting mode `?m`.
         - For each such rover/camera pair:
           - Calculate cost:
             - Get rover's current location.
             - If `(calibrated ?c ?r)` is false:
               - Find calibration target `?o_cal` for `?c`.
               - Add minimum distance from rover's current location to any waypoint visible from `?o_cal`.
               - Add 1 for `calibrate`.
               - Update rover's conceptual location to the chosen calibration waypoint.
             - If `(have_image ?r ?o ?m)` is false:
               - Add minimum distance from rover's current conceptual location to any waypoint visible from `?o`.
               - Add 1 for `take_image`.
               - Update rover's conceptual location to the chosen image waypoint.
             - Add minimum distance from rover's current conceptual location to any communication waypoint.
             - Add 1 for `communicate_image_data`.
           - Take the minimum cost over all suitable rover/camera pairs. Add this minimum to the total heuristic. If no suitable pair or no path exists, add a large value (infinity).
    4. Return the total heuristic cost. If the state is a goal state, this sum will be 0. If any necessary step is impossible (e.g., unreachable waypoint), return infinity.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # --- Extract Static Information ---
        self.lander_location = None
        self.rover_capabilities = {} # rover -> set(capabilities)
        self.camera_info = {} # camera -> {'on_board': rover, 'supports': set(modes), 'calibration_target': objective}
        self.objective_visibility = {} # objective -> set(waypoints)
        self.sample_locations = {'soil': set(), 'rock': set()} # type -> set(waypoints)
        self.rover_to_store = {} # rover -> store
        self.communication_waypoints = set() # waypoints visible from lander

        # Build waypoint graph and precompute distances
        self.waypoint_graph, all_waypoints = build_graph(static_facts)
        self.distances = {}
        for wp in all_waypoints:
             self.distances[wp] = bfs(wp, self.waypoint_graph)

        # Parse static facts
        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            if parts[0] == 'at_lander' and len(parts) == 3:
                self.lander_location = parts[2]
            elif parts[0] == 'equipped_for_soil_analysis' and len(parts) == 2:
                rover = parts[1]
                if rover not in self.rover_capabilities: self.rover_capabilities[rover] = set()
                self.rover_capabilities[rover].add('soil')
            elif parts[0] == 'equipped_for_rock_analysis' and len(parts) == 2:
                rover = parts[1]
                if rover not in self.rover_capabilities: self.rover_capabilities[rover] = set()
                self.rover_capabilities[rover].add('rock')
            elif parts[0] == 'equipped_for_imaging' and len(parts) == 2:
                rover = parts[1]
                if rover not in self.rover_capabilities: self.rover_capabilities[rover] = set()
                self.rover_capabilities[rover].add('imaging')
            elif parts[0] == 'store_of' and len(parts) == 3:
                store, rover = parts[1], parts[2]
                self.rover_to_store[rover] = store
            elif parts[0] == 'at_soil_sample' and len(parts) == 2:
                self.sample_locations['soil'].add(parts[1])
            elif parts[0] == 'at_rock_sample' and len(parts) == 2:
                self.sample_locations['rock'].add(parts[1])
            elif parts[0] == 'visible_from' and len(parts) == 3:
                objective, waypoint = parts[1], parts[2]
                if objective not in self.objective_visibility: self.objective_visibility[objective] = set()
                self.objective_visibility[objective].add(waypoint)
            elif parts[0] == 'on_board' and len(parts) == 3:
                camera, rover = parts[1], parts[2]
                if camera not in self.camera_info: self.camera_info[camera] = {}
                self.camera_info[camera]['on_board'] = rover
            elif parts[0] == 'supports' and len(parts) == 3:
                camera, mode = parts[1], parts[2]
                if camera not in self.camera_info: self.camera_info[camera] = {}
                if 'supports' not in self.camera_info[camera]: self.camera_info[camera]['supports'] = set()
                self.camera_info[camera]['supports'].add(mode)
            elif parts[0] == 'calibration_target' and len(parts) == 3:
                camera, objective = parts[1], parts[2]
                if camera not in self.camera_info: self.camera_info[camera] = {}
                self.camera_info[camera]['calibration_target'] = objective

        # Identify communication waypoints (visible from lander)
        if self.lander_location:
             for fact in static_facts:
                 if match(fact, "visible", "*", self.lander_location):
                     wp = get_parts(fact)[1]
                     if wp in self.waypoint_graph: # Ensure waypoint is in our graph
                         self.communication_waypoints.add(wp)
                 if match(fact, "visible", self.lander_location, "*"):
                     wp = get_parts(fact)[2]
                     if wp in self.waypoint_graph: # Ensure waypoint is in our graph
                         self.communication_waypoints.add(wp)


        # Extract specific goals
        self.goal_soil_samples = set() # waypoint
        self.goal_rock_samples = set() # waypoint
        self.goal_images = set() # (objective, mode)

        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue # Skip malformed goals

            if parts[0] == 'communicated_soil_data' and len(parts) == 2:
                self.goal_soil_samples.add(parts[1])
            elif parts[0] == 'communicated_rock_data' and len(parts) == 2:
                self.goal_rock_samples.add(parts[1])
            elif parts[0] == 'communicated_image_data' and len(parts) == 3:
                self.goal_images.add((parts[1], parts[2]))

    def get_distance(self, wp1, wp2):
        """Lookup precomputed distance between two waypoints."""
        if wp1 not in self.distances or wp2 not in self.distances[wp1]:
             return float('inf') # Unreachable or waypoint not in graph
        return self.distances[wp1][wp2]

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        total_cost = 0

        # Get current rover locations and store status
        rover_locations = {}
        rover_stores_empty = {}
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            if parts[0] == 'at' and len(parts) == 3 and parts[1].startswith('rover'):
                rover_locations[parts[1]] = parts[2]
            elif parts[0] == 'empty' and len(parts) == 2 and parts[1].startswith('rover') and parts[1].endswith('store'):
                 # Find the rover for this store
                 for r, s in self.rover_to_store.items():
                     if s == parts[1]:
                         rover_stores_empty[r] = True
                         break
        # Assume stores are not empty if not explicitly stated as empty in the current state
        for rover in self.rover_to_store:
             if rover not in rover_stores_empty:
                 rover_stores_empty[rover] = False


        # --- Calculate cost for unmet soil goals ---
        for soil_wp in self.goal_soil_samples:
            goal_fact = f'(communicated_soil_data {soil_wp})'
            if goal_fact not in state:
                min_soil_cost = float('inf')

                # Find equipped rovers
                equipped_rovers = [r for r, caps in self.rover_capabilities.items() if 'soil' in caps]

                for rover in equipped_rovers:
                    cost = 0
                    current_rover_location = rover_locations.get(rover) # Get rover's current location

                    if current_rover_location is None: # Rover not found in state (shouldn't happen in valid states)
                         continue

                    # Step 1: Get sample
                    have_sample_fact = f'(have_soil_analysis {rover} {soil_wp})'
                    if have_sample_fact not in state:
                        dist_to_sample = self.get_distance(current_rover_location, soil_wp)
                        if dist_to_sample == float('inf'): continue # Cannot reach sample location

                        cost += dist_to_sample # Move to sample
                        cost += 1 # take_soil_sample

                        # Check if store needs dropping
                        if not rover_stores_empty.get(rover, False): # Store is not empty
                             cost += 1 # drop_sample action
                             # Note: This doesn't model which sample is dropped, just that the store needs freeing.

                        current_rover_location = soil_wp # Rover is now conceptually at sample location

                    # Step 2: Communicate
                    min_comm_dist = float('inf')
                    for comm_wp in self.communication_waypoints:
                        min_comm_dist = min(min_comm_dist, self.get_distance(current_rover_location, comm_wp))

                    if min_comm_dist == float('inf'): continue # Cannot reach communication point

                    cost += min_comm_dist # Move to communication point
                    cost += 1 # communicate_soil_data

                    min_soil_cost = min(min_soil_cost, cost)

                if min_soil_cost != float('inf'):
                    total_cost += min_soil_cost
                else:
                    # If a goal is unreachable by any means, return a large value
                    return float('inf') # Or a large constant

        # --- Calculate cost for unmet rock goals ---
        for rock_wp in self.goal_rock_samples:
            goal_fact = f'(communicated_rock_data {rock_wp})'
            if goal_fact not in state:
                min_rock_cost = float('inf')

                # Find equipped rovers
                equipped_rovers = [r for r, caps in self.rover_capabilities.items() if 'rock' in caps]

                for rover in equipped_rovers:
                    cost = 0
                    current_rover_location = rover_locations.get(rover)

                    if current_rover_location is None: continue

                    # Step 1: Get sample
                    have_sample_fact = f'(have_rock_analysis {rover} {rock_wp})'
                    if have_sample_fact not in state:
                        dist_to_sample = self.get_distance(current_rover_location, rock_wp)
                        if dist_to_sample == float('inf'): continue # Cannot reach sample location

                        cost += dist_to_sample # Move to sample
                        cost += 1 # take_rock_sample

                        # Check if store needs dropping
                        if not rover_stores_empty.get(rover, False): # Store is not empty
                             cost += 1 # drop_sample action

                        current_rover_location = rock_wp # Rover is now conceptually at sample location

                    # Step 2: Communicate
                    min_comm_dist = float('inf')
                    for comm_wp in self.communication_waypoints:
                        min_comm_dist = min(min_comm_dist, self.get_distance(current_rover_location, comm_wp))

                    if min_comm_dist == float('inf'): continue # Cannot reach communication point

                    cost += min_comm_dist # Move to communication point
                    cost += 1 # communicate_rock_data

                    min_rock_cost = min(min_rock_cost, cost)

                if min_rock_cost != float('inf'):
                    total_cost += min_rock_cost
                else:
                    return float('inf') # Unreachable goal

        # --- Calculate cost for unmet image goals ---
        for obj_img, mode_img in self.goal_images:
            goal_fact = f'(communicated_image_data {obj_img} {mode_img})'
            if goal_fact not in state:
                min_image_cost = float('inf')

                # Find rovers/cameras supporting this mode
                suitable_rover_cameras = []
                for camera, info in self.camera_info.items():
                    if 'on_board' in info and 'supports' in info and mode_img in info['supports']:
                        suitable_rover_cameras.append((info['on_board'], camera))

                for rover, camera in suitable_rover_cameras:
                    cost = 0
                    current_rover_location = rover_locations.get(rover)

                    if current_rover_location is None: continue

                    # Step 1: Calibrate camera
                    calibrated_fact = f'(calibrated {camera} {rover})'
                    if calibrated_fact not in state:
                        if camera not in self.camera_info or 'calibration_target' not in self.camera_info[camera]:
                             continue # Camera has no calibration target defined statically

                        cal_obj = self.camera_info[camera]['calibration_target']
                        if cal_obj not in self.objective_visibility or not self.objective_visibility[cal_obj]:
                             continue # Calibration target not visible from any waypoint

                        min_cal_dist = float('inf')
                        best_cal_wp = None
                        for cal_wp in self.objective_visibility[cal_obj]:
                            dist = self.get_distance(current_rover_location, cal_wp)
                            if dist < min_cal_dist:
                                min_cal_dist = dist
                                best_cal_wp = cal_wp

                        if min_cal_dist == float('inf'): continue # Cannot reach calibration point

                        cost += min_cal_dist # Move to calibration point
                        cost += 1 # calibrate
                        current_rover_location = best_cal_wp # Rover is now conceptually at calibration point

                    # Step 2: Take image
                    have_image_fact = f'(have_image {rover} {obj_img} {mode_img})'
                    if have_image_fact not in state:
                        if obj_img not in self.objective_visibility or not self.objective_visibility[obj_img]:
                             continue # Objective not visible from any waypoint

                        min_img_dist = float('inf')
                        best_img_wp = None
                        for img_wp in self.objective_visibility[obj_img]:
                            dist = self.get_distance(current_rover_location, img_wp)
                            if dist < min_img_dist:
                                min_img_dist = dist
                                best_img_wp = img_wp

                        if min_img_dist == float('inf'): continue # Cannot reach image point

                        cost += min_img_dist # Move to image point
                        cost += 1 # take_image
                        current_rover_location = best_img_wp # Rover is now conceptually at image point

                    # Step 3: Communicate
                    min_comm_dist = float('inf')
                    for comm_wp in self.communication_waypoints:
                        min_comm_dist = min(min_comm_dist, self.get_distance(current_rover_location, comm_wp))

                    if min_comm_dist == float('inf'): continue # Cannot reach communication point

                    cost += min_comm_dist # Move to communication point
                    cost += 1 # communicate_image_data

                    min_image_cost = min(min_image_cost, cost)

                if min_image_cost != float('inf'):
                    total_cost += min_image_cost
                else:
                    return float('inf') # Unreachable goal


        # If total_cost is 0, it means no unmet goals were found, which is a goal state.
        # If any goal was unreachable, we returned infinity earlier.
        return total_cost
