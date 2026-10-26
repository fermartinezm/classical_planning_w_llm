from fnmatch import fnmatch
from collections import deque
import math
from heuristics.heuristic_base import Heuristic

# Helper functions for parsing PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at rover1 waypoint2)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# Helper functions for graph traversal (BFS for shortest paths)
def bfs_shortest_paths(graph, start_node):
    """Computes shortest path distances from start_node to all other nodes in the graph."""
    distances = {node: math.inf for node in graph}
    if start_node in distances: # Ensure start_node is in the graph nodes
        distances[start_node] = 0
        queue = deque([start_node])

        while queue:
            current_node = queue.popleft()

            if current_node in graph:
                for neighbor in graph[current_node]:
                    if distances[neighbor] == math.inf:
                        distances[neighbor] = distances[current_node] + 1
                        queue.append(neighbor)
    return distances

def compute_all_pairs_shortest_paths(graph):
    """Computes shortest path distances between all pairs of nodes in the graph."""
    all_pairs_distances = {}
    all_nodes = set(graph.keys())
    for neighbors in graph.values():
        all_nodes.update(neighbors)

    # Ensure all nodes are keys in the graph dictionary, even if they have no neighbors
    full_graph = {node: set() for node in all_nodes}
    for u, neighbors in graph.items():
        full_graph[u].update(neighbors)

    for start_node in full_graph:
        all_pairs_distances[start_node] = bfs_shortest_paths(full_graph, start_node)
    return all_pairs_distances


class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to satisfy all goal conditions.
    It breaks down each unachieved goal (communicating soil, rock, or image data)
    into the necessary steps: acquiring the data (sampling or imaging) and then
    communicating it. It estimates the cost of each step, including navigation,
    sampling/imaging actions, calibration (for images), and dropping samples
    if the store is full. Navigation cost is estimated using precomputed shortest
    paths between waypoints.

    # Assumptions
    - The waypoint graph defined by `visible` facts is connected for all relevant waypoints in solvable problems.
    - Required objects (soil/rock samples, objectives, calibration targets) exist at locations reachable by equipped rovers in solvable problems.
    - Each rover has at most one store.
    - Each camera is on board at most one rover (statically).
    - The lander location is static.
    - The heuristic considers the first suitable equipped rover/camera found for a task.
    - Resource conflicts (e.g., multiple rovers needing the same waypoint or store) are ignored.
    - Action costs are uniform (1 per action).

    # Heuristic Initialization
    The heuristic precomputes static information from the task:
    - The lander's location.
    - Which rovers are equipped for soil, rock, and imaging tasks.
    - The mapping from rovers to their stores.
    - The mapping from cameras to supported modes.
    - The mapping from objectives to waypoints visible from them.
    - The mapping from cameras to their calibration targets.
    - The mapping from cameras to the rovers they are on board.
    - The waypoint graph based on `visible` connections.
    - All-pairs shortest path distances between waypoints using BFS on the `visible` graph.
    - The set of waypoints visible from the lander (communication points).

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic computes the sum of estimated costs for each goal fact that is not yet satisfied:

    1.  Initialize total cost to 0.
    2.  Extract dynamic state information: rover locations, sample locations, analyses/images held by rovers, store statuses, camera calibration statuses, and communication statuses.
    3.  Iterate through each goal fact specified in the task.
    4.  If a goal fact is already true in the current state, its cost is 0. Continue to the next goal.
    5.  If the goal is `(communicated_soil_data ?w)`:
        - Check if `(have_soil_analysis ?r ?w)` is true for any equipped rover `?r`.
            - If yes: Find the rover `r_with_analysis`. Estimate cost as navigation from `r_with_analysis`'s location to a communication waypoint + 1 (communicate).
            - If no (sampling is needed): Find an equipped rover `?r_eq`. Estimate cost as navigation from `?r_eq`'s location to `?w` + (1 if store full) + 1 (sample) + navigation from `?w` to a communication waypoint + 1 (communicate).
        - Add this estimated cost to the total cost. If any step is impossible (e.g., unreachable waypoint), the cost for this goal becomes infinity.
    6.  If the goal is `(communicated_rock_data ?w)`:
        - Similar logic as for soil data.
        - Add this estimated cost to the total cost. If any step is impossible, the cost for this goal becomes infinity.
    7.  If the goal is `(communicated_image_data ?o ?m)`:
        - Check if `(have_image ?r ?o ?m)` is true for any equipped rover `?r`.
            - If yes: Find the rover `r_with_image`. Estimate cost as navigation from `r_with_image`'s location to a communication waypoint + 1 (communicate).
            - If no (imaging is needed): Find an equipped rover `?r_eq` with a suitable camera `?i`. Estimate cost as navigation from `?r_eq`'s location to a calibration waypoint (if needed) + (1 if calibrate needed) + navigation to an image waypoint + 1 (take_image) + navigation to a communication waypoint + 1 (communicate).
        - Add this estimated cost to the total cost. If any step is impossible, the cost for this goal becomes infinity.
    8.  Return the total accumulated cost. If the total cost is infinity, the goal is unreachable from this state according to the heuristic.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # Precompute static information
        self.lander_waypoint = None
        self.equipped_rovers = {'soil': set(), 'rock': set(), 'imaging': set()}
        self.rover_stores = {} # {rover: store}
        self.camera_modes = {} # {camera: {mode1, mode2}}
        self.objective_waypoints = {} # {objective: {wp1, wp2}} (waypoints visible from objective)
        self.calibration_targets = {} # {camera: objective}
        self.rover_cameras = {} # {rover: {cam1, cam2}} (cameras on board rover)
        self.camera_rovers = {} # {camera: rover} # Map camera to the rover it's on (static)
        self.all_waypoints = set()
        self.waypoint_graph = {} # {wp: {neighbor1, neighbor2}} based on visible

        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue # Skip empty facts

            predicate = parts[0]
            if predicate == 'at_lander':
                self.lander_waypoint = parts[2]
            elif predicate == 'equipped_for_soil_analysis':
                self.equipped_rovers['soil'].add(parts[1])
            elif predicate == 'equipped_for_rock_analysis':
                self.equipped_rovers['rock'].add(parts[1])
            elif predicate == 'equipped_for_imaging':
                self.equipped_rovers['imaging'].add(parts[1])
            elif predicate == 'store_of':
                self.rover_stores[parts[2]] = parts[1] # map rover to store
            elif predicate == 'supports':
                camera, mode = parts[1], parts[2]
                self.camera_modes.setdefault(camera, set()).add(mode)
            elif predicate == 'visible_from':
                objective, waypoint = parts[1], parts[2]
                self.objective_waypoints.setdefault(objective, set()).add(waypoint)
                self.all_waypoints.add(waypoint)
            elif predicate == 'calibration_target':
                self.calibration_targets[parts[1]] = parts[2] # map camera to target objective
            elif predicate == 'on_board':
                camera, rover = parts[1], parts[2]
                self.rover_cameras.setdefault(rover, set()).add(camera)
                self.camera_rovers[camera] = rover # Assuming a camera is on only one rover statically
            elif predicate == 'visible':
                wp1, wp2 = parts[1], parts[2]
                self.waypoint_graph.setdefault(wp1, set()).add(wp2)
                self.waypoint_graph.setdefault(wp2, set()).add(wp1) # Graph is symmetric
                self.all_waypoints.add(wp1)
                self.all_waypoints.add(wp2)
            # Add waypoints from can_traverse too, just to be safe
            elif predicate == 'can_traverse':
                 wp1, wp2 = parts[2], parts[3]
                 self.all_waypoints.add(wp1)
                 self.all_waypoints.add(wp2)

        # Ensure all waypoints mentioned anywhere are in the graph keys for BFS
        for wp in self.all_waypoints:
            self.waypoint_graph.setdefault(wp, set())

        # Compute all-pairs shortest paths
        self.waypoint_distances = compute_all_pairs_shortest_paths(self.waypoint_graph)

        # Compute communication waypoints (visible from lander)
        self.comm_waypoints = set()
        if self.lander_waypoint and self.lander_waypoint in self.waypoint_graph:
             self.comm_waypoints = self.waypoint_graph[self.lander_waypoint]


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Extract dynamic state information
        rover_locations = {} # {rover: waypoint}
        soil_samples_at = set() # {waypoint}
        rock_samples_at = set() # {waypoint}
        have_soil = {} # {rover: {waypoint}}
        have_rock = {} # {rover: {waypoint}}
        have_image = {} # {rover: {(objective, mode)}}
        store_status = {} # {store: 'empty' or 'full'}
        calibrated_cameras = {} # {rover: {camera}}
        communicated_soil = set() # {waypoint}
        communicated_rock = set() # {waypoint}
        communicated_image = set() # {(objective, mode)}

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip empty facts

            predicate = parts[0]
            if predicate == 'at':
                rover_locations[parts[1]] = parts[2]
            elif predicate == 'at_soil_sample':
                soil_samples_at.add(parts[1])
            elif predicate == 'at_rock_sample':
                rock_samples_at.add(parts[1])
            elif predicate == 'have_soil_analysis':
                rover, wp = parts[1], parts[2]
                have_soil.setdefault(rover, set()).add(wp)
            elif predicate == 'have_rock_analysis':
                rover, wp = parts[1], parts[2]
                have_rock.setdefault(rover, set()).add(wp)
            elif predicate == 'have_image':
                rover, obj, mode = parts[1], parts[2], parts[3]
                have_image.setdefault(rover, set()).add((obj, mode))
            elif predicate == 'empty':
                store_status[parts[1]] = 'empty'
            elif predicate == 'full':
                store_status[parts[1]] = 'full'
            elif predicate == 'calibrated':
                camera, rover = parts[1], parts[2]
                calibrated_cameras.setdefault(rover, set()).add(camera)
            elif predicate == 'communicated_soil_data':
                communicated_soil.add(parts[1])
            elif predicate == 'communicated_rock_data':
                communicated_rock.add(parts[1])
            elif predicate == 'communicated_image_data':
                communicated_image.add((parts[1], parts[2]))

        total_cost = 0

        # Process goals
        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue # Skip empty goals

            predicate = parts[0]

            if predicate == 'communicated_soil_data':
                wp = parts[1]
                if wp in communicated_soil:
                    continue # Goal already met

                # Cost for this specific soil goal
                step_cost = 0

                # Check if analysis exists
                analysis_exists = any(wp in have_soil.get(r, set()) for r in self.equipped_rovers['soil'])

                if analysis_exists:
                    # Find a rover that has the analysis
                    r_with_analysis = next((r for r in self.equipped_rovers['soil'] if wp in have_soil.get(r, set())), None)
                    if r_with_analysis:
                        rover_loc = rover_locations.get(r_with_analysis) # Get current location
                        current_wp = rover_loc

                        # Navigate to a communication waypoint
                        if current_wp and self.comm_waypoints:
                             if current_wp not in self.comm_waypoints:
                                 min_dist = math.inf
                                 for cw in self.comm_waypoints:
                                     if current_wp in self.waypoint_distances and cw in self.waypoint_distances[current_wp]:
                                         min_dist = min(min_dist, self.waypoint_distances[current_wp][cw])
                                 if min_dist != math.inf:
                                     step_cost += min_dist
                                 else: step_cost += math.inf # Cannot reach comm waypoint
                             # else: already at a comm waypoint, nav cost is 0
                        else: step_cost += math.inf # Cannot find comm waypoints or rover location unknown

                        step_cost += 1 # communicate action

                    else: # Should not happen if analysis_exists is true
                         step_cost += math.inf

                else: # Analysis needs to be sampled
                    # Find an equipped rover
                    equipped_rover = next(iter(self.equipped_rovers['soil']), None) # Pick any equipped rover
                    if equipped_rover:
                        rover_loc = rover_locations.get(equipped_rover)
                        current_wp = rover_loc

                        # 1. Navigate to Sample location
                        sample_wp = wp
                        if current_wp and sample_wp:
                            if current_wp != sample_wp:
                                step_cost += self.waypoint_distances.get(current_wp, {}).get(sample_wp, math.inf)
                            current_wp = sample_wp # Assume it reaches
                        else: step_cost += math.inf # Cannot find sample waypoint or rover location unknown

                        # Check if store is full (needs drop before sample)
                        store = self.rover_stores.get(equipped_rover)
                        if store and store_status.get(store) == 'full':
                            step_cost += 1 # Cost for drop action

                        # 2. Sample
                        step_cost += 1 # Cost for sample action

                        # 3. Navigate to Communication waypoint
                        if current_wp and self.comm_waypoints:
                            if current_wp not in self.comm_waypoints:
                                min_dist = math.inf
                                for cw in self.comm_waypoints:
                                     if current_wp in self.waypoint_distances and cw in self.waypoint_distances[current_wp]:
                                         min_dist = min(min_dist, self.waypoint_distances[current_wp][cw])
                                if min_dist != math.inf:
                                    step_cost += min_dist
                                else: step_cost += math.inf # Cannot reach comm waypoint
                            # else: already at a comm waypoint, nav cost is 0
                        else: step_cost += math.inf # Cannot find comm waypoints or previous step failed

                        # 4. Communicate
                        step_cost += 1 # communicate action

                    else: # No equipped rover found
                         step_cost += math.inf # Cannot achieve soil goal

                total_cost += step_cost


            elif predicate == 'communicated_rock_data':
                wp = parts[1]
                if wp in communicated_rock:
                    continue # Goal already met

                # Cost for this specific rock goal
                step_cost = 0

                # Check if analysis exists
                analysis_exists = any(wp in have_rock.get(r, set()) for r in self.equipped_rovers['rock'])

                if analysis_exists:
                    # Find a rover that has the analysis
                    r_with_analysis = next((r for r in self.equipped_rovers['rock'] if wp in have_rock.get(r, set())), None)
                    if r_with_analysis:
                        rover_loc = rover_locations.get(r_with_analysis) # Get current location
                        current_wp = rover_loc

                        # Add navigation cost to a communication waypoint
                        if current_wp and self.comm_waypoints:
                             if current_wp not in self.comm_waypoints:
                                 min_dist = math.inf
                                 for cw in self.comm_waypoints:
                                     if current_wp in self.waypoint_distances and cw in self.waypoint_distances[current_wp]:
                                         min_dist = min(min_dist, self.waypoint_distances[current_wp][cw])
                                 if min_dist != math.inf:
                                     step_cost += min_dist
                                 else: step_cost += math.inf # Cannot reach comm waypoint
                             # else: already at a comm waypoint, nav cost is 0
                        else: step_cost += math.inf # Cannot find comm waypoints or rover location unknown

                        step_cost += 1 # communicate action

                    else: # Should not happen if analysis_exists is true
                         step_cost += math.inf

                else: # Analysis needs to be sampled
                    # Find an equipped rover
                    equipped_rover = next(iter(self.equipped_rovers['rock']), None) # Pick any equipped rover
                    if equipped_rover:
                        rover_loc = rover_locations.get(equipped_rover)
                        current_wp = rover_loc

                        # 1. Navigate to Sample location
                        sample_wp = wp
                        if current_wp and sample_wp:
                            if current_wp != sample_wp:
                                step_cost += self.waypoint_distances.get(current_wp, {}).get(sample_wp, math.inf)
                            current_wp = sample_wp # Assume it reaches
                        else: step_cost += math.inf # Cannot find sample waypoint or rover location unknown

                        # Check if store is full (needs drop before sample)
                        store = self.rover_stores.get(equipped_rover)
                        if store and store_status.get(store) == 'full':
                            step_cost += 1 # Cost for drop action

                        # 2. Sample
                        step_cost += 1 # Cost for sample action

                        # 3. Navigate to Communication waypoint
                        if current_wp and self.comm_waypoints:
                            if current_wp not in self.comm_waypoints:
                                min_dist = math.inf
                                for cw in self.comm_waypoints:
                                     if current_wp in self.waypoint_distances and cw in self.waypoint_distances[current_wp]:
                                         min_dist = min(min_dist, self.waypoint_distances[current_wp][cw])
                                if min_dist != math.inf:
                                    step_cost += min_dist
                                else: step_cost += math.inf # Cannot reach comm waypoint
                            # else: already at a comm waypoint, nav cost is 0
                        else: step_cost += math.inf # Cannot find comm waypoints or previous step failed

                        # 4. Communicate
                        step_cost += 1 # communicate action

                    else: # No equipped rover found
                         step_cost += math.inf # Cannot achieve rock goal

                total_cost += step_cost


            elif predicate == 'communicated_image_data':
                obj, mode = parts[1], parts[2]
                if (obj, mode) in communicated_image:
                    continue # Goal already met

                # Cost for this specific image goal
                step_cost = 0

                # Check if image exists
                image_exists = any((obj, mode) in have_image.get(r, set()) for r in self.equipped_rovers['imaging'])

                if image_exists:
                    # Find a rover that has the image
                    r_with_image = next((r for r in self.equipped_rovers['imaging'] if (obj, mode) in have_image.get(r, set())), None)
                    if r_with_image:
                        rover_loc = rover_locations.get(r_with_image) # Get current location
                        current_wp = rover_loc

                        # Navigate to a communication waypoint
                        if current_wp and self.comm_waypoints:
                             if current_wp not in self.comm_waypoints:
                                 min_dist = math.inf
                                 for cw in self.comm_waypoints:
                                     if current_wp in self.waypoint_distances and cw in self.waypoint_distances[current_wp]:
                                         min_dist = min(min_dist, self.waypoint_distances[current_wp][cw])
                                 if min_dist != math.inf:
                                     step_cost += min_dist
                                 else: step_cost += math.inf # Cannot reach comm waypoint
                             # else: already at a comm waypoint, nav cost is 0
                        else: step_cost += math.inf # Cannot find comm waypoints or rover location unknown

                        step_cost += 1 # communicate action

                    else: # Should not happen if image_exists is true
                         step_cost += math.inf

                else: # Image needs to be taken
                    # Find an equipped rover with a suitable camera
                    equipped_rover = None
                    suitable_camera = None
                    for r in self.equipped_rovers['imaging']:
                        for cam in self.rover_cameras.get(r, set()):
                            if mode in self.camera_modes.get(cam, set()):
                                equipped_rover = r
                                suitable_camera = cam
                                break
                        if equipped_rover: break # Found a suitable rover/camera

                    if equipped_rover and suitable_camera:
                        rover_loc = rover_locations.get(equipped_rover)
                        current_wp = rover_loc # Start navigation from current location

                        # Find image waypoint visible from objective
                        image_waypoint = next(iter(self.objective_waypoints.get(obj, set())), None) # Pick any suitable waypoint

                        # Find calibration target and waypoint
                        cal_target = self.calibration_targets.get(suitable_camera)
                        cal_waypoint = next(iter(self.objective_waypoints.get(cal_target, set())), None) # Pick any suitable waypoint

                        # Check if camera is calibrated
                        is_calibrated = suitable_camera in calibrated_cameras.get(equipped_rover, set())

                        # --- Navigation and Actions Sequence ---

                        # 1. Calibrate (if needed)
                        if not is_calibrated:
                            if current_wp and cal_waypoint:
                                if current_wp != cal_waypoint:
                                     step_cost += self.waypoint_distances.get(current_wp, {}).get(cal_waypoint, math.inf)
                                current_wp = cal_waypoint # Assume it reaches
                                step_cost += 1 # calibrate
                            else: # Cannot find cal waypoint or rover location unknown
                                step_cost += math.inf # Cannot calibrate

                        # 2. Take Image
                        if current_wp and image_waypoint:
                            if current_wp != image_waypoint:
                                step_cost += self.waypoint_distances.get(current_wp, {}).get(image_waypoint, math.inf)
                            current_wp = image_waypoint # Assume it reaches
                            step_cost += 1 # take_image
                        else: # Cannot find image waypoint or previous step failed
                            step_cost += math.inf # Cannot take image

                        # 3. Navigate to Communication waypoint
                        if current_wp and self.comm_waypoints:
                            if current_wp not in self.comm_waypoints:
                                min_dist = math.inf
                                for cw in self.comm_waypoints:
                                     if current_wp in self.waypoint_distances and cw in self.waypoint_distances[current_wp]:
                                         min_dist = min(min_dist, self.waypoint_distances[current_wp][cw])
                                if min_dist != math.inf:
                                    step_cost += min_dist
                                else: step_cost += math.inf # Cannot reach comm waypoint
                            # else: already at a comm waypoint, nav cost is 0
                        else: step_cost += math.inf # Cannot find comm waypoints or previous step failed

                        # 4. Communicate
                        step_cost += 1 # communicate action

                        total_cost += step_cost

                    else: # No equipped rover/camera found
                         step_cost += math.inf # Cannot achieve image goal


        # The total_cost will be 0 if and only if all goals were already in the state.
        # If any step_cost was math.inf, total_cost will be math.inf.
        # This is appropriate for unreachable goals in GBFS.

        return total_cost
