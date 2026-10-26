from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque
import sys

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty facts or malformed strings defensively
    if not isinstance(fact, str) or len(fact) < 2 or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at rover1 waypoint1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # The fact must have at least as many parts as there are arguments in the pattern
    if len(parts) < len(args):
        return False
    # Check if each part matches the corresponding argument pattern
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def bfs(graph, start_node):
    """Perform Breadth-First Search to find distances from a start node."""
    distances = {node: float('inf') for node in graph}
    if start_node in distances: # Ensure start_node is a valid node in the graph
        distances[start_node] = 0
        queue = deque([start_node])
        while queue:
            current_node = queue.popleft()
            # Check if current_node is still valid (e.g., not a disconnected node)
            if current_node in graph:
                for neighbor in graph.get(current_node, []):
                    if distances[neighbor] == float('inf'):
                        distances[neighbor] = distances[current_node] + 1
                        queue.append(neighbor)
    return distances

def compute_all_pairs_shortest_paths(graph):
    """Compute shortest path distances between all pairs of nodes in a graph."""
    all_distances = {}
    # Ensure all nodes that appear as neighbors are also in the graph keys
    all_nodes = set(graph.keys())
    for neighbors in graph.values():
        all_nodes.update(neighbors)

    # Rebuild graph to include all nodes as keys, even if they have no outgoing edges
    full_graph = {node: graph.get(node, []) for node in all_nodes}

    for start_node in full_graph:
        all_distances[start_node] = bfs(full_graph, start_node)
    return all_distances

def min_dist_from_set(source_rovers_or_wps, target_wp, current_rover_locations, distances):
    """
    Calculate the minimum distance from a set of source rovers/waypoints to a target waypoint.
    Returns float('inf') if no source can reach the target.
    """
    min_d = float('inf')
    source_wps = set()
    for source in source_rovers_or_wps:
        if source in current_rover_locations: # It's a rover object name
            source_wps.add(current_rover_locations[source])
        else: # Assume it's a waypoint string
            source_wps.add(source)

    if not source_wps:
        return float('inf') # No valid source locations provided

    found_reachable_source = False
    for swp in source_wps:
        # Check if source waypoint exists in the distance map and target is reachable from it
        if swp in distances and target_wp in distances[swp]:
             dist = distances[swp][target_wp]
             if dist != float('inf'): # Only consider reachable targets
                 min_d = min(min_d, dist)
                 found_reachable_source = True

    return min_d if found_reachable_source else float('inf')


class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve all goal
    conditions by summing the estimated costs for each unsatisfied goal
    independently. The cost for each goal is estimated based on the current
    state of data collection (sample, analyze, image) and the movement
    required to perform the necessary actions (sampling/imaging location,
    lander location for communication). Movement costs are estimated using
    shortest path distances on the waypoint graph.

    # Assumptions
    - Each unsatisfied goal is treated independently. Synergies (e.g., one trip
      to collect multiple samples, or one trip to the lander to communicate
      multiple data types) are not explicitly modeled or optimized.
    - Rovers have static capabilities (soil, rock, imaging equipment).
    - Cameras have static properties (on-board rover, supported modes,
      calibration target).
    - Objective visibility from waypoints is static.
    - Sample locations are static.
    - Lander location is static.
    - The waypoint graph is undirected and traversable by all rovers based on
      `visible` facts. `can_traverse` facts are assumed to mirror `visible`.
    - The cost of any single action (move, sample, analyze, calibrate,
      take_image, communicate) is 1. Movement cost between waypoints is the
      shortest path distance (number of `move` actions).
    - For a goal requiring a specific type of equipped rover or a camera with
      specific properties, it is assumed that at least one such rover/camera
      exists if the problem instance is solvable. The heuristic will attempt
      to find the closest available resource.
    - `at_soil_data` and `at_rock_data` are treated as global facts once produced.
      `have_image` is tied to the specific rover that took the image.
    - Communication of soil/rock data can be done by any rover at the lander
      waypoint if the `at_soil_data`/`at_rock_data` fact exists.
    - Communication of image data must be done by the specific rover that has
      the `have_image` fact, while that rover is at the lander waypoint.

    # Heuristic Initialization
    - The waypoint graph is constructed from `visible` facts.
    - All-pairs shortest path distances between waypoints are precomputed using BFS.
    - The lander's waypoint is identified.
    - Static information about rover capabilities, camera properties, objective
      visibility, and sample locations is extracted.
    - The set of goal conditions is stored.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Parse the current state to identify the location of each rover, which
       rovers have collected which samples/images, which data has been analyzed
       (`at_soil_data`, `at_rock_data`), and which cameras are calibrated.
    2. Initialize the total heuristic cost `h` to 0.
    3. Iterate through each goal condition:
       a. If the goal condition is already satisfied in the current state, add 0
          cost for this goal and continue to the next goal.
       b. If the goal is `(communicated_soil_data W)`:
          - If `(at_soil_data W)` is true: Add 1 (communicate) + minimum distance
            from any rover's current location to the lander waypoint.
          - Else if `(have_soil_sample R W)` is true for some rover R: Add 1
            (analyze) + 1 (communicate) + minimum distance from waypoint W
            (where analysis is assumed to happen) to the lander waypoint.
          - Else (need to sample): Add 1 (sample) + 1 (analyze) + 1 (communicate)
            + minimum distance from any soil-equipped rover's current location
            to waypoint W + minimum distance from waypoint W to the lander waypoint.
       c. If the goal is `(communicated_rock_data W)`: Follow the same logic as
          soil data, replacing soil with rock and using rock-equipped rovers.
       d. If the goal is `(communicated_image_data O M)`:
          - Find a camera C that supports mode M and is a calibration target for
            objective O, and the rover R it is on. (Assume one such pair exists).
          - Find waypoints visible from objective O.
          - If `(have_image R O M)` is true for rover R: Add 1 (communicate) +
            distance from rover R's current location to the lander waypoint.
          - Else if `(calibrated C O)` is true for camera C: Add 1 (take_image)
            + 1 (communicate) + minimum distance from rover R's current location
            to any waypoint visible from O + minimum distance from any waypoint
            visible from O to the lander waypoint.
          - Else (need to calibrate): Add 1 (calibrate) + 1 (take_image) + 1
            (communicate) + minimum distance from rover R's current location
            to any waypoint visible from O + minimum distance from any waypoint
            visible from O to the lander waypoint.
       e. If any required movement stage has infinite distance (e.g., target
          unreachable), the cost for this goal is infinite.
    4. Sum the costs for all unsatisfied goals to get the total heuristic value.
    5. Return the total heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # Precompute static information
        self.waypoint_graph = {}
        self.lander_waypoint = None
        self.rover_capabilities = {} # {rover: {'soil', 'rock', 'imaging'}}
        self.camera_info = {} # {camera: {'on_board': rover, 'supports': {mode}, 'calibration_target': objective}}
        self.objective_visibility = {} # {objective: {waypoint}}
        self.soil_sample_locations = set()
        self.rock_sample_locations = set()
        self.all_rovers = set() # Keep track of all rover names

        # Parse static facts
        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue
            pred = parts[0]
            if pred == 'visible':
                w1, w2 = parts[1], parts[2]
                self.waypoint_graph.setdefault(w1, []).append(w2)
                self.waypoint_graph.setdefault(w2, []).append(w1) # Assuming visible is symmetric
            elif pred == 'at_lander':
                # lander, wp = parts[1], parts[2] # lander object name is not needed
                self.lander_waypoint = parts[2]
            elif pred.startswith('equipped_for_'):
                cap_type = pred.split('_')[2] # e.g., 'soil', 'rock', 'imaging'
                rover = parts[1]
                self.rover_capabilities.setdefault(rover, set()).add(cap_type)
                self.all_rovers.add(rover)
            elif pred == 'on_board':
                camera, rover = parts[1], parts[2]
                self.camera_info.setdefault(camera, {})['on_board'] = rover
                self.all_rovers.add(rover)
            elif pred == 'supports':
                camera, mode = parts[1], parts[2]
                self.camera_info.setdefault(camera, {}).setdefault('supports', set()).add(mode)
            elif pred == 'calibration_target':
                camera, objective = parts[1], parts[2]
                self.camera_info.setdefault(camera, {})['calibration_target'] = objective
            elif pred == 'visible_from':
                objective, waypoint = parts[1], parts[2]
                self.objective_visibility.setdefault(objective, set()).add(waypoint)
            elif pred == 'at_soil_sample':
                self.soil_sample_locations.add(parts[1])
            elif pred == 'at_rock_sample':
                self.rock_sample_locations.add(parts[1])
            # Also identify rovers from 'at' facts in initial state if available, though static facts usually don't include 'at'
            # For robustness, could parse initial state here too, but problem description implies static facts are enough for capabilities etc.
            # Let's rely on equipped_for_* and on_board to identify rovers.

        # Compute all-pairs shortest paths
        self.distances = compute_all_pairs_shortest_paths(self.waypoint_graph)

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Parse current state information
        current_rover_locations = {} # {rover: waypoint}
        current_have_soil = {} # {rover: {waypoint}}
        current_have_rock = {} # {rover: {waypoint}}
        current_have_image = {} # {rover: {(objective, mode)}}
        current_at_soil_data = set() # {waypoint}
        current_at_rock_data = set() # {waypoint}
        current_calibrated = {} # {camera: {objective}}

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue
            pred = parts[0]
            if pred == 'at' and parts[1].startswith('rover'):
                rover, wp = parts[1], parts[2]
                current_rover_locations[rover] = wp
            elif pred == 'have_soil_sample':
                rover, wp = parts[1], parts[2]
                current_have_soil.setdefault(rover, set()).add(wp)
            elif pred == 'have_rock_sample':
                rover, wp = parts[1], parts[2]
                current_have_rock.setdefault(rover, set()).add(wp)
            elif pred == 'have_image':
                rover, obj, mode = parts[1], parts[2], parts[3]
                current_have_image.setdefault(rover, set()).add((obj, mode))
            elif pred == 'at_soil_data':
                wp = parts[1]
                current_at_soil_data.add(wp)
            elif pred == 'at_rock_data':
                wp = parts[1]
                current_at_rock_data.add(wp)
            elif pred == 'calibrated':
                camera, obj = parts[1], parts[2]
                current_calibrated.setdefault(camera, set()).add(obj)

        total_cost = 0

        for goal in self.goals:
            if goal in state:
                continue # Goal already satisfied

            parts = get_parts(goal)
            if not parts: continue
            pred = parts[0]
            goal_cost = 0 # Cost for this specific goal

            if pred == 'communicated_soil_data':
                W = parts[1]
                if W in current_at_soil_data:
                    # Need communicate
                    goal_cost += 1
                    move_cost = min_dist_from_set(current_rover_locations.keys(), self.lander_waypoint, current_rover_locations, self.distances)
                    if move_cost == float('inf'): return float('inf') # Impossible goal
                    goal_cost += move_cost
                else:
                    rover_with_sample = None
                    for r, samples in current_have_soil.items():
                        if W in samples:
                            rover_with_sample = r
                            break

                    if rover_with_sample:
                        # Need analyze and communicate
                        goal_cost += 1 # analyze
                        goal_cost += 1 # communicate
                        # Assume analysis happens at W, move from W to lander
                        move_cost = min_dist_from_set({W}, self.lander_waypoint, current_rover_locations, self.distances)
                        if move_cost == float('inf'): return float('inf') # Impossible goal
                        goal_cost += move_cost
                    else:
                        # Need sample, analyze, communicate
                        goal_cost += 1 # sample
                        goal_cost += 1 # analyze
                        goal_cost += 1 # communicate
                        equipped_rovers = [r for r, caps in self.rover_capabilities.items() if 'soil' in caps]
                        if not equipped_rovers: return float('inf') # No equipped rover available

                        move_cost_to_sample = min_dist_from_set(equipped_rovers, W, current_rover_locations, self.distances)
                        if move_cost_to_sample == float('inf'): return float('inf') # Cannot reach sample location
                        goal_cost += move_cost_to_sample

                        move_cost_to_lander = min_dist_from_set({W}, self.lander_waypoint, current_rover_locations, self.distances) # Move from W to lander
                        if move_cost_to_lander == float('inf'): return float('inf') # Cannot reach lander from sample location
                        goal_cost += move_cost_to_lander


            elif pred == 'communicated_rock_data':
                W = parts[1]
                if W in current_at_rock_data:
                    # Need communicate
                    goal_cost += 1
                    move_cost = min_dist_from_set(current_rover_locations.keys(), self.lander_waypoint, current_rover_locations, self.distances)
                    if move_cost == float('inf'): return float('inf') # Impossible goal
                    goal_cost += move_cost
                else:
                    rover_with_sample = None
                    for r, samples in current_have_rock.items():
                        if W in samples:
                            rover_with_sample = r
                            break

                    if rover_with_sample:
                        # Need analyze and communicate
                        goal_cost += 1 # analyze
                        goal_cost += 1 # communicate
                        # Assume analysis happens at W, move from W to lander
                        move_cost = min_dist_from_set({W}, self.lander_waypoint, current_rover_locations, self.distances)
                        if move_cost == float('inf'): return float('inf') # Impossible goal
                        goal_cost += move_cost
                    else:
                        # Need sample, analyze, communicate
                        goal_cost += 1 # sample
                        goal_cost += 1 # analyze
                        goal_cost += 1 # communicate
                        equipped_rovers = [r for r, caps in self.rover_capabilities.items() if 'rock' in caps]
                        if not equipped_rovers: return float('inf') # No equipped rover available

                        move_cost_to_sample = min_dist_from_set(equipped_rovers, W, current_rover_locations, self.distances)
                        if move_cost_to_sample == float('inf'): return float('inf') # Cannot reach sample location
                        goal_cost += move_cost_to_sample

                        move_cost_to_lander = min_dist_from_set({W}, self.lander_waypoint, current_rover_locations, self.distances) # Move from W to lander
                        if move_cost_to_lander == float('inf'): return float('inf') # Cannot reach lander from sample location
                        goal_cost += move_cost_to_lander


            elif pred == 'communicated_image_data':
                O, M = parts[1], parts[2]

                rover_with_image = None
                for r, images in current_have_image.items():
                    if (O, M) in images:
                        rover_with_image = r
                        break

                if rover_with_image:
                    # Need communicate
                    goal_cost += 1
                    move_cost = min_dist_from_set({rover_with_image}, self.lander_waypoint, current_rover_locations, self.distances)
                    if move_cost == float('inf'): return float('inf') # Impossible goal
                    goal_cost += move_cost
                else:
                    # Find a suitable camera/rover
                    suitable_rover = None
                    suitable_camera = None
                    for c, info in self.camera_info.items():
                        if M in info.get('supports', set()) and info.get('calibration_target') == O:
                            r = info.get('on_board')
                            # Ensure rover exists, is currently placed, and is equipped for imaging
                            if r and r in current_rover_locations and 'imaging' in self.rover_capabilities.get(r, set()):
                                suitable_rover = r
                                suitable_camera = c
                                break # Found a suitable rover/camera

                    if suitable_rover is None:
                         # Should not happen in solvable instances, but indicates this goal is impossible
                         return float('inf')

                    visible_wps = self.objective_visibility.get(O, set())
                    if not visible_wps:
                        # Should not happen in solvable instances
                        return float('inf')

                    if O in current_calibrated.get(suitable_camera, set()):
                        # Need take_image and communicate
                        goal_cost += 1 # take_image
                        goal_cost += 1 # communicate
                        move_cost_to_visible = min_dist_from_set({suitable_rover}, visible_wps, current_rover_locations, self.distances) # Move to visible wp
                        if move_cost_to_visible == float('inf'): return float('inf') # Cannot reach visible waypoint
                        goal_cost += move_cost_to_visible

                        move_cost_to_lander = min_dist_from_set(visible_wps, self.lander_waypoint, current_rover_locations, self.distances) # Move from visible wp to lander
                        if move_cost_to_lander == float('inf'): return float('inf') # Cannot reach lander from visible waypoint
                        goal_cost += move_cost_to_lander
                    else:
                        # Need calibrate, take_image, communicate
                        goal_cost += 1 # calibrate
                        goal_cost += 1 # take_image
                        goal_cost += 1 # communicate
                        move_cost_to_visible = min_dist_from_set({suitable_rover}, visible_wps, current_rover_locations, self.distances) # Move to visible wp
                        if move_cost_to_visible == float('inf'): return float('inf') # Cannot reach visible waypoint
                        goal_cost += move_cost_to_visible

                        move_cost_to_lander = min_dist_from_set(visible_wps, self.lander_waypoint, current_rover_locations, self.distances) # Move from visible wp to lander
                        if move_cost_to_lander == float('inf'): return float('inf') # Cannot reach lander from visible waypoint
                        goal_cost += move_cost_to_lander

            # If goal_cost is still 0 here, it means the goal type was not recognized or had no cost added (shouldn't happen for unsatisfied goals)
            # Add the cost for this goal to the total
            total_cost += goal_cost

        return total_cost
