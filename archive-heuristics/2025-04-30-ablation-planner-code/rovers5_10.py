# Need to import Heuristic base class, assuming it's available in the environment
# from heuristics.heuristic_base import Heuristic

from fnmatch import fnmatch
import collections

# Helper functions to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle empty or invalid fact string case defensively
    if not fact or not isinstance(fact, str) or fact[0] != '(' or fact[-1] != ')':
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

# BFS for shortest path on unweighted graph
def bfs(graph, start_node):
    """Computes shortest distances from start_node to all reachable nodes in the graph."""
    distances = {node: float('inf') for node in graph}
    if start_node not in graph:
         # Start node might not be in graph if it's an object of a different type
         # or if the graph is empty or isolated.
         return distances # All distances remain inf

    distances[start_node] = 0
    queue = collections.deque([start_node])
    while queue:
        current_node = queue.popleft()
        # Check if current_node is a valid key in the graph (should be if added from graph keys)
        if current_node in graph:
            for neighbor in graph.get(current_node, []):
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)
    return distances

def compute_all_pairs_shortest_paths(graph):
    """Computes shortest distances between all pairs of nodes in the graph."""
    all_distances = {}
    # Collect all unique nodes that appear as keys or values in the graph
    all_nodes = set(graph.keys())
    for neighbors in graph.values():
        all_nodes.update(neighbors)

    # Ensure all nodes are keys in the distance map, even if isolated
    for node in all_nodes:
         all_distances[node] = bfs(graph, node)
    return all_distances

# Assume Heuristic base class is available in the environment
# class Heuristic:
#     def __init__(self, task):
#         self.goals = task.goals
#         self.static = task.static
#     def __call__(self, node):
#         raise NotImplementedError

class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the cost to reach a goal state by summing the estimated
    costs for each unmet communication goal. The cost for each goal includes the
    necessary actions (sample, calibrate, take_image, communicate) and the estimated
    movement costs between required locations (sample waypoint, calibration waypoint,
    image waypoint, lander-visible waypoint). Movement costs are calculated as
    shortest paths on the waypoint graph defined by 'visible' predicates.

    # Assumptions
    - Action costs are uniform (1).
    - The graph for movement is defined by the 'visible' predicates, and all rovers
      can traverse any visible link.
    - If a sample is not at its original waypoint and no rover has it, it is assumed
      to be on a rover for heuristic calculation purposes (simplification).
    - When multiple rovers/cameras/waypoint options exist for a task step (e.g.,
      sampling, imaging, calibrating), the heuristic estimates the cost based on
      the option that minimizes the required sequential movement for that specific
      task, picking the best intermediate waypoints greedily based on distance from
      the previous step's location.
    - Unreachable required waypoints result in an infinite heuristic value.

    # Heuristic Initialization
    The constructor parses the static facts from the task to build necessary data
    structures:
    - Set of goal predicates.
    - Lander location and set of waypoints visible from the lander.
    - Waypoint graph based on 'visible' predicates.
    - All-pairs shortest path distances between waypoints using BFS.
    - Sets of rovers equipped for soil, rock, and imaging.
    - Mapping from store to its owning rover.
    - Mapping from camera to supported modes.
    - Mapping from camera to its calibration target objective.
    - Mapping from camera to its onboard rover.
    - Mapping from objective to set of waypoints it's visible from.
    - Sets of all objects by type (rovers, waypoints, cameras, objectives, modes, stores).

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Parse the current state to identify: rover locations, locations of soil/rock samples,
       which rovers have which samples/images, which cameras are calibrated, and which
       stores are full.
    2. Initialize the total heuristic cost `h` to 0.
    3. For each goal predicate in the task's goals:
       - If the goal predicate is already true in the current state, continue to the next goal.
       - If the goal is `(communicated_soil_data w)`:
         - Add 1 for the `communicate_soil_data` action.
         - Check if any rover has `(have_soil_analysis r w)`.
         - If no rover has the sample:
           - Add 1 for the `sample_soil` action.
           - If `(at_soil_sample w)` is true:
             - Estimate movement cost to sample: Find the equipped soil rover closest to `w`. Add the distance from its current location to `w`.
             - Add 1 for the `drop` action if the chosen rover's store is full.
             - Estimate movement cost after sampling: Add the minimum distance from `w` to any lander-visible waypoint.
           - If `(at_soil_sample w)` is false (sample already collected but not in `have_soil_analysis` - simplified assumption): No additional sampling/movement cost for sampling phase.
         - If a rover *does* have the sample:
           - Estimate movement cost to communicate: Find a rover with the sample. Add the minimum distance from its current location to any lander-visible waypoint.
         - If any required movement path is unreachable (distance is infinity), the goal is unreachable, set `h` to infinity and return.
       - If the goal is `(communicated_rock_data w)`: Follow similar logic as soil data, using rock-specific predicates and rovers.
       - If the goal is `(communicated_image_data o m)`:
         - Add 1 for the `communicate_image_data` action.
         - Check if any rover has `(have_image r o m)`.
         - If no rover has the image:
           - Add 1 for the `take_image` action.
           - Find suitable rover(s) and camera(s) (equipped for imaging, camera on board, supports mode m).
           - Estimate the minimum sequential movement cost for the calibrate (if needed) -> image -> communicate sequence, minimized over suitable rovers/cameras and optimal intermediate waypoints.
           - If calibration is needed: Add 1 for the `calibrate` action.
           - Add the calculated minimum sequential movement cost.
           - If any required movement path is unreachable, the goal is unreachable, set `h` to infinity and return.
         - If a rover *does* have the image:
           - Estimate movement cost to communicate: Find a rover with the image. Add the minimum distance from its current location to any lander-visible waypoint.
         - If any required movement path is unreachable, the goal is unreachable, set `h` to infinity and return.
    4. Return the total heuristic cost `h`.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals

        # Data structures to store static information
        self.lander_location = None
        self.lander_visible_wps = set()
        self.waypoint_graph = collections.defaultdict(list)
        self.equipped_soil = set()
        self.equipped_rock = set()
        self.equipped_imaging = set()
        self.store_owner = {} # {store: rover}
        self.camera_supports = collections.defaultdict(set) # {camera: {mode, ...}}
        self.camera_cal_target = {} # {camera: objective}
        self.camera_on_board = {} # {camera: rover}
        self.objective_visible_from = collections.defaultdict(set) # {objective: {waypoint, ...}}

        # Sets of all objects by type (for easier iteration/lookup)
        self.all_rovers = set()
        self.all_waypoints = set()
        self.all_cameras = set()
        self.all_objectives = set()
        self.all_modes = set()
        self.all_stores = set()

        # Collect all objects first
        for fact_str in task.init + task.static + list(task.goals):
             parts = get_parts(fact_str)
             if not parts: continue
             predicate = parts[0]
             args = parts[1:]

             # Infer object types from predicates and arguments
             if predicate == "at": # (at ?x - rover ?y - waypoint) or (at_lander ?x - lander ?y - waypoint)
                 if args[0].startswith('rover'): self.all_rovers.add(args[0])
                 if args[0].startswith('general'): pass # Assuming general is the lander
                 self.all_waypoints.add(args[1])
             elif predicate == "at_lander": # (at_lander ?x - lander ?y - waypoint)
                 # Assuming lander type is 'lander' and name is 'general'
                 self.lander_location = args[1]
                 self.all_waypoints.add(args[1])
             elif predicate == "can_traverse": # (can_traverse ?r - rover ?x - waypoint ?y - waypoint)
                 self.all_rovers.add(args[0])
                 self.all_waypoints.add(args[1])
                 self.all_waypoints.add(args[2])
             elif predicate in ["equipped_for_soil_analysis", "equipped_for_rock_analysis", "equipped_for_imaging"]: # (?r - rover)
                 self.all_rovers.add(args[0])
             elif predicate in ["empty", "full"]: # (?s - store)
                 self.all_stores.add(args[0])
             elif predicate in ["have_rock_analysis", "have_soil_analysis"]: # (?r - rover ?w - waypoint)
                 self.all_rovers.add(args[0])
                 self.all_waypoints.add(args[1])
             elif predicate == "calibrated": # (?c - camera ?r - rover)
                 self.all_cameras.add(args[0])
                 self.all_rovers.add(args[1])
             elif predicate == "supports": # (?c - camera ?m - mode)
                 self.all_cameras.add(args[0])
                 self.all_modes.add(args[1])
             elif predicate == "visible": # (?w - waypoint ?p - waypoint)
                 self.all_waypoints.add(args[0])
                 self.all_waypoints.add(args[1])
             elif predicate == "have_image": # (?r - rover ?o - objective ?m - mode)
                 self.all_rovers.add(args[0])
                 self.all_objectives.add(args[1])
                 self.all_modes.add(args[2])
             elif predicate in ["communicated_soil_data", "communicated_rock_data"]: # (?w - waypoint)
                 self.all_waypoints.add(args[0])
             elif predicate == "communicated_image_data": # (?o - objective ?m - mode)
                 self.all_objectives.add(args[0])
                 self.all_modes.add(args[1])
             elif predicate in ["at_soil_sample", "at_rock_sample"]: # (?w - waypoint)
                 self.all_waypoints.add(args[0])
             elif predicate == "visible_from": # (?o - objective ?w - waypoint)
                 self.all_objectives.add(args[0])
                 self.all_waypoints.add(args[1])
             elif predicate == "store_of": # (?s - store ?r - rover)
                 self.all_stores.add(args[0])
                 self.all_rovers.add(args[1])
             elif predicate == "calibration_target": # (?i - camera ?o - objective)
                 self.all_cameras.add(args[0])
                 self.all_objectives.add(args[1])
             elif predicate == "on_board": # (?i - camera ?r - rover)
                 self.all_cameras.add(args[0])
                 self.all_rovers.add(args[1])

        # Parse static facts again, now that we have object sets
        for fact_str in task.static:
            parts = get_parts(fact_str)
            if not parts: continue

            predicate = parts[0]
            args = parts[1:]

            if predicate == "at_lander":
                self.lander_location = args[1] # Assuming only one lander
            elif predicate == "visible":
                wp1, wp2 = args
                self.waypoint_graph[wp1].append(wp2)
                self.waypoint_graph[wp2].append(wp1) # Assuming visible is symmetric
            elif predicate == "equipped_for_soil_analysis":
                self.equipped_soil.add(args[0])
            elif predicate == "equipped_for_rock_analysis":
                self.equipped_rock.add(args[0])
            elif predicate == "equipped_for_imaging":
                self.equipped_imaging.add(args[0])
            elif predicate == "store_of":
                self.store_owner[args[0]] = args[1]
            elif predicate == "supports":
                self.camera_supports[args[0]].add(args[1])
            elif predicate == "calibration_target":
                self.camera_cal_target[args[0]] = args[1]
            elif predicate == "on_board":
                self.camera_on_board[args[0]] = args[1]
            elif predicate == "visible_from":
                self.objective_visible_from[args[0]].add(args[1])

        # Ensure all waypoints are keys in the graph, even if isolated
        full_waypoint_graph = {wp: self.waypoint_graph.get(wp, []) for wp in self.all_waypoints}
        self.waypoint_graph = full_waypoint_graph

        # Compute all-pairs shortest paths
        self.dist = compute_all_pairs_shortest_paths(self.waypoint_graph)

        # Precompute lander-visible waypoints
        self.lander_visible_wps = set()
        if self.lander_location:
             # Find all waypoints v where (visible v lander_location) is true
             # The 'visible' predicate is symmetric.
             for fact_str in task.static:
                 if match(fact_str, "visible", "*", self.lander_location):
                     self.lander_visible_wps.add(get_parts(fact_str)[1])
                 elif match(fact_str, "visible", self.lander_location, "*"):
                      self.lander_visible_wps.add(get_parts(fact_str)[2])

             # Also add the lander location itself if it's a waypoint
             if self.lander_location in self.all_waypoints:
                  self.lander_visible_wps.add(self.lander_location)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        h = 0

        # Parse current state facts
        rover_locations = {} # {rover: waypoint}
        soil_samples_at_wp = set() # {waypoint}
        rock_samples_at_wp = set() # {waypoint}
        have_soil_data = set() # {(rover, waypoint)}
        have_rock_data = set() # {(rover, waypoint)}
        have_image_data = set() # {(rover, objective, mode)}
        calibrated_cameras = set() # {(camera, rover)}
        full_stores = set() # {store}

        for fact_str in state:
            parts = get_parts(fact_str)
            if not parts: continue

            predicate = parts[0]
            args = parts[1:]

            if predicate == "at":
                # Could be rover or lander, but we only care about rover locations here
                if args[0] in self.all_rovers:
                    rover_locations[args[0]] = args[1]
            elif predicate == "at_soil_sample":
                soil_samples_at_wp.add(args[0])
            elif predicate == "at_rock_sample":
                rock_samples_at_wp.add(args[0])
            elif predicate == "have_soil_analysis":
                have_soil_data.add((args[0], args[1]))
            elif predicate == "have_rock_analysis":
                have_rock_data.add((args[0], args[1]))
            elif predicate == "have_image":
                have_image_data.add((args[0], args[1], args[2]))
            elif predicate == "calibrated":
                calibrated_cameras.add((args[0], args[1]))
            elif predicate == "full":
                full_stores.add(args[0])

        # Iterate through goals and sum costs for unmet goals
        for goal_str in self.goals:
            if goal_str in state:
                continue # Goal already met

            goal_parts = get_parts(goal_str)
            if not goal_parts: continue

            goal_predicate = goal_parts[0]
            goal_args = goal_parts[1:]

            if goal_predicate == "communicated_soil_data":
                w = goal_args[0]
                h += 1 # Cost for communicate action

                # Check if sample is collected
                has_sample = any((r, w) in have_soil_data for r in self.all_rovers)

                if not has_sample:
                    h += 1 # Cost for sample action
                    if w in soil_samples_at_wp:
                        # Need to move equipped rover to w
                        min_dist_to_sample_wp = float('inf')
                        best_sampler_rover = None
                        for r in self.equipped_soil:
                            if r in rover_locations:
                                r_loc = rover_locations[r]
                                if r_loc in self.dist and w in self.dist[r_loc]:
                                    if self.dist[r_loc][w] < min_dist_to_sample_wp:
                                        min_dist_to_sample_wp = self.dist[r_loc][w]
                                        best_sampler_rover = r

                        if best_sampler_rover:
                           h += min_dist_to_sample_wp # Move to sample wp
                           # Check store of this specific rover
                           if best_sampler_rover in self.store_owner and self.store_owner[best_sampler_rover] in full_stores:
                               h += 1 # Drop action
                           # Movement after sampling at w to comm_wp
                           min_dist_sample_wp_to_comm_wp = float('inf')
                           for lander_wp in self.lander_visible_wps:
                                if w in self.dist and lander_wp in self.dist[w]:
                                    min_dist_sample_wp_to_comm_wp = min(min_dist_sample_wp_to_comm_wp, self.dist[w][lander_wp])
                           if min_dist_sample_wp_to_comm_wp != float('inf'):
                                h += min_dist_sample_wp_to_comm_wp
                           else:
                               h = float('inf') # Cannot reach comm wp from sample wp
                               return h # Goal unreachable
                        else:
                            # Cannot reach sample wp from any equipped rover
                            h = float('inf')
                            return h # Goal unreachable
                    else:
                        # Sample not at wp, assume on a rover (but not in have_soil_data - simplified)
                        # This case is problematic in a strict sense, but for heuristic,
                        # assume the sample is on a rover and just needs communication.
                        # This falls into the 'has_sample' case for communication movement below.
                        pass # No extra cost for sampling/moving to sample wp if sample is gone from wp.

                # Movement to communication waypoint (either from sample wp or from rover holding sample)
                if has_sample:
                    # Find a rover that has the sample
                    rover_with_data_loc = None
                    for r, wp in have_soil_data:
                        if wp == w and r in rover_locations:
                            rover_with_data_loc = rover_locations[r]
                            break
                    if rover_with_data_loc:
                        # Movement from rover_with_data_loc to comm_wp
                        min_dist_r_to_l = float('inf')
                        for lander_wp in self.lander_visible_wps:
                            if rover_with_data_loc in self.dist and lander_wp in self.dist[rover_with_data_loc]:
                                min_dist_r_to_l = min(min_dist_r_to_l, self.dist[rover_with_data_loc][lander_wp])
                        if min_dist_r_to_l != float('inf'):
                            h += min_dist_r_to_l
                        else:
                            h = float('inf') # Cannot reach comm wp from rover loc
                            return h # Goal unreachable
                    else:
                        # Should not happen in a solvable state if has_sample is true
                        # Fallback: Assume some rover has it and estimate move from any rover
                        min_dist_r_to_l = float('inf')
                        for r_loc in rover_locations.values():
                             for lander_wp in self.lander_visible_wps:
                                 if r_loc in self.dist and lander_wp in self.dist[r_loc]:
                                     min_dist_r_to_l = min(min_dist_r_to_l, self.dist[r_loc][lander_wp])
                         if min_dist_r_to_l != float('inf'):
                              h += min_dist_r_to_l
                         else:
                              h = float('inf') # Cannot reach comm wp from any rover loc
                              return h # Goal unreachable
                # If not has_sample and sample was not at wp, we skipped sampling cost.
                # We still need to communicate. The movement cost is implicitly covered
                # by the assumption that a rover has it and needs to move.

            elif goal_predicate == "communicated_rock_data":
                 w = goal_args[0]
                 h += 1 # Cost for communicate action

                 # Check if sample is collected
                 has_sample = any((r, w) in have_rock_data for r in self.all_rovers)

                 if not has_sample:
                     h += 1 # Cost for sample action
                     if w in rock_samples_at_wp:
                         # Need to move equipped rover to w
                         min_dist_to_sample_wp = float('inf')
                         best_sampler_rover = None
                         for r in self.equipped_rock:
                             if r in rover_locations:
                                 r_loc = rover_locations[r]
                                 if r_loc in self.dist and w in self.dist[r_loc]:
                                     if self.dist[r_loc][w] < min_dist_to_sample_wp:
                                         min_dist_to_sample_wp = self.dist[r_loc][w]
                                         best_sampler_rover = r

                         if best_sampler_rover:
                            h += min_dist_to_sample_wp # Move to sample wp
                            # Check store of this specific rover
                            if best_sampler_rover in self.store_owner and self.store_owner[best_sampler_rover] in full_stores:
                                h += 1 # Drop action
                            # Movement after sampling at w to comm_wp
                            min_dist_sample_wp_to_comm_wp = float('inf')
                            for lander_wp in self.lander_visible_wps:
                                 if w in self.dist and lander_wp in self.dist[w]:
                                     min_dist_sample_wp_to_comm_wp = min(min_dist_sample_wp_to_comm_wp, self.dist[w][lander_wp])
                            if min_dist_sample_wp_to_comm_wp != float('inf'):
                                 h += min_dist_sample_wp_to_comm_wp
                            else:
                                h = float('inf') # Cannot reach comm wp from sample wp
                                return h # Goal unreachable
                         else:
                             # Cannot reach sample wp from any equipped rover
                             h = float('inf')
                             return h # Goal unreachable
                     else:
                         # Sample not at wp, assume on a rover (but not in have_rock_data - simplified)
                         pass # No extra cost for sampling/moving to sample wp if sample is gone from wp.

                 # Movement to communication waypoint (either from sample wp or from rover holding sample)
                 if has_sample:
                     # Find a rover that has the sample
                     rover_with_data_loc = None
                     for r, wp in have_rock_data:
                         if wp == w and r in rover_locations:
                             rover_with_data_loc = rover_locations[r]
                             break
                     if rover_with_data_loc:
                         # Movement from rover_with_data_loc to comm_wp
                         min_dist_r_to_l = float('inf')
                         for lander_wp in self.lander_visible_wps:
                             if rover_with_data_loc in self.dist and lander_wp in self.dist[rover_with_data_loc]:
                                 min_dist_r_to_l = min(min_dist_r_to_l, self.dist[rover_with_data_loc][lander_wp])
                         if min_dist_r_to_l != float('inf'):
                             h += min_dist_r_to_l
                         else:
                             h = float('inf') # Cannot reach comm wp from rover loc
                             return h # Goal unreachable
                     else:
                         # Should not happen in a solvable state if has_sample is true
                         # Fallback: Assume some rover has it and estimate move from any rover
                         min_dist_r_to_l = float('inf')
                         for r_loc in rover_locations.values():
                              for lander_wp in self.lander_visible_wps:
                                  if r_loc in self.dist and lander_wp in self.dist[r_loc]:
                                      min_dist_r_to_l = min(min_dist_r_to_l, self.dist[r_loc][lander_wp])
                         if min_dist_r_to_l != float('inf'):
                              h += min_dist_r_to_l
                         else:
                              h = float('inf') # Cannot reach comm wp from any rover loc
                              return h # Goal unreachable
                 # If not has_sample and sample was not at wp, we skipped sampling cost.
                 # We still need to communicate. The movement cost is implicitly covered
                 # by the assumption that a rover has it and needs to move.


            elif goal_predicate == "communicated_image_data":
                o, m = goal_args
                h += 1 # Cost for communicate action

                # Check if image is taken
                has_image = any((r, o, m) in have_image_data for r in self.all_rovers)

                if not has_image:
                    h += 1 # Cost for take_image action

                    # Find suitable rover+camera candidates
                    candidate_rovers_cameras = []
                    for r in self.equipped_imaging:
                        for c in self.all_cameras:
                            if (c, r) in self.camera_on_board and m in self.camera_supports.get(c, set()):
                                candidate_rovers_cameras.append((r, c))

                    min_move_cost_sequence = float('inf')

                    for r, c in candidate_rovers_cameras:
                        if r not in rover_locations: continue
                        r_loc = rover_locations[r]

                        is_calibrated = (c, r) in calibrated_cameras

                        current_sequence_cost = 0
                        current_wp = r_loc # Start movement from rover's current location

                        if not is_calibrated:
                            current_sequence_cost += 1 # Calibrate action
                            # Find best cal target t for c, best cal wp w
                            cal_target = self.camera_cal_target.get(c)
                            if not cal_target: continue # Cannot calibrate this camera
                            cal_wps = self.objective_visible_from.get(cal_target, set())
                            if not cal_wps: continue # No waypoint to calibrate from

                            # Find best w in cal_wps minimizing dist(current_wp, w)
                            min_dist_current_to_w = float('inf')
                            best_cal_wp = None
                            for w in cal_wps:
                                if current_wp in self.dist and w in self.dist[current_wp]:
                                    if self.dist[current_wp][w] < min_dist_current_to_w:
                                        min_dist_current_to_w = self.dist[current_wp][w]
                                        best_cal_wp = w

                            if best_cal_wp is None: continue # Cannot reach any cal wp

                            current_sequence_cost += min_dist_current_to_w # Move to cal wp
                            current_wp = best_cal_wp # Update current location after move

                        # Find best p in image_wps minimizing dist(current_wp, p)
                        image_wps = self.objective_visible_from.get(o, set())
                        if not image_wps: continue # No waypoint to image from
                        min_dist_current_to_p = float('inf')
                        best_image_wp = None
                        for p in image_wps:
                             if current_wp in self.dist and p in self.dist[current_wp]:
                                 if self.dist[current_wp][p] < min_dist_current_to_p:
                                     min_dist_current_to_p = self.dist[current_wp][p]
                                     best_image_wp = p

                        if best_image_wp is None: continue # Cannot reach any image wp

                        current_sequence_cost += min_dist_current_to_p # Move to image wp
                        current_wp = best_image_wp # Update current location after move

                        # Find best l in lander_visible_wps minimizing dist(current_wp, l)
                        min_dist_current_to_l = float('inf')
                        for l in self.lander_visible_wps:
                             if current_wp in self.dist and l in self.dist[current_wp]:
                                 min_dist_current_to_l = min(min_dist_current_to_l, self.dist[current_wp][l])

                        if min_dist_current_to_l == float('inf'): continue # Cannot reach lander from image wp

                        current_sequence_cost += min_dist_current_to_l # Move to comm wp

                        min_move_cost_sequence = min(min_move_cost_sequence, current_sequence_cost)

                    if min_move_cost_sequence != float('inf'):
                        h += min_move_cost_sequence
                    else:
                        h = float('inf') # Cannot complete image task
                        return h # Goal unreachable

                else: # Has image
                    # Find a rover that has the image
                    rover_with_image_loc = None
                    for r, o_img, m_img in have_image_data:
                        if o_img == o and m_img == m and r in rover_locations:
                            rover_with_image_loc = rover_locations[r]
                            break
                    if rover_with_image_loc:
                        # Movement from rover_with_image_loc to comm_wp
                        min_dist_r_to_l = float('inf')
                        for lander_wp in self.lander_visible_wps:
                            if rover_with_image_loc in self.dist and lander_wp in self.dist[rover_with_image_loc]:
                                min_dist_r_to_l = min(min_dist_r_to_l, self.dist[rover_with_image_loc][lander_wp])
                        if min_dist_r_to_l != float('inf'):
                            h += min_dist_r_to_l
                        else:
                            h = float('inf') # Cannot reach comm wp from rover loc
                            return h # Goal unreachable
                    else:
                        # Should not happen in a solvable state if have_image is true
                        # Fallback: Assume some rover has it and estimate move from any rover
                        min_dist_r_to_l = float('inf')
                        for r_loc in rover_locations.values():
                             for lander_wp in self.lander_visible_wps:
                                 if r_loc in self.dist and lander_wp in self.dist[r_loc]:
                                     min_dist_r_to_l = min(min_dist_r_to_l, self.dist[r_loc][lander_wp])
                         if min_dist_r_to_l != float('inf'):
                              h += min_dist_r_to_l
                         else:
                              h = float('inf') # Cannot reach comm wp from any rover loc
                              return h # Goal unreachable


        return h
