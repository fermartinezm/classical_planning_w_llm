import collections
import math

# Assume the Operator and Task classes are available from code-file-task
# For self-containment during testing, you might include them, but the problem
# description implies they are provided by the environment.

# Helper functions (as developed in thought process)
def parse_fact(fact_str):
    """Parses a PDDL fact string into a tuple."""
    # Remove surrounding brackets and split by spaces
    parts = fact_str.strip("()").split()
    return tuple(parts)

def get_facts_by_predicate(facts, predicate_name):
    """Filters a set of fact strings by predicate name."""
    # Use startswith for efficiency
    prefix = f'({predicate_name} '
    return {parse_fact(f) for f in facts if f.startswith(prefix)}

def bfs(graph, start_node, all_nodes):
    """Computes shortest path distances from start_node to all reachable nodes."""
    distances = {node: float('inf') for node in all_nodes}
    if start_node in all_nodes:
        distances[start_node] = 0
        queue = collections.deque([start_node])
    else:
        # Start node is not in the set of relevant nodes, cannot start BFS
        return distances # All distances remain inf

    while queue:
        current_node = queue.popleft()

        # Ensure current_node is a key in the graph before iterating neighbors
        if current_node in graph:
            for neighbor in graph[current_node]:
                # Ensure neighbor is one of the relevant nodes
                if neighbor in all_nodes and distances[neighbor] == float('inf'):
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)

    return distances


def compute_all_pairs_shortest_paths(graph, nodes):
    """Computes shortest path distances between all pairs of nodes."""
    all_dist = {}
    # Ensure all nodes that might be start/end points are included
    all_relevant_nodes = set(nodes)
    all_relevant_nodes.update(graph.keys())
    for neighbors in graph.values():
        all_relevant_nodes.update(neighbors)

    for start_node in all_relevant_nodes:
        all_dist[start_node] = bfs(graph, start_node, all_relevant_nodes)
    return all_dist

def min_dist_to_set(dist_map_from_w, set_of_ws):
    """Finds the minimum distance from a waypoint w to any waypoint in a set."""
    min_d = float('inf')
    # dist_map_from_w might be None or empty if w was not in the graph keys or not a relevant node
    if dist_map_from_w:
        for target_w in set_of_ws:
            min_d = min(min_d, dist_map_from_w.get(target_w, float('inf')))
    return min_d


class roversHeuristic:
    """
    Domain-dependent heuristic for the Rovers domain.

    Summary:
    This heuristic estimates the cost to reach a goal state by summing the
    estimated costs for each individual unachieved goal predicate. For each
    goal, it estimates the minimum number of actions and navigation steps
    required to satisfy it, considering rover capabilities, locations,
    required resources (store, calibrated camera), and necessary waypoints
    (sample locations, image locations, calibration locations, communication
    locations). Navigation costs are estimated using precomputed shortest
    path distances on the waypoint graph. The heuristic is non-admissible;
    it ignores negative effects (like camera uncalibration after taking an
    image or store becoming full) and resource contention between goals.

    Assumptions:
    - The 'visible' predicate defines the traversable graph for all rovers.
      Navigation is assumed to be symmetric between visible waypoints.
    - Calibration targets are objectives, and visibility from a target
      objective is given by 'visible_from'.
    - A rover must have an empty store to sample soil/rock.
    - A rover must be at a waypoint visible from the lander to communicate data.
    - If a soil/rock sample predicate (`at_soil_sample`, `at_rock_sample`) is
      false in the current state, but the corresponding communication goal is
      not met, it is assumed that the sample was taken and some rover must
      hold the analysis data (`have_soil_analysis`, `have_rock_analysis`) for
      the goal to be reachable. If the sample is gone and no rover has the data,
      the goal is considered impossible.

    Heuristic Initialization:
    1. Parse static facts from the task to extract:
       - The lander's location.
       - Each rover's equipment capabilities (soil, rock, imaging).
       - The store associated with each rover.
       - Information about cameras on board rovers, including supported modes
         and calibration targets.
       - Waypoints from which each objective is visible (`visible_from`).
       - The waypoint graph based on 'visible' predicates.
    2. Identify all relevant waypoints mentioned in static facts or initial state.
    3. Compute all-pairs shortest path distances between all relevant waypoints
       using Breadth-First Search (BFS) on the waypoint graph.
    4. Identify the set of waypoints from which communication with the lander
       is possible (waypoints visible from the lander's location).
    5. Store all extracted and computed information in instance variables for
       efficient access during heuristic computation.

    Step-By-Step Thinking for Computing Heuristic:
    1. Initialize the total heuristic value `h` to 0. Define a large penalty
       value for goals deemed impossible.
    2. Parse the current state to extract dynamic facts:
       - The current location of each rover.
       - The fullness status of each store.
       - Which rovers have collected soil/rock analysis data for which waypoints.
       - Which rovers have taken images of which objectives in which modes.
       - Which cameras on which rovers are currently calibrated.
       - Which soil/rock samples are still available at which waypoints.
    3. Identify the set of goal predicates from `task.goals` that are not
       present in the current state.
    4. For each unachieved goal predicate:
       a. Initialize the minimum estimated cost for this specific goal to infinity.
       b. Determine the type of goal (soil, rock, or image communication).
       c. **For Soil/Rock Communication Goals (`communicated_soil_data ?w` or `communicated_rock_data ?w`):**
          i. Check if any rover currently possesses the required analysis data (`have_soil_analysis ?r ?w` or `have_rock_analysis ?r ?w`).
          ii. If yes, the cost to obtain the data is 0. The remaining cost is solely for communication. Iterate through rovers holding the data, find their current location, calculate the shortest navigation distance from that location to any communication waypoint, and add 1 for the `communicate` action. The minimum of these values over all rovers holding the data is the estimated cost for this goal.
          iii. If no rover holds the data, check if the sample (`at_soil_sample ?w` or `at_rock_sample ?w`) is still present at waypoint `?w` in the current state. If not, the goal is considered impossible from this state (sample gone, data not collected), and the cost remains infinity.
          iv. If the sample is present, the data must be sampled. Iterate through rovers equipped for the required analysis type. For each equipped rover:
              - Calculate the navigation cost from the rover's current location to waypoint `?w`.
              - Add 1 for the `sample` action.
              - Add 1 if the rover's store is currently full (for the `drop` action needed before sampling).
              - This sum is the estimated cost to obtain the data for this rover.
              - After sampling, the rover is at waypoint `?w`. Calculate the shortest navigation distance from `?w` to any communication waypoint, and add 1 for the `communicate` action. This is the estimated cost to communicate.
              - The total estimated cost for this rover is the sum of the obtain cost and the communicate cost.
              - Update the minimum estimated cost for this goal with the minimum cost found across all equipped rovers.
       d. **For Image Communication Goals (`communicated_image_data ?o ?m`):**
          i. Check if any rover currently possesses the required image (`have_image ?r ?o ?m`).
          ii. If yes, the cost to obtain the image is 0. The remaining cost is solely for communication. Iterate through rovers holding the image, find their current location, calculate the shortest navigation distance from that location to any communication waypoint, and add 1 for the `communicate` action. The minimum of these values over all rovers holding the image is the estimated cost for this goal.
          iii. If no rover holds the image, it must be taken. Iterate through rovers equipped for imaging that have a camera supporting the required mode `?m`. For each such rover/camera pair:
              - Find waypoints `?p` from which objective `?o` is visible (`visible_from ?o ?p`). If no such waypoint exists, this rover/camera cannot take the image via this objective.
              - For each suitable image waypoint `?p`:
                  - Calculate the navigation cost from the rover's current location to `?p`.
                  - Add 1 for the `take_image` action.
                  - If the camera is not currently calibrated: Calculate the estimated calibration cost. Find the calibration target `?t` for the camera. Find waypoints `?w` visible from `?t` (`visible_from ?t ?w`). Calculate the minimum cost to navigate from `?p` to any `?w`, add 1 for the `calibrate` action, and add the minimum cost to navigate back from that `?w` to `?p`. Add this minimum calibration cost. If calibration is impossible (no target or no visible waypoint for target), this path is impossible.
                  - Sum the navigation cost to `?p`, the `take_image` cost, and the calibration cost. This is the estimated cost to obtain the image at `?p`.
                  - After taking the image at `?p`, the rover is at `?p`. Calculate the shortest navigation distance from `?p` to any communication waypoint, and add 1 for the `communicate` action. This is the estimated cost to communicate.
                  - The total estimated cost for this path (rover/camera/image_waypoint) is the sum of the obtain cost and the communicate cost.
                  - Update the minimum estimated cost for this goal with the minimum cost found across all suitable image waypoints `?p`.
              - Update the minimum estimated cost for this goal with the minimum cost found across all suitable rover/camera pairs.
       e. If the minimum estimated cost for the goal is still infinity, it means the goal is likely unreachable in this simplified model. Add the large penalty value to `h`.
       f. Otherwise, add the minimum estimated cost for the goal to `h`.
    5. Return `h`. If the final calculated `h` is infinity (due to summing penalties), return a large integer value instead.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by processing static facts and precomputing
        waypoint distances.
        """
        self.task = task
        static_facts = task.static

        # 1. Parse static facts
        self.lander_location = None
        lander_facts = get_facts_by_predicate(static_facts, 'at_lander')
        if lander_facts:
            # Assuming only one lander
            self.lander_location = list(lander_facts)[0][2]

        self.rover_capabilities = {} # {rover: {'soil': bool, 'rock': bool, 'imaging': bool}}
        self.rover_stores = {}       # {rover: store}
        self.rover_cameras = {}      # {rover: {camera: {'modes': set(mode), 'cal_target': objective}}}
        self.objective_visible_from = {} # {objective: set(waypoint)}
        self.waypoint_graph = collections.defaultdict(set) # {waypoint: set(neighbors)}
        all_waypoints = set()

        # Collect all waypoints mentioned in static facts and initial state
        for fact_str in static_facts:
            fact = parse_fact(fact_str)
            if fact[0] in ('visible', 'visible_from', 'at_lander'):
                 for item in fact[1:]:
                     # Simple check if it looks like a waypoint
                     if isinstance(item, str) and item.startswith('waypoint'):
                         all_waypoints.add(item)
            elif fact[0] in ('equipped_for_soil_analysis', 'equipped_for_rock_analysis', 'equipped_for_imaging'):
                 rover = fact[1]
                 self.rover_capabilities.setdefault(rover, {})
            elif fact[0] == 'store_of':
                 store, rover = fact[1], fact[2]
                 self.rover_stores[rover] = store
                 self.rover_capabilities.setdefault(rover, {}) # Ensure rover exists
            elif fact[0] == 'on_board':
                 camera, rover = fact[1], fact[2]
                 self.rover_cameras.setdefault(rover, {}).setdefault(camera, {}).update({'modes': set()})
                 self.rover_capabilities.setdefault(rover, {}) # Ensure rover exists


        for fact_str in task.initial_state:
             fact = parse_fact(fact_str)
             if fact[0] == 'at':
                 rover, waypoint = fact[1], fact[2]
                 all_waypoints.add(waypoint)
                 self.rover_capabilities.setdefault(rover, {}) # Ensure rover exists


        # Process detailed static facts after collecting all waypoints
        for fact_str in static_facts:
            fact = parse_fact(fact_str)
            if fact[0] == 'equipped_for_soil_analysis':
                rover = fact[1]
                self.rover_capabilities[rover]['soil'] = True
            elif fact[0] == 'equipped_for_rock_analysis':
                rover = fact[1]
                self.rover_capabilities[rover]['rock'] = True
            elif fact[0] == 'equipped_for_imaging':
                rover = fact[1]
                self.rover_capabilities[rover]['imaging'] = True
            elif fact[0] == 'store_of':
                store, rover = fact[1], fact[2]
                self.rover_stores[rover] = store
            elif fact[0] == 'on_board':
                camera, rover = fact[1], fact[2]
                # Handled during initial waypoint collection to ensure rover exists
                pass
            elif fact[0] == 'supports':
                camera, mode = fact[1], fact[2]
                # Find which rover has this camera
                for r, cameras_info in self.rover_cameras.items():
                    if camera in cameras_info:
                        self.rover_cameras[r][camera]['modes'].add(mode)
                        break # Assume camera names are unique or first match is sufficient
            elif fact[0] == 'calibration_target':
                camera, target = fact[1], fact[2]
                # Find which rover has this camera
                for r, cameras_info in self.rover_cameras.items():
                    if camera in cameras_info:
                        self.rover_cameras[r][camera]['cal_target'] = target
                        break # Assume camera names are unique or first match is sufficient
            elif fact[0] == 'visible_from':
                objective, waypoint = fact[1], fact[2]
                self.objective_visible_from.setdefault(objective, set()).add(waypoint)
            elif fact[0] == 'visible':
                w1, w2 = fact[1], fact[2]
                self.waypoint_graph[w1].add(w2)
                self.waypoint_graph[w2].add(w1) # Assuming visible is symmetric for traversal


        # 2. & 3. Build graph and compute distances
        self.dist = compute_all_pairs_shortest_paths(self.waypoint_graph, list(all_waypoints))

        # 4. Identify communication waypoints
        self.comm_waypoints = set()
        if self.lander_location:
            # Waypoints visible *from* the lander's location
            for fact_str in static_facts:
                fact = parse_fact(fact_str)
                if fact[0] == 'visible' and fact[1] == self.lander_location:
                    self.comm_waypoints.add(fact[2])
            # Waypoints *to* which the lander's location is visible
            for fact_str in static_facts:
                fact = parse_fact(fact_str)
                if fact[0] == 'visible' and fact[2] == self.lander_location:
                    self.comm_waypoints.add(fact[1])
            # The lander location itself might be a comm waypoint if a rover can be there
            self.comm_waypoints.add(self.lander_location)


    def __call__(self, state):
        """
        Computes the heuristic value for the given state.
        """
        h = 0
        large_penalty = 1000 # Penalty for seemingly impossible goals

        # 2. Parse current state
        rover_locations = {} # {rover: waypoint}
        store_full = {}      # {store: bool}
        have_soil = set()    # {(rover, waypoint)}
        have_rock = set()    # {(rover, waypoint)}
        have_image = set()   # {(rover, objective, mode)}
        calibrated_cameras = set() # {(camera, rover)}
        soil_samples = set() # {waypoint}
        rock_samples = set() # {waypoint}

        for fact_str in state:
            fact = parse_fact(fact_str)
            if fact[0] == 'at':
                rover, waypoint = fact[1], fact[2]
                rover_locations[rover] = waypoint
            elif fact[0] == 'full':
                store = fact[1]
                store_full[store] = True
            elif fact[0] == 'empty':
                 store = fact[1]
                 store_full[store] = False # Explicitly set False if empty fact exists
            elif fact[0] == 'have_soil_analysis':
                rover, waypoint = fact[1], fact[2]
                have_soil.add((rover, waypoint))
            elif fact[0] == 'have_rock_analysis':
                rover, waypoint = fact[1], fact[2]
                have_rock.add((rover, waypoint))
            elif fact[0] == 'have_image':
                rover, objective, mode = fact[1], fact[2], fact[3]
                have_image.add((rover, objective, mode))
            elif fact[0] == 'calibrated':
                camera, rover = fact[1], fact[2]
                calibrated_cameras.add((camera, rover))
            elif fact[0] == 'at_soil_sample':
                waypoint = fact[1]
                soil_samples.add(waypoint)
            elif fact[0] == 'at_rock_sample':
                waypoint = fact[1]
                rock_samples.add(waypoint)

        # Ensure all known rovers have a location entry (should be in state if valid)
        # Ensure all known stores have a full/empty status entry (default to empty)
        for r, store in self.rover_stores.items():
             store_full.setdefault(store, False)


        # 3. & 4. Iterate through unachieved goals
        for goal_str in self.task.goals:
            if goal_str in state:
                continue # Goal already achieved

            goal = parse_fact(goal_str)
            goal_predicate = goal[0]
            min_goal_cost = float('inf')

            if goal_predicate == 'communicated_soil_data':
                waypoint_w = goal[1]
                rovers_with_data = [r for r in rover_locations if (r, waypoint_w) in have_soil]
                rovers_can_sample = [r for r in rover_locations if self.rover_capabilities.get(r, {}).get('soil', False)]
                can_sample_w_now = waypoint_w in soil_samples

                if rovers_with_data:
                    # Cost is just communication from current location of rover that has it
                    for r in rovers_with_data:
                        current_w = rover_locations.get(r)
                        if current_w is None: continue # Rover location unknown

                        nav_to_comm = min_dist_to_set(self.dist.get(current_w, {}), self.comm_waypoints)
                        if nav_to_comm != float('inf'):
                            min_goal_cost = min(min_goal_cost, nav_to_comm + 1)
                elif can_sample_w_now and rovers_can_sample:
                    # Need to sample and then communicate
                    min_cost_to_sample_and_comm = float('inf')
                    for r in rovers_can_sample:
                        current_w = rover_locations.get(r)
                        if current_w is None: continue # Rover location unknown

                        store = self.rover_stores.get(r) # Get store for this rover
                        if store is None: continue # Rover has no store? (Shouldn't happen based on domain)

                        # Cost to sample at waypoint_w
                        dist_to_sample_w = self.dist.get(current_w, {}).get(waypoint_w, float('inf'))
                        if dist_to_sample_w == float('inf'): continue # Cannot reach sample waypoint

                        cost_to_sample = dist_to_sample_w + (1 if store_full.get(store, False) else 0) + 1 # Nav + Drop (if full) + Sample

                        # After sampling at waypoint_w, need to communicate from waypoint_w
                        nav_to_comm = min_dist_to_set(self.dist.get(waypoint_w, {}), self.comm_waypoints)
                        if nav_to_comm == float('inf'): continue # Cannot reach comm waypoint from sample waypoint

                        cost_to_communicate = nav_to_comm + 1 # Nav + Communicate

                        min_cost_to_sample_and_comm = min(min_cost_to_sample_and_comm, cost_to_sample + cost_to_communicate)

                    min_goal_cost = min(min_goal_cost, min_cost_to_sample_and_comm)
                # else: Goal is impossible (sample gone and no data, or no rover can sample) - min_goal_cost remains inf

            elif goal_predicate == 'communicated_rock_data':
                waypoint_w = goal[1]
                rovers_with_data = [r for r in rover_locations if (r, waypoint_w) in have_rock]
                rovers_can_sample = [r for r in rover_locations if self.rover_capabilities.get(r, {}).get('rock', False)]
                can_sample_w_now = waypoint_w in rock_samples

                if rovers_with_data:
                    # Cost is just communication
                    for r in rovers_with_data:
                        current_w = rover_locations.get(r)
                        if current_w is None: continue # Rover location unknown

                        nav_to_comm = min_dist_to_set(self.dist.get(current_w, {}), self.comm_waypoints)
                        if nav_to_comm != float('inf'):
                            min_goal_cost = min(min_goal_cost, nav_to_comm + 1)
                elif can_sample_w_now and rovers_can_sample:
                    # Need to sample and then communicate
                    min_cost_to_sample_and_comm = float('inf')
                    for r in rovers_can_sample:
                        current_w = rover_locations.get(r)
                        if current_w is None: continue # Rover location unknown

                        store = self.rover_stores.get(r) # Get store for this rover
                        if store is None: continue # Rover has no store?

                        # Cost to sample at waypoint_w
                        dist_to_sample_w = self.dist.get(current_w, {}).get(waypoint_w, float('inf'))
                        if dist_to_sample_w == float('inf'): continue # Cannot reach sample waypoint

                        cost_to_sample = dist_to_sample_w + (1 if store_full.get(store, False) else 0) + 1 # Nav + Drop (if full) + Sample

                        # After sampling at waypoint_w, need to communicate from waypoint_w
                        nav_to_comm = min_dist_to_set(self.dist.get(waypoint_w, {}), self.comm_waypoints)
                        if nav_to_comm == float('inf'): continue # Cannot reach comm waypoint from sample waypoint

                        cost_to_communicate = nav_to_comm + 1 # Nav + Communicate

                        min_cost_to_sample_and_comm = min(min_cost_to_sample_and_comm, cost_to_sample + cost_to_communicate)

                    min_goal_cost = min(min_goal_cost, min_cost_to_sample_and_comm)
                # else: Goal is impossible - min_goal_cost remains inf

            elif goal_predicate == 'communicated_image_data':
                objective_o, mode_m = goal[1], goal[2]
                rovers_with_image = [r for r in rover_locations if (r, objective_o, mode_m) in have_image]
                suitable_rovers = [r for r in rover_locations if self.rover_capabilities.get(r, {}).get('imaging', False)]

                if rovers_with_image:
                    # Cost is just communication
                    for r in rovers_with_image:
                        current_w = rover_locations.get(r)
                        if current_w is None: continue # Rover location unknown

                        nav_to_comm = min_dist_to_set(self.dist.get(current_w, {}), self.comm_waypoints)
                        if nav_to_comm != float('inf'):
                            min_goal_cost = min(min_goal_cost, nav_to_comm + 1)
                elif suitable_rovers:
                    # Need to take image and then communicate
                    min_cost_to_take_and_comm = float('inf')
                    for r in suitable_rovers:
                        current_w = rover_locations.get(r)
                        if current_w is None: continue # Rover location unknown

                        suitable_cameras = [i for i, cam_info in self.rover_cameras.get(r, {}).items() if mode_m in cam_info.get('modes', set())]

                        if not suitable_cameras: continue # Rover has no suitable camera for this mode

                        min_cost_for_this_rover = float('inf')
                        for i in suitable_cameras:
                            cal_target = self.rover_cameras.get(r, {}).get(i, {}).get('cal_target')
                            if cal_target is None: continue # Camera has no calibration target? (Shouldn't happen)

                            calibrated = (i, r) in calibrated_cameras

                            min_cost_for_this_camera = float('inf')
                            image_waypoints = self.objective_visible_from.get(objective_o, set())

                            if not image_waypoints: continue # No waypoint to view objective

                            for p in image_waypoints:
                                dist_to_p = self.dist.get(current_w, {}).get(p, float('inf'))
                                if dist_to_p == float('inf'): continue # Cannot reach image waypoint

                                cost_to_reach_p = dist_to_p
                                cost_to_take = 1 # take_image action

                                cost_to_calibrate = 0
                                if not calibrated:
                                    cal_waypoints = self.objective_visible_from.get(cal_target, set())
                                    if not cal_waypoints:
                                        cost_to_calibrate = float('inf') # Cannot calibrate

                                    if cost_to_calibrate != float('inf'):
                                        # Simplified calibration cost: Nav from p to cal_w + Calibrate + Nav from cal_w back to p
                                        min_cost_nav_cal_nav = float('inf')
                                        for w in cal_waypoints:
                                            dist_p_w = self.dist.get(p, {}).get(w, float('inf'))
                                            dist_w_p = self.dist.get(w, {}).get(p, float('inf'))
                                            if dist_p_w != float('inf') and dist_w_p != float('inf'):
                                                min_cost_nav_cal_nav = min(min_cost_nav_cal_nav, dist_p_w + 1 + dist_w_p)

                                        cost_to_calibrate = min_cost_nav_cal_nav # Could still be infinity

                                if cost_to_calibrate == float('inf'): continue # Cannot calibrate camera for this image

                                total_cost_to_get_image_at_p = cost_to_reach_p + cost_to_calibrate + cost_to_take

                                # After taking image at p, need to communicate from p
                                nav_to_comm = min_dist_to_set(self.dist.get(p, {}), self.comm_waypoints)
                                if nav_to_comm == float('inf'): continue # Cannot reach comm waypoint from image waypoint

                                cost_to_communicate = nav_to_comm + 1 # Nav + Communicate

                                total_cost_for_this_path = total_cost_to_get_image_at_p + cost_to_communicate
                                min_cost_for_this_camera = min(min_cost_for_this_camera, total_cost_for_this_path)

                            min_cost_for_this_rover = min(min_cost_for_this_rover, min_cost_for_this_camera)

                        min_cost_to_take_and_comm = min(min_cost_to_take_and_comm, min_cost_for_this_rover)

                    min_goal_cost = min(min_goal_cost, min_cost_to_take_and_comm)
                # else: Goal is impossible (no rover can take images) - min_goal_cost remains inf


            # Add cost for this goal to total heuristic
            if min_goal_cost == float('inf'):
                 h += large_penalty # Add a large penalty if goal seems impossible
            else:
                 h += min_goal_cost

        # Return 0 if goal is reached, otherwise return the computed sum.
        # The sum will be 0 only if all goals were initially in the state.
        # If any goal was unreachable (added penalty), h will be >= large_penalty.
        # If h is still float('inf') (e.g., if large_penalty was inf), return a large int.
        if h == float('inf'):
             # Return a value larger than any possible finite heuristic
             # Max possible finite h could be roughly (num_goals * (max_nav * 3 + 5))
             # A large constant based on the number of goals seems reasonable
             return large_penalty * len(self.task.goals) + 1
        return h

