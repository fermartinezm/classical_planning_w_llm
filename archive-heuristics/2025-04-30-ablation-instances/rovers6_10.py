# Import necessary modules
import sys
from collections import deque, defaultdict
from fnmatch import fnmatch
# Assuming heuristic_base is available in the execution environment
# from heuristics.heuristic_base import Heuristic

# Dummy Heuristic base for standalone testing if needed
# Remove this block when integrating into the planner framework
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    class Heuristic:
        def __init__(self, task):
            pass
        def __call__(self, node):
            pass
    # print("Warning: heuristics.heuristic_base not found. Using dummy base class.")


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact string or malformed fact
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
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

def bfs(start_node, graph, all_nodes):
    """
    Performs BFS starting from start_node on the given graph.
    Returns a dictionary mapping each reachable node to its distance from start_node.
    Initializes distances for all nodes in all_nodes.
    """
    distances = {node: float('inf') for node in all_nodes} # Initialize for all nodes
    if start_node in distances: # Ensure start_node is one of the known nodes
        distances[start_node] = 0
        queue = deque([start_node])

        while queue:
            current_node = queue.popleft()

            # Check if current_node has neighbors in the graph
            if current_node in graph:
                for neighbor in graph[current_node]:
                    # Ensure neighbor is one of the known nodes before updating distance
                    if neighbor in distances and distances[neighbor] == float('inf'):
                        distances[neighbor] = distances[current_node] + 1
                        queue.append(neighbor)
    return distances

# Helper function to find minimum in an iterable, returning infinity if empty or all infinity
def safe_min(iterable):
    min_val = float('inf')
    found_finite = False
    for val in iterable:
        if val < min_val:
            min_val = val
            found_finite = True
    return min_val if found_finite else float('inf')


class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the minimum number of actions required to achieve
    all goal conditions. It sums the estimated costs for each unachieved goal.
    For data communication goals (soil, rock, image), the cost is estimated
    as the minimum cost to first acquire the data (if not already held) and
    then communicate it from a lander-visible location. Acquisition costs
    include travel, sampling/imaging actions, and potentially dropping samples
    or calibrating cameras. Communication costs include travel to a lander-visible
    waypoint and the communication action. Travel costs are estimated using
    precomputed shortest path distances on the waypoint graph for each rover.

    # Assumptions
    - All action costs are 1.
    - The waypoint graph defined by `can_traverse` is static and provides
      the movement possibilities for rovers.
    - `visible` predicate defines communication links to the lander.
    - `visible_from` predicate defines visibility for imaging and calibration.
    - `at_soil_sample` and `at_rock_sample` facts are only removed by successful
      sampling actions. If a goal requires data from a waypoint where the sample
      is no longer present and no rover holds the data, the goal is unreachable
      from this state (heuristic returns infinity).
    - Similarly, image goals are unreachable if required visibility or calibration
      targets/waypoints do not exist or are inaccessible.
    - The heuristic assumes the problem is solvable if it doesn't return infinity.

    # Heuristic Initialization
    The constructor precomputes static information needed for heuristic calculation:
    - Identifies all objects of relevant types (rovers, waypoints, cameras, etc.).
    - Maps static relationships like `store_of`, `equipped_for_*`, `on_board`,
      `supports`, `calibration_target`, `visible_from`.
    - Determines lander location(s) and identifies all waypoints visible from
      any lander location (`comm_waypoints`).
    - Builds the waypoint graph for each rover based on `can_traverse`.
    - Computes all-pairs shortest paths (APSP) for each rover on its waypoint
      graph using BFS, storing distances in `self.rover_distances`.
    - Stores the initial locations of soil and rock samples from the task's
      initial state to check reachability for sampling goals.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic `h(s)` is computed as follows:

    1.  Parse the current state to extract dynamic facts: rover locations,
        held samples/images, store status, current sample locations, camera
        calibration status, and already communicated data.
    2.  Initialize `total_cost = 0`.
    3.  Iterate through each goal fact specified in `task.goals`.
    4.  If a goal fact `G` is already present in the current state, it contributes 0 to the heuristic.
    5.  If `G` is not in the current state, estimate the minimum cost to achieve it:
        a.  **If `G` is `(communicated_soil_data ?w)`:**
            - Check if any rover `r` currently holds `(have_soil_analysis r ?w)`.
            - If YES: The cost is the minimum travel cost for one such rover `r` from its current location to any lander communication waypoint (`comm_w`), plus 1 for the `communicate_soil_data` action. Minimize this over all rovers holding the data.
            - If NO: The data needs to be acquired and then communicated.
                - Check if `(at_soil_sample ?w)` is in the *current* state. If not, this goal is unreachable from this state; add `infinity` to `total_cost` and stop processing goals (or continue and return infinity at the end).
                - If `(at_soil_sample ?w)` is in the state: The minimum cost is the minimum over all soil-equipped rovers `r` of:
                    - Travel cost from `r`'s current location to `?w`.
                    - Plus 1 if `r`'s store is full (for `drop` action).
                    - Plus 1 for the `sample_soil` action.
                    - Plus minimum travel cost from `?w` to any `comm_w`.
                    - Plus 1 for the `communicate_soil_data` action.
                - Add this minimum acquisition+communication cost to `total_cost`. If no equipped rover can reach `?w` or any `comm_w`, this cost will be infinity.
        b.  **If `G` is `(communicated_rock_data ?w)`:**
            - Similar logic as for soil data, using rock predicates and rock-equipped rovers. Check `(at_rock_sample ?w)` in the current state if no rover holds the data.
        c.  **If `G` is `(communicated_image_data ?o ?m)`:**
            - Check if any rover `r` currently holds `(have_image r ?o ?m)`.
            - If YES: The cost is the minimum travel cost for one such rover `r` from its current location to any `comm_w`, plus 1 for the `communicate_image_data` action. Minimize over rovers holding the image.
            - If NO: The image needs to be acquired and then communicated.
                - Find all (rover `r`, camera `i`) pairs capable of taking the image (`equipped_for_imaging r`, `on_board i r`, `supports i m`). If none, unreachable; add `infinity`.
                - Find all waypoints `p` where `visible_from ?o ?p`. If none, unreachable; add `infinity`.
                - For each capable (r, i) pair:
                    - Find all waypoints `w` where `visible_from ?t ?w` for some `calibration_target i t`. If none, this (r, i) cannot calibrate; skip this pair.
                    - The minimum acquisition path cost for this (r, i) is the minimum over all image waypoints `p` and calibration waypoints `w` of: travel cost from `r`'s current location to `w`, plus travel cost from `w` to `p`. If infinity for all p/w combinations, this (r, i) cannot acquire the image; skip.
                    - If a path exists: Acquisition cost is `min_acq_path_cost` + 1 (`calibrate`) + 1 (`take_image`).
                    - Communication cost is minimum travel cost from *any* suitable image waypoint `p` to any `comm_w` + 1 (`communicate_image_data`). (Simplified assumption: communication can happen from any image waypoint after acquisition).
                    - Total cost for this (r, i) is acquisition cost + communication cost.
                - The minimum acquisition+communication cost for the goal is the minimum over all capable (r, i) pairs. If minimum is infinity, unreachable; add `infinity`.
                - Add this minimum cost to `total_cost`.
    6.  Return `total_cost`. If `total_cost` is infinity, the state is likely unsolvable.

    This heuristic is non-admissible because it sums costs for goals independently, ignoring potential positive interactions (e.g., one trip can collect multiple samples, one trip to lander can communicate multiple data items). However, it captures essential costs like travel, sampling, imaging, calibration, and communication, aiming to guide the search effectively.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static
        initial_state = task.initial_state # Need initial state to check for lost samples

        # --- Precompute Static Information ---

        self.rovers = set()
        self.waypoints = set()
        self.stores = set()
        self.cameras = set()
        self.modes = set()
        self.objectives = set()
        self.landers = set()

        self.store_to_rover = {} # {store: rover}
        self.rover_equipment = defaultdict(set) # {rover: {soil, rock, imaging}}
        self.rover_cameras = defaultdict(set) # {rover: {camera}}
        self.camera_modes = defaultdict(set) # {camera: {mode}}
        self.camera_calib_target = {} # {camera: objective}
        self.objective_visible_from = defaultdict(set) # {objective: {waypoint}}
        self.lander_locations = set() # {waypoint}
        self.waypoint_visibility = defaultdict(set) # {waypoint: {visible_waypoint}}
        self.rover_can_traverse = defaultdict(lambda: defaultdict(set)) # {rover: {wp_from: {wp_to}}}

        # Store initial sample locations
        self.initial_soil_samples = set() # {waypoint}
        self.initial_rock_samples = set() # {waypoint}

        # Parse static facts
        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == 'rover': self.rovers.add(parts[1])
            elif predicate == 'waypoint': self.waypoints.add(parts[1])
            elif predicate == 'store': self.stores.add(parts[1])
            elif predicate == 'camera': self.cameras.add(parts[1])
            elif predicate == 'mode': self.modes.add(parts[1])
            elif predicate == 'objective': self.objectives.add(parts[1])
            elif predicate == 'lander': self.landers.add(parts[1])
            elif predicate == 'store_of':
                 if len(parts) == 3: self.store_to_rover[parts[1]] = parts[2]
            elif predicate == 'equipped_for_soil_analysis':
                 if len(parts) == 2: self.rover_equipment[parts[1]].add('soil')
            elif predicate == 'equipped_for_rock_analysis':
                 if len(parts) == 2: self.rover_equipment[parts[1]].add('rock')
            elif predicate == 'equipped_for_imaging':
                 if len(parts) == 2: self.rover_equipment[parts[1]].add('imaging')
            elif predicate == 'on_board':
                 if len(parts) == 3: self.rover_cameras[parts[2]].add(parts[1])
            elif predicate == 'supports':
                 if len(parts) == 3: self.camera_modes[parts[1]].add(parts[2])
            elif predicate == 'calibration_target':
                 if len(parts) == 3: self.camera_calib_target[parts[1]] = parts[2]
            elif predicate == 'visible_from':
                 if len(parts) == 3: self.objective_visible_from[parts[1]].add(parts[2])
            elif predicate == 'at_lander':
                 if len(parts) == 3: self.lander_locations.add(parts[2])
            elif predicate == 'visible':
                if len(parts) == 3: # Ensure correct number of parts for visible
                    self.waypoint_visibility[parts[1]].add(parts[2])
                    self.waypoint_visibility[parts[2]].add(parts[1]) # Visibility is symmetric
            elif predicate == 'can_traverse':
                 if len(parts) == 4: # Ensure correct number of parts for can_traverse
                     # Assuming can_traverse is symmetric for heuristic pathfinding
                     self.rover_can_traverse[parts[1]][parts[2]].add(parts[3])
                     self.rover_can_traverse[parts[1]][parts[3]].add(parts[2])

        # Identify communication waypoints (visible from any lander location)
        self.comm_waypoints = set()
        for lander_loc in self.lander_locations:
             if lander_loc in self.waypoint_visibility:
                 self.comm_waypoints.update(self.waypoint_visibility[lander_loc])

        # Precompute shortest paths for each rover
        self.rover_distances = {} # {rover: {start_wp: {end_wp: distance}}}
        for rover in self.rovers:
            self.rover_distances[rover] = {}
            waypoint_graph = self.rover_can_traverse[rover]

            # Run BFS from each waypoint to all other waypoints
            for start_wp in self.waypoints: # Iterate over all known waypoints
                 self.rover_distances[rover][start_wp] = bfs(start_wp, waypoint_graph, self.waypoints)

        # Store initial sample locations from the initial state
        for fact in initial_state:
             if match(fact, "at_soil_sample", "*"):
                 self.initial_soil_samples.add(get_parts(fact)[1])
             elif match(fact, "at_rock_sample", "*"):
                 self.initial_rock_samples.add(get_parts(fact)[1])


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # --- Parse Current State ---
        current_rover_loc = {} # {rover: waypoint}
        rover_has_soil = set() # {(rover, waypoint)}
        rover_has_rock = set() # {(rover, waypoint)}
        rover_has_image = set() # {(rover, objective, mode)}
        store_is_full = {} # {store: bool}
        at_soil_sample_locs = set() # {waypoint}
        at_rock_sample_locs = set() # {waypoint}
        calibrated_cameras = set() # {(camera, rover)}
        communicated_soil = set() # {waypoint}
        communicated_rock = set() # {waypoint}
        communicated_image = set() # {(objective, mode)}

        # Initialize store status (assume empty unless full fact exists)
        for store in self.stores:
             store_is_full[store] = False

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]
            if predicate == 'at':
                if len(parts) == 3:
                    obj, loc = parts[1], parts[2]
                    if obj in self.rovers:
                        current_rover_loc[obj] = loc
            elif predicate == 'have_soil_analysis':
                 if len(parts) == 3: rover_has_soil.add((parts[1], parts[2]))
            elif predicate == 'have_rock_analysis':
                 if len(parts) == 3: rover_has_rock.add((parts[1], parts[2]))
            elif predicate == 'have_image':
                 if len(parts) == 4: rover_has_image.add((parts[1], parts[2], parts[3]))
            elif predicate == 'full':
                 if len(parts) == 2: store_is_full[parts[1]] = True
            elif predicate == 'at_soil_sample':
                 if len(parts) == 2: at_soil_sample_locs.add(parts[1])
            elif predicate == 'at_rock_sample':
                 if len(parts) == 2: at_rock_sample_locs.add(parts[1])
            elif predicate == 'calibrated':
                 if len(parts) == 3: calibrated_cameras.add((parts[1], parts[2]))
            elif predicate == 'communicated_soil_data':
                 if len(parts) == 2: communicated_soil.add(parts[1])
            elif predicate == 'communicated_rock_data':
                 if len(parts) == 2: communicated_rock.add(parts[1])
            elif predicate == 'communicated_image_data':
                 if len(parts) == 3: communicated_image.add((parts[1], parts[2]))

        total_cost = 0

        # --- Estimate Cost for Ungratified Goals ---
        for goal in self.goals:
            g_parts = get_parts(goal)
            if not g_parts: continue

            g_predicate = g_parts[0]

            if g_predicate == 'communicated_soil_data':
                if len(g_parts) != 2: continue # Malformed goal
                waypoint = g_parts[1]
                if waypoint in communicated_soil:
                    continue # Goal already achieved

                min_goal_cost = float('inf')
                found_have_soil = False

                # Option 1: Communicate if data is already held by a rover
                min_comm_cost_if_held = float('inf')
                for rover, wp in rover_has_soil:
                    if wp == waypoint:
                        found_have_soil = True
                        # Cost to get this rover to a communication waypoint + communicate
                        current_loc = current_rover_loc.get(rover)
                        if current_loc and self.comm_waypoints:
                             cost_to_comm_wp = safe_min(self.rover_distances[rover][current_loc].get(comm_wp, float('inf')) for comm_wp in self.comm_waypoints)
                             if cost_to_comm_wp != float('inf'):
                                 min_comm_cost_if_held = min(min_comm_cost_if_held, cost_to_comm_wp + 1) # +1 for communicate action

                if found_have_soil:
                    min_goal_cost = min(min_goal_cost, min_comm_cost_if_held)

                # Option 2: Acquire data (if sample exists) and then communicate
                if waypoint in at_soil_sample_locs:
                     min_acq_comm_cost = float('inf')
                     for rover in self.rovers:
                         if 'soil' in self.rover_equipment.get(rover, set()):
                             # Cost to acquire
                             current_loc = current_rover_loc.get(rover)
                             if current_loc:
                                 cost_to_sample_wp = self.rover_distances[rover][current_loc].get(waypoint, float('inf'))
                                 if cost_to_sample_wp != float('inf'):
                                     acq_cost = cost_to_sample_wp
                                     store = next((s for s, r in self.store_to_rover.items() if r == rover), None)
                                     if store and store_is_full.get(store, False):
                                         acq_cost += 1 # drop action
                                     acq_cost += 1 # sample_soil action

                                     # Cost to communicate from sample waypoint (or after moving)
                                     if self.comm_waypoints:
                                         cost_from_sample_to_comm_wp = safe_min(self.rover_distances[rover][waypoint].get(comm_wp, float('inf')) for comm_wp in self.comm_waypoints)
                                         if cost_from_sample_to_comm_wp != float('inf'):
                                             comm_cost = cost_from_sample_to_comm_wp + 1 # communicate action
                                             min_acq_comm_cost = min(min_acq_comm_cost, acq_cost + comm_cost)

                     min_goal_cost = min(min_goal_cost, min_acq_comm_cost)

                # If goal is still not achieved and sample is gone, it's unreachable
                if min_goal_cost == float('inf') and waypoint not in at_soil_sample_locs and not found_have_soil:
                     # Check if sample was initially present. If not, something is wrong with goal/domain.
                     # If it was initially present but is now gone and no rover has it, it's unreachable.
                     if waypoint in self.initial_soil_samples:
                          total_cost = float('inf') # Unreachable goal
                          break # Stop processing goals
                     # else: # Goal requires data from a waypoint that never had a sample? Treat as unreachable.
                     #      total_cost = float('inf')
                     #      break # Stop processing goals
                     # Removed the else branch here. If it wasn't initially present, it's not a valid goal anyway.
                     # If it was initially present and is now gone and no rover has it, it's unreachable.
                     # The current logic correctly sets min_goal_cost to inf if acquisition/communication is impossible.
                     # If after checking both options, min_goal_cost is still inf, it's unreachable.

                if min_goal_cost == float('inf'):
                     total_cost = float('inf')
                     break # Stop processing goals
                else:
                    total_cost += min_goal_cost


            elif g_predicate == 'communicated_rock_data':
                if len(g_parts) != 2: continue # Malformed goal
                waypoint = g_parts[1]
                if waypoint in communicated_rock:
                    continue # Goal already achieved

                min_goal_cost = float('inf')
                found_have_rock = False

                # Option 1: Communicate if data is already held by a rover
                min_comm_cost_if_held = float('inf')
                for rover, wp in rover_has_rock:
                    if wp == waypoint:
                        found_have_rock = True
                        # Cost to get this rover to a communication waypoint + communicate
                        current_loc = current_rover_loc.get(rover)
                        if current_loc and self.comm_waypoints:
                             cost_to_comm_wp = safe_min(self.rover_distances[rover][current_loc].get(comm_wp, float('inf')) for comm_wp in self.comm_waypoints)
                             if cost_to_comm_wp != float('inf'):
                                 min_comm_cost_if_held = min(min_comm_cost_if_held, cost_to_comm_wp + 1) # +1 for communicate action

                if found_have_rock:
                    min_goal_cost = min(min_goal_cost, min_comm_cost_if_held)

                # Option 2: Acquire data (if sample exists) and then communicate
                if waypoint in at_rock_sample_locs:
                     min_acq_comm_cost = float('inf')
                     for rover in self.rovers:
                         if 'rock' in self.rover_equipment.get(rover, set()):
                             # Cost to acquire
                             current_loc = current_rover_loc.get(rover)
                             if current_loc:
                                 cost_to_sample_wp = self.rover_distances[rover][current_loc].get(waypoint, float('inf'))
                                 if cost_to_sample_wp != float('inf'):
                                     acq_cost = cost_to_sample_wp
                                     store = next((s for s, r in self.store_to_rover.items() if r == rover), None)
                                     if store and store_is_full.get(store, False):
                                         acq_cost += 1 # drop action
                                     acq_cost += 1 # sample_rock action

                                     # Cost to communicate from sample waypoint (or after moving)
                                     if self.comm_waypoints:
                                         cost_from_sample_to_comm_wp = safe_min(self.rover_distances[rover][waypoint].get(comm_wp, float('inf')) for comm_wp in self.comm_waypoints)
                                         if cost_from_sample_to_comm_wp != float('inf'):
                                             comm_cost = cost_from_sample_to_comm_wp + 1 # communicate action
                                             min_acq_comm_cost = min(min_acq_comm_cost, acq_cost + comm_cost)

                     min_goal_cost = min(min_goal_cost, min_acq_comm_cost)

                # If goal is still not achieved and sample is gone, it's unreachable
                if min_goal_cost == float('inf') and waypoint not in at_rock_sample_locs and not found_have_rock:
                     if waypoint in self.initial_rock_samples:
                          total_cost = float('inf') # Unreachable goal
                          break # Stop processing goals
                     # else: # Goal requires data from a waypoint that never had a sample? Treat as unreachable.
                     #      total_cost = float('inf')
                     #      break # Stop processing goals

                if min_goal_cost == float('inf'):
                     total_cost = float('inf')
                     break # Stop processing goals
                else:
                    total_cost += min_goal_cost


            elif g_predicate == 'communicated_image_data':
                if len(g_parts) != 3: continue # Malformed goal
                objective, mode = g_parts[1], g_parts[2]
                if (objective, mode) in communicated_image:
                    continue # Goal already achieved

                min_goal_cost = float('inf')
                found_have_image = False

                # Option 1: Communicate if data is already held by a rover
                min_comm_cost_if_held = float('inf')
                for rover, obj, m in rover_has_image:
                    if obj == objective and m == mode:
                        found_have_image = True
                        # Cost to get this rover to a communication waypoint + communicate
                        current_loc = current_rover_loc.get(rover)
                        if current_loc and self.comm_waypoints:
                             cost_to_comm_wp = safe_min(self.rover_distances[rover][current_loc].get(comm_wp, float('inf')) for comm_wp in self.comm_waypoints)
                             if cost_to_comm_wp != float('inf'):
                                 min_comm_cost_if_held = min(min_comm_cost_if_held, cost_to_comm_wp + 1) # +1 for communicate action

                if found_have_image:
                    min_goal_cost = min(min_goal_cost, min_comm_cost_if_held)

                # Option 2: Acquire image and then communicate
                suitable_image_wps = self.objective_visible_from.get(objective, set())
                if not suitable_image_wps:
                     # Cannot see the objective from anywhere
                     min_acq_comm_cost = float('inf') # Unreachable
                else:
                    min_acq_comm_cost = float('inf')
                    # Find suitable (rover, camera) pairs
                    for rover in self.rovers:
                        if 'imaging' in self.rover_equipment.get(rover, set()):
                            for camera in self.rover_cameras.get(rover, set()):
                                if mode in self.camera_modes.get(camera, set()):
                                    # Found a capable (rover, camera) pair

                                    # Find calibration waypoints for this camera
                                    calib_target = self.camera_calib_target.get(camera)
                                    suitable_calib_wps = self.objective_visible_from.get(calib_target, set()) if calib_target else set()

                                    if not suitable_calib_wps:
                                         # Cannot calibrate this camera
                                         continue # Skip this (rover, camera) pair

                                    # Cost to acquire image with this rover/camera
                                    current_loc = current_rover_loc.get(rover)
                                    if current_loc:
                                        min_path_cost = float('inf')
                                        # Find min cost path: current -> calib_wp -> image_wp
                                        for calib_wp in suitable_calib_wps:
                                            dist_curr_to_calib = self.rover_distances[rover][current_loc].get(calib_wp, float('inf'))
                                            if dist_curr_to_calib != float('inf'):
                                                for image_wp in suitable_image_wps:
                                                    dist_calib_to_image = self.rover_distances[rover][calib_wp].get(image_wp, float('inf'))
                                                    if dist_calib_to_image != float('inf'):
                                                        min_path_cost = min(min_path_cost, dist_curr_to_calib + dist_calib_to_image)

                                        if min_path_cost != float('inf'):
                                            acq_cost = min_path_cost + 1 # calibrate
                                            acq_cost += 1 # take_image

                                            # Cost to communicate from image waypoint (or after moving)
                                            # Assume communication happens from *any* suitable image waypoint
                                            min_dist_image_to_comm = float('inf')
                                            if self.comm_waypoints:
                                                min_dist_image_to_comm = safe_min(
                                                    self.rover_distances[rover][img_wp].get(comm_wp, float('inf'))
                                                    for img_wp in suitable_image_wps
                                                    for comm_wp in self.comm_waypoints
                                                )

                                            if min_dist_image_to_comm != float('inf'):
                                                 comm_cost = min_dist_image_to_comm + 1 # communicate
                                                 min_acq_comm_cost = min(min_acq_comm_cost, acq_cost + comm_cost)


                min_goal_cost = min(min_goal_cost, min_acq_comm_cost)

                # If goal is still not achieved and acquisition is impossible, it's unreachable
                if min_goal_cost == float('inf') and not found_have_image:
                     total_cost = float('inf') # Unreachable goal
                     break # Stop processing goals
                else:
                    total_cost += min_goal_cost

            # If at any point total_cost becomes infinity, stop and return infinity
            if total_cost == float('inf'):
                 break

        # Heuristic is 0 only if all goals are achieved.
        # If total_cost is 0 here, it means no ungratified goals were found.
        # If total_cost is infinity, it means at least one ungratified goal is unreachable.
        # Otherwise, it's the sum of estimated costs for ungratified but reachable goals.
        return total_cost
