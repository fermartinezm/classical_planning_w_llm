from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import collections
import sys

# Define infinity for unreachable states
INF = float('inf')

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact strings or malformed facts gracefully
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

def bfs_shortest_path(graph, start, targets):
    """
    Find the shortest path distance from start to any node in targets
    using BFS on the given graph.
    graph: adjacency list {node: [neighbor1, neighbor2, ...]}
    start: start node
    targets: set of target nodes
    Returns: shortest distance, or INF if no path exists or start/targets are invalid
    """
    if start is None or not targets:
        return INF
    if start in targets:
        return 0

    queue = collections.deque([(start, 0)])
    visited = {start}

    while queue:
        current_node, dist = queue.popleft()

        if current_node in graph: # Ensure the node exists in the graph keys
            for neighbor in graph[current_node]:
                if neighbor in targets:
                    return dist + 1
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))

    return INF # No path found

def bfs_all_pairs_shortest_path(graph, nodes):
    """
    Compute shortest path distances between all pairs of nodes in the graph.
    graph: adjacency list {node: [neighbor1, neighbor2, ...]}
    nodes: list of all possible nodes
    Returns: dictionary {(start, end): distance}
    """
    distances = {}
    for start_node in nodes:
        queue = collections.deque([(start_node, 0)])
        visited = {start_node}
        distances[(start_node, start_node)] = 0

        while queue:
            current_node, dist = queue.popleft()

            if current_node in graph: # Ensure the node exists in the graph keys
                for neighbor in graph[current_node]:
                    if (start_node, neighbor) not in distances:
                         distances[(start_node, neighbor)] = dist + 1
                         visited.add(neighbor)
                         queue.append((neighbor, dist + 1))
    return distances


class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve all
    unmet goal conditions. It calculates the minimum cost for each unachieved
    communication goal independently and sums these minimum costs. The cost
    for a single communication goal is estimated by finding the cheapest way
    to either communicate existing data or acquire the data (sample/image)
    and then communicate it, considering navigation, sampling/imaging,
    calibration, and dropping actions. Navigation costs are precomputed
    using BFS on the traversal graph for each rover.

    # Assumptions
    - Goal conditions are always of the form `(communicated_soil_data ?w)`,
      `(communicated_rock_data ?w)`, or `(communicated_image_data ?o ?m)`.
    - Samples (`at_soil_sample`, `at_rock_sample`) are consumed upon sampling.
      If a communication goal for a sample waypoint is not met, and no rover
      currently has the corresponding `have_*_analysis` fact, and the sample
      fact (`at_*_sample`) is no longer in the state, the goal is considered
      unreachable (cost INF).
    - Calibration is needed before taking an image if the camera is not
      calibrated. The heuristic adds a fixed cost of 1 for calibration,
      ignoring the navigation required to a calibration target.
    - The heuristic sums the minimum costs for individual goals, ignoring
      potential synergies (e.g., one trip achieving multiple goals) or
      conflicts (e.g., resource contention like stores or cameras).
    - All rovers, waypoints, cameras, etc., mentioned in the initial state
      or static facts are relevant.

    # Heuristic Initialization
    The constructor performs the following steps:
    1. Collects all objects (rovers, waypoints, cameras, etc.) present in
       the initial state and static facts.
    2. Parses static facts to build data structures:
       - Lander location.
       - Rover capabilities (soil, rock, imaging).
       - Mapping of stores to rovers.
       - Camera information (which rover it's on, supported modes, calibration target).
       - Objective visibility (waypoints from which an objective is visible).
       - Initial locations of soil and rock samples.
       - Waypoints visible from the lander location (communication points).
       - Traversal graphs (adjacency lists) for each rover based on `can_traverse`.
    3. Precomputes all-pairs shortest path distances for each rover's
       traversal graph using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Parse the dynamic facts from the state to get:
       - Current location of each rover.
       - Status of each store (full/empty).
       - Calibration status of each camera.
       - Data collected by rovers (`have_soil_analysis`, `have_rock_analysis`, `have_image`).
       - Data already communicated (`communicated_soil_data`, etc.).
    2. Initialize the total heuristic cost to 0.
    3. Identify all goal facts that are not present in the current state.
    4. For each unachieved goal fact:
       a. Initialize the minimum cost for this specific goal to infinity.
       b. If the goal is `(communicated_soil_data ?w)`:
          - Check if any rover currently holds `(have_soil_analysis ?r ?w)`.
          - If yes, calculate the cost to navigate that rover to any communication waypoint and communicate (Navigation + 1). Minimize over such rovers.
          - If no, check if `(at_soil_sample ?w)` is still in the state.
          - If yes, calculate the cost for any equipped rover to navigate to `?w`, sample (potentially dropping first), navigate to any communication waypoint, and communicate (Navigation1 + Drop(if needed) + Sample + Navigation2 + Communicate). Minimize over equipped rovers.
          - The minimum of the costs from these options (communicate existing data vs. sample and communicate) is the cost for this goal.
       c. If the goal is `(communicated_rock_data ?w)`: Follow similar logic as soil data.
       d. If the goal is `(communicated_image_data ?o ?m)`:
          - Check if any rover currently holds `(have_image ?r ?o ?m)`.
          - If yes, calculate the cost to navigate that rover to any communication waypoint and communicate (Navigation + 1). Minimize over such rovers.
          - If no, find equipped rovers with cameras supporting mode `?m`. Find waypoints visible from objective `?o`.
          - Calculate the cost for any such rover/camera pair to navigate to a visible waypoint `?p`, calibrate (if needed), take the image, navigate from `?p` to any communication waypoint, and communicate (Navigation1 + Calibrate(if needed) + Image + Navigation2 + Communicate). Minimize over suitable rover/camera pairs and visible waypoints `?p`.
          - The minimum of the costs from these options is the cost for this goal.
       e. Add the minimum cost for this goal to the total heuristic cost. If the minimum cost was infinity, the total cost becomes infinity.
    5. Return the total heuristic cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        initial_state = task.initial_state
        static_facts = task.static

        # 1. Collect all objects by type from initial state and static facts
        self.rovers = set()
        self.waypoints = set()
        self.stores = set()
        self.cameras = set()
        self.modes = set()
        self.landers = set()
        self.objectives = set()

        all_facts = initial_state | static_facts

        # Infer types based on common predicates and argument positions in rovers domain
        for fact_str in all_facts:
            parts = get_parts(fact_str)
            if not parts: continue
            predicate = parts[0]

            if predicate in ['at', 'at_lander'] and len(parts) == 3:
                 # (at ?x ?y) or (at_lander ?x ?y)
                 obj_name = parts[1]
                 loc_name = parts[2]
                 # Distinguish rover/lander based on other predicates or naming conventions
                 # Simple heuristic: if it's not a known lander, assume it's a rover if it appears in rover predicates
                 if match(fact_str, 'at_lander', obj_name, '*'):
                      self.landers.add(obj_name)
                 elif any(match(f, 'equipped_for_*', obj_name) for f in all_facts) or \
                      any(match(f, 'can_traverse', obj_name, '*', '*') for f in all_facts) or \
                      any(match(f, 'store_of', '*', obj_name) for f in all_facts) or \
                      any(match(f, 'on_board', '*', obj_name) for f in all_facts) or \
                      any(match(f, 'have_*_analysis', obj_name, '*') for f in all_facts) or \
                      any(match(f, 'have_image', obj_name, '*', '*') for f in all_facts):
                      self.rovers.add(obj_name)
                 self.waypoints.add(loc_name)
            elif predicate == 'can_traverse' and len(parts) == 4: # (can_traverse ?r ?x ?y)
                 self.rovers.add(parts[1])
                 self.waypoints.add(parts[2])
                 self.waypoints.add(parts[3])
            elif predicate in ['equipped_for_soil_analysis', 'equipped_for_rock_analysis', 'equipped_for_imaging'] and len(parts) == 2: self.rovers.add(parts[1])
            elif predicate in ['empty', 'full'] and len(parts) == 2: self.stores.add(parts[1])
            elif predicate in ['have_rock_analysis', 'have_soil_analysis'] and len(parts) == 3: self.rovers.add(parts[1]); self.waypoints.add(parts[2])
            elif predicate == 'calibrated' and len(parts) == 3: self.cameras.add(parts[1]); self.rovers.add(parts[2])
            elif predicate == 'supports' and len(parts) == 3: self.cameras.add(parts[1]); self.modes.add(parts[2])
            elif predicate == 'visible' and len(parts) == 3: self.waypoints.add(parts[1]); self.waypoints.add(parts[2])
            elif predicate == 'have_image' and len(parts) == 4: self.rovers.add(parts[1]); self.objectives.add(parts[2]); self.modes.add(parts[3])
            elif predicate in ['communicated_soil_data', 'communicated_rock_data'] and len(parts) == 2: self.waypoints.add(parts[1])
            elif predicate == 'communicated_image_data' and len(parts) == 3: self.objectives.add(parts[1]); self.modes.add(parts[2])
            elif predicate in ['at_soil_sample', 'at_rock_sample'] and len(parts) == 2: self.waypoints.add(parts[1])
            elif predicate == 'visible_from' and len(parts) == 3: self.objectives.add(parts[1]); self.waypoints.add(parts[2])
            elif predicate == 'store_of' and len(parts) == 3: self.stores.add(parts[1]); self.rovers.add(parts[2])
            elif predicate == 'calibration_target' and len(parts) == 3: self.cameras.add(parts[1]); self.objectives.add(parts[2])
            elif predicate == 'on_board' and len(parts) == 3: self.cameras.add(parts[1]); self.rovers.add(parts[2])

        # Ensure all objects are captured even if only mentioned in goals
        for goal_fact_str in self.goals:
             parts = get_parts(goal_fact_str)
             if not parts: continue
             predicate = parts[0]
             if predicate in ['communicated_soil_data', 'communicated_rock_data'] and len(parts) == 2:
                 self.waypoints.add(parts[1])
             elif predicate == 'communicated_image_data' and len(parts) == 3:
                 self.objectives.add(parts[1])
                 self.modes.add(parts[2])


        # 2. Parse static facts into useful structures
        self.lander_location = None
        self.rover_capabilities = collections.defaultdict(lambda: {'soil': False, 'rock': False, 'imaging': False})
        self.store_to_rover = {}
        self.camera_info = collections.defaultdict(lambda: {'rover': None, 'modes': set(), 'calibration_target': None})
        self.objective_visibility = collections.defaultdict(set)
        self.initial_soil_sample_locations = set() # Samples initially present
        self.initial_rock_sample_locations = set() # Samples initially present
        self.communication_waypoint_candidates = set() # Waypoints visible from lander location
        self.rover_traversal_graphs = collections.defaultdict(lambda: collections.defaultdict(list))

        for fact_str in static_facts:
            parts = get_parts(fact_str)
            if not parts: continue
            predicate = parts[0]
            if predicate == 'at_lander':
                self.lander_location = parts[2]
            elif predicate == 'equipped_for_soil_analysis':
                self.rover_capabilities[parts[1]]['soil'] = True
            elif predicate == 'equipped_for_rock_analysis':
                self.rover_capabilities[parts[1]]['rock'] = True
            elif predicate == 'equipped_for_imaging':
                self.rover_capabilities[parts[1]]['imaging'] = True
            elif predicate == 'store_of':
                self.store_to_rover[parts[1]] = parts[2]
            elif predicate == 'on_board':
                self.camera_info[parts[1]]['rover'] = parts[2]
            elif predicate == 'supports':
                self.camera_info[parts[1]]['modes'].add(parts[2])
            elif predicate == 'calibration_target':
                self.camera_info[parts[1]]['calibration_target'] = parts[2]
            elif predicate == 'visible_from':
                self.objective_visibility[parts[1]].add(parts[2])
            elif predicate == 'can_traverse':
                rover, wp1, wp2 = parts[1:]
                self.rover_traversal_graphs[rover][wp1].append(wp2)
                # Assuming can_traverse is symmetric if visible is symmetric (based on examples)
                self.rover_traversal_graphs[rover][wp2].append(wp1)
            elif predicate == 'visible':
                 wp1, wp2 = parts[1:]
                 # Check if this waypoint is visible from the lander location
                 if self.lander_location: # Ensure lander location is known
                     if wp2 == self.lander_location:
                         self.communication_waypoint_candidates.add(wp1)
                     if wp1 == self.lander_location:
                         self.communication_waypoint_candidates.add(wp2)

        # Collect initial sample locations from initial state
        for fact_str in initial_state:
             parts = get_parts(fact_str)
             if not parts: continue
             if parts[0] == 'at_soil_sample':
                 self.initial_soil_sample_locations.add(parts[1])
             elif parts[0] == 'at_rock_sample':
                 self.initial_rock_sample_locations.add(parts[1])

        # 3. Precompute navigation distances for each rover
        self.navigation_distances = {}
        all_waypoints_list = list(self.waypoints) # Use the collected set of all waypoints

        for rover in self.rovers:
            graph = self.rover_traversal_graphs[rover]
            self.navigation_distances[rover] = bfs_all_pairs_shortest_path(graph, all_waypoints_list)


    def get_navigation_cost(self, rover, start_wp, end_wp):
         """Lookup precomputed navigation cost."""
         if start_wp is None or end_wp is None: # Handle cases where location is unknown
             return INF
         if start_wp == end_wp:
             return 0
         # Use .get() with default INF for robustness
         return self.navigation_distances.get(rover, {}).get((start_wp, end_wp), INF)

    def get_navigation_cost_to_set(self, rover, start_wp, target_waypoints):
         """Find minimum navigation cost from start_wp to any waypoint in the target_waypoints set."""
         if start_wp is None or not target_waypoints:
             return INF
         min_dist = INF
         for target_wp in target_waypoints:
             min_dist = min(min_dist, self.get_navigation_cost(rover, start_wp, target_wp))
         return min_dist

    def get_store_for_rover(self, rover):
        """Find the store associated with a rover."""
        # This mapping is static, built in __init__
        return next((store for store, r in self.store_to_rover.items() if r == rover), None)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Parse dynamic state facts
        rover_locations = {} # rover -> waypoint
        store_status = {} # store -> 'empty' or 'full'
        camera_calibration = {} # camera -> 'calibrated' or 'uncalibrated'
        soil_data_collected = set() # (rover, waypoint)
        rock_data_collected = set() # (rover, waypoint)
        image_data_collected = set() # (rover, objective, mode)
        communicated_soil_data = set() # waypoint
        communicated_rock_data = set() # waypoint
        communicated_image_data = set() # (objective, mode)
        current_soil_sample_locations = set() # waypoints with samples currently
        current_rock_sample_locations = set() # waypoints with samples currently


        for fact_str in state:
            parts = get_parts(fact_str)
            if not parts: continue
            predicate = parts[0]
            if predicate == 'at' and len(parts) == 3:
                 obj_name = parts[1]
                 if obj_name in self.rovers: # Only track rover locations
                     rover_locations[obj_name] = parts[2]
            elif predicate == 'empty':
                 store_status[parts[1]] = 'empty'
            elif predicate == 'full':
                 store_status[parts[1]] = 'full'
            elif predicate == 'calibrated':
                 camera_calibration[parts[1]] = 'calibrated'
            elif predicate == 'have_soil_analysis':
                 soil_data_collected.add((parts[1], parts[2]))
            elif predicate == 'have_rock_analysis':
                 rock_data_collected.add((parts[1], parts[2]))
            elif predicate == 'have_image':
                 image_data_collected.add((parts[1], parts[2], parts[3]))
            elif predicate == 'communicated_soil_data':
                 communicated_soil_data.add(parts[1])
            elif predicate == 'communicated_rock_data':
                 communicated_rock_data.add(parts[1])
            elif predicate == 'communicated_image_data':
                 communicated_image_data.add((parts[1], parts[2]))
            elif predicate == 'at_soil_sample':
                 current_soil_sample_locations.add(parts[1])
            elif predicate == 'at_rock_sample':
                 current_rock_sample_locations.add(parts[1])


        # Assume cameras not explicitly calibrated are uncalibrated
        for cam in self.cameras:
             if cam not in camera_calibration:
                 camera_calibration[cam] = 'uncalibrated'

        # Assume stores not explicitly full are empty (initial state default)
        # In PDDL, predicates are typically explicit. If (full s) is not there, (empty s) must be, or vice versa.
        # We rely on the state containing either (full s) or (empty s) for each store.


        total_cost = 0

        # Identify unachieved goals
        # Filter goals to only include communication goals
        communication_goals = {g for g in self.goals if get_parts(g)[0] in ['communicated_soil_data', 'communicated_rock_data', 'communicated_image_data']}
        unachieved_communication_goals = communication_goals - state

        if not unachieved_communication_goals and not (self.goals - communication_goals - state):
             # All communication goals are met, and any other potential goals are also met.
             # In this domain, goals are only communication goals based on the examples.
             # If the goal set contains non-communication goals, this heuristic might be incomplete.
             # Assuming goals are *only* communication goals based on the problem description and examples.
             return 0 # Goal state reached

        for goal_fact_str in unachieved_communication_goals:
            goal_parts = get_parts(goal_fact_str)
            predicate = goal_parts[0]

            min_cost_for_goal = INF

            if predicate == 'communicated_soil_data':
                waypoint = goal_parts[1]

                # --- Option 1: Communicate existing data ---
                rovers_with_data = [r for r, sampled_w in soil_data_collected if sampled_w == waypoint]
                for rover in rovers_with_data:
                    nav_cost = self.get_navigation_cost_to_set(rover, rover_locations.get(rover), self.communication_waypoint_candidates)
                    cost = nav_cost + 1 # communicate_soil_data
                    min_cost_for_goal = min(min_cost_for_goal, cost)

                # --- Option 2: Sample and Communicate ---
                # Only possible if the sample still exists at the waypoint
                if waypoint in current_soil_sample_locations:
                    equipped_rovers = [r for r, caps in self.rover_capabilities.items() if caps['soil']]
                    for rover in equipped_rovers:
                        nav_to_sample = self.get_navigation_cost(rover, rover_locations.get(rover), waypoint)
                        if nav_to_sample == INF: continue # Cannot reach sample location

                        sample_cost = nav_to_sample
                        rover_store = self.get_store_for_rover(rover)
                        # Check if store is full. If store_of fact is missing or store is not in state, assume not full.
                        if rover_store and store_status.get(rover_store) == 'full':
                            sample_cost += 1 # drop action
                        sample_cost += 1 # sample_soil action

                        nav_to_comm = self.get_navigation_cost_to_set(rover, waypoint, self.communication_waypoint_candidates)
                        if nav_to_comm == INF: continue # Cannot reach comm location from sample location

                        comm_cost = nav_to_comm + 1 # communicate_soil_data

                        cost = sample_cost + comm_cost
                        min_cost_for_goal = min(min_cost_for_goal, cost)

            elif predicate == 'communicated_rock_data':
                waypoint = goal_parts[1]

                # --- Option 1: Communicate existing data ---
                rovers_with_data = [r for r, sampled_w in rock_data_collected if sampled_w == waypoint]
                for rover in rovers_with_data:
                    nav_cost = self.get_navigation_cost_to_set(rover, rover_locations.get(rover), self.communication_waypoint_candidates)
                    cost = nav_cost + 1 # communicate_rock_data
                    min_cost_for_goal = min(min_cost_for_goal, cost)

                # --- Option 2: Sample and Communicate ---
                if waypoint in current_rock_sample_locations:
                    equipped_rovers = [r for r, caps in self.rover_capabilities.items() if caps['rock']]
                    for rover in equipped_rovers:
                        nav_to_sample = self.get_navigation_cost(rover, rover_locations.get(rover), waypoint)
                        if nav_to_sample == INF: continue

                        sample_cost = nav_to_sample
                        rover_store = self.get_store_for_rover(rover)
                        if rover_store and store_status.get(rover_store) == 'full':
                            sample_cost += 1 # drop action
                        sample_cost += 1 # sample_rock action

                        nav_to_comm = self.get_navigation_cost_to_set(rover, waypoint, self.communication_waypoint_candidates)
                        if nav_to_comm == INF: continue

                        comm_cost = nav_to_comm + 1 # communicate_rock_data

                        cost = sample_cost + comm_cost
                        min_cost_for_goal = min(min_cost_for_goal, cost)

            elif predicate == 'communicated_image_data':
                objective = goal_parts[1]
                mode = goal_parts[2]

                # --- Option 1: Communicate existing data ---
                rovers_with_data = [r for r, o, m in image_data_collected if o == objective and m == mode]
                for rover in rovers_with_data:
                    nav_cost = self.get_navigation_cost_to_set(rover, rover_locations.get(rover), self.communication_waypoint_candidates)
                    cost = nav_cost + 1 # communicate_image_data
                    min_cost_for_goal = min(min_cost_for_goal, cost)

                # --- Option 2: Take Image and Communicate ---
                # Find equipped rovers with suitable cameras
                suitable_rover_camera_pairs = []
                for rover in self.rovers:
                    if self.rover_capabilities[rover]['imaging']:
                        for camera, info in self.camera_info.items():
                            if info['rover'] == rover and mode in info['modes']:
                                suitable_rover_camera_pairs.append((rover, camera))

                # Find waypoints visible from the objective
                imaging_waypoint_candidates = self.objective_visibility.get(objective, set())

                if suitable_rover_camera_pairs and imaging_waypoint_candidates:
                    for rover, camera in suitable_rover_camera_pairs:
                        # Need to navigate to imaging location, calibrate (if needed), take image, navigate to comm location, communicate

                        # Find the best imaging waypoint for this rover
                        best_imaging_wp = None
                        min_nav_to_imaging = INF
                        for imaging_wp in imaging_waypoint_candidates:
                            nav_cost = self.get_navigation_cost(rover, rover_locations.get(rover), imaging_wp)
                            if nav_cost < min_nav_to_imaging:
                                min_nav_to_imaging = nav_cost
                                best_imaging_wp = imaging_wp

                        if best_imaging_wp is None or min_nav_to_imaging == INF: continue # Cannot reach any imaging location

                        image_cost = min_nav_to_imaging
                        # Check if camera is calibrated. If camera_info is missing or not in state, assume uncalibrated.
                        if camera_calibration.get(camera) != 'calibrated':
                            # Simplified calibration cost
                            image_cost += 1 # calibrate action
                            # Note: This ignores navigation to calibration target and back.

                        image_cost += 1 # take_image action

                        nav_to_comm = self.get_navigation_cost_to_set(rover, best_imaging_wp, self.communication_waypoint_candidates)
                        if nav_to_comm == INF: continue

                        comm_cost = nav_to_comm + 1 # communicate_image_data

                        cost = image_cost + comm_cost
                        min_cost_for_goal = min(min_cost_for_goal, cost)
                else:
                     # No suitable rover/camera or no imaging waypoint exists
                     pass # min_cost_for_goal remains INF

            # Add the minimum cost for this goal to the total
            total_cost += min_cost_for_goal

        # If total_cost is INF, it means at least one unachieved goal is unreachable.
        # Otherwise, it's the sum of minimum costs for each unachieved goal.
        return total_cost

