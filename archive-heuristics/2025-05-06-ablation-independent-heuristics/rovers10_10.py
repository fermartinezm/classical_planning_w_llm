import heapq
import logging
from collections import defaultdict, deque

from heuristics.heuristic_base import Heuristic
from task import Operator, Task # Assuming these are available in the environment


# Helper function to parse PDDL fact strings
def parse_fact(fact_string):
    """
    Parses a PDDL fact string into a predicate and its arguments.
    e.g., '(at rover1 waypoint1)' -> ('at', ['rover1', 'waypoint1'])
    """
    # Remove surrounding brackets and split by space
    parts = fact_string[1:-1].split()
    predicate = parts[0]
    args = parts[1:]
    return predicate, args


class roversHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Rovers domain.

    Summary:
    The heuristic estimates the cost to reach the goal state by summing up
    estimated costs for each unsatisfied goal literal. For each unsatisfied
    goal, it identifies the necessary actions (sample/take_image/calibrate/drop,
    communicate) and the required navigation between relevant waypoints
    (sample location, image location, calibration location, communication location).
    It finds the minimum cost sequence of actions and navigation for each goal,
    considering available rovers and resources, and sums these minimum costs.
    Navigation costs are estimated using precomputed shortest paths on the
    rover-specific traversal graphs.

    Assumptions:
    - The task object provides access to initial state, goals, static facts,
      and operators.
    - All necessary objects (rovers, waypoints, cameras, etc.) and their types
      can be inferred from the initial state and static facts.
    - The waypoint graph defined by (can_traverse ?r ?x ?y) is used for rover
      navigation. It is assumed that if (can_traverse ?r ?x ?y) is true, then
      (visible ?x ?y) is also true, as required by the navigate action.
    - Goals are reachable in principle from the initial state. If a goal is
      determined to be unreachable by the heuristic's logic based on available
      resources and graph connectivity, it returns infinity.
    - The cost of each action is 1.

    Heuristic Initialization:
    The constructor preprocesses the task information, primarily from static facts
    and the initial state, to build efficient data structures:
    - Identifies all objects by type (rovers, waypoints, cameras, etc.) by
      parsing initial state and static facts.
    - Builds rover-specific traversal graphs based on (can_traverse ?r ?x ?y).
    - Precomputes all-pairs shortest paths for each rover's graph using BFS.
    - Stores static predicates like (at_lander), (visible), (equipped_for_*),
      (store_of), (on_board), (supports), (calibration_target), (visible_from),
      (at_soil_sample - initial), (at_rock_sample - initial).
    - Identifies communication waypoints (visible from lander location).

    Step-By-Step Thinking for Computing Heuristic:
    1. Get the current state from the provided node.
    2. Parse the current state to get dynamic information: rover locations,
       store fullness, camera calibration status, and data/image possession
       by rovers.
    3. Initialize the total heuristic value `h` to 0.
    4. Iterate through each goal literal defined in `self.goals`.
    5. If a goal literal is already satisfied in the current state, skip it.
    6. If a goal literal is unsatisfied:
        a. Determine the type of goal (communicated_soil_data, communicated_rock_data, communicated_image_data).
        b. For `(communicated_soil_data W)`:
            - Calculate the minimum cost to achieve this goal across all suitable rovers.
            - The cost includes: 1 (communicate) + (1 if sample needed) + (1 if drop needed) + navigation cost.
            - Navigation cost if sample needed: dist(current_loc, W) + dist(W, comm_wp).
            - Navigation cost if sample not needed: dist(current_loc, comm_wp).
            - Minimize the total cost over suitable rovers and communication waypoints.
            - If no path is found for any suitable rover, the goal is unreachable (return infinity).
            - Add the minimum cost to `h`.
        c. For `(communicated_rock_data W)`: Similar logic as soil data.
        d. For `(communicated_image_data O M)`:
            - Calculate the minimum cost to achieve this goal across all suitable rovers and cameras.
            - The cost includes: 1 (communicate) + (1 if take_image needed) + (1 if calibrate needed) + navigation cost.
            - Navigation cost if calibrate and take_image needed: dist(current_loc, cal_wp) + dist(cal_wp, img_wp) + dist(img_wp, comm_wp).
            - Navigation cost if only take_image needed: dist(current_loc, img_wp) + dist(img_wp, comm_wp).
            - Navigation cost if only communicate needed: dist(current_loc, comm_wp).
            - Minimize the total cost over suitable rovers, cameras, calibration waypoints, image waypoints, and communication waypoints.
            - If no path is found for any suitable combination, the goal is unreachable (return infinity).
            - Add the minimum cost to `h`.
    7. Return the total heuristic value `h`. If any goal was unreachable, infinity was returned earlier.
    """

    def __init__(self, task):
        super().__init__()
        self.task = task
        self.goals = task.goals
        self.static_facts = task.static
        self.initial_state = task.initial_state

        # --- Preprocessing Static and Initial State Information ---

        # Infer object types by iterating through all facts
        all_relevant_facts = set(self.static_facts) | set(self.initial_state)
        self.rovers = set()
        self.waypoints = set()
        self.stores = set()
        self.cameras = set()
        self.modes = set()
        self.landers = set()
        self.objectives = set()

        for fact_string in all_relevant_facts:
            pred, args = parse_fact(fact_string)
            # Simple inference based on common predicate argument positions
            if pred in ['at', 'can_traverse']: # at(rover, wp), can_traverse(rover, wp, wp)
                if len(args) > 0: self.rovers.add(args[0])
                if len(args) > 1: self.waypoints.add(args[1])
                if len(args) > 2: self.waypoints.add(args[2])
            elif pred == 'at_lander': # at_lander(lander, wp)
                if len(args) > 0: self.landers.add(args[0])
                if len(args) > 1: self.waypoints.add(args[1])
            elif pred.startswith('equipped_for_'): # equipped_for_soil_analysis(rover)
                if len(args) > 0: self.rovers.add(args[0])
            elif pred in ['empty', 'full']: # empty(store), full(store)
                 if len(args) > 0: self.stores.add(args[0])
            elif pred == 'store_of': # store_of(store, rover)
                 if len(args) > 0: self.stores.add(args[0])
                 if len(args) > 1: self.rovers.add(args[1])
            elif pred in ['have_soil_analysis', 'have_rock_analysis']: # have_soil_analysis(rover, wp)
                 if len(args) > 0: self.rovers.add(args[0])
                 if len(args) > 1: self.waypoints.add(args[1])
            elif pred == 'have_image': # have_image(rover, objective, mode)
                 if len(args) > 0: self.rovers.add(args[0])
                 if len(args) > 1: self.objectives.add(args[1])
                 if len(args) > 2: self.modes.add(args[2])
            elif pred == 'calibrated': # calibrated(camera, rover)
                if len(args) > 0: self.cameras.add(args[0])
                if len(args) > 1: self.rovers.add(args[1])
            elif pred == 'supports': # supports(camera, mode)
                 if len(args) > 0: self.cameras.add(args[0])
                 if len(args) > 1: self.modes.add(args[1])
            elif pred == 'visible': # visible(wp, wp)
                 if len(args) > 0: self.waypoints.add(args[0])
                 if len(args) > 1: self.waypoints.add(args[1])
            elif pred in ['communicated_soil_data', 'communicated_rock_data']: # communicated_soil_data(wp)
                 if len(args) > 0: self.waypoints.add(args[0])
            elif pred == 'communicated_image_data': # communicated_image_data(objective, mode)
                 if len(args) > 0: self.objectives.add(args[0])
                 if len(args) > 1: self.modes.add(args[1])
            elif pred in ['at_soil_sample', 'at_rock_sample']: # at_soil_sample(wp)
                 if len(args) > 0: self.waypoints.add(args[0])
            elif pred == 'visible_from': # visible_from(objective, wp)
                 if len(args) > 0: self.objectives.add(args[0])
                 if len(args) > 1: self.waypoints.add(args[1])
            elif pred == 'calibration_target': # calibration_target(camera, objective)
                 if len(args) > 0: self.cameras.add(args[0])
                 if len(args) > 1: self.objectives.add(args[1])
            elif pred == 'on_board': # on_board(camera, rover)
                 if len(args) > 0: self.cameras.add(args[0])
                 if len(args) > 1: self.rovers.add(args[1])


        # Static predicate storage
        self.lander_location = None
        self.rover_capabilities = defaultdict(set) # rover -> {soil, rock, imaging}
        self.rover_stores = {} # rover -> store
        self.rover_cameras = defaultdict(list) # rover -> [camera1, camera2, ...]
        self.camera_modes = defaultdict(set) # camera -> {mode1, mode2, ...}
        self.camera_calibration_target = {} # camera -> objective
        self.objective_visible_from = defaultdict(set) # objective -> {waypoint1, waypoint2, ...}
        self.initial_soil_samples = set() # {waypoint1, waypoint2, ...}
        self.initial_rock_samples = set() # {waypoint1, waypoint2, ...}
        self.rover_traversal_graphs = {} # rover -> {wp1: {neighbor1, neighbor2}, ...}


        # Build traversal graphs and store static info
        can_traverse_edges = defaultdict(set) # rover -> {(wp_from, wp_to), ...}
        visible_edges = set() # {(wp1, wp2), ...}

        for fact_string in self.static_facts:
            pred, args = parse_fact(fact_string)
            if pred == 'at_lander':
                if len(args) == 2: self.lander_location = args[1]
            elif pred == 'equipped_for_soil_analysis':
                if len(args) == 1: self.rover_capabilities[args[0]].add('soil')
            elif pred == 'equipped_for_rock_analysis':
                if len(args) == 1: self.rover_capabilities[args[0]].add('rock')
            elif pred == 'equipped_for_imaging':
                if len(args) == 1: self.rover_capabilities[args[0]].add('imaging')
            elif pred == 'store_of':
                if len(args) == 2: self.rover_stores[args[1]] = args[0] # rover -> store
            elif pred == 'on_board':
                if len(args) == 2: self.rover_cameras[args[1]].append(args[0]) # rover -> camera
            elif pred == 'supports':
                if len(args) == 2: self.camera_modes[args[0]].add(args[1]) # camera -> mode
            elif pred == 'calibration_target':
                if len(args) == 2: self.camera_calibration_target[args[0]] = args[1] # camera -> objective
            elif pred == 'visible_from':
                if len(args) == 2: self.objective_visible_from[args[0]].add(args[1]) # objective -> waypoint
            elif pred == 'can_traverse':
                 if len(args) == 3:
                    can_traverse_edges[args[0]].add((args[1], args[2]))
            elif pred == 'visible':
                 if len(args) == 2:
                    visible_edges.add((args[0], args[1]))


        # Initial samples are in the initial state, not static facts
        for fact_string in self.initial_state:
             pred, args = parse_fact(fact_string)
             if pred == 'at_soil_sample':
                 if len(args) == 1: self.initial_soil_samples.add(args[0])
             elif pred == 'at_rock_sample':
                 if len(args) == 1: self.initial_rock_samples.add(args[0])

        # Build rover traversal graphs based on can_traverse
        # Note: The navigate action also requires (visible ?y ?z).
        # We assume here that (can_traverse r y z) implies (visible y z) for simplicity
        # or that the can_traverse facts only exist for visible pairs.
        for rover in self.rovers:
            graph = defaultdict(set)
            for wp1, wp2 in can_traverse_edges[rover]:
                 graph[wp1].add(wp2)
            self.rover_traversal_graphs[rover] = graph

        # Precompute shortest paths for each rover
        self.rover_dist = {} # rover -> {wp1: {wp2: dist, ...}, ...}
        for rover, graph in self.rover_traversal_graphs.items():
            self.rover_dist[rover] = self._all_pairs_shortest_paths(graph)

        # Precompute communication waypoints (visible from lander location)
        self.comm_wps = set()
        if self.lander_location:
            # Build a simple visible graph (undirected)
            visible_adj = defaultdict(set)
            for wp1, wp2 in visible_edges:
                visible_adj[wp1].add(wp2)
                visible_adj[wp2].add(wp1) # Assuming visible is symmetric

            # Communication waypoints are those visible from the lander location
            self.comm_wps = set(visible_adj.get(self.lander_location, []))
            # A rover can also be at the lander location itself if it's a waypoint
            if self.lander_location in self.waypoints:
                 self.comm_wps.add(self.lander_location)


    def _all_pairs_shortest_paths(self, graph):
        """Computes all-pairs shortest paths using BFS."""
        dist = {}
        # Collect all unique waypoints present in the graph edges or the overall waypoint set
        waypoints = set(graph.keys())
        for neighbors in graph.values():
            waypoints.update(neighbors)
        waypoints.update(self.waypoints) # Include all known waypoints
        waypoints = list(waypoints) # Convert to list for consistent iteration order

        for start_node in waypoints:
            dist[start_node] = {}
            q = deque([(start_node, 0)])
            visited = {start_node}
            dist[start_node][start_node] = 0

            while q:
                curr_node, curr_dist = q.popleft()

                for neighbor in graph.get(curr_node, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        dist[start_node][neighbor] = curr_dist + 1
                        q.append((neighbor, curr_dist + 1))

        # Fill in unreachable pairs with infinity for all known waypoints
        all_known_waypoints = self.waypoints # Use the set of all waypoints identified
        for wp1 in all_known_waypoints:
             if wp1 not in dist: dist[wp1] = {}
             for wp2 in all_known_waypoints:
                if wp2 not in dist[wp1]:
                    dist[wp1][wp2] = float('inf')

        return dist

    def _get_dist(self, rover, wp1, wp2):
        """Get shortest distance for a rover between two waypoints."""
        # Ensure rover and waypoints are valid and distance is precomputed
        if rover not in self.rover_dist or \
           wp1 not in self.rover_dist[rover] or \
           wp2 not in self.rover_dist[rover][wp1]:
             # This happens if a waypoint is not part of the rover's graph or no path exists
             return float('inf')
        return self.rover_dist[rover][wp1][wp2]

    def _min_dist_to_set(self, rover, start_wp, target_wps_set):
        """Get minimum shortest distance for a rover from start_wp to any waypoint in target_wps_set."""
        if not target_wps_set or start_wp not in self.waypoints: # Check if start_wp is a known waypoint
            return float('inf')
        min_d = float('inf')
        for target_wp in target_wps_set:
            min_d = min(min_d, self._get_dist(rover, start_wp, target_wp))
        return min_d


    def __call__(self, node):
        state = node.state

        # Check if goal is reached
        if self.task.goal_reached(state):
            return 0

        # --- Extract Dynamic State Information ---
        rover_locations = {} # rover -> waypoint
        store_full = {} # store -> bool
        calibrated_cameras = set() # {(camera, rover), ...}
        have_soil = set() # {(rover, waypoint), ...}
        have_rock = set() # {(rover, waypoint), ...}
        have_image = set() # {(rover, objective, mode), ...}

        for fact_string in state:
            pred, args = parse_fact(fact_string)
            if pred == 'at':
                if len(args) == 2: rover_locations[args[0]] = args[1]
            elif pred == 'full':
                if len(args) == 1: store_full[args[0]] = True
            elif pred == 'empty':
                 if len(args) == 1: store_full[args[0]] = False # Explicitly mark as not full
            elif pred == 'calibrated':
                if len(args) == 2: calibrated_cameras.add((args[0], args[1]))
            elif pred == 'have_soil_analysis':
                if len(args) == 2: have_soil.add((args[0], args[1]))
            elif pred == 'have_rock_analysis':
                if len(args) == 2: have_rock.add((args[0], args[1]))
            elif pred == 'have_image':
                if len(args) == 3: have_image.add((args[0], args[1], args[2]))

        # Ensure all rovers have a location (they should, from initial state)
        # If not, this state is likely invalid or unreachable, return inf.
        if any(rover not in rover_locations for rover in self.rovers):
             return float('inf')

        # Ensure all stores have a full/empty status (they should, from initial state)
        # If not, assume empty as per initial state convention.
        for store in self.stores:
             if store not in store_full:
                 store_full[store] = False


        # --- Compute Heuristic ---
        h = 0

        for goal_string in self.goals:
            if goal_string in state:
                continue # Goal already satisfied

            pred, args = parse_fact(goal_string)

            if pred == 'communicated_soil_data':
                waypoint_w = args[0]
                min_goal_cost = float('inf')

                # Find best rover path
                for rover in self.rovers:
                    if 'soil' in self.rover_capabilities.get(rover, set()):
                        current_loc = rover_locations[rover]
                        rover_cost = 0

                        # Check if rover already has the data
                        if (rover, waypoint_w) in have_soil:
                            # Rover has data, just need to communicate
                            nav_cost = self._min_dist_to_set(rover, current_loc, self.comm_wps)
                            if nav_cost != float('inf'):
                                 rover_cost = 1 + nav_cost # 1 for communicate action
                                 min_goal_cost = min(min_goal_cost, rover_cost)

                        else: # Need to sample
                            if waypoint_w in self.initial_soil_samples:
                                store_s = self.rover_stores.get(rover)
                                if store_s is None: continue # Rover has no store

                                sample_cost = 1 # sample_soil action
                                drop_cost = 0
                                if store_full.get(store_s, False):
                                    drop_cost = 1 # drop action

                                # Navigation: current -> W -> comm_wp
                                nav_cost_to_sample = self._get_dist(rover, current_loc, waypoint_w)
                                nav_cost_sample_to_comm = self._min_dist_to_set(rover, waypoint_w, self.comm_wps)

                                if nav_cost_to_sample != float('inf') and nav_cost_sample_to_comm != float('inf'):
                                     total_nav_cost = nav_cost_to_sample + nav_cost_sample_to_comm
                                     rover_cost = sample_cost + drop_cost + total_nav_cost + 1 # +1 for communicate action
                                     min_goal_cost = min(min_goal_cost, rover_cost)

                if min_goal_cost == float('inf'):
                    return float('inf') # Goal unreachable

                h += min_goal_cost

            elif pred == 'communicated_rock_data':
                waypoint_w = args[0]
                min_goal_cost = float('inf')

                # Find best rover path
                for rover in self.rovers:
                    if 'rock' in self.rover_capabilities.get(rover, set()):
                        current_loc = rover_locations[rover]
                        rover_cost = 0

                        # Check if rover already has the data
                        if (rover, waypoint_w) in have_rock:
                            # Rover has data, just need to communicate
                            nav_cost = self._min_dist_to_set(rover, current_loc, self.comm_wps)
                            if nav_cost != float('inf'):
                                 rover_cost = 1 + nav_cost # 1 for communicate action
                                 min_goal_cost = min(min_goal_cost, rover_cost)

                        else: # Need to sample
                            if waypoint_w in self.initial_rock_samples:
                                store_s = self.rover_stores.get(rover)
                                if store_s is None: continue # Rover has no store

                                sample_cost = 1 # sample_rock action
                                drop_cost = 0
                                if store_full.get(store_s, False):
                                    drop_cost = 1 # drop action

                                # Navigation: current -> W -> comm_wp
                                nav_cost_to_sample = self._get_dist(rover, current_loc, waypoint_w)
                                nav_cost_sample_to_comm = self._min_dist_to_set(rover, waypoint_w, self.comm_wps)

                                if nav_cost_to_sample != float('inf') and nav_cost_sample_to_comm != float('inf'):
                                     total_nav_cost = nav_cost_to_sample + nav_cost_sample_to_comm
                                     rover_cost = sample_cost + drop_cost + total_nav_cost + 1 # +1 for communicate action
                                     min_goal_cost = min(min_goal_cost, rover_cost)

                if min_goal_cost == float('inf'):
                    return float('inf') # Goal unreachable

                h += min_goal_cost


            elif pred == 'communicated_image_data':
                objective_o = args[0]
                mode_m = args[1]
                min_goal_cost = float('inf')

                # Find best rover/camera path
                image_wps = self.objective_visible_from.get(objective_o, set())
                if not image_wps:
                     # Cannot take image of this objective
                     return float('inf') # Goal unreachable

                for rover in self.rovers:
                    if 'imaging' in self.rover_capabilities.get(rover, set()):
                        current_loc = rover_locations[rover]
                        for camera in self.rover_cameras.get(rover, []):
                            if mode_m in self.camera_modes.get(camera, set()):
                                # Found suitable rover and camera
                                rover_camera_cost = 0

                                # Check if rover already has the image
                                if (rover, objective_o, mode_m) in have_image:
                                    # Rover has image, just need to communicate
                                    nav_cost = self._min_dist_to_set(rover, current_loc, self.comm_wps)
                                    if nav_cost != float('inf'):
                                         rover_camera_cost = 1 + nav_cost # 1 for communicate action
                                         min_goal_cost = min(min_goal_cost, rover_camera_cost)

                                else: # Need to take image
                                    cal_target = self.camera_calibration_target.get(camera)
                                    if not cal_target: continue # Camera cannot be calibrated

                                    cal_wps = self.objective_visible_from.get(cal_target, set())
                                    if not cal_wps: continue # Cannot calibrate camera at any known waypoint

                                    take_image_cost = 1 # take_image action
                                    calibrate_cost = 0
                                    nav_cost = float('inf')

                                    if (camera, rover) not in calibrated_cameras:
                                        calibrate_cost = 1 # calibrate action
                                        # Navigation: current -> cal_wp -> img_wp -> comm_wp
                                        # min_{W in cal_wps, P in image_wps} (dist(current, W) + dist(W, P) + min_dist(P, comm_wps))
                                        min_nav_cost_seq = float('inf')
                                        for W in cal_wps:
                                            for P in image_wps:
                                                d1 = self._get_dist(rover, current_loc, W)
                                                d2 = self._get_dist(rover, W, P)
                                                d3 = self._min_dist_to_set(rover, P, self.comm_wps)
                                                if d1 != float('inf') and d2 != float('inf') and d3 != float('inf'):
                                                    min_nav_cost_seq = min(min_nav_cost_seq, d1 + d2 + d3)
                                        nav_cost = min_nav_cost_seq

                                    else: # calibrated
                                        # Navigation: current -> img_wp -> comm_wp
                                        # min_{P in image_wps} (dist(current, P) + min_dist(P, comm_wps))
                                        min_nav_cost_seq = float('inf')
                                        for P in image_wps:
                                            d1 = self._get_dist(rover, current_loc, P)
                                            d2 = self._min_dist_to_set(rover, P, self.comm_wps)
                                            if d1 != float('inf') and d2 != float('inf'):
                                                min_nav_cost_seq = min(min_nav_cost_seq, d1 + d2)
                                        nav_cost = min_nav_cost_seq

                                    if nav_cost != float('inf'):
                                         rover_camera_cost = take_image_cost + calibrate_cost + nav_cost + 1 # +1 for communicate action
                                         min_goal_cost = min(min_goal_cost, rover_camera_cost)

                if min_goal_cost == float('inf'):
                    return float('inf') # Goal unreachable

                h += min_goal_cost

            # Add other goal types if any (only soil, rock, image in this domain)
            # ...

        return h
