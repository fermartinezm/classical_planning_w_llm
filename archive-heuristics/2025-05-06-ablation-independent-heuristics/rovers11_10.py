import heapq
from collections import deque
import math

from heuristics.heuristic_base import Heuristic
from task import Operator, Task

# Helper function to parse PDDL facts represented as strings
def parse_fact(fact_string):
    """Parses a PDDL fact string into a predicate and arguments."""
    # Remove leading/trailing brackets and split by space
    parts = fact_string[1:-1].split()
    if not parts:
        return None, [] # Handle empty string case
    return parts[0], parts[1:] # predicate, args

class roversHeuristic(Heuristic):
    """
    Summary:
    Domain-dependent heuristic for the rovers domain. Estimates the cost to
    reach the goal by summing the estimated costs for each unachieved goal
    fact. The cost for each goal fact is estimated based on the sequence of
    actions required (navigate, sample/image, navigate, communicate) and
    the capabilities and current locations of the rovers. Navigation costs
    are estimated using precomputed shortest path distances (BFS) on the
    rover-specific traversal graphs.

    Assumptions:
    - The heuristic assumes that if a sample was initially present at a
      waypoint but is no longer in the state and no rover has the data,
      the goal is unreachable (does not model resampling after dropping).
    - The heuristic simplifies the imaging process; it estimates the cost
      for a single image capture and communication sequence and does not
      explicitly model the need for recalibration between multiple images
      taken by the same camera for different objectives/modes within the
      same goal achievement path (it assumes calibration is needed before
      each take_image if the camera is not calibrated, which is always true
      after a take_image action).
    - The heuristic adds a fixed cost (1 action for 'drop') if a rover's
      store is full when it needs to sample.
    - The heuristic sums costs for individual goals, potentially overestimating
      if actions contribute to multiple goals (non-admissible).
    - All objects referenced in facts (especially waypoints) are valid and
      exist in the parsed static/initial state information.

    Heuristic Initialization:
    The constructor precomputes static information and navigation data:
    - Parses static facts to identify lander locations, rover equipment,
      store ownership, camera details (on-board rover, supported modes,
      calibration target), waypoint visibility, and rover traversal graphs.
    - Extracts all unique objects of relevant types (rovers, waypoints, etc.)
      from the task definition (initial state, goals, static facts, operators).
    - Identifies initial soil and rock sample locations from the initial state.
    - Identifies communication waypoints (visible from lander locations).
    - Computes all-pairs shortest path distances for each rover on its
      specific traversal graph using BFS. This is stored in
      self.rover_distances[rover][start_waypoint][end_waypoint].

    Step-By-Step Thinking for Computing Heuristic:
    1. Initialize the total heuristic value `h` to 0.
    2. Parse the current state to extract dynamic information: rover locations,
       store states (full/empty) per rover, camera calibration states, and
       collected data (have_soil_analysis, have_rock_analysis, have_image).
    3. Iterate through each goal fact in the task's goals.
    4. If a goal fact is already present in the current state, its contribution
       to the heuristic is 0. Continue to the next goal.
    5. If a goal fact is not in the current state, estimate the minimum cost
       to achieve it:
       - For `(communicated_soil_data ?w)`:
         - Check if any rover currently has the soil analysis data `(have_soil_analysis ?r ?w)`.
           - If yes: The remaining task is to communicate the data. Find the minimum cost over all rovers `?r` having the data: (shortest path distance from `?r`'s current location to any communication waypoint reachable by `?r`) + 1 (communicate action). Add this minimum cost over suitable rovers to `h`.
           - If no: The data needs to be sampled and then communicated. Check if `?w` was an initial soil sample location. If not, the goal is impossible (add infinity to `h`). If yes: Find the minimum cost over all rovers `?r` equipped for soil analysis: (shortest path distance from `?r`'s current location to `?w`) + 1 (sample action) + (cost for store if full, 1 if full, 0 if empty/none) + (shortest path distance from `?w` to any communication waypoint reachable by `?r`) + 1 (communicate action). Add this minimum cost over equipped rovers to `h`.
       - For `(communicated_rock_data ?w)`: Follow a similar logic as for soil data, using rock-specific predicates and equipment.
       - For `(communicated_image_data ?o ?m)`:
         - Check if any rover currently has the image data `(have_image ?r ?o ?m)`.
           - If yes: The remaining task is to communicate the data. Find the minimum cost over all rovers `?r` having the data: (shortest path distance from `?r`'s current location to any communication waypoint reachable by `?r`) + 1 (communicate action). Add this minimum cost over suitable rovers to `h`.
           - If no: The image needs to be taken and then communicated. Find the minimum cost over all rovers `?r` equipped for imaging with a camera `?i` supporting mode `?m`: (shortest path distance from `?r`'s current location to a calibration waypoint for `?i`'s target) + 1 (calibrate action) + (shortest path distance from the calibration waypoint to an imaging waypoint for `?o`) + 1 (take_image action) + (shortest path distance from the imaging waypoint to any communication waypoint reachable by `?r`) + 1 (communicate action). This minimum is taken over all suitable rovers/cameras and all possible calibration/imaging waypoints. Add this minimum cost to `h`.
    6. After iterating through all goals, if the current state is the goal state, return 0. Otherwise, return the calculated `h`. If `h` is infinity (because an unachievable goal was detected), return infinity.
    """

    def __init__(self, task):
        super().__init__()
        self.goals = task.goals

        # --- Parse Static Information ---
        self.lander_location = {} # {lander: waypoint}
        self.rover_equipment = {} # {rover: {soil, rock, imaging}}
        self.store_owner = {} # {store: rover}
        self.camera_info = {} # {camera: {rover: r, modes: {m}, cal_target: t}}
        self.waypoint_graph = {} # {w1: {w2}} based on visible
        self.rover_traversal = {} # {rover: {w1: {w2}}} based on can_traverse
        self.objective_visibility = {} # {objective: {waypoint}}
        self.all_waypoints = set()

        # Collect all objects and initialize structures
        rovers = set()
        landers = set()
        stores = set()
        cameras = set()
        modes = set()
        objectives = set()

        # Collect objects and waypoints from all facts in the task
        all_facts_strings = task.initial_state | task.goals | task.static | set(f for op in task.operators for f in op.preconditions | op.add_effects | op.del_effects)
        for fact_string in all_facts_strings:
             pred, args = parse_fact(fact_string)
             if not pred: continue # Skip empty facts

             # Collect waypoints
             if pred in ('at', 'at_lander', 'at_soil_sample', 'at_rock_sample'):
                 if len(args) > 0: self.all_waypoints.add(args[-1]) # Last arg is usually waypoint
             elif pred in ('can_traverse', 'visible'):
                 if len(args) > 1:
                     self.all_waypoints.add(args[0])
                     self.all_waypoints.add(args[1])
             elif pred == 'visible_from':
                 if len(args) > 1: self.all_waypoints.add(args[1])

             # Collect objects by type based on predicate arguments
             if pred == 'at': rovers.add(args[0])
             elif pred == 'at_lander': landers.add(args[0])
             elif pred == 'can_traverse': rovers.add(args[0])
             elif pred in ('equipped_for_soil_analysis', 'equipped_for_rock_analysis', 'equipped_for_imaging'):
                 if len(args) > 0: rovers.add(args[0])
             elif pred in ('empty', 'full'):
                 if len(args) > 0: stores.add(args[0])
             elif pred in ('have_rock_analysis', 'have_soil_analysis', 'have_image'):
                 if len(args) > 0: rovers.add(args[0])
             elif pred == 'calibrated':
                 if len(args) > 1: cameras.add(args[0]); rovers.add(args[1])
             elif pred == 'supports':
                 if len(args) > 1: cameras.add(args[0]); modes.add(args[1])
             elif pred == 'have_image':
                 if len(args) > 2: objectives.add(args[1]); modes.add(args[2])
             elif pred == 'communicated_image_data':
                 if len(args) > 1: objectives.add(args[0]); modes.add(args[1])
             elif pred == 'visible_from':
                 if len(args) > 0: objectives.add(args[0])
             elif pred == 'store_of':
                 if len(args) > 1: stores.add(args[0]); rovers.add(args[1])
             elif pred == 'calibration_target':
                 if len(args) > 1: cameras.add(args[0]); objectives.add(args[1])
             elif pred == 'on_board':
                 if len(args) > 1: cameras.add(args[0]); rovers.add(args[1])


        for r in rovers: self.rover_equipment[r] = set()
        for i in cameras: self.camera_info[i] = {'modes': set()}
        for w in self.all_waypoints:
             self.waypoint_graph[w] = set()
             for r in rovers:
                 if r not in self.rover_traversal: self.rover_traversal[r] = {}
                 self.rover_traversal[r][w] = set()
        for o in objectives: self.objective_visibility[o] = set()


        for fact_string in task.static:
            pred, args = parse_fact(fact_string)
            if not pred: continue
            if pred == 'at_lander': self.lander_location[args[0]] = args[1]
            elif pred == 'equipped_for_soil_analysis': self.rover_equipment[args[0]].add('soil')
            elif pred == 'equipped_for_rock_analysis': self.rover_equipment[args[0]].add('rock')
            elif pred == 'equipped_for_imaging': self.rover_equipment[args[0]].add('imaging')
            elif pred == 'store_of': self.store_owner[args[0]] = args[1]
            elif pred == 'visible': self.waypoint_graph[args[0]].add(args[1])
            elif pred == 'can_traverse': self.rover_traversal[args[0]][args[1]].add(args[2])
            elif pred == 'on_board': self.camera_info[args[0]]['rover'] = args[1]
            elif pred == 'supports': self.camera_info[args[0]]['modes'].add(args[1])
            elif pred == 'calibration_target': self.camera_info[args[0]]['cal_target'] = args[1]
            elif pred == 'visible_from': self.objective_visibility[args[0]].add(args[1])

        # Identify initial sample locations
        self.initial_soil_samples = set()
        self.initial_rock_samples = set()
        for fact_string in task.initial_state:
            pred, args = parse_fact(fact_string)
            if not pred: continue
            if pred == 'at_soil_sample': self.initial_soil_samples.add(args[0])
            elif pred == 'at_rock_sample': self.initial_rock_samples.add(args[0])

        # Identify communication waypoints
        self.comm_waypoints = set()
        lander_locs = set(self.lander_location.values())
        for w1 in self.all_waypoints:
            for w2 in lander_locs:
                # Check visibility is symmetric? Domain says visible w1 w2 and visible w2 w1
                # We need waypoints w1 such that (visible w1 w2) is true for a lander location w2
                if w1 in self.waypoint_graph and w2 in self.waypoint_graph[w1]:
                    self.comm_waypoints.add(w1)
                # Also check the other direction just in case visible is not symmetric in the instance
                if w2 in self.waypoint_graph and w1 in self.waypoint_graph[w2]:
                     self.comm_waypoints.add(w1)


        # Precompute BFS distances for each rover
        self.rover_distances = {} # {rover: {start_w: {end_w: dist}}}
        for r in rovers:
            self.rover_distances[r] = {}
            # Use the traversal graph built for this rover
            graph = self.rover_traversal.get(r, {})

            # BFS from every waypoint that is relevant (appears in task)
            relevant_waypoints = self.all_waypoints

            # Ensure all relevant waypoints are keys in the graph dict for BFS
            # even if they have no outgoing edges
            graph_for_bfs = {w: set(neighbors) for w, neighbors in graph.items()}
            for w in relevant_waypoints:
                 if w not in graph_for_bfs:
                     graph_for_bfs[w] = set()

            for start_w in relevant_waypoints:
                 self.rover_distances[r][start_w] = self.bfs(graph_for_bfs, start_w)


    def bfs(self, graph, start_node):
        """Computes shortest path distances from start_node to all reachable nodes."""
        # Ensure start_node is in the graph keys for BFS to work correctly
        if start_node not in graph:
             # If start_node is not a source node, add it with no outgoing edges
             graph[start_node] = set()

        distances = {node: math.inf for node in graph}
        if start_node in distances: # Check if start_node is one of the nodes in the graph
            distances[start_node] = 0
            queue = deque([start_node])

            while queue:
                curr = queue.popleft()
                # Ensure curr is a key in the graph dict before iterating neighbors
                if curr in graph:
                    for neighbor in graph[curr]:
                        if neighbor in distances and distances[neighbor] == math.inf:
                            distances[neighbor] = distances[curr] + 1
                            queue.append(neighbor)
        # If start_node was not in the graph initially, distances will be all inf except possibly start_node itself if added.
        # The dict returned will contain distances only for nodes present in the graph keys.
        return distances

    def min_dist_to_set(self, rover, start_w, target_waypoints):
        """Finds the minimum shortest path distance from start_w to any waypoint in target_waypoints for the given rover."""
        # Check if rover exists and start_w is a known waypoint for this rover
        if rover not in self.rover_distances or start_w not in self.rover_distances[rover]:
            return math.inf # Rover cannot move or start_w is not in its graph data

        min_d = math.inf
        dists_from_start = self.rover_distances[rover][start_w]

        # Handle the case where start_w is one of the targets (distance is 0)
        if start_w in target_waypoints:
             return 0

        for tw in target_waypoints:
            if tw in dists_from_start:
                min_d = min(min_d, dists_from_start[tw])
        return min_d

    def __call__(self, node):
        state = node.state

        # Goal check first
        if self.goals <= state:
            return 0

        h = 0

        # --- Parse Dynamic Information from State ---
        rover_locations = {} # {rover: waypoint}
        store_states = {} # {rover: 'full'/'empty'}
        camera_calibrated = {} # {camera: True/False}
        have_soil = set() # {(rover, waypoint)}
        have_rock = set() # {(rover, waypoint)}
        have_image = set() # {(rover, objective, mode)}
        # at_soil_sample_state = set() # {waypoint} # Not strictly needed for heuristic logic
        # at_rock_sample_state = set() # {waypoint} # Not strictly needed for heuristic logic

        # Initialize camera calibrated state to False
        for cam in self.camera_info:
             camera_calibrated[cam] = False

        # Initialize store state to empty for rovers with stores
        for store, rover in self.store_owner.items():
             store_states[rover] = 'empty'

        # Initialize rover locations (assume rovers exist based on static/initial state)
        all_rovers = set(self.rover_equipment.keys()) # Rovers mentioned in equipped_for...
        for r in all_rovers:
             # Find initial location if not in state? No, state always has current location.
             # If a rover is not in state, it might not exist in this instance or is irrelevant.
             # We only care about rovers currently in the state.
             pass # Locations will be populated from state parsing


        for fact_string in state:
            pred, args = parse_fact(fact_string)
            if not pred: continue
            if pred == 'at': rover_locations[args[0]] = args[1]
            elif pred == 'full':
                 # Find the rover owning this store
                 store = args[0]
                 if store in self.store_owner:
                     rover = self.store_owner[store]
                     store_states[rover] = 'full'
            # elif pred == 'empty': # Initialized to empty, full overrides
            #      store = args[0]
            #      if store in self.store_owner:
            #          rover = self.store_owner[store]
            #          store_states[rover] = 'empty'
            elif pred == 'calibrated':
                 if len(args) > 0: camera_calibrated[args[0]] = True
            elif pred == 'have_soil_analysis':
                 if len(args) > 1: have_soil.add((args[0], args[1]))
            elif pred == 'have_rock_analysis':
                 if len(args) > 1: have_rock.add((args[0], args[1]))
            elif pred == 'have_image':
                 if len(args) > 2: have_image.add((args[0], args[1], args[2]))
            # elif pred == 'at_soil_sample': at_soil_sample_state.add(args[0])
            # elif pred == 'at_rock_sample': at_rock_sample_state.add(args[0])


        # --- Estimate Cost for Unachieved Goals ---
        for goal_string in self.goals:
            if goal_string in state:
                continue # Goal already achieved

            pred, args = parse_fact(goal_string)
            if not pred: continue # Should not happen for goal facts

            if pred == 'communicated_soil_data':
                w = args[0]
                found_rover_with_data = False
                min_comm_cost = math.inf

                # Check if any rover already has the data
                for r, soil_w in have_soil:
                    if soil_w == w:
                        found_rover_with_data = True
                        if r in rover_locations:
                            curr_w = rover_locations[r]
                            # Cost to communicate: navigate to comm + communicate
                            cost = self.min_dist_to_set(r, curr_w, self.comm_waypoints)
                            if cost != math.inf:
                                cost += 1 # communicate action
                                min_comm_cost = min(min_comm_cost, cost)
                        # else: rover location unknown (shouldn't happen if have_soil is true)

                if found_rover_with_data:
                    h += min_comm_cost
                else: # Need to sample and communicate
                    # Check if sample was initially present. If not, impossible.
                    if w not in self.initial_soil_samples:
                         h += math.inf
                         break # Unreachable goal makes total cost infinity

                    min_sample_comm_cost = math.inf
                    # Find best equipped rover
                    for r in self.rover_equipment:
                        if 'soil' in self.rover_equipment[r] and r in rover_locations:
                            curr_w = rover_locations[r]

                            # Cost to sample: nav to w + sample + (store cost)
                            cost_nav_sample = self.min_dist_to_set(r, curr_w, {w})
                            if cost_nav_sample == math.inf: continue

                            # Check store state for this rover
                            store_cost = 0
                            # Check if rover has a store and if it's full
                            if r in store_states and store_states[r] == 'full':
                                store_cost = 1 # Cost of drop action

                            cost_sample_step = cost_nav_sample + 1 + store_cost # nav + sample + drop(if needed)

                            # Cost to communicate after sampling (conceptually at w): nav to comm + communicate
                            cost_nav_comm = self.min_dist_to_set(r, w, self.comm_waypoints)
                            if cost_nav_comm == math.inf: continue
                            cost_comm_step = cost_nav_comm + 1 # nav + communicate

                            min_sample_comm_cost = min(min_sample_comm_cost, cost_sample_step + cost_comm_step)

                    h += min_sample_comm_cost

            elif pred == 'communicated_rock_data':
                w = args[0]
                found_rover_with_data = False
                min_comm_cost = math.inf

                # Check if any rover already has the data
                for r, rock_w in have_rock:
                    if rock_w == w:
                        found_rover_with_data = True
                        if r in rover_locations:
                            curr_w = rover_locations[r]
                            # Cost to communicate: navigate to comm + communicate
                            cost = self.min_dist_to_set(r, curr_w, self.comm_waypoints)
                            if cost != math.inf:
                                cost += 1 # communicate action
                                min_comm_cost = min(min_comm_cost, cost)
                        # else: rover location unknown

                if found_rover_with_data:
                    h += min_comm_cost
                else: # Need to sample and communicate
                    # Check if sample was initially present. If not, impossible.
                    if w not in self.initial_rock_samples:
                         h += math.inf
                         break # Unreachable goal makes total cost infinity

                    min_sample_comm_cost = math.inf
                    # Find best equipped rover
                    for r in self.rover_equipment:
                        if 'rock' in self.rover_equipment[r] and r in rover_locations:
                            curr_w = rover_locations[r]

                            # Cost to sample: nav to w + sample + (store cost)
                            cost_nav_sample = self.min_dist_to_set(r, curr_w, {w})
                            if cost_nav_sample == math.inf: continue

                            # Check store state for this rover
                            store_cost = 0
                            # Check if rover has a store and if it's full
                            if r in store_states and store_states[r] == 'full':
                                store_cost = 1 # Cost of drop action

                            cost_sample_step = cost_nav_sample + 1 + store_cost # nav + sample + drop(if needed)

                            # Cost to communicate after sampling (conceptually at w): nav to comm + communicate
                            cost_nav_comm = self.min_dist_to_set(r, w, self.comm_waypoints)
                            if cost_nav_comm == math.inf: continue
                            cost_comm_step = cost_nav_comm + 1 # nav + communicate

                            min_sample_comm_cost = min(min_sample_comm_cost, cost_sample_step + cost_comm_step)

                    h += min_sample_comm_cost

            elif pred == 'communicated_image_data':
                o = args[0]
                m = args[1]
                found_rover_with_data = False
                min_comm_cost = math.inf

                # Check if any rover already has the data
                for r, img_o, img_m in have_image:
                    if img_o == o and img_m == m:
                        found_rover_with_data = True
                        if r in rover_locations:
                            curr_w = rover_locations[r]
                            # Cost to communicate: navigate to comm + communicate
                            cost = self.min_dist_to_set(r, curr_w, self.comm_waypoints)
                            if cost != math.inf:
                                cost += 1 # communicate action
                                min_comm_cost = min(min_comm_cost, cost)
                        # else: rover location unknown

                if found_rover_with_data:
                    h += min_comm_cost
                else: # Need to image and communicate
                    min_image_comm_cost = math.inf
                    # Find best equipped rover/camera
                    for r in self.rover_equipment:
                        if 'imaging' in self.rover_equipment[r] and r in rover_locations:
                            for i in self.camera_info:
                                if self.camera_info[i].get('rover') == r and m in self.camera_info[i]['modes']:
                                    curr_w = rover_locations[r]
                                    cal_target = self.camera_info[i].get('cal_target')
                                    possible_cal_w = self.objective_visibility.get(cal_target, set())
                                    possible_img_w = self.objective_visibility.get(o, set())

                                    if not possible_cal_w or not possible_img_w:
                                        continue # Cannot calibrate or image this objective/mode

                                    # Cost = Nav(curr, cal) + Cal + Nav(cal, img) + Img + Nav(img, comm) + Comm
                                    best_path_cost_for_rover_cam = math.inf
                                    for w_cal in possible_cal_w:
                                        cost_nav1 = self.min_dist_to_set(r, curr_w, {w_cal})
                                        if cost_nav1 == math.inf: continue

                                        for p in possible_img_w:
                                            cost_nav2 = self.min_dist_to_set(r, w_cal, {p})
                                            if cost_nav2 == math.inf: continue

                                            cost_nav3 = self.min_dist_to_set(r, p, self.comm_waypoints)
                                            if cost_nav3 == math.inf: continue

                                            # Total cost for this path (curr->w_cal->p->comm)
                                            cost = cost_nav1 + 1 + cost_nav2 + 1 + cost_nav3 + 1
                                            best_path_cost_for_rover_cam = min(best_path_cost_for_rover_cam, cost)

                                    min_image_comm_cost = min(min_image_comm_cost, best_path_cost_for_rover_cam)

                    h += min_image_comm_cost

            # If at any point h becomes infinity, the rest don't matter
            if h == math.inf:
                 break

        return h
