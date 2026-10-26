import heapq
import math
import logging

from heuristics.heuristic_base import Heuristic
from task import Operator, Task

class roversHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Rovers domain.

    Summary:
    This heuristic estimates the cost to reach a goal state by summing the
    estimated costs for each unachieved goal fact. For each unachieved goal,
    it calculates the minimum cost required to satisfy that specific goal fact,
    considering whether the necessary data (soil/rock analysis, image) has
    already been collected or if it needs to be collected first. The cost
    estimation includes navigation steps (estimated by shortest path distance
    on the waypoint graph), sampling/imaging actions, calibration, dropping
    samples, and the final communication action. It considers the capabilities
    and current locations of all rovers to find the minimum cost path for
    each goal.

    Assumptions:
    - The waypoint graph defined by 'visible' predicates is undirected and
      represents traversable links for all rovers (as 'can_traverse' seems
      to mirror 'visible' in the domain definition).
    - The cost of each action is 1.
    - The heuristic assumes that any equipped/capable rover can be used to
      achieve a goal component (sampling, imaging, communicating) and selects
      the one that appears cheapest based on current location and required steps.
    - Calibration cost is simplified to 1 if the camera is not currently
      calibrated, without explicitly calculating navigation to a calibration
      target waypoint. Navigation to the imaging waypoint is included.
    - The heuristic relies on precomputed shortest path distances between
      all waypoints.

    Heuristic Initialization:
    The constructor performs precomputation based on the static facts of the
    planning task:
    1. Parses all facts in the domain/problem to identify all waypoints.
    2. Extracts static information such as lander location, rover equipment,
       store-to-rover mapping, camera details (on-board, supported modes,
       calibration targets), waypoint visibility (to build the graph), and
       objective visibility from waypoints.
    3. Identifies communication points (waypoints visible from the lander).
    4. Precomputes all-pairs shortest path distances between all waypoints
       using Breadth-First Search (BFS) on the waypoint graph derived from
       'visible' predicates.
    5. Stores the initial locations of soil and rock samples for feasibility
       checks during heuristic computation.

    Step-By-Step Thinking for Computing Heuristic (__call__ method):
    1. Initialize the total heuristic value `h` to 0.
    2. Extract dynamic information from the current state: rover locations,
       store fullness, collected soil/rock data, taken images, and camera
       calibration status.
    3. Iterate through each goal fact defined in the task.
    4. If a goal fact is already present in the current state, its contribution
       to the heuristic is 0, and we continue to the next goal.
    5. If a goal fact is not in the state, estimate the minimum cost to achieve it:
        a.  **For `(communicated_soil_data ?w)`:**
            -   Calculate `cost_if_collected`: If `(have_soil_analysis ?r ?w)` is true for any equipped rover `?r`, the cost is 1 (communicate) plus the navigation cost for that rover from its current location to the nearest communication point. Minimize this cost over all rovers that have the data. If no rover has the data, this cost is infinity.
            -   Calculate `cost_if_not_collected`: If a soil sample initially existed at `?w` and there is an equipped rover `?r_equip`, the cost involves sampling and communicating. The cost is the navigation cost for `?r_equip` from its current location to `?w`, plus 1 if its store is full (for dropping), plus 1 (sample action), plus the navigation cost from `?w` to the nearest communication point, plus 1 (communicate action). Minimize this total cost over all equipped rovers `?r_equip`. If no initial sample or no equipped rover, this cost is infinity.
            -   The estimated cost for this goal is the minimum of `cost_if_collected` and `cost_if_not_collected`. Add this to `h`.
        b.  **For `(communicated_rock_data ?w)`:** Similar logic as soil data, checking for rock samples and rock analysis equipment.
        c.  **For `(communicated_image_data ?o ?m)`:**
            -   Calculate `cost_if_collected`: If `(have_image ?r ?o ?m)` is true for any capable rover `?r` (equipped for imaging, with a camera supporting the mode), the cost is 1 (communicate) plus the navigation cost for that rover from its current location to the nearest communication point. Minimize over rovers that have the image. If no rover has the image, this cost is infinity.
            -   Calculate `cost_if_not_collected`: If there is a capable rover `?r_capable` and a waypoint `?image_w` visible from `?o`, the cost involves imaging and communicating. The cost is the navigation cost for `?r_capable` from its current location to `?image_w`, plus 1 if the camera needs calibration, plus 1 (take_image action), plus the navigation cost from `?image_w` to the nearest communication point, plus 1 (communicate action). Minimize this total cost over all capable rovers `?r_capable` and all suitable imaging waypoints `?image_w`. If no capable rover or no suitable imaging waypoint, this cost is infinity.
            -   The estimated cost for this goal is the minimum of `cost_if_collected` and `cost_if_not_collected`. Add this to `h`.
    6. If at any point the estimated cost for a goal is infinity (meaning it's impossible to achieve from the initial state based on static facts and initial samples), the total heuristic is infinity.
    7. Return the total heuristic value `h`.
    """

    def __init__(self, task):
        super().__init__(task)
        self.all_facts = task.facts # Store all facts to find all objects
        self.lander_location = None
        self.equipped_soil = set()
        self.equipped_rock = set()
        self.equipped_imaging = set()
        self.store_to_rover = {} # store -> rover
        self.camera_on_rover = {} # camera -> rover
        self.camera_modes = {} # camera -> set of modes
        self.camera_cal_target = {} # camera -> objective
        self.waypoint_graph = {} # waypoint -> set of neighbors
        self.obj_visible_from_waypoints = {} # objective -> set of waypoints
        self.communication_points = set()
        self.initial_soil_samples = set()
        self.initial_rock_samples = set()
        self.dist = {} # waypoint -> waypoint -> distance

        self._parse_static(task.static)
        self._precompute_distances()

        # Store initial state samples for feasibility checks
        for fact_str in task.initial_state:
             pred, args = self._parse_fact(fact_str)
             if pred == 'at_soil_sample':
                 self.initial_soil_samples.add(args[0])
             elif pred == 'at_rock_sample':
                 self.initial_rock_samples.add(args[0])

    def _parse_fact(self, fact_str):
         """Helper to parse fact string into predicate and arguments."""
         parts = fact_str.strip('()').split()
         predicate = parts[0]
         args = parts[1:]
         return predicate, args

    def _parse_static(self, static_facts):
        """Extracts information from static facts."""
        all_waypoints = set()

        # First pass to collect all waypoints from relevant static facts
        for fact_str in static_facts:
            pred, args = self._parse_fact(fact_str)
            if pred == 'visible' or pred == 'can_traverse':
                 all_waypoints.add(args[1])
                 all_waypoints.add(args[2])
            elif pred == 'at_lander':
                 self.lander_location = args[1]
                 all_waypoints.add(args[1]) # Lander is at a waypoint
            elif pred == 'visible_from':
                 all_waypoints.add(args[1])

        # Second pass to extract structured static info
        for fact_str in static_facts:
            pred, args = self._parse_fact(fact_str)
            if pred == 'at_lander':
                 self.lander_location = args[1]
            elif pred == 'equipped_for_soil_analysis':
                 self.equipped_soil.add(args[0])
            elif pred == 'equipped_for_rock_analysis':
                 self.equipped_rock.add(args[0])
            elif pred == 'equipped_for_imaging':
                 self.equipped_imaging.add(args[0])
            elif pred == 'store_of':
                 self.store_to_rover[args[0]] = args[1]
            elif pred == 'on_board':
                 self.camera_on_rover[args[0]] = args[1]
            elif pred == 'supports':
                 camera, mode = args
                 if camera not in self.camera_modes:
                     self.camera_modes[camera] = set()
                 self.camera_modes[camera].add(mode)
            elif pred == 'calibration_target':
                 self.camera_cal_target[args[0]] = args[1]
            elif pred == 'visible_from':
                 objective, waypoint = args
                 if objective not in self.obj_visible_from_waypoints:
                     self.obj_visible_from_waypoints[objective] = set()
                 self.obj_visible_from_waypoints[objective].add(waypoint)

        # Build waypoint graph from visible facts
        for fact_str in static_facts:
            pred, args = self._parse_fact(fact_str)
            if pred == 'visible':
                w1, w2 = args
                if w1 not in self.waypoint_graph:
                    self.waypoint_graph[w1] = set()
                if w2 not in self.waypoint_graph:
                    self.waypoint_graph[w2] = set()
                self.waypoint_graph[w1].add(w2)
                self.waypoint_graph[w2].add(w1) # Assuming visible is symmetric

        # Identify communication points
        if self.lander_location:
             # Iterate through all waypoints mentioned in the graph or lander location
             all_known_waypoints = set(self.waypoint_graph.keys())
             if self.lander_location:
                 all_known_waypoints.add(self.lander_location)

             for w in all_known_waypoints:
                 # Check if (visible w self.lander_location) is in static facts
                 if f'(visible {w} {self.lander_location})' in static_facts:
                     self.communication_points.add(w)
                 # Also check the reverse, as visible is symmetric
                 if f'(visible {self.lander_location} {w})' in static_facts:
                      self.communication_points.add(w)


    def _precompute_distances(self):
        """Computes all-pairs shortest paths between waypoints using BFS."""
        # Need a list of all possible waypoints in the domain
        all_waypoints = set()
        for fact_str in self.all_facts:
             pred, args = self._parse_fact(fact_str)
             # Identify waypoints from predicates that use them
             if pred in ['at', 'at_lander', 'can_traverse', 'visible',
                         'at_soil_sample', 'at_rock_sample', 'visible_from']:
                 for arg in args:
                     # Simple domain-specific check: if arg starts with 'waypoint', assume it's a waypoint
                     if arg.startswith('waypoint'):
                         all_waypoints.add(arg)

        waypoints_list = list(all_waypoints)

        for start_w in waypoints_list:
            self.dist[start_w] = {}
            for w in waypoints_list:
                self.dist[start_w][w] = math.inf
            self.dist[start_w][start_w] = 0
            q = [start_w]
            visited = {start_w}

            while q:
                u = q.pop(0)
                neighbors = self.waypoint_graph.get(u, set())
                for v in neighbors:
                    if v not in visited:
                        visited.add(v)
                        self.dist[start_w][v] = self.dist[start_w][u] + 1
                        q.append(v)

    def __call__(self, node):
        """
        Computes the domain-dependent heuristic value for the given state.
        """
        state = node.state
        goals = self.goals
        h = 0

        # Extract dynamic state information
        rover_locations = {} # rover -> waypoint
        store_full = {} # store -> bool
        have_soil = {} # (rover, waypoint) -> bool
        have_rock = {} # (rover, waypoint) -> bool
        have_image = {} # (rover, objective, mode) -> bool
        calibrated = {} # (camera, rover) -> bool

        for fact_str in state:
            pred, args = self._parse_fact(fact_str)
            if pred == 'at':
                rover_locations[args[0]] = args[1]
            elif pred == 'full':
                store_full[args[0]] = True
            elif pred == 'have_soil_analysis':
                have_soil[(args[0], args[1])] = True
            elif pred == 'have_rock_analysis':
                have_rock[(args[0], args[1])] = True
            elif pred == 'have_image':
                have_image[(args[0], args[1], args[2])] = True
            elif pred == 'calibrated':
                calibrated[(args[0], args[1])] = True

        # Check each goal
        for goal_str in goals:
            if goal_str in state:
                continue # Goal already achieved

            pred, args = self._parse_fact(goal_str)

            if pred == 'communicated_soil_data':
                w = args[0]
                cost_if_collected = math.inf
                cost_if_not_collected = math.inf

                # Case 1: Data already collected by some rover
                # Consider only rovers that are equipped for soil analysis
                roaming_rovers = set(rover_locations.keys())
                rovers_with_data = {r for r in roaming_rovers if have_soil.get((r, w), False)}

                if rovers_with_data:
                    min_comm_cost_for_collected = math.inf
                    for r_data in rovers_with_data:
                         current_w = rover_locations[r_data]
                         nav_cost = math.inf
                         if self.communication_points:
                             nav_cost = min(self.dist.get(current_w, {}).get(comm_w, math.inf) for comm_w in self.communication_points)
                         min_comm_cost_for_collected = min(min_comm_cost_for_collected, 1 + nav_cost) # 1 for communicate action
                    cost_if_collected = min_comm_cost_for_collected


                # Case 2: Data not yet collected
                if w in self.initial_soil_samples: # Check if sample exists initially
                    R_equip = self.equipped_soil.intersection(roaming_rovers) # Only consider equipped rovers that are currently active
                    if R_equip:
                        min_cost_for_any_rover = math.inf
                        for r_equip in R_equip:
                            current_w = rover_locations[r_equip]

                            # Cost to sample = nav to w + drop (if needed) + 1
                            cost_to_w = self.dist.get(current_w, {}).get(w, math.inf)
                            if cost_to_w == math.inf: continue # Cannot reach sample location

                            store = None
                            # Find store for this rover
                            for s, rover in self.store_to_rover.items():
                                if rover == r_equip:
                                    store = s
                                    break
                            drop_cost = 1 if store is not None and store_full.get(store, False) else 0

                            # Cost to communicate after sampling = nav from w to comm point + 1
                            cost_from_w_to_comm = math.inf
                            if self.communication_points:
                                 cost_from_w_to_comm = min(self.dist.get(w, {}).get(comm_w, math.inf) for comm_w in self.communication_points)
                            if cost_from_w_to_comm == math.inf: continue # Cannot reach comm point from sample location

                            total_rover_cost = cost_to_w + drop_cost + 1 + cost_from_w_to_comm + 1 # 1 for sample, 1 for communicate
                            min_cost_for_any_rover = min(min_cost_for_any_rover, total_rover_cost)
                        cost_if_not_collected = min_cost_for_any_rover

                h += min(cost_if_collected, cost_if_not_collected)

            elif pred == 'communicated_rock_data':
                w = args[0]
                cost_if_collected = math.inf
                cost_if_not_collected = math.inf

                # Case 1: Data already collected by some rover
                roaming_rovers = set(rover_locations.keys())
                rovers_with_data = {r for r in roaming_rovers if have_rock.get((r, w), False)}

                if rovers_with_data:
                    min_comm_cost_for_collected = math.inf
                    for r_data in rovers_with_data:
                         current_w = rover_locations[r_data]
                         nav_cost = math.inf
                         if self.communication_points:
                             nav_cost = min(self.dist.get(current_w, {}).get(comm_w, math.inf) for comm_w in self.communication_points)
                         min_comm_cost_for_collected = min(min_comm_cost_for_collected, 1 + nav_cost) # 1 for communicate action
                    cost_if_collected = min_comm_cost_for_collected

                # Case 2: Data not yet collected
                if w in self.initial_rock_samples: # Check if sample exists initially
                    R_equip = self.equipped_rock.intersection(roaming_rovers) # Only consider equipped rovers that are currently active
                    if R_equip:
                        min_cost_for_any_rover = math.inf
                        for r_equip in R_equip:
                            current_w = rover_locations[r_equip]

                            # Cost to sample = nav to w + drop (if needed) + 1
                            cost_to_w = self.dist.get(current_w, {}).get(w, math.inf)
                            if cost_to_w == math.inf: continue # Cannot reach sample location

                            store = None
                            # Find store for this rover
                            for s, rover in self.store_to_rover.items():
                                if rover == r_equip:
                                    store = s
                                    break
                            drop_cost = 1 if store is not None and store_full.get(store, False) else 0

                            # Cost to communicate after sampling = nav from w to comm point + 1
                            cost_from_w_to_comm = math.inf
                            if self.communication_points:
                                 cost_from_w_to_comm = min(self.dist.get(w, {}).get(comm_w, math.inf) for comm_w in self.communication_points)
                            if cost_from_w_to_comm == math.inf: continue # Cannot reach comm point from sample location

                            total_rover_cost = cost_to_w + drop_cost + 1 + cost_from_w_to_comm + 1 # 1 for sample, 1 for communicate
                            min_cost_for_any_rover = min(min_cost_for_any_rover, total_rover_cost)
                        cost_if_not_collected = min_cost_for_any_rover

                h += min(cost_if_collected, cost_if_not_collected)


            elif pred == 'communicated_image_data':
                o, m = args
                cost_if_collected = math.inf
                cost_if_not_collected = math.inf

                # Case 1: Image already taken by some rover
                roaming_rovers = set(rover_locations.keys())
                # Find rovers capable of taking this image (equipped, with camera supporting mode)
                R_capable_of_image_static = set()
                for r in self.equipped_imaging:
                    for cam, rover_with_cam in self.camera_on_rover.items():
                        if rover_with_cam == r and m in self.camera_modes.get(cam, set()):
                            R_capable_of_image_static.add(r)
                            break # Found a camera for this rover/mode

                rovers_with_image = {r for r in R_capable_of_image_static.intersection(roaming_rovers) if have_image.get((r, o, m), False)}

                if rovers_with_image:
                    min_comm_cost_for_collected = math.inf
                    for r_image in rovers_with_image:
                        current_w = rover_locations[r_image]
                        nav_cost = math.inf
                        if self.communication_points:
                            nav_cost = min(self.dist.get(current_w, {}).get(comm_w, math.inf) for comm_w in self.communication_points)
                        min_comm_cost_for_collected = min(min_comm_cost_for_collected, 1 + nav_cost) # 1 for communicate action
                    cost_if_collected = min_comm_cost_for_collected


                # Case 2: Image not yet taken
                W_visible = self.obj_visible_from_waypoints.get(o, set())
                R_capable_of_image_active = R_capable_of_image_static.intersection(roaming_rovers)

                if R_capable_of_image_active and W_visible:
                    min_cost_for_any_rover = math.inf
                    for r_capable in R_capable_of_image_active:
                        current_w = rover_locations[r_capable]

                        # Find the camera on this rover supporting the mode
                        camera_for_mode = None
                        for cam, rover_with_cam in self.camera_on_rover.items():
                            if rover_with_cam == r_capable and m in self.camera_modes.get(cam, set()):
                                camera_for_mode = cam
                                break
                        if camera_for_mode is None: continue # Should not happen if r_capable is in R_capable_of_image_static

                        # Cost to take image = nav to image_w + calibrate (if needed) + 1
                        min_cost_via_any_image_w = math.inf
                        for image_w in W_visible:
                            cost_to_image_w = self.dist.get(current_w, {}).get(image_w, math.inf)
                            if cost_to_image_w == math.inf: continue # Cannot reach image location

                            cal_cost = 1 if not calibrated.get((camera_for_mode, r_capable), False) else 0

                            # Cost to communicate after imaging = nav from image_w to comm point + 1
                            cost_from_image_w_to_comm = math.inf
                            if self.communication_points:
                                 cost_from_image_w_to_comm = min(self.dist.get(image_w, {}).get(comm_w, math.inf) for comm_w in self.communication_points)
                            if cost_from_image_w_to_comm == math.inf: continue # Cannot reach comm point from image location

                            total_path_cost = cost_to_image_w + cal_cost + 1 + cost_from_image_w_to_comm + 1 # 1 for take_image, 1 for communicate
                            min_cost_via_any_image_w = min(min_cost_via_any_image_w, total_path_cost)

                        min_cost_for_any_rover = min(min_cost_for_any_rover, min_cost_via_any_image_w)
                    cost_if_not_collected = min_cost_for_any_rover

                h += min(cost_if_collected, cost_if_not_collected)

            # If h is already infinity, no need to continue
            if h == math.inf:
                return math.inf

        return h

