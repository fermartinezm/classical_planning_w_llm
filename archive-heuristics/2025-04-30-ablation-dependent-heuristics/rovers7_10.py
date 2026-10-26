# Required imports
import collections
import math

# Helper function to parse a PDDL fact string
def parse_fact(fact_string):
    """
    Parses a PDDL fact string into a tuple (predicate, arg1, arg2, ...).
    Ignores the surrounding parentheses.
    """
    # Remove parentheses and split by space
    parts = fact_string.strip("()").split()
    return tuple(parts)

# Helper function to get objects from a fact string, ignoring predicate
def get_objects_from_fact(fact_string):
    """
    Parses a PDDL fact string and returns a tuple of its objects (arguments),
    ignoring the predicate and surrounding parentheses.
    """
    # Remove parentheses and split by space, take parts after the first one
    parts = fact_string.strip("()").split()
    return tuple(parts[1:]) if len(parts) > 1 else ()

# BFS for shortest path on the waypoint graph
def bfs(graph, start_node):
    """
    Performs Breadth-First Search on the graph starting from start_node
    to compute shortest distances to all reachable nodes.

    @param graph: Adjacency list representation of the graph.
    @param start_node: The starting node for the BFS.
    @return: A dictionary mapping reachable nodes to their shortest distance
             from the start_node. Nodes not in the graph or unreachable
             will not be in the result or implicitly have infinite distance.
    """
    # Initialize distances for all nodes present in the graph keys or values
    all_nodes = set(graph.keys())
    for neighbors in graph.values():
        all_nodes.update(neighbors)

    distances = {node: math.inf for node in all_nodes}

    if start_node not in all_nodes:
         # Start node is not in the graph at all (e.g., isolated waypoint)
         # Distances to all other nodes remain infinity. Distance to itself is 0.
         distances[start_node] = 0
         return distances

    distances[start_node] = 0
    queue = collections.deque([start_node])
    while queue:
        current_node = queue.popleft()
        # Ensure current_node is a valid key in the graph (it should be if in all_nodes)
        if current_node in graph:
            for neighbor in graph[current_node]:
                if distances[neighbor] == math.inf:
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)
    return distances


class roversHeuristic:
    """
    Domain-dependent heuristic for the Rovers domain.

    Summary:
    The heuristic estimates the cost to reach the goal by summing the estimated
    costs for each unachieved goal fact. For each unachieved goal fact
    (communicating soil data, rock data, or image data), it estimates the
    minimum number of actions required to achieve it, considering the steps
    needed to acquire the data/image (sampling, imaging, calibration, dropping)
    and the steps needed to communicate it (moving to a lander-visible waypoint).
    Navigation costs are estimated using precomputed shortest paths on the
    waypoint graph. The heuristic is non-admissible as it sums costs for
    potentially overlapping subgoals and simplifies some interactions (like
    calibration consumption).

    Assumptions:
    - The problem instance is solvable. Necessary equipment and traversable
      paths exist for goals to be achievable.
    - The waypoint graph defined by 'visible' facts (and implicitly 'can_traverse')
      is static and represents the navigation possibilities for all rovers.
      (Note: The PDDL shows 'can_traverse' is rover specific, but examples
       often make it uniform. We use 'visible' for graph building as it's
       required by 'navigate' and seems to define the topology).
    - Lander location is static.
    - Calibration targets, objective visibility, camera capabilities, and
      store-rover mapping are static.
    - Initial soil/rock samples at waypoints are static until sampled.

    Heuristic Initialization:
    The constructor precomputes static information from the task's static facts:
    - Lander location.
    - Rover capabilities (equipped for soil, rock, imaging).
    - Store mapping for each rover.
    - Waypoint graph based on 'visible' facts.
    - All-pairs shortest paths between waypoints using BFS.
    - Camera information (on-board rover, supported modes, calibration target).
    - Waypoints visible from objectives and calibration targets.
    - Waypoints visible from the lander location.

    Step-By-Step Thinking for Computing Heuristic:
    1.  Check if the current state is a goal state. If yes, return 0.
    2.  Identify the set of goal facts that are not true in the current state
        (`unachieved_goals`).
    3.  Initialize the total heuristic value `h = 0`.
    4.  For each `goal_fact` in `unachieved_goals`:
        a.  Parse the `goal_fact` to identify the predicate and arguments.
        b.  If `goal_fact` is `(communicated_soil_data ?w)`:
            i.  Find the minimum estimated cost to achieve this goal fact
                among all rovers equipped for soil analysis.
            ii. For a given equipped rover `r`:
                -   Check if `(have_soil_analysis r' ?w)` is in state for *any* equipped rover `r'`. Let this be `soil_data_already_acquired`.
                -   If `soil_data_already_acquired`:
                    -   Cost for rover `r` to communicate = 1 (communicate) + `min_dist_rover_to_lander_visible_waypoint(state, r)`.
                -   Else (need to sample):
                    -   If `(at_soil_sample ?w)` is NOT in state:
                        -   Cost is infinity (cannot sample).
                    -   Else:
                        -   Cost for rover `r` to sample and communicate:
                            -   `c = 1` (sample) + 1 (communicate)
                            -   `c += dist(current_rover_location(r), ?w)`
                            -   If the rover's store is full: `c += 1` (drop).
                            -   `c += min_dist_rover_to_lander_visible_waypoint(state, r)`.
                -   Update the minimum cost found so far for this goal fact.
            iii. Add the minimum estimated cost over all equipped rovers to `h`. If the minimum is infinity, the goal is unreachable from this state, and the total heuristic will be infinity.
        c.  If `goal_fact` is `(communicated_rock_data ?w)`:
            i.  Similar logic as for soil data, using rock analysis capabilities and facts. Check `(at_rock_sample ?w)`.
        d.  If `goal_fact` is `(communicated_image_data ?o ?m)`:
            i.  Find the minimum estimated cost to achieve this goal fact
                among all rover-camera pairs `(r, i)` where `r` is equipped
                for imaging, `i` is on board `r`, and `i` supports mode `m`.
            ii. For a given suitable pair `(r, i)`:
                -   Check if `(have_image r' ?o ?m)` is in state for *any* suitable pair `(r', i')`. Let this be `image_already_acquired`.
                -   If `image_already_acquired`:
                    -   Cost for pair `(r, i)` to communicate = 1 (communicate) + `min_dist_rover_to_lander_visible_waypoint(state, r)`.
                -   Else (need to take image):
                    -   Cost for pair `(r, i)` to take image and communicate:
                        -   `c = 1` (take image) + 1 (communicate).
                        -   `c += min_dist_rover_to_objective_visible_waypoint(state, r, ?o)`.
                        -   If `(calibrated i r)` not in state:
                            -   `c += 1` (calibrate).
                            -   `c += min_dist_rover_to_calibration_target_visible_waypoint(state, r, i)`.
                        -   `c += min_dist_rover_to_lander_visible_waypoint(state, r)`.
                -   Update the minimum cost found so far for this goal fact.
            iii. Add the minimum estimated cost over all suitable pairs to `h`. If the minimum is infinity, the goal is unreachable from this state, and the total heuristic will be infinity.
    5.  Return the total heuristic value `h`. If `h` is infinity (due to unreachable subgoals), return a large finite number or infinity depending on planner requirements (infinity is safer to indicate unsolvability, but a large number might be needed if the planner doesn't handle infinity well; let's use infinity).

    """
    def __init__(self, task):
        """
        Initializes the heuristic by precomputing static information.

        @param task: The planning task object.
        """
        self.task = task
        self.static_facts = task.static
        self.goal_facts = task.goals

        # --- Precompute Static Information ---

        self.lander_waypoint = None
        self.equipped_soil = set()
        self.equipped_rock = set()
        self.equipped_imaging = set()
        self.rover_store_map = {}
        self.waypoint_graph = collections.defaultdict(list)
        self.all_waypoints = set()
        self.camera_rover_map = {}
        self.camera_modes_map = collections.defaultdict(set)
        self.camera_caltarget_map = {}
        self.objective_visible_waypoints_map = collections.defaultdict(set)
        # Calibration targets are objectives, so their visible_from facts
        # are already captured in objective_visible_waypoints_map.
        # We will look up the calibration target objective name in that map.

        # Parse static facts to populate initial data structures
        for fact_string in self.static_facts:
            predicate, *objects = parse_fact(fact_string)

            if predicate == 'at_lander':
                # (at_lander ?x - lander ?y - waypoint)
                self.lander_waypoint = objects[1]
            elif predicate == 'equipped_for_soil_analysis':
                # (equipped_for_soil_analysis ?r - rover)
                self.equipped_soil.add(objects[0])
            elif predicate == 'equipped_for_rock_analysis':
                # (equipped_for_rock_analysis ?r - rover)
                self.equipped_rock.add(objects[0])
            elif predicate == 'equipped_for_imaging':
                # (equipped_for_imaging ?r - rover)
                self.equipped_imaging.add(objects[0])
            elif predicate == 'store_of':
                # (store_of ?s - store ?r - rover)
                self.rover_store_map[objects[1]] = objects[0]
            elif predicate == 'visible':
                # (visible ?w1 - waypoint ?w2 - waypoint)
                w1, w2 = objects
                self.waypoint_graph[w1].append(w2)
                self.waypoint_graph[w2].append(w1) # Assuming visible is symmetric
                self.all_waypoints.add(w1)
                self.all_waypoints.add(w2)
            elif predicate == 'on_board':
                # (on_board ?i - camera ?r - rover)
                self.camera_rover_map[objects[0]] = objects[1]
            elif predicate == 'supports':
                # (supports ?c - camera ?m - mode)
                self.camera_modes_map[objects[0]].add(objects[1])
            elif predicate == 'calibration_target':
                # (calibration_target ?i - camera ?t - objective)
                self.camera_caltarget_map[objects[0]] = objects[1]
            elif predicate == 'visible_from':
                # (visible_from ?o - objective ?w - waypoint)
                objective, waypoint = objects
                self.objective_visible_waypoints_map[objective].add(waypoint)


        # Add all waypoints mentioned in initial state facts (at, at_soil_sample, at_rock_sample)
        # and can_traverse facts to ensure all potential locations are considered in the graph nodes
        all_initial_facts = self.task.initial_state | self.static_facts
        for fact_string in all_initial_facts:
             predicate, *objects = parse_fact(fact_string)
             if predicate in ['at', 'at_lander', 'at_soil_sample', 'at_rock_sample']:
                 if objects: # Ensure objects list is not empty
                    self.all_waypoints.add(objects[-1]) # Waypoint is usually the last object
             elif predicate == 'can_traverse':
                 if len(objects) >= 3: # Ensure enough objects for rover, from_wp, to_wp
                    self.all_waypoints.add(objects[1])
                    self.all_waypoints.add(objects[2])


        # Ensure all waypoints from all_waypoints are keys in graph, even if isolated
        # This is necessary so BFS can be called on any waypoint.
        for wp in self.all_waypoints:
             if wp not in self.waypoint_graph:
                 self.waypoint_graph[wp] = []


        # Compute all-pairs shortest paths using BFS from each waypoint
        self.dist = {}
        for start_wp in self.waypoint_graph:
            self.dist[start_wp] = bfs(self.waypoint_graph, start_wp)

        # Find waypoints visible from the lander location
        self.lander_visible_waypoints_set = set()
        if self.lander_waypoint and self.lander_waypoint in self.waypoint_graph:
             # A waypoint ?x is lander-visible if (visible ?x lander_waypoint) is true.
             # This means ?x is a neighbor of lander_waypoint in the graph.
             self.lander_visible_waypoints_set = set(self.waypoint_graph.get(self.lander_waypoint, []))


    def get_rover_location(self, state, rover):
        """Finds the current waypoint of the given rover in the state."""
        for fact_string in state:
            predicate, *objects = parse_fact(fact_string)
            if predicate == 'at' and objects and objects[0] == rover:
                return objects[1]
        return None # Rover location not found (shouldn't happen in valid states)

    def is_store_full(self, state, rover):
        """Checks if the rover's store is full in the state."""
        store = self.rover_store_map.get(rover)
        if store:
            return f'(full {store})' in state
        return False # Rover has no store or store mapping is missing

    def min_dist_rover_to_waypoints(self, state, rover, target_waypoints):
        """
        Finds the minimum navigation distance from the rover's current location
        to any of the target waypoints. Returns math.inf if no target waypoint
        is reachable or the rover's location is unknown/isolated.
        """
        if not target_waypoints:
             return math.inf # No target waypoints to reach

        rover_loc = self.get_rover_location(state, rover)
        if rover_loc is None or rover_loc not in self.dist:
            return math.inf # Rover location unknown or isolated, or isolated waypoint

        min_d = math.inf
        # Look up distances from the rover's current location
        distances_from_rover_loc = self.dist.get(rover_loc, {})

        for target_wp in target_waypoints:
            min_d = min(min_d, distances_from_rover_loc.get(target_wp, math.inf))

        return min_d

    def __call__(self, state, goal):
        """
        Computes the heuristic value for the given state and goal.

        @param state: The current state (frozenset of facts).
        @param goal: The goal state (frozenset of facts).
        @return: The estimated number of actions to reach the goal.
        """
        # If the state is a goal state, the heuristic is 0.
        if goal <= state:
            return 0

        h = 0
        unachieved_goals = goal - state

        # Pre-check if any rover has soil/rock data or image for goals
        # This avoids re-calculating this check for every rover in the loop below
        have_soil_data_in_state = {} # waypoint -> bool
        have_rock_data_in_state = {} # waypoint -> bool
        have_image_in_state = {}    # (objective, mode) -> bool

        for goal_fact in unachieved_goals:
             predicate, *objects = parse_fact(goal_fact)
             if predicate == 'communicated_soil_data':
                 waypoint = objects[0]
                 if waypoint not in have_soil_data_in_state:
                     have_soil_data_in_state[waypoint] = any(
                         f'(have_soil_analysis {r} {waypoint})' in state
                         for r in self.equipped_soil
                     )
             elif predicate == 'communicated_rock_data':
                 waypoint = objects[0]
                 if waypoint not in have_rock_data_in_state:
                     have_rock_data_in_state[waypoint] = any(
                         f'(have_rock_analysis {r} {waypoint})' in state
                         for r in self.equipped_rock
                     )
             elif predicate == 'communicated_image_data':
                 objective, mode = objects
                 key = (objective, mode)
                 if key not in have_image_in_state:
                     # Check if any imaging rover has the image
                     have_image_in_state[key] = any(
                         f'(have_image {r} {objective} {mode})' in state
                         for r in self.equipped_imaging # Consider only imaging rovers
                         # No need to check camera here, as have_image fact is rover-specific
                         # and implies a suitable camera was used.
                     )


        for goal_fact in unachieved_goals:
            predicate, *objects = parse_fact(goal_fact)

            if predicate == 'communicated_soil_data':
                waypoint = objects[0]
                min_cost_for_goal = math.inf

                soil_data_already_acquired = have_soil_data_in_state.get(waypoint, False)
                sample_still_at_waypoint = f'(at_soil_sample {waypoint})' in state

                # If data is not acquired and sample is gone, this goal is unreachable
                if not soil_data_already_acquired and not sample_still_at_waypoint:
                     min_cost_for_goal = math.inf
                else:
                    # Iterate through rovers capable of soil analysis
                    for rover in self.equipped_soil:
                        rover_loc = self.get_rover_location(state, rover)
                        if rover_loc is None or rover_loc not in self.dist:
                             continue # Cannot use this rover if location unknown or isolated

                        current_rover_cost = 0

                        if not soil_data_already_acquired:
                            # Need to sample
                            current_rover_cost += 1 # sample_soil action
                            nav_cost_to_sample = self.dist[rover_loc].get(waypoint, math.inf)
                            if nav_cost_to_sample == math.inf: continue # Cannot reach sample location
                            current_rover_cost += nav_cost_to_sample

                            if self.is_store_full(state, rover):
                                current_rover_cost += 1 # drop action

                        # Need to communicate
                        current_rover_cost += 1 # communicate_soil_data action
                        nav_cost_to_lander_visible = self.min_dist_rover_to_waypoints(state, rover, self.lander_visible_waypoints_set)
                        if nav_cost_to_lander_visible == math.inf: continue # Cannot reach communication location
                        current_rover_cost += nav_cost_to_lander_visible

                        min_cost_for_goal = min(min_cost_for_goal, current_rover_cost)

                h += min_cost_for_goal

            elif predicate == 'communicated_rock_data':
                waypoint = objects[0]
                min_cost_for_goal = math.inf

                rock_data_already_acquired = have_rock_data_in_state.get(waypoint, False)
                sample_still_at_waypoint = f'(at_rock_sample {waypoint})' in state

                # If data is not acquired and sample is gone, this goal is unreachable
                if not rock_data_already_acquired and not sample_still_at_waypoint:
                     min_cost_for_goal = math.inf
                else:
                    # Iterate through rovers capable of rock analysis
                    for rover in self.equipped_rock:
                        rover_loc = self.get_rover_location(state, rover)
                        if rover_loc is None or rover_loc not in self.dist:
                             continue # Cannot use this rover

                        current_rover_cost = 0

                        if not rock_data_already_acquired:
                            # Need to sample
                            current_rover_cost += 1 # sample_rock action
                            nav_cost_to_sample = self.dist[rover_loc].get(waypoint, math.inf)
                            if nav_cost_to_sample == math.inf: continue # Cannot reach sample location
                            current_rover_cost += nav_cost_to_sample

                            if self.is_store_full(state, rover):
                                current_rover_cost += 1 # drop action

                        # Need to communicate
                        current_rover_cost += 1 # communicate_rock_data action
                        nav_cost_to_lander_visible = self.min_dist_rover_to_waypoints(state, rover, self.lander_visible_waypoints_set)
                        if nav_cost_to_lander_visible == math.inf: continue # Cannot reach communication location
                        current_rover_cost += nav_cost_to_lander_visible

                        min_cost_for_goal = min(min_cost_for_goal, current_rover_cost)

                h += min_cost_for_goal

            elif predicate == 'communicated_image_data':
                objective, mode = objects
                min_cost_for_goal = math.inf

                image_already_acquired = have_image_in_state.get((objective, mode), False)

                # Find suitable rover-camera pairs
                suitable_pairs = []
                for rover in self.equipped_imaging:
                    # Find cameras on this rover that support the mode
                    for camera, r_on_board in self.camera_rover_map.items():
                        if r_on_board == rover and mode in self.camera_modes_map.get(camera, set()):
                             suitable_pairs.append((rover, camera))

                # If no suitable equipment exists and image is not acquired, this goal is unreachable
                if not suitable_pairs and not image_already_acquired:
                     min_cost_for_goal = math.inf
                else:
                    # Iterate through suitable rover-camera pairs
                    for rover, camera in suitable_pairs:
                        rover_loc = self.get_rover_location(state, rover)
                        if rover_loc is None or rover_loc not in self.dist:
                             continue # Cannot use this rover

                        current_pair_cost = 0

                        if not image_already_acquired:
                            # Need to take image
                            current_pair_cost += 1 # take_image action

                            # Nav cost to objective-visible waypoint
                            obj_visible_wps = self.objective_visible_waypoints_map.get(objective, set())
                            nav_cost_to_obj_visible = self.min_dist_rover_to_waypoints(state, rover, obj_visible_wps)
                            if nav_cost_to_obj_visible == math.inf: continue # Cannot reach objective-visible location
                            current_pair_cost += nav_cost_to_obj_visible

                            # Calibration cost if needed
                            if f'(calibrated {camera} {rover})' not in state:
                                cal_target = self.camera_caltarget_map.get(camera)
                                if cal_target:
                                    current_pair_cost += 1 # calibrate action
                                    # Calibration targets are objectives, so their visible_from facts are in objective_visible_waypoints_map
                                    cal_visible_wps = self.objective_visible_waypoints_map.get(cal_target, set())
                                    nav_cost_to_cal_visible = self.min_dist_rover_to_waypoints(state, rover, cal_visible_wps)
                                    if nav_cost_to_cal_visible == math.inf: continue # Cannot reach calibration location
                                    current_pair_cost += nav_cost_to_cal_visible
                                else:
                                     # Camera requires calibration but has no calibration target defined
                                     # This makes calibration impossible.
                                     current_pair_cost = math.inf # Cannot calibrate
                                     if current_pair_cost == math.inf: continue # Cannot proceed with this pair

                        # Need to communicate
                        current_pair_cost += 1 # communicate_image_data action
                        nav_cost_to_lander_visible = self.min_dist_rover_to_waypoints(state, rover, self.lander_visible_waypoints_set)
                        if nav_cost_to_lander_visible == math.inf: continue # Cannot reach communication location
                        current_pair_cost += nav_pair_cost_to_lander_visible

                        min_cost_for_goal = min(min_cost_for_goal, current_pair_cost)

                h += min_cost_for_goal

            # If any goal is unreachable (cost is infinity), the total heuristic is infinity
            if h == math.inf:
                 return math.inf # Return infinity to indicate unsolvability from this state

        return h
