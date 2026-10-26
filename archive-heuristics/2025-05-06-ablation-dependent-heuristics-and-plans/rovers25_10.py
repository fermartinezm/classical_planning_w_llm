import collections
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at_lander general waypoint1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def bfs_shortest_path(graph, start):
    """
    Computes shortest path distances from a start node in a graph.
    Graph is represented as a dictionary: node -> list of neighbors.
    Returns a dictionary: node -> distance.
    """
    # Initialize distances for all nodes in the graph
    distances = {node: float('inf') for node in graph}

    # If the start node is not in the graph, we cannot compute paths from it.
    # This shouldn't happen if the graph is built correctly to include all waypoints.
    if start not in graph:
         return distances # All distances remain infinity

    distances[start] = 0
    queue = collections.deque([start])
    while queue:
        current = queue.popleft()
        # Ensure current node has neighbors in the graph (should always be true here)
        if current in graph:
            for neighbor in graph[current]:
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = distances[current] + 1
                    queue.append(neighbor)
    return distances


class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve all
    unmet goal conditions. It breaks down the goal conditions into sub-tasks
    (sampling, imaging, communicating) and estimates the minimum actions
    needed for each, including navigation costs (precomputed shortest paths).
    The total heuristic is the sum of the estimated costs for each unmet goal.

    # Assumptions
    - Navigation cost between adjacent visible waypoints is 1.
    - Sampling (soil/rock) costs 1 action (assuming an empty store is available or can be emptied).
    - Taking an image costs 1 action (assuming calibrated camera and correct location).
    - Calibrating a camera costs 1 action (assuming correct location and target visibility).
    - Communicating data costs 1 action (assuming correct location relative to lander).
    - If a sample or image is required by a goal but is not currently held by any rover
      and is no longer at its original location (for samples), it is considered
      unreachable, and a large penalty is applied.
    - The heuristic assumes the existence of suitable rovers, cameras, and waypoints
      based on static facts, but accounts for their current state (location, calibration).
    - Store capacity is simplified: assumes a store is either empty or full, and can be
      emptied with one drop action if full. The heuristic adds 1 action for sampling
      if the store is currently full.
    - The heuristic sums costs for individual unmet goals, ignoring potential resource
      conflicts or synergies between goals.

    # Heuristic Initialization
    - Identifies all objects (rovers, waypoints, etc.) by parsing initial state and static facts.
    - Parses goal conditions to identify required data/images.
    - Parses static facts to build navigation graphs for each rover based on `can_traverse` and `visible` facts.
    - Computes shortest path distances between all pairs of waypoints for each rover using BFS.
    - Stores static information: lander location, communication waypoints (visible from lander),
      rover capabilities (equipment, cameras, stores), camera properties (modes, calibration targets),
      and visibility of objectives/targets from waypoints.

    # Step-By-Step Thinking for Computing Heuristic
    1.  **Precomputation (`__init__`)**:
        - Collect all objects (rovers, waypoints, etc.) by iterating through initial state and static facts and identifying arguments based on predicate positions and common domain structure.
        - Build a navigation graph for each rover. Nodes are all identified waypoints. Edges exist between `w1` and `w2` if `(can_traverse R w1 w2)` and `(visible w1 w2)` are static facts.
        - Compute shortest path distances between all pairs of waypoints for each rover using BFS on their respective graphs. Store these distances.
        - Store static information: lander location, communication waypoints (visible from lander),
          rover capabilities (equipment: soil, rock, imaging), cameras on board rovers, modes supported by cameras,
          calibration targets for cameras, and waypoints from which objectives and calibration targets are visible.
        - Store which store belongs to which rover.

    2.  **Heuristic Calculation (`__call__`)**:
        - Initialize total heuristic cost to 0.
        - Extract current locations of all objects (especially rovers) from the current state.
        - Extract current status of samples (`have_soil_analysis`, `have_rock_analysis`),
          images (`have_image`), and camera calibration (`calibrated`) from the current state.
        - Extract current store status (`empty`, `full`) from the current state.

        - Iterate through each goal literal in `task.goals`:
            - If the goal literal is already true in the current state, add 0 to the total cost and continue to the next goal.
            - If the goal is `(communicated_soil_data W)`:
                - Estimate the minimum cost to achieve this goal. This involves either communicating an already held sample or sampling it first and then communicating.
                - If any rover `R` currently has `(have_soil_analysis R W)`: The cost is 1 (communicate) + navigation cost for `R` from its current location to the nearest communication waypoint. Find the minimum such cost over all rovers holding the sample.
                - If no rover has the sample but `(at_soil_sample W)` is true: The cost involves navigation for a soil-equipped rover `R` from its current location to `W`, plus 1 (sample action, potentially +1 for drop if store is full), plus navigation from `W` to the nearest communication waypoint, plus 1 (communicate action). Find the minimum such cost over all soil-equipped rovers and communication waypoints.
                - If the sample is neither held nor at `W`: Add a large penalty (e.g., 1000) as the goal might be unreachable.
                - Add the minimum estimated cost for this goal to the total heuristic.
            - If the goal is `(communicated_rock_data W)`:
                - Estimate cost similarly to soil data, using rock-specific predicates and rock-equipped rovers. Add the minimum estimated cost to the total.
            - If the goal is `(communicated_image_data O M)`:
                - Estimate the minimum cost to achieve this goal. This involves either communicating an already held image or taking the image first and then communicating.
                - If any rover `R` currently has `(have_image R O M)`: The cost is 1 (communicate) + navigation cost for `R` from its current location to the nearest communication waypoint. Find the minimum such cost over all rovers holding the image.
                - If no rover has the image: The cost involves finding a suitable imaging-equipped rover `R` with a camera `I` supporting mode `M`.
                    - If camera `I` on `R` is calibrated: Cost is navigation from `R`'s location to a waypoint `P` where `O` is visible, plus 1 (take_image action), plus navigation from `P` to the nearest communication waypoint, plus 1 (communicate action).
                    - If camera `I` on `R` needs calibration: Cost is navigation from `R`'s location to a waypoint `W` where `I`'s calibration target is visible, plus 1 (calibrate action), plus navigation from `W` to a waypoint `P` where `O` is visible, plus 1 (take_image action), plus navigation from `P` to the nearest communication waypoint, plus 1 (communicate action).
                    - Find the minimum total cost over all suitable rovers, cameras, and waypoints (W and P).
                - If no suitable rover/camera/waypoint combination exists or is reachable: Add a large penalty (e.g., 1000).
                - Add the minimum estimated cost for this goal to the total heuristic.

        - Return the total heuristic cost. If the cost is infinity (due to accumulating penalties), return a large finite number (e.g., 1000).
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static
        initial_state = task.initial_state

        # --- Collect all objects by type and waypoints ---
        self.rovers = set()
        self.waypoints = set()
        self.stores = set()
        self.cameras = set()
        self.modes = set()
        self.landers = set()
        self.objectives = set()

        # Use predicates to identify objects and types
        # Iterate through all facts in initial state and static facts
        for fact in static_facts | initial_state:
            parts = get_parts(fact)
            predicate = parts[0]

            # Infer types based on predicate structure (common in rovers domain)
            if predicate in ['at', 'at_lander']: # (at <obj> <waypoint>), (at_lander <lander> <waypoint>)
                 if len(parts) == 3:
                      obj, loc = parts[1:]
                      # Cannot reliably distinguish rover/lander/package here without type info
                      # Let's rely on other predicates to confirm types
                      self.waypoints.add(loc)
            elif predicate == 'can_traverse': # (can_traverse <rover> <waypoint> <waypoint>)
                 if len(parts) == 4:
                      r, w1, w2 = parts[1:]
                      self.rovers.add(r)
                      self.waypoints.add(w1)
                      self.waypoints.add(w2)
            elif predicate == 'visible': # (visible <waypoint> <waypoint>)
                 if len(parts) == 3:
                      w1, w2 = parts[1:]
                      self.waypoints.add(w1)
                      self.waypoints.add(w2)
            elif predicate in ['equipped_for_soil_analysis', 'equipped_for_rock_analysis', 'equipped_for_imaging']: # (<equip> <rover>)
                 if len(parts) == 2:
                      self.rovers.add(parts[1])
            elif predicate == 'empty' or predicate == 'full': # (<status> <store>)
                 if len(parts) == 2:
                      self.stores.add(parts[1])
            elif predicate in ['have_rock_analysis', 'have_soil_analysis']: # (<have> <rover> <waypoint>)
                 if len(parts) == 3:
                      self.rovers.add(parts[1])
                      self.waypoints.add(parts[2])
            elif predicate == 'calibrated': # (calibrated <camera> <rover>)
                 if len(parts) == 3:
                      self.cameras.add(parts[1])
                      self.rovers.add(parts[2])
            elif predicate == 'supports': # (supports <camera> <mode>)
                 if len(parts) == 3:
                      self.cameras.add(parts[1])
                      self.modes.add(parts[2])
                      self.camera_modes[parts[1]].add(parts[2])
            elif predicate == 'visible_from': # (visible_from <objective> <waypoint>)
                 if len(parts) == 3:
                      self.objectives.add(parts[1])
                      self.waypoints.add(parts[2])
                      self.objective_visible_from[parts[1]].add(parts[2])
            elif predicate in ['at_soil_sample', 'at_rock_sample']: # (<sample> <waypoint>)
                 if len(parts) == 2:
                      self.waypoints.add(parts[1])
            elif predicate == 'store_of': # (store_of <store> <rover>)
                 if len(parts) == 3:
                      self.stores.add(parts[1])
                      self.rovers.add(parts[2])
                      self.rover_stores[parts[2]] = parts[1] # rover -> store
            elif predicate == 'calibration_target': # (calibration_target <camera> <objective>)
                 if len(parts) == 3:
                      self.cameras.add(parts[1])
                      self.objectives.add(parts[2])
                      self.camera_calibration_target[parts[1]] = parts[2] # camera -> objective (target)
            elif predicate == 'on_board': # (on_board <camera> <rover>)
                 if len(parts) == 3:
                      self.cameras.add(parts[1])
                      self.rovers.add(parts[2])
                      self.rover_cameras[parts[2]].add(parts[1])
            elif predicate in ['communicated_soil_data', 'communicated_rock_data']: # (<comm> <waypoint>) - these are goals, not init/static facts usually
                 if len(parts) == 2:
                      self.waypoints.add(parts[1])
            elif predicate == 'communicated_image_data': # (<comm> <objective> <mode>) - these are goals
                 if len(parts) == 3:
                      self.objectives.add(parts[1])
                      self.modes.add(parts[2])
            elif predicate == 'at_lander': # (at_lander <lander> <waypoint>)
                 if len(parts) == 3:
                      self.landers.add(parts[1])
                      self.lander_location = parts[2]
                      self.waypoints.add(parts[2])


        # Build navigation graph edges from static facts (can_traverse and visible)
        can_traverse_edges = collections.defaultdict(list) # rover -> list of (w1, w2)
        visible_edges = collections.defaultdict(list) # w1 -> list of w2

        for fact in static_facts:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == 'can_traverse':
                r, w1, w2 = parts[1:]
                can_traverse_edges[r].append((w1, w2))
            elif predicate == 'visible':
                w1, w2 = parts[1:]
                visible_edges[w1].append(w2)


        # Now that we have all waypoints and the lander location, populate comm_waypoints
        self.comm_waypoints = set()
        if self.lander_location:
             for w1 in self.waypoints:
                 if self.lander_location in visible_edges.get(w1, []):
                     self.comm_waypoints.add(w1)
                 if w1 in visible_edges.get(self.lander_location, []): # Visible is symmetric
                     self.comm_waypoints.add(w1)


        # Build combined graph (can_traverse AND visible) for each rover
        rover_graphs = {}
        for rover in self.rovers:
            graph = collections.defaultdict(list)
            # Ensure all waypoints are nodes in the graph, even if isolated
            for w in self.waypoints:
                 graph[w] = [] # Initialize all waypoints as nodes

            for w1, w2 in can_traverse_edges.get(rover, []):
                if w2 in visible_edges.get(w1, []):
                    graph[w1].append(w2)
            rover_graphs[rover] = graph

        # Compute shortest paths for each rover
        self.rover_dist = {}
        for rover, graph in rover_graphs.items():
            self.rover_dist[rover] = {}
            for start_node in self.waypoints: # Compute distances from all known waypoints
                 self.rover_dist[rover][start_node] = bfs_shortest_path(graph, start_node)


        # Populate static equipment info
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'equipped_for_soil_analysis':
                self.rover_equipment[parts[1]].add('soil')
            elif parts[0] == 'equipped_for_rock_analysis':
                self.rover_equipment[parts[1]].add('rock')
            elif parts[0] == 'equipped_for_imaging':
                self.rover_equipment[parts[1]].add('imaging')

        # Populate static calibration target info
        for fact in static_facts:
             parts = get_parts(fact)
             if parts[0] == 'calibration_target':
                 cam, obj = parts[1:]
                 self.camera_calibration_target[cam] = obj
                 # Also populate calibration_target_visible_from based on objective_visible_from
                 if obj in self.objective_visible_from:
                      self.calibration_target_visible_from[obj] = self.objective_visible_from[obj]


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        total_heuristic = 0
        infinity = float('inf')
        UNREACHABLE_PENALTY = 1000 # Penalty for goals that seem unreachable

        # Helper to get object location from the current state
        def get_object_location(obj):
            for fact in state:
                if match(fact, 'at', obj, '*'):
                    return get_parts(fact)[2]
            return None # Object not found at any waypoint

        # Helper to check if a rover's store is empty in the current state
        def is_store_empty(rover):
            store = self.rover_stores.get(rover)
            if store:
                return f'(empty {store})' in state
            # If rover has no store or store_of fact is missing, assume it cannot sample
            return False

        # Helper to check if a rover's store is full in the current state
        def is_store_full(rover):
             store = self.rover_stores.get(rover)
             if store:
                 return f'(full {store})' in state
             # If rover has no store or store_of fact is missing, assume it cannot sample
             return False

        # Helper to get min navigation cost from start_w to any comm_w for a rover
        def min_nav_to_comm(rover, start_w):
            if rover not in self.rover_dist or start_w not in self.rover_dist[rover]:
                 return infinity # Rover or start_w not in precomputed graph

            min_dist = infinity
            # Ensure comm_waypoints is not empty and start_w has computed distances
            if not self.comm_waypoints or start_w not in self.rover_dist[rover]:
                 return infinity

            for comm_w in self.comm_waypoints:
                dist = self.rover_dist[rover][start_w].get(comm_w, infinity)
                min_dist = min(min_dist, dist)
            return min_dist

        # Helper to get nav cost between two waypoints for a rover
        def nav_cost(rover, start_w, end_w):
             if rover not in self.rover_dist or start_w not in self.rover_dist[rover]:
                  return infinity
             return self.rover_dist[rover][start_w].get(end_w, infinity)


        # Process each goal
        for goal in self.goals:
            if goal in state:
                continue # Goal already achieved

            parts = get_parts(goal)
            predicate = parts[0]

            if predicate == 'communicated_soil_data':
                w = parts[1]
                goal_cost = infinity

                # Option 1: Sample is already held by a rover
                best_comm_cost = infinity
                for r in self.rovers:
                    if f'(have_soil_analysis {r} {w})' in state:
                        loc_r = get_object_location(r)
                        if loc_r is not None:
                            cost = 1 + min_nav_to_comm(r, loc_r) # Communicate + Nav to comm point
                            best_comm_cost = min(best_comm_cost, cost)

                if best_comm_cost != infinity:
                    goal_cost = best_comm_cost
                else:
                    # Option 2: Sample is at the waypoint, need to sample and communicate
                    if f'(at_soil_sample {w})' in state:
                        min_sample_comm_cost = infinity
                        soil_rovers = [r for r, equip in self.rover_equipment.items() if 'soil' in equip]
                        for r in soil_rovers:
                            loc_r = get_object_location(r)
                            if loc_r is not None:
                                nav_to_w = nav_cost(r, loc_r, w)
                                if nav_to_w == infinity: continue

                                # Check if rover has a store and if it's empty or can be emptied
                                store = self.rover_stores.get(r)
                                if store is None: continue # Rover cannot sample without a store

                                sample_cost = 1 # Base cost for sample action
                                if is_store_full(r):
                                    sample_cost += 1 # Add cost for drop action if store is full

                                nav_w_to_comm = min_nav_to_comm(r, w)
                                if nav_w_to_comm == infinity: continue

                                comm_cost = 1
                                current_total_cost = nav_to_w + sample_cost + nav_w_to_comm + comm_cost
                                min_sample_comm_cost = min(min_sample_comm_cost, current_total_cost)

                        if min_sample_comm_cost != infinity:
                            goal_cost = min_sample_comm_cost
                        else:
                             # Sample at W, but no reachable equipped rover with store or comm point
                             goal_cost = UNREACHABLE_PENALTY
                    else:
                        # Sample not at W and no rover has it - likely unreachable
                        goal_cost = UNREACHABLE_PENALTY

                total_heuristic += goal_cost


            elif predicate == 'communicated_rock_data':
                w = parts[1]
                goal_cost = infinity

                # Option 1: Sample is already held by a rover
                best_comm_cost = infinity
                for r in self.rovers:
                    if f'(have_rock_analysis {r} {w})' in state:
                        loc_r = get_object_location(r)
                        if loc_r is not None:
                            cost = 1 + min_nav_to_comm(r, loc_r) # Communicate + Nav to comm point
                            best_comm_cost = min(best_comm_cost, cost)

                if best_comm_cost != infinity:
                    goal_cost = best_comm_cost
                else:
                    # Option 2: Sample is at the waypoint, need to sample and communicate
                    if f'(at_rock_sample {w})' in state:
                        min_sample_comm_cost = infinity
                        rock_rovers = [r for r, equip in self.rover_equipment.items() if 'rock' in equip]
                        for r in rock_rovers:
                            loc_r = get_object_location(r)
                            if loc_r is not None:
                                nav_to_w = nav_cost(r, loc_r, w)
                                if nav_to_w == infinity: continue

                                # Check if rover has a store and if it's empty or can be emptied
                                store = self.rover_stores.get(r)
                                if store is None: continue # Rover cannot sample without a store

                                sample_cost = 1 # Base cost for sample action
                                if is_store_full(r):
                                    sample_cost += 1 # Add cost for drop action if store is full

                                nav_w_to_comm = min_nav_to_comm(r, w)
                                if nav_w_to_comm == infinity: continue

                                comm_cost = 1
                                current_total_cost = nav_to_w + sample_cost + nav_w_to_comm + comm_cost
                                min_sample_comm_cost = min(min_sample_comm_cost, current_total_cost)

                        if min_sample_comm_cost != infinity:
                            goal_cost = min_sample_comm_cost
                        else:
                             goal_cost = UNREACHABLE_PENALTY
                    else:
                        # Sample not at W and no rover has it - likely unreachable
                        goal_cost = UNREACHABLE_PENALTY

                total_heuristic += goal_cost


            elif predicate == 'communicated_image_data':
                o, m = parts[1:]
                goal_cost = infinity

                # Option 1: Image is already held by a rover
                best_comm_cost = infinity
                for r in self.rovers:
                    if f'(have_image {r} {o} {m})' in state:
                        loc_r = get_object_location(r)
                        if loc_r is not None:
                            cost = 1 + min_nav_to_comm(r, loc_r) # Communicate + Nav to comm point
                            best_comm_cost = min(best_comm_cost, cost)

                if best_comm_cost != infinity:
                    goal_cost = best_comm_cost
                else:
                    # Option 2: Need to take the image and communicate
                    min_take_comm_cost = infinity
                    imaging_rovers = [r for r, equip in self.rover_equipment.items() if 'imaging' in equip]

                    for r in imaging_rovers:
                        loc_r = get_object_location(r)
                        if loc_r is None: continue # Rover location unknown

                        # Find cameras on this rover supporting the mode
                        suitable_cameras = [
                            cam for cam in self.rover_cameras.get(r, set())
                            if m in self.camera_modes.get(cam, set())
                        ]

                        for cam in suitable_cameras:
                            is_calibrated = f'(calibrated {cam} {r})' in state
                            cal_target_obj = self.camera_calibration_target.get(cam)

                            # Find waypoints where the objective O is visible from
                            obj_visible_wps = self.objective_visible_from.get(o, set())

                            if is_calibrated:
                                # Already calibrated, just need to navigate to a visible waypoint and take image
                                for p in obj_visible_wps:
                                    nav_to_p = nav_cost(r, loc_r, p)
                                    if nav_to_p == infinity: continue

                                    take_image_cost = 1
                                    nav_p_to_comm = min_nav_to_comm(r, p)
                                    if nav_p_to_comm == infinity: continue

                                    comm_cost = 1
                                    current_total_cost = nav_to_p + take_image_cost + nav_p_to_comm + comm_cost
                                    min_take_comm_cost = min(min_take_comm_cost, current_total_cost)
                            else:
                                # Need to calibrate first
                                if cal_target_obj is None: continue # Camera has no calibration target

                                # Find waypoints where the calibration target is visible from
                                cal_target_visible_wps = self.calibration_target_visible_from.get(cal_target_obj, set())

                                for w in cal_target_visible_wps:
                                    nav_to_w = nav_cost(r, loc_r, w)
                                    if nav_to_w == infinity: continue

                                    calibrate_cost = 1

                                    for p in obj_visible_wps:
                                        nav_w_to_p = nav_cost(r, w, p)
                                        if nav_w_to_p == infinity: continue

                                        take_image_cost = 1
                                        nav_p_to_comm = min_nav_to_comm(r, p)
                                        if nav_p_to_comm == infinity: continue

                                        comm_cost = 1
                                        current_total_cost = nav_to_w + calibrate_cost + nav_w_to_p + take_image_cost + nav_p_to_comm + comm_cost
                                        min_take_comm_cost = min(min_take_comm_cost, current_total_cost)

                    if min_take_comm_cost != infinity:
                        goal_cost = min_take_comm_cost
                    else:
                        # Need to take image, but no reachable equipped rover/camera/waypoint
                        goal_cost = UNREACHABLE_PENALTY

                total_heuristic += goal_cost

            # If a goal is not a communicated_* goal and not in state, it's likely an unsupported goal type.
            # Add a penalty.
            # else:
            #     total_heuristic += UNREACHABLE_PENALTY


        # Ensure heuristic is 0 only at goal state.
        # If total_heuristic is 0, it means all goals were found in the state.
        # The task.goal_reached check is the definitive one.
        # If task.goal_reached is False but total_heuristic is 0, it implies
        # the heuristic missed some unmet goal, which shouldn't happen if it
        # iterates through task.goals.
        # The only case where total_heuristic could be infinity is if UNREACHABLE_PENALTY
        # was added for at least one goal. We return a large finite number instead.
        if total_heuristic == infinity:
             return UNREACHABLE_PENALTY # Or a larger number if multiple penalties possible

        return total_heuristic
