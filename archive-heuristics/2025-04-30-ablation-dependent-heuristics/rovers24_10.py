import collections

def parse_fact(fact_string):
    """
    Parses a PDDL fact string into a predicate and its objects.
    e.g., '(at rover1 waypoint1)' -> ('at', ['rover1', 'waypoint1'])
    """
    # Removes parentheses and splits into parts
    parts = fact_string.strip('()').split()
    predicate = parts[0]
    objects = parts[1:]
    return predicate, objects

def bfs(graph, start_node):
    """
    Performs Breadth-First Search to find shortest distances from a start node
    in a graph represented as an adjacency list.
    Assumes graph keys are the nodes.
    """
    distances = {node: float('inf') for node in graph}
    if start_node not in graph:
        # If start_node is not in the graph keys, it's an isolated node
        # or doesn't exist in the defined graph structure.
        # Distance to itself is 0, others infinity.
        distances[start_node] = 0 # Add it to distances
        # No neighbors to explore, queue remains empty.
        return distances

    # Standard BFS if start_node is in the graph keys
    distances[start_node] = 0
    queue = collections.deque([start_node])
    while queue:
        current_node = queue.popleft()
        for neighbor in graph.get(current_node, []):
            if distances[neighbor] == float('inf'):
                distances[neighbor] = distances[current_node] + 1
                queue.append(neighbor)
    return distances

def compute_all_pairs_shortest_paths(graph):
    """
    Computes shortest paths from every node to every other node in the graph.
    Ensures all nodes present in the graph (keys or values) are considered as start nodes.
    """
    all_distances = {}
    # Collect all unique nodes from keys and values
    all_nodes = set(graph.keys())
    for neighbors in graph.values():
        all_nodes.update(neighbors)

    for start_node in all_nodes:
        # Need to ensure the graph passed to BFS includes all nodes from 'all_nodes'
        # so that distances dictionary in BFS is complete.
        # Create a temporary graph view for BFS that includes all nodes.
        full_graph_view = {node: graph.get(node, set()) for node in all_nodes}
        all_distances[start_node] = bfs(full_graph_view, start_node)
    return all_distances

def get_min_dist_to_set(distances_from_start, target_waypoints):
    """
    Finds the minimum distance from the start node (for which distances_from_start
    was computed) to any waypoint in the target_waypoints set.
    """
    min_d = float('inf')
    for target_w in target_waypoints:
        min_d = min(min_d, distances_from_start.get(target_w, float('inf')))
    return min_d


class roversHeuristic:
    """
    Domain-dependent heuristic for the Rovers domain.

    Summary:
    The heuristic estimates the remaining cost to reach the goal by summing
    the estimated costs for each individual unmet communication goal.
    For each unmet goal (communicating soil data, rock data, or image data),
    it checks if the required sample/image has already been collected/taken.
    - If collected/taken: The cost is estimated as the navigation cost for
      the rover holding the item to reach a communication waypoint plus 1
      action for communication.
    - If not collected/taken: The cost is estimated as the cost to collect/take
      the item (navigation to sample/image location, sampling/imaging actions,
      potentially dropping a full store, calibration for images) plus the
      navigation cost from the sample/image location to a communication
      waypoint plus 1 action for communication.
    Navigation costs are estimated using precomputed shortest path distances
    for each rover on its specific traversal graph. The minimum cost over
    suitable rovers and locations is used.

    Assumptions:
    - The planning instance is solvable (all necessary locations are reachable).
    - Traversal links are symmetric if visible links are symmetric (handled
      during graph construction).
    - The cost of dropping a sample is 1 action, needed if the store is full
      before sampling.
    - Calibration is needed before each 'take_image' action with that camera.
    - The heuristic sums costs for individual unmet goals, which might overestimate
      if actions contribute to multiple goals (e.g., navigation). This is a
      common simplification in non-admissible heuristics for efficiency.

    Heuristic Initialization:
    - Parses static facts to identify lander location, rover capabilities,
      store ownership, camera properties (on-board, supports, target),
      waypoint visibility, objective/target visibility from waypoints,
      and rover traversal capabilities.
    - Constructs a navigation graph for each rover based on 'can_traverse'
      and 'visible' facts.
    - Computes all-pairs shortest paths for each rover's navigation graph
      using BFS and stores them.
    - Identifies waypoints visible from the lander's location (communication waypoints).

    Step-By-Step Thinking for Computing Heuristic:
    1.  Check if the current state is a goal state. If yes, return 0.
    2.  Parse the current state to extract dynamic information: rover locations,
        samples/images held by rovers, store statuses, camera calibration statuses,
        and remaining samples at waypoints.
    3.  Identify the set of communication goals that are not yet achieved
        in the current state.
    4.  Initialize the total heuristic value `h = 0`.
    5.  For each unmet goal `g`:
        a.  If `g` is `(communicated_soil_data ?w)`:
            i.  Check if any rover `?r` currently holds `(have_soil_analysis ?r ?w)`.
            ii. If yes (sample collected): Find the rover `?r` holding the sample. Calculate the minimum navigation cost for `?r` from its current location to any communication waypoint (`min_dist(?r_loc, comm_waypoints)`). Add this cost + 1 (for `communicate_soil_data` action) to `h`.
            iii. If no (sample not collected): Find the minimum cost over all soil-equipped rovers `?r`. The cost for a rover `?r` is estimated as: navigation from `?r_loc` to `?w` (`dist(?r_loc, ?w)`) + 1 (for `sample_soil`) + (1 if `?r`'s store is full, for `drop`) + navigation from `?w` to a communication waypoint (`min_dist(?w, comm_waypoints)`) + 1 (for `communicate_soil_data`). Add the minimum such cost to `h`.
        b.  If `g` is `(communicated_rock_data ?w)`: Similar logic as soil data, using rock-equipped rovers, `at_rock_sample`, `have_rock_analysis`, and `sample_rock`.
        c.  If `g` is `(communicated_image_data ?o ?m)`:
            i.  Check if any rover `?r` currently holds `(have_image ?r ?o ?m)`.
            ii. If yes (image taken): Find the rover `?r` holding the image. Calculate the minimum navigation cost for `?r` from its current location to any communication waypoint (`min_dist(?r_loc, comm_waypoints)`). Add this cost + 1 (for `communicate_image_data` action) to `h`.
            iii. If no (image not taken): Find the minimum cost over all imaging-equipped rovers `?r` that have a camera `?i` supporting mode `?m`. For a suitable `?r` and `?i`, find the best calibration waypoint `?cal_loc` (visible from `?i`'s target `?t`) and the best image waypoint `?img_loc` (visible from `?o`). The cost for this `?r`, `?i`, `?cal_loc`, `?img_loc` is estimated as: navigation from `?r_loc` to `?cal_loc` (`dist(?r_loc, ?cal_loc)`) + 1 (for `calibrate`) + navigation from `?cal_loc` to `?img_loc` (`dist(?cal_loc, ?img_loc)`) + 1 (for `take_image`) + navigation from `?img_loc` to a communication waypoint (`min_dist(?img_loc, comm_waypoints)`) + 1 (for `communicate_image_data`). Add the minimum such cost over all choices to `h`.
    6.  Return the total heuristic value `h`. Handle cases where required waypoints are unreachable (should not happen in solvable instances, but return infinity if necessary). Use a large number like 1000 for infinity in practical terms.
    """
    def __init__(self, task):
        """
        Initializes the heuristic by parsing static facts and computing
        shortest paths for each rover.
        """
        self.task = task
        self._parse_static(task.static)
        self._compute_rover_distances()

    def _parse_static(self, static_facts):
        # Initialize static data structures
        self.lander_location = None
        self.rover_equipment = collections.defaultdict(set)
        self.rover_stores = {} # rover -> store
        self.store_rover = {} # store -> rover
        self.rover_cameras = collections.defaultdict(set)
        self.camera_supports = collections.defaultdict(set)
        self.camera_targets = {} # camera -> objective (target)
        self.objective_visible_from = collections.defaultdict(set) # objective -> {waypoint}
        self.target_visible_from = collections.defaultdict(set) # objective (target) -> {waypoint}
        self.rover_traversal_graphs = collections.defaultdict(lambda: collections.defaultdict(set)) # rover -> waypoint -> {neighbor_waypoint}
        self.all_waypoints = set()
        self.all_rovers = set()
        self.all_cameras = set()
        self.all_objectives = set()
        self.all_modes = set()
        self.all_stores = set()
        self.all_landers = set()

        visible_graph = collections.defaultdict(set)

        for fact_str in static_facts:
            pred, objs = parse_fact(fact_str)
            if pred == 'at_lander':
                self.lander_location = objs[1]
                self.all_landers.add(objs[0])
                self.all_waypoints.add(objs[1])
            elif pred == 'equipped_for_soil_analysis':
                self.rover_equipment[objs[0]].add('soil')
                self.all_rovers.add(objs[0])
            elif pred == 'equipped_for_rock_analysis':
                self.rover_equipment[objs[0]].add('rock')
                self.all_rovers.add(objs[0])
            elif pred == 'equipped_for_imaging':
                self.rover_equipment[objs[0]].add('imaging')
                self.all_rovers.add(objs[0])
            elif pred == 'store_of':
                self.store_rover[objs[0]] = objs[1]
                self.rover_stores[objs[1]] = objs[0]
                self.all_stores.add(objs[0])
                self.all_rovers.add(objs[1])
            elif pred == 'visible':
                w1, w2 = objs
                # Assuming visible is symmetric
                visible_graph[w1].add(w2)
                visible_graph[w2].add(w1)
                self.all_waypoints.add(w1)
                self.all_waypoints.add(w2)
            elif pred == 'can_traverse':
                r, w1, w2 = objs
                # We build the potential traversal graph here.
                # It will be filtered by visible links later.
                self.rover_traversal_graphs[r][w1].add(w2)
                self.rover_traversal_graphs[r][w2].add(w1) # Assuming traversal is symmetric if visible is symmetric
                self.all_rovers.add(r)
                self.all_waypoints.add(w1)
                self.all_waypoints.add(w2)
            elif pred == 'on_board':
                c, r = objs
                self.rover_cameras[r].add(c)
                self.all_cameras.add(c)
                self.all_rovers.add(r)
            elif pred == 'supports':
                c, m = objs
                self.camera_supports[c].add(m)
                self.all_cameras.add(c)
                self.all_modes.add(m)
            elif pred == 'calibration_target':
                c, o = objs
                self.camera_targets[c] = o
                self.all_cameras.add(c)
                self.all_objectives.add(o)
            elif pred == 'visible_from':
                o, w = objs
                # This predicate is used for both objective visibility (for take_image)
                # and calibration target visibility (for calibrate).
                # The calibration target is an objective.
                self.objective_visible_from[o].add(w)
                # Add to target_visible_from if this objective is a calibration target for any camera
                # We can refine this check later if needed, but adding all visible_from O W where O is an objective is safe.
                self.target_visible_from[o].add(w)
                self.all_objectives.add(o)
                self.all_waypoints.add(w)

        # Refine traversal graphs using visible facts
        # A rover can traverse w1->w2 if can_traverse w1->w2 AND visible w1->w2
        for rover, graph in list(self.rover_traversal_graphs.items()): # Use list to modify while iterating
             filtered_graph = collections.defaultdict(set)
             # Ensure all waypoints this rover *could* traverse from/to are in the filtered graph keys
             all_possible_nodes = set(graph.keys()) | set(w for neighbors in graph.values() for w in neighbors)
             for w1 in all_possible_nodes:
                 # If w1 was in the original graph, check its neighbors
                 if w1 in graph:
                     for w2 in graph[w1]:
                         if w2 in visible_graph.get(w1, set()):
                             filtered_graph[w1].add(w2)
                 # If w1 was not in the original graph keys but was a neighbor, it's an isolated node for this rover initially
                 # It will be added to the filtered_graph keys when BFS is run on all_waypoints

             self.rover_traversal_graphs[rover] = filtered_graph


        # Identify communication waypoints
        self.comm_waypoints = visible_graph.get(self.lander_location, set())


    def _compute_rover_distances(self):
        """
        Computes all-pairs shortest paths for each rover's traversal graph.
        """
        self.rover_distances = {}
        for rover, graph in self.rover_traversal_graphs.items():
            # Ensure all waypoints in the problem are considered as potential start/end nodes for BFS
            # even if they are isolated in the rover's specific graph.
            all_nodes_in_problem = set(self.all_waypoints)

            # Create a graph structure that includes all waypoints from the problem,
            # even if a specific rover cannot traverse to/from them. Distances will be infinity.
            full_graph_nodes = {node: graph.get(node, set()) for node in all_nodes_in_problem}

            self.rover_distances[rover] = compute_all_pairs_shortest_paths(full_graph_nodes)

    def __call__(self, state):
        """
        Computes the domain-dependent heuristic value for the given state.
        """
        # Use a large number for unreachable costs
        infinity = 1000

        if self.task.goal_reached(state):
            return 0

        # Parse dynamic state facts
        rover_locations = {}
        rover_soil_samples = collections.defaultdict(set)
        rover_rock_samples = collections.defaultdict(set)
        rover_images = collections.defaultdict(set) # rover -> {(objective, mode)}
        store_status = {} # store -> 'empty' or 'full'
        camera_calibrated = {} # (camera, rover) -> bool

        for fact_str in state:
            pred, objs = parse_fact(fact_str)
            if pred == 'at':
                rover_locations[objs[0]] = objs[1]
            elif pred == 'have_soil_analysis':
                rover_soil_samples[objs[0]].add(objs[1])
            elif pred == 'have_rock_analysis':
                rover_rock_samples[objs[0]].add(objs[1])
            elif pred == 'have_image':
                rover_images[objs[0]].add((objs[1], objs[2])) # (objective, mode)
            elif pred == 'empty':
                store_status[objs[0]] = 'empty'
            elif pred == 'full':
                store_status[objs[0]] = 'full'
            elif pred == 'calibrated':
                camera_calibrated[(objs[0], objs[1])] = True
            # We don't need at_soil_sample/at_rock_sample from state for this heuristic logic,
            # as we only care if the sample is *had* by a rover or *communicated*.
            # The presence of samples at waypoints is static/initial state info relevant for *applicability* of sample actions,
            # but the heuristic estimates cost based on *needing* the sample, not its current ground truth location.
            # The PDDL shows sample actions consume the at_soil_sample/at_rock_sample fact, so it's dynamic,
            # but the heuristic doesn't need to track *where* the sample is, only *if* it's needed and not yet collected.

        h = 0

        # Identify unmet communication goals
        unmet_goals = self.task.goals - state

        # Process each unmet goal
        for goal_fact_str in unmet_goals:
            pred, objs = parse_fact(goal_fact_str)

            if pred == 'communicated_soil_data':
                waypoint = objs[0]
                # Check if sample is already collected by any rover
                sample_collected = any(waypoint in samples for samples in rover_soil_samples.values())

                if sample_collected:
                    # Cost to communicate collected sample
                    min_comm_cost = infinity
                    for rover, samples in rover_soil_samples.items():
                        if waypoint in samples:
                            rover_loc = rover_locations.get(rover)
                            # Ensure rover_loc is valid and rover has distance info from that location
                            if rover_loc and rover in self.rover_distances and rover_loc in self.rover_distances[rover]:
                                dist_to_comm = get_min_dist_to_set(self.rover_distances[rover][rover_loc], self.comm_waypoints)
                                if dist_to_comm != float('inf'):
                                     min_comm_cost = min(min_comm_cost, dist_to_comm + 1) # +1 for communicate action
                    h += min_comm_cost if min_comm_cost != infinity else infinity # Add cost only if reachable
                else:
                    # Cost to collect sample and then communicate
                    min_total_cost = infinity
                    # Find best equipped rover
                    for rover in self.all_rovers:
                        if 'soil' in self.rover_equipment.get(rover, set()):
                            rover_loc = rover_locations.get(rover)
                            # Ensure rover_loc is valid and rover has distance info from that location
                            if rover_loc and rover in self.rover_distances and rover_loc in self.rover_distances[rover]:
                                # Cost to navigate to sample location + sample action + (drop if needed)
                                dist_to_sample = self.rover_distances[rover][rover_loc].get(waypoint, float('inf'))
                                if dist_to_sample != float('inf'):
                                    sample_action_cost = 1 # sample_soil
                                    # Check if rover has a store and if it's full
                                    rover_store = self.rover_stores.get(rover)
                                    drop_cost = 1 if rover_store and store_status.get(rover_store, 'empty') == 'full' else 0

                                    # Cost to navigate from sample location to communication waypoint + communicate action
                                    # Need distances from the sample waypoint itself
                                    if waypoint in self.rover_distances[rover]:
                                        dist_sample_to_comm = get_min_dist_to_set(self.rover_distances[rover][waypoint], self.comm_waypoints)
                                        if dist_sample_to_comm != float('inf'):
                                             cost_to_communicate = dist_sample_to_comm + 1 # communicate_soil_data

                                             total_cost_for_rover = dist_to_sample + sample_action_cost + drop_cost + cost_to_communicate
                                             min_total_cost = min(min_total_cost, total_cost_for_rover)
                                    # else: waypoint is unreachable from itself? Should not happen if dist_to_sample is finite.

                    h += min_total_cost if min_total_cost != infinity else infinity # Add cost only if reachable

            elif pred == 'communicated_rock_data':
                waypoint = objs[0]
                # Check if sample is already collected by any rover
                sample_collected = any(waypoint in samples for samples in rover_rock_samples.values())

                if sample_collected:
                    # Cost to communicate collected sample
                    min_comm_cost = infinity
                    for rover, samples in rover_rock_samples.items():
                        if waypoint in samples:
                            rover_loc = rover_locations.get(rover)
                            # Ensure rover_loc is valid and rover has distance info from that location
                            if rover_loc and rover in self.rover_distances and rover_loc in self.rover_distances[rover]:
                                dist_to_comm = get_min_dist_to_set(self.rover_distances[rover][rover_loc], self.comm_waypoints)
                                if dist_to_comm != float('inf'):
                                     min_comm_cost = min(min_comm_cost, dist_to_comm + 1) # +1 for communicate action
                    h += min_comm_cost if min_comm_cost != infinity else infinity # Add cost only if reachable
                else:
                    # Cost to collect sample and then communicate
                    min_total_cost = infinity
                    # Find best equipped rover
                    for rover in self.all_rovers:
                        if 'rock' in self.rover_equipment.get(rover, set()):
                            rover_loc = rover_locations.get(rover)
                            # Ensure rover_loc is valid and rover has distance info from that location
                            if rover_loc and rover in self.rover_distances and rover_loc in self.rover_distances[rover]:
                                # Cost to navigate to sample location + sample action + (drop if needed)
                                dist_to_sample = self.rover_distances[rover][rover_loc].get(waypoint, float('inf'))
                                if dist_to_sample != float('inf'):
                                    sample_action_cost = 1 # sample_rock
                                    # Check if rover has a store and if it's full
                                    rover_store = self.rover_stores.get(rover)
                                    drop_cost = 1 if rover_store and store_status.get(rover_store, 'empty') == 'full' else 0

                                    # Cost to navigate from sample location to communication waypoint + communicate action
                                    # Need distances from the sample waypoint itself
                                    if waypoint in self.rover_distances[rover]:
                                        dist_sample_to_comm = get_min_dist_to_set(self.rover_distances[rover][waypoint], self.comm_waypoints)
                                        if dist_sample_to_comm != float('inf'):
                                             cost_to_communicate = dist_sample_to_comm + 1 # communicate_rock_data

                                             total_cost_for_rover = dist_to_sample + sample_action_cost + drop_cost + cost_to_communicate
                                             min_total_cost = min(min_total_cost, total_cost_for_rover)
                                    # else: waypoint is unreachable from itself? Should not happen if dist_to_sample is finite.

                    h += min_total_cost if min_total_cost != infinity else infinity # Add cost only if reachable

            elif pred == 'communicated_image_data':
                objective, mode = objs
                # Check if image is already taken by any rover
                image_taken = any((objective, mode) in images for images in rover_images.values())

                if image_taken:
                    # Cost to communicate taken image
                    min_comm_cost = infinity
                    for rover, images in rover_images.items():
                        if (objective, mode) in images:
                            rover_loc = rover_locations.get(rover)
                            # Ensure rover_loc is valid and rover has distance info from that location
                            if rover_loc and rover in self.rover_distances and rover_loc in self.rover_distances[rover]:
                                dist_to_comm = get_min_dist_to_set(self.rover_distances[rover][rover_loc], self.comm_waypoints)
                                if dist_to_comm != float('inf'):
                                     min_comm_cost = min(min_comm_cost, dist_to_comm + 1) # +1 for communicate action
                    h += min_comm_cost if min_comm_cost != infinity else infinity # Add cost only if reachable
                else:
                    # Cost to take image and then communicate
                    min_total_cost = infinity
                    # Find best equipped rover with suitable camera
                    for rover in self.all_rovers:
                        if 'imaging' in self.rover_equipment.get(rover, set()):
                            rover_loc = rover_locations.get(rover)
                            # Ensure rover_loc is valid and rover has distance info from that location
                            if rover_loc and rover in self.rover_distances and rover_loc in self.rover_distances[rover]:
                                for camera in self.rover_cameras.get(rover, set()):
                                    if mode in self.camera_supports.get(camera, set()):
                                        calibration_target = self.camera_targets.get(camera)
                                        if calibration_target:
                                            # Find best calibration waypoint
                                            cal_waypoints = self.target_visible_from.get(calibration_target, set())
                                            if cal_waypoints:
                                                min_cost_via_cal = infinity
                                                for cal_loc in cal_waypoints:
                                                    dist_to_cal = self.rover_distances[rover][rover_loc].get(cal_loc, float('inf'))
                                                    if dist_to_cal != float('inf'):
                                                        # Find best image waypoint
                                                        img_waypoints = self.objective_visible_from.get(objective, set())
                                                        if img_waypoints:
                                                            min_cost_via_img = infinity
                                                            for img_loc in img_waypoints:
                                                                # Need distances from the calibration waypoint itself
                                                                if cal_loc in self.rover_distances[rover]:
                                                                    dist_cal_to_img = self.rover_distances[rover][cal_loc].get(img_loc, float('inf'))
                                                                    if dist_cal_to_img != float('inf'):
                                                                        # Cost to take image: nav to cal + calibrate + nav to img + take_image
                                                                        cost_to_take = dist_to_cal + 1 + dist_cal_to_img + 1

                                                                        # Cost to communicate: nav from img to comm + communicate
                                                                        # Need distances from the image waypoint itself
                                                                        if img_loc in self.rover_distances[rover]:
                                                                            dist_img_to_comm = get_min_dist_to_set(self.rover_distances[rover][img_loc], self.comm_waypoints)
                                                                            if dist_img_to_comm != float('inf'):
                                                                                 cost_to_communicate = dist_img_to_comm + 1

                                                                                 total_cost_for_path = cost_to_take + cost_to_communicate
                                                                                 min_cost_via_img = min(min_cost_via_img, total_cost_for_path)
                                                                        # else: img_loc unreachable from itself? Should not happen if dist_cal_to_img is finite.
                                                            min_cost_via_cal = min(min_cost_via_cal, min_cost_via_img)
                                                min_total_cost = min(min_total_cost, min_cost_via_cal)

                    h += min_total_cost if min_total_cost != infinity else infinity # Add cost only if reachable

        # Return the total heuristic value. Cap at infinity if any goal was unreachable.
        return h if h != float('inf') else infinity
