import collections

# Helper function to parse PDDL fact strings
def parse_fact(fact_str):
    """
    Parses a PDDL fact string like '(predicate arg1 arg2)' into a tuple
    (predicate, [arg1, arg2]).
    """
    # Removes leading/trailing parens and splits by space
    parts = fact_str.strip('()').split()
    if not parts:
        return None, [] # Handle empty string case
    return parts[0], parts[1:]

# Helper function for BFS
def bfs(graph, start_node):
    """
    Performs Breadth-First Search to find shortest distances from start_node
    to all reachable nodes in the graph.
    Graph is represented as a dictionary: node -> set of neighbors.
    Returns a dictionary: node -> distance.
    """
    # Initialize distances for all nodes in the graph
    distances = {node: float('inf') for node in graph}

    # If start_node is not a valid node in the graph, return the inf distances
    if start_node not in graph:
         return distances

    distances[start_node] = 0
    queue = collections.deque([start_node])

    while queue:
        current_node = queue.popleft()

        # Check if current_node has neighbors in the graph dict
        if current_node in graph:
            for neighbor in graph[current_node]:
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)
    return distances

# Helper function to compute all-pairs shortest paths
def compute_all_pairs_shortest_paths(graph, nodes):
    """
    Computes shortest distances between all pairs of nodes in the graph
    using BFS from each node.
    Returns a dictionary: start_node -> (target_node -> distance).
    """
    all_distances = {}
    # Compute distances starting from every node in the provided list/set
    for start_node in nodes:
        all_distances[start_node] = bfs(graph, start_node)
    return all_distances


class roversHeuristic:
    """
    Domain-dependent heuristic for the Rovers domain.

    Summary:
    This heuristic estimates the cost to reach a goal state by summing up
    the estimated costs for each unachieved goal fact independently. The
    estimated cost for a single goal fact is the minimum number of actions
    and travel steps required for a suitable rover (or rover/camera pair)
    to achieve that specific goal, assuming other goals do not interfere.
    Travel costs are estimated using precomputed shortest path distances
    on the waypoint graph for each rover.

    Assumptions:
    - The heuristic assumes that achieving different goal facts can be done
      independently by potentially different rovers/cameras, and sums their
      individual minimum costs. This is a relaxation and makes the heuristic
      non-admissible.
    - For image goals, a specific sequence of steps is assumed: travel to a
      calibration waypoint, calibrate, travel to an imaging waypoint, take
      the image, travel to a communication waypoint, and communicate. The
      travel cost is the sum of shortest path distances between these points.
    - The heuristic assumes that solvable problems have finite paths between
      necessary waypoints for capable rovers. If no path exists according
      to the precomputed graph, the cost is considered infinite for that
      specific path segment. If the minimum cost for a goal remains infinite,
      it does not contribute a finite value to the heuristic sum.
    - The heuristic returns 0 if and only if the state is a goal state.
      For non-goal states where the calculated sum is 0, it returns 1 to
      distinguish them from goal states.

    Heuristic Initialization:
    Upon initialization with a Task object, the heuristic precomputes static
    information from the task's static facts and initial state:
    1.  Identifies all waypoints and rovers present in the problem by parsing
        relevant facts in the static information and initial state.
    2.  Parses static facts to determine:
        - Lander location.
        - Which rovers are equipped for soil analysis, rock analysis, and imaging.
        - Which store belongs to which rover.
        - Which waypoints have soil or rock samples.
        - Which camera is on which rover, which modes each camera supports,
          and the calibration target objective for each camera.
        - Which waypoints are visible from each objective.
    3.  Builds a navigation graph for each rover. An edge exists from waypoint
        `w1` to `w2` for a rover if the static facts include both
        `(can_traverse rover w1 w2)` and `(visible w1 w2)`. The graph includes
        all identified waypoints as nodes, even if they have no edges.
    4.  Computes all-pairs shortest path distances for each rover on its
        respective navigation graph using BFS.
    5.  Identifies the set of communication waypoints (waypoints visible from
        the lander's location).
    6.  Identifies the set of calibration waypoints for each camera (waypoints
        visible from the camera's calibration target objective).
    7.  Identifies the set of imaging waypoints for each objective (waypoints
        visible from the objective).

    Step-By-Step Thinking for Computing Heuristic:
    For a given state:
    1.  Check if the state is a goal state using `task.goal_reached(state)`. If yes, return 0.
    2.  Initialize the heuristic value `h` to 0.
    3.  Convert the state (frozenset) to a set for efficient fact lookups.
    4.  Iterate through each goal fact specified in `task.goals`.
    5.  If a goal fact is already true in the current state, skip it.
    6.  If a goal fact is not true, estimate the minimum cost to achieve it by
        considering all suitable rovers (and cameras for image goals) and
        calculating the cost for each, taking the minimum:
        -   **For `(communicated_soil_data ?w)`:**
            -   Iterate through all rovers equipped for soil analysis.
            -   If the rover already has the sample `(have_soil_analysis ?r ?w)`, the cost is the shortest distance from its current location to any communication waypoint + 1 (communicate action).
            -   If the rover does not have the sample but `?w` has a soil sample (`(at_soil_sample ?w)` is static), the cost is the shortest distance from its current location to `?w` + 1 (sample action) + shortest distance from `?w` to any communication waypoint + 1 (communicate action). Add 1 more action if the rover's store is full (for `drop`).
            -   The minimum of these costs across all suitable rovers is the contribution for this goal.
        -   **For `(communicated_rock_data ?w)`:** Analogous to soil data, using rock-equipped rovers and rock samples.
        -   **For `(communicated_image_data ?o ?m)`:**
            -   Iterate through all rovers equipped for imaging that have a camera supporting mode `?m`.
            -   For each such rover/camera pair, check if the rover already has the image `(have_image ?r ?o ?m)`.
                -   If yes, the cost is the shortest distance from its current location to any communication waypoint + 1 (communicate action).
            -   If the rover does not have the image, the cost is estimated as the minimum travel cost for the sequence: rover's current location -> a calibration waypoint for the camera -> an imaging waypoint for the objective -> a communication waypoint, plus 3 actions (calibrate, take_image, communicate). The minimum travel cost is calculated using precomputed distances over all combinations of intermediate and final waypoints from the respective sets (calibration waypoints, imaging waypoints, communication waypoints).
            -   The minimum of these costs across all suitable rover/camera pairs is the contribution for this goal.
    7.  Add the minimum estimated cost for the unachieved goal fact (if finite) to `h`. If the minimum cost is infinite (no path found by the simple logic), this goal contributes 0 to the sum, effectively assuming it's not achievable via the modeled steps or by any capable rover.
    8.  After iterating through all goal facts, if the total calculated `h` is 0 but the state is not a goal state (checked in step 1), return 1 to ensure non-goal states have a positive heuristic value. Otherwise, return `h`.
    """
    def __init__(self, task):
        self.task = task
        self.static_facts = task.static
        self.goals = task.goals

        # Precompute static information
        self.lander_location = None
        self.soil_equipped_rovers = set()
        self.rock_equipped_rovers = set()
        self.imaging_equipped_rovers = set()
        self.store_to_rover = {} # store -> rover
        self.waypoint_has_soil = set() # waypoint
        self.waypoint_has_rock = set() # waypoint
        self.camera_on_rover = {} # camera -> rover
        self.camera_supports_mode = collections.defaultdict(set) # camera -> set of modes
        self.camera_cal_target = {} # camera -> objective
        self.objective_visible_from = collections.defaultdict(set) # objective -> set of waypoints

        # Collect all waypoints and rovers for graph building
        all_waypoints_set = set()
        all_rovers_set = set()
        visible_waypoints_raw = collections.defaultdict(set) # waypoint -> set of visible neighbors

        # Collect objects and raw graph info from static facts
        for fact_str in self.static_facts:
            pred, args = parse_fact(fact_str)
            if pred == 'at_lander':
                self.lander_location = args[1]
                all_waypoints_set.add(args[1])
            elif pred == 'equipped_for_soil_analysis':
                self.soil_equipped_rovers.add(args[0])
                all_rovers_set.add(args[0])
            elif pred == 'equipped_for_rock_analysis':
                self.rock_equipped_rovers.add(args[0])
                all_rovers_set.add(args[0])
            elif pred == 'equipped_for_imaging':
                self.imaging_equipped_rovers.add(args[0])
                all_rovers_set.add(args[0])
            elif pred == 'store_of':
                self.store_to_rover[args[0]] = args[1]
            elif pred == 'at_soil_sample':
                self.waypoint_has_soil.add(args[0])
                all_waypoints_set.add(args[0])
            elif pred == 'at_rock_sample':
                self.waypoint_has_rock.add(args[0])
                all_waypoints_set.add(args[0])
            elif pred == 'on_board':
                self.camera_on_rover[args[0]] = args[1]
            elif pred == 'supports':
                self.camera_supports_mode[args[0]].add(args[1])
            elif pred == 'calibration_target':
                self.camera_cal_target[args[0]] = args[1]
            elif pred == 'visible_from':
                self.objective_visible_from[args[0]].add(args[1])
                all_waypoints_set.add(args[1]) # Waypoints visible from objectives are important
            elif pred == 'can_traverse':
                rover, w1, w2 = args
                all_rovers_set.add(rover)
                all_waypoints_set.add(w1)
                all_waypoints_set.add(w2)
            elif pred == 'visible':
                 w1, w2 = args
                 visible_waypoints_raw[w1].add(w2)
                 visible_waypoints_raw[w2].add(w1) # Visible is symmetric
                 all_waypoints_set.add(w1)
                 all_waypoints_set.add(w2)
            # Add objects from initial state as well (especially rover locations)
        for fact_str in self.task.initial_state:
             pred, args = parse_fact(fact_str)
             if pred == 'at':
                  all_rovers_set.add(args[0])
                  all_waypoints_set.add(args[1])
             # Add other object types if needed, but waypoints and rovers are key for graph/distances

        self.all_waypoints = list(all_waypoints_set) # Store as list
        self.all_rovers = list(all_rovers_set)

        # Build rover graphs based on explicit can_traverse and visible facts
        self.rover_graph = collections.defaultdict(lambda: collections.defaultdict(set))
        for rover in self.all_rovers:
             # Initialize graph with all relevant waypoints as nodes
             for w in self.all_waypoints:
                  self.rover_graph[rover][w] = set()

        for fact_str in self.static_facts:
             pred, args = parse_fact(fact_str)
             if pred == 'can_traverse':
                  rover, w1, w2 = args
                  # An edge exists if (can_traverse rover w1 w2) AND (visible w1 w2)
                  if w2 in visible_waypoints_raw.get(w1, set()):
                       self.rover_graph[rover][w1].add(w2)


        # Compute all-pairs shortest paths for each rover
        self.rover_distances = {} # rover -> (waypoint -> (waypoint -> distance))
        for rover in self.all_rovers:
             # Compute distances for all waypoints that are keys in the rover's graph
             # This ensures we compute distances from/to all waypoints the rover might be at or need to visit.
             self.rover_distances[rover] = compute_all_pairs_shortest_paths(self.rover_graph[rover], list(self.rover_graph[rover].keys()))


        # Identify communication waypoints
        # A waypoint is a communication waypoint if it is visible from the lander location.
        self.comm_waypoint_set = visible_waypoints_raw.get(self.lander_location, set())

        # Identify calibration waypoints for each camera
        self.camera_cal_waypoints = collections.defaultdict(set) # camera -> set of waypoints
        for camera, target_objective in self.camera_cal_target.items():
             # Calibration waypoint for camera i is any waypoint visible from its target objective t
             self.camera_cal_waypoints[camera] = self.objective_visible_from.get(target_objective, set())

        # Identify imaging waypoints for each objective
        # Imaging waypoint for objective o is any waypoint visible from objective o
        self.objective_imaging_waypoints = self.objective_visible_from # Same data structure

        # Store mapping from rover to its store
        self.rover_to_store = {v: k for k, v in self.store_to_rover.items()}


    def get_rover_location(self, state_facts, rover):
        """Finds the current waypoint of the given rover in the state."""
        for fact_str in state_facts:
            pred, args = parse_fact(fact_str)
            if pred == 'at' and args[0] == rover:
                return args[1]
        return None # Rover location not found (should not happen in valid state)

    def get_min_dist(self, rover, start_waypoint, target_waypoints):
        """
        Calculates the minimum shortest distance from start_waypoint to any
        waypoint in target_waypoints for the given rover.
        Returns float('inf') if start_waypoint is None, target_waypoints is empty,
        or no target waypoint is reachable.
        """
        if start_waypoint is None or not target_waypoints:
            return float('inf')

        min_d = float('inf')
        # Get precomputed distances from start_waypoint for this rover
        # Ensure start_waypoint is a key in the distances dict
        start_distances = self.rover_distances.get(rover, {}).get(start_waypoint, {})

        # Iterate through target waypoints and find the minimum distance
        for target_w in target_waypoints:
            # Ensure target_w is a key in the distances dict from start_waypoint
            min_d = min(min_d, start_distances.get(target_w, float('inf')))

        return min_d

    def get_min_travel_cost_sequence(self, rover, start_w, waypoints1, waypoints2, waypoints3):
        """
        Calculates min_{w1 in waypoints1, w2 in waypoints2, w3 in waypoints3}
        (dist(start_w, w1) + dist(w1, w2) + dist(w2, w3)) for the given rover.
        Returns float('inf') if any waypoint set is empty or no path exists.
        """
        if start_w is None or not waypoints1 or not waypoints2 or not waypoints3:
             return float('inf')

        min_total_dist = float('inf')
        rover_dists = self.rover_distances.get(rover, {})
        start_dists = rover_dists.get(start_w, {})

        # Iterate through all combinations of intermediate waypoints
        for w1 in waypoints1:
            # Ensure w1 is a key in the distances dict from start_w
            dist_start_w1 = start_dists.get(w1, float('inf'))
            if dist_start_w1 == float('inf'): continue

            w1_dists = rover_dists.get(w1, {})
            for w2 in waypoints2:
                # Ensure w2 is a key in the distances dict from w1
                dist_w1_w2 = w1_dists.get(w2, float('inf'))
                if dist_w1_w2 == float('inf'): continue

                w2_dists = rover_dists.get(w2, {})
                for w3 in waypoints3:
                    # Ensure w3 is a key in the distances dict from w2
                    dist_w2_w3 = w2_dists.get(w3, float('inf'))
                    if dist_w2_w3 == float('inf'): continue

                    min_total_dist = min(min_total_dist, dist_start_w1 + dist_w1_w2 + dist_w2_w3)

        return min_total_dist


    def __call__(self, state):
        """
        Computes the heuristic value for the given state.
        """
        # If goal is reached, heuristic is 0
        if self.task.goal_reached(state):
            return 0

        h = 0
        state_facts = set(state) # Convert frozenset to set for faster lookups

        # Check each goal fact
        for goal_fact_str in self.goals:
            if goal_fact_str in state_facts:
                continue # Goal already achieved

            pred, args = parse_fact(goal_fact_str)

            if pred == 'communicated_soil_data':
                waypoint = args[0]
                min_goal_cost = float('inf')

                # Find a rover that can achieve this goal
                for rover in self.soil_equipped_rovers:
                    rover_at_sample_w = f'(have_soil_analysis {rover} {waypoint})'

                    if rover_at_sample_w in state_facts:
                        # Rover already has the sample, just needs to communicate
                        current_w = self.get_rover_location(state_facts, rover)
                        dist_to_comm = self.get_min_dist(rover, current_w, self.comm_waypoint_set)
                        if dist_to_comm != float('inf'):
                            min_goal_cost = min(min_goal_cost, dist_to_comm + 1) # +1 for communicate

                    elif waypoint in self.waypoint_has_soil:
                        # Rover needs to sample and then communicate
                        current_w = self.get_rover_location(state_facts, rover)
                        dist_to_sample = self.get_min_dist(rover, current_w, {waypoint})
                        dist_sample_to_comm = self.get_min_dist(rover, waypoint, self.comm_waypoint_set)

                        if dist_to_sample != float('inf') and dist_sample_to_comm != float('inf'):
                            # Cost to sample: travel to waypoint + 1 (sample)
                            cost_sample = dist_to_sample + 1

                            # Cost to communicate: travel from sample waypoint to comm waypoint + 1 (communicate)
                            cost_comm = dist_sample_to_comm + 1

                            # Check if drop is needed before sampling
                            store = self.rover_to_store.get(rover)
                            cost_drop = 0
                            if store and f'(full {store})' in state_facts:
                                cost_drop = 1 # +1 for drop action

                            min_goal_cost = min(min_goal_cost, cost_sample + cost_comm + cost_drop)

                if min_goal_cost != float('inf'):
                    h += min_goal_cost


            elif pred == 'communicated_rock_data':
                waypoint = args[0]
                min_goal_cost = float('inf')

                # Find a rover that can achieve this goal
                for rover in self.rock_equipped_rovers:
                    rover_at_sample_w = f'(have_rock_analysis {rover} {waypoint})'

                    if rover_at_sample_w in state_facts:
                        # Rover already has the sample, just needs to communicate
                        current_w = self.get_rover_location(state_facts, rover)
                        dist_to_comm = self.get_min_dist(rover, current_w, self.comm_waypoint_set)
                        if dist_to_comm != float('inf'):
                            min_goal_cost = min(min_goal_cost, dist_to_comm + 1) # +1 for communicate

                    elif waypoint in self.waypoint_has_rock:
                        # Rover needs to sample and then communicate
                        current_w = self.get_rover_location(state_facts, rover)
                        dist_to_sample = self.get_min_dist(rover, current_w, {waypoint})
                        dist_sample_to_comm = self.get_min_dist(rover, waypoint, self.comm_waypoint_set)

                        if dist_to_sample != float('inf') and dist_sample_to_comm != float('inf'):
                            # Cost to sample: travel to waypoint + 1 (sample)
                            cost_sample = dist_to_sample + 1

                            # Cost to communicate: travel from sample waypoint to comm waypoint + 1 (communicate)
                            cost_comm = dist_sample_to_comm + 1

                            # Check if drop is needed before sampling
                            store = self.rover_to_store.get(rover)
                            cost_drop = 0
                            if store and f'(full {store})' in state_facts:
                                cost_drop = 1 # +1 for drop action

                            min_goal_cost = min(min_goal_cost, cost_sample + cost_comm + cost_drop)

                if min_goal_cost != float('inf'):
                    h += min_goal_cost


            elif pred == 'communicated_image_data':
                objective, mode = args
                min_goal_cost = float('inf')

                # Find a rover/camera that can achieve this goal
                for rover in self.imaging_equipped_rovers:
                    # Find cameras on this rover that support the mode
                    suitable_cameras = [
                        cam for cam, r in self.camera_on_rover.items() if r == rover and mode in self.camera_supports_mode.get(cam, set())
                    ]

                    for camera in suitable_cameras:
                        rover_has_image = f'(have_image {rover} {objective} {mode})'

                        if rover_has_image in state_facts:
                            # Rover already has the image, just needs to communicate
                            current_w = self.get_rover_location(state_facts, rover)
                            dist_to_comm = self.get_min_dist(rover, current_w, self.comm_waypoint_set)
                            if dist_to_comm != float('inf'):
                                min_goal_cost = min(min_goal_cost, dist_to_comm + 1) # +1 for communicate

                        else:
                            # Rover needs to take image and then communicate
                            current_w = self.get_rover_location(state_facts, rover)

                            cal_target = self.camera_cal_target.get(camera)
                            cal_waypoints = self.camera_cal_waypoints.get(camera, set())
                            img_waypoints = self.objective_imaging_waypoints.get(objective, set())

                            if cal_target and cal_waypoints and img_waypoints:
                                # Calculate min travel cost: current -> cal_w -> img_p -> comm_w
                                travel_cost = self.get_min_travel_cost_sequence(
                                    rover, current_w, cal_waypoints, img_waypoints, self.comm_waypoint_set
                                )

                                if travel_cost != float('inf'):
                                    # Actions: 1 (calibrate) + 1 (take_image) + 1 (communicate)
                                    action_cost = 3
                                    min_goal_cost = min(min_goal_cost, travel_cost + action_cost)

                if min_goal_cost != float('inf'):
                    h += min_goal_cost

        # If h is still 0 but goal is not reached, return a small positive value
        # to distinguish from goal states.
        if h == 0 and not self.task.goal_reached(state):
             return 1

        return h

