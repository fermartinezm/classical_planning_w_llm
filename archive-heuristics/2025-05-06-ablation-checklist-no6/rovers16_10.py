from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import defaultdict, deque

# Helper functions to parse PDDL facts
def parse_fact(fact_string):
    """Parses a PDDL fact string into a tuple (predicate, arg1, ...)."""
    # Remove parentheses and split by spaces
    parts = fact_string[1:-1].split()
    return tuple(parts)

def match_fact(fact_tuple, *pattern):
    """Checks if a parsed fact matches a given pattern (with wildcards)."""
    if len(fact_tuple) != len(pattern):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(fact_tuple, pattern))

# Helper functions for navigation graph and shortest paths
def build_nav_graph(static_facts, rover_name, visible_waypoints):
    """
    Builds the navigation graph for a specific rover based on can_traverse
    and visible predicates.
    """
    graph = defaultdict(set)
    can_traverse_edges = set()
    for fact_string in static_facts:
        fact = parse_fact(fact_string)
        if match_fact(fact, "can_traverse", rover_name, "*", "*"):
            can_traverse_edges.add(fact)

    for fact in can_traverse_edges:
        _, r, wp1, wp2 = fact
        # Check if visible condition also holds for the navigate action
        if wp1 in visible_waypoints and wp2 in visible_waypoints[wp1]:
             graph[wp1].add(wp2)
    return graph

def bfs(graph, start_node):
    """Computes shortest path distances from start_node to all reachable nodes."""
    distances = {start_node: 0}
    queue = deque([start_node])
    while queue:
        current_node = queue.popleft()
        current_dist = distances[current_node]
        # Ensure current_node is in graph keys before iterating neighbors
        if current_node in graph:
            for neighbor in graph[current_node]:
                if neighbor not in distances:
                    distances[neighbor] = current_dist + 1
                    queue.append(neighbor)
    return distances

def precompute_shortest_paths(nav_graphs):
    """Precomputes shortest paths for all rovers and all start nodes."""
    all_shortest_paths = {}
    for rover, graph in nav_graphs.items():
        rover_paths = {}
        # Collect all waypoints involved in can_traverse for this rover
        all_waypoints = set(graph.keys())
        for neighbors in graph.values():
             all_waypoints.update(neighbors)

        for start_wp in all_waypoints:
             rover_paths[start_wp] = bfs(graph, start_wp)
        all_shortest_paths[rover] = rover_paths
    return all_shortest_paths

def get_distance(shortest_paths, rover, start_wp, end_wp):
    """Gets the shortest distance, returns infinity if unreachable."""
    if rover not in shortest_paths or start_wp not in shortest_paths[rover] or end_wp not in shortest_paths[rover][start_wp]:
        return float('inf')
    return shortest_paths[rover][start_wp][end_wp]


class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    Estimates the cost to reach the goal by summing the estimated costs
    for each unachieved goal fact (communicating data).

    The cost for each unachieved communication goal is estimated based on
    whether the required data (sample or image) is already collected or
    needs to be collected, plus the estimated navigation costs using
    precomputed shortest paths. It assumes a single rover performs all
    steps for a specific data item goal (collect/take, navigate, communicate).
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static
        initial_state = task.initial_state # Need initial state for initial samples

        # --- Extract Static Information ---
        self.lander_location = None
        self.visible_waypoints = defaultdict(set) # wp1 -> {wp2, wp3, ...}
        self.lander_visible_waypoints = set()
        self.rover_capabilities = defaultdict(set) # rover -> {soil, rock, imaging}
        self.store_to_rover = {} # store -> rover
        self.camera_info = defaultdict(dict) # camera -> {on_board_rover: r, supported_modes: {m1, m2}, calibration_target: t}
        self.objective_image_waypoints = defaultdict(set) # obj -> {wp1, wp2} (waypoints where obj is visible from)
        self.calibration_target_waypoints = defaultdict(set) # obj (target) -> {wp1, wp2} (waypoints where target is visible from)
        self.rover_nav_graphs = {} # rover -> graph (adj list)
        self.all_rovers = set() # Collect all rover names

        # Process static facts
        for fact_string in static_facts:
            fact = parse_fact(fact_string)
            if match_fact(fact, "at_lander", "*", "*"):
                _, lander, wp = fact
                self.lander_location = wp
            elif match_fact(fact, "visible", "*", "*"):
                _, wp1, wp2 = fact
                self.visible_waypoints[wp1].add(wp2)
            elif match_fact(fact, "equipped_for_soil_analysis", "rover*"):
                _, rover = fact
                self.rover_capabilities[rover].add("soil")
                self.all_rovers.add(rover)
            elif match_fact(fact, "equipped_for_rock_analysis", "rover*"):
                _, rover = fact
                self.rover_capabilities[rover].add("rock")
                self.all_rovers.add(rover)
            elif match_fact(fact, "equipped_for_imaging", "rover*"):
                _, rover = fact
                self.rover_capabilities[rover].add("imaging")
                self.all_rovers.add(rover)
            elif match_fact(fact, "store_of", "rover*store", "rover*"):
                _, store, rover = fact
                self.store_to_rover[store] = rover
                self.all_rovers.add(rover)
            elif match_fact(fact, "supports", "camera*", "*"):
                _, camera, mode = fact
                self.camera_info[camera].setdefault('supported_modes', set()).add(mode)
            elif match_fact(fact, "calibration_target", "camera*", "*"):
                _, camera, target_obj = fact
                self.camera_info[camera]['calibration_target'] = target_obj
            elif match_fact(fact, "on_board", "camera*", "rover*"):
                _, camera, rover = fact
                self.camera_info[camera]['on_board_rover'] = rover
                self.all_rovers.add(rover)
            elif match_fact(fact, "visible_from", "*", "*"):
                _, obj, wp = fact
                # An objective can be an image target or a calibration target
                self.objective_image_waypoints[obj].add(wp)
                self.calibration_target_waypoints[obj].add(wp)

        # Determine lander visible waypoints
        if self.lander_location and self.lander_location in self.visible_waypoints:
             self.lander_visible_waypoints = self.visible_waypoints[self.lander_location]

        # Build navigation graphs and precompute shortest paths for each rover
        for rover in self.all_rovers:
             self.rover_nav_graphs[rover] = build_nav_graph(static_facts, rover, self.visible_waypoints)

        self.rover_shortest_paths = precompute_shortest_paths(self.rover_nav_graphs)

        # --- Extract Goal Information ---
        self.goal_data = set() # {(type, arg1, ...)}
        for goal_string in self.goals:
            goal = parse_fact(goal_string)
            if match_fact(goal, "communicated_soil_data", "*"):
                _, wp = goal
                self.goal_data.add(("soil", wp))
            elif match_fact(goal, "communicated_rock_data", "*"):
                _, wp = goal
                self.goal_data.add(("rock", wp))
            elif match_fact(goal, "communicated_image_data", "*", "*"):
                _, obj, mode = goal
                self.goal_data.add(("image", obj, mode))


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # --- Extract Current State Information ---
        current_rover_locations = {}
        rover_has_sample = defaultdict(set) # Stores ("soil", waypoint) or ("rock", waypoint) tuples
        rover_has_image = defaultdict(set) # Stores (objective, mode) tuples
        rover_store_full = {} # Stores boolean for each rover
        rover_calibrated_cameras = defaultdict(set) # rover -> {camera1, camera2}
        communicated_data = set() # Stores (type, ...) tuples achieved in this state
        current_soil_samples_at_wp = set() # Waypoints where soil samples are currently located
        current_rock_samples_at_wp = set() # Waypoints where rock samples are currently located


        for fact_string in state:
            fact = parse_fact(fact_string)
            if match_fact(fact, "at", "rover*", "*"):
                _, rover, wp = fact
                current_rover_locations[rover] = wp
            elif match_fact(fact, "have_soil_analysis", "rover*", "*"):
                _, rover, wp = fact
                rover_has_sample[rover].add(("soil", wp))
            elif match_fact(fact, "have_rock_analysis", "rover*", "*"):
                _, rover, wp = fact
                rover_has_sample[rover].add(("rock", wp))
            elif match_fact(fact, "have_image", "rover*", "*", "*"):
                _, rover, obj, mode = fact
                rover_has_image[rover].add((obj, mode))
            elif match_fact(fact, "full", "rover*store"):
                _, store = fact
                rover = self.store_to_rover.get(store)
                if rover:
                    rover_store_full[rover] = True
            elif match_fact(fact, "calibrated", "camera*", "rover*"):
                _, camera, rover = fact
                rover_calibrated_cameras[rover].add(camera)
            elif match_fact(fact, "communicated_soil_data", "*"):
                _, wp = fact
                communicated_data.add(("soil", wp))
            elif match_fact(fact, "communicated_rock_data", "*"):
                _, wp = fact
                communicated_data.add(("rock", wp))
            elif match_fact(fact, "communicated_image_data", "*", "*"):
                _, obj, mode = fact
                communicated_data.add(("image", obj, mode))
            elif match_fact(fact, "at_soil_sample", "*"):
                _, wp = fact
                current_soil_samples_at_wp.add(wp)
            elif match_fact(fact, "at_rock_sample", "*"):
                _, wp = fact
                current_rock_samples_at_wp.add(wp)


        # Ensure all rovers known from init are in relevant state dicts (defaulting)
        for rover in self.all_rovers:
             # Rovers should always have an @at fact if they exist, but default defensively
             current_rover_locations.setdefault(rover, None)
             rover_store_full.setdefault(rover, False)


        # --- Compute Heuristic Cost ---
        total_cost = 0

        for goal_item in self.goal_data:
            if goal_item in communicated_data:
                continue # Goal already achieved

            data_type = goal_item[0]

            if data_type == "soil":
                _, wp = goal_item
                # Check if any rover has the sample
                rover_with_sample = next((r for r, samples in rover_has_sample.items() if ("soil", wp) in samples), None)

                if rover_with_sample:
                    # Cost to communicate: Nav from current loc to lander-visible + Communicate
                    min_comm_cost = float('inf')
                    rover_loc = current_rover_locations.get(rover_with_sample)
                    if rover_loc:
                         min_nav_to_lander = min((get_distance(self.rover_shortest_paths, rover_with_sample, rover_loc, lander_wp)
                                                  for lander_wp in self.lander_visible_waypoints), default=float('inf'))
                         if min_nav_to_lander != float('inf'):
                             min_comm_cost = min_nav_to_lander + 1 # +1 for communicate action

                    if min_comm_cost != float('inf'):
                         total_cost += min_comm_cost
                    else:
                         # Cannot reach lander from current location with sample.
                         # Add a penalty.
                         total_cost += 10 # Penalty for difficult communication

                else: # Sample needs collecting and communicating
                    # Check if sample is still at the waypoint
                    if wp not in current_soil_samples_at_wp:
                        # Sample is gone from its initial location and no rover has it.
                        # Goal likely impossible via standard sample action.
                        total_cost += 100 # High penalty
                    else:
                        # Sample is still at the waypoint, needs collecting
                        # Cost = Nav to sample + Sample (+ Drop if store full) + Nav from sample loc to lander-visible + Communicate
                        min_total_cost_for_sample = float('inf')

                        # Find suitable rovers (equipped for soil)
                        suitable_rovers = [r for r, caps in self.rover_capabilities.items() if "soil" in caps]

                        for rover in suitable_rovers:
                             rover_loc = current_rover_locations.get(rover)
                             if not rover_loc: continue # Should not happen if rover exists

                             # Cost to get sample
                             nav_to_sample_cost = get_distance(self.rover_shortest_paths, rover, rover_loc, wp)
                             if nav_to_sample_cost == float('inf'): continue # Cannot reach sample location

                             sample_action_cost = 1 # sample_soil action
                             drop_action_cost = 1 if rover_store_full.get(rover, False) else 0 # drop action if needed

                             get_sample_cost = nav_to_sample_cost + sample_action_cost + drop_action_cost

                             # Cost to communicate sample from sample location
                             min_nav_from_sample_to_lander = min((get_distance(self.rover_shortest_paths, rover, wp, lander_wp)
                                                                  for lander_wp in self.lander_visible_waypoints), default=float('inf'))
                             if min_nav_from_sample_to_lander == float('inf'): continue # Cannot reach lander from sample location

                             comm_action_cost = 1 # communicate_soil_data action
                             comm_cost_from_sample_loc = min_nav_from_sample_to_lander + comm_action_cost

                             total_cost_for_this_rover = get_sample_cost + comm_cost_from_sample_loc
                             min_total_cost_for_sample = min(min_total_cost_for_sample, total_cost_for_this_rover)

                        if min_total_cost_for_sample != float('inf'):
                             total_cost += min_total_cost_for_sample
                        else:
                             # Sample is at waypoint, but no equipped rover can reach it and lander.
                             total_cost += 100 # High penalty

            elif data_type == "rock":
                _, wp = goal_item
                # Check if any rover has the sample
                rover_with_sample = next((r for r, samples in rover_has_sample.items() if ("rock", wp) in samples), None)

                if rover_with_sample:
                    # Cost to communicate: Nav from current loc to lander-visible + Communicate
                    min_comm_cost = float('inf')
                    rover_loc = current_rover_locations.get(rover_with_sample)
                    if rover_loc:
                         min_nav_to_lander = min((get_distance(self.rover_shortest_paths, rover_with_sample, rover_loc, lander_wp)
                                                  for lander_wp in self.lander_visible_waypoints), default=float('inf'))
                         if min_nav_to_lander != float('inf'):
                             min_comm_cost = min_nav_to_lander + 1

                    if min_comm_cost != float('inf'):
                         total_cost += min_comm_cost
                    else:
                         total_cost += 10 # Penalty

                else: # Sample needs collecting and communicating
                    # Check if sample is still at the waypoint
                    if wp not in current_rock_samples_at_wp:
                        # Sample is gone from its initial location and no rover has it.
                        total_cost += 100 # High penalty
                    else:
                        # Sample is still at the waypoint, needs collecting
                        # Cost = Nav to sample + Sample (+ Drop if store full) + Nav from sample loc to lander-visible + Communicate
                        min_total_cost_for_sample = float('inf')
                        suitable_rovers = [r for r, caps in self.rover_capabilities.items() if "rock" in caps]

                        for rover in suitable_rovers:
                             rover_loc = current_rover_locations.get(rover)
                             if not rover_loc: continue

                             nav_to_sample_cost = get_distance(self.rover_shortest_paths, rover, rover_loc, wp)
                             if nav_to_sample_cost == float('inf'): continue

                             sample_action_cost = 1
                             drop_action_cost = 1 if rover_store_full.get(rover, False) else 0

                             get_sample_cost = nav_to_sample_cost + sample_action_cost + drop_action_cost

                             min_nav_from_sample_to_lander = min((get_distance(self.rover_shortest_paths, rover, wp, lander_wp)
                                                                  for lander_wp in self.lander_visible_waypoints), default=float('inf'))
                             if min_nav_from_sample_to_lander == float('inf'): continue

                             comm_action_cost = 1
                             comm_cost_from_sample_loc = min_nav_from_sample_to_lander + comm_action_cost

                             total_cost_for_this_rover = get_sample_cost + comm_cost_from_sample_loc
                             min_total_cost_for_sample = min(min_total_cost_for_sample, total_cost_for_this_rover)

                        if min_total_cost_for_sample != float('inf'):
                             total_cost += min_total_cost_for_sample
                        else:
                             total_cost += 100 # High penalty

            elif data_type == "image":
                _, obj, mode = goal_item
                # Check if any rover has the image
                rover_with_image = next((r for r, images in rover_has_image.items() if (obj, mode) in images), None)

                if rover_with_image:
                    # Cost to communicate: Nav from current loc to lander-visible + Communicate
                    min_comm_cost = float('inf')
                    rover_loc = current_rover_locations.get(rover_with_image)
                    if rover_loc:
                         min_nav_to_lander = min((get_distance(self.rover_shortest_paths, rover_with_image, rover_loc, lander_wp)
                                                  for lander_wp in self.lander_visible_waypoints), default=float('inf'))
                         if min_nav_to_lander != float('inf'):
                             min_comm_cost = min_nav_to_lander + 1

                    if min_comm_cost != float('inf'):
                         total_cost += min_comm_cost
                    else:
                         total_cost += 10 # Penalty

                else: # Image needs taking and communicating
                    # Cost = Nav to cal_wp + Calibrate + Nav from cal_wp to image_wp + Take_image + Nav from image_wp to lander-visible + Communicate
                    min_total_cost_for_image = float('inf')

                    # Find suitable rovers (equipped for imaging)
                    suitable_rovers = [r for r, caps in self.rover_capabilities.items() if "imaging" in caps]

                    for rover in suitable_rovers:
                         rover_loc = current_rover_locations.get(rover)
                         if not rover_loc: continue

                         # Find suitable cameras on this rover supporting the mode
                         suitable_cameras = [
                             cam for cam, info in self.camera_info.items()
                             if info.get('on_board_rover') == rover and mode in info.get('supported_modes', set())
                         ]

                         for camera in suitable_cameras:
                             cal_target = self.camera_info[camera].get('calibration_target')
                             if not cal_target: continue # Camera must have a calibration target

                             # Find suitable image waypoints for this objective
                             image_wps = self.objective_image_waypoints.get(obj, set())
                             # Find suitable calibration waypoints for this camera's target
                             cal_wps = self.calibration_target_waypoints.get(cal_target, set())

                             for image_wp in image_wps:
                                 for cal_wp in cal_wps:
                                     # Cost to get image
                                     nav_to_cal_cost = get_distance(self.rover_shortest_paths, rover, rover_loc, cal_wp)
                                     if nav_to_cal_cost == float('inf'): continue

                                     cal_action_cost = 1 # calibrate action

                                     nav_cal_to_image_cost = get_distance(self.rover_shortest_paths, rover, cal_wp, image_wp)
                                     if nav_cal_to_image_cost == float('inf'): continue

                                     take_image_action_cost = 1 # take_image action

                                     get_image_total_cost = nav_to_cal_cost + cal_action_cost + nav_cal_to_image_cost + take_image_action_cost

                                     # Cost to communicate image from image location
                                     min_nav_from_image_to_lander = min((get_distance(self.rover_shortest_paths, rover, image_wp, lander_wp)
                                                                         for lander_wp in self.lander_visible_waypoints), default=float('inf'))
                                     if min_nav_from_image_to_lander == float('inf'): continue

                                     comm_action_cost = 1 # communicate_image_data action
                                     comm_cost_from_image_loc = min_nav_from_image_to_lander + comm_action_cost

                                     total_cost_for_this_path = get_image_total_cost + comm_cost_from_image_loc
                                     min_total_cost_for_image = min(min_total_cost_for_image, total_cost_for_this_path)

                    if min_total_cost_for_image != float('inf'):
                         total_cost += min_total_cost_for_image
                    else:
                         # No suitable combination allows achieving this image goal.
                         total_cost += 100 # High penalty


        # Ensure heuristic is 0 for goal state
        # Check if all goal items are in communicated_data
        all_goals_communicated = all(item in communicated_data for item in self.goal_data)

        if all_goals_communicated:
            return 0

        # Handle cases where total_cost might still be infinity if penalties weren't added
        if total_cost == float('inf'):
             return 1000 # Return a large finite number for unsolvable/very hard states

        return total_cost
