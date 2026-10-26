# Assume heuristic_base is available in the environment
# from heuristics.heuristic_base import Heuristic

# Define a dummy Heuristic base class for standalone testing if needed
# Remove this block when integrating into the planner environment
class Heuristic:
    def __init__(self, task):
        pass
    def __call__(self, node):
        pass
# End of dummy class


from fnmatch import fnmatch
from collections import deque, defaultdict
import math # Use math.inf for infinity

# Helper functions for parsing PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact string or malformed fact
    if not fact or not isinstance(fact, str) or fact[0] != '(' or fact[-1] != ')':
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

# BFS implementation for shortest paths
def bfs(graph, start_node):
    """
    Performs BFS to find shortest distances from start_node to all other nodes.

    Args:
        graph: Adjacency list representation {node: {neighbor1, neighbor2, ...}}
        start_node: The starting node for BFS.

    Returns:
        A dictionary {node: distance} containing shortest distances.
        Returns infinity for unreachable nodes.
    """
    distances = {node: math.inf for node in graph}
    if start_node not in graph:
         # Start node might not be in the graph if it's an object of a different type
         # or if the graph is empty. Return distances only for nodes in the graph.
         return distances

    distances[start_node] = 0
    queue = deque([start_node])

    while queue:
        current_node = queue.popleft()

        if current_node not in graph:
             continue # Should not happen if start_node was in graph

        for neighbor in graph[current_node]:
            if distances[neighbor] == math.inf:
                distances[neighbor] = distances[current_node] + 1
                queue.append(neighbor)

    return distances

# Helper to find min distance between a set of start waypoints and a set of end waypoints
def min_dist(shortest_paths_map, start_wps, end_wps):
    """
    Finds the minimum shortest path distance between any waypoint in start_wps
    and any waypoint in end_wps using precomputed shortest paths.

    Args:
        shortest_paths_map: The result of running BFS from all nodes {start_wp: {end_wp: dist}}
        start_wps: A set of starting waypoints.
        end_wps: A set of ending waypoints.

    Returns:
        The minimum distance, or math.inf if no path exists between any pair.
    """
    min_d = math.inf
    if not start_wps or not end_wps:
        return min_d

    for sw in start_wps:
        if sw in shortest_paths_map:
            for ew in end_wps:
                if ew in shortest_paths_map[sw]:
                    min_d = min(min_d, shortest_paths_map[sw][ew])
    return min_d


class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve all
    goal conditions. It sums the estimated costs for each unachieved goal
    fact. The cost for a single goal fact (e.g., communicated soil data for
    waypoint W) is estimated as the minimum number of actions required
    for any suitable rover to collect the necessary data (sample or image)
    and then navigate to a communication point to transmit it. Navigation
    costs are estimated using shortest path distances on the traversal graph
    for each rover.

    # Assumptions
    - The heuristic assumes that all goal conditions are achievable in the
      given problem instance.
    - For sampling goals, it assumes the sample is either still at the
      waypoint or already collected by a rover that can communicate it.
      It does not handle scenarios where a sample was collected but the
      data was subsequently lost (e.g., by dropping a full store containing
      the only sample data).
    - For imaging goals, it assumes a suitable camera and calibration target
      exist and are reachable, and that imaging and calibration waypoints
      are reachable.
    - The cost of actions is assumed to be 1.
    - The heuristic is non-admissible; it may overestimate the cost.

    # Heuristic Initialization
    The heuristic initializes by parsing the static facts and initial state
    to build data structures representing:
    - Goal facts.
    - Lander location.
    - Rover capabilities (soil, rock, imaging).
    - Rover stores.
    - Cameras on board rovers, supported modes, and calibration targets.
    - Objective visibility from waypoints.
    - Waypoint visibility (for communication points).
    - Initial locations of soil and rock samples.
    - Rover traversal graphs.
    It then precomputes all-pairs shortest paths for each rover's traversal
    graph using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic calculates the sum of estimated costs
    for each goal fact that is not yet satisfied in the state.

    For each unachieved goal fact G:
    1.  **Identify the type of goal:** Is it `communicated_soil_data`,
        `communicated_rock_data`, or `communicated_image_data`?
    2.  **Find communication points:** Determine the lander location and
        identify all waypoints visible from the lander. These are potential
        communication points.
    3.  **Estimate cost based on goal type:**
        *   **Soil/Rock Data Goal (for waypoint W):**
            *   If any rover R already has the analysis data for W:
                *   The cost is the minimum navigation distance for R from its
                    current location to any communication point, plus 1 action
                    for communication. Minimize this cost over all rovers that
                    have the data.
            *   Else (no rover has the data):
                *   If the sample is still at waypoint W:
                    *   Find rovers R equipped for this analysis type.
                    *   For each such rover R, calculate the cost to:
                        *   Navigate R to W (distance).
                        *   (If R's store is full, add 1 for drop action).
                        *   Sample at W (1 action).
                        *   Navigate R from W to any communication point (minimum distance).
                        *   Communicate (1 action).
                    *   The cost for this goal is the minimum of this total cost
                        over all suitable rovers.
                *   If the sample is no longer at W and no rover has the data,
                    the heuristic assumes this path is not possible from this state
                    (or assigns a very high cost, effectively infinity).
        *   **Image Data Goal (for objective O and mode M):**
            *   If any rover R already has the image data for (O, M):
                *   The cost is the minimum navigation distance for R from its
                    current location to any communication point, plus 1 action
                    for communication. Minimize this cost over all rovers that
                    have the image.
            *   Else (no rover has the image):
                *   Find rovers R equipped for imaging that have a camera I
                    supporting mode M.
                *   Find calibration waypoints (visible from I's calibration target)
                    and imaging waypoints (visible from O).
                *   For each suitable rover R and camera I, calculate the cost to:
                    *   Navigate R to a calibration waypoint (minimum distance).
                    *   Calibrate camera I (1 action).
                    *   Navigate R from the calibration waypoint to an imaging
                        waypoint (minimum distance between the sets of waypoints).
                    *   Take the image (1 action).
                    *   Navigate R from the imaging waypoint to a communication
                        point (minimum distance between the sets of waypoints).
                    *   Communicate (1 action).
                *   The cost for this goal is the minimum of this total cost
                    over all suitable rover/camera combinations and choices
                    of calibration/imaging/communication waypoints. Handle cases
                    where necessary waypoints are unreachable.
    4.  **Sum Costs:** The total heuristic value is the sum of the minimum
        estimated costs for all unachieved goal facts.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goal_facts = task.goals

        # Data structures for static information
        self.lander_location = {}  # lander -> wp
        self.rovers = set()
        self.waypoint_names = set() # Use a different name to avoid conflict with state parsing
        self.objectives = set()
        self.modes = set()
        self.cameras = set()
        self.stores = set()

        self.rover_capabilities = defaultdict(set)  # rover -> {soil, rock, imaging}
        self.rover_stores = {}  # rover -> store
        self.rover_cameras = defaultdict(list)  # rover -> [camera]
        self.camera_modes = defaultdict(set)  # camera -> {mode}
        self.camera_calibration_target = {}  # camera -> objective
        self.objective_visible_from = defaultdict(set)  # objective -> {wp}
        self.waypoint_visibility = defaultdict(set)  # wp -> {wp}
        self.rover_traversal_graph = defaultdict(lambda: defaultdict(set)) # rover -> {wp -> {neighbor_wp}}

        self.initial_soil_samples = set()  # {wp}
        self.initial_rock_samples = set()  # {wp}

        # Collect all object names and types from static, initial, and goals
        # This is a heuristic way to find objects given the input format
        all_facts = set(task.initial_state) | set(task.goals) | set(task.static)
        for fact in all_facts:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            pred = parts[0]
            args = parts[1:]

            # Infer object types based on predicates and argument positions/prefixes
            if pred in ['at', 'can_traverse', 'equipped_for_soil_analysis',
                        'equipped_for_rock_analysis', 'equipped_for_imaging',
                        'have_rock_analysis', 'have_soil_analysis', 'have_image',
                        'calibrated', 'on_board']:
                if len(args) > 0: self.rovers.add(args[0])
            if pred in ['at', 'at_lander', 'at_soil_sample', 'at_rock_sample',
                        'visible_from', 'can_traverse', 'visible',
                        'sample_soil', 'sample_rock', 'calibrate', 'take_image',
                        'communicate_soil_data', 'communicate_rock_data',
                        'communicate_image_data']:
                 if len(args) > 0:
                     # Waypoints appear in various positions, collect all args from relevant predicates
                     for arg in args:
                         # Simple check: waypoints often start with 'waypoint'
                         if arg.startswith('waypoint'):
                             self.waypoint_names.add(arg)
            if pred in ['empty', 'full', 'store_of', 'sample_soil', 'sample_rock', 'drop']:
                 if len(args) > 0: self.stores.add(args[0])
            if pred in ['calibrated', 'supports', 'calibration_target', 'on_board', 'calibrate', 'take_image']:
                 if len(args) > 0: self.cameras.add(args[0])
            if pred in ['supports', 'take_image', 'communicate_image_data']:
                 if len(args) > 0: self.modes.add(args[-1]) # Mode is usually the last arg
            if pred in ['at_lander', 'communicate_soil_data', 'communicate_rock_data', 'communicate_image_data']:
                 if len(args) > 0: # Lander is usually the first arg after predicate
                     # Simple check: landers often start with 'general' or 'lander'
                     if args[0].startswith('general') or args[0].startswith('lander'):
                         pass # We only need lander location, not the object itself in a set
            if pred in ['calibration_target', 'visible_from', 'have_image', 'communicate_image_data']:
                 if len(args) > 0: self.objectives.add(args[0]) # Objective is usually the first arg after predicate


        # Parse static facts
        for fact in task.static:
            parts = get_parts(fact)
            if not parts: continue
            pred = parts[0]
            args = parts[1:]

            if match(fact, 'at_lander', '*', '*'):
                self.lander_location[args[0]] = args[1]
            elif match(fact, 'equipped_for_soil_analysis', '*'):
                self.rover_capabilities[args[0]].add('soil')
            elif match(fact, 'equipped_for_rock_analysis', '*'):
                self.rover_capabilities[args[0]].add('rock')
            elif match(fact, 'equipped_for_imaging', '*'):
                self.rover_capabilities[args[0]].add('imaging')
            elif match(fact, 'store_of', '*', '*'):
                self.rover_stores[args[1]] = args[0] # store_of ?s ?r
            elif match(fact, 'on_board', '*', '*'):
                self.rover_cameras[args[1]].append(args[0]) # on_board ?i ?r
            elif match(fact, 'supports', '*', '*'):
                self.camera_modes[args[0]].add(args[1]) # supports ?c ?m
            elif match(fact, 'calibration_target', '*', '*'):
                self.camera_calibration_target[args[0]] = args[1] # calibration_target ?i ?t
            elif match(fact, 'visible_from', '*', '*'):
                self.objective_visible_from[args[0]].add(args[1]) # visible_from ?o ?w
            elif match(fact, 'visible', '*', '*'):
                self.waypoint_visibility[args[0]].add(args[1]) # visible ?w1 ?w2
            elif match(fact, 'can_traverse', '*', '*', '*'):
                self.rover_traversal_graph[args[0]][args[1]].add(args[2]) # can_traverse ?r ?y ?z

        # Parse initial state for initial sample locations
        for fact in task.initial_state:
             if match(fact, 'at_soil_sample', '*'):
                 self.initial_soil_samples.add(get_parts(fact)[1])
             elif match(fact, 'at_rock_sample', '*'):
                 self.initial_rock_samples.add(get_parts(fact)[1])

        # Compute all-pairs shortest paths for each rover
        self.rover_shortest_paths = {}
        for rover in self.rovers:
            self.rover_shortest_paths[rover] = {}
            # Ensure all waypoints are in the graph keys, even if no edges from them
            # This is important for BFS to calculate distances to all waypoints
            for wp in self.waypoint_names:
                 if wp not in self.rover_traversal_graph[rover]:
                      self.rover_traversal_graph[rover][wp] = set()

            for start_wp in self.waypoint_names:
                self.rover_shortest_paths[rover][start_wp] = bfs(self.rover_traversal_graph[rover], start_wp)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Parse current state
        current_rover_location = {} # rover -> wp
        current_rover_store_status = {} # rover -> 'empty' or 'full'
        current_rover_soil_samples = defaultdict(set) # rover -> {wp}
        current_rover_rock_samples = defaultdict(set) # rover -> {wp}
        current_rover_images = defaultdict(set) # rover -> {(obj, mode)}
        current_rover_camera_calibration = defaultdict(set) # rover -> {camera}
        current_soil_samples_at_waypoint = set() # {wp}
        current_rock_samples_at_waypoint = set() # {wp}
        current_communicated_soil = set() # {wp}
        current_communicated_rock = set() # {wp}
        current_communicated_image = set() # {(obj, mode)}

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue
            pred = parts[0]
            args = parts[1:]

            if match(fact, 'at', 'rover*', '*'):
                current_rover_location[args[0]] = args[1]
            elif match(fact, 'empty', 'store*'):
                # Find the rover for this store
                store = args[0]
                rover = next((r for r, s in self.rover_stores.items() if s == store), None)
                if rover:
                    current_rover_store_status[rover] = 'empty'
            elif match(fact, 'full', 'store*'):
                 # Find the rover for this store
                store = args[0]
                rover = next((r for r, s in self.rover_stores.items() if s == store), None)
                if rover:
                    current_rover_store_status[rover] = 'full'
            elif match(fact, 'have_soil_analysis', 'rover*', '*'):
                current_rover_soil_samples[args[0]].add(args[1])
            elif match(fact, 'have_rock_analysis', 'rover*', '*'):
                current_rover_rock_samples[args[0]].add(args[1])
            elif match(fact, 'have_image', 'rover*', '*', '*'):
                current_rover_images[args[0]].add((args[1], args[2]))
            elif match(fact, 'calibrated', 'camera*', 'rover*'):
                current_rover_camera_calibration[args[1]].add(args[0])
            elif match(fact, 'at_soil_sample', '*'):
                current_soil_samples_at_waypoint.add(args[0])
            elif match(fact, 'at_rock_sample', '*'):
                current_rock_samples_at_waypoint.add(args[0])
            elif match(fact, 'communicated_soil_data', '*'):
                current_communicated_soil.add(args[0])
            elif match(fact, 'communicated_rock_data', '*'):
                current_communicated_rock.add(args[0])
            elif match(fact, 'communicated_image_data', '*', '*'):
                current_communicated_image.add((args[0], args[1]))

        total_cost = 0
        # Assuming there is at least one lander and getting its location
        lander_wp = list(self.lander_location.values())[0] if self.lander_location else None
        comm_wps = self.waypoint_visibility.get(lander_wp, set()) if lander_wp else set()

        # If no communication points are defined or lander location is unknown,
        # communication goals are impossible. Heuristic will reflect this via inf.
        # For solvable problems, this should not be an issue.

        for goal in self.goal_facts:
            if goal in state:
                continue # Goal already achieved

            parts = get_parts(goal)
            if not parts: continue
            pred = parts[0]
            args = parts[1:]

            min_cost_for_goal = math.inf

            if pred == 'communicated_soil_data':
                wp = args[0]
                data_had = False
                # Check if any rover already has the data
                for r in self.rovers:
                    if wp in current_rover_soil_samples.get(r, set()):
                        data_had = True
                        rover_wp = current_rover_location.get(r)
                        if rover_wp:
                            nav_cost = min_dist(self.rover_shortest_paths.get(r, {}), {rover_wp}, comm_wps)
                            if nav_cost != math.inf:
                                cost = nav_cost + 1 # communicate
                                min_cost_for_goal = min(min_cost_for_goal, cost)

                # If no rover has the data, check if sample is available to collect
                if not data_had and wp in current_soil_samples_at_waypoint:
                    for r in self.rovers:
                        if 'soil' in self.rover_capabilities.get(r, set()):
                            rover_wp = current_rover_location.get(r)
                            if rover_wp:
                                # Cost to sample
                                nav_to_sample = self.rover_shortest_paths.get(r, {}).get(rover_wp, {}).get(wp, math.inf)
                                if nav_to_sample != math.inf:
                                    drop_cost = 1 if current_rover_store_status.get(r) == 'full' else 0
                                    sample_cost = 1
                                    # Cost to communicate
                                    nav_to_comm = min_dist(self.rover_shortest_paths.get(r, {}), {wp}, comm_wps)
                                    if nav_to_comm != math.inf:
                                        communicate_cost = 1
                                        cost = nav_to_sample + drop_cost + sample_cost + nav_to_comm + communicate_cost
                                        min_cost_for_goal = min(min_cost_for_goal, cost)

            elif pred == 'communicated_rock_data':
                wp = args[0]
                data_had = False
                # Check if any rover already has the data
                for r in self.rovers:
                    if wp in current_rover_rock_samples.get(r, set()):
                        data_had = True
                        rover_wp = current_rover_location.get(r)
                        if rover_wp:
                            nav_cost = min_dist(self.rover_shortest_paths.get(r, {}), {rover_wp}, comm_wps)
                            if nav_cost != math.inf:
                                cost = nav_cost + 1 # communicate
                                min_cost_for_goal = min(min_cost_for_goal, cost)

                # If no rover has the data, check if sample is available to collect
                if not data_had and wp in current_rock_samples_at_waypoint:
                    for r in self.rovers:
                        if 'rock' in self.rover_capabilities.get(r, set()):
                            rover_wp = current_rover_location.get(r)
                            if rover_wp:
                                # Cost to sample
                                nav_to_sample = self.rover_shortest_paths.get(r, {}).get(rover_wp, {}).get(wp, math.inf)
                                if nav_to_sample != math.inf:
                                    drop_cost = 1 if current_rover_store_status.get(r) == 'full' else 0
                                    sample_cost = 1
                                    # Cost to communicate
                                    nav_to_comm = min_dist(self.rover_shortest_paths.get(r, {}), {wp}, comm_wps)
                                    if nav_to_comm != math.inf:
                                        communicate_cost = 1
                                        cost = nav_to_sample + drop_cost + sample_cost + nav_to_comm + communicate_cost
                                        min_cost_for_goal = min(min_cost_for_goal, cost)

            elif pred == 'communicated_image_data':
                obj, mode = args
                data_had = False
                 # Check if any rover already has the image
                for r in self.rovers:
                    if (obj, mode) in current_rover_images.get(r, set()):
                        data_had = True
                        rover_wp = current_rover_location.get(r)
                        if rover_wp:
                            nav_cost = min_dist(self.rover_shortest_paths.get(r, {}), {rover_wp}, comm_wps)
                            if nav_cost != math.inf:
                                cost = nav_cost + 1 # communicate
                                min_cost_for_goal = min(min_cost_for_goal, cost)

                # If no rover has the image, estimate cost to take and communicate
                if not data_had:
                    img_wps = self.objective_visible_from.get(obj, set())
                    if img_wps:
                        for r in self.rovers:
                            if 'imaging' in self.rover_capabilities.get(r, set()):
                                rover_wp = current_rover_location.get(r)
                                if rover_wp:
                                    for i in self.rover_cameras.get(r, []):
                                        if mode in self.camera_modes.get(i, set()):
                                            cal_target = self.camera_calibration_target.get(i)
                                            if cal_target:
                                                cal_wps = self.objective_visible_from.get(cal_target, set())
                                                if cal_wps:
                                                    # Cost to get image
                                                    image_cost = math.inf
                                                    # Option: If camera is calibrated: nav to img_wp + take_image
                                                    if i in current_rover_camera_calibration.get(r, set()):
                                                        nav_to_img = min_dist(self.rover_shortest_paths.get(r, {}), {rover_wp}, img_wps)
                                                        if nav_to_img != math.inf:
                                                            image_cost = nav_to_img + 1 # nav to img, take_image
                                                    # Option: If not calibrated: nav to cal_wp + calibrate + nav from cal_wp to img_wp + take_image
                                                    else:
                                                        nav_to_cal = min_dist(self.rover_shortest_paths.get(r, {}), {rover_wp}, cal_wps)
                                                        nav_cal_to_img = min_dist(self.rover_shortest_paths.get(r, {}), cal_wps, img_wps)
                                                        if nav_to_cal != math.inf and nav_cal_to_img != math.inf:
                                                            image_cost = nav_to_cal + 1 + nav_cal_to_img + 1 # nav to cal, calibrate, nav from cal to img, take_image

                                                    # Cost to communicate
                                                    nav_img_to_comm = min_dist(self.rover_shortest_paths.get(r, {}), img_wps, comm_wps)
                                                    communicate_cost = math.inf
                                                    if nav_img_to_comm != math.inf:
                                                        communicate_cost = nav_img_to_comm + 1 # nav to comm, communicate

                                                    if image_cost != math.inf and communicate_cost != math.inf:
                                                        cost = image_cost + communicate_cost
                                                        min_cost_for_goal = min(min_cost_for_goal, cost)


            # Add the minimum cost for this goal to the total heuristic
            # Only add if the goal is considered achievable (cost is not infinity)
            if min_cost_for_goal != math.inf:
                total_cost += min_cost_for_goal
            # else: The goal is unachievable from this state based on our model.
            # For greedy BFS, leaving it as infinity (or not adding anything if we
            # assume solvable problems) is standard. If it's truly unachievable,
            # the search might explore states trying to reach it, but the heuristic
            # won't guide it towards it.

        return total_cost
