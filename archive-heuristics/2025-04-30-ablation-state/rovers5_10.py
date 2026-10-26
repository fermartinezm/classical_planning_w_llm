# Import necessary modules
from fnmatch import fnmatch
from collections import defaultdict, deque
# Assuming heuristic_base is available in the specified path
from heuristics.heuristic_base import Heuristic
# import sys # Not strictly needed, float('inf') is sufficient

# Helper functions for parsing PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential leading/trailing whitespace or multiple spaces
    return fact.strip()[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at rover1 waypoint1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# Helper function for BFS
def bfs_shortest_paths(graph, start_node):
    """
    Computes shortest path distances from a start_node to all other nodes
    in a graph using Breadth-First Search. Assumes unit edge costs.

    Args:
        graph: A dictionary where keys are nodes and values are sets of neighbor nodes.
               Represents directed edges.
        start_node: The node to start the BFS from.

    Returns:
        A dictionary mapping each reachable node to its distance from the start_node.
        Nodes not reachable are not included.
    """
    distances = {start_node: 0}
    queue = deque([start_node])
    while queue:
        current_node = queue.popleft()
        current_dist = distances[current_node]
        if current_node in graph: # Handle nodes with no outgoing edges
            for neighbor in graph[current_node]:
                if neighbor not in distances:
                    distances[neighbor] = current_dist + 1
                    queue.append(neighbor)
    return distances

class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve all
    unmet goal conditions. It sums the estimated minimum cost for each
    individual goal fact, ignoring potential synergies between goals.
    The cost for each goal fact is estimated by finding the minimum cost
    sequence of actions (navigation, sampling/imaging, communication)
    required for any capable rover to achieve that specific fact, assuming
    other goal facts are independent. Navigation costs are estimated using
    precomputed shortest path distances (BFS) on the rover's traverse graph.

    # Assumptions
    - The cost of each action is 1.
    - Goals are independent (summing individual goal costs).
    - The lander location is static.
    - Rover capabilities (equipped_for_...), camera properties (on_board, supports, calibration_target),
      initial sample locations (at_soil_sample, at_rock_sample), objective visibility (visible_from),
      and waypoint visibility (visible) are static.
    - Navigation is possible between waypoints if (can_traverse ?r ?y ?z) and (visible ?y ?z) hold.
      The BFS graph for a rover `r` from `y` to `z` includes an edge if `(can_traverse r y z)` AND `(visible y z)`.

    # Heuristic Initialization
    - Extracts goal conditions.
    - Extracts static facts including:
        - Lander location.
        - Rover capabilities (soil, rock, imaging).
        - Store ownership.
        - Initial sample locations.
        - Waypoint visibility graph.
        - Rover traversability graphs.
        - Camera information (onboard, supported modes, calibration target).
        - Objective visibility waypoints.
    - Precomputes shortest path distances between all waypoints for each rover
      using BFS on the combined `can_traverse` and `visible` graph.
    - Identifies waypoints visible from the lander.
    - Stores mappings for quick lookup of static information.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Initialize total heuristic cost to 0.
    2. Identify the current location of each rover and the state of their stores (empty/full).
    3. For each goal fact `g` in the task's goals:
        a. If `g` is already true in the current state, continue to the next goal.
        b. If `g` is `(communicated_soil_data ?w)`:
            - Find all rovers `?r` equipped for soil analysis.
            - For each such rover `?r`, calculate the estimated cost to achieve this goal:
                - Cost to sample: If `(have_soil_analysis ?r ?w)` is false:
                    - Navigation cost from `?r`'s current location to `?w`.
                    - Add 1 for `sample_soil` action.
                    - Add 1 for `drop` action if `?r`'s store is currently full.
                    - Rover's location becomes `?w`.
                - Cost to communicate:
                    - Find the minimum navigation cost from the rover's current location (after sampling, if applicable) to any waypoint visible from the lander.
                    - Add 1 for `communicate_soil_data` action.
            - The cost for this goal fact is the minimum estimated cost over all capable rovers. If no rover can achieve it, the cost is infinity.
        c. If `g` is `(communicated_rock_data ?w)`:
            - Similar logic to soil data, using rock-specific predicates and capabilities.
        d. If `g` is `(communicated_image_data ?o ?m)`:
            - Find all rovers `?r` equipped for imaging with a camera `?i` that supports mode `?m` and is onboard `?r`.
            - For each such rover/camera pair (`?r`, `?i`):
                - Cost to image: If `(have_image ?r ?o ?m)` is false:
                    - Cost to calibrate: If `(calibrated ?i ?r)` is false:
                        - Find a waypoint `cal_wp` visible from `?i`'s calibration target.
                        - Navigation cost from `?r`'s current location to `cal_wp`.
                        - Add 1 for `calibrate` action.
                        - Rover's location becomes `cal_wp`.
                    - Find a waypoint `img_wp` visible from objective `?o`.
                    - Navigation cost from the rover's current location (after calibration, if applicable) to `img_wp`.
                    - Add 1 for `take_image` action.
                    - Rover's location becomes `img_wp`.
                - Cost to communicate:
                    - Find the minimum navigation cost from the rover's current location (after imaging, if applicable) to any waypoint visible from the lander.
                    - Add 1 for `communicate_image_data` action.
            - The cost for this goal fact is the minimum estimated cost over all capable rover/camera pairs. If no pair can achieve it, the cost is infinity.
    4. Sum the costs calculated for each unachieved goal fact to get the total heuristic value.
    5. Return the total cost. If the total cost is infinity (meaning at least one goal is unreachable), return infinity. Returning 0 for goal state is handled by step 3a.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # --- Extract Static Information ---

        self.lander_location = None
        for fact in static_facts:
            if match(fact, "at_lander", "*", "*"):
                self.lander_location = get_parts(fact)[2]
                break # Assuming only one lander

        self.equipped_soil = set()
        self.equipped_rock = set()
        self.equipped_imaging = set()
        self.rover_store = {} # rover -> store name
        self.initial_soil_samples = set() # waypoint -> True (if sample was initially there)
        self.initial_rock_samples = set() # waypoint -> True (if sample was initially there)
        self.waypoint_visibility = defaultdict(set) # visible w1 w2 : w1 -> {w2}
        self.rover_traversability = defaultdict(lambda: defaultdict(set)) # can_traverse r w1 w2 : r -> w1 -> {w2}
        self.camera_info = {} # camera -> {'rover': r, 'modes': {m}, 'cal_target': t}
        self.objective_visible_from = defaultdict(set) # objective -> {wp}

        all_waypoints = set()
        all_rovers = set()
        all_cameras = set()
        all_objectives = set()
        all_modes = set()

        # First pass to collect all objects and build basic static mappings
        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue # Skip empty facts

            predicate = parts[0]

            if predicate == 'at_lander':
                 if len(parts) > 2: all_waypoints.add(parts[2])
            elif predicate == 'equipped_for_soil_analysis':
                if len(parts) > 1: self.equipped_soil.add(parts[1])
                if len(parts) > 1: all_rovers.add(parts[1])
            elif predicate == 'equipped_for_rock_analysis':
                if len(parts) > 1: self.equipped_rock.add(parts[1])
                if len(parts) > 1: all_rovers.add(parts[1])
            elif predicate == 'equipped_for_imaging':
                if len(parts) > 1: self.equipped_imaging.add(parts[1])
                if len(parts) > 1: all_rovers.add(parts[1])
            elif predicate == 'store_of':
                if len(parts) > 2:
                    store, rover = parts[1], parts[2]
                    self.rover_store[rover] = store
                    all_rovers.add(rover)
            elif predicate == 'at_soil_sample':
                if len(parts) > 1: self.initial_soil_samples.add(parts[1])
                if len(parts) > 1: all_waypoints.add(parts[1])
            elif predicate == 'at_rock_sample':
                if len(parts) > 1: self.initial_rock_samples.add(parts[1])
                if len(parts) > 1: all_waypoints.add(parts[1])
            elif predicate == 'visible':
                if len(parts) > 2:
                    w1, w2 = parts[1], parts[2]
                    self.waypoint_visibility[w1].add(w2)
                    all_waypoints.add(w1)
                    all_waypoints.add(w2)
            elif predicate == 'can_traverse':
                if len(parts) > 3:
                    r, w1, w2 = parts[1], parts[2], parts[3]
                    self.rover_traversability[r][w1].add(w2)
                    all_rovers.add(r)
                    all_waypoints.add(w1)
                    all_waypoints.add(w2)
            elif predicate == 'on_board':
                if len(parts) > 2:
                    camera, rover = parts[1], parts[2]
                    if camera not in self.camera_info:
                        self.camera_info[camera] = {'rover': None, 'modes': set(), 'cal_target': None}
                    self.camera_info[camera]['rover'] = rover
                    all_cameras.add(camera)
                    all_rovers.add(rover)
            elif predicate == 'supports':
                if len(parts) > 2:
                    camera, mode = parts[1], parts[2]
                    if camera not in self.camera_info:
                        self.camera_info[camera] = {'rover': None, 'modes': set(), 'cal_target': None}
                    self.camera_info[camera]['modes'].add(mode)
                    all_cameras.add(camera)
                    all_modes.add(mode)
            elif predicate == 'calibration_target':
                if len(parts) > 2:
                    camera, objective = parts[1], parts[2]
                    if camera not in self.camera_info:
                        self.camera_info[camera] = {'rover': None, 'modes': set(), 'cal_target': None}
                    self.camera_info[camera]['cal_target'] = objective
                    all_cameras.add(camera)
                    all_objectives.add(objective)
            elif predicate == 'visible_from':
                if len(parts) > 2:
                    objective, waypoint = parts[1], parts[2]
                    self.objective_visible_from[objective].add(waypoint)
                    all_objectives.add(objective)
                    all_waypoints.add(waypoint)
            # Collect all waypoints mentioned in goals too
            elif predicate in ['communicated_soil_data', 'communicated_rock_data']:
                 if len(parts) > 1: all_waypoints.add(parts[1])
            # For communicated_image_data, objective and mode are involved, waypoints are derived from visible_from

        self.all_waypoints = list(all_waypoints) # Convert to list for consistent iteration/lookup
        self.all_rovers = list(all_rovers)
        self.all_cameras = list(all_cameras)
        self.all_objectives = list(all_objectives)
        self.all_modes = list(all_modes)


        # --- Precompute Navigation Distances ---
        self.rover_distances = {} # rover -> start_wp -> end_wp -> distance
        self.communication_waypoints = set() # Waypoints visible from lander

        if self.lander_location:
             # Waypoints visible from the lander's location
             self.communication_waypoints = self.waypoint_visibility.get(self.lander_location, set())

        for rover in self.all_rovers:
            rover_graph = defaultdict(set)
            # Build the navigation graph for this rover: (can_traverse r w1 w2) AND (visible w1 w2)
            if rover in self.rover_traversability:
                for w1, neighbors in self.rover_traversability[rover].items():
                    for w2 in neighbors:
                         # The navigate action requires (visible ?y ?z) where ?y is current and ?z is destination.
                         # So edge w1 -> w2 exists if (can_traverse r w1 w2) and (visible w1 w2)
                         if w2 in self.waypoint_visibility.get(w1, set()):
                             rover_graph[w1].add(w2)

            self.rover_distances[rover] = {}
            # Compute BFS from every waypoint to every other waypoint for this rover
            for start_wp in self.all_waypoints:
                 self.rover_distances[rover][start_wp] = bfs_shortest_paths(rover_graph, start_wp)

        # --- Precompute Camera Calibration Waypoints ---
        self.camera_cal_visible_from = defaultdict(set) # camera -> {wp}
        for camera, info in self.camera_info.items():
            cal_target = info.get('cal_target')
            if cal_target and cal_target in self.objective_visible_from:
                self.camera_cal_visible_from[camera] = self.objective_visible_from[cal_target]


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Check if goal is already reached
        if self.goals <= state:
            return 0

        # --- Extract Dynamic Information from State ---
        current_rover_locations = {}
        # Create a set of facts that are currently true in the state for quick lookup
        state_facts = set(state)

        for fact in state_facts:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]

            if predicate == 'at' and len(parts) == 3 and parts[1].startswith('rover'):
                rover, wp = parts[1], parts[2]
                current_rover_locations[rover] = wp


        total_heuristic_cost = 0
        infinity = float('inf') # Use infinity for unreachable goals

        # --- Estimate Cost for Each Ungoaled Fact ---
        for goal in self.goals:
            if goal in state_facts:
                continue # Goal already achieved

            parts = get_parts(goal)
            if not parts: continue # Should not happen for goal facts

            goal_type = parts[0]

            if goal_type == 'communicated_soil_data':
                waypoint_goal = parts[1]
                min_cost_for_goal = infinity

                # Find rovers capable of soil analysis
                capable_rovers = [r for r in self.all_rovers if r in self.equipped_soil]

                for rover in capable_rovers:
                    rover_current_wp = current_rover_locations.get(rover)
                    if not rover_current_wp:
                         continue # Rover location unknown

                    cost_to_get_have = 0
                    rover_loc_after_have = rover_current_wp # Default if already have it

                    # Check if rover already has the soil analysis
                    have_fact = f"(have_soil_analysis {rover} {waypoint_goal})"
                    if have_fact not in state_facts:
                        # Need to sample
                        # Check if sample was initially at this waypoint (static)
                        if waypoint_goal not in self.initial_soil_samples:
                             continue # Cannot sample soil here if it wasn't there initially

                        # Cost to navigate to sample location
                        dist_to_sample = self.rover_distances[rover].get(rover_current_wp, {}).get(waypoint_goal, infinity)
                        if dist_to_sample == infinity:
                            continue # Cannot reach sample location

                        cost_to_get_have += dist_to_sample
                        cost_to_get_have += 1 # sample_soil action

                        # Check if store is full, need drop action before sampling
                        rover_store_name = self.rover_store.get(rover)
                        # Check if the store exists and is full
                        if rover_store_name and f"(full {rover_store_name})" in state_facts:
                             cost_to_get_have += 1 # drop action

                        rover_loc_after_have = waypoint_goal # Rover is at sample location after sampling

                    # Cost to communicate
                    min_comm_nav_cost = infinity
                    if self.communication_waypoints:
                        for comm_wp in self.communication_waypoints:
                            dist_to_comm = self.rover_distances[rover].get(rover_loc_after_have, {}).get(comm_wp, infinity)
                            min_comm_nav_cost = min(min_comm_nav_cost, dist_to_comm)

                    if min_comm_nav_cost == infinity:
                         continue # Cannot reach any communication waypoint

                    cost_to_communicate = min_comm_nav_cost + 1 # communicate_soil_data action

                    total_cost_for_rover = cost_to_get_have + cost_to_communicate
                    min_cost_for_goal = min(min_cost_for_goal, total_cost_for_rover)

                if min_cost_for_goal != infinity:
                    total_heuristic_cost += min_cost_for_goal
                else:
                    # If a goal is unreachable by any rover, the problem is likely unsolvable
                    # or this path is bad. Return infinity.
                    return infinity # Goal is unreachable

            elif goal_type == 'communicated_rock_data':
                waypoint_goal = parts[1]
                min_cost_for_goal = infinity

                # Find rovers capable of rock analysis
                capable_rovers = [r for r in self.all_rovers if r in self.equipped_rock]

                for rover in capable_rovers:
                    rover_current_wp = current_rover_locations.get(rover)
                    if not rover_current_wp:
                         continue

                    cost_to_get_have = 0
                    rover_loc_after_have = rover_current_wp # Default if already have it

                    # Check if rover already has the rock analysis
                    have_fact = f"(have_rock_analysis {rover} {waypoint_goal})"
                    if have_fact not in state_facts:
                        # Need to sample
                        # Check if sample was initially at this waypoint (static)
                        if waypoint_goal not in self.initial_rock_samples:
                             continue # Cannot sample rock here if it wasn't there initially

                        # Cost to navigate to sample location
                        dist_to_sample = self.rover_distances[rover].get(rover_current_wp, {}).get(waypoint_goal, infinity)
                        if dist_to_sample == infinity:
                            continue # Cannot reach sample location

                        cost_to_get_have += dist_to_sample
                        cost_to_get_have += 1 # sample_rock action

                        # Check if store is full, need drop action before sampling
                        rover_store_name = self.rover_store.get(rover)
                        # Check if the store exists and is full
                        if rover_store_name and f"(full {rover_store_name})" in state_facts:
                             cost_to_get_have += 1 # drop action

                        rover_loc_after_have = waypoint_goal # Rover is at sample location after sampling

                    # Cost to communicate
                    min_comm_nav_cost = infinity
                    if self.communication_waypoints:
                        for comm_wp in self.communication_waypoints:
                            dist_to_comm = self.rover_distances[rover].get(rover_loc_after_have, {}).get(comm_wp, infinity)
                            min_comm_nav_cost = min(min_comm_nav_cost, dist_to_comm)

                    if min_comm_nav_cost == infinity:
                         continue # Cannot reach any communication waypoint

                    cost_to_communicate = min_comm_nav_cost + 1 # communicate_rock_data action

                    total_cost_for_rover = cost_to_get_have + cost_to_communicate
                    min_cost_for_goal = min(min_cost_for_goal, total_cost_for_rover)

                if min_cost_for_goal != infinity:
                    total_heuristic_cost += min_cost_for_goal
                else:
                    return infinity # Goal is unreachable

            elif goal_type == 'communicated_image_data':
                objective_goal, mode_goal = parts[1], parts[2]
                min_cost_for_goal = infinity

                # Find rovers capable of imaging
                capable_rovers = [r for r in self.all_rovers if r in self.equipped_imaging]

                for rover in capable_rovers:
                    rover_current_wp = current_rover_locations.get(rover)
                    if not rover_current_wp:
                         continue

                    # Find cameras onboard this rover that support the goal mode
                    capable_cameras = [
                        cam for cam, info in self.camera_info.items()
                        if info.get('rover') == rover and mode_goal in info.get('modes', set())
                    ]

                    for camera in capable_cameras:
                        cost_to_get_have = 0
                        rover_loc_after_have = rover_current_wp # Default if already have it

                        # Check if rover already has the image
                        have_fact = f"(have_image {rover} {objective_goal} {mode_goal})"
                        if have_fact not in state_facts:
                            # Need to take image

                            cost_to_calibrate = 0
                            rover_loc_after_cal = rover_current_wp # Default if already calibrated

                            # Check if camera is calibrated
                            calibrated_fact = f"(calibrated {camera} {rover})"
                            if calibrated_fact not in state_facts:
                                # Need to calibrate
                                cal_waypoints = self.camera_cal_visible_from.get(camera, set())
                                if not cal_waypoints:
                                     continue # No waypoint to calibrate from

                                min_cal_nav_cost = infinity
                                # Find the closest calibration waypoint
                                closest_cal_wp = None
                                for cal_wp in cal_waypoints:
                                    dist_to_cal = self.rover_distances[rover].get(rover_current_wp, {}).get(cal_wp, infinity)
                                    if dist_to_cal < min_cal_nav_cost:
                                        min_cal_nav_cost = dist_to_cal
                                        closest_cal_wp = cal_wp

                                if min_cal_nav_cost == infinity:
                                     continue # Cannot reach any calibration waypoint

                                cost_to_calibrate += min_cal_nav_cost
                                cost_to_calibrate += 1 # calibrate action
                                rover_loc_after_cal = closest_cal_wp # Rover is at calibration waypoint after calibrating

                            # Cost to navigate to image location and take image
                            img_waypoints = self.objective_visible_from.get(objective_goal, set())
                            if not img_waypoints:
                                 continue # No waypoint to image from

                            min_img_nav_cost = infinity
                            # Find the closest image waypoint
                            closest_img_wp = None
                            for img_wp in img_waypoints:
                                dist_to_img = self.rover_distances[rover].get(rover_loc_after_cal, {}).get(img_wp, infinity)
                                if dist_to_img < min_img_nav_cost:
                                     min_img_nav_cost = dist_to_img
                                     closest_img_wp = img_wp


                            if min_img_nav_cost == infinity:
                                 continue # Cannot reach any image waypoint

                            cost_to_get_have += cost_to_calibrate # Add cost of calibration path
                            cost_to_get_have += min_img_nav_cost # Add cost of navigation to image waypoint
                            cost_to_get_have += 1 # take_image action
                            rover_loc_after_have = closest_img_wp # Rover is at image waypoint after taking image


                        # Cost to communicate
                        min_comm_nav_cost = infinity
                        if self.communication_waypoints:
                            # Find the closest communication waypoint
                            for comm_wp in self.communication_waypoints:
                                dist_to_comm = self.rover_distances[rover].get(rover_loc_after_have, {}).get(comm_wp, infinity)
                                min_comm_nav_cost = min(min_comm_nav_cost, dist_to_comm)

                        if min_comm_nav_cost == infinity:
                             continue # Cannot reach any communication waypoint

                        cost_to_communicate = min_comm_nav_cost + 1 # communicate_image_data action

                        total_cost_for_pair = cost_to_get_have + cost_to_communicate
                        min_cost_for_goal = min(min_cost_for_goal, total_cost_for_pair)

                if min_cost_for_goal != infinity:
                    total_heuristic_cost += min_cost_for_goal
                else:
                    return infinity # Goal is unreachable

            # No other goal types specified in the domain file provided

        # Return infinity if any goal was unreachable, otherwise return the sum
        # If total_heuristic_cost is infinity, it means one of the min_cost_for_goal was infinity
        # which was already handled by returning infinity early.
        # If we reached here, all goals are either met or reachable.
        return total_heuristic_cost
