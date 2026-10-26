# <code-file-heuristic-rovers>
from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL fact strings
def get_parts(fact):
    """
    Removes surrounding parentheses from a PDDL fact string and splits it
    into a list of parts (predicate and arguments).
    e.g., '(at rover1 waypoint1)' -> ['at', 'rover1', 'waypoint1']
    """
    # Remove parentheses and split by spaces
    return fact[1:-1].split()

# Helper function to match fact patterns
def match(fact, *args):
    """
    Checks if a PDDL fact string matches a given pattern of arguments.
    Uses fnmatch for pattern matching.
    e.g., match('(at rover1 waypoint1)', 'at', '*', 'waypoint1') -> True
    """
    parts = get_parts(fact)
    # Check if the number of parts matches the number of args, and if each part matches the corresponding arg pattern
    return len(parts) == len(args) and all(fnmatch(part, arg) for part, arg in zip(parts, args))

# BFS function to find shortest path distance for a specific rover
def get_distance(rover, start_waypoint, end_waypoint, traversal_graph):
    """
    Calculates the shortest path distance (number of navigation actions)
    for a specific rover between two waypoints using BFS on the rover's
    traversal graph.
    Returns float('inf') if the end waypoint is unreachable by the rover
    from the start waypoint.
    """
    if start_waypoint == end_waypoint:
        return 0

    queue = deque([(start_waypoint, 0)])
    visited = {start_waypoint}

    graph = traversal_graph.get(rover)
    if not graph:
        return float('inf') # Rover cannot traverse anywhere

    while queue:
        current_wp, dist = queue.popleft()

        # Check if current_wp is in the graph for this rover.
        # This provides safety if the graph wasn't fully populated with all waypoints.
        if current_wp not in graph:
             continue

        for neighbor_wp in graph[current_wp]:
            if neighbor_wp not in visited:
                visited.add(neighbor_wp)
                if neighbor_wp == end_waypoint:
                    return dist + 1
                queue.append((neighbor_wp, dist + 1))

    return float('inf') # Not reachable

class roversHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Rovers domain.

    Summary:
    The heuristic estimates the cost to reach the goal by summing the estimated
    minimum costs for each unachieved goal fact independently. For communication
    goals (soil, rock, image), it estimates the cost to first acquire the
    necessary data (sampling or taking image) if not already possessed by a
    rover, and then the cost to navigate a rover holding the data to a location
    visible from the lander and perform the communication action. Navigation
    costs are estimated using shortest path distances (BFS) on the rover's
    traversal graph. Action costs (sample, calibrate, take_image, communicate)
    are assumed to be 1.

    Assumptions:
    - The heuristic assumes that if a communication goal for a sample (soil/rock)
      from a waypoint `W` is present and no rover currently holds the sample
      `(have_soil_analysis ?r W)` or `(have_rock_analysis ?r W)`, then the
      sample `(at_soil_sample W)` or `(at_rock_sample W)` must still be present
      at waypoint `W` in the current state for the goal to be reachable. If the
      sample is not at `W` and not held by any rover, the goal is considered
      unreachable.
    - The heuristic assumes that if an image goal `(communicated_image_data O M)`
      is present and no rover currently holds the image `(have_image ?r O M)`,
      then there exists an equipped rover with a camera supporting mode `M`,
      a waypoint `P` visible from objective `O`, a calibration target `T` for
      that camera, and a waypoint `W` visible from `T`, allowing the image to
      be taken. If any of these prerequisites (suitable rover/camera, image
      waypoint, calibration target/waypoint) are missing, the goal is considered
      unreachable.
    - The heuristic calculates costs for acquiring and communicating each goal
      item independently, ignoring potential synergies (e.g., one rover collecting
      multiple samples, or one navigation path serving multiple goals).
    - Store capacity and the `drop` action are ignored for simplicity in sample
      acquisition cost estimation.
    - Camera calibration state is handled by assuming calibration is needed
      before taking an image if the image is not already possessed. The cost
      includes navigation to a calibration waypoint and the calibrate action,
      followed by navigation to the image waypoint and the take_image action.
      It does not track calibration state across multiple image goals for the
      same camera/rover.

    Heuristic Initialization:
    The constructor pre-processes static facts from the task description to build
    data structures needed for heuristic calculation:
    - `rover_capabilities`: Maps rovers to their equipment (soil, rock, imaging).
    - `rover_cameras`: Maps rovers to the cameras they have on board.
    - `camera_supports`: Maps cameras to the modes they support.
    - `camera_calibration_targets`: Maps cameras to their calibration objectives.
    - `objective_visible_from`: Maps objectives to waypoints they are visible from.
    - `lander_location`: The waypoint where the lander is located.
    - `communication_waypoints`: List of waypoints visible from the lander location.
    - `rover_traversal_graphs`: Adjacency list representation of the navigation
      graph for each rover, built from `can_traverse` facts. This is used for BFS.
      Includes all waypoints mentioned in `can_traverse` or `visible` or `at_lander`
      or `visible_from` to ensure graph lookups don't fail.

    Step-By-Step Thinking for Computing Heuristic:
    1. Initialize `total_cost` to 0. Define a large finite number `LARGE_FINITE`
       to represent unreachable goals. Use `INF = float('inf')` for internal BFS results.
    2. Parse the current state to extract dynamic information: rover locations,
       locations of soil/rock samples, which rovers have which samples/images.
    3. Iterate through each goal fact specified in `task.goals`.
    4. If a goal fact `g` is already true in the current state, its cost is 0. Continue to the next goal.
    5. If `g` is not true:
        a. If `g` is `(communicated_soil_data ?w)`:
            i. Find the minimum cost to achieve this goal. Initialize `min_goal_cost` to `INF`.
            ii. Check if any rover `R` currently has `(have_soil_analysis R ?w)`.
            iii. If yes: The cost is the minimum cost for `R` to navigate from its current location (`R_loc`) to a communication waypoint (`comm_wp`) and perform the `communicate_soil_data` action (+1 cost). Calculate `min_dist(R_loc, comm_wp) + 1` over all communication waypoints `comm_wp`. Update `min_goal_cost` with the minimum found cost.
            iv. If no: The sample needs to be acquired and then communicated. Check if `(at_soil_sample ?w)` is in the state. If not, the goal is considered unreachable (`min_goal_cost` remains `INF`). If yes: Find equipped rovers `R`. For each `R`, calculate cost to sample: `dist_to_sample = get_distance(R, R_loc, ?w)`. If reachable (`dist_to_sample != INF`), cost is `dist_to_sample + 1` (sample action). Then calculate cost to communicate from `?w`: `dist_to_comm = min(get_distance(R, ?w, comm_wp) + 1 for comm_wp in communication_waypoints)`. If reachable (`dist_to_comm != INF`), total cost for this rover is `dist_to_sample + 1 + dist_to_comm`. Update `min_goal_cost` with the minimum total cost over all equipped rovers.
            v. Add `min_goal_cost` to `total_cost`. If `min_goal_cost` was `INF`, `total_cost` will become `INF`.
        b. If `g` is `(communicated_rock_data ?w)`: Similar logic as soil data, using rock-specific predicates and equipment.
        c. If `g` is `(communicated_image_data ?o ?m)`:
            i. Find the minimum cost to achieve this goal. Initialize `min_goal_cost` to `INF`.
            ii. Check if any rover `R` currently has `(have_image R ?o ?m)`.
            iii. If yes: The cost is the minimum cost for `R` to navigate from its current location (`R_loc`) to a communication waypoint (`comm_wp`) and perform the `communicate_image_data` action (+1 cost). Calculate `min_dist(R_loc, comm_wp) + 1` over all communication waypoints `comm_wp`. Update `min_goal_cost` with the minimum found cost.
            iv. If no: The image needs to be taken and then communicated. Find equipped rovers `R` with cameras `I` supporting `M`. For each such `(R, I)` pair: Find waypoints `P` visible from `O` and `W` visible from `I`'s calibration target `T`. If no such `P` or `W` exists, this rover/camera cannot achieve the goal, skip this pair. Otherwise, calculate cost to take image: `min_img_acq_cost_for_pair = min(get_distance(R, R_loc, W) + 1 + get_distance(R, W, P) + 1 for W in Ws for P in Ps)`. If reachable (`min_img_acq_cost_for_pair != INF`), calculate cost to communicate from `P`: `min_comm_cost_from_img = min(get_distance(R, P, comm_wp) + 1 for P in Ps for comm_wp in communication_waypoints)`. If reachable (`min_comm_cost_from_img != INF`), total cost for this rover/camera is `min_img_acq_cost_for_pair + min_comm_cost_from_img`. Update `min_goal_cost` with the minimum total cost over all suitable rover/camera pairs.
            v. Add `min_goal_cost` to `total_cost`. If `min_goal_cost` was `INF`, `total_cost` will become `INF`.
    6. After processing all goals, if `total_cost` is `INF` (meaning at least one unachieved goal is unreachable), return `LARGE_FINITE`.
    7. Otherwise, return `total_cost`. This will be 0 if all goals were initially met, and positive otherwise.
    """
    def __init__(self, task):
        self.goals = task.goals
        static_facts = task.static

        # Data structures to store static information
        self.rover_capabilities = {} # {rover: {soil: bool, rock: bool, imaging: bool}}
        self.rover_cameras = {}      # {rover: [camera]}
        self.camera_supports = {}    # {camera: [mode]}
        self.camera_calibration_targets = {} # {camera: objective}
        self.objective_visible_from = {} # {objective: [waypoint]}
        self.lander_location = None
        self.communication_waypoints = [] # [waypoint] visible from lander
        self.rover_traversal_graphs = {} # {rover: {waypoint: [neighbor_waypoint]}}

        # Collect all relevant objects and waypoints mentioned in static facts
        all_waypoints = set()
        all_rovers = set()
        all_cameras = set()
        all_objectives = set()
        all_modes = set()
        all_stores = set()
        all_landers = set()

        visible_map = {} # {wp: [neighbor_wp]}

        for fact in static_facts:
            parts = get_parts(fact)
            predicate = parts[0]

            if predicate == 'at_lander':
                self.lander_location = parts[2]
                all_landers.add(parts[1])
                all_waypoints.add(parts[2])
            elif predicate == 'equipped_for_soil_analysis':
                rover = parts[1]
                self.rover_capabilities.setdefault(rover, {}).update({'soil': True})
                all_rovers.add(rover)
            elif predicate == 'equipped_for_rock_analysis':
                rover = parts[1]
                self.rover_capabilities.setdefault(rover, {}).update({'rock': True})
                all_rovers.add(rover)
            elif predicate == 'equipped_for_imaging':
                rover = parts[1]
                self.rover_capabilities.setdefault(rover, {}).update({'imaging': True})
                all_rovers.add(rover)
            elif predicate == 'store_of':
                store, rover = parts[1], parts[2]
                all_stores.add(store)
                all_rovers.add(rover)
            elif predicate == 'on_board':
                camera, rover = parts[1], parts[2]
                self.rover_cameras.setdefault(rover, []).append(camera)
                all_cameras.add(camera)
                all_rovers.add(rover)
            elif predicate == 'supports':
                camera, mode = parts[1], parts[2]
                self.camera_supports.setdefault(camera, []).append(mode)
                all_cameras.add(camera)
                all_modes.add(mode)
            elif predicate == 'calibration_target':
                camera, objective = parts[1], parts[2]
                self.camera_calibration_targets[camera] = objective
                all_cameras.add(camera)
                all_objectives.add(objective)
            elif predicate == 'visible_from':
                objective, waypoint = parts[1], parts[2]
                self.objective_visible_from.setdefault(objective, []).append(waypoint)
                all_objectives.add(objective)
                all_waypoints.add(waypoint)
            elif predicate == 'visible':
                wp1, wp2 = parts[1], parts[2]
                visible_map.setdefault(wp1, []).append(wp2)
                visible_map.setdefault(wp2, []).append(wp1) # Visible is symmetric
                all_waypoints.add(wp1)
                all_waypoints.add(wp2)
            elif predicate == 'can_traverse':
                rover, wp1, wp2 = parts[1], parts[2], parts[3]
                self.rover_traversal_graphs.setdefault(rover, {}).setdefault(wp1, []).append(wp2)
                all_rovers.add(rover)
                all_waypoints.add(wp1)
                all_waypoints.add(wp2)
            # Note: Other predicates like 'rover', 'waypoint', etc. are types and not processed here.

        # Ensure all rovers have capability entries, even if empty
        for r in all_rovers:
             self.rover_capabilities.setdefault(r, {}).setdefault('soil', False)
             self.rover_capabilities[r].setdefault('rock', False)
             self.rover_capabilities[r].setdefault('imaging', False)

        # Ensure all waypoints mentioned in static facts are in traversal graphs for all rovers
        # This prevents BFS from failing on KeyError if a waypoint has no outgoing edges.
        for rover in all_rovers:
            self.rover_traversal_graphs.setdefault(rover, {})
            for wp in all_waypoints:
                 self.rover_traversal_graphs[rover].setdefault(wp, [])


        # Build communication waypoints (visible from lander location)
        if self.lander_location and self.lander_location in visible_map:
             self.communication_waypoints = visible_map[self.lander_location]


    def __call__(self, node):
        state = node.state
        total_cost = 0
        INF = float('inf') # Use float('inf') internally for calculations
        LARGE_FINITE = 1000000 # Return this for unreachable states

        # 2. Parse current state
        rover_locations = {} # {rover: waypoint}
        soil_samples_at = set() # {waypoint}
        rock_samples_at = set() # {waypoint}
        rover_soil_samples = {} # {rover: [waypoint]}
        rover_rock_samples = {} # {rover: [waypoint]}
        rover_images = {}       # {rover: [(objective, mode)]}
        # store_status = {} # {store: 'empty' or 'full'} # Ignoring store for simplicity
        # camera_calibration_status = {} # {camera: {rover: bool}} # Ignoring calibration state

        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]

            if predicate == 'at':
                # Assuming first arg of 'at' is always a rover based on domain structure
                # Check if the object is a known rover
                if parts[1] in self.rover_capabilities:
                     rover_locations[parts[1]] = parts[2]
            elif predicate == 'at_soil_sample':
                soil_samples_at.add(parts[1])
            elif predicate == 'at_rock_sample':
                rock_samples_at.add(parts[1])
            elif predicate == 'have_soil_analysis':
                rover_soil_samples.setdefault(parts[1], []).append(parts[2])
            elif predicate == 'have_rock_analysis':
                rover_rock_samples.setdefault(parts[1], []).append(parts[2])
            elif predicate == 'have_image':
                rover_images.setdefault(parts[1], []).append((parts[2], parts[3]))
            # elif predicate == 'empty':
            #     store_status[parts[1]] = 'empty'
            # elif predicate == 'full':
            #     store_status[parts[1]] = 'full'
            # elif predicate == 'calibrated':
            #     camera, rover = parts[1], parts[2]
            #     camera_calibration_status.setdefault(camera, {})[rover] = True


        # 3. Iterate through goals and calculate cost for unachieved ones
        for goal in self.goals:
            # Goals are represented as strings like '(communicated_soil_data waypoint1)'
            if goal in state:
                continue # Goal already achieved

            goal_parts = get_parts(goal)
            goal_predicate = goal_parts[0]

            min_goal_cost = INF

            if goal_predicate == 'communicated_soil_data':
                waypoint_w = goal_parts[1]
                goal_item_possessed = False

                # Check if any rover already has the sample
                for rover, samples in rover_soil_samples.items():
                    if waypoint_w in samples:
                        goal_item_possessed = True
                        # Rover R has the sample, need to communicate
                        rover_loc = rover_locations.get(rover)
                        if rover_loc:
                            # Find min cost to communicate from current location
                            comm_costs = [get_distance(rover, rover_loc, comm_wp, self.rover_traversal_graphs) + 1
                                          for comm_wp in self.communication_waypoints]
                            min_comm_cost = min(comm_costs) if comm_costs else INF
                            min_goal_cost = min(min_goal_cost, min_comm_cost)

                # If no rover has the sample, it needs to be sampled and communicated
                if not goal_item_possessed:
                    # Check if sample is still at the waypoint
                    if waypoint_w not in soil_samples_at:
                        # Sample is gone and no one has it -> Unreachable goal
                        min_goal_cost = INF
                    else: # Sample is at the waypoint, need to sample and communicate
                        for rover, capabilities in self.rover_capabilities.items():
                            if capabilities.get('soil'): # Find equipped rover
                                rover_loc = rover_locations.get(rover)
                                if rover_loc:
                                    # Cost to sample = navigate to W + sample action
                                    dist_to_sample = get_distance(rover, rover_loc, waypoint_w, self.rover_traversal_graphs)
                                    cost_to_sample = dist_to_sample + 1 if dist_to_sample != INF else INF

                                    if cost_to_sample != INF:
                                        # Cost to communicate from W = navigate from W to comm_wp + communicate action
                                        comm_costs_from_w = [get_distance(rover, waypoint_w, comm_wp, self.rover_traversal_graphs) + 1
                                                             for comm_wp in self.communication_waypoints]
                                        min_comm_cost_from_w = min(comm_costs_from_w) if comm_costs_from_w else INF

                                        if min_comm_cost_from_w != INF:
                                            total_cost_for_rover = cost_to_sample + min_comm_cost_from_w
                                            min_goal_cost = min(min_goal_cost, total_cost_for_rover)


            elif goal_predicate == 'communicated_rock_data':
                waypoint_w = goal_parts[1]
                goal_item_possessed = False

                # Check if any rover already has the sample
                for rover, samples in rover_rock_samples.items():
                    if waypoint_w in samples:
                        goal_item_possessed = True
                        # Rover R has the sample, need to communicate
                        rover_loc = rover_locations.get(rover)
                        if rover_loc:
                            # Find min cost to communicate from current location
                            comm_costs = [get_distance(rover, rover_loc, comm_wp, self.rover_traversal_graphs) + 1
                                          for comm_wp in self.communication_waypoints]
                            min_comm_cost = min(comm_costs) if comm_costs else INF
                            min_goal_cost = min(min_goal_cost, min_comm_cost)

                # If no rover has the sample, it needs to be sampled and communicated
                if not goal_item_possessed:
                    # Check if sample is still at the waypoint
                    if waypoint_w not in rock_samples_at:
                         min_goal_cost = INF # Cannot sample
                    else: # Sample is at the waypoint, need to sample and communicate
                        for rover, capabilities in self.rover_capabilities.items():
                            if capabilities.get('rock'): # Find equipped rover
                                rover_loc = rover_locations.get(rover)
                                if rover_loc:
                                    # Cost to sample = navigate to W + sample action
                                    dist_to_sample = get_distance(rover, rover_loc, waypoint_w, self.rover_traversal_graphs)
                                    cost_to_sample = dist_to_sample + 1 if dist_to_sample != INF else INF

                                    if cost_to_sample != INF:
                                        # Cost to communicate from W = navigate from W to comm_wp + communicate action
                                        comm_costs_from_w = [get_distance(rover, waypoint_w, comm_wp, self.rover_traversal_graphs) + 1
                                                             for comm_wp in self.communication_waypoints]
                                        min_comm_cost_from_w = min(comm_costs_from_w) if comm_costs_from_w else INF

                                        if min_comm_cost_from_w != INF:
                                            total_cost_for_rover = cost_to_sample + min_comm_cost_from_w
                                            min_goal_cost = min(min_goal_cost, total_cost_for_rover)


            elif goal_predicate == 'communicated_image_data':
                objective_o = goal_parts[1]
                mode_m = goal_parts[2]
                goal_item_possessed = False

                # Check if any rover already has the image
                for rover, images in rover_images.items():
                    if (objective_o, mode_m) in images:
                        goal_item_possessed = True
                        # Rover R has the image, need to communicate
                        rover_loc = rover_locations.get(rover)
                        if rover_loc:
                            # Find min cost to communicate from current location
                            comm_costs = [get_distance(rover, rover_loc, comm_wp, self.rover_traversal_graphs) + 1
                                          for comm_wp in self.communication_waypoints]
                            min_comm_cost = min(comm_costs) if comm_costs else INF
                            min_goal_cost = min(min_goal_cost, min_comm_cost)

                # If no rover has the image, it needs to be taken and communicated
                if not goal_item_possessed:
                    # Find suitable rovers, cameras, waypoints
                    suitable_rover_camera_pairs = []
                    for rover, capabilities in self.rover_capabilities.items():
                        if capabilities.get('imaging'):
                            for camera in self.rover_cameras.get(rover, []):
                                if mode_m in self.camera_supports.get(camera, []):
                                    # Check if calibration target and image waypoints exist
                                    cal_target = self.camera_calibration_targets.get(camera)
                                    if cal_target:
                                        cal_wps = self.objective_visible_from.get(cal_target, [])
                                        img_wps = self.objective_visible_from.get(objective_o, [])
                                        if cal_wps and img_wps:
                                            suitable_rover_camera_pairs.append((rover, camera, cal_wps, img_wps))

                    if not suitable_rover_camera_pairs:
                         min_goal_cost = INF # Cannot take image
                    else: # Can take image, need to take and communicate
                        for rover, camera, cal_wps, img_wps in suitable_rover_camera_pairs:
                            rover_loc = rover_locations.get(rover)
                            if rover_loc:
                                # Cost to take image = navigate to cal_wp + calibrate + navigate to img_wp + take_image
                                # Find min cost path R_loc -> cal_wp -> img_wp
                                min_img_acq_cost_for_pair = INF
                                for cal_wp in cal_wps:
                                    dist_to_cal = get_distance(rover, rover_loc, cal_wp, self.rover_traversal_graphs)
                                    if dist_to_cal != INF:
                                        for img_wp in img_wps:
                                            dist_cal_to_img = get_distance(rover, cal_wp, img_wp, self.rover_traversal_graphs)
                                            if dist_cal_to_img != INF:
                                                # Cost: nav1 + calibrate + nav2 + take_image
                                                img_acq_cost = dist_to_cal + 1 + dist_cal_to_img + 1
                                                min_img_acq_cost_for_pair = min(min_img_acq_cost_for_pair, img_acq_cost)

                                if min_img_acq_cost_for_pair != INF:
                                    # Cost to communicate from img_wp = navigate from img_wp to comm_wp + communicate action
                                    min_comm_cost_from_img = INF
                                    for img_wp in img_wps:
                                         comm_costs_from_img = [get_distance(rover, img_wp, comm_wp, self.rover_traversal_graphs) + 1
                                                               for comm_wp in self.communication_waypoints]
                                         min_comm_cost_from_img = min(min_comm_cost_from_img, min(comm_costs_from_img) if comm_costs_from_img else INF)

                                    if min_comm_cost_from_img != INF:
                                        total_cost_for_pair = min_img_acq_cost_for_pair + min_comm_cost_from_img
                                        min_goal_cost = min(min_goal_cost, total_cost_for_pair)


            # Add the minimum cost for this goal to the total heuristic cost
            if min_goal_cost == INF:
                 # If any goal is unreachable, the whole state is effectively infinite cost
                 # Return a large finite number immediately
                 return LARGE_FINITE
            total_cost += min_goal_cost

        # If the loop finished and total_cost is 0, it means all goals were in the state.
        # If total_cost > 0, it means there were unachieved goals, and they were reachable.
        # The heuristic is 0 only if all goals are met.
        return total_cost

# </code-file-heuristic-rovers>
