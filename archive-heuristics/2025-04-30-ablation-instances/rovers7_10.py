from collections import defaultdict, deque
from heuristics.heuristic_base import Heuristic

# Helper function to parse facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

# Helper function for BFS shortest path
def shortest_path(graph, start, end_set):
    """
    Finds the shortest path distance from a start waypoint to any waypoint in a set of end waypoints
    in a given graph.

    Args:
        graph: Adjacency list representation of the graph {waypoint: [neighbor1, neighbor2, ...]}
               This graph is specific to a rover's can_traverse.
        start: The starting waypoint.
        end_set: A set of destination waypoints.

    Returns:
        The minimum distance (number of edges) to reach any waypoint in end_set,
        or float('inf') if no waypoint in end_set is reachable.
    """
    if not end_set: # Cannot reach an empty set of destinations
        return float('inf')
    if start in end_set:
        return 0
    if start not in graph: # Start node not in this rover's graph
        return float('inf')

    queue = deque([(start, 0)])
    visited = {start}

    while queue:
        current_wp, dist = queue.popleft()

        # Check if current_wp is in graph keys before iterating neighbors
        if current_wp in graph:
            for neighbor in graph[current_wp]:
                if neighbor in end_set:
                    return dist + 1
                # Ensure neighbor is also a valid node in the graph before adding
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))

    return float('inf') # No path found to any waypoint in end_set


class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the total number of actions required to satisfy all
    outstanding goal conditions. It calculates the minimum cost for each unsatisfied
    goal independently and sums these costs. The cost for each goal involves
    estimating navigation steps using shortest paths on the rover's traverse graph,
    plus the costs of sampling/imaging and communication actions.

    # Assumptions
    - The heuristic assumes that resources (rovers, stores, cameras) can be
      reused for different goals without conflict (ignores resource contention).
    - It assumes that required samples, calibration targets, and imaging
      objectives are available at their specified locations.
    - Navigation cost between waypoints for a specific rover is estimated
      using the shortest path distance on the graph defined by the
      `can_traverse` predicates for that rover.
    - For imaging goals requiring calibration, the navigation cost is estimated
      as the sum of shortest paths between the rover's location, a calibration
      waypoint, an imaging waypoint, and a communication waypoint. This is a
      relaxation that doesn't guarantee a single continuous path but provides
      a lower bound estimate for the navigation sequence.

    # Heuristic Initialization
    During initialization, the heuristic processes static facts from the task
    to build data structures needed for cost calculation:
    - `lander_location`: The waypoint where the lander is located.
    - `communication_points`: A set of waypoints visible from the lander location.
    - `rover_capabilities`: Maps each rover to its analysis/imaging capabilities.
    - `rover_stores`: Maps each rover to its store object.
    - `rover_cameras`: Maps each rover to the cameras it has on board.
    - `camera_modes`: Maps each camera to the set of modes it supports.
    - `camera_calibration_targets`: Maps each camera to its calibration target objective.
    - `visible_from_objective`: Maps each objective to the set of waypoints from which it is visible.
    - `visible_from_calibration_target`: Maps each calibration target objective to the set of waypoints from which it is visible.
    - `can_traverse_graph`: Maps each rover to its waypoint adjacency list based on `can_traverse` facts.
    - `goals`: Stores the set of goal predicates.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1.  Identify the current location of each rover, which stores are full,
        which data (soil, rock, image) each rover possesses, and which cameras
        are calibrated. Also, identify which communication goals are already met.
    2.  Initialize the total heuristic cost to 0.
    3.  Iterate through each goal predicate defined in the task.
    4.  If a goal predicate is already satisfied in the current state, add 0 cost for this goal.
    5.  If a goal predicate is *not* satisfied, estimate the minimum cost to satisfy it:
        *   **For `(communicated_soil_data ?w)`:**
            *   Check if any rover already has `(have_soil_analysis ?r ?w)`.
                *   If yes: The cost is the minimum navigation steps for any such rover `r` from its current location to *any* communication point, plus 1 for the `communicate_soil_data` action. Find the minimum cost over all rovers that have the data.
                *   If no: The cost is the minimum cost over all soil-equipped rovers `r`. The cost for a rover `r` is:
                    *   Navigation from its current location to waypoint `w`.
                    *   Plus 1 if its store is full (for the `drop` action).
                    *   Plus 1 for the `sample_soil` action.
                    *   Plus minimum navigation steps from waypoint `w` to *any* communication point.
                    *   Plus 1 for the `communicate_soil_data` action.
        *   **For `(communicated_rock_data ?w)`:** Similar calculation as for soil data, using rock-equipped rovers and rock samples.
        *   **For `(communicated_image_data ?o ?m)`:**
            *   Check if any rover already has `(have_image ?r ?o ?m)`.
                *   If yes: The cost is the minimum navigation steps for any such rover `r` from its current location to *any* communication point, plus 1 for the `communicate_image_data` action. Find the minimum cost over all rovers that have the image.
                *   If no: The cost is the minimum cost over all imaging-equipped rovers `r` with a camera `i` supporting mode `m`. The cost for a rover `r` and camera `i` depends on whether the camera is currently calibrated:
                    *   If camera `i` on rover `r` is calibrated:
                        *   Cost is minimum navigation from `r`'s current location to *any* waypoint visible from objective `o`.
                        *   Plus 1 for the `take_image` action.
                        *   Plus minimum navigation from *any* waypoint visible from objective `o` to *any* communication point.
                        *   Plus 1 for the `communicate_image_data` action.
                    *   If camera `i` on rover `r` is *not* calibrated:
                        *   Cost is minimum navigation from `r`'s current location to *any* waypoint visible from `i`'s calibration target.
                        *   Plus 1 for the `calibrate` action.
                        *   Plus minimum navigation from *any* waypoint visible from `i`'s calibration target to *any* waypoint visible from objective `o`.
                        *   Plus 1 for the `take_image` action.
                        *   Plus minimum navigation from *any* waypoint visible from objective `o` to *any* communication point.
                        *   Plus 1 for the `communicate_image_data` action.
                    *   Find the minimum cost over all suitable rovers and cameras.
        *   If the minimum cost for a goal is infinity (e.g., required waypoints are unreachable), the total heuristic for the state is infinity.
    6.  Sum the minimum costs for all unsatisfied goals to get the total heuristic value.
    7.  Return the total heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # Data structures to store static information
        self.lander_location = None
        self.communication_points = set()
        self.rover_capabilities = defaultdict(set) # {rover: {cap1, cap2}}
        self.rover_stores = {} # {rover: store}
        self.rover_cameras = defaultdict(list) # {rover: [cam1, cam2]}
        self.camera_modes = defaultdict(set) # {camera: {mode1, mode2}}
        self.camera_calibration_targets = {} # {camera: objective}
        self.visible_from_objective = defaultdict(set) # {objective: {wp1, wp2}}
        self.visible_from_calibration_target = defaultdict(set) # {objective: {wp1, wp2}}
        self.can_traverse_graph = defaultdict(lambda: defaultdict(list)) # {rover: {wp1: [wp2, wp3]}}

        # Parse static facts
        for fact in static_facts:
            parts = get_parts(fact)
            pred = parts[0]

            if pred == 'at_lander':
                # Assuming only one lander
                self.lander_location = parts[2]
            elif pred == 'visible':
                wp1, wp2 = parts[1:]
                # We need waypoints ?x such that (visible ?x lander_loc) for communication
                if self.lander_location:
                     if wp2 == self.lander_location:
                         self.communication_points.add(wp1)
                     if wp1 == self.lander_location: # visible is often bidirectional, handle both orders
                         self.communication_points.add(wp2)
            elif pred.startswith('equipped_for_'):
                cap_type = pred[len('equipped_for_'):] # soil_analysis, rock_analysis, imaging
                rover = parts[1]
                self.rover_capabilities[rover].add(cap_type)
            elif pred == 'store_of':
                store, rover = parts[1:]
                self.rover_stores[rover] = store
            elif pred == 'on_board':
                camera, rover = parts[1:]
                self.rover_cameras[rover].append(camera)
            elif pred == 'supports':
                camera, mode = parts[1:]
                self.camera_modes[camera].add(mode)
            elif pred == 'calibration_target':
                camera, objective = parts[1:]
                self.camera_calibration_targets[camera] = objective
            elif pred == 'visible_from':
                objective, waypoint = parts[1:]
                self.visible_from_objective[objective].add(waypoint)
            elif pred == 'can_traverse':
                rover, w1, w2 = parts[1:]
                self.can_traverse_graph[rover][w1].append(w2)
                self.can_traverse_graph[rover][w2].append(w1) # Assuming bidirectional traverse

        # Populate visible_from_calibration_target based on visible_from_objective and calibration_target
        for cam, cal_obj in self.camera_calibration_targets.items():
             if cal_obj in self.visible_from_objective:
                 self.visible_from_calibration_target[cal_obj].update(self.visible_from_objective[cal_obj])


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Extract relevant information from the current state
        rover_locations = {}
        rover_stores_full = set()
        rover_soil_data = defaultdict(set) # {rover: {wp1, wp2}}
        rover_rock_data = defaultdict(set) # {rover: {wp1, wp2}}
        rover_image_data = defaultdict(set) # {rover: {(obj1, mode1), (obj2, mode2)}}
        rover_calibrated_cameras = defaultdict(set) # {rover: {cam1, cam2}}
        communicated_soil = set()
        communicated_rock = set()
        communicated_image = set() # {(obj1, mode1), (obj2, mode2)}

        for fact in state:
            parts = get_parts(fact)
            pred = parts[0]

            if pred == 'at':
                obj, loc = parts[1:]
                if obj.startswith('rover'):
                     rover_locations[obj] = loc
            elif pred == 'full':
                store = parts[1]
                # Find which rover this store belongs to
                for r, s in self.rover_stores.items():
                    if s == store:
                        rover_stores_full.add(r)
                        break
            elif pred == 'have_soil_analysis':
                r, wp = parts[1:]
                rover_soil_data[r].add(wp)
            elif pred == 'have_rock_analysis':
                r, wp = parts[1:]
                rover_rock_data[r].add(wp)
            elif pred == 'have_image':
                r, obj, mode = parts[1:]
                rover_image_data[r].add((obj, mode))
            elif pred == 'calibrated':
                cam, r = parts[1:]
                rover_calibrated_cameras[r].add(cam)
            elif pred == 'communicated_soil_data':
                wp = parts[1]
                communicated_soil.add(wp)
            elif pred == 'communicated_rock_data':
                wp = parts[1]
                communicated_rock.add(wp)
            elif pred == 'communicated_image_data':
                obj, mode = parts[1:]
                communicated_image.add((obj, mode))

        total_cost = 0

        # Iterate through goals and sum costs for unsatisfied ones
        for goal_fact in self.goals:
            parts = get_parts(goal_fact)
            pred = parts[0]

            if pred == 'communicated_soil_data':
                wp = parts[1]
                if wp in communicated_soil:
                    continue # Goal already met

                min_goal_cost = float('inf')

                # Case 1: Data already collected by some rover
                found_have_data = False
                for r, wps_with_data in rover_soil_data.items():
                    if wp in wps_with_data:
                        found_have_data = True
                        loc_r = rover_locations.get(r)
                        if loc_r is None: continue # Should not happen in a valid state

                        # Cost = Nav(current_loc(r) -> comm_point) + 1
                        graph_r = self.can_traverse_graph.get(r, {})
                        nav_cost = shortest_path(graph_r, loc_r, self.communication_points)
                        if nav_cost != float('inf'):
                            min_goal_cost = min(min_goal_cost, nav_cost + 1)

                # Case 2: Need to sample and communicate
                if not found_have_data:
                    # Find capable rovers
                    capable_rovers = [r for r, caps in self.rover_capabilities.items() if 'soil_analysis' in caps]
                    if not capable_rovers:
                         min_goal_cost = float('inf') # Cannot achieve this goal
                    else:
                        for r in capable_rovers:
                            loc_r = rover_locations.get(r)
                            if loc_r is None: continue

                            # Cost = Nav(loc_r -> w) + (1 if store full) + 1 (sample) + Nav(w -> comm_point) + 1 (communicate)
                            graph_r = self.can_traverse_graph.get(r, {})
                            nav_to_sample = shortest_path(graph_r, loc_r, {wp})
                            if nav_to_sample == float('inf'): continue

                            store_full_cost = 1 if r in rover_stores_full else 0
                            sample_cost = 1

                            nav_to_comm = shortest_path(graph_r, wp, self.communication_points)
                            if nav_to_comm == float('inf'): continue

                            communicate_cost = 1

                            current_rover_cost = nav_to_sample + store_full_cost + sample_cost + nav_to_comm + communicate_cost
                            min_goal_cost = min(min_goal_cost, current_rover_cost)

                if min_goal_cost == float('inf'):
                    return float('inf') # Indicate unreachable goal
                total_cost += min_goal_cost


            elif pred == 'communicated_rock_data':
                wp = parts[1]
                if wp in communicated_rock:
                    continue # Goal already met

                min_goal_cost = float('inf')

                # Case 1: Data already collected by some rover
                found_have_data = False
                for r, wps_with_data in rover_rock_data.items():
                    if wp in wps_with_data:
                        found_have_data = True
                        loc_r = rover_locations.get(r)
                        if loc_r is None: continue

                        # Cost = Nav(current_loc(r) -> comm_point) + 1
                        graph_r = self.can_traverse_graph.get(r, {})
                        nav_cost = shortest_path(graph_r, loc_r, self.communication_points)
                        if nav_cost != float('inf'):
                            min_goal_cost = min(min_goal_cost, nav_cost + 1)

                # Case 2: Need to sample and communicate
                if not found_have_data:
                    # Find capable rovers
                    capable_rovers = [r for r, caps in self.rover_capabilities.items() if 'rock_analysis' in caps]
                    if not capable_rovers:
                         min_goal_cost = float('inf')
                    else:
                        for r in capable_rovers:
                            loc_r = rover_locations.get(r)
                            if loc_r is None: continue

                            # Cost = Nav(loc_r -> w) + (1 if store full) + 1 (sample) + Nav(w -> comm_point) + 1 (communicate)
                            graph_r = self.can_traverse_graph.get(r, {})
                            nav_to_sample = shortest_path(graph_r, loc_r, {wp})
                            if nav_to_sample == float('inf'): continue

                            store_full_cost = 1 if r in rover_stores_full else 0
                            sample_cost = 1

                            nav_to_comm = shortest_path(graph_r, wp, self.communication_points)
                            if nav_to_comm == float('inf'): continue

                            communicate_cost = 1

                            current_rover_cost = nav_to_sample + store_full_cost + sample_cost + nav_to_comm + communicate_cost
                            min_goal_cost = min(min_goal_cost, current_rover_cost)

                if min_goal_cost == float('inf'):
                    return float('inf') # Unreachable goal
                total_cost += min_goal_cost


            elif pred == 'communicated_image_data':
                obj, mode = parts[1:]
                if (obj, mode) in communicated_image:
                    continue # Goal already met

                min_goal_cost = float('inf')

                # Case 1: Image already taken by some rover
                found_have_image = False
                for r, images_taken in rover_image_data.items():
                    if (obj, mode) in images_taken:
                        found_have_image = True
                        loc_r = rover_locations.get(r)
                        if loc_r is None: continue

                        # Cost = Nav(current_loc(r) -> comm_point) + 1
                        graph_r = self.can_traverse_graph.get(r, {})
                        nav_cost = shortest_path(graph_r, loc_r, self.communication_points)
                        if nav_cost != float('inf'):
                            min_goal_cost = min(min_goal_cost, nav_cost + 1)

                # Case 2: Need to take image and communicate
                if not found_have_image:
                    # Find capable rovers/cameras
                    suitable_rovers_cameras = [] # List of (rover, camera)
                    for r, caps in self.rover_capabilities.items():
                        if 'imaging' in caps:
                            for cam in self.rover_cameras.get(r, []):
                                if mode in self.camera_modes.get(cam, set()):
                                    suitable_rovers_cameras.append((r, cam))

                    if not suitable_rovers_cameras:
                        min_goal_cost = float('inf')
                    else:
                        for r, cam in suitable_rovers_cameras:
                            loc_r = rover_locations.get(r)
                            if loc_r is None: continue

                            graph_r = self.can_traverse_graph.get(r, {})

                            # Cost if already calibrated (check state)
                            if r in rover_calibrated_cameras and cam in rover_calibrated_cameras[r]:
                                # Need to image and communicate
                                # Cost = Nav(loc_r -> img_point) + 1 (take_image) + Nav(img_point -> comm_point) + 1 (communicate)
                                img_points = self.visible_from_objective.get(obj, set())
                                if not img_points: continue # Cannot image this objective

                                nav_loc_to_img = shortest_path(graph_r, loc_r, img_points)
                                if nav_loc_to_img == float('inf'): continue

                                # Min nav from any img point to any comm point
                                min_nav_img_to_comm = float('inf')
                                for img_p in img_points:
                                     nav_img_to_comm = shortest_path(graph_r, img_p, self.communication_points)
                                     min_nav_img_to_comm = min(min_nav_img_to_comm, nav_img_to_comm)

                                if min_nav_img_to_comm == float('inf'): continue

                                current_rover_cost_calibrated = nav_loc_to_img + 1 + min_nav_img_to_comm + 1
                                min_goal_cost = min(min_goal_cost, current_rover_cost_calibrated)

                            else:
                                # Need to calibrate, then image, then communicate
                                # Cost = Nav(loc_r -> cal_point) + 1 (calibrate) + Nav(cal_point -> img_point) + 1 (take_image) + Nav(img_point -> comm_point) + 1 (communicate)

                                cal_target = self.camera_calibration_targets.get(cam)
                                if not cal_target: continue # Camera has no calibration target?
                                cal_points = self.visible_from_calibration_target.get(cal_target, set())
                                if not cal_points: continue # No waypoint to calibrate from?

                                img_points = self.visible_from_objective.get(obj, set())
                                if not img_points: continue # Cannot image this objective

                                # Nav from current location to any calibration point
                                nav_loc_to_cal = shortest_path(graph_r, loc_r, cal_points)
                                if nav_loc_to_cal == float('inf'): continue

                                # Min nav from any calibration point to any image point
                                min_nav_cal_to_img = float('inf')
                                for cal_p in cal_points:
                                    nav_cal_to_img = shortest_path(graph_r, cal_p, img_points)
                                    min_nav_cal_to_img = min(min_cal_to_img, nav_cal_to_img) # Typo fixed: min_cal_to_img -> min_nav_cal_to_img
                                if min_nav_cal_to_img == float('inf'): continue

                                # Min nav from any image point to any communication point
                                min_nav_img_to_comm = float('inf')
                                for img_p in img_points:
                                    nav_img_to_comm = shortest_path(graph_r, img_p, self.communication_points)
                                    min_nav_img_to_comm = min(min_nav_img_to_comm, nav_img_to_comm)
                                if min_nav_img_to_comm == float('inf'): continue

                                current_rover_cost_calibrate = nav_loc_to_cal + 1 + min_nav_cal_to_img + 1 + min_nav_img_to_comm + 1
                                min_goal_cost = min(min_goal_cost, current_rover_cost_calibrate)

                if min_goal_cost == float('inf'):
                    return float('inf') # Indicate unreachable goal
                total_cost += min_goal_cost

        return total_cost
