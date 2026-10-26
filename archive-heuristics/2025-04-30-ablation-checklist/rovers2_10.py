import sys
from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic

# Define a large number to represent infinity for unreachable states/goals
INF = float('inf')

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty strings or malformed facts gracefully
    if not fact or not isinstance(fact, str) or len(fact) < 2:
        return []
    # Remove outer parentheses and split by whitespace
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

def find_fact(state, *pattern):
    """Find the first fact in the state that matches the pattern."""
    for fact in state:
        if match(fact, *pattern):
            return fact
    return None

def find_all_facts(state, *pattern):
    """Find all facts in the state that match the pattern."""
    return [fact for fact in state if match(fact, *pattern)]

def bfs(graph, start_node):
    """
    Perform Breadth-First Search to find shortest distances from a start node
    in an unweighted graph.

    Args:
        graph: Adjacency list representation {node: [neighbor1, neighbor2, ...]}
        start_node: The node to start the search from.

    Returns:
        A dictionary {node: distance} containing shortest distances from start_node
        to all reachable nodes. Unreachable nodes are not included.
    """
    distances = {start_node: 0}
    queue = deque([start_node])
    visited = {start_node}

    while queue:
        current_node = queue.popleft()

        if current_node not in graph:
             continue # Handle nodes that might be in distances/queue but not in graph

        for neighbor in graph[current_node]:
            if neighbor not in visited:
                visited.add(neighbor)
                distances[neighbor] = distances[current_node] + 1
                queue.append(neighbor)

    return distances

class RoversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    This heuristic estimates the cost to reach the goal by summing the estimated
    costs for each unachieved goal predicate independently. For each goal, it
    estimates the minimum actions required by finding the best-suited rover
    and calculating the steps needed (navigation, sampling/imaging, calibration,
    communication), using precomputed shortest path distances for navigation.

    It is non-admissible as it ignores negative interactions and resource
    constraints (like store capacity beyond the current sample, or multiple
    rovers needing the same resource/location simultaneously) and sums costs
    for independent goals.

    Heuristic Components for an unachieved goal (e.g., communicated_soil_data W):
    1. Cost to acquire the data (if not already acquired):
       - Navigation to the sample/image location.
       - Dropping sample if store is full (for soil/rock).
       - Calibration if camera not calibrated (for image).
       - The sample/image action itself.
    2. Cost to communicate the data:
       - Navigation to a waypoint visible from the lander.
       - The communication action itself.

    The heuristic finds the minimum cost for each goal over all suitable rovers
    and relevant waypoints (sample/image locations, calibration locations, lander locations)
    using precomputed navigation distances.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and static facts,
        and precomputing navigation distances.
        """
        self.goals = task.goals
        self.initial_state = task.initial_state # Needed to check initial sample locations
        static_facts = task.static

        # Precompute static information
        self._precompute_static(static_facts)

        # Precompute navigation distances for each rover
        self._precompute_nav_distances()

    def _precompute_static(self, static_facts):
        """Extract and organize static facts for quick lookup."""
        self.lander_waypoints = set()
        self.initial_soil_samples = set()
        self.initial_rock_samples = set()
        self.objective_visible_waypoints = {} # objective -> set of waypoints
        self.camera_calibration_targets = {} # camera -> objective
        self.calibration_target_visible_waypoints = {} # objective (calibration target) -> set of waypoints
        self.rover_capabilities = {} # rover -> set of capabilities ('soil', 'rock', 'imaging')
        self.rover_stores = {} # rover -> store
        self.rover_cameras = {} # rover -> set of cameras
        self.camera_modes = {} # camera -> set of modes
        self.rover_graph = {} # rover -> {waypoint -> set of neighbors}

        all_waypoints = set()
        all_rovers = set()
        all_cameras = set()
        all_objectives = set()

        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]
            if predicate == "at_lander":
                self.lander_waypoints.add(parts[2])
            elif predicate == "at_soil_sample":
                 # Samples can be removed, so check initial state later
                 pass # Initial samples are in task.initial_state
            elif predicate == "at_rock_sample":
                 # Samples can be removed, so check initial state later
                 pass # Initial samples are in task.initial_state
            elif predicate == "equipped_for_soil_analysis":
                rover = parts[1]
                self.rover_capabilities.setdefault(rover, set()).add('soil')
                all_rovers.add(rover)
            elif predicate == "equipped_for_rock_analysis":
                rover = parts[1]
                self.rover_capabilities.setdefault(rover, set()).add('rock')
                all_rovers.add(rover)
            elif predicate == "equipped_for_imaging":
                rover = parts[1]
                self.rover_capabilities.setdefault(rover, set()).add('imaging')
                all_rovers.add(rover)
            elif predicate == "store_of":
                store, rover = parts[1], parts[2]
                self.rover_stores[rover] = store
                all_rovers.add(rover)
            elif predicate == "on_board":
                camera, rover = parts[1], parts[2]
                self.rover_cameras.setdefault(rover, set()).add(camera)
                all_rovers.add(rover)
                all_cameras.add(camera)
            elif predicate == "supports":
                camera, mode = parts[1], parts[2]
                self.camera_modes.setdefault(camera, set()).add(mode)
                all_cameras.add(camera)
            elif predicate == "visible_from":
                objective, waypoint = parts[1], parts[2]
                self.objective_visible_waypoints.setdefault(objective, set()).add(waypoint)
                all_objectives.add(objective)
                all_waypoints.add(waypoint)
            elif predicate == "calibration_target":
                camera, objective = parts[1], parts[2]
                self.camera_calibration_targets[camera] = objective
                all_cameras.add(camera)
                all_objectives.add(objective)
            elif predicate == "visible":
                 wp1, wp2 = parts[1], parts[2]
                 all_waypoints.add(wp1)
                 all_waypoints.add(wp2)
                 # Graph edges built from can_traverse and visible
            elif predicate == "can_traverse":
                 rover, wp1, wp2 = parts[1], parts[2], parts[3]
                 all_rovers.add(rover)
                 all_waypoints.add(wp1)
                 all_waypoints.add(wp2)
                 # Graph edges built from can_traverse and visible

        # Build the navigation graph for each rover
        for rover in all_rovers:
            self.rover_graph[rover] = {}
            for wp in all_waypoints:
                self.rover_graph[rover][wp] = set()

        for fact in static_facts:
             parts = get_parts(fact)
             if not parts or parts[0] != "can_traverse": continue
             rover, wp1, wp2 = parts[1], parts[2], parts[3]
             # Check if wp1 and wp2 are visible from each other (action precondition)
             visible_fact = find_fact(static_facts, "visible", wp1, wp2)
             if visible_fact:
                 self.rover_graph[rover][wp1].add(wp2)
             # The domain states visible is symmetric, but action only requires visible y z, not z y
             # Let's check both directions for robustness based on instance files
             visible_fact_rev = find_fact(static_facts, "visible", wp2, wp1)
             if visible_fact_rev and (wp1, wp2) != (wp2, wp1): # Avoid adding self loop if visible is symmetric
                  # If can_traverse is also symmetric, this might add redundant edges, but BFS handles it.
                  # The action navigate ?x ?y ?z requires (can_traverse ?x ?y ?z) AND (visible ?y ?z).
                  # So an edge exists from y to z for rover x if can_traverse x y z and visible y z.
                  # My graph building was slightly off. Let's fix it.
                  pass # Rebuild graph correctly below

        # Correct graph building: Edge from Y to Z for rover X if (can_traverse X Y Z) and (visible Y Z)
        self.rover_graph = {r: {wp: set() for wp in all_waypoints} for r in all_rovers}
        visible_pairs = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue
            if parts[0] == "visible":
                visible_pairs.add((parts[1], parts[2]))

        for fact in static_facts:
            parts = get_parts(fact)
            if not parts or parts[0] != "can_traverse": continue
            rover, wp_from, wp_to = parts[1], parts[2], parts[3]
            if (wp_from, wp_to) in visible_pairs:
                 self.rover_graph[rover][wp_from].add(wp_to)


        # Populate calibration target visible waypoints
        for camera, target_objective in self.camera_calibration_targets.items():
             if target_objective in self.objective_visible_waypoints:
                  self.calibration_target_visible_waypoints[target_objective] = self.objective_visible_waypoints[target_objective]
             else:
                  self.calibration_target_visible_waypoints[target_objective] = set() # No waypoints visible from target

        # Populate initial sample locations
        self.initial_soil_samples = {get_parts(fact)[1] for fact in task.initial_state if match(fact, "at_soil_sample", "*")}
        self.initial_rock_samples = {get_parts(fact)[1] for fact in task.initial_state if match(fact, "at_rock_sample", "*")}


    def _precompute_nav_distances(self):
        """Compute all-pairs shortest paths for each rover."""
        self.rover_nav_distances = {}
        for rover, graph in self.rover_graph.items():
            self.rover_nav_distances[rover] = {}
            all_waypoints = list(graph.keys()) # Get all waypoints for this rover's graph
            for start_wp in all_waypoints:
                self.rover_nav_distances[rover][start_wp] = bfs(graph, start_wp)

    def nav_dist(self, rover, start_wp, end_wp):
        """Lookup precomputed navigation distance."""
        if rover not in self.rover_nav_distances or \
           start_wp not in self.rover_nav_distances[rover] or \
           end_wp not in self.rover_nav_distances[rover][start_wp]:
            return INF # Unreachable or waypoint not in graph for this rover
        return self.rover_nav_distances[rover][start_wp][end_wp]

    def __call__(self, node):
        """Compute the heuristic estimate for the given state."""
        state = node.state

        # If goal is reached, heuristic is 0
        if self.goals <= state:
            return 0

        total_heuristic = 0
        unachieved_goals = self.goals - state

        for goal in unachieved_goals:
            parts = get_parts(goal)
            if not parts or parts[0] != "communicated_soil_data" and \
                           parts[0] != "communicated_rock_data" and \
                           parts[0] != "communicated_image_data":
                 # Ignore non-communication goals if any exist (shouldn't in this domain)
                 continue

            predicate = parts[0]
            min_goal_cost = INF

            if predicate == "communicated_soil_data":
                waypoint = parts[1]
                min_goal_cost = self._estimate_soil_goal_cost(state, waypoint)

            elif predicate == "communicated_rock_data":
                waypoint = parts[1]
                min_goal_cost = self._estimate_rock_goal_cost(state, waypoint)

            elif predicate == "communicated_image_data":
                objective, mode = parts[1], parts[2]
                min_goal_cost = self._estimate_image_goal_cost(state, objective, mode)

            # If any single goal is impossible, the state is likely a dead end
            if min_goal_cost == INF:
                return INF

            total_heuristic += min_goal_cost

        return total_heuristic

    def _estimate_soil_goal_cost(self, state, waypoint):
        """Estimate cost for a single communicated_soil_data goal."""
        min_cost = INF

        # Check if sample was ever at this waypoint initially
        if f"(at_soil_sample {waypoint})" not in self.initial_state:
             return INF # Sample never existed here

        # Find rovers equipped for soil analysis
        soil_rovers = [r for r, caps in self.rover_capabilities.items() if 'soil' in caps]
        if not soil_rovers:
             return INF # No rover can collect soil

        # Check if sample is still at the waypoint in the current state
        sample_still_present = f"(at_soil_sample {waypoint})" in state

        for rover in soil_rovers:
            rover_cost = 0
            current_rover_loc_fact = find_fact(state, "at", rover, "*")
            if not current_rover_loc_fact: continue # Rover location unknown (shouldn't happen in rovers)
            current_rover_loc = get_parts(current_rover_loc_fact)[2]

            have_analysis = f"(have_soil_analysis {rover} {waypoint})" in state

            # Cost to get the sample (if not already have it)
            loc_after_acquisition = current_rover_loc # Default if already have data
            if not have_analysis:
                if not sample_still_present:
                    continue # Sample is gone and this rover doesn't have the data

                nav_to_sample_cost = self.nav_dist(rover, current_rover_loc, waypoint)
                if nav_to_sample_cost == INF: continue # Cannot reach sample location

                rover_cost += nav_to_sample_cost

                # Check store status
                store = self.rover_stores.get(rover)
                if store and f"(full {store})" in state:
                    rover_cost += 1 # Cost for drop action

                rover_cost += 1 # Cost for sample_soil action
                loc_after_acquisition = waypoint # Rover is at sample location after sampling

            # Cost to communicate the data
            min_nav_to_lander = INF
            for lander_wp in self.lander_waypoints:
                 # Communicate action requires rover at X, lander at Y, and visible X Y
                 # We need a waypoint X visible from lander_wp Y, where rover can reach X
                 # Let's assume the rover needs to navigate to a waypoint X visible from lander_wp
                 # A simpler heuristic: just navigate to the lander_wp itself if visible
                 # The domain says visible ?x ?y for communicate, where ?x is rover loc, ?y is lander loc.
                 # So rover needs to be at a waypoint visible from the lander's waypoint.
                 # Find a waypoint R_WP reachable by the rover from its current location (after acquisition)
                 # such that R_WP is visible from some lander_wp.
                 # The simplest estimate is navigating to *any* lander waypoint visible from *some* waypoint.
                 # Let's find the minimum navigation cost from the rover's location (after acquisition)
                 # to *any* waypoint that is visible from *any* lander waypoint.
                 # This is still complex. Let's simplify: Assume the rover navigates directly to the lander waypoint.
                 # This is not strictly correct based on the predicate (visible ?x ?y), but simplifies the heuristic.
                 # A slightly better simplification: Find the closest waypoint R_WP to the rover (after acquisition)
                 # such that R_WP is visible from *any* lander waypoint.
                 # Let's try the simplest: Navigate to the lander waypoint itself.
                 # The precondition is (visible ?x ?y) where ?x is rover loc, ?y is lander loc.
                 # So the rover must be at a waypoint visible from the lander waypoint.
                 # Let's find the minimum navigation cost from the rover's location (after acquisition)
                 # to *any* waypoint R_WP such that R_WP is visible from *any* lander waypoint L_WP.

                 # Find closest waypoint R_WP visible from any lander_wp
                 closest_visible_from_lander_wp = INF
                 for potential_rover_wp in self.rover_graph.get(rover, {}).keys(): # Any waypoint rover can reach
                      is_visible_from_lander = False
                      for lander_wp_check in self.lander_waypoints:
                           if find_fact(self.initial_state, "visible", potential_rover_wp, lander_wp_check): # Visible is static
                                is_visible_from_lander = True
                                break
                      if is_visible_from_lander:
                           nav = self.nav_dist(rover, loc_after_acquisition, potential_rover_wp)
                           closest_visible_from_lander_wp = min(closest_visible_from_lander_wp, nav)

                 if closest_visible_from_lander_wp == INF:
                      continue # Cannot reach any waypoint visible from lander

                 min_nav_to_lander = closest_visible_from_lander_wp


            if min_nav_to_lander == INF:
                 continue # Cannot reach a communication point

            rover_cost += min_nav_to_lander
            rover_cost += 1 # Cost for communicate_soil_data action

            min_cost = min(min_cost, rover_cost)

        return min_cost

    def _estimate_rock_goal_cost(self, state, waypoint):
        """Estimate cost for a single communicated_rock_data goal."""
        min_cost = INF

        # Check if sample was ever at this waypoint initially
        if f"(at_rock_sample {waypoint})" not in self.initial_state:
             return INF # Sample never existed here

        # Find rovers equipped for rock analysis
        rock_rovers = [r for r, caps in self.rover_capabilities.items() if 'rock' in caps]
        if not rock_rovers:
             return INF # No rover can collect rock

        # Check if sample is still at the waypoint in the current state
        sample_still_present = f"(at_rock_sample {waypoint})" in state

        for rover in rock_rovers:
            rover_cost = 0
            current_rover_loc_fact = find_fact(state, "at", rover, "*")
            if not current_rover_loc_fact: continue
            current_rover_loc = get_parts(current_rover_loc_fact)[2]

            have_analysis = f"(have_rock_analysis {rover} {waypoint})" in state

            # Cost to get the sample (if not already have it)
            loc_after_acquisition = current_rover_loc # Default if already have data
            if not have_analysis:
                if not sample_still_present:
                    continue # Sample is gone and this rover doesn't have the data

                nav_to_sample_cost = self.nav_dist(rover, current_rover_loc, waypoint)
                if nav_to_sample_cost == INF: continue

                rover_cost += nav_to_sample_cost

                # Check store status
                store = self.rover_stores.get(rover)
                if store and f"(full {store})" in state:
                    rover_cost += 1 # Cost for drop action

                rover_cost += 1 # Cost for sample_rock action
                loc_after_acquisition = waypoint # Rover is at sample location after sampling

            # Cost to communicate the data
            min_nav_to_lander = INF
            # Find closest waypoint R_WP visible from any lander_wp
            closest_visible_from_lander_wp = INF
            for potential_rover_wp in self.rover_graph.get(rover, {}).keys(): # Any waypoint rover can reach
                 is_visible_from_lander = False
                 for lander_wp_check in self.lander_waypoints:
                      if find_fact(self.initial_state, "visible", potential_rover_wp, lander_wp_check):
                           is_visible_from_lander = True
                           break
                 if is_visible_from_lander:
                      nav = self.nav_dist(rover, loc_after_acquisition, potential_rover_wp)
                      closest_visible_from_lander_wp = min(closest_visible_from_lander_wp, nav)

            if closest_visible_from_lander_wp == INF:
                 continue # Cannot reach a communication point

            rover_cost += closest_visible_from_lander_wp
            rover_cost += 1 # Cost for communicate_rock_data action

            min_cost = min(min_cost, rover_cost)

        return min_cost


    def _estimate_image_goal_cost(self, state, objective, mode):
        """Estimate cost for a single communicated_image_data goal."""
        min_cost = INF

        # Find waypoints visible from the objective
        image_wps = self.objective_visible_waypoints.get(objective, set())
        if not image_wps:
             return INF # Cannot take image of this objective

        # Find rovers equipped for imaging
        imaging_rovers = [r for r, caps in self.rover_capabilities.items() if 'imaging' in caps]
        if not imaging_rovers:
             return INF # No rover can take images

        for rover in imaging_rovers:
            # Find cameras on board this rover that support the mode
            suitable_cameras = [c for c in self.rover_cameras.get(rover, set()) if mode in self.camera_modes.get(c, set())]
            if not suitable_cameras: continue # Rover has no suitable camera

            current_rover_loc_fact = find_fact(state, "at", rover, "*")
            if not current_rover_loc_fact: continue
            current_rover_loc = get_parts(current_rover_loc_fact)[2]

            for camera in suitable_cameras:
                rover_cam_cost = 0
                have_image = f"(have_image {rover} {objective} {mode})" in state

                # Cost to take the image (if not already have it)
                loc_after_acquisition = current_rover_loc # Default if already have data
                if not have_image:
                    # Find best image waypoint P
                    min_nav_to_P = INF
                    best_P = None
                    for p in image_wps:
                        nav = self.nav_dist(rover, current_rover_loc, p)
                        if nav < min_nav_to_P:
                            min_nav_to_P = nav
                            best_P = p

                    if best_P is None or min_nav_to_P == INF:
                         continue # Cannot reach any image waypoint

                    rover_cam_cost += min_nav_to_P
                    loc_at_image_time = best_P

                    # Cost to calibrate (if camera not calibrated)
                    if f"(calibrated {camera} {rover})" not in state:
                        cal_target = self.camera_calibration_targets.get(camera)
                        if not cal_target:
                             continue # Camera has no calibration target

                        cal_wps = self.calibration_target_visible_waypoints.get(cal_target, set())
                        if not cal_wps:
                             continue # No waypoints visible from calibration target

                        # Find best calibration waypoint W from current image location (loc_at_image_time)
                        min_nav_P_to_W = INF
                        best_W = None
                        for w in cal_wps:
                            nav = self.nav_dist(rover, loc_at_image_time, w)
                            if nav < min_nav_P_to_W:
                                min_nav_P_to_W = nav
                                best_W = w

                        if best_W is None or min_nav_P_to_W == INF:
                             continue # Cannot reach any calibration waypoint from image location

                        rover_cam_cost += min_nav_P_to_W # Nav from image loc to cal loc
                        rover_cam_cost += 1 # calibrate action
                        rover_cam_cost += self.nav_dist(rover, best_W, loc_at_image_time) # Nav back to image loc

                    rover_cam_cost += 1 # take_image action
                    loc_after_acquisition = loc_at_image_time # Rover is at image location after taking image

                # Cost to communicate the data
                min_nav_to_lander = INF
                # Find closest waypoint R_WP visible from any lander_wp
                closest_visible_from_lander_wp = INF
                for potential_rover_wp in self.rover_graph.get(rover, {}).keys(): # Any waypoint rover can reach
                     is_visible_from_lander = False
                     for lander_wp_check in self.lander_waypoints:
                          if find_fact(self.initial_state, "visible", potential_rover_wp, lander_wp_check):
                               is_visible_from_lander = True
                               break
                     if is_visible_from_lander:
                          nav = self.nav_dist(rover, loc_after_acquisition, potential_rover_wp)
                          closest_visible_from_lander_wp = min(closest_visible_from_lander_wp, nav)

                if closest_visible_from_lander_wp == INF:
                     continue # Cannot reach a communication point

                rover_cam_cost += closest_visible_from_lander_wp
                rover_cam_cost += 1 # Cost for communicate_image_data action

                min_cost = min(min_cost, rover_cam_cost)

        return min_cost

# Example usage (assuming Task and Node classes are available)
# task = Task(...) # Load your PDDL task
# heuristic = RoversHeuristic(task)
# state = frozenset(...) # Your current state
# node = Node(state, parent=None, action=None, cost=0) # Create a dummy node
# h_value = heuristic(node)
# print(f"Heuristic value: {h_value}")
