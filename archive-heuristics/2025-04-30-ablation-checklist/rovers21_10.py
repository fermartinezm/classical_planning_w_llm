import sys
from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic

# Set a large value for unreachable goals
UNREACHABLE = sys.maxsize

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty facts or malformed strings gracefully
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

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

def build_rover_graph(rover, static_facts):
    """Build an adjacency list graph for a specific rover's navigation."""
    graph = {}
    waypoints = set()

    # Collect all waypoints and initialize graph
    for fact in static_facts:
        if match(fact, "at", rover, "*"):
             waypoints.add(get_parts(fact)[2])
        if match(fact, "at_lander", "*", "*"):
             waypoints.add(get_parts(fact)[2])
        if match(fact, "visible", "*", "*"):
             waypoints.add(get_parts(fact)[1])
             waypoints.add(get_parts(fact)[2])
        if match(fact, "can_traverse", rover, "*", "*"):
             waypoints.add(get_parts(fact)[2])
             waypoints.add(get_parts(fact)[3])

    for wp in waypoints:
        graph[wp] = []

    # Add edges based on can_traverse and visible
    for fact in static_facts:
        if match(fact, "can_traverse", rover, "*", "*"):
            _, r, wp1, wp2 = get_parts(fact)
            if r == rover:
                 # Check if visible is also true (precondition for navigate)
                 if any(match(v_fact, "visible", wp1, wp2) for v_fact in static_facts):
                    if wp1 in graph and wp2 in graph: # Ensure waypoints are in our collected set
                        graph[wp1].append(wp2)

    return graph

def bfs(start_wp, end_wp, graph):
    """
    Perform BFS to find the shortest path cost (number of moves)
    between two waypoints in a given graph.
    """
    if start_wp == end_wp:
        return 0

    queue = deque([(start_wp, 0)])
    visited = {start_wp}

    while queue:
        current_wp, dist = queue.popleft()

        if current_wp == end_wp:
            return dist

        if current_wp in graph:
            for neighbor in graph[current_wp]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))

    return UNREACHABLE # Not reachable

def get_rover_location(state, rover):
    """Find the current waypoint of a specific rover in the state."""
    for fact in state:
        if match(fact, "at", rover, "*"):
            return get_parts(fact)[2]
    return None # Rover location not found (shouldn't happen in valid states)

class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    Estimates the required number of actions to reach a goal state by summing
    the estimated minimum cost for each unachieved goal predicate independently.

    # Summary of Cost Estimation per Goal Type:
    - Communicate Soil/Rock Data (W):
        - If sample is already held by a rover: Cost is min(nav_cost(rover_loc, comm_wp)) + 1 (communicate).
        - If sample needs to be taken at W: Cost is min(nav_cost(rover_loc, W) + (1 if store full) + 1 (sample) + nav_cost(W, comm_wp) + 1 (communicate)).
        - If sample at W is gone and no rover has it: Unreachable (Inf).
    - Communicate Image Data (O, M):
        - If image is already held by a rover: Cost is min(nav_cost(rover_loc, comm_wp)) + 1 (communicate).
        - If image needs to be taken: Cost is min(nav_cost(rover_loc, CalibWP) + 1 (calibrate) + nav_cost(CalibWP, ImageWP) + 1 (take_image) + nav_cost(ImageWP, comm_wp) + 1 (communicate)).
        - If no suitable rover/camera/waypoint exists: Unreachable (Inf).

    Navigation costs (nav_cost) are calculated using BFS on the rover's specific
    traversal graph. Minimums are taken over relevant rovers, waypoints, etc.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static_facts = task.static

        # --- Precompute Static Information ---

        self.rovers = {get_parts(fact)[1] for fact in self.static_facts if match(fact, "equipped_for_*", "*")} | \
                      {get_parts(fact)[2] for fact in self.static_facts if match(fact, "on_board", "*", "*")} | \
                      {get_parts(fact)[2] for fact in self.static_facts if match(fact, "store_of", "*", "*")}

        self.waypoints = {get_parts(fact)[2] for fact in self.static_facts if match(fact, "at_lander", "*", "*")} | \
                         {get_parts(fact)[1] for fact in self.static_facts if match(fact, "visible", "*", "*")} | \
                         {get_parts(fact)[2] for fact in self.static_facts if match(fact, "visible", "*", "*")} | \
                         {get_parts(fact)[2] for fact in self.static_facts if match(fact, "can_traverse", "*", "*", "*")} | \
                         {get_parts(fact)[3] for fact in self.static_facts if match(fact, "can_traverse", "*", "*", "*")} | \
                         {get_parts(fact)[2] for fact in self.static_facts if match(fact, "visible_from", "*", "*")} | \
                         {get_parts(fact)[2] for fact in task.initial_state if match(fact, "at_soil_sample", "*")} | \
                         {get_parts(fact)[2] for fact in task.initial_state if match(fact, "at_rock_sample", "*")}


        self.landers = {get_parts(fact)[1] for fact in self.static_facts if match(fact, "at_lander", "*", "*")}
        self.lander_locations = {get_parts(fact)[1]: get_parts(fact)[2] for fact in self.static_facts if match(fact, "at_lander", "*", "*")}

        self.cameras = {get_parts(fact)[1] for fact in self.static_facts if match(fact, "on_board", "*", "*")}
        self.objectives = {get_parts(fact)[1] for fact in self.static_facts if match(fact, "visible_from", "*", "*")} | \
                          {get_parts(fact)[2] for fact in self.static_facts if match(fact, "calibration_target", "*", "*")}

        self.modes = {get_parts(fact)[2] for fact in self.static_facts if match(fact, "supports", "*", "*")}

        self.rover_nav_graphs = {rover: build_rover_graph(rover, self.static_facts) for rover in self.rovers}

        self.comm_waypoins = set()
        for lander_loc in self.lander_locations.values():
            for wp1 in self.waypoints:
                if any(match(fact, "visible", wp1, lander_loc) for fact in self.static_facts):
                    self.comm_waypoins.add(wp1)

        self.rover_stores = {get_parts(fact)[2]: get_parts(fact)[1] for fact in self.static_facts if match(fact, "store_of", "*", "*")}
        self.rover_equipment = {}
        for rover in self.rovers:
            self.rover_equipment[rover] = set()
            if any(match(fact, "equipped_for_soil_analysis", rover) for fact in self.static_facts):
                self.rover_equipment[rover].add("soil")
            if any(match(fact, "equipped_for_rock_analysis", rover) for fact in self.static_facts):
                self.rover_equipment[rover].add("rock")
            if any(match(fact, "equipped_for_imaging", rover) for fact in self.static_facts):
                self.rover_equipment[rover].add("imaging")

        self.rover_cameras = {}
        for rover in self.rovers:
            self.rover_cameras[rover] = {get_parts(fact)[1] for fact in self.static_facts if match(fact, "on_board", "*", rover)}

        self.camera_modes = {}
        for camera in self.cameras:
            self.camera_modes[camera] = {get_parts(fact)[2] for fact in self.static_facts if match(fact, "supports", camera, "*")}

        self.camera_targets = {get_parts(fact)[1]: get_parts(fact)[2] for fact in self.static_facts if match(fact, "calibration_target", "*", "*")}

        self.objective_visible_wps = {}
        for obj in self.objectives:
            self.objective_visible_wps[obj] = {get_parts(fact)[2] for fact in self.static_facts if match(fact, "visible_from", obj, "*")}

        self.target_visible_wps = {}
        for cam in self.cameras:
            if cam in self.camera_targets:
                target = self.camera_targets[cam]
                self.target_visible_wps[target] = {get_parts(fact)[2] for fact in self.static_facts if match(fact, "visible_from", target, "*")}


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        total_cost = 0

        # Keep track of samples/images already accounted for in the heuristic sum
        # This is a simple way to avoid double-counting if multiple rovers have the same sample/image
        accounted_soil_goals = set()
        accounted_rock_goals = set()
        accounted_image_goals = set()

        for goal in self.goals:
            if goal in state:
                continue # Goal already achieved

            goal_parts = get_parts(goal)
            goal_type = goal_parts[0]

            if goal_type == "communicated_soil_data":
                waypoint = goal_parts[1]
                if waypoint in accounted_soil_goals:
                    continue # Already accounted for this waypoint's soil data goal

                min_goal_cost = UNREACHABLE

                # Check if any rover already has the sample
                rover_with_sample = None
                for rover in self.rovers:
                    if f"(have_soil_analysis {rover} {waypoint})" in state:
                        rover_with_sample = rover
                        break # Found a rover with the sample

                if rover_with_sample:
                    # Sample is held, just need to communicate
                    current_wp = get_rover_location(state, rover_with_sample)
                    if current_wp:
                        min_nav_to_comm = UNREACHABLE
                        for comm_wp in self.comm_waypoins:
                            nav_cost = bfs(current_wp, comm_wp, self.rover_nav_graphs[rover_with_sample])
                            min_nav_to_comm = min(min_nav_to_comm, nav_cost)

                        if min_nav_to_comm != UNREACHABLE:
                            min_goal_cost = min_nav_to_comm + 1 # +1 for communicate action

                else:
                    # Sample needs to be taken and communicated
                    if f"(at_soil_sample {waypoint})" in state: # Check if sample still exists at the waypoint
                        for rover in self.rovers:
                            if "soil" in self.rover_equipment.get(rover, set()): # Check if rover is equipped
                                current_wp = get_rover_location(state, rover)
                                if current_wp:
                                    nav_cost_to_sample = bfs(current_wp, waypoint, self.rover_nav_graphs[rover])

                                    if nav_cost_to_sample != UNREACHABLE:
                                        rover_cost = nav_cost_to_sample
                                        store = self.rover_stores.get(rover)
                                        if store and f"(full {store})" in state:
                                            rover_cost += 1 # +1 for drop action
                                        rover_cost += 1 # +1 for sample_soil action

                                        # Now need to navigate from sample waypoint to communication waypoint
                                        min_nav_to_comm = UNREACHABLE
                                        for comm_wp in self.comm_waypoins:
                                            nav_cost = bfs(waypoint, comm_wp, self.rover_nav_graphs[rover])
                                            min_nav_to_comm = min(min_nav_to_comm, nav_cost)

                                        if min_nav_to_comm != UNREACHABLE:
                                            rover_cost += min_nav_to_comm + 1 # +1 for communicate action
                                            min_goal_cost = min(min_goal_cost, rover_cost)
                    # If sample is not in state and no rover has it, it's unreachable for sampling.

                if min_goal_cost != UNREACHABLE:
                    total_cost += min_goal_cost
                    accounted_soil_goals.add(waypoint) # Mark this waypoint's soil goal as accounted

            elif goal_type == "communicated_rock_data":
                waypoint = goal_parts[1]
                if waypoint in accounted_rock_goals:
                    continue # Already accounted for this waypoint's rock data goal

                min_goal_cost = UNREACHABLE

                # Check if any rover already has the sample
                rover_with_sample = None
                for rover in self.rovers:
                    if f"(have_rock_analysis {rover} {waypoint})" in state:
                        rover_with_sample = rover
                        break # Found a rover with the sample

                if rover_with_sample:
                    # Sample is held, just need to communicate
                    current_wp = get_rover_location(state, rover_with_sample)
                    if current_wp:
                        min_nav_to_comm = UNREACHABLE
                        for comm_wp in self.comm_waypoins:
                            nav_cost = bfs(current_wp, comm_wp, self.rover_nav_graphs[rover_with_sample])
                            min_nav_to_comm = min(min_nav_to_comm, nav_cost)

                        if min_nav_to_comm != UNREACHABLE:
                            min_goal_cost = min_nav_to_comm + 1 # +1 for communicate action

                else:
                    # Sample needs to be taken and communicated
                    if f"(at_rock_sample {waypoint})" in state: # Check if sample still exists at the waypoint
                        for rover in self.rovers:
                            if "rock" in self.rover_equipment.get(rover, set()): # Check if rover is equipped
                                current_wp = get_rover_location(state, rover)
                                if current_wp:
                                    nav_cost_to_sample = bfs(current_wp, waypoint, self.rover_nav_graphs[rover])

                                    if nav_cost_to_sample != UNREACHABLE:
                                        rover_cost = nav_cost_to_sample
                                        store = self.rover_stores.get(rover)
                                        if store and f"(full {store})" in state:
                                            rover_cost += 1 # +1 for drop action
                                        rover_cost += 1 # +1 for sample_rock action

                                        # Now need to navigate from sample waypoint to communication waypoint
                                        min_nav_to_comm = UNREACHABLE
                                        for comm_wp in self.comm_waypoins:
                                            nav_cost = bfs(waypoint, comm_wp, self.rover_nav_graphs[rover])
                                            min_nav_to_comm = min(min_nav_to_comm, nav_cost)

                                        if min_nav_to_comm != UNREACHABLE:
                                            rover_cost += min_nav_to_comm + 1 # +1 for communicate action
                                            min_goal_cost = min(min_goal_cost, rover_cost)
                    # If sample is not in state and no rover has it, it's unreachable for sampling.

                if min_goal_cost != UNREACHABLE:
                    total_cost += min_goal_cost
                    accounted_rock_goals.add(waypoint) # Mark this waypoint's rock goal as accounted


            elif goal_type == "communicated_image_data":
                objective, mode = goal_parts[1], goal_parts[2]
                image_goal_key = (objective, mode)
                if image_goal_key in accounted_image_goals:
                    continue # Already accounted for this image goal

                min_goal_cost = UNREACHABLE

                # Check if any rover already has the image
                rover_with_image = None
                for rover in self.rovers:
                    if f"(have_image {rover} {objective} {mode})" in state:
                        rover_with_image = rover
                        break # Found a rover with the image

                if rover_with_image:
                    # Image is held, just need to communicate
                    current_wp = get_rover_location(state, rover_with_image)
                    if current_wp:
                        min_nav_to_comm = UNREACHABLE
                        for comm_wp in self.comm_waypoins:
                            nav_cost = bfs(current_wp, comm_wp, self.rover_nav_graphs[rover_with_image])
                            min_nav_to_comm = min(min_nav_to_comm, nav_cost)

                        if min_nav_to_comm != UNREACHABLE:
                            min_goal_cost = min_nav_to_comm + 1 # +1 for communicate action

                else:
                    # Image needs to be taken and communicated
                    for rover in self.rovers:
                        if "imaging" in self.rover_equipment.get(rover, set()): # Check if rover is equipped
                            current_wp = get_rover_location(state, rover)
                            if not current_wp: continue

                            for camera in self.rover_cameras.get(rover, set()):
                                if mode in self.camera_modes.get(camera, set()): # Check if camera supports mode
                                    calib_target = self.camera_targets.get(camera)
                                    if not calib_target: continue # Camera has no calibration target

                                    calib_wps = self.target_visible_wps.get(calib_target, set())
                                    image_wps = self.objective_visible_wps.get(objective, set())

                                    if not calib_wps or not image_wps: continue # Cannot calibrate or take image

                                    min_take_and_comm_cost_for_rover_cam = UNREACHABLE

                                    for calib_wp in calib_wps:
                                        nav_cost_to_calib = bfs(current_wp, calib_wp, self.rover_nav_graphs[rover])
                                        if nav_cost_to_calib == UNREACHABLE: continue

                                        for image_wp in image_wps:
                                            nav_cost_calib_to_image = bfs(calib_wp, image_wp, self.rover_nav_graphs[rover])
                                            if nav_cost_calib_to_image == UNREACHABLE: continue

                                            # Cost to take image: Nav to Calib + Calib + Nav to Image + Take Image
                                            take_image_cost = nav_cost_to_calib + 1 + nav_cost_calib_to_image + 1

                                            # Cost to communicate: Nav from ImageWP to CommWP + Communicate
                                            min_nav_image_to_comm = UNREACHABLE
                                            for comm_wp in self.comm_waypoins:
                                                nav_cost = bfs(image_wp, comm_wp, self.rover_nav_graphs[rover])
                                                min_nav_image_to_comm = min(min_nav_image_to_comm, nav_cost)

                                            if min_nav_image_to_comm != UNREACHABLE:
                                                total_action_cost = take_image_cost + min_nav_image_to_comm + 1 # +1 for communicate
                                                min_take_and_comm_cost_for_rover_cam = min(min_take_and_comm_cost_for_rover_cam, total_action_cost)

                                    min_goal_cost = min(min_goal_cost, min_take_and_comm_cost_for_rover_cam)


                if min_goal_cost != UNREACHABLE:
                    total_cost += min_goal_cost
                    accounted_image_goals.add(image_goal_key) # Mark this image goal as accounted

        # If any goal is unreachable, the total heuristic should be infinity
        if total_cost >= UNREACHABLE:
             return UNREACHABLE
        return total_cost

