import collections
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle cases like '(at rover1 waypoint1)' -> ['at', 'rover1', 'waypoint1']
    # Handle cases like '(communicated_image_data objective1 high_res)' -> ['communicated_image_data', 'objective1', 'high_res']
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

# BFS helper function
def bfs_distance(graph, start_nodes, target_nodes):
    """
    Finds the minimum shortest path distance from any node in start_nodes
    to any node in target_nodes in the given graph.
    Returns float('inf') if no path exists.
    """
    if not start_nodes or not target_nodes:
        return float('inf')

    start_nodes = set(start_nodes) if not isinstance(start_nodes, set) else start_nodes
    target_nodes = set(target_nodes) if not isinstance(target_nodes, set) else target_nodes

    # Optimization: If any start node is a target node, distance is 0.
    if start_nodes.intersection(target_nodes):
        return 0

    # Filter start nodes to only include those present as keys in the graph
    # This ensures we only start BFS from known waypoints.
    valid_start_nodes = {node for node in start_nodes if node in graph}
    if not valid_start_nodes:
        return float('inf') # Cannot start from any provided node

    queue = collections.deque([(node, 0) for node in valid_start_nodes])
    visited = set(valid_start_nodes)

    while queue:
        (current_node, dist) = queue.popleft()

        if current_node in target_nodes:
            return dist

        # current_node is guaranteed to be in graph keys by valid_start_nodes filter
        for neighbor in graph[current_node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))

    return float('inf') # No path found


class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal state
    by summing the estimated costs for each unachieved goal fact. The cost for
    each goal fact is estimated based on the sequence of actions and navigations
    needed to achieve it from the current state. Navigation costs are estimated
    using Breadth-First Search (BFS) on the waypoint graph.

    # Assumptions
    - All rovers can traverse any edge defined by the `visible` predicate (ignoring `can_traverse` differences for simplicity, based on example instances).
    - Waypoint graph edges are bidirectional if `visible` is defined in both directions.
    - Sampling a soil/rock sample consumes the sample from the waypoint.
    - Taking an image consumes the calibration state of the camera.
    - Store capacity and state (`empty`/`full`) are ignored for sampling costs (simplification).
    - Calibration is always needed before taking an image with a camera unless the camera is currently calibrated (simplification).
    - The heuristic greedily picks the best available rover/waypoint combination for each goal independently.
    - Unreachable goals contribute infinity to the heuristic value.

    # Heuristic Initialization
    - Extract static information from the task:
        - Waypoint graph from `visible` predicates.
        - Lander location and waypoints visible from the lander.
        - Rover capabilities (`equipped_for_soil_analysis`, etc.).
        - Cameras on board rovers and modes they support.
        - Camera calibration targets.
        - Waypoints from which objectives are visible.
        - Initial locations of soil and rock samples.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize total heuristic cost to 0.
    2. Extract relevant dynamic facts from the current state (rover locations, `have_soil_analysis`, `have_rock_analysis`, `have_image`, `communicated_*`, `calibrated`, `empty`, `at_soil_sample`, `at_rock_sample`).
    3. For each goal fact in `task.goals`:
        a. If the goal fact is already true in the current state, continue to the next goal.
        b. If the goal is `(communicated_soil_data ?w)`:
            i. Add 1 for the `communicate_soil_data` action.
            ii. Check if any rover currently has `(have_soil_analysis ?w)`.
            iii. If yes: Find the location of such a rover. Add the BFS distance from this location to any waypoint visible from the lander. If unreachable, return infinity.
            iv. If no (need to sample):
                - Add 1 for the `sample_soil` action.
                - Check if `(at_soil_sample ?w)` is true in the current state. If not, this goal is unreachable (sample is gone or never existed), return infinity.
                - Find the best soil-equipped rover (one that minimizes navigation).
                - Add the BFS distance from the rover's location to `?w` (sample location).
                - Add the BFS distance from `?w` to any waypoint visible from the lander. If unreachable, return infinity.
                - (Store state is ignored).
        c. If the goal is `(communicated_rock_data ?w)`: Follow similar logic as soil data, replacing soil with rock and `equipped_for_soil_analysis` with `equipped_for_rock_analysis`. Check `(at_rock_sample ?w)`.
        d. If the goal is `(communicated_image_data ?o ?m)`:
            i. Add 1 for the `communicate_image_data` action.
            ii. Check if any rover currently has `(have_image ?o ?m)`.
            iii. If yes: Find the location of such a rover. Add the BFS distance from this location to any waypoint visible from the lander. If unreachable, return infinity.
            iv. If no (need to take image):
                - Add 1 for the `take_image` action.
                - Find the best imaging-equipped rover with a camera supporting mode `?m`.
                - Determine if calibration is needed for that camera on that rover (`(calibrated ?i ?r)` is false).
                - Find waypoints `image_wps` visible from objective `?o`. If none, return infinity.
                - If calibration is needed:
                    - Add 1 for the `calibrate` action.
                    - Find calibration target `?t` for the camera and waypoints `cal_wps` visible from `?t`. If none, return infinity.
                    - Add BFS distance from rover's location to any `cal_wp`.
                    - Add BFS distance from any `cal_wp` to any `image_wp`.
                - If calibration is not needed:
                    - Add BFS distance from rover's location to any `image_wp`.
                - Add BFS distance from any `image_wp` to any waypoint visible from the lander. If unreachable, return infinity.
        e. Add the calculated cost for the unachieved goal to the total heuristic cost.
    4. Return the total heuristic cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static/initial facts."""
        self.goals = task.goals

        # --- Extract Static and Initial Information ---

        # Waypoint graph from visible predicates
        self.waypoint_graph = collections.defaultdict(set)
        for fact in task.static:
            if match(fact, "visible", "*", "*"):
                _, wp1, wp2 = get_parts(fact)
                self.waypoint_graph[wp1].add(wp2)
                self.waypoint_graph[wp2].add(wp1) # Assuming bidirectional visibility implies bidirectional traversability

        # Add all waypoints mentioned in visible_from to the graph keys, even if isolated
        for fact in task.static:
             if match(fact, "visible_from", "*", "*"):
                 _, objective, waypoint = get_parts(fact)
                 if waypoint not in self.waypoint_graph:
                     self.waypoint_graph[waypoint] = set()


        # Lander location and visible waypoints
        self.lander_location = None
        for fact in task.initial_state:
             if match(fact, "at_lander", "*", "*"):
                 _, lander, wp = get_parts(fact)
                 self.lander_location = wp
                 # Add lander location to graph keys if not already present
                 if self.lander_location not in self.waypoint_graph:
                     self.waypoint_graph[self.lander_location] = set()
                 break # Assuming only one lander

        self.lander_visible_waypoints = set()
        if self.lander_location:
             # Find waypoints X such that (visible X lander_location) is true
             for fact in task.static:
                 if match(fact, "visible", "*", self.lander_location):
                     _, wp1, wp2 = get_parts(fact)
                     self.lander_visible_waypoints.add(wp1)
             # Also include the lander location itself if it's a waypoint (it should be)
             self.lander_visible_waypoints.add(self.lander_location)


        # Rover capabilities
        self.rover_capabilities = collections.defaultdict(set)
        for fact in task.static:
            if match(fact, "equipped_for_soil_analysis", "*"):
                _, rover = get_parts(fact)
                self.rover_capabilities[rover].add('soil')
            elif match(fact, "equipped_for_rock_analysis", "*"):
                _, rover = get_parts(fact)
                self.rover_capabilities[rover].add('rock')
            elif match(fact, "equipped_for_imaging", "*"):
                _, rover = get_parts(fact)
                self.rover_capabilities[rover].add('imaging')

        # Camera information
        self.rover_cameras = collections.defaultdict(set) # {rover: {camera}}
        self.camera_supports_modes = collections.defaultdict(set) # {camera: {mode}}
        self.camera_calibration_target = {} # {camera: objective}
        self.objective_visible_from = collections.defaultdict(set) # {objective: {waypoint}}
        # self.store_of_rover = {} # {store: rover} # Not strictly needed if ignoring store state

        for fact in task.static:
            if match(fact, "on_board", "*", "*"):
                _, camera, rover = get_parts(fact)
                self.rover_cameras[rover].add(camera)
            elif match(fact, "supports", "*", "*"):
                _, camera, mode = get_parts(fact)
                self.camera_supports_modes[camera].add(mode)
            elif match(fact, "calibration_target", "*", "*"):
                _, camera, objective = get_parts(fact)
                self.camera_calibration_target[camera] = objective
            elif match(fact, "visible_from", "*", "*"):
                _, objective, waypoint = get_parts(fact)
                self.objective_visible_from[objective].add(waypoint)
            # elif match(fact, "store_of", "*", "*"):
            #      _, store, rover = get_parts(fact)
            #      self.store_of_rover[store] = rover


        # Initial sample locations (consumed by sampling)
        self.initial_soil_sample_locations = set()
        self.initial_rock_sample_locations = set()
        for fact in task.initial_state:
            if match(fact, "at_soil_sample", "*"):
                _, wp = get_parts(fact)
                self.initial_soil_sample_locations.add(wp)
                # Add sample location to graph keys if not already present
                if wp not in self.waypoint_graph:
                     self.waypoint_graph[wp] = set()
            elif match(fact, "at_rock_sample", "*"):
                _, wp = get_parts(fact)
                self.initial_rock_sample_locations.add(wp)
                 # Add sample location to graph keys if not already present
                if wp not in self.waypoint_graph:
                     self.waypoint_graph[wp] = set()


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # --- Extract Dynamic Information from State ---
        current_rover_locations = {} # {rover: waypoint}
        current_have_soil = set() # {(rover, waypoint)}
        current_have_rock = set() # {(rover, waypoint)}
        current_have_image = set() # {(rover, objective, mode)}
        current_communicated_soil = set() # {waypoint}
        current_communicated_rock = set() # {waypoint}
        current_communicated_image = set() # {(objective, mode)}
        current_calibrated_cameras = set() # {(camera, rover)}
        # current_empty_stores = set() # {store} # Not needed if ignoring store state
        current_at_soil_sample = set() # {waypoint}
        current_at_rock_sample = set() # {waypoint}


        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip empty facts

            predicate = parts[0]
            if predicate == "at" and len(parts) == 3 and parts[1].startswith('rover'):
                current_rover_locations[parts[1]] = parts[2]
            elif predicate == "have_soil_analysis" and len(parts) == 3:
                current_have_soil.add(tuple(parts[1:]))
            elif predicate == "have_rock_analysis" and len(parts) == 3:
                current_have_rock.add(tuple(parts[1:]))
            elif predicate == "have_image" and len(parts) == 4:
                current_have_image.add(tuple(parts[1:]))
            elif predicate == "communicated_soil_data" and len(parts) == 2:
                current_communicated_soil.add(parts[1])
            elif predicate == "communicated_rock_data" and len(parts) == 2:
                current_communicated_rock.add(parts[1])
            elif predicate == "communicated_image_data" and len(parts) == 3:
                current_communicated_image.add(tuple(parts[1:]))
            elif predicate == "calibrated" and len(parts) == 3:
                current_calibrated_cameras.add(tuple(parts[1:]))
            # elif predicate == "empty" and len(parts) == 2:
            #      current_empty_stores.add(parts[1])
            elif predicate == "at_soil_sample" and len(parts) == 2:
                 current_at_soil_sample.add(parts[1])
            elif predicate == "at_rock_sample" and len(parts) == 2:
                 current_at_rock_sample.add(parts[1])


        total_cost = 0

        # --- Estimate Cost for Each Unachieved Goal ---
        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue # Skip empty goals

            predicate = parts[0]

            if predicate == "communicated_soil_data" and len(parts) == 2:
                w = parts[1]
                if w in current_communicated_soil:
                    continue # Goal already achieved

                cost = 1 # communicate action is always needed

                # Check if sample data is already collected by any rover
                have_fact = any((r, w) in current_have_soil for r in current_rover_locations)

                if have_fact:
                    # Find the rover with the sample and calculate nav cost to lander
                    rover_with_sample_loc = None
                    for r, loc in current_rover_locations.items():
                         if (r, w) in current_have_soil:
                             rover_with_sample_loc = loc
                             break

                    if rover_with_sample_loc:
                        nav_cost = bfs_distance(self.waypoint_graph, {rover_with_sample_loc}, self.lander_visible_waypoints)
                        if nav_cost is float('inf'): return float('inf') # Cannot reach lander
                        cost += nav_cost
                    else:
                         # Should not happen if have_fact is true and rover_locations are tracked
                         # This indicates an inconsistency, treat as unreachable
                         return float('inf')

                else: # Need to sample first
                    cost += 1 # sample action

                    # Check if sample is still at the waypoint or was initially there
                    if w not in current_at_soil_sample and w in self.initial_soil_sample_locations:
                         # Sample was initially there but is gone and no rover has it. Unreachable.
                         return float('inf')
                    elif w not in self.initial_soil_sample_locations:
                         # Goal requires sampling a waypoint that never had a sample. Unreachable.
                         return float('inf')


                    # Find the best soil-equipped rover to sample and communicate
                    best_sample_comm_nav = float('inf')
                    for r, r_loc in current_rover_locations.items():
                        if 'soil' in self.rover_capabilities.get(r, set()):
                             # Ignoring empty store constraint for simplicity
                            sample_nav = bfs_distance(self.waypoint_graph, {r_loc}, {w})
                            comm_nav = bfs_distance(self.waypoint_graph, {w}, self.lander_visible_waypoints)
                            if sample_nav is not float('inf') and comm_nav is not float('inf'):
                                best_sample_comm_nav = min(best_sample_comm_nav, sample_nav + comm_nav)

                    if best_sample_comm_nav is float('inf'):
                        return float('inf') # Cannot reach sample location or lander
                    cost += best_sample_comm_nav

                total_cost += cost

            elif predicate == "communicated_rock_data" and len(parts) == 2:
                w = parts[1]
                if w in current_communicated_rock:
                    continue # Goal already achieved

                cost = 1 # communicate action is always needed

                # Check if sample data is already collected by any rover
                have_fact = any((r, w) in current_have_rock for r in current_rover_locations)

                if have_fact:
                    # Find the rover with the sample and calculate nav cost to lander
                    rover_with_sample_loc = None
                    for r, loc in current_rover_locations.items():
                         if (r, w) in current_have_rock:
                             rover_with_sample_loc = loc
                             break

                    if rover_with_sample_loc:
                        nav_cost = bfs_distance(self.waypoint_graph, {rover_with_sample_loc}, self.lander_visible_waypoints)
                        if nav_cost is float('inf'): return float('inf') # Cannot reach lander
                        cost += nav_cost
                    else:
                         return float('inf') # State inconsistency

                else: # Need to sample first
                    cost += 1 # sample action

                    # Check if sample is still at the waypoint or was initially there
                    if w not in current_at_rock_sample and w in self.initial_rock_sample_locations:
                         return float('inf') # Sample gone
                    elif w not in self.initial_rock_sample_locations:
                         return float('inf') # Sample never existed

                    # Find the best rock-equipped rover to sample and communicate
                    best_sample_comm_nav = float('inf')
                    for r, r_loc in current_rover_locations.items():
                        if 'rock' in self.rover_capabilities.get(r, set()):
                             # Ignoring empty store constraint
                            sample_nav = bfs_distance(self.waypoint_graph, {r_loc}, {w})
                            comm_nav = bfs_distance(self.waypoint_graph, {w}, self.lander_visible_waypoints)
                            if sample_nav is not float('inf') and comm_nav is not float('inf'):
                                best_sample_comm_nav = min(best_sample_comm_nav, sample_nav + comm_nav)

                    if best_sample_comm_nav is float('inf'):
                        return float('inf') # Cannot reach sample location or lander
                    cost += best_sample_comm_nav

                total_cost += cost

            elif predicate == "communicated_image_data" and len(parts) == 3:
                o, m = parts[1:]
                if (o, m) in current_communicated_image:
                    continue # Goal already achieved

                cost = 1 # communicate action is always needed

                # Check if image is already taken by any rover
                have_fact = any((r, o, m) in current_have_image for r in current_rover_locations)

                if have_fact:
                    # Find the rover with the image and calculate nav cost to lander
                    rover_with_image_loc = None
                    for r, loc in current_rover_locations.items():
                         if (r, o, m) in current_have_image:
                             rover_with_image_loc = loc
                             break

                    if rover_with_image_loc:
                        nav_cost = bfs_distance(self.waypoint_graph, {rover_with_image_loc}, self.lander_visible_waypoints)
                        if nav_cost is float('inf'): return float('inf') # Cannot reach lander
                        cost += nav_cost
                    else:
                         return float('inf') # State inconsistency

                else: # Need to take image first
                    cost += 1 # take_image action

                    # Find the best imaging-equipped rover/camera/waypoints combination
                    best_imaging_sequence_cost = float('inf') # Includes nav steps and calibrate action

                    image_wps = self.objective_visible_from.get(o, set())
                    if not image_wps:
                         return float('inf') # Cannot image this objective from anywhere

                    for r, r_loc in current_rover_locations.items():
                        if 'imaging' in self.rover_capabilities.get(r, set()):
                            for camera in self.rover_cameras.get(r, set()):
                                if m in self.camera_supports_modes.get(camera, set()):

                                    calibration_needed = (camera, r) not in current_calibrated_cameras

                                    current_path_cost = 0 # Cost for imaging part (nav + calibrate + take_image)

                                    if calibration_needed:
                                        cal_target = self.camera_calibration_target.get(camera)
                                        cal_wps = self.objective_visible_from.get(cal_target, set()) if cal_target else set()
                                        if not cal_wps:
                                             continue # Cannot calibrate this camera

                                        # Sequence: Nav(rover_loc, cal_wp) + Calibrate + Nav(cal_wp, image_wp)
                                        nav1 = bfs_distance(self.waypoint_graph, {r_loc}, cal_wps)
                                        nav2 = bfs_distance(self.waypoint_graph, cal_wps, image_wps)

                                        if nav1 is not float('inf') and nav2 is not float('inf'):
                                            current_path_cost = nav1 + 1 + nav2 # nav + calibrate + nav
                                        else:
                                            continue # Cannot complete this sequence

                                    else: # Calibration not needed
                                         # Sequence: Nav(rover_loc, image_wp)
                                        nav1 = bfs_distance(self.waypoint_graph, {r_loc}, image_wps)
                                        if nav1 is not float('inf'):
                                            current_path_cost = nav1 # nav
                                        else:
                                            continue # Cannot reach image waypoint

                                    # Now add navigation from image_wp to lander-visible wp
                                    nav_image_to_lander = bfs_distance(self.waypoint_graph, image_wps, self.lander_visible_waypoints)

                                    if nav_image_to_lander is not float('inf'):
                                        current_path_cost += nav_image_to_lander
                                        best_imaging_sequence_cost = min(best_imaging_sequence_cost, current_path_cost)
                                    else:
                                        continue # Cannot reach lander from image waypoints

                    if best_imaging_sequence_cost is float('inf'):
                        return float('inf') # Cannot complete the imaging and communication sequence for this goal
                    cost += best_imaging_sequence_cost

                total_cost += cost

            # Add other goal types if any, though domain file only shows communicated_*

        return total_cost
