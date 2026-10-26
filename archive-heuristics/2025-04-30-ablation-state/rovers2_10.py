from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic

# Helper functions to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure we have the same number of parts as args, unless args has a wildcard at the end
    if len(parts) != len(args) and args[-1] != '*':
         return False
    # Check if each part matches the corresponding arg pattern
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# BFS implementation for shortest path calculation
def bfs(graph, start_node):
    """Computes shortest path distances from start_node to all reachable nodes."""
    distances = {node: float('inf') for node in graph}
    if start_node not in graph:
        return distances # Start node not in graph

    distances[start_node] = 0
    queue = deque([start_node])

    while queue:
        current_node = queue.popleft()

        # Ensure current_node is a valid key in the graph dictionary
        if current_node not in graph:
             continue

        for neighbor in graph.get(current_node, []):
            if distances[neighbor] == float('inf'):
                distances[neighbor] = distances[current_node] + 1
                queue.append(neighbor)
    return distances

def compute_all_pairs_shortest_paths(graph):
    """Computes shortest path distances between all pairs of nodes."""
    all_distances = {}
    # Only compute for nodes that are actually keys in the graph dictionary
    for start_node in graph.keys():
        all_distances[start_node] = bfs(graph, start_node)
    return all_distances


class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the total number of actions required to achieve all
    unmet goal conditions. It sums the estimated minimum cost for each individual
    unmet goal predicate, assuming goals can be pursued independently by capable
    rovers. The cost includes navigation (estimated by shortest path BFS) and
    specific actions (sampling, dropping, calibrating, taking images, communicating).

    # Assumptions
    - The waypoint graph defined by `can_traverse` and `visible` is undirected.
    - Navigation cost between adjacent waypoints is 1.
    - Action costs (sample, drop, calibrate, take_image, communicate) are 1.
    - A rover needs an empty store to sample soil or rock. If its store is full,
      it needs to drop the current sample (cost 1) before sampling.
    - Taking an image requires calibration. This heuristic assumes calibration
      is needed before taking an image unless the image is already obtained.
    - Goals are assumed to be achievable if the necessary locations (samples,
      objectives, calibration targets, communication points) and equipped rovers
      exist and are connected in the navigation graph. If a goal component
      appears unreachable via the calculated paths, the heuristic returns infinity.
    - Solvable instances guarantee that all goal components are reachable.

    # Heuristic Initialization
    The constructor precomputes static information from the task:
    - Lander locations.
    - Communication waypoints (visible from lander locations).
    - Rover capabilities (soil, rock, imaging).
    - Store ownership for each rover.
    - Initial locations of soil and rock samples.
    - Waypoints from which objectives are visible.
    - Camera information (which rover has it, supported modes, calibration target).
    - Rover navigation graphs based on `can_traverse` and `visible`.
    - All-pairs shortest path distances for each rover's navigation graph using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic calculates the sum of estimated costs for
    each goal predicate that is not yet satisfied in the state.

    1.  **Parse State:** Extract current locations of rovers, sample holdings,
        image holdings, store statuses, and camera calibration statuses into
        easily accessible data structures (dictionaries/sets). Also, track
        which soil/rock samples are still available at their initial locations.

    2.  **Initialize Total Cost:** `total_cost = 0`.

    3.  **Iterate Through Goals:** For each goal predicate in the task's goals:
        *   If the goal predicate is already true in the current state, add 0 cost for this goal and continue to the next goal.
        *   If the goal predicate is not true:
            *   Calculate the minimum estimated cost to achieve this specific goal predicate. This minimum is taken over all capable rovers and relevant resources (cameras, stores). If no way is found to achieve the goal component (e.g., required sample is gone and no rover has it, or required waypoints are unreachable), the minimum cost is infinity.

            *   **For `(communicated_soil_data ?w)`:**
                *   Find rovers equipped for soil analysis.
                *   Check if the sample `?w` is already held by *any* capable rover or is still available at location `?w`.
                *   If held by rover `r`: Cost is navigation from `r`'s current location to nearest communication waypoint + 1 (communicate).
                *   If available at `w`: Minimum cost over all capable rovers `r`: Navigation from `r`'s current location to `w` + 1 (sample) + (1 if store is full) + Navigation from `w` to nearest communication waypoint + 1 (communicate).
                *   If sample is gone and not held: Cost is infinity.
                *   The minimum cost for the goal `(communicated_soil_data ?w)` is the minimum found across all options.

            *   **For `(communicated_rock_data ?w)`:** Similar calculation as soil data.

            *   **For `(communicated_image_data ?o ?m)`:**
                *   Find rovers equipped for imaging with cameras supporting mode `?m`.
                *   Check if the image `(o, m)` is already held by *any* capable rover.
                *   If held by rover `r`: Cost is navigation from `r`'s current location to nearest communication waypoint + 1 (communicate).
                *   If not held: Minimum cost over all capable rovers `r` and suitable cameras `i`:
                    Minimum cost over all pairs of (calibration waypoint `w_cal`, image waypoint `w_img`):
                    Navigation from `r`'s current location to `w_cal` + 1 (calibrate) + Navigation from `w_cal` to `w_img` + 1 (take_image) + Navigation from `w_img` to nearest communication waypoint + 1 (communicate).
                *   If no path is found (no suitable waypoints, or unreachable): Cost is infinity.
                *   The minimum cost for the goal `(communicated_image_data ?o ?m)` is the minimum found across all options.

            *   **Add Minimum Goal Cost:** If the minimum estimated cost for the current unachieved goal predicate is finite, add it to `total_cost`. If it is infinity, return `float('inf')` immediately, as the state is likely unsolvable or a dead end according to the heuristic.

    4.  **Return Total Cost:** If the loop completes without returning infinity, return the accumulated `total_cost`.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # --- Precompute Static Information ---

        self.lander_locations = set()
        self.rover_capabilities = {} # {rover_name: set of capabilities ('soil', 'rock', 'imaging')}
        self.rover_stores = {} # {rover_name: store_name}
        self.soil_sample_locations = set() # {waypoint_name} (initial locations)
        self.rock_sample_locations = set() # {waypoint_name} (initial locations)
        self.objective_visible_from = {} # {objective_name: set of waypoint_names}
        self.camera_info = {} # {camera_name: {'rover': rover_name, 'modes': set of mode_names, 'calibration_target': objective_name}}
        self.rover_navigation_graphs = {} # {rover_name: {waypoint: [neighbor_waypoint, ...]}}
        self.waypoints = set() # Set of all waypoints mentioned in relevant static facts
        visible_connections = set() # Set of (w1, w2) tuples for visible predicate

        # First pass to collect basic info and all waypoints
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'at_lander':
                self.lander_locations.add(parts[2])
                self.waypoints.add(parts[2])
            elif parts[0] == 'equipped_for_soil_analysis':
                rover = parts[1]
                self.rover_capabilities.setdefault(rover, set()).add('soil')
            elif parts[0] == 'equipped_for_rock_analysis':
                rover = parts[1]
                self.rover_capabilities.setdefault(rover, set()).add('rock')
            elif parts[0] == 'equipped_for_imaging':
                rover = parts[1]
                self.rover_capabilities.setdefault(rover, set()).add('imaging')
            elif parts[0] == 'store_of':
                store, rover = parts[1], parts[2]
                self.rover_stores[rover] = store
            elif parts[0] == 'at_soil_sample':
                self.soil_sample_locations.add(parts[1])
                self.waypoints.add(parts[1])
            elif parts[0] == 'at_rock_sample':
                self.rock_sample_locations.add(parts[1])
                self.waypoints.add(parts[1])
            elif parts[0] == 'visible_from':
                objective, waypoint = parts[1], parts[2]
                self.objective_visible_from.setdefault(objective, set()).add(waypoint)
                self.waypoints.add(waypoint)
            elif parts[0] == 'calibration_target':
                camera, objective = parts[1], parts[2]
                self.camera_info.setdefault(camera, {})['calibration_target'] = objective
            elif parts[0] == 'on_board':
                camera, rover = parts[1], parts[2]
                self.camera_info.setdefault(camera, {})['rover'] = rover
            elif parts[0] == 'supports':
                camera, mode = parts[1], parts[2]
                self.camera_info.setdefault(camera, {}).setdefault('modes', set()).add(mode)
            elif parts[0] == 'can_traverse':
                 _, r, w1, w2 = parts
                 self.waypoints.add(w1)
                 self.waypoints.add(w2)
            elif parts[0] == 'visible':
                 w1, w2 = parts[1], parts[2]
                 self.waypoints.add(w1)
                 self.waypoints.add(w2)
                 visible_connections.add((w1, w2))
                 visible_connections.add((w2, w1)) # Assume visible is symmetric

        # Identify communication waypoints (visible from any lander location)
        self.comm_waypoints = set()
        for w1, w2 in visible_connections:
            if w2 in self.lander_locations:
                self.comm_waypoints.add(w1)

        # Build navigation graphs per rover
        rover_traverse_facts = [fact for fact in static_facts if get_parts(fact)[0] == 'can_traverse']
        # Only build graph for rovers that exist and have capabilities, and for waypoints mentioned in traverse/visible
        all_waypoints_in_graph = set()
        for fact in rover_traverse_facts:
             _, r, w1, w2 = get_parts(fact)
             if r in self.rover_capabilities:
                 all_waypoints_in_graph.add(w1)
                 all_waypoints_in_graph.add(w2)
        for w1, w2 in visible_connections:
             all_waypoints_in_graph.add(w1)
             all_waypoints_in_graph.add(w2)
        # Also include goal locations, sample locations, lander locations, cal/img waypoints
        for goal in self.goals:
             parts = get_parts(goal)
             if parts[0] in ['communicated_soil_data', 'communicated_rock_data']:
                 all_waypoints_in_graph.add(parts[1]) # Sample waypoint
             elif parts[0] == 'communicated_image_data':
                 obj, mode = parts[1], parts[2]
                 # Need to find cal/img waypoints for this goal
                 # This is complex to do fully in init, but we can add known locations
                 # Add all waypoints from visible_from for any objective mentioned in goals
                 if obj in self.objective_visible_from:
                     all_waypoints_in_graph.update(self.objective_visible_from[obj])
                 # Add all waypoints from visible_from for any calibration target of cameras
                 # supporting modes in goals
                 for cam, info in self.camera_info.items():
                     if mode in info.get('modes', set()) and 'calibration_target' in info:
                         cal_target = info['calibration_target']
                         if cal_target in self.objective_visible_from:
                             all_waypoints_in_graph.update(self.objective_visible_from[cal_target])

        all_waypoints_in_graph.update(self.lander_locations)
        all_waypoints_in_graph.update(self.soil_sample_locations)
        all_waypoints_in_graph.update(self.rock_sample_locations)
        all_waypoints_in_graph.update(self.comm_waypoints)


        for rover in self.rover_capabilities.keys():
             graph = {wp: [] for wp in all_waypoints_in_graph} # Initialize graph with relevant waypoints
             # Add edges based on can_traverse and visible
             for fact in rover_traverse_facts:
                 _, r, w1, w2 = get_parts(fact)
                 if r == rover:
                     # Check if w1 and w2 are visible from each other
                     if (w1, w2) in visible_connections:
                         if w1 in graph and w2 in graph: # Only add if both are in our graph set
                             graph[w1].append(w2)
                             graph[w2].append(w1) # Assume can_traverse is symmetric if visible is symmetric

             # Remove duplicate edges
             for wp in graph:
                 graph[wp] = list(set(graph[wp]))

             self.rover_navigation_graphs[rover] = graph

        # Compute all-pairs shortest paths for each rover
        self.rover_shortest_paths = {}
        for rover, graph in self.rover_navigation_graphs.items():
             self.rover_shortest_paths[rover] = compute_all_pairs_shortest_paths(graph)


    def get_distance(self, rover, start, end):
        """Looks up precomputed shortest path distance for a rover."""
        if rover not in self.rover_shortest_paths:
             return float('inf') # Rover doesn't exist or has no graph

        rover_paths = self.rover_shortest_paths[rover]

        # Check if start and end are valid nodes in the precomputed paths
        if start not in rover_paths or end not in rover_paths.get(start, {}):
            return float('inf')

        return rover_paths[start][end]


    def get_distance_to_any_comm_waypoint(self, rover, start):
        """Finds min distance from start to any communication waypoint for a rover."""
        if not self.comm_waypoints:
            return float('inf') # No communication points defined

        min_dist = float('inf')
        for comm_wp in self.comm_waypoints:
            dist = self.get_distance(rover, start, comm_wp)
            min_dist = min(min_dist, dist)
        return min_dist


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # --- Parse Current State ---
        rover_locations = {} # {rover: waypoint}
        has_soil = set() # {(rover, waypoint)}
        has_rock = set() # {(rover, waypoint)}
        has_image = set() # {(rover, objective, mode)}
        store_status = {} # {store: 'empty' or 'full'}
        calibrated_cameras = set() # {(camera, rover)}
        soil_samples_at_state = set() # {waypoint} - samples currently at location
        rock_samples_at_state = set() # {waypoint} - samples currently at location

        # Initialize samples from static facts, then update based on state
        soil_samples_at_state.update(self.soil_sample_locations)
        rock_samples_at_state.update(self.rock_sample_locations)

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at' and parts[1] in self.rover_capabilities: # Check if it's a rover with capabilities
                rover_locations[parts[1]] = parts[2]
            elif parts[0] == 'have_soil_analysis':
                has_soil.add((parts[1], parts[2]))
                # If a rover has the sample, it's no longer at the location (unless multiple samples?)
                # Assuming one sample per waypoint, having it means it's not at the location anymore.
                if parts[2] in soil_samples_at_state:
                     soil_samples_at_state.remove(parts[2])
            elif parts[0] == 'have_rock_analysis':
                has_rock.add((parts[1], parts[2]))
                 # If a rover has the sample, it's no longer at the location
                if parts[2] in rock_samples_at_state:
                     rock_samples_at_state.remove(parts[2])
            elif parts[0] == 'have_image':
                has_image.add((parts[1], parts[2], parts[3]))
            elif parts[0] == 'empty':
                store_status[parts[1]] = 'empty'
            elif parts[0] == 'full':
                store_status[parts[1]] = 'full'
            elif parts[0] == 'calibrated':
                calibrated_cameras.add((parts[1], parts[2]))
            # Note: at_soil_sample and at_rock_sample are handled by removing from the initial set

        # --- Compute Heuristic Cost ---
        total_cost = 0

        for goal in self.goals:
            # Check if goal is already satisfied
            if goal in state:
                continue

            parts = get_parts(goal)
            predicate = parts[0]
            min_goal_cost = float('inf')

            if predicate == 'communicated_soil_data':
                w = parts[1]
                # Find rovers equipped for soil
                capable_rovers = [r for r, caps in self.rover_capabilities.items() if 'soil' in caps]

                # Option 1: Is the sample already held by *any* capable rover?
                rover_with_sample = None
                for r_cap in capable_rovers:
                    if (r_cap, w) in has_soil:
                        rover_with_sample = r_cap
                        break

                if rover_with_sample:
                    # Sample is held, just need to communicate
                    r = rover_with_sample
                    current_loc = rover_locations.get(r)
                    if current_loc:
                        nav_cost_to_comm = self.get_distance_to_any_comm_waypoint(r, current_loc)
                        if nav_cost_to_comm is not float('inf'):
                            min_goal_cost = min(min_goal_cost, nav_cost_to_comm + 1) # +1 communicate
                elif w in soil_samples_at_state:
                    # Sample is at the location, need to sample and communicate
                    for r in capable_rovers:
                        current_loc = rover_locations.get(r)
                        store = self.rover_stores.get(r)
                        if current_loc and store:
                            nav_cost_to_sample = self.get_distance(r, current_loc, w)
                            if nav_cost_to_sample is not float('inf'):
                                cost_get_sample = nav_cost_to_sample + 1 # sample_soil
                                if store_status.get(store) != 'empty':
                                     cost_get_sample += 1 # drop (assume drop is always possible if full)

                                nav_cost_to_comm = self.get_distance_to_any_comm_waypoint(r, w) # From sample loc
                                if nav_cost_to_comm is not float('inf'):
                                    min_goal_cost = min(min_goal_cost, cost_get_sample + nav_cost_to_comm + 1) # +1 communicate

            elif predicate == 'communicated_rock_data':
                w = parts[1]
                # Find rovers equipped for rock
                capable_rovers = [r for r, caps in self.rover_capabilities.items() if 'rock' in caps]

                # Option 1: Is the sample already held by *any* capable rover?
                rover_with_sample = None
                for r_cap in capable_rovers:
                    if (r_cap, w) in has_rock:
                        rover_with_sample = r_cap
                        break

                if rover_with_sample:
                    # Sample is held, just need to communicate
                    r = rover_with_sample
                    current_loc = rover_locations.get(r)
                    if current_loc:
                        nav_cost_to_comm = self.get_distance_to_any_comm_waypoint(r, current_loc)
                        if nav_cost_to_comm is not float('inf'):
                            min_goal_cost = min(min_goal_cost, nav_cost_to_comm + 1) # +1 communicate
                elif w in rock_samples_at_state:
                    # Sample is at the location, need to sample and communicate
                    for r in capable_rovers:
                        current_loc = rover_locations.get(r)
                        store = self.rover_stores.get(r)
                        if current_loc and store:
                            nav_cost_to_sample = self.get_distance(r, current_loc, w)
                            if nav_cost_to_sample is not float('inf'):
                                cost_get_sample = nav_cost_to_sample + 1 # sample_rock
                                if store_status.get(store) != 'empty':
                                     cost_get_sample += 1 # drop (assume drop is always possible if full)

                                nav_cost_to_comm = self.get_distance_to_any_comm_waypoint(r, w) # From sample loc
                                if nav_cost_to_comm is not float('inf'):
                                    min_goal_cost = min(min_goal_cost, cost_get_sample + nav_cost_to_comm + 1) # +1 communicate

            elif predicate == 'communicated_image_data':
                o, m = parts[1], parts[2]
                # Find rovers equipped for imaging
                capable_rovers = [r for r, caps in self.rover_capabilities.items() if 'imaging' in caps]

                # Option 1: Is the image already held by *any* capable rover?
                rover_with_image = None
                for r_cap in capable_rovers:
                    if (r_cap, o, m) in has_image:
                        rover_with_image = r_cap
                        break

                if rover_with_image:
                    # Image is held, just need to communicate
                    r = rover_with_image
                    current_loc = rover_locations.get(r)
                    if current_loc:
                        nav_cost_to_comm = self.get_distance_to_any_comm_waypoint(r, current_loc)
                        if nav_cost_to_comm is not float('inf'):
                            min_goal_cost = min(min_goal_cost, nav_cost_to_comm + 1) # +1 communicate
                else:
                    # Need to calibrate, image, and communicate
                    for r in capable_rovers:
                        current_loc = rover_locations.get(r)
                        if not current_loc: continue

                        # Find cameras on this rover supporting this mode
                        suitable_cameras = [cam for cam, info in self.camera_info.items() if info.get('rover') == r and m in info.get('modes', set())]

                        for i in suitable_cameras:
                            cal_target = self.camera_info[i].get('calibration_target')
                            cal_waypoints = self.objective_visible_from.get(cal_target, set())
                            image_waypoints = self.objective_visible_from.get(o, set())

                            if cal_target and cal_waypoints and image_waypoints:
                                min_path_cost_for_camera = float('inf')

                                for w_cal in cal_waypoints:
                                    nav_curr_to_cal = self.get_distance(r, current_loc, w_cal)
                                    if nav_curr_to_cal is not float('inf'):
                                        for w_img in image_waypoints:
                                            nav_cal_to_img = self.get_distance(r, w_cal, w_img)
                                            if nav_cal_to_img is not float('inf'):
                                                nav_img_to_comm = self.get_distance_to_any_comm_waypoint(r, w_img)
                                                if nav_img_to_comm is not float('inf'):
                                                    # Cost: Nav(curr->cal) + Calibrate + Nav(cal->img) + TakeImage + Nav(img->comm) + Communicate
                                                    path_cost = nav_curr_to_cal + 1 + nav_cal_to_img + 1 + nav_img_to_comm + 1
                                                    min_path_cost_for_camera = min(min_path_cost_for_camera, path_cost)

                                min_goal_cost = min(min_goal_cost, min_path_cost_for_camera)


            # Add the minimum cost found for this goal component
            if min_goal_cost is not float('inf'):
                total_cost += min_goal_cost
            else:
                # This goal component seems unreachable from this state based on our model.
                # In a solvable problem, this implies the state is not on a path to the goal.
                # Returning infinity is appropriate here.
                return float('inf') # State is likely unsolvable or heuristic limitation


        # If we reached here, all unachieved goals had a finite estimated cost.
        return total_cost
