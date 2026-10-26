from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic

# Helper functions to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact string or invalid format defensively
    if not fact or not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
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

# BFS implementation for shortest path distances
def bfs_distances(graph, start_node):
    """Computes shortest path distances from a start_node in a graph."""
    # Initialize distances for all nodes that are keys in the graph
    distances = {node: float('inf') for node in graph}

    # If start_node is not in the graph keys, it's unreachable from anywhere
    # within this graph structure, or the graph is empty.
    if start_node not in graph:
         return distances # All distances remain infinity

    distances[start_node] = 0
    queue = deque([start_node])
    visited = {start_node}

    while queue:
        current_node = queue.popleft()

        # Ensure current_node is a valid key before accessing its neighbors
        if current_node in graph:
            for neighbor in graph[current_node]:
                # Ensure neighbor is also a valid node in the graph structure
                if neighbor in graph and neighbor not in visited:
                    visited.add(neighbor)
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)

    return distances

def bfs_all_pairs(graph, nodes):
    """Computes shortest path distances between all pairs of nodes in a graph."""
    all_distances = {}
    # Ensure we only run BFS from nodes that are actually in the graph keys
    graph_nodes = set(graph.keys())
    for start_node in nodes:
        if start_node in graph_nodes:
             all_distances[start_node] = bfs_distances(graph, start_node)
        else:
             # If a node is listed but not in the graph keys, it's isolated or invalid
             # Distances from it to anywhere are infinity.
             all_distances[start_node] = {node: float('inf') for node in nodes}

    # Ensure distances are recorded for all target nodes, even if unreachable
    for start_node in nodes:
        if start_node not in all_distances:
             all_distances[start_node] = {node: float('inf') for node in nodes}
        else:
             for end_node in nodes:
                 if end_node not in all_distances[start_node]:
                      all_distances[start_node][end_node] = float('inf')

    return all_distances


class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to satisfy all
    goal conditions. It sums the estimated cost for each unsatisfied goal.
    The cost for a goal depends on whether the required data (soil sample,
    rock sample, or image) has already been collected. If not collected,
    it includes costs for sampling/imaging actions and navigation to
    collection points. If collected, it includes costs for navigation to
    a communication point. Navigation costs are estimated using precomputed
    shortest path distances on the rover's traversable waypoint graph.

    # Assumptions
    - The problem instance is solvable (implies necessary equipment, samples,
      visibility, calibration targets, traversability exist).
    - Action costs are 1.
    - Each rover has at most one store and at most one camera.

    # Heuristic Initialization
    - Extracts static information from the task: lander location, waypoint
      visibility graph, rover traversability graphs, rover capabilities
      (equipment, stores, cameras, camera modes, calibration targets),
      and objective visibility from waypoints.
    - Identifies all unique waypoints, rovers, cameras, objectives, modes, stores.
    - Precomputes shortest path distances between all pairs of waypoints
      for each rover's traversable graph using BFS.
    - Identifies waypoints visible from the lander (communication points).
    - Precomputes the shortest distance from every waypoint to the nearest
      communication point for each rover.

    # Step-By-Step Thinking for Computing Heuristic
    The heuristic value for a state is the sum of estimated costs for each
    goal condition that is not yet satisfied in the state.

    For each unsatisfied goal G:
    1.  If G is `(communicated_soil_data W)`:
        - Add 1 for the `communicate_soil_data` action.
        - Check if `(have_soil_analysis R W)` is true for any rover R in the current state.
        - If NO rover has the soil data for W:
            - Add 1 for the `sample_soil` action.
            - Estimate navigation cost: Find the soil-equipped rover R that minimizes the sum of distances: (distance from R's current location to W) + (distance from W to the nearest communication waypoint). Add this minimum navigation cost. Also add 1 for the `drop` action if the chosen rover's store is full in the current state.
        - If YES, some rover R has the soil data for W:
            - Estimate navigation cost: Find the rover R (that has the data) that minimizes the distance from R's current location to the nearest communication waypoint. Add this minimum navigation cost.

    2.  If G is `(communicated_rock_data W)`:
        - Similar logic as for soil data, using rock-specific predicates and equipped rovers.

    3.  If G is `(communicated_image_data O M)`:
        - Add 1 for the `communicate_image_data` action.
        - Check if `(have_image R O M)` is true for any rover R in the current state.
        - If NO rover has the image data for (O M):
            - Add 1 for the `take_image` action.
            - Add 1 for the `calibrate` action (simplified cost, assuming needed and possible).
            - Estimate navigation cost: Find the imaging-equipped rover R with a camera C supporting mode M that minimizes the navigation cost for the sequence: R's current location -> Collection Waypoint(s) -> Communication Waypoint (W_comm).
                - Find the nearest waypoint P visible from O.
                - Find the nearest waypoint P_cal visible from a calibration target for C.
                - Calculate navigation cost for two primary path options:
                    - Option 1 (Calibrate then Image): R_loc -> P_cal -> P -> W_comm. Cost = dist(R_loc, P_cal) + dist(P_cal, P) + dist(P, W_comm). This option is only possible if P_cal exists and paths are traversable.
                    - Option 2 (Image at Cal-suitable P): R_loc -> P -> W_comm. Cost = dist(R_loc, P) + dist(P, W_comm). This option is only possible if P exists, P is suitable for calibration, and paths are traversable.
                - Take the minimum navigation cost over all capable rovers and valid options. Add this minimum navigation cost. If no valid path exists for any capable rover, the goal is unreachable (infinite cost).
        - If YES, some rover R has the image data for (O M):
            - Estimate navigation cost: Find the rover R (that has the data) that minimizes the distance from R's current location to the nearest communication waypoint. Add this minimum navigation cost.

    The total heuristic value is the sum of these estimated costs for all unsatisfied goals. If any required navigation or capability is impossible, the cost for that goal component is infinite, resulting in an infinite total heuristic.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static_facts = task.static

        # --- Parse Static Facts ---
        self.lander_location = None
        self.waypoint_graph = {} # For visible relation (bidirectional)
        self.rover_graphs = {} # rover -> waypoint -> set(neighbors) (unidirectional)
        self.equipped_soil_rovers = set()
        self.equipped_rock_rovers = set()
        self.equipped_imaging_rovers = set()
        self.rover_stores = {} # rover -> store
        self.rover_cameras = {} # rover -> camera
        self.camera_modes = {} # camera -> set(modes)
        self.camera_calibration_target = {} # camera -> objective
        self.objective_visible_from = {} # objective -> set(waypoints)
        self.all_waypoints = set()
        self.all_rovers = set()
        self.all_cameras = set()
        self.all_objectives = set()
        self.all_modes = set()
        self.all_stores = set()

        for fact in self.static_facts:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]

            if predicate == 'at_lander':
                if len(parts) == 3: self.lander_location = parts[2]
            elif predicate == 'visible':
                if len(parts) == 3:
                    wp1, wp2 = parts[1], parts[2]
                    self.waypoint_graph.setdefault(wp1, set()).add(wp2)
                    self.waypoint_graph.setdefault(wp2, set()).add(wp1) # Visible is symmetric
                    self.all_waypoints.add(wp1)
                    self.all_waypoints.add(wp2)
            elif predicate == 'can_traverse':
                if len(parts) == 4:
                    rover, wp1, wp2 = parts[1], parts[2], parts[3]
                    self.rover_graphs.setdefault(rover, {}).setdefault(wp1, set()).add(wp2)
                    self.all_rovers.add(rover)
                    self.all_waypoints.add(wp1)
                    self.all_waypoints.add(wp2)
            elif predicate == 'equipped_for_soil_analysis':
                if len(parts) == 2: self.equipped_soil_rovers.add(parts[1])
            elif predicate == 'equipped_for_rock_analysis':
                if len(parts) == 2: self.equipped_rock_rovers.add(parts[1])
            elif predicate == 'equipped_for_imaging':
                if len(parts) == 2: self.equipped_imaging_rovers.add(parts[1])
            elif predicate == 'store_of':
                if len(parts) == 3:
                    self.rover_stores[parts[2]] = parts[1] # rover -> store
                    self.all_stores.add(parts[1])
                    self.all_rovers.add(parts[2])
            elif predicate == 'on_board':
                if len(parts) == 3:
                    self.rover_cameras[parts[2]] = parts[1] # rover -> camera
                    self.all_cameras.add(parts[1])
                    self.all_rovers.add(parts[2])
            elif predicate == 'supports':
                if len(parts) == 3:
                    self.camera_modes.setdefault(parts[1], set()).add(parts[2])
                    self.all_cameras.add(parts[1])
                    self.all_modes.add(parts[2])
            elif predicate == 'calibration_target':
                if len(parts) == 3:
                    self.camera_calibration_target[parts[1]] = parts[2] # camera -> objective
                    self.all_cameras.add(parts[1])
                    self.all_objectives.add(parts[2])
            elif predicate == 'visible_from':
                if len(parts) == 3:
                    self.objective_visible_from.setdefault(parts[1], set()).add(parts[2])
                    self.all_objectives.add(parts[1])
                    self.all_waypoints.add(parts[2])
            # Collect object types explicitly listed
            elif len(parts) == 2 and parts[1].startswith('- '):
                 obj_type = parts[1][2:]
                 obj_name = parts[0]
                 if obj_type == 'rover': self.all_rovers.add(obj_name)
                 elif obj_type == 'waypoint': self.all_waypoints.add(obj_name)
                 elif obj_type == 'store': self.all_stores.add(obj_name)
                 elif obj_type == 'camera': self.all_cameras.add(obj_name)
                 elif obj_type == 'mode': self.all_modes.add(obj_name)
                 elif obj_type == 'objective': self.all_objectives.add(obj_name)


        # Ensure all rovers and waypoints mentioned in static facts are in the graph keys/nodes list
        for rover in self.all_rovers:
             self.rover_graphs.setdefault(rover, {})
             for wp in self.all_waypoints:
                 self.rover_graphs[rover].setdefault(wp, set())


        # --- Precompute Distances ---
        self.rover_distances = {}
        # Collect all nodes that might appear in any rover graph
        all_possible_graph_nodes = set(self.all_waypoints)
        for graph in self.rover_graphs.values():
             all_possible_graph_nodes.update(graph.keys())
             for neighbors in graph.values():
                 all_possible_graph_nodes.update(neighbors)


        for rover, graph in self.rover_graphs.items():
             # Use all_possible_graph_nodes to ensure all relevant nodes are considered for all-pairs BFS
             self.rover_distances[rover] = bfs_all_pairs(graph, list(all_possible_graph_nodes))


        # Identify communication waypoints (visible from lander)
        self.comm_waypoint_candidates = set()
        if self.lander_location and self.lander_location in self.waypoint_graph:
             # Find waypoints visible *from* the lander location
             # The visible predicate is symmetric, so visible(A, B) implies visible(B, A)
             # We need waypoints X such that visible(X, lander_location) is true.
             # This is equivalent to finding neighbors of lander_location in the symmetric waypoint_graph.
             self.comm_waypoint_candidates = self.waypoint_graph.get(self.lander_location, set())


        # Precompute distance from every waypoint to the nearest communication waypoint for each rover
        self.rover_dist_to_comm = {}
        for rover in self.all_rovers:
            self.rover_dist_to_comm[rover] = {}
            all_dists_for_rover = self.rover_distances.get(rover, {})

            for start_wp in self.all_waypoints: # Only need distances from actual waypoints
                min_dist = float('inf')
                # Find the minimum distance from start_wp to any comm_waypoint_candidate
                if start_wp in all_dists_for_rover:
                    dists_from_start = all_dists_for_rover[start_wp]
                    for comm_wp in self.comm_waypoint_candidates:
                        if comm_wp in dists_from_start and dists_from_start[comm_wp] != float('inf'):
                             min_dist = min(min_dist, dists_from_start[comm_wp])

                if min_dist == float('inf'):
                    self.rover_dist_to_comm[rover][start_wp] = None # Cannot reach any comm point
                else:
                    self.rover_dist_to_comm[rover][start_wp] = min_dist


    def get_rover_location(self, state, rover):
        """Find the current waypoint of a rover in the state."""
        for fact in state:
            if match(fact, "at", rover, "*"):
                return get_parts(fact)[2]
        return None # Rover location not found (shouldn't happen in valid states)

    def get_distance(self, rover, start_wp, end_wp):
        """Get the precomputed shortest distance between two waypoints for a rover."""
        if rover not in self.rover_distances or start_wp not in self.rover_distances[rover] or end_wp not in self.rover_distances[rover][start_wp]:
             return None # Path not found or invalid waypoints/rover
        dist = self.rover_distances[rover][start_wp][end_wp]
        return dist if dist != float('inf') else None # Return None if unreachable

    def get_dist_to_nearest_comm_waypoint(self, rover, start_wp):
        """Get the precomputed shortest distance from a waypoint to the nearest communication waypoint for a rover."""
        if rover not in self.rover_dist_to_comm or start_wp not in self.rover_dist_to_comm[rover]:
             return None # Invalid waypoint/rover
        return self.rover_dist_to_comm[rover].get(start_wp, None) # Use .get for safety

    def find_nearest_waypoint(self, rover, start_wp, target_wps):
        """Find the nearest waypoint in target_wps to start_wp for a rover."""
        min_dist = float('inf')
        nearest_wp = None
        if start_wp not in self.rover_distances.get(rover, {}):
             return None, None # Invalid start waypoint

        dists_from_start = self.rover_distances[rover].get(start_wp, {})

        for target_wp in target_wps:
            if target_wp in dists_from_start:
                dist = dists_from_start[target_wp]
                if dist != float('inf') and dist < min_dist:
                    min_dist = dist
                    nearest_wp = target_wp

        return nearest_wp, (min_dist if min_dist != float('inf') else None)

    def rovers_with_soil_data(self, state, waypoint):
        """Find rovers in the state that have soil data for the given waypoint."""
        return {get_parts(fact)[1] for fact in state if match(fact, "have_soil_analysis", "*", waypoint)}

    def rovers_with_rock_data(self, state, waypoint):
        """Find rovers in the state that have rock data for the given waypoint."""
        return {get_parts(fact)[1] for fact in state if match(fact, "have_rock_analysis", "*", waypoint)}

    def rovers_with_image_data(self, state, objective, mode):
        """Find rovers in the state that have image data for the given objective and mode."""
        return {get_parts(fact)[1] for fact in state if match(fact, "have_image", "*", objective, mode)}

    def get_rover_store(self, rover):
        """Get the store associated with a rover."""
        return self.rover_stores.get(rover)

    def get_rover_camera(self, rover):
        """Get the camera associated with a rover."""
        return self.rover_cameras.get(rover)

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # If goal is already reached, heuristic is 0
        if self.goals <= state:
            return 0

        total_cost = 0

        for goal in self.goals:
            if goal in state:
                continue # Goal already satisfied

            parts = get_parts(goal)
            if not parts: continue

            predicate = parts[0]

            if predicate == 'communicated_soil_data':
                waypoint = parts[1]
                # Cost includes communicate action + (sample + drop + nav) if needed OR (nav) if data exists
                cost_for_goal = 1 # Communicate action

                rovers_with_data = self.rovers_with_soil_data(state, waypoint)

                if not rovers_with_data:
                    # Need to sample
                    cost_for_goal += 1 # Sample action
                    # Find best rover to sample and then communicate
                    capable_rovers = list(self.equipped_soil_rovers)
                    if not capable_rovers: return float('inf') # Cannot sample soil if no rover is equipped

                    min_path_cost = float('inf')
                    for rover in capable_rovers:
                        rover_loc = self.get_rover_location(state, rover)
                        if rover_loc is None: continue # Rover not in state?

                        dist_to_sample = self.get_distance(rover, rover_loc, waypoint)
                        if dist_to_sample is None: continue # Cannot reach sample location

                        dist_sample_to_comm = self.get_dist_to_nearest_comm_waypoint(rover, waypoint)
                        if dist_sample_to_comm is None: continue # Cannot reach comm location from sample location

                        # Check if store is full - add drop cost before sampling
                        store = self.get_rover_store(rover)
                        drop_cost = 1 if store and f'(full {store})' in state else 0

                        min_path_cost = min(min_path_cost, dist_to_sample + drop_cost + dist_sample_to_comm)

                    if min_path_cost == float('inf'): return float('inf') # Cannot achieve this goal
                    cost_for_goal += min_path_cost

                else:
                    # Data exists, just need to communicate
                    min_path_cost = float('inf')
                    for rover in rovers_with_data:
                        rover_loc = self.get_rover_location(state, rover)
                        if rover_loc is None: continue

                        dist_to_comm = self.get_dist_to_nearest_comm_waypoint(rover, rover_loc)
                        if dist_to_comm is None: continue
                        min_path_cost = min(min_path_cost, dist_to_comm)

                    if min_path_cost == float('inf'): return float('inf') # Cannot achieve this goal
                    cost_for_goal += min_path_cost

                total_cost += cost_for_goal

            elif predicate == 'communicated_rock_data':
                 waypoint = parts[1]
                 # Cost includes communicate action + (sample + drop + nav) if needed OR (nav) if data exists
                 cost_for_goal = 1 # Communicate action

                 rovers_with_data = self.rovers_with_rock_data(state, waypoint)

                 if not rovers_with_data:
                     # Need to sample
                     cost_for_goal += 1 # Sample action
                     # Find best rover to sample and then communicate
                     capable_rovers = list(self.equipped_rock_rovers)
                     if not capable_rovers: return float('inf') # Cannot sample rock

                     min_path_cost = float('inf')
                     for rover in capable_rovers:
                         rover_loc = self.get_rover_location(state, rover)
                         if rover_loc is None: continue

                         dist_to_sample = self.get_distance(rover, rover_loc, waypoint)
                         if dist_to_sample is None: continue

                         dist_sample_to_comm = self.get_dist_to_nearest_comm_waypoint(rover, waypoint)
                         if dist_sample_to_comm is None: continue

                         # Check if store is full - add drop cost before sampling
                         store = self.get_rover_store(rover)
                         drop_cost = 1 if store and f'(full {store})' in state else 0

                         min_path_cost = min(min_path_cost, dist_to_sample + drop_cost + dist_sample_to_comm)

                     if min_path_cost == float('inf'): return float('inf') # Cannot achieve this goal
                     cost_for_goal += min_path_cost

                 else:
                     # Data exists, just need to communicate
                     min_path_cost = float('inf')
                     for rover in rovers_with_data:
                         rover_loc = self.get_rover_location(state, rover)
                         if rover_loc is None: continue

                         dist_to_comm = self.get_dist_to_nearest_comm_waypoint(rover, rover_loc)
                         if dist_to_comm is None: continue
                         min_path_cost = min(min_path_cost, dist_to_comm)

                     if min_path_cost == float('inf'): return float('inf') # Cannot achieve this goal
                     cost_for_goal += min_path_cost

                 total_cost += cost_for_goal

            elif predicate == 'communicated_image_data':
                 objective, mode = parts[1], parts[2]
                 # Cost includes communicate action + (take_image + calibrate + nav) if needed OR (nav) if data exists
                 cost_for_goal = 1 # Communicate action

                 rovers_with_data = self.rovers_with_image_data(state, objective, mode)

                 if not rovers_with_data:
                     # Need to take image
                     cost_for_goal += 1 # Take image action
                     cost_for_goal += 1 # Calibrate action (simplified)

                     # Find best rover to take image and then communicate
                     capable_rovers = [r for r in self.equipped_imaging_rovers if self.get_rover_camera(r) is not None and mode in self.camera_modes.get(self.get_rover_camera(r), set())]
                     if not capable_rovers: return float('inf') # Cannot take image

                     min_path_cost = float('inf')
                     for rover in capable_rovers:
                         rover_loc = self.get_rover_location(state, rover)
                         if rover_loc is None: continue

                         camera = self.get_rover_camera(rover)
                         target = self.camera_calibration_target.get(camera)

                         # Find nearest P visible from O
                         nearest_P, dist_to_P = self.find_nearest_waypoint(rover, rover_loc, self.objective_visible_from.get(objective, set()))
                         if nearest_P is None: continue # Cannot reach image waypoint

                         dist_P_to_comm = self.get_dist_to_nearest_comm_waypoint(rover, nearest_P)
                         if dist_P_to_comm is None: continue # Cannot communicate from P

                         # Find nearest P_cal visible from target for C (if target exists)
                         nearest_P_cal = None
                         dist_to_P_cal = None
                         if target is not None:
                             nearest_P_cal, dist_to_P_cal = self.find_nearest_waypoint(rover, rover_loc, self.objective_visible_from.get(target, set()))
                             # P_cal might be None if no visible waypoint for target

                         # Estimate navigation cost for collection (getting to P and potentially P_cal) and then communication
                         # Option 1: Go to P_cal, then P, then W_comm
                         nav_cost_option1 = float('inf')
                         if nearest_P_cal is not None:
                             dist_P_cal_to_P = self.get_distance(rover, nearest_P_cal, nearest_P)
                             if dist_to_P_cal is not None and dist_P_cal_to_P is not None:
                                 nav_cost_option1 = dist_to_P_cal + dist_P_cal_to_P + dist_P_to_comm

                         # Option 2: Go directly to P (if P is also suitable for calibration), then W_comm
                         nav_cost_option2 = float('inf')
                         cal_at_P_possible = target is not None and nearest_P in self.objective_visible_from.get(target, set())
                         if cal_at_P_possible:
                             if dist_to_P is not None:
                                 nav_cost_option2 = dist_to_P + dist_P_to_comm

                         collection_comm_nav_cost = min(nav_cost_option1, nav_cost_option2)

                         if collection_comm_nav_cost != float('inf'):
                             min_path_cost = min(min_path_cost, collection_comm_nav_cost)

                     if min_path_cost == float('inf'): return float('inf') # Cannot achieve this goal
                     cost_for_goal += min_path_cost

                 else:
                     # Data exists, just need to communicate
                     min_path_cost = float('inf')
                     for rover in rovers_with_data:
                         rover_loc = self.get_rover_location(state, rover)
                         if rover_loc is None: continue

                         dist_to_comm = self.get_dist_to_nearest_comm_waypoint(rover, rover_loc)
                         if dist_to_comm is None: continue
                         min_path_cost = min(min_path_cost, dist_to_comm)

                     if min_path_cost == float('inf'): return float('inf') # Cannot achieve this goal
                     cost_for_goal += min_path_cost

                 total_cost += cost_for_goal


        return total_cost
