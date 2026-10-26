from fnmatch import fnmatch
from collections import deque
import math # For float('inf')

# Assume Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

# Define a dummy Heuristic base class for standalone testing if needed
class Heuristic:
    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static

    def __call__(self, node):
        raise NotImplementedError

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string before attempting slicing
    if not isinstance(fact, str):
        return [] # Or raise an error, depending on desired behavior
    # Handle empty fact string or malformed facts
    if len(fact) < 2 or fact[0] != '(' or fact[-1] != ')':
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

def bfs(graph, start_node):
    """Computes shortest path distances from start_node to all other nodes."""
    distances = {node: math.inf for node in graph}
    if start_node in distances: # Ensure start_node is in the graph nodes
        distances[start_node] = 0
        queue = deque([start_node])

        while queue:
            current_node = queue.popleft()

            # If current_node has no outgoing edges in the graph, continue
            if current_node not in graph:
                 continue

            for neighbor in graph[current_node]:
                if distances[neighbor] == math.inf:
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)
    return distances

def compute_all_pairs_shortest_paths(graph):
    """Computes shortest path distances between all pairs of nodes."""
    all_pairs_dist = {}
    # Ensure we iterate over all nodes that are keys or values in the graph
    all_nodes = set(graph.keys())
    for neighbors in graph.values():
        all_nodes.update(neighbors)

    for start_node in all_nodes:
        # Need to handle nodes that might be in the graph values but not keys (no outgoing edges)
        # BFS handles this by only exploring reachable nodes.
        # We still want a distance entry for every node, even if unreachable from start_node.
        all_pairs_dist[start_node] = bfs(graph, start_node)

    return all_pairs_dist


class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    Estimates the number of actions required to achieve uncommunicated goals.
    It sums the minimum estimated cost for each unachieved goal fact,
    considering rover capabilities, locations, sample/objective availability,
    calibration status, and communication points. Navigation costs are
    estimated using precomputed shortest paths on rover-specific traverse graphs.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and static facts.
        Precomputes navigation distances for each rover.
        """
        self.goals = task.goals
        static_facts = task.static

        # --- Precompute static information ---

        self.lander_pos = None
        self.rover_capabilities = {} # rover -> set of capabilities ('soil', 'rock', 'imaging')
        self.store_to_rover = {}     # store -> rover
        self.rover_to_store = {}     # rover -> store
        self.rover_cameras = {}      # rover -> set of cameras on board
        self.camera_modes = {}       # camera -> set of supported modes
        self.camera_calib_target = {} # camera -> calibration target objective
        self.objective_visible_wps = {} # objective -> set of visible waypoints
        self.calib_target_visible_wps = {} # objective (cal target) -> set of visible waypoints
        self.rover_waypoint_graphs = {} # rover -> waypoint graph (adjacency list)
        self.rover_dist = {}         # rover -> all-pairs shortest path distances

        all_waypoints = set()
        all_rovers = set()
        all_cameras = set()
        all_objectives = set()
        all_modes = set()
        all_stores = set()

        # First pass to identify all objects of relevant types
        for fact in static_facts:
             parts = get_parts(fact)
             if not parts: continue

             pred = parts[0]
             if pred == 'at_lander':
                 self.lander_pos = parts[2]
             elif pred == 'equipped_for_soil_analysis':
                 rover = parts[1]
                 self.rover_capabilities.setdefault(rover, set()).add('soil')
                 all_rovers.add(rover)
             elif pred == 'equipped_for_rock_analysis':
                 rover = parts[1]
                 self.rover_capabilities.setdefault(rover, set()).add('rock')
                 all_rovers.add(rover)
             elif pred == 'equipped_for_imaging':
                 rover = parts[1]
                 self.rover_capabilities.setdefault(rover, set()).add('imaging')
                 all_rovers.add(rover)
             elif pred == 'store_of':
                 store, rover = parts[1], parts[2]
                 self.store_to_rover[store] = rover
                 self.rover_to_store[rover] = store
                 all_stores.add(store)
                 all_rovers.add(rover)
             elif pred == 'on_board':
                 camera, rover = parts[1], parts[2]
                 self.rover_cameras.setdefault(rover, set()).add(camera)
                 all_cameras.add(camera)
                 all_rovers.add(rover)
             elif pred == 'supports':
                 camera, mode = parts[1], parts[2]
                 self.camera_modes.setdefault(camera, set()).add(mode)
                 all_cameras.add(camera)
                 all_modes.add(mode)
             elif pred == 'calibration_target':
                 camera, objective = parts[1], parts[2]
                 self.camera_calib_target[camera] = objective
                 all_cameras.add(camera)
                 all_objectives.add(objective)
             elif pred == 'visible_from':
                 objective, waypoint = parts[1], parts[2]
                 self.objective_visible_wps.setdefault(objective, set()).add(waypoint)
                 self.calib_target_visible_wps.setdefault(objective, set()).add(waypoint) # Cal targets are objectives
                 all_objectives.add(objective)
                 all_waypoints.add(waypoint)
             elif pred == 'visible':
                 wp1, wp2 = parts[1], parts[2]
                 all_waypoints.add(wp1)
                 all_waypoints.add(wp2)
             elif pred == 'can_traverse':
                 rover, wp1, wp2 = parts[1], parts[2], parts[3]
                 all_rovers.add(rover)
                 all_waypoints.add(wp1)
                 all_waypoints.add(wp2)

        # Initialize graphs for all rovers with all known waypoints
        for rover in all_rovers:
            self.rover_waypoint_graphs[rover] = {wp: [] for wp in all_waypoints}

        # Second pass to build rover-specific traverse graphs
        for fact in static_facts:
            if match(fact, "can_traverse", "*", "*", "*"):
                parts = get_parts(fact)
                rover, wp_from, wp_to = parts[1], parts[2], parts[3]
                if wp_from in self.rover_waypoint_graphs[rover]:
                     self.rover_waypoint_graphs[rover][wp_from].append(wp_to)
                # Ensure the 'to' waypoint is also a key if it has no outgoing edges
                if wp_to not in self.rover_waypoint_graphs[rover]:
                     self.rover_waypoint_graphs[rover][wp_to] = []


        # Precompute all-pairs shortest paths for each rover
        for rover, graph in self.rover_waypoint_graphs.items():
             self.rover_dist[rover] = compute_all_pairs_shortest_paths(graph)

        # Find waypoints visible from the lander
        self.visible_from_lander_wps = set()
        if self.lander_pos:
            for fact in static_facts:
                if match(fact, "visible", "*", self.lander_pos):
                    self.visible_from_lander_wps.add(get_parts(fact)[1])
                elif match(fact, "visible", self.lander_pos, "*"):
                     # Assuming visible is symmetric, but check both directions just in case
                     # The domain definition shows visible is used bidirectionally for can_traverse
                     # but visible itself is not explicitly symmetric in the predicates.
                     # However, the example instance shows both directions. Let's assume symmetry for visible.
                     self.visible_from_lander_wps.add(get_parts(fact)[2])


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # --- Parse current state information ---

        current_rover_pos = {} # rover -> waypoint
        current_soil_samples = set() # waypoint
        current_rock_samples = set() # waypoint
        current_have_soil = {} # rover -> set of waypoints
        current_have_rock = {} # rover -> set of waypoints
        current_have_image = {} # rover -> set of (objective, mode) tuples
        current_full_stores = set() # store
        current_calibrated_cameras = set() # (camera, rover) tuple
        current_communicated = set() # communicated facts

        # Initialize have_data sets for all rovers
        for rover in self.rover_capabilities:
            current_have_soil[rover] = set()
            current_have_rock[rover] = set()
            current_have_image[rover] = set()


        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            pred = parts[0]
            if pred == 'at':
                obj_type = None # Need to determine if it's a rover
                # This is inefficient. Better to iterate rovers and check state.
                # Let's assume the first argument of 'at' in state is always a rover for simplicity
                # based on the domain and examples.
                if parts[1] in self.rover_capabilities: # Check if it's a known rover
                     current_rover_pos[parts[1]] = parts[2]
            elif pred == 'at_soil_sample':
                current_soil_samples.add(parts[1])
            elif pred == 'at_rock_sample':
                current_rock_samples.add(parts[1])
            elif pred == 'have_soil_analysis':
                current_have_soil.setdefault(parts[1], set()).add(parts[2])
            elif pred == 'have_rock_analysis':
                current_have_rock.setdefault(parts[1], set()).add(parts[2])
            elif pred == 'have_image':
                current_have_image.setdefault(parts[1], set()).add((parts[2], parts[3]))
            elif pred == 'full':
                current_full_stores.add(parts[1])
            elif pred == 'calibrated':
                current_calibrated_cameras.add((parts[1], parts[2]))
            elif pred.startswith('communicated_'):
                current_communicated.add(fact)

        # --- Compute heuristic cost ---

        total_cost = 0
        unachieved_goals = self.goals - current_communicated

        if not unachieved_goals:
            return 0 # Goal state reached

        all_unreachable = True # Track if all remaining goals are unreachable

        for goal in unachieved_goals:
            parts = get_parts(goal)
            if not parts: continue

            goal_type = parts[0]
            min_goal_cost = math.inf

            if goal_type == 'communicated_soil_data':
                w = parts[1]
                # Consider rovers equipped for soil analysis
                for rover, capabilities in self.rover_capabilities.items():
                    if 'soil' in capabilities and rover in current_rover_pos:
                        rover_wp = current_rover_pos[rover]
                        rover_cost = math.inf

                        # Find nearest communication waypoint reachable by this rover
                        min_dist_to_lander = math.inf
                        nearest_comm_wp = None
                        if rover in self.rover_dist and rover_wp in self.rover_dist[rover]:
                            for comm_wp in self.visible_from_lander_wps:
                                dist = self.rover_dist[rover][rover_wp].get(comm_wp, math.inf)
                                if dist < min_dist_to_lander:
                                    min_dist_to_lander = dist
                                    nearest_comm_wp = comm_wp

                        # If rover cannot reach any communication point, skip
                        if nearest_comm_wp is None or min_dist_to_lander == math.inf:
                             continue

                        # Case 1: Rover already has the soil data
                        if w in current_have_soil.get(rover, set()):
                            rover_cost = min_dist_to_lander + 1 # communicate

                        # Case 2: Rover needs to collect the soil data
                        elif w in current_soil_samples: # Sample is available
                            dist_to_sample = self.rover_dist[rover][rover_wp].get(w, math.inf)

                            if dist_to_sample != math.inf:
                                collect_cost = dist_to_sample + 1 # sample_soil

                                # Add drop cost if store is full
                                store_of_r = self.rover_to_store.get(rover)
                                if store_of_r and (store_of_r in current_full_stores):
                                    collect_cost += 1 # drop

                                # Cost to communicate from sample location
                                dist_sample_to_comm = self.rover_dist[rover][w].get(nearest_comm_wp, math.inf)

                                if dist_sample_to_comm != math.inf:
                                     comm_cost = dist_sample_to_comm + 1 # communicate
                                     rover_cost = collect_cost + comm_cost

                        min_goal_cost = min(min_goal_cost, rover_cost)

            elif goal_type == 'communicated_rock_data':
                 w = parts[1]
                 # Consider rovers equipped for rock analysis
                 for rover, capabilities in self.rover_capabilities.items():
                     if 'rock' in capabilities and rover in current_rover_pos:
                         rover_wp = current_rover_pos[rover]
                         rover_cost = math.inf

                         # Find nearest communication waypoint reachable by this rover
                         min_dist_to_lander = math.inf
                         nearest_comm_wp = None
                         if rover in self.rover_dist and rover_wp in self.rover_dist[rover]:
                             for comm_wp in self.visible_from_lander_wps:
                                 dist = self.rover_dist[rover][rover_wp].get(comm_wp, math.inf)
                                 if dist < min_dist_to_lander:
                                     min_dist_to_lander = dist
                                     nearest_comm_wp = comm_wp

                         # If rover cannot reach any communication point, skip
                         if nearest_comm_wp is None or min_dist_to_lander == math.inf:
                              continue

                         # Case 1: Rover already has the rock data
                         if w in current_have_rock.get(rover, set()):
                             rover_cost = min_dist_to_lander + 1 # communicate

                         # Case 2: Rover needs to collect the rock data
                         elif w in current_rock_samples: # Sample is available
                             dist_to_sample = self.rover_dist[rover][rover_wp].get(w, math.inf)

                             if dist_to_sample != math.inf:
                                 collect_cost = dist_to_sample + 1 # sample_rock

                                 # Add drop cost if store is full
                                 store_of_r = self.rover_to_store.get(rover)
                                 if store_of_r and (store_of_r in current_full_stores):
                                     collect_cost += 1 # drop

                                 # Cost to communicate from sample location
                                 dist_sample_to_comm = self.rover_dist[rover][w].get(nearest_comm_wp, math.inf)

                                 if dist_sample_to_comm != math.inf:
                                      comm_cost = dist_sample_to_comm + 1 # communicate
                                      rover_cost = collect_cost + comm_cost

                         min_goal_cost = min(min_goal_cost, rover_cost)


            elif goal_type == 'communicated_image_data':
                o, m = parts[1], parts[2]
                # Consider rovers equipped for imaging
                for rover, capabilities in self.rover_capabilities.items():
                    if 'imaging' in capabilities and rover in current_rover_pos:
                        # Consider cameras on this rover supporting the mode
                        for camera in self.rover_cameras.get(rover, set()):
                            if m in self.camera_modes.get(camera, set()):
                                rover_camera_cost = math.inf
                                rover_wp = current_rover_pos[rover]

                                # Find nearest communication waypoint reachable by this rover
                                min_dist_to_lander = math.inf
                                nearest_comm_wp = None
                                if rover in self.rover_dist and rover_wp in self.rover_dist[rover]:
                                    for comm_wp in self.visible_from_lander_wps:
                                        dist = self.rover_dist[rover][rover_wp].get(comm_wp, math.inf)
                                        if dist < min_dist_to_lander:
                                            min_dist_to_lander = dist
                                            nearest_comm_wp = comm_wp

                                # If rover cannot reach any communication point, skip this camera/rover combo
                                if nearest_comm_wp is None or min_dist_to_lander == math.inf:
                                     continue

                                # Case 1: Rover already has the image
                                if (o, m) in current_have_image.get(rover, set()):
                                    rover_camera_cost = min_dist_to_lander + 1 # communicate

                                # Case 2: Rover needs to take the image
                                else:
                                    # Find nearest image waypoint reachable by this rover
                                    min_dist_to_img_wp = math.inf
                                    nearest_img_wp = None
                                    if o in self.objective_visible_wps and rover in self.rover_dist and rover_wp in self.rover_dist[rover]:
                                        for img_wp in self.objective_visible_wps[o]:
                                            dist = self.rover_dist[rover][rover_wp].get(img_wp, math.inf)
                                            if dist < min_dist_to_img_wp:
                                                min_dist_to_img_wp = dist
                                                nearest_img_wp = img_wp

                                    # If rover cannot reach any image waypoint, skip
                                    if nearest_img_wp is None or min_dist_to_img_wp == math.inf:
                                         continue

                                    # Subcase 2a: Camera is calibrated
                                    if (camera, rover) in current_calibrated_cameras:
                                        take_cost = min_dist_to_img_wp + 1 # take_image

                                        # Cost to communicate from image location
                                        dist_img_to_comm = self.rover_dist[rover][nearest_img_wp].get(nearest_comm_wp, math.inf)

                                        if dist_img_to_comm != math.inf:
                                            comm_cost = dist_img_to_comm + 1 # communicate
                                            rover_camera_cost = take_cost + comm_cost

                                    # Subcase 2b: Camera needs calibration
                                    else:
                                        calib_target = self.camera_calib_target.get(camera)
                                        if calib_target:
                                            # Find nearest calibration waypoint reachable by this rover
                                            min_dist_to_cal_wp = math.inf
                                            nearest_cal_wp = None
                                            if calib_target in self.calib_target_visible_wps and rover in self.rover_dist and rover_wp in self.rover_dist[rover]:
                                                for cal_wp in self.calib_target_visible_wps[calib_target]:
                                                    dist = self.rover_dist[rover][rover_wp].get(cal_wp, math.inf)
                                                    if dist < min_dist_to_cal_wp:
                                                        min_dist_to_cal_wp = dist
                                                        nearest_cal_wp = cal_wp

                                            # If rover cannot reach any calibration waypoint, skip
                                            if nearest_cal_wp is None or min_dist_to_cal_wp == math.inf:
                                                 continue

                                            # Cost to calibrate
                                            cal_cost = min_dist_to_cal_wp + 1 # calibrate

                                            # Cost to navigate from calibration wp to image wp
                                            dist_cal_to_img = self.rover_dist[rover][nearest_cal_wp].get(nearest_img_wp, math.inf)

                                            if dist_cal_to_img != math.inf:
                                                take_cost = dist_cal_to_img + 1 # take_image

                                                # Cost to communicate from image location
                                                dist_img_to_comm = self.rover_dist[rover][nearest_img_wp].get(nearest_comm_wp, math.inf)

                                                if dist_img_to_comm != math.inf:
                                                    comm_cost = dist_img_to_comm + 1 # communicate
                                                    rover_camera_cost = cal_cost + take_cost + comm_cost

                                min_goal_cost = min(min_goal_cost, rover_camera_cost)

            # Add the minimum cost for this goal to the total
            if min_goal_cost != math.inf:
                total_cost += min_goal_cost
                all_unreachable = False # At least one goal is reachable

        # If there were unachieved goals but all were unreachable, return infinity
        if unachieved_goals and all_unreachable:
             return math.inf

        return total_cost

