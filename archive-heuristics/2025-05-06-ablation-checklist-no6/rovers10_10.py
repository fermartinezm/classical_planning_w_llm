from collections import deque
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# BFS implementation
def bfs_all_pairs_shortest_path(graph, nodes):
    """Compute shortest path distances between all pairs of nodes in a graph."""
    dist = {}
    for start_node in nodes:
        dist[start_node] = {}
        visited = {start_node}
        queue = deque([(start_node, 0)])
        dist[start_node][start_node] = 0

        while queue:
            current_wp, d = queue.popleft()

            if current_wp in graph:
                for neighbor in graph[current_wp]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        dist[start_node][neighbor] = d + 1
                        queue.append((neighbor, d + 1))
    return dist


class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    Estimates the cost to reach the goal by summing the estimated costs
    for each unachieved goal fact. The cost for each goal fact includes
    actions (sample/image/calibrate, communicate) and navigation,
    using precomputed shortest path distances between waypoints.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static facts and precomputing
        waypoint distances.
        """
        self.goals = task.goals
        static_facts = task.static

        # Precompute static information
        self.lander_loc = None
        self.lander_visible_wps = set()
        self.waypoint_graph = {} # Adjacency list
        self.rover_capabilities = {
            'soil': set(),
            'rock': set(),
            'imaging': set()
        }
        self.camera_modes = {} # camera -> set of modes
        self.camera_rover = {} # camera -> rover
        self.camera_cal_target = {} # camera -> objective (calibration target)
        self.cal_wps = {} # objective (calibration target) -> set of waypoints visible from it
        self.img_wps = {} # objective -> set of waypoints visible from it
        self.store_rover = {} # store -> rover
        self.all_waypoints = set()
        self.initial_soil_samples = set()
        self.initial_rock_samples = set()

        # First pass to build waypoint graph and collect basic info
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'at_lander':
                self.lander_loc = parts[2]
                self.all_waypoints.add(self.lander_loc)
            elif parts[0] == 'can_traverse':
                 # Build graph from can_traverse facts (assuming symmetry and uniform traversability)
                 wp1, wp2 = parts[2], parts[3]
                 self.waypoint_graph.setdefault(wp1, set()).add(wp2)
                 self.waypoint_graph.setdefault(wp2, set()).add(wp1)
                 self.all_waypoints.add(wp1)
                 self.all_waypoints.add(wp2)
            elif parts[0] == 'equipped_for_soil_analysis':
                self.rover_capabilities['soil'].add(parts[1])
            elif parts[0] == 'equipped_for_rock_analysis':
                self.rover_capabilities['rock'].add(parts[1])
            elif parts[0] == 'equipped_for_imaging':
                self.rover_capabilities['imaging'].add(parts[1])
            elif parts[0] == 'supports':
                camera, mode = parts[1], parts[2]
                self.camera_modes.setdefault(camera, set()).add(mode)
            elif parts[0] == 'on_board':
                camera, rover = parts[1], parts[2]
                self.camera_rover[camera] = rover
            elif parts[0] == 'calibration_target':
                camera, objective = parts[1], parts[2]
                self.camera_cal_target[camera] = objective
            elif parts[0] == 'store_of':
                store, rover = parts[1], parts[2]
                self.store_rover[store] = rover
            elif parts[0] == 'at_soil_sample':
                 self.initial_soil_samples.add(parts[1])
                 self.all_waypoints.add(parts[1])
            elif parts[0] == 'at_rock_sample':
                 self.initial_rock_samples.add(parts[1])
                 self.all_waypoints.add(parts[1])
            elif parts[0] == 'visible_from':
                 objective, waypoint = parts[1], parts[2]
                 # visible_from facts are needed for imaging and calibration waypoints
                 # Map objective to waypoints visible from it
                 self.img_wps.setdefault(objective, set()).add(waypoint)
                 self.all_waypoints.add(waypoint)


        # Second pass to link calibration targets to visible waypoints
        # We need calibration targets (objectives) and waypoints visible *from* them
        for fact in static_facts:
             parts = get_parts(fact)
             if parts[0] == 'visible_from':
                 objective, waypoint = parts[1], parts[2]
                 # Check if this objective is a calibration target for any camera
                 if objective in self.camera_cal_target.values():
                      self.cal_wps.setdefault(objective, set()).add(waypoint)


        # Precompute lander visible waypoints
        if self.lander_loc:
             for fact in static_facts:
                 parts = get_parts(fact)
                 if parts[0] == 'visible':
                     wp1, wp2 = parts[1], parts[2]
                     if wp1 == self.lander_loc:
                         self.lander_visible_wps.add(wp2)
                     elif wp2 == self.lander_loc: # Assuming visible is symmetric
                         self.lander_visible_wps.add(wp1)
             self.all_waypoints.update(self.lander_visible_wps) # Ensure lander visible wps are in all_waypoints


        # Ensure all waypoints mentioned in the graph are in the set
        for wp in self.waypoint_graph:
            self.all_waypoints.add(wp)
            for neighbor in self.waypoint_graph[wp]:
                self.all_waypoints.add(neighbor)

        # Precompute all-pairs shortest paths
        self.dist = bfs_all_pairs_shortest_path(self.waypoint_graph, list(self.all_waypoints)) # Use list for consistent ordering

    def get_dist(self, wp1, wp2):
        """Get precomputed distance between two waypoints."""
        return self.dist.get(wp1, {}).get(wp2, float('inf'))

    def get_min_dist_to_set(self, wp, wp_set):
        """Get minimum distance from a waypoint to any waypoint in a set."""
        if not wp_set:
            return float('inf')
        min_d = float('inf')
        for target_wp in wp_set:
            d = self.get_dist(wp, target_wp)
            if d != float('inf'):
                 min_d = min(min_d, d)
        return min_d

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Check if goal is reached
        if self.goals <= state:
            return 0

        # Extract current state information
        rover_locs = {}
        store_full = {}
        has_soil = {} # rover -> waypoint -> bool
        has_rock = {} # rover -> waypoint -> bool
        has_image = {} # rover -> objective -> mode -> bool
        is_calibrated = {} # camera -> rover -> bool
        at_soil_sample = set()
        at_rock_sample = set()

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at' and parts[1].startswith('rover'):
                rover_locs[parts[1]] = parts[2]
            elif parts[0] == 'full' and parts[1].startswith('store'):
                store_full[parts[1]] = True
            elif parts[0] == 'have_soil_analysis':
                has_soil.setdefault(parts[1], {})[parts[2]] = True
            elif parts[0] == 'have_rock_analysis':
                has_rock.setdefault(parts[1], {})[parts[2]] = True
            elif parts[0] == 'have_image':
                has_image.setdefault(parts[1], {}).setdefault(parts[2], {})[parts[3]] = True
            elif parts[0] == 'calibrated':
                is_calibrated.setdefault(parts[1], {})[parts[2]] = True
            elif parts[0] == 'at_soil_sample':
                 at_soil_sample.add(parts[1])
            elif parts[0] == 'at_rock_sample':
                 at_rock_sample.add(parts[1])

        total_cost = 0

        # Process unachieved goals
        for goal in self.goals:
            if goal in state:
                continue # Goal already achieved

            parts = get_parts(goal)
            predicate = parts[0]

            if predicate == 'communicated_soil_data':
                w = parts[1]
                cost_g = 1 # communicate action

                need_sample = True
                rover_with_data = None
                for r in self.rover_capabilities['soil']:
                    if has_soil.get(r, {}).get(w, False):
                        need_sample = False
                        rover_with_data = r # Found a rover with the data
                        break

                if need_sample:
                    # Check if sample exists at the waypoint
                    if w not in self.initial_soil_samples and w not in at_soil_sample:
                         return float('inf') # Sample already taken and not present, cannot resample

                    cost_g += 1 # sample action
                    min_sample_path_cost = float('inf')
                    best_sampler = None

                    for r_s in self.rover_capabilities['soil']:
                        current_loc = rover_locs.get(r_s)
                        if current_loc:
                            dist_to_sample = self.get_dist(current_loc, w)
                            if dist_to_sample != float('inf'):
                                if dist_to_sample < min_sample_path_cost:
                                    min_sample_path_cost = dist_to_sample
                                    best_sampler = r_s

                    if best_sampler is None or min_sample_path_cost == float('inf'):
                        return float('inf') # No equipped rover or unreachable sample waypoint

                    cost_g += min_sample_path_cost

                    store = next((s for s, r in self.store_rover.items() if r == best_sampler), None)
                    if store and store_full.get(store, False):
                        cost_g += 1 # drop action

                    rover_with_data = best_sampler # This rover will have the data

                if rover_with_data is None:
                     # This case should ideally not be reached if need_sample logic is correct
                     return float('inf')

                current_loc = rover_locs.get(rover_with_data)
                if current_loc is None: return float('inf') # Rover location unknown

                min_comm_path_cost = self.get_min_dist_to_set(current_loc, self.lander_visible_wps)
                if min_comm_path_cost == float('inf'):
                    return float('inf') # Cannot reach lander visible waypoint

                cost_g += min_comm_path_cost
                total_cost += cost_g

            elif predicate == 'communicated_rock_data':
                w = parts[1]
                cost_g = 1 # communicate action

                need_sample = True
                rover_with_data = None
                for r in self.rover_capabilities['rock']:
                    if has_rock.get(r, {}).get(w, False):
                        need_sample = False
                        rover_with_data = r # Found a rover with the data
                        break

                if need_sample:
                    # Check if sample exists at the waypoint
                    if w not in self.initial_rock_samples and w not in at_rock_sample:
                         return float('inf') # Sample already taken and not present, cannot resample

                    cost_g += 1 # sample action
                    min_sample_path_cost = float('inf')
                    best_sampler = None

                    for r_s in self.rover_capabilities['rock']:
                        current_loc = rover_locs.get(r_s)
                        if current_loc:
                            dist_to_sample = self.get_dist(current_loc, w)
                            if dist_to_sample != float('inf'):
                                if dist_to_sample < min_sample_path_cost:
                                    min_sample_path_cost = dist_to_sample
                                    best_sampler = r_s

                    if best_sampler is None or min_sample_path_cost == float('inf'):
                        return float('inf') # No equipped rover or unreachable sample waypoint

                    cost_g += min_sample_path_cost

                    store = next((s for s, r in self.store_rover.items() if r == best_sampler), None)
                    if store and store_full.get(store, False):
                        cost_g += 1 # drop action

                    rover_with_data = best_sampler # This rover will have the data

                if rover_with_data is None:
                     return float('inf')

                current_loc = rover_locs.get(rover_with_data)
                if current_loc is None: return float('inf') # Rover location unknown

                min_comm_path_cost = self.get_min_dist_to_set(current_loc, self.lander_visible_wps)
                if min_comm_path_cost == float('inf'):
                    return float('inf') # Cannot reach lander visible waypoint

                cost_g += min_comm_path_cost
                total_cost += cost_g

            elif predicate == 'communicated_image_data':
                o, m = parts[1], parts[2]
                cost_g = 1 # communicate action

                need_image = True
                rover_with_data = None
                for r in self.rover_capabilities['imaging']:
                    if has_image.get(r, {}).get(o, {}).get(m, False):
                        need_image = False
                        rover_with_data = r # Found a rover with the image
                        break

                if need_image:
                    cost_g += 1 # take_image action
                    cost_g += 1 # calibrate action (assuming needed before each image)

                    min_total_nav_cost = float('inf')
                    best_imager = None

                    # Find suitable rovers (equipped for imaging, have camera supporting mode m)
                    suitable_rovers = []
                    for r in self.rover_capabilities['imaging']:
                        for i, modes in self.camera_modes.items():
                            if self.camera_rover.get(i) == r and m in modes:
                                suitable_rovers.append((r, i)) # Store rover and a suitable camera
                                break # Found a suitable camera for this rover/mode

                    if not suitable_rovers: return float('inf') # No suitable rover/camera combo

                    possible_img_wps = self.img_wps.get(o, set())
                    if not possible_img_wps: return float('inf') # Cannot image objective from anywhere

                    for r_i, i in suitable_rovers:
                         current_loc = rover_locs.get(r_i)
                         if current_loc is None: continue # Rover location unknown

                         cal_target = self.camera_cal_target.get(i)
                         possible_cal_wps = self.cal_wps.get(cal_target, set()) if cal_target else set()

                         # Path options: current -> cal_wp -> img_wp
                         if possible_cal_wps:
                             for cal_wp in possible_cal_wps:
                                 dist_curr_to_cal = self.get_dist(current_loc, cal_wp)
                                 if dist_curr_to_cal == float('inf'): continue
                                 for img_wp in possible_img_wps:
                                     dist_cal_to_img = self.get_dist(cal_wp, img_wp)
                                     if dist_cal_to_img == float('inf'): continue
                                     total_nav_cost = dist_curr_to_cal + dist_cal_to_img
                                     if total_nav_cost < min_total_nav_cost:
                                         min_total_nav_cost = total_nav_cost
                                         best_imager = r_i
                         else:
                             # If no calibration waypoint exists for the camera's target,
                             # this goal is unachievable via this camera/rover.
                             pass # min_total_nav_cost remains infinity

                    if best_imager is None or min_total_nav_cost == float('inf'):
                        return float('inf') # No suitable rover/camera or unreachable waypoints

                    cost_g += min_total_nav_cost
                    rover_with_data = best_imager # This rover will have the data

                if rover_with_data is None:
                     return float('inf')

                current_loc = rover_locs.get(rover_with_data)
                if current_loc is None: return float('inf') # Rover location unknown

                min_comm_path_cost = self.get_min_dist_to_set(current_loc, self.lander_visible_wps)
                if min_comm_path_cost == float('inf'):
                    return float('inf') # Cannot reach lander visible waypoint

                cost_g += min_comm_path_cost
                total_cost += cost_g

            # Add other goal types if any (not present in domain file provided)
            # elif predicate == 'some_other_goal':
            #    ...

        return total_cost
