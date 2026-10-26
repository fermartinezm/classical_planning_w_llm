import math
from collections import deque
from fnmatch import fnmatch
# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

# Define a dummy Heuristic base class if the actual one is not provided
# In a real scenario, you would import the actual Heuristic base class.
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    print("Warning: heuristics.heuristic_base not found. Using dummy base class.")
    class Heuristic:
        def __init__(self, task):
            self.goals = task.goals
            self.static = task.static
            # Placeholder for precomputed data
            self.waypoints = set()
            self.rovers = set()
            self.landers = set()
            self.lander_locations = set()
            self.visible_graph = {}
            self.dist = {}
            self.comm_points = set()
            self.min_dist_to_comm = {}
            self.initial_soil_samples = set()
            self.initial_rock_samples = set()
            self.objective_visibility = {} # {objective: {waypoint}}
            self.camera_info = {} # {camera: {rover: r, modes: {m}, cal_target: t}}
            self.equipped_soil = set()
            self.equipped_rock = set()
            self.equipped_imaging = set()
            self.rover_stores = {} # {rover: store}

        def __call__(self, node):
            # Placeholder
            return 0


# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact strings or malformed facts defensively
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
    # Check if the number of parts matches the number of arguments
    if len(parts) != len(args):
        return False
    # Check if each part matches the corresponding argument pattern
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def bfs(graph, start_node):
    """
    Performs Breadth-First Search on a graph to find shortest distances
    from a start node to all reachable nodes.

    Args:
        graph: A dictionary representing the graph {node: set of neighbors}.
        start_node: The node to start the BFS from.

    Returns:
        A dictionary {node: distance} for all reachable nodes.
        Distance is float('inf') for unreachable nodes.
    """
    distances = {node: float('inf') for node in graph}
    if start_node not in graph:
         # Start node is not in the graph, no paths possible from here
         return distances

    distances[start_node] = 0
    queue = deque([start_node])

    while queue:
        current = queue.popleft()

        if current in graph: # Ensure current node is a key in the graph
            for neighbor in graph[current]:
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = distances[current] + 1
                    queue.append(neighbor)

    return distances

def get_rover_pos(state, rover):
    """Find the current waypoint of a given rover in the state."""
    for fact in state:
        if match(fact, "at", rover, "*"):
            return get_parts(fact)[2]
    return None # Rover location not found in state

def get_store_of_rover(static_facts, rover):
    """Find the store object associated with a given rover from static facts."""
    for fact in static_facts:
        if match(fact, "store_of", "*", rover):
            return get_parts(fact)[1]
    return None # Store not found for this rover


class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    Estimates the cost to reach the goal by summing the estimated costs
    for each unachieved goal condition independently.

    The cost for each goal is estimated based on the actions and navigation
    required to achieve it from the current state, using precomputed
    shortest path distances on the visible waypoint graph. This heuristic
    is not admissible but aims to be informative for greedy best-first search.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by precomputing static information and distances.
        """
        super().__init__(task)

        # --- Precomputation ---

        # Collect all objects by type based on predicate arguments in initial and static facts
        all_facts = set(task.initial_state) | set(task.static)

        for fact in all_facts:
             parts = get_parts(fact)
             if not parts: continue
             pred = parts[0]
             if pred in ['at', 'can_traverse', 'visible', 'at_soil_sample', 'at_rock_sample', 'visible_from']:
                 for obj in parts[1:]:
                     if obj.startswith('waypoint'): self.waypoints.add(obj)
             elif pred in ['at', 'can_traverse', 'equipped_for_soil_analysis', 'equipped_for_rock_analysis', 'equipped_for_imaging', 'have_rock_analysis', 'have_soil_analysis', 'store_of', 'calibrated', 'on_board', 'have_image', 'communicate_soil_data', 'communicate_rock_data', 'communicate_image_data']:
                  for obj in parts[1:]:
                     if obj.startswith('rover'): self.rovers.add(obj)
             elif pred in ['empty', 'full', 'store_of']:
                  for obj in parts[1:]:
                     if obj.startswith('store'): self.rover_stores[obj] = None # Will fill rover later
             elif pred in ['calibrated', 'supports', 'calibration_target', 'on_board', 'have_image', 'communicate_image_data']:
                  for obj in parts[1:]:
                     if obj.startswith('camera'): self.camera_info[obj] = self.camera_info.get(obj, {'modes': set()}) # Will fill other info later
                     if obj.startswith('mode'): pass # Modes are collected within camera_info
             elif pred in ['at_lander', 'communicate_soil_data', 'communicate_rock_data', 'communicate_image_data']:
                  for obj in parts[1:]:
                     if obj.startswith('lander'): self.landers.add(obj)
             elif pred in ['calibration_target', 'visible_from', 'have_image', 'communicated_image_data']:
                  for obj in parts[1:]:
                     if obj.startswith('objective'): pass # Objectives are collected within other structures

        # Refine rover_stores, camera_info, equipped sets from static facts
        for fact in task.static:
            if match(fact, "store_of", "*", "*"):
                s, r = get_parts(fact)[1:3]
                self.rover_stores[r] = s # Map rover to its store
            elif match(fact, "on_board", "*", "*"):
                c, r = get_parts(fact)[1:3]
                self.camera_info[c] = self.camera_info.get(c, {'modes': set()})
                self.camera_info[c]['rover'] = r
            elif match(fact, "supports", "*", "*"):
                c, m = get_parts(fact)[1:3]
                self.camera_info[c] = self.camera_info.get(c, {'modes': set()})
                self.camera_info[c]['modes'].add(m)
            elif match(fact, "calibration_target", "*", "*"):
                c, t = get_parts(fact)[1:3]
                self.camera_info[c] = self.camera_info.get(c, {'modes': set()})
                self.camera_info[c]['cal_target'] = t
            elif match(fact, "equipped_for_soil_analysis", "*"):
                self.equipped_soil.add(get_parts(fact)[1])
            elif match(fact, "equipped_for_rock_analysis", "*"):
                self.equipped_rock.add(get_parts(fact)[1])
            elif match(fact, "equipped_for_imaging", "*"):
                self.equipped_imaging.add(get_parts(fact)[1])
            elif match(fact, "at_lander", "*", "*"):
                 self.lander_locations.add(get_parts(fact)[2])
            elif match(fact, "visible", "*", "*"):
                 w1, w2 = get_parts(fact)[1:3]
                 self.visible_graph.setdefault(w1, set()).add(w2)
                 self.visible_graph.setdefault(w2, set()).add(w1) # Assuming visible is symmetric

        # Collect initial sample locations from initial state
        for fact in task.initial_state:
            if match(fact, "at_soil_sample", "*"):
                self.initial_soil_samples.add(get_parts(fact)[1])
            elif match(fact, "at_rock_sample", "*"):
                self.initial_rock_samples.add(get_parts(fact)[1])

        # Collect objective visibility from static facts
        self.objective_visibility = {}
        for fact in task.static:
            if match(fact, "visible_from", "*", "*"):
                o, w = get_parts(fact)[1:3]
                self.objective_visibility.setdefault(o, set()).add(w)

        # Ensure all waypoints mentioned in facts are in the visible graph structure
        for wp in self.waypoints:
             if wp not in self.visible_graph:
                 self.visible_graph[wp] = set() # Add isolated waypoints

        # Compute all-pairs shortest paths on the visible graph
        self.dist = {}
        for start_wp in self.waypoints:
            self.dist[start_wp] = bfs(self.visible_graph, start_wp)

        # Compute communication points (waypoints visible from any lander location)
        self.comm_points = set()
        for lander_loc in self.lander_locations:
            # Waypoints visible from lander location
            if lander_loc in self.visible_graph:
                self.comm_points.update(self.visible_graph[lander_loc])
            # Lander location itself is also a communication point
            self.comm_points.add(lander_loc)

        # Compute minimum distance from any waypoint to any communication point
        self.min_dist_to_comm = {}
        for wp in self.waypoints:
            min_d = float('inf')
            if wp in self.dist: # Ensure waypoint is in the distance map
                for comm_wp in self.comm_points:
                    if comm_wp in self.dist[wp]:
                        min_d = min(min_d, self.dist[wp][comm_wp])
            self.min_dist_to_comm[wp] = min_d


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions
        to reach a goal state from the current state.
        """
        state = node.state

        # --- Parse Current State ---
        current_rover_pos = {} # {rover: waypoint}
        current_have_soil = set() # {(rover, waypoint)}
        current_have_rock = set() # {(rover, waypoint)}
        current_have_image = set() # {(rover, objective, mode)}
        current_calibrated = set() # {(camera, rover)}
        current_full_stores = set() # {store}
        current_communicated_soil = set() # {waypoint}
        current_communicated_rock = set() # {waypoint}
        current_communicated_image = set() # {(objective, mode)}

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue
            pred = parts[0]
            if pred == "at" and len(parts) == 3 and parts[1].startswith('rover'):
                current_rover_pos[parts[1]] = parts[2]
            elif pred == "have_soil_analysis" and len(parts) == 3:
                current_have_soil.add((parts[1], parts[2]))
            elif pred == "have_rock_analysis" and len(parts) == 3:
                current_have_rock.add((parts[1], parts[2]))
            elif pred == "have_image" and len(parts) == 4:
                current_have_image.add((parts[1], parts[2], parts[3]))
            elif pred == "calibrated" and len(parts) == 3:
                current_calibrated.add((parts[1], parts[2]))
            elif pred == "full" and len(parts) == 2:
                current_full_stores.add(parts[1])
            elif pred == "communicated_soil_data" and len(parts) == 2:
                current_communicated_soil.add(parts[1])
            elif pred == "communicated_rock_data" and len(parts) == 2:
                current_communicated_rock.add(parts[1])
            elif pred == "communicated_image_data" and len(parts) == 3:
                current_communicated_image.add((parts[1], parts[2]))

        total_cost = 0

        # --- Estimate Cost for Each Goal ---
        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue
            pred = parts[0]

            if pred == "communicated_soil_data" and len(parts) == 2:
                w = parts[1]
                if w in current_communicated_soil:
                    continue # Goal already achieved

                cost_g = 1 # Cost for the communicate action

                # Check if sample exists on any rover
                sample_exists = any((r, w) in current_have_soil for r in self.rovers)

                if not sample_exists:
                    # Need to sample
                    cost_g += 1 # Cost for sample_soil action

                    # Check if sample exists at the waypoint initially
                    if w not in self.initial_soil_samples:
                        return float('inf') # Goal impossible (sample not available)

                    # Cost to get an equipped rover to sample location with empty store
                    min_sample_prep_cost = float('inf')
                    for r_eq in self.equipped_soil:
                        if r_eq in current_rover_pos:
                            current_pos = current_rover_pos[r_eq]
                            if current_pos not in self.dist or w not in self.dist[current_pos] or self.dist[current_pos][w] == float('inf'):
                                continue # Cannot reach sample location

                            nav_cost = self.dist[current_pos][w]
                            prep_cost = nav_cost

                            # Check if this specific rover's store is full
                            store_r = self.rover_stores.get(r_eq)
                            if store_r and store_r in current_full_stores:
                                prep_cost += 1 # Cost for drop action

                            min_sample_prep_cost = min(min_sample_prep_cost, prep_cost)

                    if min_sample_prep_cost == float('inf'):
                        return float('inf') # Cannot sample (no equipped rover can reach or all stores full and cannot drop)

                    cost_g += min_sample_prep_cost

                    # After sampling, the rover is at waypoint `w`. Cost to get it to a comm point:
                    if w not in self.min_dist_to_comm or self.min_dist_to_comm[w] == float('inf'):
                         return float('inf') # Sample location cannot reach any comm point
                    cost_g += self.min_dist_to_comm[w]

                else:
                    # Sample already exists on some rover(s). Cost to get one of them to a comm point:
                    min_comm_nav_cost = float('inf')
                    for r, sample_w in current_have_soil:
                        if sample_w == w and r in current_rover_pos:
                            current_pos = current_rover_pos[r]
                            if current_pos in self.min_dist_to_comm:
                                min_comm_nav_cost = min(min_comm_nav_cost, self.min_dist_to_comm[current_pos])

                    if min_comm_nav_cost == float('inf'):
                        return float('inf') # Rover with sample cannot reach any comm point
                    cost_g += min_comm_nav_cost

                total_cost += cost_g

            elif pred == "communicated_rock_data" and len(parts) == 2:
                w = parts[1]
                if w in current_communicated_rock:
                    continue # Goal already achieved

                cost_g = 1 # Cost for the communicate action

                # Check if sample exists on any rover
                sample_exists = any((r, w) in current_have_rock for r in self.rovers)

                if not sample_exists:
                    # Need to sample
                    cost_g += 1 # Cost for sample_rock action

                    # Check if sample exists at the waypoint initially
                    if w not in self.initial_rock_samples:
                        return float('inf') # Goal impossible (sample not available)

                    # Cost to get an equipped rover to sample location with empty store
                    min_sample_prep_cost = float('inf')
                    for r_eq in self.equipped_rock:
                        if r_eq in current_rover_pos:
                            current_pos = current_rover_pos[r_eq]
                            if current_pos not in self.dist or w not in self.dist[current_pos] or self.dist[current_pos][w] == float('inf'):
                                continue # Cannot reach sample location

                            nav_cost = self.dist[current_pos][w]
                            prep_cost = nav_cost

                            # Check if this specific rover's store is full
                            store_r = self.rover_stores.get(r_eq)
                            if store_r and store_r in current_full_stores:
                                prep_cost += 1 # Cost for drop action

                            min_sample_prep_cost = min(min_sample_prep_cost, prep_cost)

                    if min_sample_prep_cost == float('inf'):
                        return float('inf') # Cannot sample

                    cost_g += min_sample_prep_cost

                    # After sampling, the rover is at waypoint `w`. Cost to get it to a comm point:
                    if w not in self.min_dist_to_comm or self.min_dist_to_comm[w] == float('inf'):
                         return float('inf') # Sample location cannot reach any comm point
                    cost_g += self.min_dist_to_comm[w]

                else:
                    # Sample already exists on some rover(s). Cost to get one of them to a comm point:
                    min_comm_nav_cost = float('inf')
                    for r, sample_w in current_have_rock:
                        if sample_w == w and r in current_rover_pos:
                            current_pos = current_rover_pos[r]
                            if current_pos in self.min_dist_to_comm:
                                min_comm_nav_cost = min(min_comm_nav_cost, self.min_dist_to_comm[current_pos])

                    if min_comm_nav_cost == float('inf'):
                        return float('inf') # Rover with sample cannot reach any comm point
                    cost_g += min_comm_nav_cost

                total_cost += cost_g

            elif pred == "communicated_image_data" and len(parts) == 3:
                o, m = parts[1], parts[2]
                if (o, m) in current_communicated_image:
                    continue # Goal already achieved

                cost_g = 1 # Cost for the communicate action

                # Check if image exists on any rover
                image_exists = any((r, o, m) in current_have_image for r in self.rovers)

                if not image_exists:
                    # Need to take image
                    cost_g += 1 # Cost for take_image action

                    # Find suitable camera/rover combinations
                    suitable_cameras = []
                    for r_img in self.equipped_imaging:
                        for i_img, info in self.camera_info.items():
                            if info.get('rover') == r_img and m in info.get('modes', set()):
                                suitable_cameras.append((r_img, i_img))

                    if not suitable_cameras:
                        return float('inf') # No suitable camera/rover

                    # Find waypoints visible from the objective
                    image_waypoints = self.objective_visibility.get(o, set())
                    if not image_waypoints:
                        return float('inf') # Objective not visible from anywhere

                    # Cost to get a suitable rover to an image location with calibrated camera
                    min_image_prep_cost = float('inf')
                    for r_img, i_img in suitable_cameras:
                        if r_img in current_rover_pos:
                            current_pos = current_rover_pos[r_img]

                            # Cost to get to an image waypoint
                            min_nav_to_image_wp = float('inf')
                            for p in image_waypoints:
                                if current_pos in self.dist and p in self.dist[current_pos] and self.dist[current_pos][p] != float('inf'):
                                    min_nav_to_image_wp = min(min_nav_to_image_wp, self.dist[current_pos][p])

                            if min_nav_to_image_wp == float('inf'):
                                continue # Cannot reach any image waypoint

                            prep_cost = min_nav_to_image_wp

                            # Cost to calibrate the camera if needed
                            if (i_img, r_img) not in current_calibrated:
                                prep_cost += 1 # Cost for calibrate action
                                cal_target = self.camera_info.get(i_img, {}).get('cal_target')
                                if not cal_target:
                                     return float('inf') # Camera has no calibration target

                                cal_waypoints = self.objective_visibility.get(cal_target, set())
                                if not cal_waypoints:
                                    return float('inf') # Calibration target not visible from anywhere

                                # Cost to get rover to a calibration waypoint
                                min_nav_to_cal_wp = float('inf')
                                for w in cal_waypoints:
                                    if current_pos in self.dist and w in self.dist[current_pos] and self.dist[current_pos][w] != float('inf'):
                                        min_nav_to_cal_wp = min(min_nav_to_cal_wp, self.dist[current_pos][w])

                                if min_nav_to_cal_wp == float('inf'):
                                    return float('inf') # Cannot reach any calibration waypoint
                                prep_cost += min_nav_to_cal_wp

                            min_image_prep_cost = min(min_image_prep_cost, prep_cost)

                    if min_image_prep_cost == float('inf'):
                        return float('inf') # Cannot take image (no suitable rover can reach location/calibrate)
                    cost_g += min_image_prep_cost

                    # After taking image, the rover is at an image waypoint (one of `image_waypoints`).
                    # Cost to get from an image waypoint to a comm point:
                    min_dist_image_wp_to_comm = float('inf')
                    for p in image_waypoints:
                        if p in self.min_dist_to_comm and self.min_dist_to_comm[p] != float('inf'):
                             min_dist_image_wp_to_comm = min(min_dist_image_wp_to_comm, self.min_dist_to_comm[p])

                    if min_dist_image_wp_to_comm == float('inf'):
                        return float('inf') # Image location cannot reach any comm point
                    cost_g += min_dist_image_wp_to_comm

                else:
                    # Image already exists on some rover(s). Cost to get one of them to a comm point:
                    min_comm_nav_cost = float('inf')
                    for r, img_o, img_m in current_have_image:
                        if img_o == o and img_m == m and r in current_rover_pos:
                            current_pos = current_rover_pos[r]
                            if current_pos in self.min_dist_to_comm:
                                min_comm_nav_cost = min(min_comm_nav_cost, self.min_dist_to_comm[current_pos])

                    if min_comm_nav_cost == float('inf'):
                        return float('inf') # Rover with image cannot reach any comm point
                    cost_g += min_comm_nav_cost

                total_cost += cost_g

            # Add other goal types if any (though the problem description only shows communicated_*)

        return total_cost

