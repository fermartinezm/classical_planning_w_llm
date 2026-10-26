from fnmatch import fnmatch
from collections import deque, defaultdict
import math # Import math for infinity

# Assume Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

# Define a dummy Heuristic base class for standalone testing if needed
class Heuristic:
    def __init__(self, task):
        pass
    def __call__(self, node):
        raise NotImplementedError

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact string or malformed facts
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
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

def bfs(graph, start_node):
    """
    Perform Breadth-First Search to find shortest distances from a start node
    in a graph represented as an adjacency dictionary.
    """
    distances = {node: float('inf') for node in graph}
    if start_node not in graph:
         # Start node is not in the graph (e.g., isolated waypoint)
         # Distances to all other nodes remain infinity
         return distances

    distances[start_node] = 0
    queue = deque([start_node])

    while queue:
        current_node = queue.popleft()
        # Ensure current_node is a valid key in the graph
        if current_node in graph:
            for neighbor in graph[current_node]:
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)
    return distances


class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the total number of actions required to satisfy all
    pending goal conditions. It sums the estimated costs for each unsatisfied
    goal, assuming goals are pursued somewhat independently. The cost for each
    goal includes navigation steps (estimated using shortest paths on the
    traversal graph) and the necessary actions (sampling, imaging, communicating,
    dropping, calibrating).

    # Assumptions
    - The heuristic assumes a sequential process for achieving each goal:
      - For soil/rock goals: Navigate to sample location -> (Drop if store full) -> Sample -> Navigate to lander-visible location -> Communicate.
      - For image goals: Navigate to calibration location -> Calibrate -> Navigate to imaging location -> Take Image -> Navigate to lander-visible location -> Communicate.
    - It uses precomputed shortest path distances for navigation estimates.
    - It assumes any equipped rover can perform the necessary tasks for a goal,
      and picks the one that seems cheapest based on current location and precomputed distances.
    - It simplifies the imaging process by assuming calibration is always needed
      as part of the sequence if the image is not already held, ignoring the
      'calibrated' predicate state except for checking if the image is already taken.
    - It sums the costs for each unsatisfied goal, which is non-admissible.

    # Heuristic Initialization
    The heuristic precomputes several pieces of information from the static facts:
    - The set of goal predicates.
    - The location of the lander.
    - Which rovers have which capabilities (soil, rock, imaging).
    - The mapping from stores to rovers.
    - Information about cameras (which rover they are on, supported modes, calibration target).
    - The waypoint visibility graph (for communication locations).
    - The objective visibility mapping (which objectives are visible from which waypoints).
    - The traversal graph for each rover.
    - Shortest path distances between all pairs of waypoints for each rover using BFS.
    - Minimum distances from any waypoint to any lander-visible waypoint for each rover.
    - Minimum distances for imaging tasks:
        - From any waypoint to any calibration waypoint for a camera.
        - From any waypoint to any imaging waypoint for an objective.
        - From any calibration waypoint to any imaging waypoint for a rover/camera/objective combination.
        - From any imaging waypoint for an objective to any lander-visible waypoint for a rover.
        - From any waypoint to any calibration waypoint and then to any imaging waypoint for a rover/camera/objective combination.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify all goal conditions that are not yet satisfied in the current state.
    2. For each unsatisfied goal, estimate the minimum number of actions required to achieve it:
       - **For `(communicated_soil_data ?w)`:**
         - If any equipped rover already has `(have_soil_analysis ?r ?w)`, the cost is estimated as the minimum navigation cost for such a rover from its current location to a lander-visible waypoint plus 1 (for the communicate action).
         - Otherwise (if no rover has the sample), the cost is estimated as the minimum over all equipped rovers `r` of: navigation from `r`'s current location to `w`, plus 1 (for drop if store is full), plus 1 (for sample_soil), plus navigation from `w` to a lander-visible waypoint, plus 1 (for communicate_soil_data).
       - **For `(communicated_rock_data ?w)`:** Similar logic to soil data, using rock-equipped rovers and rock samples.
       - **For `(communicated_image_data ?o ?m)`:**
         - If any suitable rover (imaging-equipped with a camera supporting mode `m`) already has `(have_image ?r ?o ?m)`, the cost is estimated as the minimum navigation cost for such a rover from its current location to a lander-visible waypoint plus 1 (for the communicate action).
         - Otherwise (if no rover has the image), the cost is estimated as the minimum over all suitable rover/camera pairs `(r, i)` of: navigation from `r`'s current location to a calibration waypoint then to an imaging waypoint for `o` (precomputed), plus 1 (for calibrate), plus 1 (for take_image), plus navigation from an imaging waypoint for `o` to a lander-visible waypoint (precomputed), plus 1 (for communicate_image_data).
    3. The total heuristic value is the sum of the estimated costs for all unsatisfied goals.
    4. If any necessary waypoint (sample location, calibration target waypoint, imaging waypoint, lander-visible waypoint) is unreachable from the rover's current location or from other required waypoints, the cost for that goal is considered infinite. The total heuristic will be infinite if any unsatisfied goal is unreachable.
    5. If all goals are satisfied, the heuristic value is 0.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = set(str(g) for g in task.goals) # Store goals as strings

        # --- Parse Static Facts ---
        self.rovers = set()
        self.waypoints = set()
        self.stores = set()
        self.cameras = set()
        self.modes = set()
        self.landers = set()
        self.objectives = set()

        self.lander_location = None # Assuming one lander
        self.rover_capabilities = defaultdict(set) # rover -> {soil, rock, imaging}
        self.store_map = {} # store -> rover
        self.camera_info = {} # camera -> {'rover': r, 'modes': {m}, 'cal_target': o}
        self.waypoint_visibility_graph = defaultdict(set) # w -> {p where (visible w p)}
        self.lander_visible_waypoints = set() # waypoints visible from lander
        self.objective_visibility = defaultdict(set) # objective -> {waypoint where visible_from}
        self.rover_traversal_graphs = defaultdict(lambda: defaultdict(set)) # rover -> w_from -> {w_to}

        # First pass to identify all objects
        for fact in task.static:
             parts = get_parts(fact)
             if not parts: continue
             pred = parts[0]
             if pred == 'rover': self.rovers.add(parts[1])
             elif pred == 'waypoint': self.waypoints.add(parts[1])
             elif pred == 'store': self.stores.add(parts[1])
             elif pred == 'camera': self.cameras.add(parts[1])
             elif pred == 'mode': self.modes.add(parts[1])
             elif pred == 'lander': self.landers.add(parts[1])
             elif pred == 'objective': self.objectives.add(parts[1])

        # Second pass to parse relationships and build graphs
        for fact in task.static:
            parts = get_parts(fact)
            if not parts: continue

            pred = parts[0]
            if pred == 'at_lander': self.lander_location = parts[2] # Assume one lander
            elif pred == 'equipped_for_soil_analysis': self.rover_capabilities[parts[1]].add('soil')
            elif pred == 'equipped_for_rock_analysis': self.rover_capabilities[parts[1]].add('rock')
            elif pred == 'equipped_for_imaging': self.rover_capabilities[parts[1]].add('imaging')
            elif pred == 'store_of': self.store_map[parts[1]] = parts[2]
            elif pred == 'on_board':
                cam, r = parts[1], parts[2]
                if cam not in self.camera_info: self.camera_info[cam] = {'rover': None, 'modes': set(), 'cal_target': None}
                self.camera_info[cam]['rover'] = r
            elif pred == 'supports':
                cam, mode = parts[1], parts[2]
                if cam not in self.camera_info: self.camera_info[cam] = {'rover': None, 'modes': set(), 'cal_target': None}
                self.camera_info[cam]['modes'].add(mode)
            elif pred == 'calibration_target':
                cam, obj = parts[1], parts[2]
                if cam not in self.camera_info: self.camera_info[cam] = {'rover': None, 'modes': set(), 'cal_target': None}
                self.camera_info[cam]['cal_target'] = obj
            elif pred == 'visible': self.waypoint_visibility_graph[parts[1]].add(parts[2])
            elif pred == 'visible_from': self.objective_visibility[parts[1]].add(parts[2])
            elif pred == 'can_traverse': self.rover_traversal_graphs[parts[1]][parts[2]].add(parts[3])

        # Compute lander visible waypoints
        if self.lander_location:
             # Lander location is visible from itself
            self.lander_visible_waypoints.add(self.lander_location)
            # Add waypoints visible from lander location
            for w in self.waypoints:
                 if w in self.waypoint_visibility_graph and self.lander_location in self.waypoint_visibility_graph[w]:
                     self.lander_visible_waypoints.add(w)


        # --- Compute Shortest Paths ---
        self.rover_dist = {} # rover -> w_from -> w_to -> dist
        for r in self.rovers:
            self.rover_dist[r] = {}
            graph = self.rover_traversal_graphs.get(r, {})
            # Ensure all waypoints are keys in the graph dicts, even if isolated
            for u in self.waypoints:
                 self.rover_dist[r][u] = bfs(graph, u)


        # --- Compute Derived Distances ---
        self.min_dist_to_lander_visible_waypoint = {} # rover -> w_from -> min_dist
        for r in self.rovers:
            self.min_dist_to_lander_visible_waypoint[r] = {}
            for u in self.waypoints:
                self.min_dist_to_lander_visible_waypoint[r][u] = min((self.rover_dist[r][u][lv] for lv in self.lander_visible_waypoints), default=float('inf'))

        self.min_dist_to_cal = {} # rover -> w_from -> camera -> min_dist
        self.min_dist_to_img = {} # rover -> w_from -> objective -> min_dist
        self.min_dist_cal_to_img = {} # rover -> camera -> objective -> min_dist
        self.min_dist_from_loc_to_cal_then_img = {} # rover -> w_from -> camera -> objective -> min_dist
        self.min_dist_from_img_to_lander_visible = {} # rover -> objective -> min_dist

        for r in self.rovers:
            self.min_dist_to_cal[r] = {}
            self.min_dist_to_img[r] = {}
            self.min_dist_cal_to_img[r] = {}
            self.min_dist_from_loc_to_cal_then_img[r] = {}
            self.min_dist_from_img_to_lander_visible[r] = {}

            for u in self.waypoints:
                self.min_dist_to_cal[r][u] = {}
                self.min_dist_to_img[r][u] = {}
                self.min_dist_from_loc_to_cal_then_img[r][u] = {}

                for i, info in self.camera_info.items():
                    if info['rover'] == r: # Camera is on this rover
                        cal_target = info['cal_target']
                        cal_wps = self.objective_visibility.get(cal_target, set())
                        self.min_dist_to_cal[r][u][i] = min((self.rover_dist[r][u][w] for w in cal_wps), default=float('inf'))

                        self.min_dist_cal_to_img[r][i] = {}
                        self.min_dist_from_loc_to_cal_then_img[r][u][i] = {}

                        for o in self.objectives:
                            img_wps = self.objective_visibility.get(o, set())

                            # min_dist_to_img[r][u][o]
                            self.min_dist_to_img[r][u][o] = min((self.rover_dist[r][u][p] for p in img_wps), default=float('inf'))

                            # min_dist_cal_to_img[r][i][o]
                            self.min_dist_cal_to_img[r][i][o] = min((self.rover_dist[r][w][p] for w in cal_wps for p in img_wps), default=float('inf'))

                            # min_dist_from_loc_to_cal_then_img[r][u][i][o]
                            # This is min_{w in cal_wps, p in img_wps} (dist(u, w) + dist(w, p))
                            self.min_dist_from_loc_to_cal_then_img[r][u][i][o] = float('inf')
                            for w in cal_wps:
                                for p in img_wps:
                                    if self.rover_dist[r][u][w] != float('inf') and self.rover_dist[r][w][p] != float('inf'):
                                        self.min_dist_from_loc_to_cal_then_img[r][u][i][o] = min(self.min_dist_from_loc_to_cal_then_img[r][u][i][o], self.rover_dist[r][u][w] + self.rover_dist[r][w][p])


                for o in self.objectives:
                     # min_dist_from_img_to_lander_visible[r][o]
                     img_wps = self.objective_visibility.get(o, set())
                     self.min_dist_from_img_to_lander_visible[r][o] = min((self.min_dist_to_lander_visible_waypoint[r][p] for p in img_wps), default=float('inf'))


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        total_cost = 0

        # --- Extract State Information ---
        current_rover_loc = {}
        rover_has_soil = set()
        rover_has_rock = set()
        rover_has_image = set()
        full_stores = set()
        # calibrated_cameras = set() # Not used in this simplified heuristic logic

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue
            pred = parts[0]
            if pred == 'at' and parts[1] in self.rovers:
                current_rover_loc[parts[1]] = parts[2]
            elif pred == 'have_soil_analysis':
                rover_has_soil.add((parts[1], parts[2]))
            elif pred == 'have_rock_analysis':
                rover_has_rock.add((parts[1], parts[2]))
            elif pred == 'have_image':
                rover_has_image.add((parts[1], parts[2], parts[3]))
            elif pred == 'full' and parts[1] in self.stores:
                full_stores.add(parts[1])
            # elif pred == 'calibrated' and parts[1] in self.cameras:
            #      calibrated_cameras.add((parts[1], parts[2]))


        # --- Calculate Cost for Unsatisfied Goals ---
        for goal_str in self.goals:
            if goal_str in state:
                continue # Goal already achieved

            g_parts = get_parts(goal_str)
            g_pred = g_parts[0]

            if g_pred == 'communicated_soil_data':
                w = g_parts[1]
                min_goal_cost = float('inf')
                soil_rovers = {r for r, caps in self.rover_capabilities.items() if 'soil' in caps}

                # Find rovers that already have the sample
                rovers_with_sample = {r for r, w_ in rover_has_soil if w_ == w and r in soil_rovers}

                if rovers_with_sample:
                    # Case 1: Already have sample, just need to communicate
                    for r in rovers_with_sample:
                        if r not in current_rover_loc: continue # Should not happen in valid states
                        nav_cost = self.min_dist_to_lander_visible_waypoint[r][current_rover_loc[r]]
                        if nav_cost != float('inf'):
                            cost = nav_cost + 1 # Nav + Communicate
                            min_goal_cost = min(min_goal_cost, cost)

                # Case 2: Need to sample and communicate (only consider if Case 1 didn't provide a finite cost)
                if min_goal_cost == float('inf'):
                     for r in soil_rovers:
                        if r not in current_rover_loc: continue

                        # Cost to sample
                        nav_to_sample = self.rover_dist[r][current_rover_loc[r]][w]
                        if nav_to_sample == float('inf'): continue # Cannot reach sample waypoint

                        cost = nav_to_sample # Nav to sample
                        store = next((s for s, r_ in self.store_map.items() if r_ == r), None)
                        if store and store in full_stores:
                            cost += 1 # Drop
                        cost += 1 # Sample

                        # Cost to communicate from sample waypoint
                        nav_to_communicate = self.min_dist_to_lander_visible_waypoint[r][w]
                        if nav_to_communicate == float('inf'): continue # Cannot reach lander visible from sample waypoint

                        cost += nav_to_communicate + 1 # Nav + Communicate

                        min_goal_cost = min(min_goal_cost, cost)

                total_cost += min_goal_cost

            elif g_pred == 'communicated_rock_data':
                w = g_parts[1]
                min_goal_cost = float('inf')
                rock_rovers = {r for r, caps in self.rover_capabilities.items() if 'rock' in caps}

                # Find rovers that already have the sample
                rovers_with_sample = {r for r, w_ in rover_has_rock if w_ == w and r in rock_rovers}

                if rovers_with_sample:
                    # Case 1: Already have sample, just need to communicate
                    for r in rovers_with_sample:
                        if r not in current_rover_loc: continue
                        nav_cost = self.min_dist_to_lander_visible_waypoint[r][current_rover_loc[r]]
                        if nav_cost != float('inf'):
                            cost = nav_cost + 1 # Nav + Communicate
                            min_goal_cost = min(min_goal_cost, cost)

                # Case 2: Need to sample and communicate (only consider if Case 1 didn't provide a finite cost)
                if min_goal_cost == float('inf'):
                    for r in rock_rovers:
                        if r not in current_rover_loc: continue

                        # Cost to sample
                        nav_to_sample = self.rover_dist[r][current_rover_loc[r]][w]
                        if nav_to_sample == float('inf'): continue # Cannot reach sample waypoint

                        cost = nav_to_sample # Nav to sample
                        store = next((s for s, r_ in self.store_map.items() if r_ == r), None)
                        if store and store in full_stores:
                            cost += 1 # Drop
                        cost += 1 # Sample

                        # Cost to communicate from sample waypoint
                        nav_to_communicate = self.min_dist_to_lander_visible_waypoint[r][w]
                        if nav_to_communicate == float('inf'): continue # Cannot reach lander visible from sample waypoint

                        cost += nav_to_communicate + 1 # Nav + Communicate

                        min_goal_cost = min(min_goal_cost, cost)

                total_cost += min_goal_cost

            elif g_pred == 'communicated_image_data':
                o, m = g_parts[1], g_parts[2]
                min_goal_cost = float('inf')

                # Find suitable rover/camera pairs
                suitable_rc_pairs = []
                for r, caps in self.rover_capabilities.items():
                    if 'imaging' in caps and r in current_rover_loc:
                        for i, info in self.camera_info.items():
                            if info['rover'] == r and m in info['modes']:
                                suitable_rc_pairs.append((r, i))

                # Case 1: Already have image, just need to communicate
                rovers_with_image = {r for r, o_, m_ in rover_has_image if o_ == o and m_ == m and r in current_rover_loc}
                if rovers_with_image:
                     for r in rovers_with_image:
                        if r not in current_rover_loc: continue
                        nav_cost = self.min_dist_to_lander_visible_waypoint[r][current_rover_loc[r]]
                        if nav_cost != float('inf'):
                            cost = nav_cost + 1 # Nav + Communicate
                            min_goal_cost = min(min_goal_cost, cost)

                # Case 2: Need to image and communicate (only consider if Case 1 didn't provide a finite cost)
                if min_goal_cost == float('inf'):
                    for r, i in suitable_rc_pairs:
                        if r not in current_rover_loc: continue

                        # Cost to image (includes calibration nav and action) + Cost to communicate (includes nav)
                        # Assumes sequence: Nav(curr->cal->img) -> Calibrate -> TakeImage -> Nav(img->LV) -> Communicate
                        nav_cost_to_cal_then_img = self.min_dist_from_loc_to_cal_then_img[r][current_rover_loc[r]][i][o]
                        nav_cost_from_img_to_comm = self.min_dist_from_img_to_lander_visible[r][o]

                        if nav_cost_to_cal_then_img != float('inf') and nav_cost_from_img_to_comm != float('inf'):
                             # Nav(curr->cal->img) + Calibrate + TakeImage + Nav(img->LV) + Communicate
                            cost = nav_cost_to_cal_then_img + 1 + 1 + nav_cost_from_img_to_comm + 1
                            min_goal_cost = min(min_goal_cost, cost)

                total_cost += min_goal_cost

        # Return infinity if any goal is unreachable
        if total_cost == float('inf'):
             return math.inf # Use math.inf for consistency

        return total_cost

