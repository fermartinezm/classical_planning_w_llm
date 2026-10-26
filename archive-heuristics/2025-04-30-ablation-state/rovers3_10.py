from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import collections

# Helper functions to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact string or malformed fact
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

# BFS helper functions for shortest path calculation
import collections

def bfs(graph, start):
    """Computes shortest path distances from a start node in a graph."""
    distances = {node: float('inf') for node in graph}
    if start not in graph: # Handle case where start node is not in graph (e.g., isolated waypoint)
         return distances # All unreachable
    distances[start] = 0
    queue = collections.deque([start])
    while queue:
        current = queue.popleft()
        for neighbor in graph.get(current, []):
            if distances[neighbor] == float('inf'):
                distances[neighbor] = distances[current] + 1
                queue.append(neighbor)
    return distances

def compute_all_pairs_shortest_paths(graph):
    """Computes shortest path distances between all pairs of nodes in a graph."""
    all_dist = {}
    # Need all nodes, not just nodes with edges
    all_nodes = set(graph.keys())
    for neighbors in graph.values():
        all_nodes.update(neighbors)

    for start_node in all_nodes:
        all_dist[start_node] = bfs(graph, start_node)
    return all_dist


class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve all unmet
    goal conditions. It sums the estimated cost for each individual unmet goal.
    The cost for each goal is estimated based on the minimum actions needed
    to collect the required data/image and communicate it, considering rover
    capabilities, locations, and necessary intermediate steps like navigation,
    sampling, dropping, calibrating, and imaging.

    # Assumptions
    - Navigation cost between visible waypoints is 1.
    - All other actions (sample, drop, calibrate, take_image, communicate) cost 1.
    - The waypoint graph is defined by the 'visible' predicate.
    - All rovers can traverse any edge defined by 'visible'. (Simplification; 'can_traverse' is rover-specific in PDDL but often uniform in instances).
    - If a soil/rock sample is no longer at its initial location and no rover has the corresponding data, the goal requiring that data is unreachable.
    - If an objective or calibration target is not visible from any waypoint, image goals related to it are unreachable.
    - If no rover has the necessary equipment/camera, related goals are unreachable.
    - Communication requires the rover to be at any waypoint visible from the lander.
    - Calibration is required before taking an image and is consumed by the take_image action.
    - A rover needs an empty store to sample soil or rock. If its store is full, a 'drop' action is needed first.
    - The heuristic does not attempt complex resource allocation (e.g., assigning specific goals to specific rovers) but estimates the cost for *a* capable rover.

    # Heuristic Initialization
    - Parses static facts to build the waypoint graph, compute all-pairs shortest paths,
      identify lander locations and lander-visible waypoints, store rover capabilities,
      camera information (on-board, supports, calibration target), objective visibility,
      and store ownership.

    # Step-By-Step Thinking for Computing Heuristic
    The heuristic iterates through each goal condition that is not yet satisfied in the current state.
    For each unmet goal, it estimates the minimum number of actions required to satisfy it,
    assuming the task is performed by the most suitable available rover (e.g., closest capable rover).
    The total heuristic value is the sum of these individual goal costs.

    For an unmet goal `(communicated_soil_data ?w)`:
    1.  Check if any equipped rover already has `(have_soil_analysis ?r ?w)`.
    2.  If yes: The cost is the navigation from the closest rover that has the data to the nearest lander-visible waypoint, plus 1 for the communicate action.
    3.  If no:
        a.  Check if `(at_soil_sample ?w)` is true in the state.
        b.  If yes (sample exists): Cost includes 1 (sample_soil) + navigation for the closest equipped rover to `?w`. If that rover's store is full, add 1 (drop). Then, add navigation from `?w` to the nearest lander-visible waypoint, plus 1 (communicate).
        c.  If no (sample gone): This goal is impossible if no rover has the data. Return infinity.

    For an unmet goal `(communicated_rock_data ?w)`:
    1.  Similar logic as soil data, using rock analysis equipment and rock samples.

    For an unmet goal `(communicated_image_data ?o ?m)`:
    1.  Check if any capable rover/camera already has `(have_image ?r ?o ?m)`.
    2.  If yes: The cost is the navigation from the closest rover that has the image to the nearest lander-visible waypoint, plus 1 for the communicate action.
    3.  If no:
        a.  Need to take the image. This requires calibration first.
        b.  Find a capable rover/camera and its calibration target. Find waypoints visible from the calibration target (`cal_wps`) and from the objective (`img_wps`). If any are missing, return infinity.
        c.  Cost includes 1 (calibrate) + navigation for the closest capable rover to the nearest `cal_wp`. Let this waypoint be `wp_cal`.
        d.  Then, add 1 (take_image) + navigation from `wp_cal` to the nearest `img_wp`. Let this waypoint be `wp_img`.
        e.  Then, add navigation from `wp_img` to the nearest lander-visible waypoint, plus 1 (communicate).

    The total heuristic is the sum of costs for all unmet goals. If any goal cost is infinity, the total is infinity.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # --- Extract Static Information ---

        # Waypoint graph from visible facts
        self.graph = collections.defaultdict(list)
        all_waypoints = set()
        for fact in static_facts:
            if match(fact, "visible", "*", "*"):
                _, w1, w2 = get_parts(fact)
                self.graph[w1].append(w2)
                self.graph[w2].append(w1) # Assuming visible is symmetric
                all_waypoints.add(w1)
                all_waypoints.add(w2)

        # Ensure all waypoints mentioned in other static facts are in the graph nodes
        for fact in static_facts:
             parts = get_parts(fact)
             if len(parts) > 1 and parts[1].startswith('waypoint'):
                 all_waypoints.add(parts[1])
             if len(parts) > 2 and parts[2].startswith('waypoint'):
                 all_waypoints.add(parts[2])

        # Add isolated waypoints to the graph structure
        for wp in all_waypoints:
             if wp not in self.graph:
                 self.graph[wp] = []

        # All-pairs shortest paths
        self.dist = compute_all_pairs_shortest_paths(self.graph)

        # Lander locations
        self.lander_locations = {get_parts(fact)[2] for fact in static_facts if match(fact, "at_lander", "*", "*")}

        # Waypoints visible from lander locations
        self.lander_visible_waypoints = set()
        for lander_loc in self.lander_locations:
            # Find waypoints x such that (visible x lander_loc) or (visible lander_loc x)
            # Since we assume visible is symmetric and graph is built symmetrically,
            # we just need neighbors of lander_loc in the graph.
            self.lander_visible_waypoints.update(self.graph.get(lander_loc, []))
            # Also include the lander location itself if it's a waypoint (it is)
            self.lander_visible_waypoints.add(lander_loc)


        # Rover capabilities
        self.rover_capabilities = collections.defaultdict(lambda: {'soil': False, 'rock': False, 'imaging': False})
        for fact in static_facts:
            if match(fact, "equipped_for_soil_analysis", "*"):
                _, rover = get_parts(fact)
                self.rover_capabilities[rover]['soil'] = True
            elif match(fact, "equipped_for_rock_analysis", "*"):
                _, rover = get_parts(fact)
                self.rover_capabilities[rover]['rock'] = True
            elif match(fact, "equipped_for_imaging", "*"):
                _, rover = get_parts(fact)
                self.rover_capabilities[rover]['imaging'] = True

        # Camera information
        self.camera_info = collections.defaultdict(lambda: {'rover': None, 'supports': set(), 'cal_target': None})
        for fact in static_facts:
            if match(fact, "on_board", "*", "*"):
                _, camera, rover = get_parts(fact)
                self.camera_info[camera]['rover'] = rover
            elif match(fact, "supports", "*", "*"):
                _, camera, mode = get_parts(fact)
                self.camera_info[camera]['supports'].add(mode)
            elif match(fact, "calibration_target", "*", "*"):
                _, camera, objective = get_parts(fact)
                self.camera_info[camera]['cal_target'] = objective

        # Objective visibility
        self.objective_visible_waypoints = collections.defaultdict(set)
        for fact in static_facts:
            if match(fact, "visible_from", "*", "*"):
                _, objective, waypoint = get_parts(fact)
                self.objective_visible_waypoints[objective].add(waypoint)

        # Store ownership
        self.store_of = {}
        for fact in static_facts:
            if match(fact, "store_of", "*", "*"):
                _, store, rover = get_parts(fact)
                self.store_of[store] = rover


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        total_cost = 0

        # --- Parse Current State ---
        rover_locations = {}
        store_full = {}
        have_soil_analysis = collections.defaultdict(set) # {rover: {waypoint, ...}}
        have_rock_analysis = collections.defaultdict(set) # {rover: {waypoint, ...}}
        have_image = collections.defaultdict(set) # {rover: {(objective, mode), ...}}
        calibrated_cameras = set() # {(camera, rover), ...}
        current_soil_samples = set() # {waypoint, ...}
        current_rock_samples = set() # {waypoint, ...}
        communicated_soil_data = set() # {waypoint, ...}
        communicated_rock_data = set() # {waypoint, ...}
        communicated_image_data = set() # {(objective, mode), ...}


        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            args = parts[1:]

            if predicate == "at" and len(args) == 2 and args[0].startswith('rover'):
                rover_locations[args[0]] = args[1]
            elif predicate == "full" and len(args) == 1 and args[0].startswith('store'):
                store_full[args[0]] = True
            elif predicate == "have_soil_analysis" and len(args) == 2 and args[0].startswith('rover'):
                have_soil_analysis[args[0]].add(args[1])
            elif predicate == "have_rock_analysis" and len(args) == 2 and args[0].startswith('rover'):
                have_rock_analysis[args[0]].add(args[1])
            elif predicate == "have_image" and len(args) == 3 and args[0].startswith('rover'):
                 have_image[args[0]].add((args[1], args[2]))
            elif predicate == "calibrated" and len(args) == 2 and args[0].startswith('camera'):
                calibrated_cameras.add((args[0], args[1]))
            elif predicate == "at_soil_sample" and len(args) == 1 and args[0].startswith('waypoint'):
                current_soil_samples.add(args[0])
            elif predicate == "at_rock_sample" and len(args) == 1 and args[0].startswith('waypoint'):
                current_rock_samples.add(args[0])
            elif predicate == "communicated_soil_data" and len(args) == 1 and args[0].startswith('waypoint'):
                communicated_soil_data.add(args[0])
            elif predicate == "communicated_rock_data" and len(args) == 1 and args[0].startswith('waypoint'):
                communicated_rock_data.add(args[0])
            elif predicate == "communicated_image_data" and len(args) == 2 and args[0].startswith('objective'):
                communicated_image_data.add((args[0], args[1]))


        # --- Estimate Cost for Unmet Goals ---

        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue # Skip malformed goals

            predicate = parts[0]
            args = parts[1:]

            if predicate == "communicated_soil_data" and len(args) == 1:
                waypoint = args[0]
                if waypoint not in communicated_soil_data:
                    goal_cost = self._estimate_soil_goal_cost(waypoint, rover_locations, have_soil_analysis, current_soil_samples, store_full)
                    if goal_cost == float('inf'): return float('inf') # Propagate infinity
                    total_cost += goal_cost

            elif predicate == "communicated_rock_data" and len(args) == 1:
                waypoint = args[0]
                if waypoint not in communicated_rock_data:
                    goal_cost = self._estimate_rock_goal_cost(waypoint, rover_locations, have_rock_analysis, current_rock_samples, store_full)
                    if goal_cost == float('inf'): return float('inf') # Propagate infinity
                    total_cost += goal_cost

            elif predicate == "communicated_image_data" and len(args) == 2:
                objective, mode = args
                if (objective, mode) not in communicated_image_data:
                    goal_cost = self._estimate_image_goal_cost(objective, mode, rover_locations, have_image, calibrated_cameras)
                    if goal_cost == float('inf'): return float('inf') # Propagate infinity
                    total_cost += goal_cost

            # If goal is not one of the communicated types, it's unhandled by this heuristic.
            # Assuming all goals are communication goals based on examples.
            # If other goal types exist and are not met, the heuristic might be inaccurate.


        return total_cost

    def _get_distance(self, wp1, wp2):
        """Helper to get shortest distance between two waypoints."""
        if wp1 is None or wp2 is None: return float('inf')
        return self.dist.get(wp1, {}).get(wp2, float('inf'))

    def _find_closest_rover(self, rovers, current_locations, target_waypoint):
        """Finds the rover in the given set closest to the target waypoint."""
        min_dist = float('inf')
        closest_rover = None
        for r in rovers:
            current_loc = current_locations.get(r)
            if current_loc:
                dist = self._get_distance(current_loc, target_waypoint)
                if dist < min_dist:
                    min_dist = dist
                    closest_rover = r
        return closest_rover, min_dist

    def _find_closest_waypoint(self, start_waypoint, target_waypoints):
        """Finds the waypoint in the target set closest to the start waypoint."""
        min_dist = float('inf')
        closest_wp = None
        if not target_waypoints: return None, float('inf')
        for tw in target_waypoints:
            dist = self._get_distance(start_waypoint, tw)
            if dist < min_dist:
                min_dist = dist
                closest_wp = tw
        return closest_wp, min_dist


    def _estimate_soil_goal_cost(self, waypoint, rover_locations, have_soil_analysis, current_soil_samples, store_full):
        """Estimates cost for a single communicated_soil_data goal."""
        goal_cost = 0
        soil_rovers = {r for r, caps in self.rover_capabilities.items() if caps['soil']}
        if not soil_rovers: return float('inf') # No rover can do soil analysis

        # Check if data already exists on any capable rover
        data_exists = any(waypoint in have_soil_analysis.get(r, set()) for r in soil_rovers)

        task_waypoint = None # Waypoint where data was obtained

        if not data_exists:
            # Need to sample
            sample_exists = waypoint in current_soil_samples
            if not sample_exists:
                # Sample is gone, and no rover has the data -> Impossible
                return float('inf')

            # Sample exists, need to perform sampling
            goal_cost += 1 # sample_soil action

            # Find closest capable rover to the sample waypoint
            sample_rover, nav_cost_to_sample = self._find_closest_rover(soil_rovers, rover_locations, waypoint)
            if sample_rover is None or nav_cost_to_sample == float('inf'): return float('inf') # Cannot reach sample
            goal_cost += nav_cost_to_sample # navigate to sample

            # Check store for the chosen rover
            store = next((s for s, r in self.store_of.items() if r == sample_rover), None)
            if store and store_full.get(store, False):
                 goal_cost += 1 # drop action

            task_waypoint = waypoint # Sampling finished at this waypoint

        # Now calculate communication cost
        comm_start_waypoint = None
        if task_waypoint:
            # Rover is at task_waypoint after sampling
            comm_start_waypoint = task_waypoint
        else:
            # Data already existed. Find the location of a rover that has the data.
            # For simplicity, pick the first one found.
            rover_with_data = next((r for r in soil_rovers if waypoint in have_soil_analysis.get(r, set())), None)
            if rover_with_data:
                 comm_start_waypoint = rover_locations.get(rover_with_data)
            # else: This case should not happen if data_exists was True

        if not comm_start_waypoint: return float('inf') # Safety check (e.g., rover location unknown)

        # Find closest lander-visible waypoint from the communication start waypoint
        lander_comm_wp, nav_cost_to_lander = self._find_closest_waypoint(comm_start_waypoint, self.lander_visible_waypoints)
        if lander_comm_wp is None or nav_cost_to_lander == float('inf'): return float('inf') # Cannot reach lander waypoint

        goal_cost += nav_cost_to_lander # navigate to lander waypoint
        goal_cost += 1 # communicate action

        return goal_cost

    def _estimate_rock_goal_cost(self, waypoint, rover_locations, have_rock_analysis, current_rock_samples, store_full):
        """Estimates cost for a single communicated_rock_data goal."""
        goal_cost = 0
        rock_rovers = {r for r, caps in self.rover_capabilities.items() if caps['rock']}
        if not rock_rovers: return float('inf') # No rover can do rock analysis

        # Check if data already exists on any capable rover
        data_exists = any(waypoint in have_rock_analysis.get(r, set()) for r in rock_rovers)

        task_waypoint = None # Waypoint where data was obtained

        if not data_exists:
            # Need to sample
            sample_exists = waypoint in current_rock_samples
            if not sample_exists:
                # Sample is gone, and no rover has the data -> Impossible
                return float('inf')

            # Sample exists, need to perform sampling
            goal_cost += 1 # sample_rock action

            # Find closest capable rover to the sample waypoint
            sample_rover, nav_cost_to_sample = self._find_closest_rover(rock_rovers, rover_locations, waypoint)
            if sample_rover is None or nav_cost_to_sample == float('inf'): return float('inf') # Cannot reach sample
            goal_cost += nav_cost_to_sample # navigate to sample

            # Check store for the chosen rover
            store = next((s for s, r in self.store_of.items() if r == sample_rover), None)
            if store and store_full.get(store, False):
                 goal_cost += 1 # drop action

            task_waypoint = waypoint # Sampling finished at this waypoint

        # Now calculate communication cost
        comm_start_waypoint = None
        if task_waypoint:
            # Rover is at task_waypoint after sampling
            comm_start_waypoint = task_waypoint
        else:
            # Data already existed. Find the location of a rover that has the data.
            rover_with_data = next((r for r in rock_rovers if waypoint in have_rock_analysis.get(r, set())), None)
            if rover_with_data:
                 comm_start_waypoint = rover_locations.get(rover_with_data)
            # else: Should not happen if data_exists was True

        if not comm_start_waypoint: return float('inf') # Safety check

        # Find closest lander-visible waypoint from the communication start waypoint
        lander_comm_wp, nav_cost_to_lander = self._find_closest_waypoint(comm_start_waypoint, self.lander_visible_waypoints)
        if lander_comm_wp is None or nav_cost_to_lander == float('inf'): return float('inf') # Cannot reach lander waypoint

        goal_cost += nav_cost_to_lander # navigate to lander waypoint
        goal_cost += 1 # communicate action

        return goal_cost


    def _estimate_image_goal_cost(self, objective, mode, rover_locations, have_image, calibrated_cameras):
        """Estimates cost for a single communicated_image_data goal."""
        goal_cost = 0
        img_rovers = {r for r, caps in self.rover_capabilities.items() if caps['imaging']}
        if not img_rovers: return float('inf') # No rover can do imaging

        # Find capable cameras (on board a capable rover and supporting the mode)
        capable_cameras = {c for c, info in self.camera_info.items()
                           if info['rover'] in img_rovers and mode in info['supports']}
        if not capable_cameras: return float('inf') # No suitable camera

        # Check if image already exists on any capable rover
        image_exists = any((objective, mode) in have_image.get(r, set()) for r in img_rovers)

        task_waypoint = None # Waypoint where image was taken
        task_rover = None # Rover that performed the task

        if not image_exists:
            # Need to take image. Requires calibration first.
            goal_cost += 1 # calibrate action

            # Find calibration target and visible waypoints for a capable camera
            cal_target = None
            cal_target_wps = set()
            img_camera = None # The camera chosen for this task

            # Find a capable camera and its rover and calibration target
            # We need a camera whose target is visible from *some* waypoint.
            for c in capable_cameras:
                r = self.camera_info[c]['rover']
                t = self.camera_info[c].get('cal_target')
                if t:
                    wps = self.objective_visible_waypoints.get(t, set())
                    if wps:
                        img_camera = c
                        task_rover = r # Assign task rover here
                        cal_target = t
                        cal_target_wps = wps
                        break # Found a usable camera/target/rover

            if not img_camera or not cal_target_wps: return float('inf') # No usable camera/target/rover combo

            # Find closest capable rover (the chosen task_rover) to a calibration waypoint
            current_loc = rover_locations.get(task_rover)
            if not current_loc: return float('inf') # Rover location unknown

            cal_wp, nav_cost_to_cal = self._find_closest_waypoint(current_loc, cal_target_wps)
            if cal_wp is None or nav_cost_to_cal == float('inf'): return float('inf') # Cannot reach calibration waypoint
            goal_cost += nav_cost_to_cal # navigate to calibration waypoint

            # Rover is now at cal_wp. Need to take image.
            goal_cost += 1 # take_image action

            # Find waypoints visible from objective
            objective_wps = self.objective_visible_waypoints.get(objective, set())
            if not objective_wps: return float('inf') # Objective not visible from anywhere

            # Find closest objective waypoint from the calibration waypoint (where rover is)
            img_wp, nav_cost_to_img = self._find_closest_waypoint(cal_wp, objective_wps)
            if img_wp is None or nav_cost_to_img == float('inf'): return float('inf') # Cannot reach image waypoint
            goal_cost += nav_cost_to_img # navigate to image waypoint

            task_waypoint = img_wp # Image taken at this waypoint


        # Now calculate communication cost
        comm_start_waypoint = None
        if task_waypoint:
            # Rover is at task_waypoint after taking image
            comm_start_waypoint = task_waypoint
        else:
            # Image already existed. Find the location of a rover that has the image.
            rover_with_image = next((r for r in img_rovers if (objective, mode) in have_image.get(r, set())), None)
            if rover_with_image:
                 comm_start_waypoint = rover_locations.get(rover_with_image)
                 task_rover = rover_with_image # Assign task rover here for communication part
            # else: Should not happen if image_exists was True

        if not comm_start_waypoint: return float('inf') # Safety check (e.g., rover location unknown)

        # Find closest lander-visible waypoint from the communication start waypoint
        lander_comm_wp, nav_cost_to_lander = self._find_closest_waypoint(comm_start_waypoint, self.lander_visible_waypoints)
        if lander_comm_wp is None or nav_cost_to_lander == float('inf'): return float('inf') # Cannot reach lander waypoint

        goal_cost += nav_cost_to_lander # navigate to lander waypoint
        goal_cost += 1 # communicate action

        return goal_cost
