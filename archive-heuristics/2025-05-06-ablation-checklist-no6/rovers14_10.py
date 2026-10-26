from fnmatch import fnmatch
from collections import deque
# Assume Heuristic base class is available from the planning framework
# from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty string or malformed fact gracefully
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

# Helper function to match PDDL facts with patterns
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

# Helper function for Breadth-First Search to find shortest paths
def bfs(graph, start_node):
    """Computes shortest path distances from start_node to all reachable nodes."""
    distances = {node: float('inf') for node in graph}
    if start_node not in graph:
         # Start node is not in the graph (e.g., an object not a waypoint)
         # Or graph is empty. Return distances indicating unreachable.
         return distances

    distances[start_node] = 0
    queue = deque([start_node])

    while queue:
        current_node = queue.popleft()

        if current_node in graph: # Check again just in case
            for neighbor in graph.get(current_node, set()):
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)
    return distances


# Define the domain-dependent heuristic class
class roversHeuristic: # Inherit from Heuristic if available in the framework
    """
    A domain-dependent heuristic for the Rovers domain.

    Estimates the cost to reach the goal by summing the estimated costs
    for each uncommunicated goal fact. The cost for each goal fact is
    estimated by summing the costs of necessary actions (sample/image,
    calibrate, communicate, drop if needed) and estimated navigation costs
    to relevant locations. Navigation cost is estimated as the shortest
    path distance in the rover's navigation graph.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static
        initial_state = task.initial_state

        # --- Parse Static Facts ---
        self.lander_waypoint = None
        self.rover_equipment = {} # {rover: {equipment_type, ...}}
        self.rover_to_store = {} # {rover: store}
        self.rover_to_cameras = {} # {rover: [camera, ...]}
        self.camera_modes = set() # {(camera, mode), ...}
        self.camera_calibration_target = {} # {camera: objective}
        self.objective_visible_from = {} # {objective: {waypoint, ...}}
        self.can_traverse_graph = {} # {rover: {from_wp: {to_wp, ...}}}
        self.visible_graph = {} # {from_wp: {to_wp, ...}}
        self.all_waypoints = set() # Collect all waypoints
        self.all_rovers = set() # Collect all rovers

        # Store initial sample locations
        self.initial_soil_samples = {w for fact in initial_state if match(fact, "at_soil_sample", w)}
        self.initial_rock_samples = {w for fact in initial_state if match(fact, "at_rock_sample", w)}

        # Collect all rovers and waypoints from initial state, goals, and static facts
        all_facts = set(initial_state) | set(static_facts) | set(self.goals)
        for fact in all_facts:
             parts = get_parts(fact)
             if parts:
                  for part in parts[1:]:
                       if part.startswith('rover'): self.all_rovers.add(part)
                       if part.startswith('waypoint'): self.all_waypoints.add(part)


        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue

            pred = parts[0]
            if pred == "at_lander":
                if len(parts) == 3: self.lander_waypoint = parts[2]
            elif pred.startswith("equipped_for_"):
                if len(parts) == 2:
                    rover = parts[1]
                    eq_type = pred.split('_')[2] # e.g., 'soil' from 'equipped_for_soil_analysis'
                    self.rover_equipment.setdefault(rover, set()).add(eq_type)
            elif pred == "store_of":
                if len(parts) == 3: self.rover_to_store[parts[2]] = parts[1] # rover -> store
            elif pred == "on_board":
                if len(parts) == 3: self.rover_to_cameras.setdefault(parts[2], []).append(parts[1]) # rover -> camera
            elif pred == "supports":
                if len(parts) == 3: self.camera_modes.add((parts[1], parts[2])) # (camera, mode)
            elif pred == "calibration_target":
                if len(parts) == 3: self.camera_calibration_target[parts[1]] = parts[2] # camera -> objective
            elif pred == "visible_from":
                if len(parts) == 3: self.objective_visible_from.setdefault(parts[1], set()).add(parts[2]) # objective -> waypoint
            elif pred == "can_traverse":
                if len(parts) == 4:
                    rover, from_wp, to_wp = parts[1], parts[2], parts[3]
                    self.can_traverse_graph.setdefault(rover, {}).setdefault(from_wp, set()).add(to_wp)
            elif pred == "visible":
                 if len(parts) == 3:
                     from_wp, to_wp = parts[1], parts[2]
                     self.visible_graph.setdefault(from_wp, set()).add(to_wp)


        # --- Precompute Rover Navigation Distances ---
        self.rover_distances = {} # {rover: {from_wp: {to_wp: distance}}}
        for rover in self.all_rovers:
            # Build the actual navigation graph for this rover: requires both can_traverse and visible
            nav_graph = {}
            # Initialize graph with all known waypoints
            for wp in self.all_waypoints:
                 nav_graph.setdefault(wp, set())

            if rover in self.can_traverse_graph:
                for from_wp in self.can_traverse_graph[rover]:
                    if from_wp in self.visible_graph:
                        for to_wp in self.can_traverse_graph[rover][from_wp]:
                             if to_wp in self.visible_graph[from_wp]:
                                 nav_graph[from_wp].add(to_wp)

            self.rover_distances[rover] = {}
            for start_wp in self.all_waypoints: # Compute distances from *all* waypoints
                self.rover_distances[rover][start_wp] = bfs(nav_graph, start_wp)


        # --- Identify Communication Waypoints ---
        # Waypoints from which the lander waypoint is visible
        self.comm_wps = set()
        if self.lander_waypoint and self.lander_waypoint in self.all_waypoints:
            for wp in self.visible_graph:
                if self.lander_waypoint in self.visible_graph[wp]:
                    self.comm_wps.add(wp)
            # The lander waypoint itself is a communication point if a rover can be there
            # and communicate. The action requires (visible ?x ?y_lander) where ?x is rover loc.
            # If lander_waypoint is visible from itself, it's a comm_wp.
            if self.lander_waypoint in self.visible_graph.get(self.lander_waypoint, set()):
                 self.comm_wps.add(self.lander_waypoint)


    def get_distance(self, rover, from_wp, to_wp):
        """Helper to get precomputed distance, handling unreachable."""
        if rover in self.rover_distances and from_wp in self.rover_distances[rover] and to_wp in self.rover_distances[rover][from_wp]:
            return self.rover_distances[rover][from_wp][to_wp]
        return float('inf') # Unreachable

    def get_min_nav_cost(self, current_rover_locations, target_waypoints, relevant_rovers=None):
        """
        Finds the minimum navigation cost from any of the relevant rovers'
        current locations to any of the target waypoints.
        Returns float('inf') if no relevant rover can reach any target waypoint.
        """
        if not target_waypoints:
             return float('inf') # Cannot navigate to an empty set of targets

        min_nav = float('inf')
        rovers_to_check = relevant_rovers if relevant_rovers is not None else self.all_rovers # Check all rovers by default

        for r in rovers_to_check:
            if r in current_rover_locations:
                current_wp = current_rover_locations[r]
                for target_wp in target_waypoints:
                    dist = self.get_distance(r, current_wp, target_wp)
                    min_nav = min(min_nav, dist)
        return min_nav


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        total_h = 0

        # Get dynamic state info
        rover_locations = {}
        rover_soil_samples = set()
        rover_rock_samples = set()
        rover_images = set()
        full_stores = set()
        calibrated_cams = set()

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            pred = parts[0]
            if pred == "at":
                if len(parts) == 3 and parts[1].startswith('rover'):
                    rover, waypoint = parts[1], parts[2]
                    rover_locations[rover] = waypoint
            elif pred == "have_soil_analysis":
                if len(parts) == 3: rover_soil_samples.add((parts[1], parts[2]))
            elif pred == "have_rock_analysis":
                if len(parts) == 3: rover_rock_samples.add((parts[1], parts[2]))
            elif pred == "have_image":
                if len(parts) == 4: rover_images.add((parts[1], parts[2], parts[3]))
            elif pred == "full":
                if len(parts) == 2: full_stores.add(parts[1])
            elif pred == "calibrated":
                if len(parts) == 3: calibrated_cams.add((parts[1], parts[2]))


        # Check if goal is already reached
        if self.goals <= state:
            return 0

        # Calculate cost for each uncommunicated goal
        for goal in self.goals:
            if goal in state:
                continue # Already communicated

            parts = get_parts(goal)
            if not parts: continue

            pred = parts[0]

            if pred == "communicated_soil_data":
                # (communicated_soil_data ?w)
                if len(parts) != 2: continue
                w = parts[1]

                goal_cost = 0

                # 1. Need to communicate (1 action)
                goal_cost += 1

                # 2. Need to have the sample data
                has_sample_data = any((r, w) in rover_soil_samples for r in self.all_rovers) # Check any rover

                if not has_sample_data:
                    # Need to sample (1 action)
                    goal_cost += 1

                    # Need equipped rover with empty store at waypoint w
                    equipped_rovers = [r for r in self.all_rovers if 'soil' in self.rover_equipment.get(r, set())]
                    if not equipped_rovers: return float('inf') # Goal impossible

                    # Need empty store: Check if *any* equipped rover has a full store.
                    needs_drop = any(self.rover_to_store.get(r) in full_stores for r in equipped_rovers if self.rover_to_store.get(r))
                    if needs_drop:
                        goal_cost += 1 # drop action (1 action)

                    # Need to navigate to waypoint w for sampling
                    min_nav_to_sample = self.get_min_nav_cost(rover_locations, {w}, equipped_rovers)
                    if min_nav_to_sample == float('inf'):
                         # Cannot reach sample location w with any equipped rover
                         # Check if sample was even initially present. If not, goal impossible.
                         if w not in self.initial_soil_samples:
                              return float('inf')
                         # If sample was present but is gone and unreachable, assume impossible.
                         return float('inf')
                    goal_cost += min_nav_to_sample # Navigation cost

                # 3. Need to navigate to a communication point for communication
                min_nav_to_comm = self.get_min_nav_cost(rover_locations, self.comm_wps)
                if min_nav_to_comm == float('inf'):
                     # No rover can reach any communication point
                     return float('inf')
                goal_cost += min_nav_to_comm # Navigation cost

                total_h += goal_cost

            elif pred == "communicated_rock_data":
                # (communicated_rock_data ?w)
                if len(parts) != 2: continue
                w = parts[1]

                goal_cost = 0

                # 1. Need to communicate (1 action)
                goal_cost += 1

                # 2. Need to have the sample data
                has_sample_data = any((r, w) in rover_rock_samples for r in self.all_rovers) # Check any rover

                if not has_sample_data:
                    # Need to sample (1 action)
                    goal_cost += 1

                    # Need equipped rover with empty store at waypoint w
                    equipped_rovers = [r for r in self.all_rovers if 'rock' in self.rover_equipment.get(r, set())]
                    if not equipped_rovers: return float('inf') # Goal impossible

                    # Need empty store: Check if *any* equipped rover has a full store.
                    needs_drop = any(self.rover_to_store.get(r) in full_stores for r in equipped_rovers if self.rover_to_store.get(r))
                    if needs_drop:
                        goal_cost += 1 # drop action (1 action)

                    # Need to navigate to waypoint w for sampling
                    min_nav_to_sample = self.get_min_nav_cost(rover_locations, {w}, equipped_rovers)
                    if min_nav_to_sample == float('inf'):
                         # Cannot reach sample location w with any equipped rover
                         # Check if sample was even initially present. If not, goal impossible.
                         if w not in self.initial_rock_samples:
                              return float('inf')
                         # If sample was present but is gone and unreachable, assume impossible.
                         return float('inf')
                    goal_cost += min_nav_to_sample # Navigation cost

                # 3. Need to navigate to a communication point for communication
                min_nav_to_comm = self.get_min_nav_cost(rover_locations, self.comm_wps)
                if min_nav_to_comm == float('inf'):
                     return float('inf')
                goal_cost += min_nav_to_comm # Navigation cost

                total_h += goal_cost


            elif pred == "communicated_image_data":
                # (communicated_image_data ?o ?m)
                if len(parts) != 3: continue
                o, m = parts[1], parts[2]

                goal_cost = 0

                # 1. Need to communicate (1 action)
                goal_cost += 1

                # 2. Need to have the image data
                has_image_data = any((r, o, m) in rover_images for r in self.all_rovers) # Check any rover

                if not has_image_data:
                    # Need to take image (1 action)
                    goal_cost += 1

                    # Find suitable rovers and cameras: equipped for imaging, camera on board, camera supports mode
                    suitable_rovers_cams = [(r, i) for r in self.all_rovers if 'imaging' in self.rover_equipment.get(r, set()) for i in self.rover_to_cameras.get(r, []) if (i, m) in self.camera_modes]
                    if not suitable_rovers_cams: return float('inf') # Goal impossible

                    suitable_rovers = {r for r, i in suitable_rovers_cams}

                    # Need calibrated camera
                    is_calibrated = any((c, r) in calibrated_cams for r, c in suitable_rovers_cams)

                    if not is_calibrated:
                        # Need to calibrate (1 action)
                        goal_cost += 1

                        # Need to navigate to a calibration waypoint
                        cal_wps = set()
                        for r, i in suitable_rovers_cams:
                            cal_target = self.camera_calibration_target.get(i)
                            if cal_target and cal_target in self.objective_visible_from:
                                cal_wps.update(self.objective_visible_from[cal_target])

                        if not cal_wps: return float('inf') # No calibration waypoint found

                        min_nav_to_cal = self.get_min_nav_cost(rover_locations, cal_wps, suitable_rovers)
                        if min_nav_to_cal == float('inf'):
                             # Cannot reach any calibration waypoint with any suitable rover
                             return float('inf')
                        goal_cost += min_nav_to_cal # Navigation cost

                    # Need to navigate to an image target waypoint
                    img_wps = self.objective_visible_from.get(o, set())
                    if not img_wps: return float('inf') # No image waypoint found

                    min_nav_to_img = self.get_min_nav_cost(rover_locations, img_wps, suitable_rovers)
                    if min_nav_to_img == float('inf'):
                         # Cannot reach any image waypoint with any suitable rover
                         return float('inf')
                    goal_cost += min_nav_to_img # Navigation cost


                # 3. Need to navigate to a communication point for communication
                min_nav_to_comm = self.get_min_nav_cost(rover_locations, self.comm_wps)
                if min_nav_to_comm == float('inf'):
                     return float('inf')
                goal_cost += min_nav_to_comm # Navigation cost

                total_h += goal_cost

        return total_h
