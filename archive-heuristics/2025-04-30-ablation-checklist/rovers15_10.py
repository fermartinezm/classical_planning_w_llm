import sys
from fnmatch import fnmatch
from collections import deque
from math import inf

# Assume heuristic_base.py is available in the same directory or PYTHONPATH
# from heuristics.heuristic_base import Heuristic

# Define a dummy Heuristic base class if not available, for standalone testing
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    print("Warning: heuristics.heuristic_base not found. Using dummy base class.")
    class Heuristic:
        def __init__(self, task):
            self.goals = task.goals
            self.static = task.static
        def __call__(self, node):
            raise NotImplementedError

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty facts or malformed strings gracefully
    if not fact or fact[0] != '(' or fact[-1] != ')':
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
    """
    Performs Breadth-First Search on a graph to find shortest distances
    from a start node to all other reachable nodes.

    Args:
        graph: An adjacency list represented as a dictionary {node: [neighbor1, neighbor2, ...]}
        start_node: The node to start the BFS from.

    Returns:
        A dictionary {node: distance} containing the shortest distance from start_node
        to every reachable node. Unreachable nodes are not included.
    """
    distances = {start_node: 0}
    queue = deque([start_node])

    while queue:
        current_node = queue.popleft()

        if current_node in graph:
            for neighbor in graph[current_node]:
                if neighbor not in distances:
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)

    return distances

class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve each
    unmet goal fact independently, summing up these estimates. It considers
    navigation costs using precomputed shortest paths (BFS) and adds costs
    for sampling, imaging, calibration, dropping, and communication actions
    as needed for each goal.

    # Assumptions
    - Navigation costs are estimated using shortest paths on the can_traverse graph.
    - Resource constraints (store capacity, camera calibration) are simplified:
      - A drop action is added if a rover's store is full when sampling is needed.
      - A calibrate action is added if a camera is uncalibrated when imaging is needed.
      - It doesn't track which specific store is used or which camera is calibrated/uncalibrated
        beyond the immediate need for a goal.
    - It ignores potential negative interactions (e.g., taking an image uncalibrates a camera needed for another image goal).
    - It ignores positive interactions (e.g., one navigation action might position a rover for multiple tasks).
    - It assumes any equipped rover can perform its task if at the right location with resources.
    - Communication is assumed possible from any waypoint visible from the lander.

    # Heuristic Initialization
    - Parses static facts to identify:
        - Lander location(s).
        - Communication waypoints (visible from lander).
        - Rover capabilities (equipped for soil/rock/imaging).
        - Store ownership.
        - Camera properties (on_board, supports, calibration_target).
        - Objective visibility (visible_from).
        - Rover navigation graphs (can_traverse).
    - Precomputes shortest path distances between all pairs of waypoints for each rover using BFS.

    # Step-by-Step Thinking for Computing Heuristic
    For each goal fact that is NOT true in the current state:
    1. Add 1 to the total cost for the final 'communicate' action.
    2. Check if any rover currently possesses the required data/image (have_soil_analysis, have_rock_analysis, have_image).
    3. If YES:
       - Find the minimum navigation cost for any rover that has the data/image to reach *any* communication waypoint. Add this minimum cost to the total.
    4. If NO:
       - Determine the intermediate action needed (sample_soil, sample_rock, take_image). Add 1 to the total cost for this action.
       - If sampling (soil/rock):
         - Identify the sample waypoint.
         - Find equipped rovers for this task.
         - Calculate the minimum cost for any such rover:
           - Navigation cost from current location to sample waypoint.
           - +1 if the rover's store is full (for a drop action).
           - Navigation cost from sample waypoint to *any* communication waypoint.
         - Add the minimum of these costs over all suitable rovers to the total.
       - If imaging:
         - Identify the objective and mode.
         - Find equipped rovers with suitable cameras (on_board, supports mode).
         - Identify waypoints visible from the objective.
         - Calculate the minimum cost for any such rover/camera combination:
           - Navigation cost from current location to *any* waypoint visible from the objective.
           - +1 if the camera is not calibrated (for a calibrate action).
           - Navigation cost from the imaging waypoint to *any* communication waypoint.
         - Add the minimum of these costs over all suitable rover/camera/waypoint combinations to the total.

    The total heuristic value is the sum of costs for all unachieved goals.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        super().__init__(task)

        # --- Precompute Static Information ---
        self.lander_at = set()
        self.comm_waypoints = set() # Waypoints visible from lander
        self.rovers = set()
        self.waypoints = set()
        self.objectives = set()
        self.cameras = set()
        self.modes = set()
        self.stores = set()
        self.landers = set()

        self.rover_capabilities = {} # rover -> set of {'soil', 'rock', 'imaging'}
        self.rover_stores = {}       # rover -> store
        self.rover_cameras = {}      # rover -> set of cameras

        self.camera_info = {}        # camera -> {'supports': set of modes, 'calibration_target': objective, 'on_board_rover': rover}
        self.objective_info = {}     # objective -> {'visible_from': set of waypoints, 'calibration_target_for': set of cameras}
        self.waypoint_info = {}      # waypoint -> {'visible_from_objectives': set of objectives, 'calibration_target_for_cameras': set of cameras}

        rover_nav_graphs = {} # rover -> {wp1: [wp2, ...]} adjacency list

        # Parse static facts
        for fact in self.static:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            pred = parts[0]
            args = parts[1:]

            if pred == 'at_lander':
                lander, wp = args
                self.landers.add(lander)
                self.waypoints.add(wp)
                self.lander_at.add(wp)
            elif pred == 'at': # Initial rover locations are static in some problems? No, they are in init. This is likely types.
                 if len(args) == 2 and args[0].startswith('rover') and args[1].startswith('waypoint'):
                     self.rovers.add(args[0])
                     self.waypoints.add(args[1])
            elif pred == 'rover': self.rovers.add(args[0])
            elif pred == 'waypoint': self.waypoints.add(args[0])
            elif pred == 'objective': self.objectives.add(args[0])
            elif pred == 'camera': self.cameras.add(args[0])
            elif pred == 'mode': self.modes.add(args[0])
            elif pred == 'store': self.stores.add(args[0])
            elif pred == 'lander': self.landers.add(args[0])

            elif pred == 'equipped_for_soil_analysis':
                rover = args[0]
                self.rover_capabilities.setdefault(rover, set()).add('soil')
            elif pred == 'equipped_for_rock_analysis':
                rover = args[0]
                self.rover_capabilities.setdefault(rover, set()).add('rock')
            elif pred == 'equipped_for_imaging':
                rover = args[0]
                self.rover_capabilities.setdefault(rover, set()).add('imaging')
            elif pred == 'store_of':
                store, rover = args
                self.rover_stores[rover] = store
            elif pred == 'on_board':
                camera, rover = args
                self.rover_cameras.setdefault(rover, set()).add(camera)
                self.camera_info.setdefault(camera, {})['on_board_rover'] = rover
            elif pred == 'supports':
                camera, mode = args
                self.camera_info.setdefault(camera, {})['supports'] = self.camera_info.get(camera, {}).get('supports', set()) | {mode}
            elif pred == 'calibration_target':
                camera, objective = args
                self.camera_info.setdefault(camera, {})['calibration_target'] = objective
                self.objective_info.setdefault(objective, {})['calibration_target_for'] = self.objective_info.get(objective, {}).get('calibration_target_for', set()) | {camera}
            elif pred == 'visible_from':
                objective, wp = args
                self.objective_info.setdefault(objective, {})['visible_from'] = self.objective_info.get(objective, {}).get('visible_from', set()) | {wp}
                self.waypoint_info.setdefault(wp, {})['visible_from_objectives'] = self.waypoint_info.get(wp, {}).get('visible_from_objectives', set()) | {objective}
            elif pred == 'visible':
                 # Used for communication visibility and potentially traverse edges if can_traverse is missing/incomplete
                 # We primarily use can_traverse for rover movement graph
                 pass # Handled below for comm waypoints
            elif pred == 'can_traverse':
                rover, wp1, wp2 = args
                rover_nav_graphs.setdefault(rover, {}).setdefault(wp1, []).append(wp2)

        # Infer all objects if not explicitly typed in static (some domains omit types in static)
        for fact in self.task.initial_state | self.task.goals:
             parts = get_parts(fact)
             if not parts: continue
             pred = parts[0]
             args = parts[1:]
             if pred == 'at' and len(args) == 2:
                 if args[0].startswith('rover'): self.rovers.add(args[0])
                 if args[1].startswith('waypoint'): self.waypoints.add(args[1])
             elif pred == 'at_lander' and len(args) == 2:
                 if args[0].startswith('lander'): self.landers.add(args[0])
                 if args[1].startswith('waypoint'): self.waypoints.add(args[1])
             elif pred == 'equipped_for_soil_analysis' and len(args) == 1: self.rovers.add(args[0])
             elif pred == 'equipped_for_rock_analysis' and len(args) == 1: self.rovers.add(args[0])
             elif pred == 'equipped_for_imaging' and len(args) == 1: self.rovers.add(args[0])
             elif pred == 'empty' and len(args) == 1: self.stores.add(args[0])
             elif pred == 'have_rock_analysis' and len(args) == 2: self.rovers.add(args[0]); self.waypoints.add(args[1])
             elif pred == 'have_soil_analysis' and len(args) == 2: self.rovers.add(args[0]); self.waypoints.add(args[1])
             elif pred == 'full' and len(args) == 1: self.stores.add(args[0])
             elif pred == 'calibrated' and len(args) == 2: self.cameras.add(args[0]); self.rovers.add(args[1])
             elif pred == 'supports' and len(args) == 2: self.cameras.add(args[0]); self.modes.add(args[1])
             elif pred == 'visible' and len(args) == 2: self.waypoints.add(args[0]); self.waypoints.add(args[1])
             elif pred == 'have_image' and len(args) == 3: self.rovers.add(args[0]); self.objectives.add(args[1]); self.modes.add(args[2])
             elif pred == 'communicated_soil_data' and len(args) == 1: self.waypoints.add(args[0])
             elif pred == 'communicated_rock_data' and len(args) == 1: self.waypoints.add(args[0])
             elif pred == 'communicated_image_data' and len(args) == 2: self.objectives.add(args[0]); self.modes.add(args[1])
             elif pred == 'at_soil_sample' and len(args) == 1: self.waypoints.add(args[0])
             elif pred == 'at_rock_sample' and len(args) == 1: self.waypoints.add(args[0])
             elif pred == 'visible_from' and len(args) == 2: self.objectives.add(args[0]); self.waypoints.add(args[1])
             elif pred == 'store_of' and len(args) == 2: self.stores.add(args[0]); self.rovers.add(args[1])
             elif pred == 'calibration_target' and len(args) == 2: self.cameras.add(args[0]); self.objectives.add(args[1])
             elif pred == 'on_board' and len(args) == 2: self.cameras.add(args[0]); self.rovers.add(args[1])


        # Determine communication waypoints: visible from any lander location
        lander_locations = {get_parts(fact)[2] for fact in self.static if match(fact, "at_lander", "*", "*")}
        visible_facts = {tuple(get_parts(fact)[1:]) for fact in self.static if match(fact, "visible", "*", "*")}

        self.comm_waypoints = set()
        for wp1, wp2 in visible_facts:
            if wp1 in lander_locations:
                self.comm_waypoints.add(wp2)
            if wp2 in lander_locations:
                self.comm_waypoints.add(wp1)

        # Build rover navigation graphs (ensure all waypoints are included even if isolated)
        self.rover_nav_graphs = {}
        for rover in self.rovers:
             self.rover_nav_graphs[rover] = {wp: [] for wp in self.waypoints} # Initialize with all waypoints
             if rover in rover_nav_graphs:
                 for wp1, neighbors in rover_nav_graphs[rover].items():
                     self.rover_nav_graphs[rover][wp1].extend(neighbors)


        # Precompute navigation distances for each rover
        self.rover_nav_dists = {} # rover -> {start_wp: {end_wp: dist}}
        for rover in self.rovers:
            self.rover_nav_dists[rover] = {}
            # Run BFS from every waypoint as a potential start node
            for start_wp in self.waypoints:
                 self.rover_nav_dists[rover][start_wp] = bfs(self.rover_nav_graphs.get(rover, {}), start_wp)


    def get_nav_cost(self, rover, start_wp, end_wp):
        """Get precomputed navigation cost, return inf if unreachable."""
        if rover not in self.rover_nav_dists or start_wp not in self.rover_nav_dists[rover] or end_wp not in self.rover_nav_dists[rover][start_wp]:
             # This can happen if a rover cannot traverse at all, or if waypoints are isolated
             return inf
        return self.rover_nav_dists[rover][start_wp][end_wp]


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # --- Parse Current State ---
        current_rover_at = {} # rover -> waypoint
        current_store_full = {} # store -> boolean
        current_camera_calibrated = {} # camera -> boolean
        current_soil_samples_at = set() # waypoint
        current_rock_samples_at = set() # waypoint
        current_have_soil_analysis = {} # rover -> set of waypoints
        current_have_rock_analysis = {} # rover -> set of waypoints
        current_have_image = {} # rover -> set of (objective, mode)
        current_communicated_soil = set() # waypoint
        current_communicated_rock = set() # waypoint
        current_communicated_image = set() # (objective, mode)

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            pred = parts[0]
            args = parts[1:]

            if pred == 'at' and len(args) == 2 and args[0].startswith('rover'):
                current_rover_at[args[0]] = args[1]
            elif pred == 'full' and len(args) == 1 and args[0].startswith('store'):
                current_store_full[args[0]] = True
            elif pred == 'empty' and len(args) == 1 and args[0].startswith('store'):
                 current_store_full[args[0]] = False # Explicitly mark as not full
            elif pred == 'calibrated' and len(args) == 2 and args[0].startswith('camera'):
                current_camera_calibrated[args[0]] = True
            elif pred == 'at_soil_sample' and len(args) == 1 and args[0].startswith('waypoint'):
                current_soil_samples_at.add(args[0])
            elif pred == 'at_rock_sample' and len(args) == 1 and args[0].startswith('waypoint'):
                current_rock_samples_at.add(args[0])
            elif pred == 'have_soil_analysis' and len(args) == 2 and args[0].startswith('rover'):
                current_have_soil_analysis.setdefault(args[0], set()).add(args[1])
            elif pred == 'have_rock_analysis' and len(args) == 2 and args[0].startswith('rover'):
                current_have_rock_analysis.setdefault(args[0], set()).add(args[1])
            elif pred == 'have_image' and len(args) == 3 and args[0].startswith('rover'):
                current_have_image.setdefault(args[0], set()).add((args[1], args[2]))
            elif pred == 'communicated_soil_data' and len(args) == 1 and args[0].startswith('waypoint'):
                current_communicated_soil.add(args[0])
            elif pred == 'communicated_rock_data' and len(args) == 1 and args[0].startswith('waypoint'):
                current_communicated_rock.add(args[0])
            elif pred == 'communicated_image_data' and len(args) == 2 and args[0].startswith('objective'):
                current_communicated_image.add(tuple(args)) # Store as tuple (objective, mode)

        total_cost = 0

        # --- Estimate Cost for Each Unachieved Goal ---
        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue

            pred = parts[0]
            args = parts[1:]

            if pred == 'communicated_soil_data' and len(args) == 1:
                wp_sample = args[0]
                if wp_sample in current_communicated_soil:
                    continue # Goal already achieved

                # Cost for communication action
                goal_cost = 1

                # Check if any rover already has the soil data
                rover_with_data = None
                for rover in self.rovers:
                    if wp_sample in current_have_soil_analysis.get(rover, set()):
                        rover_with_data = rover # Found a rover with the data
                        break # Simple: just find one, don't need the best yet

                if rover_with_data:
                    # Rover has data, needs to navigate to a communication waypoint
                    min_nav_to_comm = inf
                    current_wp = current_rover_at.get(rover_with_data)
                    if current_wp:
                        for comm_wp in self.comm_waypoints:
                            min_nav_to_comm = min(min_nav_to_comm, self.get_nav_cost(rover_with_data, current_wp, comm_wp))
                    goal_cost += min_nav_to_comm if min_nav_to_comm != inf else 100 # Add nav cost, or large penalty if unreachable

                else:
                    # No rover has data, needs to sample first
                    goal_cost += 1 # Cost for sample_soil action

                    # Find the best rover to sample and communicate
                    min_total_task_cost = inf
                    for rover in self.rovers:
                        if 'soil' in self.rover_capabilities.get(rover, set()):
                            current_wp = current_rover_at.get(rover)
                            if current_wp:
                                nav_to_sample_cost = self.get_nav_cost(rover, current_wp, wp_sample)
                                if nav_to_sample_cost != inf:
                                    # Cost to sample: nav + sample + (drop if needed)
                                    sample_cost = nav_to_sample_cost + 1
                                    store = self.rover_stores.get(rover)
                                    if store and current_store_full.get(store, False):
                                        sample_cost += 1 # Add drop cost

                                    # Cost to communicate after sampling: nav + communicate
                                    min_nav_sample_to_comm = inf
                                    for comm_wp in self.comm_waypoints:
                                        min_nav_sample_to_comm = min(min_nav_sample_to_comm, self.get_nav_cost(rover, wp_sample, comm_wp))

                                    if min_nav_sample_to_comm != inf:
                                         # Total cost for this rover: nav_to_sample + sample + (drop) + nav_to_comm + communicate
                                         # We already added 1 for communicate and 1 for sample action at the start of the 'else' block
                                         # So just add the navigation and drop costs here
                                         current_rover_task_cost = nav_to_sample_cost + (1 if store and current_store_full.get(store, False) else 0) + min_nav_sample_to_comm
                                         min_total_task_cost = min(min_total_task_cost, current_rover_task_cost)

                    goal_cost += min_total_task_cost if min_total_task_cost != inf else 100 # Add min task cost, or large penalty

                total_cost += goal_cost

            elif pred == 'communicated_rock_data' and len(args) == 1:
                wp_sample = args[0]
                if wp_sample in current_communicated_rock:
                    continue # Goal already achieved

                # Cost for communication action
                goal_cost = 1

                # Check if any rover already has the rock data
                rover_with_data = None
                for rover in self.rovers:
                    if wp_sample in current_have_rock_analysis.get(rover, set()):
                        rover_with_data = rover
                        break

                if rover_with_data:
                    # Rover has data, needs to navigate to a communication waypoint
                    min_nav_to_comm = inf
                    current_wp = current_rover_at.get(rover_with_data)
                    if current_wp:
                        for comm_wp in self.comm_waypoints:
                            min_nav_to_comm = min(min_nav_to_comm, self.get_nav_cost(rover_with_data, current_wp, comm_wp))
                    goal_cost += min_nav_to_comm if min_nav_to_comm != inf else 100

                else:
                    # No rover has data, needs to sample first
                    goal_cost += 1 # Cost for sample_rock action

                    # Find the best rover to sample and communicate
                    min_total_task_cost = inf
                    for rover in self.rovers:
                        if 'rock' in self.rover_capabilities.get(rover, set()):
                            current_wp = current_rover_at.get(rover)
                            if current_wp:
                                nav_to_sample_cost = self.get_nav_cost(rover, current_wp, wp_sample)
                                if nav_to_sample_cost != inf:
                                    # Cost to sample: nav + sample + (drop if needed)
                                    sample_cost = nav_to_sample_cost + 1
                                    store = self.rover_stores.get(rover)
                                    if store and current_store_full.get(store, False):
                                        sample_cost += 1 # Add drop cost

                                    # Cost to communicate after sampling: nav + communicate
                                    min_nav_sample_to_comm = inf
                                    for comm_wp in self.comm_waypoints:
                                        min_nav_sample_to_comm = min(min_nav_sample_to_comm, self.get_nav_cost(rover, wp_sample, comm_wp))

                                    if min_nav_sample_to_comm != inf:
                                         # Total cost for this rover: nav_to_sample + sample + (drop) + nav_to_comm + communicate
                                         # We already added 1 for communicate and 1 for sample action
                                         current_rover_task_cost = nav_to_sample_cost + (1 if store and current_store_full.get(store, False) else 0) + min_nav_sample_to_comm
                                         min_total_task_cost = min(min_total_task_cost, current_rover_task_cost)

                    goal_cost += min_total_task_cost if min_total_task_cost != inf else 100

                total_cost += goal_cost

            elif pred == 'communicated_image_data' and len(args) == 2:
                objective, mode = args
                if (objective, mode) in current_communicated_image:
                    continue # Goal already achieved

                # Cost for communication action
                goal_cost = 1

                # Check if any rover already has the image
                rover_with_image = None
                for rover in self.rovers:
                    if (objective, mode) in current_have_image.get(rover, set()):
                        rover_with_image = rover
                        break

                if rover_with_image:
                    # Rover has image, needs to navigate to a communication waypoint
                    min_nav_to_comm = inf
                    current_wp = current_rover_at.get(rover_with_image)
                    if current_wp:
                        for comm_wp in self.comm_waypoints:
                            min_nav_to_comm = min(min_nav_to_comm, self.get_nav_cost(rover_with_image, current_wp, comm_wp))
                    goal_cost += min_nav_to_comm if min_nav_to_comm != inf else 100

                else:
                    # No rover has image, needs to take image first
                    goal_cost += 1 # Cost for take_image action

                    # Find the best rover/camera/waypoint combination to take image and communicate
                    min_total_task_cost = inf
                    suitable_imaging_wps = self.objective_info.get(objective, {}).get('visible_from', set())

                    if not suitable_imaging_wps:
                         # Cannot take image of this objective
                         min_total_task_cost = inf # Task impossible

                    else:
                        for rover in self.rovers:
                            if 'imaging' in self.rover_capabilities.get(rover, set()):
                                current_wp = current_rover_at.get(rover)
                                if current_wp:
                                    for camera in self.rover_cameras.get(rover, set()):
                                        if mode in self.camera_info.get(camera, {}).get('supports', set()):
                                            # Found a suitable rover/camera
                                            # Find best waypoint to take image from
                                            min_nav_to_imaging_wp = inf
                                            best_imaging_wp = None
                                            for imaging_wp in suitable_imaging_wps:
                                                nav_cost = self.get_nav_cost(rover, current_wp, imaging_wp)
                                                if nav_cost != inf and nav_cost < min_nav_to_imaging_wp:
                                                    min_nav_to_imaging_wp = nav_cost
                                                    best_imaging_wp = imaging_wp

                                            if best_imaging_wp: # Found a reachable imaging waypoint
                                                # Cost to image: nav + image + (calibrate if needed)
                                                image_cost = min_nav_to_imaging_wp + 1
                                                if not current_camera_calibrated.get(camera, False):
                                                    image_cost += 1 # Add calibrate cost

                                                # Cost to communicate after imaging: nav + communicate
                                                min_nav_image_to_comm = inf
                                                for comm_wp in self.comm_waypoints:
                                                    min_nav_image_to_comm = min(min_nav_image_to_comm, self.get_nav_cost(rover, best_imaging_wp, comm_wp))

                                                if min_nav_image_to_comm != inf:
                                                    # Total cost for this rover/camera: nav_to_imaging + image + (calibrate) + nav_to_comm + communicate
                                                    # We already added 1 for communicate and 1 for take_image action
                                                    current_rover_camera_task_cost = min_nav_to_imaging_wp + (1 if not current_camera_calibrated.get(camera, False) else 0) + min_nav_image_to_comm
                                                    min_total_task_cost = min(min_total_task_cost, current_rover_camera_task_cost)

                    goal_cost += min_total_task_cost if min_total_task_cost != inf else 100

                total_cost += goal_cost

        # Return 0 if all goals are met, otherwise the estimated cost
        # The loop above only adds cost for *unmet* goals. If total_cost is 0, all goals were met.
        return total_cost

