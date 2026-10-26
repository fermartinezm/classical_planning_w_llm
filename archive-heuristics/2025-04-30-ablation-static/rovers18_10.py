from fnmatch import fnmatch
from collections import deque
import math # For float('inf')

# Assuming Heuristic base class is available in heuristics.heuristic_base
# If running standalone, you might need to define a dummy base class
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    # Define a dummy Heuristic base class if the module is not found
    class Heuristic:
        def __init__(self, task):
            self.goals = task.goals
            self.static = task.static
            self.initial_state = task.initial_state # Need initial state for samples

        def __call__(self, node):
            raise NotImplementedError


# Helper function to parse PDDL fact strings
def get_parts(fact_string):
    """Extract the components of a PDDL fact string."""
    # Remove parentheses and split by spaces
    return fact_string[1:-1].split()

# Helper function for BFS
def bfs_with_target(graph, start_node, end_nodes):
    """
    Find the shortest distance and the target node reached from start_node
    to any node in end_nodes in an unweighted graph.
    graph: adjacency list {node: [neighbor1, neighbor2, ...]}
    start_node: the starting waypoint
    end_nodes: a set of target waypoints
    Returns (shortest distance, target_node), or (float('inf'), None) if no path exists.
    """
    if start_node in end_nodes:
        return (0, start_node)

    queue = deque([(start_node, 0)])
    visited = {start_node}

    while queue:
        current_node, distance = queue.popleft()

        # Check if current_node is in graph keys, handles isolated nodes
        if current_node in graph:
            for neighbor in graph[current_node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    if neighbor in end_nodes:
                        return (distance + 1, neighbor)
                    queue.append((neighbor, distance + 1))

    return (math.inf, None) # No path found

# Simple BFS for distance only
def bfs(graph, start_node, end_nodes):
     result = bfs_with_target(graph, start_node, end_nodes)
     return result[0]


class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve all
    unmet goal conditions. It sums the estimated costs for each individual
    unmet goal fact. The cost for a goal fact is the minimum number of
    actions required by any capable rover to achieve that specific fact,
    including navigation, sampling/imaging, and communication steps.

    # Assumptions
    - The navigation graph defined by `can_traverse` and `visible` is static.
      For simplicity and efficiency, this heuristic builds a single undirected
      graph from all `can_traverse` facts, assuming any rover can use any
      such link in either direction.
    - Action costs are uniform (1).
    - Soil/rock samples, objectives, calibration targets, camera capabilities,
      and rover equipment are static (defined in the initial state/static facts).
    - If a soil/rock sample required by a goal was not present in the initial
      state, that goal is considered impossible.
    - Calibration is required before each image is taken.
    - The lander location is static.

    # Heuristic Initialization
    The heuristic pre-processes static facts from the task and initial state
    to build data structures needed for efficient lookups during heuristic
    computation:
    - `lander_location`: The waypoint where the lander is located.
    - `waypoint_graph`: An adjacency list representing the navigation graph
      between waypoints based on `can_traverse` facts.
    - `comm_wps_by_lander_wp`: A set of waypoints visible from the lander's
      location, which are potential communication points.
    - `equipped_rovers`: Dictionaries mapping equipment type ('soil', 'rock',
      'imaging') to sets of rovers possessing that equipment.
    - `rover_stores`: A dictionary mapping rovers to their associated store objects.
    - `rover_cameras`: A dictionary mapping rovers to lists of cameras on board.
    - `camera_modes`: A dictionary mapping cameras to sets of modes they support.
    - `camera_cal_target`: A dictionary mapping cameras to their calibration
      target objectives.
    - `objective_visible_wps`: A dictionary mapping objectives to sets of
      waypoints from which they are visible.
    - `caltarget_visible_wps`: A dictionary mapping calibration target objectives
      to sets of waypoints visible from them (derived from `objective_visible_wps`
      and `camera_cal_target`).
    - `initial_soil_samples`: A set of waypoints where soil samples existed
      in the initial state.
    - `initial_rock_samples`: A set of waypoints where rock samples existed
      in the initial state.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic calculates the sum of estimated costs
    for each goal fact that is not yet satisfied in the state.

    For each unachieved goal fact `G`:

    1.  **If `G` is `(communicated_soil_data ?w)`:**
        *   Check if a soil sample was initially available at `?w`. If not, this goal is impossible (cost = 1000).
        *   Find the minimum cost among all rovers equipped for soil analysis:
            *   Get the rover's current location from the state.
            *   Calculate the cost to get the sample *if* the rover doesn't already have it in the state:
                *   Add 1 if the rover's store is currently full in the state (for the `drop` action).
                *   Calculate navigation cost from the rover's current location to `?w` using BFS. Add this cost.
                *   Add 1 for the `sample_soil` action.
                *   The rover's effective location for the next step becomes `?w`.
            *   Calculate the cost to communicate the sample data:
                *   Calculate navigation cost from the rover's effective location (where it got the sample, or its current location if it already had the sample) to the closest communication waypoint (visible from the lander) using BFS. Add this cost.
                *   Add 1 for the `communicate_soil_data` action.
            *   The total cost for this rover is the sum of sample cost (if needed) and communication cost.
        *   The cost for this goal fact `G` is the minimum total cost found across all capable rovers. If no capable rover can reach the necessary locations, the cost is 1000.

    2.  **If `G` is `(communicated_rock_data ?w)`:**
        *   Follow the same logic as for soil data, using rock-specific equipment, samples, and actions.

    3.  **If `G` is `(communicated_image_data ?o ?m)`:**
        *   Find the minimum cost among all (rover, camera) pairs where the rover is equipped for imaging, the camera is on board, and the camera supports mode `?m`:
            *   Get the rover's current location from the state.
            *   Calculate the cost to get the image *if* the rover doesn't already have it in the state:
                *   Find the camera's calibration target. If none, this pair cannot achieve the goal.
                *   Find waypoints visible from the calibration target and the objective `?o`. If none exist for either, this pair cannot achieve the goal.
                *   Calculate navigation cost from the rover's current location to the closest calibration waypoint using BFS. Add this cost.
                *   Add 1 for the `calibrate` action.
                *   The rover's effective location becomes the reached calibration waypoint.
                *   Calculate navigation cost from the calibration waypoint to the closest image waypoint (visible from `?o`) using BFS. Add this cost.
                *   Add 1 for the `take_image` action.
                *   The rover's effective location for the next step becomes the reached image waypoint.
            *   Calculate the cost to communicate the image data:
                *   Calculate navigation cost from the rover's effective location (where it took the image, or its current location if it already had the image) to the closest communication waypoint (visible from the lander) using BFS. Add this cost.
                *   Add 1 for the `communicate_image_data` action.
            *   The total cost for this (rover, camera) pair is the sum of image cost (if needed) and communication cost.
        *   The cost for this goal fact `G` is the minimum total cost found across all capable (rover, camera) pairs. If no capable pair can achieve the goal, the cost is 1000.

    The total heuristic value is the sum of the costs calculated for each unachieved goal fact.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static
        initial_state = task.initial_state # Need initial state to check for initial samples

        # Pre-process static facts
        self.lander_location = None
        self.waypoint_graph = {} # Adjacency list {wp: {neighbor1, ...}}
        self.comm_wps_by_lander_wp = set() # {comm_wp1, ...}
        self.equipped_rovers = {'soil': set(), 'rock': set(), 'imaging': set()}
        self.rover_stores = {} # {rover: store}
        self.rover_cameras = {} # {rover: [camera1, ...]}
        self.camera_modes = {} # {camera: {mode1, ...}}
        self.camera_cal_target = {} # {camera: objective}
        self.objective_visible_wps = {} # {objective: {wp1, ...}}
        self.caltarget_visible_wps = {} # {objective: {wp1, ...}} derived from objective_visible_wps
        self.initial_soil_samples = set() # {wp1, ...}
        self.initial_rock_samples = set() # {wp1, ...}

        all_waypoints = set()

        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'at_lander':
                # Assuming only one lander
                self.lander_location = parts[2]
                all_waypoints.add(parts[2])
            elif parts[0] == 'can_traverse':
                 # Build a combined graph using all traversable links
                 wp1, wp2 = parts[2], parts[3]
                 self.waypoint_graph.setdefault(wp1, set()).add(wp2)
                 self.waypoint_graph.setdefault(wp2, set()).add(wp1) # Assuming symmetric traverse
                 all_waypoints.add(wp1)
                 all_waypoints.add(wp2)
            elif parts[0] == 'equipped_for_soil_analysis':
                self.equipped_rovers['soil'].add(parts[1])
            elif parts[0] == 'equipped_for_rock_analysis':
                self.equipped_rovers['rock'].add(parts[1])
            elif parts[0] == 'equipped_for_imaging':
                self.equipped_rovers['imaging'].add(parts[1])
            elif parts[0] == 'store_of':
                self.rover_stores[parts[2]] = parts[1] # {rover: store}
            elif parts[0] == 'on_board':
                self.rover_cameras.setdefault(parts[2], []).append(parts[1]) # {rover: [camera]}
            elif parts[0] == 'supports':
                self.camera_modes.setdefault(parts[1], set()).add(parts[2]) # {camera: {mode}}
            elif parts[0] == 'calibration_target':
                self.camera_cal_target[parts[1]] = parts[2] # {camera: objective}
            elif parts[0] == 'visible_from':
                self.objective_visible_wps.setdefault(parts[1], set()).add(parts[2]) # {objective: {wp}}

        # Add all waypoints mentioned in initial state (at rover, at sample)
        for fact in initial_state:
             parts = get_parts(fact)
             if parts[0] == 'at' and len(parts) == 3 and parts[1].startswith('rover'):
                 all_waypoints.add(parts[2])
             elif parts[0] == 'at_soil_sample':
                 self.initial_soil_samples.add(parts[1])
                 all_waypoints.add(parts[1])
             elif parts[0] == 'at_rock_sample':
                 self.initial_rock_samples.add(parts[1])
                 all_waypoints.add(parts[1])


        # Ensure all waypoints mentioned are nodes in the graph, even if they have no edges
        for wp in all_waypoints:
             self.waypoint_graph.setdefault(wp, set())

        # Precompute communication waypoints visible from the lander location
        if self.lander_location:
             # Find all waypoints ?x such that (visible ?x lander_wp) is true.
             lander_wp = self.lander_location
             for fact in static_facts:
                 parts = get_parts(fact)
                 if parts[0] == 'visible' and len(parts) == 3 and parts[2] == lander_wp:
                     self.comm_wps_by_lander_wp.add(parts[1])

        # Precompute calibration waypoints visible from calibration targets
        self.caltarget_visible_wps = {}
        for camera, cal_target in self.camera_cal_target.items():
             if cal_target in self.objective_visible_wps:
                 self.caltarget_visible_wps[cal_target] = self.objective_visible_wps[cal_target]

        # Convert waypoint graph adjacency sets to lists for BFS
        # Ensure all nodes in the graph have an entry, even if empty list
        self.waypoint_graph = {wp: list(neighbors) for wp, neighbors in self.waypoint_graph.items()}
        for wp in all_waypoints:
             self.waypoint_graph.setdefault(wp, [])


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state (frozenset of strings)

        total_heuristic_cost = 0
        IMPOSSIBLE_COST = 1000 # Use a large value for impossible goals

        # Find current rover locations and store states
        rover_locations = {} # {rover: waypoint}
        store_full_status = {} # {store: bool}
        # Use sets for quick lookups of have_X facts
        have_soil_facts = set()
        have_rock_facts = set()
        have_image_facts = set()

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at' and len(parts) == 3 and parts[1].startswith('rover'):
                rover_locations[parts[1]] = parts[2]
            elif parts[0] == 'full' and len(parts) == 2 and parts[1].startswith('rover') and parts[1].endswith('store'):
                 store_full_status[parts[1]] = True
            # No need to track 'empty', default is not full unless 'full' is present
            elif parts[0] == 'have_soil_analysis' and len(parts) == 3:
                 have_soil_facts.add(fact)
            elif parts[0] == 'have_rock_analysis' and len(parts) == 3:
                 have_rock_facts.add(fact)
            elif parts[0] == 'have_image' and len(parts) == 4:
                 have_image_facts.add(fact)


        # Iterate through goal conditions
        for goal_fact in self.goals:
            if goal_fact in state:
                continue # Goal already achieved

            parts = get_parts(goal_fact)
            predicate = parts[0]

            if predicate == 'communicated_soil_data':
                waypoint_w = parts[1]
                goal_cost = IMPOSSIBLE_COST

                # Check if sample was initially available
                if waypoint_w not in self.initial_soil_samples:
                    # This goal is impossible from the start
                    total_heuristic_cost += goal_cost
                    continue

                min_rover_cost = math.inf

                # Consider all rovers equipped for soil analysis
                for rover in self.equipped_rovers.get('soil', set()):
                    if rover not in rover_locations: continue # Rover doesn't exist or isn't 'at' a waypoint? (Shouldn't happen in valid states)
                    r_wp = rover_locations[rover]
                    rover_cost = 0
                    current_pos_after_sample = r_wp # Default if sample already had

                    # Cost to get sample if needed
                    have_sample_fact = f'(have_soil_analysis {rover} {waypoint_w})'
                    if have_sample_fact not in have_soil_facts:
                        # Need to get sample
                        store = self.rover_stores.get(rover)
                        # Check if store is full in current state *before* navigating to sample
                        if store and store_full_status.get(store, False):
                             rover_cost += 1 # Add cost for drop action at current location

                        nav_to_sample_cost = bfs(self.waypoint_graph, r_wp, {waypoint_w})
                        if nav_to_sample_cost == math.inf: continue # Cannot reach sample location

                        rover_cost += nav_to_sample_cost + 1 # navigate + sample
                        current_pos_after_sample = waypoint_w # Rover is now at sample location

                    # Cost to communicate from current_pos_after_sample
                    comm_wps = self.comm_wps_by_lander_wp
                    if not comm_wps:
                         nav_to_comm_cost = math.inf
                    else:
                         nav_to_comm_cost = bfs(self.waypoint_graph, current_pos_after_sample, comm_wps)

                    if nav_to_comm_cost == math.inf: continue # Cannot reach a comm point

                    rover_cost += nav_to_comm_cost + 1 # navigate + communicate
                    min_rover_cost = min(min_rover_cost, rover_cost)

                goal_cost = min_rover_cost if min_rover_cost != math.inf else IMPOSSIBLE_COST
                total_heuristic_cost += goal_cost

            elif predicate == 'communicated_rock_data':
                waypoint_w = parts[1]
                goal_cost = IMPOSSIBLE_COST

                # Check if sample was initially available
                if waypoint_w not in self.initial_rock_samples:
                    # This goal is impossible from the start
                    total_heuristic_cost += goal_cost
                    continue

                min_rover_cost = math.inf

                # Consider all rovers equipped for rock analysis
                for rover in self.equipped_rovers.get('rock', set()):
                    if rover not in rover_locations: continue
                    r_wp = rover_locations[rover]
                    rover_cost = 0
                    current_pos_after_sample = r_wp # Default if sample already had

                    # Cost to get sample if needed
                    have_sample_fact = f'(have_rock_analysis {rover} {waypoint_w})'
                    if have_sample_fact not in have_rock_facts:
                        # Need to get sample
                        store = self.rover_stores.get(rover)
                        # Check if store is full in current state *before* navigating to sample
                        if store and store_full_status.get(store, False):
                             rover_cost += 1 # Add cost for drop action at current location

                        nav_to_sample_cost = bfs(self.waypoint_graph, r_wp, {waypoint_w})
                        if nav_to_sample_cost == math.inf: continue # Cannot reach sample location

                        rover_cost += nav_to_sample_cost + 1 # navigate + sample
                        current_pos_after_sample = waypoint_w # Rover is now at sample location

                    # Cost to communicate from current_pos_after_sample
                    comm_wps = self.comm_wps_by_lander_wp
                    if not comm_wps:
                         nav_to_comm_cost = math.inf
                    else:
                         nav_to_comm_cost = bfs(self.waypoint_graph, current_pos_after_sample, comm_wps)

                    if nav_to_comm_cost == math.inf: continue # Cannot reach a comm point

                    rover_cost += nav_to_comm_cost + 1 # navigate + communicate
                    min_rover_cost = min(min_rover_cost, rover_cost)

                goal_cost = min_rover_cost if min_rover_cost != math.inf else IMPOSSIBLE_COST
                total_heuristic_cost += goal_cost

            elif predicate == 'communicated_image_data':
                objective_o = parts[1]
                mode_m = parts[2]
                goal_cost = IMPOSSIBLE_COST

                min_rover_camera_cost = math.inf

                # Consider all rovers equipped for imaging
                for rover in self.equipped_rovers.get('imaging', set()):
                    if rover not in rover_locations: continue
                    r_wp = rover_locations[rover]

                    # Consider all cameras on this rover
                    for camera in self.rover_cameras.get(rover, []):
                        # Check if camera supports the required mode
                        if mode_m in self.camera_modes.get(camera, set()):
                            rover_camera_cost = 0
                            current_pos_after_image = r_wp # Default if image already had

                            # Cost to get image if needed
                            have_image_fact = f'(have_image {rover} {objective_o} {mode_m})'
                            if have_image_fact not in have_image_facts:
                                # Need to take image
                                cal_target = self.camera_cal_target.get(camera)
                                if not cal_target: continue # Camera has no calibration target

                                cal_wps = self.caltarget_visible_wps.get(cal_target, set())
                                if not cal_wps: continue # Calibration target not visible from anywhere

                                img_wps = self.objective_visible_wps.get(objective_o, set())
                                if not img_wps: continue # Objective not visible from anywhere

                                # Cost to calibrate
                                nav_result_cal = bfs_with_target(self.waypoint_graph, r_wp, cal_wps)
                                if nav_result_cal[0] == math.inf: continue # Cannot reach calibration point
                                rover_camera_cost += nav_result_cal[0] + 1 # navigate + calibrate
                                cal_wp_reached = nav_result_cal[1]

                                # Cost to take image
                                nav_result_img = bfs_with_target(self.waypoint_graph, cal_wp_reached, img_wps)
                                if nav_result_img[0] == math.inf: continue # Cannot reach image point
                                rover_camera_cost += nav_result_img[0] + 1 # navigate + take_image
                                current_pos_after_image = nav_result_img[1] # Rover is now at image location

                            # Cost to communicate from current_pos_after_image
                            comm_wps = self.comm_wps_by_lander_wp
                            if not comm_wps:
                                nav_to_comm_cost = math.inf
                            else:
                                nav_to_comm_cost = bfs(self.waypoint_graph, current_pos_after_image, comm_wps)

                            if nav_to_comm_cost == math.inf: continue # Cannot reach a comm point

                            rover_camera_cost += nav_to_comm_cost + 1 # navigate + communicate
                            min_rover_camera_cost = min(min_rover_camera_cost, rover_camera_cost)

                goal_cost = min_rover_camera_cost if min_rover_camera_cost != math.inf else IMPOSSIBLE_COST
                total_heuristic_cost += goal_cost

            # Add other goal types if any (none in this domain)

        return total_heuristic_cost
