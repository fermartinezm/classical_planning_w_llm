# Import necessary modules
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import collections

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

# Helper function to match PDDL facts
def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    Wildcards `*` are allowed in `args`.
    """
    parts = get_parts(fact)
    # Ensure we don't go out of bounds if fact has fewer parts than args
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# BFS distance helper that returns distance and the target node reached
def bfs_distance(graph, start_node, target_nodes):
    """
    Finds the shortest distance (number of edges) from start_node to any node in target_nodes
    in an unweighted graph, and returns the distance and the target node reached.

    Args:
        graph: A dictionary representing the graph {node: set_of_neighbors}.
        start_node: The starting node.
        target_nodes: A set of target nodes.

    Returns:
        A tuple (distance, reached_target_node), or (float('inf'), None) if no path exists.
    """
    if start_node in target_nodes:
        return 0, start_node

    queue = collections.deque([(start_node, 0)])
    visited = {start_node}

    while queue:
        current_node, distance = queue.popleft()

        # Ensure the node exists in the graph keys before accessing neighbors
        if current_node in graph:
            for neighbor in graph[current_node]:
                if neighbor in target_nodes:
                    return distance + 1, neighbor
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, distance + 1))

    return float('inf'), None # No path found to any target node


class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve all goal
    conditions. It sums the estimated costs for each unachieved goal fact
    independently. The cost for each goal fact is estimated by finding the
    minimum number of navigation, sampling/imaging, and communication actions
    required for the most suitable rover, considering necessary equipment and
    samples/objectives are available (or were available initially). Navigation
    costs are estimated using shortest path (BFS) on the rover's traversability graph,
    calculating distances sequentially between required locations (sample/image/calibration
    waypoint and communication waypoint).

    # Assumptions
    - The heuristic assumes that achieving one goal does not negatively impact
      the ability to achieve another (additive relaxation).
    - It assumes that a single equipped rover can perform all steps (sampling/imaging,
      moving to communication point, communicating) for a given data type goal.
    - It assumes that soil/rock samples initially present remain available until sampled
      (even though the predicate is removed, the *opportunity* to sample existed).
    - It simplifies the imaging process: it only accounts for calibration if the camera
      is currently uncalibrated and doesn't consider recalibration needs after taking
      an image for subsequent images.
    - It finds the minimum cost over suitable rovers/cameras for each goal independently.
    - Action costs are assumed to be 1.
    - If a goal requires a sample/objective/target that was not in the initial state
      or cannot be reached, that goal is considered impossible by that rover/camera path
      and contributes 0 to the heuristic.

    # Heuristic Initialization
    The initialization phase pre-computes static information from the task:
    - The traversability graph for each rover.
    - The lander's location.
    - Waypoints visible from the lander's location (communication points).
    - Which rovers are equipped for soil, rock, and imaging.
    - Which stores belong to which rovers.
    - Which cameras are on which rovers and which modes they support.
    - Calibration targets for each camera.
    - Waypoints from which objectives/targets are visible.
    - The set of initial soil and rock sample locations (needed to know if sampling is possible).
    - The set of goal conditions, categorized by type (soil, rock, image).

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic calculates the sum of costs for each goal fact that is not yet satisfied:

    1.  **Parse Current State:** Extract the current location of each rover, which rovers/waypoints have collected soil/rock/image data, which stores are full, and which cameras are calibrated. Also, identify which goal facts are already true in the current state.

    2.  **Identify Unachieved Goals:** Filter the pre-computed goal set to include only those not present in the current state.

    3.  **Calculate Cost for Each Unachieved Goal:** For each unachieved goal fact (e.g., `(communicated_soil_data w)`):
        a.  Initialize the minimum cost for this goal to infinity.
        b.  Determine the required capabilities (e.g., soil analysis equipment for soil data).
        c.  Identify all rovers possessing the required capabilities (and a store for soil/rock).
        d.  For each capable rover (and suitable camera for image goals):
            i.  Get the rover's current location.
            ii. Calculate the cost to obtain the required data (sample or image) and communicate it. This involves a sequence of steps:
                - If data is not already collected:
                    - For soil/rock: Calculate cost to move to sample waypoint using BFS from current location, add cost for drop (if store full) and sample actions. Update rover's location for the next step to the reached sample waypoint.
                    - For image: If calibration is needed, calculate cost to move to a calibration waypoint using BFS from current location, add calibrate action. Update rover's location to the reached calibration waypoint. Then, calculate cost to move from this new location to an image waypoint using BFS, add take_image action. Update rover's location to the reached image waypoint.
                - Calculate cost to move from the rover's current location (after getting data, or initial location if data was held) to a communication waypoint using BFS, add communicate action.
            iii. Sum the costs for moves and actions.
            iv. Update the minimum cost for this goal if the current rover's (or rover/camera combo's) calculated cost is lower.
        e.  If the minimum cost is still infinity after checking all relevant rovers/cameras, this goal is considered unreachable from the current state and contributes 0 to the total heuristic.

    4.  **Sum Costs:** The total heuristic value is the sum of the minimum costs calculated for each unachieved goal fact.

    5.  **Return Total Cost:** Return the calculated total heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals  # The set of facts that must hold in goal states.
        static_facts = task.static  # Facts that are not affected by actions.
        initial_state = task.initial_state # Facts true in the initial state

        # --- Pre-compute static information ---

        # Lander location
        self.lander_location = None
        for fact in static_facts:
            if match(fact, "at_lander", "*", "*"):
                self.lander_location = get_parts(fact)[2]
                break # Assuming only one lander

        # Waypoint visibility graph (used for communication points)
        self.waypoint_visibility = collections.defaultdict(set)
        for fact in static_facts:
            if match(fact, "visible", "*", "*"):
                wp1, wp2 = get_parts(fact)[1:]
                self.waypoint_visibility[wp1].add(wp2)
                self.waypoint_visibility[wp2].add(wp1) # Visibility is symmetric

        # Communication waypoints (visible from lander location)
        self.comm_waypoint_set = self.waypoint_visibility.get(self.lander_location, set())

        # Rover traversability graphs
        self.rover_traversal_graph = collections.defaultdict(lambda: collections.defaultdict(set))
        for fact in static_facts:
            if match(fact, "can_traverse", "*", "*", "*"):
                rover, wp1, wp2 = get_parts(fact)[1:]
                self.rover_traversal_graph[rover][wp1].add(wp2)
                # Assuming can_traverse is explicitly listed for both directions if symmetric

        # Equipped rovers
        self.equipped_rovers = collections.defaultdict(set) # 'soil', 'rock', 'imaging'
        for fact in static_facts:
            if match(fact, "equipped_for_soil_analysis", "*"):
                self.equipped_rovers['soil'].add(get_parts(fact)[1])
            elif match(fact, "equipped_for_rock_analysis", "*"):
                self.equipped_rovers['rock'].add(get_parts(fact)[1])
            elif match(fact, "equipped_for_imaging", "*"):
                self.equipped_rovers['imaging'].add(get_parts(fact)[1])

        # Store to rover mapping
        self.store_to_rover = {}
        for fact in static_facts:
            if match(fact, "store_of", "*", "*"):
                store, rover = get_parts(fact)[1:]
                self.store_to_rover[store] = rover

        # Camera information
        self.camera_on_board = {} # camera -> rover
        self.camera_supports_mode = collections.defaultdict(set) # camera -> set[mode]
        self.camera_calibration_target = {} # camera -> objective
        for fact in static_facts:
            if match(fact, "on_board", "*", "*"):
                camera, rover = get_parts(fact)[1:]
                self.camera_on_board[camera] = rover
            elif match(fact, "supports", "*", "*"):
                camera, mode = get_parts(fact)[1:]
                self.camera_supports_mode[camera].add(mode)
            elif match(fact, "calibration_target", "*", "*"):
                camera, target = get_parts(fact)[1:]
                self.camera_calibration_target[camera] = target

        # Objective/Target visibility waypoints
        self.objective_visible_from = collections.defaultdict(set) # objective/target -> set[waypoint]
        for fact in static_facts:
             if match(fact, "visible_from", "*", "*"):
                 obj_or_target, waypoint = get_parts(fact)[1:]
                 self.objective_visible_from[obj_or_target].add(waypoint)

        # Initial sample locations (needed to know if sampling is possible at all)
        self.initial_soil_samples = {get_parts(fact)[1] for fact in initial_state if match(fact, "at_soil_sample", "*")}
        self.initial_rock_samples = {get_parts(fact)[1] for fact in initial_state if match(fact, "at_rock_sample", "*")}


        # Categorize goal facts
        self.goal_soil_waypoints = set()
        self.goal_rock_waypoints = set()
        self.goal_image_objectives_modes = set() # set of (objective, mode) tuples

        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "communicated_soil_data":
                self.goal_soil_waypoints.add(args[0])
            elif predicate == "communicated_rock_data":
                self.goal_rock_waypoints.add(args[0])
            elif predicate == "communicated_image_data":
                self.goal_image_objectives_modes.add(tuple(args)) # (objective, mode)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # --- Parse current state information ---
        current_rover_locations = {} # rover -> waypoint
        current_rover_soil_data = collections.defaultdict(set) # rover -> set[waypoint]
        current_rover_rock_data = collections.defaultdict(set) # rover -> set[waypoint]
        current_rover_image_data = collections.defaultdict(set) # rover -> set[(objective, mode)]
        current_stores_full = set() # set of stores
        current_cameras_calibrated = set() # set of (camera, rover) tuples

        achieved_soil_goals = set() # set of waypoints
        achieved_rock_goals = set() # set of waypoints
        achieved_image_goals = set() # set of (objective, mode) tuples

        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]
            args = parts[1:]

            if predicate == "at" and args[0].startswith("rover"):
                current_rover_locations[args[0]] = args[1]
            elif predicate == "have_soil_analysis":
                current_rover_soil_data[args[0]].add(args[1])
            elif predicate == "have_rock_analysis":
                current_rover_rock_data[args[0]].add(args[1])
            elif predicate == "have_image":
                 current_rover_image_data[args[0]].add(tuple(args[1:])) # (objective, mode)
            elif predicate == "full":
                current_stores_full.add(args[0])
            elif predicate == "calibrated":
                current_cameras_calibrated.add(tuple(args)) # (camera, rover)
            elif predicate == "communicated_soil_data":
                achieved_soil_goals.add(args[0])
            elif predicate == "communicated_rock_data":
                achieved_rock_goals.add(args[0])
            elif predicate == "communicated_image_data":
                achieved_image_goals.add(tuple(args)) # (objective, mode)

        total_cost = 0

        # --- Calculate cost for unachieved soil goals ---
        for soil_wp in self.goal_soil_waypoints - achieved_soil_goals:
            min_goal_cost = float('inf')

            # Check if sampling is even possible at this waypoint (was it in the initial state?)
            if soil_wp not in self.initial_soil_samples:
                 continue # Skip impossible goals

            # Find rovers equipped for soil analysis
            soil_rovers = self.equipped_rovers.get('soil', set())

            for rover in soil_rovers:
                rover_curr_wp = current_rover_locations.get(rover)
                if rover_curr_wp is None: continue # Rover location unknown

                # Check if rover has a store (required for sampling)
                rover_store = None
                for s, r in self.store_to_rover.items():
                    if r == rover:
                        rover_store = s
                        break
                if rover_store is None:
                    continue # Rover has no store, cannot sample

                cost = 0
                rover_has_data = soil_wp in current_rover_soil_data.get(rover, set())
                current_loc_for_calc = rover_curr_wp # Start location for the sequence of moves

                if not rover_has_data:
                    # Need to sample
                    dist_to_sample, reached_sample_wp = bfs_distance(self.rover_traversal_graph.get(rover, {}), current_loc_for_calc, {soil_wp})
                    if dist_to_sample == float('inf'): continue # Cannot reach sample location

                    cost += dist_to_sample # Move to sample location
                    current_loc_for_calc = reached_sample_wp # Rover is now at sample_wp

                    # Check if store is full - need to drop first
                    if rover_store in current_stores_full:
                         cost += 1 # Cost of drop action

                    cost += 1 # Cost of sample_soil action

                # Need to communicate
                dist_to_comm, reached_comm_wp = bfs_distance(self.rover_traversal_graph.get(rover, {}), current_loc_for_calc, self.comm_waypoint_set)
                if dist_to_comm == float('inf'): continue # Cannot reach communication location

                cost += dist_to_comm # Move to communication location
                cost += 1 # Cost of communicate_soil_data action

                min_goal_cost = min(min_goal_cost, cost)

            if min_goal_cost != float('inf'):
                 total_cost += min_goal_cost
            # else: goal is impossible for any equipped rover, contributes 0

        # --- Calculate cost for unachieved rock goals ---
        for rock_wp in self.goal_rock_waypoints - achieved_rock_goals:
            min_goal_cost = float('inf')

            # Check if sampling is even possible at this waypoint (was it in the initial state?)
            if rock_wp not in self.initial_rock_samples:
                 continue # Skip impossible goals

            # Find rovers equipped for rock analysis
            rock_rovers = self.equipped_rovers.get('rock', set())

            for rover in rock_rovers:
                rover_curr_wp = current_rover_locations.get(rover)
                if rover_curr_wp is None: continue # Rover location unknown

                # Check if rover has a store (required for sampling)
                rover_store = None
                for s, r in self.store_to_rover.items():
                    if r == rover:
                        rover_store = s
                        break
                if rover_store is None:
                    continue # Rover has no store, cannot sample

                cost = 0
                rover_has_data = rock_wp in current_rover_rock_data.get(rover, set())
                current_loc_for_calc = rover_curr_wp # Start location for the sequence of moves

                if not rover_has_data:
                    # Need to sample
                    dist_to_sample, reached_sample_wp = bfs_distance(self.rover_traversal_graph.get(rover, {}), current_loc_for_calc, {rock_wp})
                    if dist_to_sample == float('inf'): continue # Cannot reach sample location

                    cost += dist_to_sample # Move to sample location
                    current_loc_for_calc = reached_sample_wp # Rover is now at sample_wp

                    # Check if store is full - need to drop first
                    if rover_store in current_stores_full:
                         cost += 1 # Cost of drop action

                    cost += 1 # Cost of sample_rock action

                # Need to communicate
                dist_to_comm, reached_comm_wp = bfs_distance(self.rover_traversal_graph.get(rover, {}), current_loc_for_calc, self.comm_waypoint_set)
                if dist_to_comm == float('inf'): continue # Cannot reach communication location

                cost += dist_to_comm # Move to communication location
                cost += 1 # Cost of communicate_rock_data action

                min_goal_cost = min(min_goal_cost, cost)

            if min_goal_cost != float('inf'):
                 total_cost += min_goal_cost
            # else: goal is impossible, contributes 0

        # --- Calculate cost for unachieved image goals ---
        for obj, mode in self.goal_image_objectives_modes - achieved_image_goals:
            min_goal_cost = float('inf')

            # Find rovers equipped for imaging
            imaging_rovers = self.equipped_rovers.get('imaging', set())

            for rover in imaging_rovers:
                rover_curr_wp = current_rover_locations.get(rover)
                if rover_curr_wp is None: continue # Rover location unknown

                rover_has_data = (obj, mode) in current_rover_image_data.get(rover, set())

                if rover_has_data:
                    # Just need to communicate
                    dist_to_comm, reached_comm_wp = bfs_distance(self.rover_traversal_graph.get(rover, {}), rover_curr_wp, self.comm_waypoint_set)
                    if dist_to_comm == float('inf'): continue # Cannot reach communication location
                    cost = dist_to_comm + 1 # Move + Communicate
                    min_goal_cost = min(min_goal_cost, cost)
                else:
                    # Need to take image and then communicate
                    # Find cameras on this rover that support the required mode
                    suitable_cameras = {
                        cam for cam, r in self.camera_on_board.items() if r == rover and mode in self.camera_supports_mode.get(cam, set())
                    }

                    for camera in suitable_cameras:
                        cost = 0
                        current_loc_for_calc = rover_curr_wp # Start location for the sequence of moves

                        # Step 1: Calibrate (if needed)
                        camera_is_calibrated = (camera, rover) in current_cameras_calibrated
                        if not camera_is_calibrated:
                            cal_target = self.camera_calibration_target.get(camera)
                            if cal_target is None: continue # Camera cannot be calibrated

                            cal_wps = self.objective_visible_from.get(cal_target, set())
                            if not cal_wps: continue # No waypoint to calibrate from

                            dist_to_cal, reached_cal_wp = bfs_distance(self.rover_traversal_graph.get(rover, {}), current_loc_for_calc, cal_wps)
                            if dist_to_cal == float('inf'): continue # Cannot reach calibration location

                            cost += dist_to_cal + 1 # Move + Calibrate
                            current_loc_for_calc = reached_cal_wp # Rover is now at this specific cal_wp

                        # Step 2: Take image
                        # Note: If camera was already calibrated, current_loc_for_calc is still rover_curr_wp.
                        # If camera was calibrated here, current_loc_for_calc is reached_cal_wp.
                        img_wps = self.objective_visible_from.get(obj, set())
                        if not img_wps: continue # Cannot take image

                        dist_to_img, reached_img_wp = bfs_distance(self.rover_traversal_graph.get(rover, {}), current_loc_for_calc, img_wps)
                        if dist_to_img == float('inf'): continue # Cannot reach image location

                        cost += dist_to_img + 1 # Move + Take Image
                        current_loc_for_calc = reached_img_wp # Rover is now at this specific img_wp

                        # Step 3: Communicate
                        comm_wps = self.comm_waypoint_set
                        if not comm_wps: continue # Cannot communicate

                        dist_to_comm, reached_comm_wp = bfs_distance(self.rover_traversal_graph.get(rover, {}), current_loc_for_calc, comm_wps)
                        if dist_to_comm == float('inf'): continue # Cannot reach communication location

                        cost += dist_to_comm + 1 # Move + Communicate

                        min_goal_cost = min(min_goal_cost, cost)


            if min_goal_cost != float('inf'):
                 total_cost += min_goal_cost
            # else: goal is impossible, contributes 0

        return total_cost
