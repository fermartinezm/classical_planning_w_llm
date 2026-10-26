import collections
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import math # Use math.inf for infinity

def get_parts(fact):
    """Helper to split a PDDL fact string into predicate and arguments."""
    # Remove parentheses and split by space
    return fact[1:-1].split()

def match(fact, *args):
    """Helper to check if a fact matches a pattern using fnmatch."""
    parts = get_parts(fact)
    # Ensure we have enough parts to match against args
    if len(parts) < len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class roversHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Rovers domain.

    Summary:
    This heuristic estimates the cost to reach a goal state by summing the
    estimated costs for each individual goal predicate that is not yet satisfied.
    For each unsatisfied goal (communicating soil, rock, or image data), it
    calculates the minimum number of actions required for any capable rover
    to first acquire the necessary data (if not already possessed) and then
    navigate to a communication point visible from the lander and perform
    the communication action. The cost estimation for each step (navigation,
    sampling, imaging, calibrating, dropping, communicating) is simplified
    to 1 action, ignoring actual path lengths for navigation. Reachability
    between waypoints for each rover is precomputed to ensure feasibility.

    Assumptions:
    - Samples (soil/rock) are static and do not reappear.
    - Calibration targets and visibility from waypoints are static.
    - Rovers retain collected data and calibration status unless explicitly
      changed by actions (e.g., take_image uncalibrates).
    - A single rover is assumed to be responsible for both acquiring data
      and communicating it for a specific goal item (e.g., rover A samples
      soil at W and rover A communicates soil data from W). This simplifies
      task assignment.
    - The navigation cost between any two reachable waypoints is simplified
      to 1 action if movement is needed, ignoring the actual path length.
    - When multiple waypoints are suitable for a task (e.g., multiple
      waypoints visible from an objective for imaging), the heuristic
      arbitrarily picks one reachable suitable waypoint.
    - The heuristic does not account for resource contention (e.g., multiple
      rovers needing the same store or camera simultaneously).

    Heuristic Initialization:
    The constructor processes the static facts from the task definition to
    build efficient lookup structures:
    - lander_location: Stores the waypoint of the lander.
    - lander_visible_waypoints: Set of waypoints visible from the lander's location.
    - rover_equipment: Maps each rover to a set of its equipment types ('soil', 'rock', 'imaging').
    - store_owner: Maps each store to its owning rover.
    - soil_sample_locations: Set of waypoints where soil samples exist.
    - rock_sample_locations: Set of waypoints where rock samples exist.
    - camera_info: Maps each camera to a dictionary containing its 'rover' owner,
                   a set of supported 'modes', and its 'calibration_target'.
    - objective_visible_from: Maps each objective to a set of waypoints
                              from which it is visible.
    - waypoint_visibility_graph: Adjacency list representing visible waypoint connections.
    - rover_traversability_graphs: Maps each rover to an adjacency list of waypoints
                                   it can traverse between (based on 'can_traverse'
                                   and 'visible' static facts).
    - rover_reachability: Maps each rover and starting waypoint to a set of all
                          waypoints reachable by that rover from that starting waypoint.
                          This is precomputed using BFS.
    - Sets of all objects by type (rovers, waypoints, cameras, etc.) are also stored.

    Step-By-Step Thinking for Computing Heuristic:
    1.  Initialize `total_cost` to 0.
    2.  Parse the current state (`node.state`) to extract dynamic information:
        -   Current location of each rover (`rover_locations`).
        -   Status of each store ('empty' or 'full') (`store_status`).
        -   Data held by each rover (soil samples by waypoint, rock samples by waypoint,
            images by (objective, mode)) (`rover_soil_data`, `rover_rock_data`, `rover_image_data`).
        -   Calibration status of each camera on each rover (`camera_calibration`).
        -   Set of already communicated data goals (`communicated_soil`, `communicated_rock`, `communicated_image`).
    3.  Iterate through each goal predicate in `self.goals`.
    4.  If the current goal predicate is already true in the state (found in the
        communicated sets), add 0 cost for this goal and continue to the next goal.
    5.  If the goal predicate is not true, determine its type
        (`communicated_soil_data`, `communicated_rock_data`, or `communicated_image_data`)
        and its specific parameters (waypoint `W` or objective `O` and mode `M`).
    6.  Initialize `min_goal_cost` for this unsatisfied goal to `math.inf`.
    7.  Iterate through each known rover `R`.
    8.  For the current rover `R`, calculate the estimated cost to achieve this
        specific goal, assuming `R` performs all necessary steps (acquire data
        if needed, navigate, communicate).
        -   Check if `R` is equipped for the task required by the goal type. If not,
            this rover cannot achieve this goal (under the single-rover assumption),
            so skip to the next rover.
        -   Get `R`'s current location (`temp_current_wp`) from the state. If the
            rover's location is unknown (should not happen in valid states), skip it.
        -   Initialize `rover_cost` to 0.
        -   **Data Acquisition Cost:**
            -   Check if the required data (soil sample `W`, rock sample `W`, or
                image `O`+`M`) is already held by rover `R`.
            -   If the data is NOT held by `R`:
                -   If it's a soil/rock goal (`communicated_X_data W`):
                    -   Check if a sample exists at `W` (from static info). If not,
                        this goal is impossible via sampling, skip this rover.
                    -   The target acquisition waypoint is `W`.
                    -   Check if `W` is reachable by `R` from `temp_current_wp`
                        using the precomputed reachability. If not, skip this rover.
                    -   If `temp_current_wp` is not `W`, add 1 to `rover_cost`
                        (for navigation). Update `temp_current_wp` to `W`.
                    -   Check `R`'s store status. Find `R`'s store (from static info).
                        If the store is 'full' in the current state, add 1 to
                        `rover_cost` (for the 'drop' action).
                    -   Add 1 to `rover_cost` (for the 'sample_soil' or 'sample_rock' action).
                -   If it's an image goal (`communicated_image_data O M`):
                    -   Find a camera `I` on `R` that supports mode `M` (from static info).
                        If none, skip this rover.
                    -   Find *a* waypoint `img_wp` from which `O` is visible (from static info)
                        that `R` can reach from `temp_current_wp`. Use `find_reachable_waypoint`.
                        If none, skip this rover.
                    -   Find the calibration target `T` for camera `I` and *a* waypoint
                        `cal_wp` from which `T` is visible (from static info) that `R`
                        can reach from `temp_current_wp`. Use `find_reachable_waypoint`.
                        If none, skip this rover.
                    -   If camera `I` is NOT calibrated on `R` in the current state:
                        -   If `temp_current_wp` is not `cal_wp`, add 1 to `rover_cost`
                            (for navigation). Update `temp_current_wp` to `cal_wp`.
                        -   Add 1 to `rover_cost` (for the 'calibrate' action).
                    -   If `temp_current_wp` is not `img_wp`, add 1 to `rover_cost`
                        (for navigation). Update `temp_current_wp` to `img_wp`.
                    -   Add 1 to `rover_cost` (for the 'take_image' action).
        -   **Communication Cost:**
            -   Find *a* waypoint `comm_wp` that is visible from the lander location
                (from static info) and that `R` can reach from the current
                `temp_current_wp` (which is the location after potential data acquisition).
                Use `find_reachable_waypoint`. If none, skip this rover.
            -   If `temp_current_wp` is not `comm_wp`, add 1 to `rover_cost`
                (for navigation). Update `temp_current_wp` to `comm_wp`.
            -   Add 1 to `rover_cost` (for the 'communicate_X_data' action).
        -   Update `min_goal_cost` = min(`min_goal_cost`, `rover_cost`).
    9.  After iterating through all rovers, if `min_goal_cost` is still `math.inf`,
        it means the goal is estimated as unreachable by any equipped rover under
        these simplified assumptions. In this case, the entire state is likely
        unsolvable, so return `math.inf` immediately.
    10. Otherwise, add `min_goal_cost` to `total_cost`.
    11. After iterating through all goals, return `total_cost`.

    """
    def __init__(self, task):
        self.goals = task.goals
        static_facts = task.static

        # --- Parse Static Facts ---
        self.lander_location = None
        self.lander_visible_waypoints = set()
        self.rover_equipment = collections.defaultdict(set) # rover -> {soil, rock, imaging}
        self.store_owner = {} # store -> rover
        self.soil_sample_locations = set() # {waypoint}
        self.rock_sample_locations = set() # {waypoint}
        self.camera_info = {} # camera -> {rover, modes, calibration_target}
        self.objective_visible_from = collections.defaultdict(set) # objective -> {waypoints}
        self.waypoint_visibility_graph = collections.defaultdict(list) # wp -> [visible_wps]
        self.rover_traversability_graphs = collections.defaultdict(lambda: collections.defaultdict(list)) # rover -> wp -> [traversable_wps]

        self.all_rovers = set()
        self.all_waypoints = set()
        self.all_cameras = set()
        self.all_objectives = set()
        self.all_modes = set()
        self.all_stores = set()
        self.all_landers = set()

        for fact in static_facts:
            parts = get_parts(fact)
            pred = parts[0]

            if pred == "at_lander":
                self.lander_location = parts[2]
                self.all_landers.add(parts[1])
                self.all_waypoints.add(parts[2])
            elif pred == "at": # Initial rover locations (static in init, dynamic in state)
                 if parts[1].startswith("rover"): # Assuming rover names start with rover
                     self.all_rovers.add(parts[1])
                     self.all_waypoints.add(parts[2])
            elif pred == "equipped_for_soil_analysis":
                self.rover_equipment[parts[1]].add('soil')
                self.all_rovers.add(parts[1])
            elif pred == "equipped_for_rock_analysis":
                self.rover_equipment[parts[1]].add('rock')
                self.all_rovers.add(parts[1])
            elif pred == "equipped_for_imaging":
                self.rover_equipment[parts[1]].add('imaging')
                self.all_rovers.add(parts[1])
            elif pred == "store_of":
                self.store_owner[parts[1]] = parts[2]
                self.all_stores.add(parts[1])
                self.all_rovers.add(parts[2])
            elif pred == "at_soil_sample":
                self.soil_sample_locations.add(parts[1])
                self.all_waypoints.add(parts[1])
            elif pred == "at_rock_sample":
                self.rock_sample_locations.add(parts[1])
                self.all_waypoints.add(parts[1])
            elif pred == "visible":
                wp1, wp2 = parts[1], parts[2]
                self.waypoint_visibility_graph[wp1].append(wp2)
                self.all_waypoints.add(wp1)
                self.all_waypoints.add(wp2)
            elif pred == "can_traverse":
                rover, wp1, wp2 = parts[1], parts[2], parts[3]
                # Add to rover-specific graph only if visible
                if f"(visible {wp1} {wp2})" in static_facts:
                     self.rover_traversability_graphs[rover][wp1].append(wp2)
                self.all_rovers.add(rover)
                self.all_waypoints.add(wp1)
                self.all_waypoints.add(wp2)
            elif pred == "on_board":
                camera, rover = parts[1], parts[2]
                if camera not in self.camera_info: self.camera_info[camera] = {}
                self.camera_info[camera]['rover'] = rover
                self.all_cameras.add(camera)
                self.all_rovers.add(rover)
            elif pred == "supports":
                camera, mode = parts[1], parts[2]
                if camera not in self.camera_info: self.camera_info[camera] = {}
                if 'modes' not in self.camera_info[camera]: self.camera_info[camera]['modes'] = set()
                self.camera_info[camera]['modes'].add(mode)
                self.all_cameras.add(camera)
                self.all_modes.add(mode)
            elif pred == "calibration_target":
                camera, objective = parts[1], parts[2]
                if camera not in self.camera_info: self.camera_info[camera] = {}
                self.camera_info[camera]['calibration_target'] = objective
                self.all_cameras.add(camera)
                self.all_objectives.add(objective)
            elif pred == "visible_from":
                objective, waypoint = parts[1], parts[2]
                self.objective_visible_from[objective].add(waypoint)
                self.all_objectives.add(objective)
                self.all_waypoints.add(waypoint)
            # Add other types if needed from object declarations if available,
            # but parsing predicates is usually sufficient to identify relevant objects.
            # Example: (rover rover1) might not be in static_facts, but (at rover1 ...) is.

        # Determine lander visible waypoints
        if self.lander_location:
            # Communication requires (visible ?x ?y) where ?x is rover location and ?y is lander location.
            # Since (visible A B) implies (visible B A), we need waypoints ?x such that (visible lander_location ?x) is true.
            self.lander_visible_waypoints = set(self.waypoint_visibility_graph.get(self.lander_location, []))
            # Also include the lander location itself if it's a waypoint a rover can be at
            if self.lander_location in self.all_waypoints:
                 self.lander_visible_waypoints.add(self.lander_location)


        # --- Precompute Rover Reachability using BFS ---
        self.rover_reachability = {} # rover -> start_wp -> {reachable_wps}
        for rover in self.all_rovers:
            graph = self.rover_traversability_graphs.get(rover, collections.defaultdict(list))
            rover_wp_reachability = {}
            # Need to consider all waypoints, even those not in the graph keys/values initially
            for start_wp in self.all_waypoints:
                reachable_set = set()
                # If the start_wp is not in the graph at all, it can only reach itself
                if start_wp not in graph and all(start_wp != neighbor for neighbors in graph.values() for neighbor in neighbors):
                     reachable_set.add(start_wp)
                else:
                    queue = collections.deque([start_wp])
                    visited = {start_wp}
                    while queue:
                        current = queue.popleft()
                        reachable_set.add(current)
                        for neighbor in graph.get(current, []):
                            if neighbor not in visited:
                                visited.add(neighbor)
                                queue.append(neighbor)
                rover_wp_reachability[start_wp] = reachable_set
            self.rover_reachability[rover] = rover_wp_reachability

    def is_reachable(self, rover, from_wp, to_wp):
        """Checks if a waypoint is reachable by a rover from another waypoint."""
        if rover not in self.rover_reachability or from_wp not in self.rover_reachability[rover]:
            return False # Rover or start_wp not in precomputed graph
        return to_wp in self.rover_reachability[rover].get(from_wp, set()) # Use .get for safety

    def find_reachable_waypoint(self, rover, from_wp, target_wps_set):
        """Finds *a* waypoint in target_wps_set reachable by rover from from_wp."""
        if not target_wps_set: return None
        if rover not in self.rover_reachability or from_wp not in self.rover_reachability[rover]:
             return None # Rover or start_wp not in precomputed graph

        reachable_from_from_wp = self.rover_reachability[rover].get(from_wp, set())
        if not reachable_from_from_wp:
             # If the start_wp itself is not in the reachability map (e.g., isolated),
             # check if any target is the start_wp itself.
             if from_wp in target_wps_set:
                  return from_wp
             return None

        # Find intersection
        possible_targets = list(target_wps_set.intersection(reachable_from_from_wp))
        if possible_targets:
            return possible_targets[0] # Return the first one found
        return None # No reachable target waypoint in the set


    def __call__(self, node):
        state = node.state

        # --- Parse Current State ---
        rover_locations = {} # rover -> waypoint
        store_status = {} # store -> 'empty' or 'full'
        rover_soil_data = collections.defaultdict(set) # rover -> {waypoints}
        rover_rock_data = collections.defaultdict(set) # rover -> {waypoints}
        rover_image_data = collections.defaultdict(set) # rover -> {(objective, mode)}
        camera_calibration = collections.defaultdict(set) # rover -> {cameras}
        communicated_soil = set() # {waypoints}
        communicated_rock = set() # {waypoints}
        communicated_image = set() # {(objective, mode)}

        for fact in state:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "at":
                if parts[1].startswith("rover"): # Assuming rover names start with rover
                    rover_locations[parts[1]] = parts[2]
            elif pred == "empty":
                store_status[parts[1]] = 'empty'
            elif pred == "full":
                store_status[parts[1]] = 'full'
            elif pred == "have_soil_analysis":
                rover_soil_data[parts[1]].add(parts[2])
            elif pred == "have_rock_analysis":
                rover_rock_data[parts[1]].add(parts[2])
            elif pred == "have_image":
                rover_image_data[parts[1]].add((parts[2], parts[3]))
            elif pred == "calibrated":
                camera_calibration[parts[2]].add(parts[1]) # calibrated ?c ?r -> rover -> camera
            elif pred == "communicated_soil_data":
                communicated_soil.add(parts[1])
            elif pred == "communicated_rock_data":
                communicated_rock.add(parts[1])
            elif pred == "communicated_image_data":
                communicated_image.add((parts[1], parts[2]))

        total_cost = 0

        # --- Estimate Cost for Unsatisfied Goals ---
        for goal in self.goals:
            parts = get_parts(goal)
            pred = parts[0]

            if pred == "communicated_soil_data":
                w = parts[1]
                if w in communicated_soil:
                    continue # Goal already satisfied

                min_goal_cost = math.inf

                # Find the minimum cost for any capable rover
                for r in self.all_rovers:
                    # Check if rover is equipped for soil analysis
                    if 'soil' not in self.rover_equipment.get(r, set()):
                        continue # Rover not equipped

                    rover_cost = 0
                    temp_current_wp = rover_locations.get(r) # Get current location from state

                    if temp_current_wp is None:
                         # Rover location unknown? Should not happen in valid states.
                         continue # Skip this rover

                    # Data Acquisition Cost
                    if w not in rover_soil_data.get(r, set()):
                        # Need to sample soil at w
                        target_acquisition_wp = w

                        # Check if sample exists at w (static fact)
                        if target_acquisition_wp not in self.soil_sample_locations:
                             continue # Sample doesn't exist, this path is blocked

                        # Check if rover can reach the sample location
                        if not self.is_reachable(r, temp_current_wp, target_acquisition_wp):
                            continue # Rover cannot reach sample location

                        if temp_current_wp != target_acquisition_wp:
                            rover_cost += 1 # navigate
                        temp_current_wp = target_acquisition_wp

                        # Check store status
                        store = None
                        for s, owner in self.store_owner.items():
                            if owner == r:
                                store = s
                                break
                        # Check if store exists and is full
                        if store and store_status.get(store) == 'full':
                            rover_cost += 1 # drop

                        rover_cost += 1 # sample_soil action

                    # Communication Cost
                    # Find a reachable waypoint visible from the lander
                    target_communication_wp = self.find_reachable_waypoint(r, temp_current_wp, self.lander_visible_waypoints)
                    if target_communication_wp is None:
                         continue # Rover cannot reach any communication location

                    if temp_current_wp != target_communication_wp:
                        rover_cost += 1 # navigate
                    temp_current_wp = target_communication_wp

                    rover_cost += 1 # communicate_soil_data action

                    min_goal_cost = min(min_goal_cost, rover_cost)

                if min_goal_cost == math.inf:
                    # If after checking all rovers, the goal is still infinitely costly,
                    # the state is likely unsolvable.
                    return math.inf
                else:
                    total_cost += min_goal_cost

            elif pred == "communicated_rock_data":
                w = parts[1]
                if w in communicated_rock:
                    continue # Goal already satisfied

                min_goal_cost = math.inf

                # Find the minimum cost for any capable rover
                for r in self.all_rovers:
                    # Check if rover is equipped for rock analysis
                    if 'rock' not in self.rover_equipment.get(r, set()):
                        continue # Rover not equipped

                    rover_cost = 0
                    temp_current_wp = rover_locations.get(r)

                    if temp_current_wp is None: continue

                    # Data Acquisition Cost
                    if w not in rover_rock_data.get(r, set()):
                        # Need to sample rock at w
                        target_acquisition_wp = w

                        # Check if sample exists at w (static fact)
                        if target_acquisition_wp not in self.rock_sample_locations:
                             continue # Sample doesn't exist, this path is blocked

                        # Check if rover can reach the sample location
                        if not self.is_reachable(r, temp_current_wp, target_acquisition_wp):
                            continue # Rover cannot reach sample location

                        if temp_current_wp != target_acquisition_wp:
                            rover_cost += 1 # navigate
                        temp_current_wp = target_acquisition_wp

                        # Check store status
                        store = None
                        for s, owner in self.store_owner.items():
                            if owner == r:
                                store = s
                                break
                        # Check if store exists and is full
                        if store and store_status.get(store) == 'full':
                            rover_cost += 1 # drop

                        rover_cost += 1 # sample_rock action

                    # Communication Cost
                    # Find a reachable waypoint visible from the lander
                    target_communication_wp = self.find_reachable_waypoint(r, temp_current_wp, self.lander_visible_waypoints)
                    if target_communication_wp is None:
                         continue # Rover cannot reach any communication location

                    if temp_current_wp != target_communication_wp:
                        rover_cost += 1 # navigate
                    temp_current_wp = target_communication_wp

                    rover_cost += 1 # communicate_rock_data action

                    min_goal_cost = min(min_goal_cost, rover_cost)

                if min_goal_cost == math.inf:
                    return math.inf
                else:
                    total_cost += min_goal_cost

            elif pred == "communicated_image_data":
                o, m = parts[1], parts[2]
                if (o, m) in communicated_image:
                    continue # Goal already satisfied

                min_goal_cost = math.inf

                # Find the minimum cost for any capable rover
                for r in self.all_rovers:
                    # Check if rover is equipped for imaging
                    if 'imaging' not in self.rover_equipment.get(r, set()):
                        continue # Rover not equipped

                    # Find camera on this rover supporting this mode
                    camera_i = None
                    for cam, info in self.camera_info.items():
                        if info.get('rover') == r and m in info.get('modes', set()):
                            camera_i = cam
                            break
                    if camera_i is None:
                        continue # No suitable camera on this rover

                    # Get calibration target and potential calibration waypoints
                    cal_target = self.camera_info[camera_i].get('calibration_target')
                    if cal_target is None:
                         continue # Camera has no calibration target?

                    cal_wps = self.objective_visible_from.get(cal_target, set())
                    if not cal_wps:
                         continue # Calibration target not visible from anywhere?

                    # Get potential image waypoints
                    img_wps = self.objective_visible_from.get(o, set())
                    if not img_wps:
                         continue # Objective not visible from anywhere?

                    rover_cost = 0
                    temp_current_wp = rover_locations.get(r)

                    if temp_current_wp is None: continue

                    # Data Acquisition Cost
                    if (o, m) not in rover_image_data.get(r, set()):
                        # Need to take image (o, m)

                        # Calibration step (only if camera is not calibrated)
                        if camera_i not in camera_calibration.get(r, set()):
                            # Need to calibrate
                            target_calibration_wp = self.find_reachable_waypoint(r, temp_current_wp, cal_wps)
                            if target_calibration_wp is None:
                                 continue # Cannot reach any calibration waypoint

                            if temp_current_wp != target_calibration_wp:
                                rover_cost += 1 # navigate
                            temp_current_wp = target_calibration_wp
                            rover_cost += 1 # calibrate action

                        # Take image step
                        target_image_wp = self.find_reachable_waypoint(r, temp_current_wp, img_wps)
                        if target_image_wp is None:
                             continue # Cannot reach any image waypoint

                        if temp_current_wp != target_image_wp:
                            rover_cost += 1 # navigate
                        temp_current_wp = target_image_wp
                        rover_cost += 1 # take_image action


                    # Communication Cost
                    # Find a reachable waypoint visible from the lander
                    target_communication_wp = self.find_reachable_waypoint(r, temp_current_wp, self.lander_visible_waypoints)
                    if target_communication_wp is None:
                         continue # Rover cannot reach any communication location

                    if temp_current_wp != target_communication_wp:
                        rover_cost += 1 # navigate
                    temp_current_wp = target_communication_wp

                    rover_cost += 1 # communicate_image_data action

                    min_goal_cost = min(min_goal_cost, rover_cost)

                if min_goal_cost == math.inf:
                    return math.inf
                else:
                    total_cost += min_goal_cost

        # If we reached here, all goals were either satisfied or had a finite estimated cost.
        # The total_cost will be 0 only if all goals were initially satisfied.
        return total_cost

