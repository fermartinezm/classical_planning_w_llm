from fnmatch import fnmatch
from collections import defaultdict, deque
import math

# Assuming the Heuristic base class is available in the environment
# from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty string or non-string input gracefully
    if not isinstance(fact, str) or len(fact) < 2:
        return []
    # Remove outer parentheses and split by whitespace
    return fact[1:-1].split()

# Helper function for BFS
def bfs(graph, start_node, all_nodes):
    """Computes shortest path distances from start_node to all other nodes in the graph."""
    distances = {node: math.inf for node in all_nodes}
    if start_node not in all_nodes:
         # Start node is not a valid waypoint in this graph
         return distances # All distances remain infinity

    distances[start_node] = 0
    queue = deque([start_node])

    while queue:
        current_node = queue.popleft()

        # If current_node is not in graph, it has no outgoing edges, just stop exploring from here
        if current_node in graph:
            for neighbor in graph[current_node]:
                if distances[neighbor] == math.inf:
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)
    return distances

# Define the Heuristic base class if it's not provided in the execution environment
# This is just a placeholder if running the heuristic code in isolation.
# In the actual planning framework, the real Heuristic class will be used.
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    # Define a dummy class if the base class is not found
    class Heuristic:
        def __init__(self, task):
            pass
        def __call__(self, node):
            raise NotImplementedError("Heuristic must implement __call__")


class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve all
    uncommunicated goal facts. It sums the estimated cost for each uncommunicated
    goal independently. The cost for a single goal is estimated by summing the
    actions needed (sample/image, calibrate, communicate) plus the minimum
    navigation cost for a capable rover to visit the necessary locations
    (sample/image waypoint, calibration waypoint, communication waypoint)
    in sequence. Navigation costs are precomputed shortest path distances.

    # Assumptions
    - Each uncommunicated goal is pursued independently.
    - The cost for a goal is the minimum cost over all capable rovers.
    - For sampling/imaging and then communicating, the rover navigates
      sequentially: current_location -> sample/image_location -> communication_location.
      For imaging that requires calibration, the sequence is:
      current_location -> calibration_location -> image_location -> communication_location.
    - Stores are tied to rovers and are either empty or full. Dropping makes a full store empty.
    - Cameras are tied to rovers and have calibration targets and supported modes.
    - Navigation costs are based on directed `can_traverse` and `visible` links.

    # Heuristic Initialization
    - Extracts static information: lander location, rover capabilities, store ownership,
      camera properties (on-board, supported modes, calibration targets), waypoint
      visibility for navigation, objective visibility for imaging, target visibility
      for calibration.
    - Builds a navigation graph for each rover based on `can_traverse` and `visible`
      predicates.
    - Precomputes all-pairs shortest paths between waypoints for each rover using BFS.
    - Identifies communication waypoints (visible from the lander).
    - Stores goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    1.  Identify all goal facts that are not currently true in the state.
    2.  For each uncommunicated goal fact (e.g., `(communicated_soil_data ?w)`):
        a.  Determine if the required data/image already exists (`have_soil_analysis`, `have_rock_analysis`, `have_image`).
        b.  Initialize the minimum cost for this goal to infinity.
        c.  Iterate through all rovers capable of achieving this goal (equipped for soil/rock/imaging, has the right camera).
        d.  For the current rover:
            i.  Get its current location. If the rover's location is unknown (not in state), skip this rover.
            ii. Calculate the cost to obtain the data/image if needed:
                - If sampling (soil/rock) is needed for waypoint `?w`: Add 1 (sample action). Add navigation cost from current location to `?w`. If the rover's store is full, add 1 (drop action). The rover is assumed to end up at `?w` for the next step.
                - If imaging is needed for objective `?o` and mode `?m` using camera `?i`: Add 1 (take_image action). Determine if calibration is needed for camera `?i` on this rover. If needed, add 1 (calibrate action). Find suitable image waypoints `?p` (where `?o` is visible from `?p`) and calibration waypoints `?w` (where the target for `?i` is visible from `?w`). Calculate the minimum navigation cost sequence: current_location -> (calibration_waypoint if needed) -> image_waypoint. Add this navigation cost. The rover is assumed to end up at the chosen image waypoint for the next step.
            iii. Calculate the cost to communicate the data/image: Add 1 (communicate action). Add minimum navigation cost from the rover's current assumed location (after sampling/imaging) to any waypoint visible from the lander.
            iv. Sum the costs (sample/image + calibrate + navigation + communicate).
            v. Update the minimum cost for this goal if the current rover's cost is lower.
        e.  If the minimum cost for the goal is still infinity (unreachable by any capable rover), the overall heuristic is infinity.
        f.  Add the minimum cost for this goal to the total heuristic cost.
    3.  Return the total heuristic cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # Data structures for static information
        self.lander_location = None
        self.soil_rovers = set()
        self.rock_rovers = set()
        self.imaging_rovers = set()
        self.store_owner = {} # store -> rover
        self.camera_on_rover = {} # camera -> rover
        self.camera_modes = defaultdict(set) # camera -> set of modes
        self.camera_target = {} # camera -> target
        self.objective_waypoints = defaultdict(set) # objective -> set of waypoints
        self.target_waypoints = defaultdict(set) # target -> set of waypoints
        self.rover_navigation_graph = defaultdict(lambda: defaultdict(set)) # rover -> waypoint -> set of reachable waypoints
        self.waypoints = set()
        self.comm_waypoints = set() # waypoints visible from lander

        # Collect all objects by type (helps in parsing facts)
        rovers_in_domain = set() # All rovers mentioned in static facts

        # First pass to collect all waypoints and rovers and build basic structures
        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]

            if predicate == "at_lander":
                self.lander_location = parts[2]
                self.waypoints.add(parts[2])
            elif predicate == "equipped_for_soil_analysis":
                self.soil_rovers.add(parts[1])
                rovers_in_domain.add(parts[1])
            elif predicate == "equipped_for_rock_analysis":
                self.rock_rovers.add(parts[1])
                rovers_in_domain.add(parts[1])
            elif predicate == "equipped_for_imaging":
                self.imaging_rovers.add(parts[1])
                rovers_in_domain.add(parts[1])
            elif predicate == "store_of":
                self.store_owner[parts[1]] = parts[2]
                rovers_in_domain.add(parts[2])
            elif predicate == "on_board":
                self.camera_on_rover[parts[1]] = parts[2]
                rovers_in_domain.add(parts[2])
            elif predicate == "supports":
                self.camera_modes[parts[1]].add(parts[2])
            elif predicate == "calibration_target":
                self.camera_target[parts[1]] = parts[2]
            elif predicate == "visible_from":
                obj_or_target, waypoint = parts[1], parts[2]
                self.objective_waypoints[obj_or_target].add(waypoint)
                self.waypoints.add(waypoint)
            elif predicate == "visible":
                 w1, w2 = parts[1], parts[2]
                 self.waypoints.add(w1)
                 self.waypoints.add(w2)
                 # Process visible later with can_traverse
                 pass
            elif predicate == "can_traverse":
                 rover, w1, w2 = parts[1], parts[2], parts[3]
                 # We need to check (visible w1 w2) here.
                 # Store potential edges first, then filter based on visible.
                 # Or, iterate visible facts to build a visible graph first.
                 # Let's rebuild the graph logic.

        # Build visible graph
        visible_graph = defaultdict(set)
        for fact in static_facts:
             parts = get_parts(fact)
             if parts and parts[0] == "visible":
                  visible_graph[parts[1]].add(parts[2])

        # Build rover navigation graphs based on can_traverse and visible
        for fact in static_facts:
             parts = get_parts(fact)
             if parts and parts[0] == "can_traverse":
                  rover, w1, w2 = parts[1], parts[2], parts[3]
                  if w2 in visible_graph.get(w1, set()): # Check if visible link exists
                       self.rover_navigation_graph[rover][w1].add(w2)
                  rovers_in_domain.add(rover) # Ensure rover is in the set

        # Precompute shortest paths for each rover
        self.dist = defaultdict(dict) # rover -> start_wp -> end_wp -> distance
        for rover in rovers_in_domain:
             if rover in self.rover_navigation_graph: # Only compute for rovers that can move
                for start_wp in self.waypoints:
                    self.dist[rover][start_wp] = bfs(self.rover_navigation_graph[rover], start_wp, self.waypoints)

        # Identify communication waypoints (visible from lander)
        if self.lander_location:
             # A waypoint is a communication waypoint if the lander location is visible from it.
             # This corresponds to the precondition (visible ?x ?y) where ?x is rover loc, ?y is lander loc.
             for fact in static_facts:
                 parts = get_parts(fact)
                 if parts and parts[0] == "visible" and parts[2] == self.lander_location:
                     self.comm_waypoints.add(parts[1])
             # Also check if lander location is visible from itself (usually true)
             if f"(visible {self.lander_location} {self.lander_location})" in static_facts:
                  self.comm_waypoints.add(self.lander_location)

        # Map rover to store for quick lookup
        self.rover_store = {v: k for k, v in self.store_owner.items()}

        # Map target to camera for quick lookup (needed for image goals)
        self.target_camera = {v: k for k, v in self.camera_target.items()}

        # Populate target_waypoints based on calibration_target and objective_waypoints
        # A waypoint is a target waypoint for target T if (visible_from T W) is true.
        # We already collected these in objective_waypoints, just need to filter by targets.
        for target in self.target_camera.keys(): # Iterate through known targets
             if target in self.objective_waypoints:
                  self.target_waypoints[target] = self.objective_waypoints[target]


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Check if goal is already reached
        if self.goals <= state:
            return 0

        # Extract relevant dynamic information from the current state
        rover_locs = {} # rover -> waypoint
        store_full = {} # store -> bool
        have_soil = set() # (rover, waypoint)
        have_rock = set() # (rover, waypoint)
        have_image = set() # (rover, objective, mode)
        calibrated = set() # (camera, rover)

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]
            if predicate == "at":
                # Check if the object is a rover based on capabilities or store ownership
                obj = parts[1]
                # Assuming any object 'at' a waypoint that is not the lander is a rover.
                # Or, check if obj is in rovers_in_domain set from __init__
                # Let's use the sets of capable rovers and store owners as proxies for 'is_rover'
                # This might miss rovers that have no capabilities or stores but exist.
                # A more robust way would be to parse object types from the problem file,
                # but we only have static facts and state facts here.
                # Let's assume rovers are the only things (besides lander) that are 'at' waypoints.
                # The lander location is handled separately.
                if obj != self.lander_location: # Simple check: if it's not the lander location object name itself
                     rover_locs[obj] = parts[2]
            elif predicate == "full":
                store_full[parts[1]] = True
            elif predicate == "empty":
                 store_full[parts[1]] = False # Assuming stores are either full or empty
            elif predicate == "have_soil_analysis":
                have_soil.add((parts[1], parts[2]))
            elif predicate == "have_rock_analysis":
                have_rock.add((parts[1], parts[2]))
            elif predicate == "have_image":
                have_image.add((parts[1], parts[2], parts[3]))
            elif predicate == "calibrated":
                calibrated.add((parts[1], parts[2]))

        total_cost = 0
        uncommunicated_goals = self.goals - state

        # Handle unreachable communication points early
        if not self.comm_waypoints:
             # If lander is not visible from any waypoint, communication is impossible
             # unless the goal is already met. Since we are here, goal is not met.
             return math.inf

        for goal in uncommunicated_goals:
            parts = get_parts(goal)
            if not parts: continue

            predicate = parts[0]

            if predicate == "communicated_soil_data":
                waypoint = parts[1]
                sample_needed = not any((r, waypoint) in have_soil for r in self.soil_rovers)
                comm_needed = True # Always needed if goal is uncommunicated

                min_goal_cost = math.inf

                for r in self.soil_rovers:
                    current_loc = rover_locs.get(r)
                    # If rover location is unknown or rover cannot move (no nav graph entry)
                    if current_loc is None or r not in self.dist or current_loc not in self.dist[r]:
                         continue

                    cost = 0
                    current_path_end = current_loc # Rover starts path from its current location

                    # Cost to sample if needed
                    if sample_needed:
                        cost += 1 # sample action
                        nav_cost_to_sample = self.dist[r][current_path_end].get(waypoint, math.inf)
                        if nav_cost_to_sample == math.inf: continue # Cannot reach sample location
                        cost += nav_cost_to_sample
                        current_path_end = waypoint # Rover is now assumed to be at sample location

                        store = self.rover_store.get(r)
                        # Check if rover has a store and if it's full
                        if store is not None and store_full.get(store, False):
                             cost += 1 # drop action

                    # Cost to communicate
                    if comm_needed:
                        cost += 1 # communicate action
                        min_nav_cost_to_comm = math.inf
                        # comm_waypoints check is done before the loop

                        for comm_wp in self.comm_waypoints:
                            nav_cost = self.dist[r][current_path_end].get(comm_wp, math.inf)
                            if nav_cost != math.inf:
                                min_nav_cost_to_comm = min(min_nav_cost_to_comm, nav_cost)

                        if min_nav_cost_to_comm == math.inf: continue # Cannot reach any comm location from sample location
                        cost += min_nav_cost_to_comm

                    min_goal_cost = min(min_goal_cost, cost)

                if min_goal_cost == math.inf:
                     # This goal is unreachable by any equipped rover
                     return math.inf
                total_cost += min_goal_cost

            elif predicate == "communicated_rock_data":
                waypoint = parts[1]
                sample_needed = not any((r, waypoint) in have_rock for r in self.rock_rovers)
                comm_needed = True # Always needed if goal is uncommunicated

                min_goal_cost = math.inf

                for r in self.rock_rovers:
                    current_loc = rover_locs.get(r)
                    if current_loc is None or r not in self.dist or current_loc not in self.dist[r]:
                         continue

                    cost = 0
                    current_path_end = current_loc # Rover starts path from its current location

                    # Cost to sample if needed
                    if sample_needed:
                        cost += 1 # sample action
                        nav_cost_to_sample = self.dist[r][current_path_end].get(waypoint, math.inf)
                        if nav_cost_to_sample == math.inf: continue # Cannot reach sample location
                        cost += nav_cost_to_sample
                        current_path_end = waypoint # Rover is now assumed to be at sample location

                        store = self.rover_store.get(r)
                        # Check if rover has a store and if it's full
                        if store is not None and store_full.get(store, False):
                             cost += 1 # drop action

                    # Cost to communicate
                    if comm_needed:
                        cost += 1 # communicate action
                        min_nav_cost_to_comm = math.inf
                        # comm_waypoints check is done before the loop

                        for comm_wp in self.comm_waypoints:
                            nav_cost = self.dist[r][current_path_end].get(comm_wp, math.inf)
                            if nav_cost != math.inf:
                                min_nav_cost_to_comm = min(min_nav_cost_to_comm, nav_cost)

                        if min_nav_cost_to_comm == math.inf: continue # Cannot reach any comm location from sample location
                        cost += min_nav_cost_to_comm

                    min_goal_cost = min(min_goal_cost, cost)

                if min_goal_cost == math.inf:
                     # This goal is unreachable by any equipped rover
                     return math.inf
                total_cost += min_goal_cost

            elif predicate == "communicated_image_data":
                objective, mode = parts[1], parts[2]
                image_needed = not any((r, objective, mode) in have_image for r in self.imaging_rovers)
                comm_needed = True # Always needed if goal is uncommunicated

                min_goal_cost = math.inf

                # Find rovers equipped for imaging that have a camera supporting the mode
                capable_rovers_cameras = [] # List of (rover, camera) tuples
                for r in self.imaging_rovers:
                    for cam, owner_r in self.camera_on_rover.items():
                        if owner_r == r and mode in self.camera_modes.get(cam, set()):
                            capable_rovers_cameras.append((r, cam))

                if not capable_rovers_cameras and image_needed:
                     # Cannot take image if no capable rover/camera exists
                     return math.inf

                for r, cam in capable_rovers_cameras:
                    current_loc = rover_locs.get(r)
                    if current_loc is None or r not in self.dist or current_loc not in self.dist[r]:
                         continue

                    cost = 0
                    current_path_end = current_loc # Rover starts path from its current location

                    # Cost to take image if needed
                    if image_needed:
                        cost += 1 # take_image action

                        calib_needed = (cam, r) not in calibrated
                        target = self.camera_target.get(cam)
                        calib_wps = self.target_waypoints.get(target, set()) if target else set()
                        image_wps = self.objective_waypoints.get(objective, set())

                        if not image_wps:
                             # Cannot take image if objective not visible from anywhere
                             continue # Try next rover/camera

                        min_nav_cost_sequence = math.inf
                        best_image_wp = None
                        # best_calib_wp = None # Keep track of the chosen calibration waypoint (not strictly needed for cost)

                        # Find best path: current -> (calib_wp) -> image_wp
                        if calib_needed:
                            cost += 1 # calibrate action
                            if not calib_wps:
                                # Cannot calibrate if target not visible from anywhere
                                continue # Try next rover/camera

                            for image_wp in image_wps:
                                for calib_wp in calib_wps:
                                    nav1 = self.dist[r][current_path_end].get(calib_wp, math.inf)
                                    nav2 = self.dist[r].get(calib_wp, {}).get(image_wp, math.inf) # Ensure calib_wp is a valid start node for dist
                                    if nav1 != math.inf and nav2 != math.inf:
                                        nav_cost = nav1 + nav2
                                        if nav_cost < min_nav_cost_sequence:
                                            min_nav_cost_sequence = nav_cost
                                            best_image_wp = image_wp
                                            # best_calib_wp = calib_wp # Store the calib waypoint used
                        else: # No calibration needed
                            for image_wp in image_wps:
                                nav_cost = self.dist[r][current_path_end].get(image_wp, math.inf)
                                if nav_cost != math.inf:
                                    if nav_cost < min_nav_cost_sequence:
                                        min_nav_cost_sequence = nav_cost
                                        best_image_wp = image_wp
                                        # best_calib_wp = None # Not used

                        if min_nav_cost_sequence == math.inf: continue # Cannot reach suitable image/calib locations

                        cost += min_nav_cost_sequence
                        current_path_end = best_image_wp # Rover is now assumed to be at image location

                    # Cost to communicate
                    if comm_needed:
                        cost += 1 # communicate action
                        min_nav_cost_to_comm = math.inf
                        # comm_waypoints check is done before the loop

                        # Ensure current_path_end is a valid start node for dist
                        if current_path_end not in self.dist.get(r, {}):
                             continue # Cannot navigate from the assumed end point

                        for comm_wp in self.comm_waypoints:
                            nav_cost = self.dist[r][current_path_end].get(comm_wp, math.inf)
                            if nav_cost != math.inf:
                                min_nav_cost_to_comm = min(min_nav_cost_to_comm, nav_cost)

                        if min_nav_cost_to_comm == math.inf: continue # Cannot reach any comm location from image location
                        cost += min_nav_cost_to_comm

                    min_goal_cost = min(min_goal_cost, cost)

                if min_goal_cost == math.inf:
                     # This goal is unreachable by any capable rover/camera combination
                     return math.inf
                total_cost += min_goal_cost

        return total_cost
