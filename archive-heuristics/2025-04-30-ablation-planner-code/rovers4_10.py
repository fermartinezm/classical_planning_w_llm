from fnmatch import fnmatch
from collections import deque
# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

# Define a dummy Heuristic base class if not provided externally
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    class Heuristic:
        def __init__(self, task):
            self.goals = task.goals
            self.static = task.static
        def __call__(self, node):
            raise NotImplementedError


# Helper functions to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
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

# BFS implementation for shortest paths
def bfs(start_node, graph):
    """
    Performs BFS to find shortest paths from start_node in a graph.

    Args:
        start_node: The starting waypoint.
        graph: A dictionary representing the graph {waypoint: {neighbor1, neighbor2, ...}}.

    Returns:
        A dictionary {waypoint: distance} from start_node. Returns infinity for unreachable nodes.
    """
    distances = {node: float('inf') for node in graph}
    if start_node not in graph:
         # Start node might not be in the graph (e.g., not a waypoint in can_traverse)
         return distances # All distances remain infinity

    distances[start_node] = 0
    queue = deque([start_node])

    while queue:
        current_node = queue.popleft()

        # Check if current_node has neighbors in the graph
        if current_node in graph:
            for neighbor in graph[current_node]:
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)
    return distances

# Main heuristic class
class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the required number of actions and minimum navigation steps
    to satisfy all goal conditions. It sums the costs for each unsatisfied goal,
    breaking down the cost into necessary actions (sample, calibrate, take_image, communicate, drop)
    and the minimum navigation required to reach the locations for these actions.

    # Assumptions
    - Action costs are uniform (1).
    - Navigation cost between adjacent waypoints is 1.
    - The heuristic sums costs for goals independently, ignoring potential synergies
      (e.g., collecting multiple samples on one trip) or conflicts (e.g., needing an empty store).
    - Assumes required samples (`at_soil_sample`, `at_rock_sample`) exist at their waypoint
      if needed for a goal and not already collected/communicated.
    - Assumes required image points (`visible_from`) and calibration points exist if needed.
    - Unreachable goals result in an infinite heuristic value.

    # Heuristic Initialization
    - Parses static facts to identify rovers, waypoints, capabilities, camera info,
      store info, lander location, visibility, and traversability.
    - Builds a navigation graph for each rover based on `can_traverse`.
    - Computes all-pairs shortest paths for each rover using BFS.
    - Identifies communication points (waypoints visible from the lander).
    - Identifies image points and calibration points.
    - Stores goal literals.

    # Step-By-Step Thinking for Computing Heuristic
    The heuristic value is the sum of costs for each unsatisfied goal literal.

    For each goal `(communicated_soil_data W)` not yet achieved:
    1. Add 1 for the `communicate_soil_data` action.
    2. Add the minimum navigation cost for any rover from its current location to any communication point. If no communication point is reachable by any rover, the goal is unreachable (infinity).
    3. If `(have_soil_analysis R W)` is not true for any rover R:
       a. Check if the soil sample at W is still available at waypoint W (`at_soil_sample W`). If not, the goal is unreachable (infinity).
       b. Add 1 for the `sample_soil` action.
       c. Add the minimum navigation cost for any *equipped* soil analysis rover from its current location to waypoint W. If no equipped rover can reach W, the goal is unreachable (infinity).
       d. Add 1 if the minimum cost to get an empty store on an equipped rover is 1 (i.e., at least one equipped rover has a full store). If no equipped rover exists, this is handled by 3c.

    For each goal `(communicated_rock_data W)` not yet achieved:
    - Follow a similar process as for soil data, using rock-specific predicates and capabilities.

    For each goal `(communicated_image_data O M)` not yet achieved:
    1. Add 1 for the `communicate_image_data` action.
    2. Add the minimum navigation cost for any rover from its current location to any communication point. If no communication point is reachable by any rover, the goal is unreachable (infinity).
    3. If `(have_image R O M)` is not true for any rover R:
       a. Check if there are any waypoints from which objective O is visible (`visible_from O P`). If not, the goal is unreachable (infinity).
       b. Add 1 for the `take_image` action.
       c. Find suitable rover-camera pairs (rover R equipped for imaging, camera I on board R, camera I supports mode M). If none exist, the goal is unreachable (infinity).
       d. Determine the minimum navigation cost and calibration cost among all suitable pairs:
          - If the camera I on rover R is already calibrated: Add the minimum navigation cost for rover R from its current location to any image point P for objective O. Calibration cost is 0.
          - If the camera I on rover R is not calibrated: Add 1 for the `calibrate` action. Add the minimum navigation cost for rover R from its current location to any calibration point W for camera I, plus the navigation cost from W to any image point P for objective O. If no calibration points or image points exist/are reachable for any suitable pair, the goal is unreachable (infinity).
          - Take the minimum navigation + calibration cost over all suitable (R, I) pairs. Add this minimum to the heuristic.

    The total heuristic is the sum of costs calculated for each unsatisfied goal. If any part of the calculation results in infinity, the total heuristic is infinity.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals

        # Precompute static information
        self.rovers = set()
        self.waypoints = set()
        self.stores = set()
        self.cameras = set()
        self.modes = set()
        self.landers = set()
        self.objectives = set()

        self.lander_loc = None
        self.equipped_soil = set()
        self.equipped_rock = set()
        self.equipped_imaging = set()
        self.rover_stores = {} # {rover: store}
        self.rover_cameras = {} # {rover: {camera}}
        self.camera_modes = {} # {camera: {mode}}
        self.can_traverse_graph = {} # {rover: {from_wp: {to_wp}}}
        self.visible_wps = set() # {(wp1, wp2)}
        self.calibration_targets = {} # {camera: objective}
        self.image_points = {} # {objective: {waypoint}}
        self.cal_points = {} # {camera: {waypoint}}

        # Collect all mentioned objects by type from static facts
        all_waypoints_set = set()
        all_rovers_set = set()

        for fact in task.static:
            parts = get_parts(fact)
            predicate = parts[0]

            if predicate == 'at_lander':
                lander, wp = parts[1], parts[2]
                self.landers.add(lander)
                self.waypoints.add(wp)
                all_waypoints_set.add(wp)
                self.lander_loc = wp
            elif predicate == 'equipped_for_soil_analysis':
                rover = parts[1]
                self.rovers.add(rover)
                all_rovers_set.add(rover)
                self.equipped_soil.add(rover)
            elif predicate == 'equipped_for_rock_analysis':
                rover = parts[1]
                self.rovers.add(rover)
                all_rovers_set.add(rover)
                self.equipped_rock.add(rover)
            elif predicate == 'equipped_for_imaging':
                rover = parts[1]
                self.rovers.add(rover)
                all_rovers_set.add(rover)
                self.equipped_imaging.add(rover)
            elif predicate == 'store_of':
                store, rover = parts[1], parts[2]
                self.stores.add(store)
                self.rovers.add(rover)
                all_rovers_set.add(rover)
                self.rover_stores[rover] = store
            elif predicate == 'visible':
                wp1, wp2 = parts[1], parts[2]
                self.waypoints.add(wp1)
                self.waypoints.add(wp2)
                all_waypoints_set.add(wp1)
                all_waypoints_set.add(wp2)
                self.visible_wps.add((wp1, wp2))
            elif predicate == 'can_traverse':
                rover, wp1, wp2 = parts[1], parts[2], parts[3]
                self.rovers.add(rover)
                self.waypoints.add(wp1)
                self.waypoints.add(wp2)
                all_rovers_set.add(rover)
                all_waypoints_set.add(wp1)
                all_waypoints_set.add(wp2)
                if rover not in self.can_traverse_graph:
                    self.can_traverse_graph[rover] = {}
                if wp1 not in self.can_traverse_graph[rover]:
                    self.can_traverse_graph[rover][wp1] = set()
                self.can_traverse_graph[rover][wp1].add(wp2)
                # Ensure all mentioned waypoints are nodes in the graph, even if no outgoing edges
                if wp2 not in self.can_traverse_graph[rover]:
                     self.can_traverse_graph[rover][wp2] = set()
            elif predicate == 'calibration_target':
                camera, objective = parts[1], parts[2]
                self.cameras.add(camera)
                self.objectives.add(objective)
                self.calibration_targets[camera] = objective
            elif predicate == 'on_board':
                camera, rover = parts[1], parts[2]
                self.cameras.add(camera)
                self.rovers.add(rover)
                all_rovers_set.add(rover)
                if rover not in self.rover_cameras:
                    self.rover_cameras[rover] = set()
                self.rover_cameras[rover].add(camera)
            elif predicate == 'supports':
                camera, mode = parts[1], parts[2]
                self.cameras.add(camera)
                self.modes.add(mode)
                if camera not in self.camera_modes:
                    self.camera_modes[camera] = set()
                self.camera_modes[camera].add(mode)
            elif predicate == 'visible_from':
                objective, wp = parts[1], parts[2]
                self.objectives.add(objective)
                self.waypoints.add(wp)
                all_waypoints_set.add(wp)
                if objective not in self.image_points:
                    self.image_points[objective] = set()
                self.image_points[objective].add(wp)
            # Initial samples are state facts, not static in this domain

        # Ensure all waypoints mentioned in static facts are nodes in the graph for each rover
        all_relevant_waypoints = self.waypoints
        for rover in self.rovers: # Iterate over rovers found in static facts
             if rover not in self.can_traverse_graph:
                  self.can_traverse_graph[rover] = {}
             for wp in all_relevant_waypoints:
                  if wp not in self.can_traverse_graph[rover]:
                       self.can_traverse_graph[rover][wp] = set()


        # Compute calibration points based on calibration targets and visible_from
        for camera, target_objective in self.calibration_targets.items():
            if target_objective in self.image_points: # visible_from is used for both image and calibration targets
                self.cal_points[camera] = self.image_points[target_objective]
            else:
                self.cal_points[camera] = set() # No waypoints visible from the calibration target

        # Compute communication points (waypoints visible from lander)
        self.comm_points = {wp for wp1, wp2 in self.visible_wps for wp in [wp1, wp2] if wp1 == self.lander_loc or wp2 == self.lander_loc}

        # Compute all-pairs shortest paths for each rover
        self.dist = {}
        for rover in self.rovers:
            self.dist[rover] = {}
            # Use the graph built from can_traverse, ensuring all relevant waypoints are nodes
            graph_for_rover = self.can_traverse_graph.get(rover, {})

            for start_wp in graph_for_rover:
                self.dist[rover][start_wp] = bfs(start_wp, graph_for_rover)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        infinity = float('inf')

        # Extract current state information
        rover_loc = {} # {rover: waypoint}
        store_full = {} # {store: True}
        camera_calibrated = {} # {camera: True}
        have_soil = {} # {rover: {waypoint}}
        have_rock = {} # {rover: {waypoint}}
        have_image = {} # {rover: {(objective, mode)}}
        soil_at_wp = set() # {waypoint}
        rock_at_wp = set() # {waypoint}
        communicated_soil = set() # {waypoint}
        communicated_rock = set() # {waypoint}
        communicated_image = set() # {(objective, mode)}

        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == 'at':
                obj, wp = parts[1], parts[2]
                if obj in self.rovers: # Assuming only rovers have 'at' location in state relevant for navigation
                    rover_loc[obj] = wp
            elif predicate == 'full':
                store = parts[1]
                store_full[store] = True
            elif predicate == 'calibrated':
                camera, rover = parts[1], parts[2]
                camera_calibrated[camera] = True
            elif predicate == 'have_soil_analysis':
                rover, wp = parts[1], parts[2]
                if rover not in have_soil:
                    have_soil[rover] = set()
                have_soil[rover].add(wp)
            elif predicate == 'have_rock_analysis':
                rover, wp = parts[1], parts[2]
                if rover not in have_rock:
                    have_rock[rover] = set()
                have_rock[rover].add(wp)
            elif predicate == 'have_image':
                rover, obj, mode = parts[1], parts[2], parts[3]
                if rover not in have_image:
                    have_image[rover] = set()
                have_image[rover].add((obj, mode))
            elif predicate == 'at_soil_sample':
                wp = parts[1]
                soil_at_wp.add(wp)
            elif predicate == 'at_rock_sample':
                wp = parts[1]
                rock_at_wp.add(wp)
            elif predicate == 'communicated_soil_data':
                wp = parts[1]
                communicated_soil.add(wp)
            elif predicate == 'communicated_rock_data':
                wp = parts[1]
                communicated_rock.add(wp)
            elif predicate == 'communicated_image_data':
                obj, mode = parts[1], parts[2]
                communicated_image.add((obj, mode))

        h = 0

        # Compute cost for unsatisfied goals
        for goal in self.goals:
            parts = get_parts(goal)
            predicate = parts[0]

            if predicate == 'communicated_soil_data':
                wp = parts[1]
                if wp not in communicated_soil:
                    # Cost for communication
                    h += 1 # communicate action
                    min_nav_to_comm = infinity
                    for rover in self.rovers:
                        current_wp = rover_loc.get(rover)
                        # Check if rover location is known and is a valid start node for BFS distances
                        if current_wp and rover in self.dist and current_wp in self.dist[rover]:
                            # Find min distance from current_wp to any comm_wp reachable by this rover
                            nav_cost = min((self.dist[rover][current_wp][comm_wp] for comm_wp in self.comm_points if comm_wp in self.dist[rover][current_wp]), default=infinity)
                            min_nav_to_comm = min(min_nav_to_comm, nav_cost)
                    if min_nav_to_comm == infinity: return infinity # Cannot reach any comm point with any rover
                    h += min_nav_to_comm

                    # Cost for getting sample if not already held by any rover
                    has_sample = any(wp in have_soil.get(r, set()) for r in self.rovers)
                    if not has_sample:
                        if wp not in soil_at_wp:
                             return infinity # Sample needed but not at waypoint

                        h += 1 # sample action
                        min_nav_to_sample = infinity
                        min_store_cost = infinity
                        found_equipped = False
                        for rover in self.equipped_soil:
                            found_equipped = True
                            current_wp = rover_loc.get(rover)
                            # Check if rover location is known and is a valid start node for BFS distances
                            if current_wp and rover in self.dist and current_wp in self.dist[rover]:
                                # Check if sample waypoint is reachable by this rover
                                if wp in self.dist[rover][current_wp]:
                                    min_nav_to_sample = min(min_nav_to_sample, self.dist[rover][current_wp][wp])
                            store = self.rover_stores.get(rover)
                            if store:
                                min_store_cost = min(min_store_cost, 1 if store_full.get(store, False) else 0)

                        if not found_equipped or min_nav_to_sample == infinity or min_store_cost == infinity:
                            return infinity # Cannot sample (no equipped rover, or cannot reach sample, or cannot manage store)

                        h += min_nav_to_sample
                        h += min_store_cost

            elif predicate == 'communicated_rock_data':
                wp = parts[1]
                if wp not in communicated_rock:
                    # Cost for communication
                    h += 1 # communicate action
                    min_nav_to_comm = infinity
                    for rover in self.rovers:
                        current_wp = rover_loc.get(rover)
                        if current_wp and rover in self.dist and current_wp in self.dist[rover]:
                            nav_cost = min((self.dist[rover][current_wp][comm_wp] for comm_wp in self.comm_points if comm_wp in self.dist[rover][current_wp]), default=infinity)
                            min_nav_to_comm = min(min_nav_to_comm, nav_cost)
                    if min_nav_to_comm == infinity: return infinity
                    h += min_nav_to_comm

                    # Cost for getting sample if not already held by any rover
                    has_sample = any(wp in have_rock.get(r, set()) for r in self.rovers)
                    if not has_sample:
                        if wp not in rock_at_wp:
                             return infinity # Sample needed but not at waypoint

                        h += 1 # sample action
                        min_nav_to_sample = infinity
                        min_store_cost = infinity
                        found_equipped = False
                        for rover in self.equipped_rock:
                            found_equipped = True
                            current_wp = rover_loc.get(rover)
                            if current_wp and rover in self.dist and current_wp in self.dist[rover] and wp in self.dist[rover][current_wp]:
                                min_nav_to_sample = min(min_nav_to_sample, self.dist[rover][current_wp][wp])
                            store = self.rover_stores.get(rover)
                            if store:
                                min_store_cost = min(min_store_cost, 1 if store_full.get(store, False) else 0)

                        if not found_equipped or min_nav_to_sample == infinity or min_store_cost == infinity:
                            return infinity # Cannot sample

                        h += min_nav_to_sample
                        h += min_store_cost

            elif predicate == 'communicated_image_data':
                obj, mode = parts[1], parts[2]
                if (obj, mode) not in communicated_image:
                    # Cost for communication
                    h += 1 # communicate action
                    min_nav_to_comm = infinity
                    for rover in self.rovers:
                        current_wp = rover_loc.get(rover)
                        if current_wp and rover in self.dist and current_wp in self.dist[rover]:
                            nav_cost = min((self.dist[rover][current_wp][comm_wp] for comm_wp in self.comm_points if comm_wp in self.dist[rover][current_wp]), default=infinity)
                            min_nav_to_comm = min(min_nav_to_comm, nav_cost)
                    if min_nav_to_comm == infinity: return infinity
                    h += min_nav_to_comm

                    # Cost for getting image if not already held by any rover
                    has_image_data = any((obj, mode) in have_image.get(r, set()) for r in self.rovers)
                    if not has_image_data:
                        if obj not in self.image_points or not self.image_points[obj]:
                            return infinity # No image points for objective

                        h += 1 # take image action
                        min_nav_to_image = infinity
                        min_cal_cost = infinity # Cost for calibrate action (0 or 1)
                        found_suitable_pair = False

                        for rover in self.equipped_imaging:
                            if rover in self.rover_cameras:
                                for camera in self.rover_cameras[rover]:
                                    if camera in self.camera_modes and mode in self.camera_modes[camera]:
                                        found_suitable_pair = True
                                        current_wp = rover_loc.get(rover)
                                        if not current_wp or rover not in self.dist or current_wp not in self.dist[rover]:
                                             continue # Rover location unknown or not in graph

                                        if camera in camera_calibrated and camera_calibrated[camera]:
                                            # Already calibrated, just need to navigate to image point
                                            if obj in self.image_points and self.image_points[obj]:
                                                nav_cost = min((self.dist[rover][current_wp][img_wp] for img_wp in self.image_points[obj] if img_wp in self.dist[rover][current_wp]), default=infinity)
                                                if nav_cost != infinity:
                                                    min_nav_to_image = min(min_nav_to_image, nav_cost)
                                                    min_cal_cost = min(min_cal_cost, 0) # No calibrate action needed
                                            # else: No image points for this objective, handled by outer check
                                        else:
                                            # Needs calibration, navigate to cal point then image point
                                            if camera in self.cal_targets and self.cal_targets[camera] in self.cal_points and self.cal_points[self.cal_targets[camera]]:
                                                cal_target_obj = self.cal_targets[camera]
                                                cal_wps = self.cal_points[cal_target_obj]
                                                img_wps = self.image_points.get(obj, set()) # Use .get with default empty set

                                                if cal_wps and img_wps: # Ensure both sets are non-empty
                                                    h_ri_nav = infinity
                                                    for cal_wp in cal_wps:
                                                        for img_wp in img_wps:
                                                            if current_wp in self.dist[rover] and cal_wp in self.dist[rover][current_wp] and img_wp in self.dist[rover].get(cal_wp, {}): # Check reachability W -> P
                                                                h_ri_nav = min(h_ri_nav, self.dist[rover][current_wp][cal_wp] + self.dist[rover][cal_wp][img_wp])

                                                    if h_ri_nav != infinity:
                                                        min_nav_to_image = min(min_nav_to_image, h_ri_nav)
                                                        min_cal_cost = min(min_cal_cost, 1) # Calibrate action needed

                        if not found_suitable_pair or min_nav_to_image == infinity or min_cal_cost == infinity:
                            return infinity # Cannot take image

                        h += min_nav_to_image
                        h += min_cal_cost

            # Handle other potential goal types if they existed (e.g. (at rover1 waypoint5))
            # Based on examples, only communicated_... goals are present.
            # If an unknown goal type is encountered, return infinity as we cannot estimate its cost.
            elif predicate not in ['communicated_soil_data', 'communicated_rock_data', 'communicated_image_data']:
                 return infinity


        return h
