from fnmatch import fnmatch
from collections import deque
import math
# Assuming Heuristic base class is available from heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

# If Heuristic base class is not provided directly, assume a minimal definition
# This is just for self-contained testing if needed, the final output should rely on the import
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    # Define a minimal placeholder if the base class is not found
    class Heuristic:
        def __init__(self, task):
            pass
        def __call__(self, node):
            raise NotImplementedError("Heuristic base class not found, using placeholder.")


# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    if not isinstance(fact, str) or len(fact) < 2 or fact[0] != '(' or fact[-1] != ')':
        # Handle unexpected fact format gracefully
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
    if not parts or len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the total number of actions required to achieve all
    unmet goal conditions. It sums the estimated costs for each individual goal,
    ignoring potential interactions (positive or negative) between goals.
    The estimated cost for each goal is the minimum number of actions
    (navigation, sampling, imaging, communication, dropping) required to
    achieve that specific goal from the current state, considering available
    rovers and locations. Navigation costs are estimated using precomputed
    shortest path distances on the waypoint graph for each rover.

    # Assumptions
    - Each unachieved goal is considered independently.
    - The cost of achieving a goal is the sum of necessary actions (sample/image/calibrate/drop/communicate)
      plus the navigation steps between required locations.
    - Navigation cost between two waypoints for a rover is the shortest path distance
      (number of `navigate` actions) in the rover's traversal graph.
    - If a sample is not at its original waypoint and no rover has it, it is assumed
      to have been communicated already (checked by the goal status).
    - If an image is not held by any rover, it needs to be taken. Taking an image
      requires calibration first. The heuristic assumes calibration is needed if the image is not held.
    - If a goal seems impossible to achieve based on static facts (e.g., no equipped
      rover, no visible waypoints for image/calibration/communication), a large
      constant cost is added for that goal.

    # Heuristic Initialization
    - Extracts static facts such as lander location, rover equipment, store ownership,
      camera capabilities (modes, calibration target, on-board rover), waypoint
      visibility, objective visibility, and rover traversal capabilities.
    - Builds a traversal graph (adjacency list) for each rover based on `can_traverse` facts.
    - Precomputes shortest path distances between all reachable pairs of waypoints
      for each rover using Breadth-First Search (BFS).

    # Step-By-Step Thinking for Computing Heuristic
    1. Parse the current state to quickly access dynamic facts (rover locations,
       collected data, communicated data, sample locations, store status, camera calibration).
    2. Initialize the total heuristic cost to 0.
    3. Identify communication waypoints (visible from the lander's location).
    4. Iterate through each goal condition specified in the task:
       - If the goal is already satisfied in the current state, add 0 to the total cost.
       - If the goal is `(communicated_soil_data ?w)` and not satisfied:
         - Find rovers that already have the soil sample from `?w`.
         - If any rover has the sample: Estimate cost as 1 (communicate) + minimum navigation cost for that rover from its current location to any communication waypoint. Minimize this over all rovers holding the sample.
         - If no rover has the sample but `(at_soil_sample ?w)` is true: Estimate cost as 1 (sample) + (1 if rover's store is full) + 1 (communicate) + minimum navigation cost for an equipped rover from its current location to `?w` and then from `?w` to any communication waypoint. Minimize this over all equipped rovers.
         - If the goal seems impossible (e.g., sample gone and not held, no equipped rover), add a large constant cost.
         - Add the minimum estimated cost for this goal to the total heuristic.
       - If the goal is `(communicated_rock_data ?w)` and not satisfied:
         - Follow symmetric logic as for soil data, using rock-specific predicates and equipment.
       - If the goal is `(communicated_image_data ?o ?m)` and not satisfied:
         - Find rovers that already have the image for `(?o, ?m)`.
         - If any rover has the image: Estimate cost as 1 (communicate) + minimum navigation cost for that rover from its current location to any communication waypoint. Minimize this over all rovers holding the image.
         - If no rover has the image: Estimate cost as 1 (calibrate) + 1 (take_image) + 1 (communicate) + minimum navigation cost for an equipped rover with a suitable camera from its current location to a calibration waypoint (for its camera's target), then to an image waypoint (for `?o`), and then to any communication waypoint. Minimize this over all equipped rovers and suitable cameras and waypoint combinations.
         - If the goal seems impossible (e.g., no equipped rover/camera, no visible waypoints), add a large constant cost.
         - Add the minimum estimated cost for this goal to the total heuristic.
    5. Return the total heuristic cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # Precompute static information
        self.lander_location = None
        self.rover_equipment = {} # rover -> set(equipment_type)
        self.rover_stores = {} # rover -> store
        self.camera_info = {} # camera -> {'rover': rover, 'supports': set(modes), 'calib_target': objective}
        self.waypoint_visibility = {} # wp -> set(visible_wps) # wp is visible *from* visible_wp
        self.objective_visibility = {} # objective -> set(visible_wps) # objective is visible *from* visible_wp
        self.rover_traversal = {} # rover -> Dict[wp, set(reachable_wps)] # Adjacency list
        self.rover_distances = {} # rover -> Dict[start_wp, Dict[end_wp, distance]]

        all_waypoints = set()
        all_rovers = set()

        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue # Skip empty facts

            if parts[0] == 'at_lander':
                self.lander_location = parts[2]
            elif parts[0] == 'equipped_for_soil_analysis':
                rover = parts[1]
                self.rover_equipment.setdefault(rover, set()).add('soil')
                all_rovers.add(rover)
            elif parts[0] == 'equipped_for_rock_analysis':
                rover = parts[1]
                self.rover_equipment.setdefault(rover, set()).add('rock')
                all_rovers.add(rover)
            elif parts[0] == 'equipped_for_imaging':
                rover = parts[1]
                self.rover_equipment.setdefault(rover, set()).add('imaging')
                all_rovers.add(rover)
            elif parts[0] == 'store_of':
                store, rover = parts[1], parts[2]
                self.rover_stores[rover] = store
            elif parts[0] == 'supports':
                camera, mode = parts[1], parts[2]
                self.camera_info.setdefault(camera, {}).setdefault('supports', set()).add(mode)
            elif parts[0] == 'visible':
                wp1, wp2 = parts[1], parts[2]
                self.waypoint_visibility.setdefault(wp1, set()).add(wp2) # wp2 is visible from wp1
                all_waypoints.add(wp1)
                all_waypoints.add(wp2)
            elif parts[0] == 'visible_from':
                obj, wp = parts[1], parts[2]
                self.objective_visibility.setdefault(obj, set()).add(wp) # obj is visible from wp
                all_waypoints.add(wp)
            elif parts[0] == 'calibration_target':
                camera, obj = parts[1], parts[2]
                self.camera_info.setdefault(camera, {})['calib_target'] = obj
            elif parts[0] == 'on_board':
                camera, rover = parts[1], parts[2]
                self.camera_info.setdefault(camera, {})['rover'] = rover
                all_rovers.add(rover)
            elif parts[0] == 'can_traverse':
                 rover, wp1, wp2 = parts[1], parts[2], parts[3]
                 self.rover_traversal.setdefault(rover, {}).setdefault(wp1, set()).add(wp2)
                 all_rovers.add(rover)
                 all_waypoints.add(wp1)
                 all_waypoints.add(wp2)

        # Ensure all rovers have equipment/traversal entries even if empty
        for rover in all_rovers:
             self.rover_equipment.setdefault(rover, set())
             self.rover_traversal.setdefault(rover, {})

        # Precompute distances for each rover
        for rover in all_rovers:
            self.rover_distances[rover] = {}
            # Find all waypoints this rover can start a traversal from or traverse to
            rover_relevant_wps = set(self.rover_traversal.get(rover, {}).keys())
            for wp_sources in self.rover_traversal.get(rover, {}).values():
                rover_relevant_wps.update(wp_sources)

            for start_wp in rover_relevant_wps:
                 self.rover_distances[rover][start_wp] = self._compute_distances(rover, start_wp, self.rover_traversal.get(rover, {}))


    def _compute_distances(self, rover, start_wp, traversal_graph):
        """Computes shortest path distances from start_wp to all reachable waypoints for a rover."""
        distances = {start_wp: 0}
        queue = deque([(start_wp, 0)])

        while queue:
            current_wp, dist = queue.popleft()

            # Get neighbors for this rover from the traversal graph
            neighbors = traversal_graph.get(current_wp, set())

            for next_wp in neighbors:
                # If we haven't found a shorter path yet
                if next_wp not in distances or distances[next_wp] > dist + 1:
                    distances[next_wp] = dist + 1
                    queue.append((next_wp, dist + 1))

        return distances


    def get_distance(self, rover, start_wp, end_wp):
        """Looks up precomputed distance or returns infinity if unreachable."""
        # Check if rover and start_wp are in the precomputed distances
        if rover not in self.rover_distances or start_wp not in self.rover_distances[rover]:
             return math.inf
        return self.rover_distances[rover][start_wp].get(end_wp, math.inf)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Parse current state for quick lookups
        rover_at = {}
        rover_has_soil = set() # (rover, waypoint)
        rover_has_rock = set() # (rover, waypoint)
        rover_has_image = set() # (rover, objective, mode)
        communicated_soil = set() # waypoint
        communicated_rock = set() # waypoint
        communicated_image = set() # (objective, mode)
        soil_samples_at = set() # waypoint
        rock_samples_at = set() # waypoint
        store_status = {} # store -> 'full' or 'empty'
        camera_calibrated = set() # (camera, rover)

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip empty facts

            if parts[0] == 'at' and parts[1].startswith('rover'):
                rover_at[parts[1]] = parts[2]
            elif parts[0] == 'have_soil_analysis':
                rover_has_soil.add((parts[1], parts[2]))
            elif parts[0] == 'have_rock_analysis':
                rover_has_rock.add((parts[1], parts[2]))
            elif parts[0] == 'have_image':
                rover_has_image.add((parts[1], parts[2], parts[3]))
            elif parts[0] == 'communicated_soil_data':
                communicated_soil.add(parts[1])
            elif parts[0] == 'communicated_rock_data':
                communicated_rock.add(parts[1])
            elif parts[0] == 'communicated_image_data':
                communicated_image.add((parts[1], parts[2]))
            elif parts[0] == 'at_soil_sample':
                soil_samples_at.add(parts[1])
            elif parts[0] == 'at_rock_sample':
                rock_samples_at.add(parts[1])
            elif parts[0] == 'full':
                store_status[parts[1]] = 'full'
            elif parts[0] == 'empty':
                 store_status[parts[1]] = 'empty'
            elif parts[0] == 'calibrated':
                camera_calibrated.add((parts[1], parts[2]))

        total_heuristic = 0
        large_cost = 1000 # Cost for seemingly impossible goals

        # Find communication waypoints (rover at wp_r needs (visible wp_r lander_wp))
        lander_wp = self.lander_location
        # Need to find waypoints X such that (visible X lander_wp) is true
        comm_wps = {wp_r for wp_r, visible_wps in self.waypoint_visibility.items() if lander_wp in visible_wps}


        # Iterate through goals
        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue # Skip empty goals
            predicate = parts[0]

            if predicate == 'communicated_soil_data':
                w = parts[1]
                if w in communicated_soil:
                    continue # Goal achieved

                min_goal_cost = math.inf

                # Check if sample is already collected by any rover
                rovers_with_sample = {r for r, sw in rover_has_soil if sw == w}

                if rovers_with_sample:
                    # Sample is collected, need to communicate
                    for r in rovers_with_sample:
                        loc_r = rover_at.get(r)
                        if loc_r is None: continue # Rover location unknown? Should not happen in valid state

                        min_dist_to_comm = math.inf
                        if comm_wps:
                            min_dist_to_comm = min((self.get_distance(r, loc_r, comm_wp) for comm_wp in comm_wps), default=math.inf)

                        if min_dist_to_comm != math.inf:
                            cost = min_dist_to_comm + 1 # communicate
                            min_goal_cost = min(min_goal_cost, cost)

                elif w in soil_samples_at:
                    # Sample is at the waypoint, need to sample and communicate
                    for r in rover_at: # Iterate through all rovers
                        if 'soil' in self.rover_equipment.get(r, set()): # Check if equipped
                            loc_r = rover_at[r]
                            store = self.rover_stores.get(r)
                            if store is None: continue # Rover has no store? Should not happen

                            dist_to_sample = self.get_distance(r, loc_r, w)

                            if dist_to_sample != math.inf:
                                min_dist_sample_to_comm = math.inf
                                if comm_wps:
                                    min_dist_sample_to_comm = min((self.get_distance(r, w, comm_wp) for comm_wp in comm_wps), default=math.inf)

                                if min_dist_sample_to_comm != math.inf:
                                    sample_cost = dist_to_sample + 1 # sample
                                    if store_status.get(store) == 'full': sample_cost += 1 # drop
                                    comm_cost = min_dist_sample_to_comm + 1 # communicate
                                    cost = sample_cost + comm_cost
                                    min_goal_cost = min(min_goal_cost, cost)

                # If min_goal_cost is still infinity, this goal is likely impossible from here
                if min_goal_cost == math.inf:
                    total_heuristic += large_cost
                else:
                    total_heuristic += min_goal_cost

            elif predicate == 'communicated_rock_data':
                w = parts[1]
                if w in communicated_rock:
                    continue # Goal achieved

                min_goal_cost = math.inf

                # Check if sample is already collected by any rover
                rovers_with_sample = {r for r, rw in rover_has_rock if rw == w}

                if rovers_with_sample:
                    # Sample is collected, need to communicate
                    for r in rovers_with_sample:
                        loc_r = rover_at.get(r)
                        if loc_r is None: continue

                        min_dist_to_comm = math.inf
                        if comm_wps:
                            min_dist_to_comm = min((self.get_distance(r, loc_r, comm_wp) for comm_wp in comm_wps), default=math.inf)

                        if min_dist_to_comm != math.inf:
                            cost = min_dist_to_comm + 1 # communicate
                            min_goal_cost = min(min_goal_cost, cost)

                elif w in rock_samples_at:
                    # Sample is at the waypoint, need to sample and communicate
                    for r in rover_at: # Iterate through all rovers
                        if 'rock' in self.rover_equipment.get(r, set()): # Check if equipped
                            loc_r = rover_at[r]
                            store = self.rover_stores.get(r)
                            if store is None: continue

                            dist_to_sample = self.get_distance(r, loc_r, w)

                            if dist_to_sample != math.inf:
                                min_dist_sample_to_comm = math.inf
                                if comm_wps:
                                    min_dist_sample_to_comm = min((self.get_distance(r, w, comm_wp) for comm_wp in comm_wps), default=math.inf)

                                if min_dist_sample_to_comm != math.inf:
                                    sample_cost = dist_to_sample + 1 # sample
                                    if store_status.get(store) == 'full': sample_cost += 1 # drop
                                    comm_cost = min_dist_sample_to_comm + 1 # communicate
                                    cost = sample_cost + comm_cost
                                    min_goal_cost = min(min_goal_cost, cost)

                if min_goal_cost == math.inf:
                    total_heuristic += large_cost
                else:
                    total_heuristic += min_goal_cost

            elif predicate == 'communicated_image_data':
                o, m = parts[1], parts[2]
                if (o, m) in communicated_image:
                    continue # Goal achieved

                min_goal_cost = math.inf

                # Check if image is already taken by any rover
                rovers_with_image = {r for r, io, im in rover_has_image if io == o and im == m}

                if rovers_with_image:
                    # Image is taken, need to communicate
                    for r in rovers_with_image:
                        loc_r = rover_at.get(r)
                        if loc_r is None: continue

                        min_dist_to_comm = math.inf
                        if comm_wps:
                            min_dist_to_comm = min((self.get_distance(r, loc_r, comm_wp) for comm_wp in comm_wps), default=math.inf)

                        if min_dist_to_comm != math.inf:
                            cost = min_dist_to_comm + 1 # communicate
                            min_goal_cost = min(min_goal_cost, cost)

                else:
                    # Need to calibrate, take image, and communicate
                    image_wps = self.objective_visibility.get(o, set())
                    # If no waypoints to view objective, this goal is impossible
                    if not image_wps:
                         total_heuristic += large_cost
                         continue

                    for r in rover_at: # Iterate through all rovers
                        if 'imaging' in self.rover_equipment.get(r, set()): # Check if equipped
                            loc_r = rover_at[r]

                            for camera, cam_info in self.camera_info.items():
                                if cam_info.get('rover') == r and m in cam_info.get('supports', set()):
                                    calib_target = cam_info.get('calib_target')
                                    if calib_target is None: continue # Camera has no calibration target?

                                    calib_wps = self.objective_visibility.get(calib_target, set())
                                    # If no waypoints to view calibration target, this camera is useless for calibration
                                    if not calib_wps: continue

                                    # Need to navigate start -> calib -> image -> comm
                                    min_nav_cost = math.inf
                                    # Iterate over all combinations of suitable waypoints
                                    for w_calib in calib_wps:
                                        for p_image in image_wps:
                                            for w_comm in comm_wps:
                                                dist1 = self.get_distance(r, loc_r, w_calib)
                                                dist2 = self.get_distance(r, w_calib, p_image)
                                                dist3 = self.get_distance(r, p_image, w_comm)
                                                if dist1 != math.inf and dist2 != math.inf and dist3 != math.inf:
                                                    nav_cost = dist1 + dist2 + dist3
                                                    min_nav_cost = min(min_nav_cost, nav_cost)

                                    if min_nav_cost != math.inf:
                                        # Cost = navigation + calibrate + take_image + communicate
                                        cost = min_nav_cost + 1 + 1 + 1
                                        min_goal_cost = min(min_goal_cost, cost)

                if min_goal_cost == math.inf:
                    total_heuristic += large_cost
                else:
                    total_heuristic += min_goal_cost

        return total_heuristic
