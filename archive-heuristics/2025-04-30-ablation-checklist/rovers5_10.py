from fnmatch import fnmatch
from collections import deque, defaultdict
from typing import Dict, Set, Tuple, Any, List, Union

# Assume Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

# Define a dummy Heuristic base class for standalone testing if needed
# In the final integrated code, remove this dummy class and use the actual import.
class Heuristic:
    def __init__(self, task):
        pass
    def __call__(self, node):
        pass

def get_parts(fact: str) -> List[str]:
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact: str, *args: str) -> bool:
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

def bfs_all_waypoints(graph: Dict[str, Set[str]], start_node: str, all_waypoints: Set[str]) -> Dict[str, int]:
    """
    Computes shortest path distances from a start node in a graph,
    considering all known waypoints.

    Args:
        graph: Adjacency list representation {node: {neighbor1, neighbor2, ...}}
               May not contain all waypoints as keys if they have no outgoing edges.
        start_node: The starting waypoint.
        all_waypoints: A set of all waypoints in the problem.

    Returns:
        A dictionary mapping each reachable waypoint to its distance from the start_node.
        Returns float('inf') for unreachable waypoints.
    """
    distances = {wp: float('inf') for wp in all_waypoints}

    if start_node not in all_waypoints:
         # Start node is not a known waypoint. Cannot start traversal.
         return distances

    distances[start_node] = 0
    queue = deque([start_node])

    while queue:
        current_node = queue.popleft()

        # Check if current_node is in the graph keys before iterating neighbors
        # A waypoint might exist but have no outgoing 'can_traverse' edges
        if current_node in graph:
            for neighbor in graph[current_node]:
                # Ensure neighbor is a known waypoint
                if neighbor in all_waypoints:
                    # If we found a shorter path (only happens first time in BFS on unweighted graph)
                    if distances[neighbor] == float('inf'):
                        distances[neighbor] = distances[current_node] + 1
                        queue.append(neighbor)

    return distances


class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    Estimates the number of actions required to achieve all uncommunicated goals.
    The heuristic sums the estimated cost for each unachieved goal independently.

    For soil/rock goals:
    Cost = (cost to navigate to sample) + (cost to drop if store full) + (sample action) + (cost to navigate to comm waypoint) + (communicate action)

    For image goals:
    Cost = (cost to navigate to calibration waypoint if needed) + (calibrate action if needed) + (cost to navigate to imaging waypoint) + (take image action) + (cost to navigate to comm waypoint) + (communicate action)

    Navigation costs are estimated using shortest path distances on the rover's
    traverse graph. Minimum costs are taken over suitable rovers/cameras and
    reachable target waypoints (sample, image, calibration, communication).
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and static facts,
        and precomputing navigation distances and relevant waypoints.
        """
        self.goals = task.goals

        # --- Precomputation from static facts ---
        self.rover_capabilities: Dict[str, Set[str]] = defaultdict(set)
        self.rover_stores: Dict[str, str] = {} # rover -> store
        self.lander_location: str = None
        self.rover_can_traverse: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set)) # rover -> {wp -> {reachable_wps}}
        self.objective_visibility: Dict[str, Set[str]] = defaultdict(set) # objective -> {visible_from_wps}
        self.camera_info: Dict[str, Dict[str, Any]] = {} # camera -> { 'rover': r, 'modes': {m}, 'cal_target': o }
        self.calibration_targets: Dict[str, str] = {} # camera -> objective
        self.all_waypoints: Set[str] = set() # Collect all waypoints

        # First pass to collect all waypoints from static facts, initial state, and goals
        potential_waypoint_preds = {
            'at_lander', 'can_traverse', 'visible', 'visible_from',
            'at_soil_sample', 'at_rock_sample', 'at'
        }
        # Collect from static facts
        for fact in task.static:
             parts = get_parts(fact)
             if not parts: continue
             if parts[0] in potential_waypoint_preds:
                  for arg in parts[1:]:
                       if arg.startswith('waypoint'):
                            self.all_waypoints.add(arg)
        # Collect from initial state and goals
        for fact in task.initial_state | task.goals:
             parts = get_parts(fact)
             if not parts: continue
             if parts[0] in potential_waypoint_preds:
                  for arg in parts[1:]:
                       if arg.startswith('waypoint'):
                            self.all_waypoints.add(arg)


        # Second pass to build structures using collected waypoints
        for fact in task.static:
            parts = get_parts(fact)
            if not parts: continue

            pred = parts[0]
            if pred == 'equipped_for_soil_analysis':
                self.rover_capabilities[parts[1]].add('soil')
            elif pred == 'equipped_for_rock_analysis':
                self.rover_capabilities[parts[1]].add('rock')
            elif pred == 'equipped_for_imaging':
                self.rover_capabilities[parts[1]].add('imaging')
            elif pred == 'store_of':
                self.rover_stores[parts[2]] = parts[1] # rover -> store
            elif pred == 'at_lander':
                self.lander_location = parts[2]
            elif pred == 'can_traverse':
                rover, wp1, wp2 = parts[1], parts[2], parts[3]
                # Ensure waypoints are in our collected set before adding to graph
                if wp1 in self.all_waypoints and wp2 in self.all_waypoints:
                    self.rover_can_traverse[rover][wp1].add(wp2)
            elif pred == 'visible_from':
                objective, waypoint = parts[1], parts[2]
                if waypoint in self.all_waypoints:
                    self.objective_visibility[objective].add(waypoint)
            elif pred == 'on_board':
                camera, rover = parts[1], parts[2]
                if camera not in self.camera_info:
                    self.camera_info[camera] = {'rover': rover, 'modes': set(), 'cal_target': None}
                self.camera_info[camera]['rover'] = rover
            elif pred == 'supports':
                camera, mode = parts[1], parts[2]
                if camera not in self.camera_info:
                     self.camera_info[camera] = {'rover': None, 'modes': set(), 'cal_target': None}
                self.camera_info[camera]['modes'].add(mode)
            elif pred == 'calibration_target':
                camera, objective = parts[1], parts[2]
                if camera not in self.camera_info:
                     self.camera_info[camera] = {'rover': None, 'modes': set(), 'cal_target': None}
                self.camera_info[camera]['cal_target'] = objective
                self.calibration_targets[camera] = objective # Store separately for easy lookup

        # Ensure all waypoints are keys in the rover graph adjacency list, even if they have no edges
        # Do this for all rovers that have any capability or store or camera
        all_rovers = set(self.rover_capabilities.keys()) | set(self.rover_stores.keys()) | {info['rover'] for info in self.camera_info.values() if info.get('rover')}
        for rover in all_rovers:
             if rover not in self.rover_can_traverse:
                  self.rover_can_traverse[rover] = {} # Initialize if rover had no can_traverse facts
             for wp in self.all_waypoints:
                  if wp not in self.rover_can_traverse[rover]:
                       self.rover_can_traverse[rover][wp] = set()


        # Compute shortest paths for each rover from every waypoint
        self.rover_distances: Dict[str, Dict[str, Dict[str, int]]] = {}
        for rover, graph in self.rover_can_traverse.items():
            self.rover_distances[rover] = {}
            for start_wp in self.all_waypoints: # BFS from all known waypoints
                 self.rover_distances[rover][start_wp] = bfs_all_waypoints(graph, start_wp, self.all_waypoints)


        # Compute communication waypoints (visible from lander)
        self.comm_wps: Set[str] = set()
        if self.lander_location:
             # Need waypoints ?x such that (visible ?x lander_wp).
             self.comm_wps = {get_parts(fact)[1] for fact in task.static if match(fact, 'visible', '*', self.lander_location)}
             # Also include the lander location itself if it's a waypoint and visible from itself (unlikely but possible)
             # Or if a rover can be *at* the lander location and communicate.
             # The predicate is (at ?r ?x) (at_lander ?l ?y) (visible ?x ?y). So ?x is the comm waypoint.
             # If lander_location is a waypoint and visible from itself, it's a comm wp.
             # Check if (visible lander_location lander_location) is in static facts.
             # This check is slightly complex as we only have fact strings. Let's assume visible is symmetric
             # and if (visible A B) is true, (visible B A) is also true.
             # So if lander_location is a waypoint, and (visible lander_location lander_location) is a static fact,
             # or if there is any (visible X lander_location) fact, then lander_location is a comm wp.
             # The current collection already gets X from (visible X lander_location).
             # Let's explicitly add lander_location if it's a waypoint and there's a visible fact involving it.
             # A simpler approach: if lander_location is a waypoint, it's likely a comm waypoint itself.
             if self.lander_location in self.all_waypoints:
                  self.comm_wps.add(self.lander_location)


        # Compute min distance to any communication waypoint for each rover from each waypoint
        self.min_dist_to_comm: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(lambda: float('inf')))
        for rover in self.rover_distances:
            for start_wp in self.all_waypoints:
                min_d = float('inf')
                for comm_wp in self.comm_wps:
                    # Ensure comm_wp is a valid destination in the distance map
                    if comm_wp in self.rover_distances[rover][start_wp]:
                         min_d = min(min_d, self.rover_distances[rover][start_wp][comm_wp])
                self.min_dist_to_comm[rover][start_wp] = min_d


    def get_min_dist_and_wp(self, rover: str, start_wp: str, target_wps: Set[str]) -> Tuple[float, str | None]:
        """Helper to find the minimum distance and the corresponding waypoint."""
        if not target_wps:
            return float('inf'), None # Cannot reach any target

        if rover not in self.rover_distances or start_wp not in self.rover_distances[rover]:
             # Rover cannot navigate from start_wp or rover is unknown
             return float('inf'), None

        min_d = float('inf')
        best_wp = None
        distances_from_start = self.rover_distances[rover][start_wp]

        for target_wp in target_wps:
            if target_wp in distances_from_start:
                dist = distances_from_start[target_wp]
                if dist < min_d:
                    min_d = dist
                    best_wp = target_wp
        return min_d, best_wp

    def __call__(self, node) -> float: # Return float to handle inf
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # --- Extract relevant info from current state ---
        current_locations: Dict[str, str] = {} # object -> waypoint
        store_status: Dict[str, str] = {} # rover -> 'empty' or 'full'
        calibrated_cameras: Set[Tuple[str, str]] = set() # (camera, rover)
        have_soil: Set[Tuple[str, str]] = set() # (rover, waypoint)
        have_rock: Set[Tuple[str, str]] = set() # (rover, waypoint)
        have_image: Set[Tuple[str, str, str]] = set() # (rover, objective, mode)

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            pred = parts[0]
            if pred == 'at':
                obj, wp = parts[1], parts[2]
                current_locations[obj] = wp
            elif pred == 'empty':
                 store = parts[1]
                 # Find the rover for this store using precomputed map
                 if store in self.rover_stores:
                      store_status[self.rover_stores[store]] = 'empty'
            elif pred == 'full':
                 store = parts[1]
                 # Find the rover for this store using precomputed map
                 if store in self.rover_stores:
                      store_status[self.rover_stores[store]] = 'full'
            elif pred == 'calibrated':
                camera, rover = parts[1], parts[2]
                calibrated_cameras.add((camera, rover))
            elif pred == 'have_soil_analysis':
                rover, wp = parts[1], parts[2]
                have_soil.add((rover, wp))
            elif pred == 'have_rock_analysis':
                rover, wp = parts[1], parts[2]
                have_rock.add((rover, wp))
            elif pred == 'have_image':
                rover, obj, mode = parts[1], parts[2], parts[3]
                have_image.add((rover, obj, mode))

        total_cost = 0.0 # Use float for summing potentially infinite costs

        # --- Estimate cost for each unachieved goal ---
        for goal in self.goals:
            if goal in state:
                continue # Goal already achieved

            parts = get_parts(goal)
            if not parts: continue

            pred = parts[0]

            if pred == 'communicated_soil_data':
                waypoint = parts[1]
                # Check if we already have the sample data for *any* rover
                have_fact_present = any((r, waypoint) in have_soil for r in self.rover_capabilities if 'soil' in self.rover_capabilities[r])

                min_goal_cost = float('inf')

                if have_fact_present:
                    # Cost to communicate: navigate to comm wp + communicate
                    min_comm_cost_for_goal = float('inf')
                    # Find the rover(s) that have the data
                    rovers_with_data = {r for r, w in have_soil if w == waypoint and r in current_locations}
                    for rover in rovers_with_data:
                         current_wp = current_locations[rover]
                         comm_nav_cost = self.min_dist_to_comm[rover].get(current_wp, float('inf'))
                         if comm_nav_cost != float('inf'):
                             min_comm_cost_for_goal = min(min_comm_cost_for_goal, comm_nav_cost + 1)
                    min_goal_cost = min_comm_cost_for_goal

                else:
                    # Cost to sample and communicate: navigate to sample + drop(opt) + sample + navigate to comm wp + communicate
                    min_sample_and_comm_cost = float('inf')
                    for rover in self.rover_capabilities:
                        if 'soil' in self.rover_capabilities[rover] and rover in current_locations:
                            current_wp = current_locations[rover]
                            store_is_full = (store_status.get(rover) == 'full') # Default to not full if status unknown
                            drop_cost = 1 if store_is_full else 0

                            # Cost to navigate to sample waypoint
                            nav_to_sample_cost, _ = self.get_min_dist_and_wp(rover, current_wp, {waypoint})

                            # Cost to navigate from sample waypoint to a communication waypoint
                            nav_to_comm_cost = self.min_dist_to_comm[rover].get(waypoint, float('inf'))

                            if nav_to_sample_cost != float('inf') and nav_to_comm_cost != float('inf'):
                                # Total cost for this rover: navigate(curr->sample) + drop(opt) + sample + navigate(sample->comm) + communicate
                                cost = nav_to_sample_cost + drop_cost + 1 + nav_to_comm_cost + 1
                                min_sample_and_comm_cost = min(min_sample_and_comm_cost, cost)
                    min_goal_cost = min_sample_and_comm_cost

                total_cost += min_goal_cost


            elif pred == 'communicated_rock_data':
                waypoint = parts[1]
                # Check if we already have the sample data for *any* rover
                have_fact_present = any((r, waypoint) in have_rock for r in self.rover_capabilities if 'rock' in self.rover_capabilities[r])

                min_goal_cost = float('inf')

                if have_fact_present:
                    # Cost to communicate: navigate to comm wp + communicate
                    min_comm_cost_for_goal = float('inf')
                    # Find the rover(s) that have the data
                    rovers_with_data = {r for r, w in have_rock if w == waypoint and r in current_locations}
                    for rover in rovers_with_data:
                         current_wp = current_locations[rover]
                         comm_nav_cost = self.min_dist_to_comm[rover].get(current_wp, float('inf'))
                         if comm_nav_cost != float('inf'):
                             min_comm_cost_for_goal = min(min_comm_cost_for_goal, comm_nav_cost + 1)
                    min_goal_cost = min_comm_cost_for_goal

                else:
                    # Cost to sample and communicate: navigate to sample + drop(opt) + sample + navigate to comm wp + communicate
                    min_sample_and_comm_cost = float('inf')
                    for rover in self.rover_capabilities:
                        if 'rock' in self.rover_capabilities[rover] and rover in current_locations:
                            current_wp = current_locations[rover]
                            store_is_full = (store_status.get(rover) == 'full')
                            drop_cost = 1 if store_is_full else 0

                            # Cost to navigate to sample waypoint
                            nav_to_sample_cost, _ = self.get_min_dist_and_wp(rover, current_wp, {waypoint})

                            # Cost to navigate from sample waypoint to a communication waypoint
                            nav_to_comm_cost = self.min_dist_to_comm[rover].get(waypoint, float('inf'))

                            if nav_to_sample_cost != float('inf') and nav_to_comm_cost != float('inf'):
                                # Total cost for this rover: navigate(curr->sample) + drop(opt) + sample + navigate(sample->comm) + communicate
                                cost = nav_to_sample_cost + drop_cost + 1 + nav_to_comm_cost + 1
                                min_sample_and_comm_cost = min(min_sample_and_comm_cost, cost)
                    min_goal_cost = min_sample_and_comm_cost

                total_cost += min_goal_cost


            elif pred == 'communicated_image_data':
                objective, mode = parts[1], parts[2]
                # Check if we already have the image data for *any* rover
                have_fact_present = any((r, objective, mode) in have_image for r in self.rover_capabilities if 'imaging' in self.rover_capabilities[r])

                min_goal_cost = float('inf')

                if have_fact_present:
                    # Cost to communicate: navigate to comm wp + communicate
                    min_comm_cost_for_goal = float('inf')
                    # Find the rover(s) that have the data
                    rovers_with_data = {r for r, o, m in have_image if o == objective and m == mode and r in current_locations}
                    for rover in rovers_with_data:
                         current_wp = current_locations[rover]
                         comm_nav_cost = self.min_dist_to_comm[rover].get(current_wp, float('inf'))
                         if comm_nav_cost != float('inf'):
                             min_comm_cost_for_goal = min(min_comm_cost_for_goal, comm_nav_cost + 1)
                    min_goal_cost = min_comm_cost_for_goal

                else:
                    # Cost to take image and communicate
                    min_take_image_and_comm_cost = float('inf')
                    # Iterate over suitable rover/camera pairs
                    for camera, info in self.camera_info.items():
                        rover = info.get('rover') # Use .get for safety
                        supported_modes = info.get('modes', set())
                        cal_target = info.get('cal_target')

                        if rover and rover in self.rover_capabilities and 'imaging' in self.rover_capabilities[rover] and mode in supported_modes and rover in current_locations:
                            current_wp = current_locations[rover]

                            # Find best imaging waypoint for this objective
                            imaging_wps = self.objective_visibility.get(objective, set())
                            if not imaging_wps: continue # Cannot image this objective

                            # Cost depends on calibration status
                            is_currently_calibrated = (camera, rover) in calibrated_cameras

                            if is_currently_calibrated:
                                # Already calibrated: navigate to image wp + take image + navigate to comm wp + communicate
                                # Find best imaging wp from current location
                                min_nav_to_image, best_image_wp = self.get_min_dist_and_wp(rover, current_wp, imaging_wps)

                                if best_image_wp is not None and min_nav_to_image != float('inf'):
                                     # Cost to get image: navigate(curr->image) + take_image
                                     cost_get_image = min_nav_to_image + 1
                                     # Cost to communicate: navigate(image->comm) + communicate
                                     cost_communicate = self.min_dist_to_comm[rover].get(best_image_wp, float('inf')) + 1
                                     if cost_communicate != float('inf'):
                                          min_take_image_and_comm_cost = min(min_take_image_and_comm_cost, cost_get_image + cost_communicate)

                            else:
                                # Not calibrated: navigate to cal wp + calibrate + navigate to image wp + take image + navigate to comm wp + communicate
                                if cal_target is None: continue # Cannot calibrate this camera

                                # Find best calibration waypoint for this camera's target
                                cal_wps = self.objective_visibility.get(cal_target, set())
                                if not cal_wps: continue # Cannot calibrate this camera

                                # Find best calibration wp from current location
                                min_nav_to_cal, best_cal_wp = self.get_min_dist_and_wp(rover, current_wp, cal_wps)

                                if best_cal_wp is not None and min_nav_to_cal != float('inf'):
                                     # Find best imaging wp from the chosen calibration waypoint
                                     min_nav_cal_to_image, best_image_wp = self.get_min_dist_and_wp(rover, best_cal_wp, imaging_wps)

                                     if best_image_wp is not None and min_nav_cal_to_image != float('inf'):
                                          # Cost to get image: navigate(curr->cal) + calibrate + navigate(cal->image) + take_image
                                          cost_get_image = min_nav_to_cal + 1 + min_nav_cal_to_image + 1
                                          # Cost to communicate: navigate(image->comm) + communicate
                                          cost_communicate = self.min_dist_to_comm[rover].get(best_image_wp, float('inf')) + 1
                                          if cost_communicate != float('inf'):
                                               min_take_image_and_comm_cost = min(min_take_image_and_comm_cost, cost_get_image + cost_communicate)

                    min_goal_cost = min_take_image_and_comm_cost

                total_cost += min_goal_cost

        # If total_cost is still float('inf'), it means at least one goal is unreachable.
        # Return infinity or a large number. Returning the float is fine.
        return total_cost

