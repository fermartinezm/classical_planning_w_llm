# Assuming heuristic_base is available in the environment
from heuristics.heuristic_base import Heuristic

from fnmatch import fnmatch
from collections import deque

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at rover1 waypoint1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure fact has at least as many parts as args, and then check pattern match
    return len(parts) >= len(args) and all(fnmatch(part, arg) for part, arg in zip(parts, args))

def bfs(graph, start_node):
    """Computes shortest distances from start_node to all reachable nodes in a graph."""
    distances = {node: float('inf') for node in graph}
    if start_node in graph: # Ensure start_node is actually in the graph keys
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

def compute_all_pairs_shortest_paths(graph):
    """Computes shortest paths between all pairs of nodes in a graph."""
    all_paths = {}
    # Collect all nodes that appear as keys or neighbors
    all_nodes = set(graph.keys())
    for neighbors in graph.values():
        all_nodes.update(neighbors)

    # Create a graph structure that includes all identified nodes,
    # even if they have no outgoing edges in the original graph.
    graph_with_all_nodes = {node: graph.get(node, set()) for node in all_nodes}

    for start_node in graph_with_all_nodes:
        all_paths[start_node] = bfs(graph_with_all_nodes, start_node)
    return all_paths


class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    Estimates the number of actions required to achieve all goal conditions.
    The heuristic sums the estimated costs for each unachieved goal independently.
    Costs include navigation (estimated by shortest path), sampling, imaging,
    calibrating, and communicating.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and static facts.
        Precomputes rover traversal graphs and shortest paths.
        """
        self.goals = task.goals
        static_facts = task.static

        # Data structures for static information
        self.lander_location = {} # lander -> waypoint
        self.rover_equipment = {} # rover -> set of {'soil', 'rock', 'imaging'}
        self.rover_stores = {} # rover -> store (assuming one store per rover)
        self.rover_cameras = {} # rover -> set of cameras on board
        self.camera_modes = {} # camera -> set of supported modes
        self.camera_calibration_target = {} # camera -> objective (calibration target)
        self.waypoint_visibility = {} # wp -> set of visible wps
        self.objective_visibility = {} # objective -> set of visible_from wps
        self.calibration_target_visibility = {} # objective (target) -> set of visible_from wps
        self.rover_traversal_graph = {} # rover -> {wp -> set of reachable wps}
        self.rover_shortest_paths = {} # rover -> {start_wp -> {end_wp -> dist}}
        self.communication_wps = set() # wps visible from any lander location

        # Collect all relevant objects and initial static relations
        all_waypoints = set()
        all_rovers = set()
        all_cameras = set()
        all_objectives = set()
        all_modes = set()
        all_landers = set()
        all_stores = set()

        for fact in static_facts:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == 'at_lander':
                lander, wp = parts[1], parts[2]
                self.lander_location[lander] = wp
                all_landers.add(lander)
                all_waypoints.add(wp)
            elif pred == 'equipped_for_soil_analysis' or pred == 'equipped_for_rock_analysis' or pred == 'equipped_for_imaging':
                all_rovers.add(parts[1])
            elif pred == 'store_of':
                store, rover = parts[1], parts[2]
                self.rover_stores[rover] = store
                all_stores.add(store)
                all_rovers.add(rover)
            elif pred == 'on_board':
                camera, rover = parts[1], parts[2]
                self.rover_cameras.setdefault(rover, set()).add(camera)
                all_cameras.add(camera)
                all_rovers.add(rover)
            elif pred == 'supports':
                camera, mode = parts[1], parts[2]
                self.camera_modes.setdefault(camera, set()).add(mode)
                all_cameras.add(camera)
                all_modes.add(mode)
            elif pred == 'calibration_target':
                camera, objective = parts[1], parts[2]
                self.camera_calibration_target[camera] = objective
                all_cameras.add(camera)
                all_objectives.add(objective)
            elif pred == 'visible':
                wp1, wp2 = parts[1], parts[2]
                all_waypoints.add(wp1)
                all_waypoints.add(wp2)
                self.waypoint_visibility.setdefault(wp1, set()).add(wp2)
            elif pred == 'visible_from':
                objective, wp = parts[1], parts[2]
                all_objectives.add(objective)
                all_waypoints.add(wp)
                self.objective_visibility.setdefault(objective, set()).add(wp)
            elif pred == 'can_traverse':
                rover, wp1, wp2 = parts[1], parts[2], parts[3]
                all_rovers.add(rover)
                all_waypoints.add(wp1)
                all_waypoints.add(wp2)
                self.rover_traversal_graph.setdefault(rover, {}).setdefault(wp1, set()).add(wp2)
                # Ensure all waypoints mentioned in can_traverse are nodes in the graph, even if no outgoing edges
                self.rover_traversal_graph[rover].setdefault(wp2, set())

        # Populate rover equipment sets (do this after identifying all rovers)
        for rover in all_rovers:
             self.rover_equipment.setdefault(rover, set()) # Ensure all rovers have an entry

        for fact in static_facts:
             parts = get_parts(fact)
             pred = parts[0]
             if pred == 'equipped_for_soil_analysis':
                 self.rover_equipment[parts[1]].add('soil')
             elif pred == 'equipped_for_rock_analysis':
                 self.rover_equipment[parts[1]].add('rock')
             elif pred == 'equipped_for_imaging':
                 self.rover_equipment[parts[1]].add('imaging')

        # Populate calibration target visibility based on objective visibility
        for camera, target_obj in self.camera_calibration_target.items():
             if target_obj in self.objective_visibility:
                  self.calibration_target_visibility[target_obj] = self.objective_visibility[target_obj]


        # Identify communication waypoints (visible from any lander location)
        lander_wps = set(self.lander_location.values())
        for lander_wp in lander_wps:
             if lander_wp in self.waypoint_visibility:
                 self.communication_wps.update(self.waypoint_visibility[lander_wp])
             # A rover at the lander waypoint can also communicate
             self.communication_wps.add(lander_wp)

        # Ensure communication_wps are not empty if there's a lander
        if not self.communication_wps and lander_wps:
             # This might happen if no waypoints are visible from the lander's location
             # or if the lander location itself is not in the visible graph.
             # As a fallback, just use the lander location(s) as communication points.
             self.communication_wps.update(lander_wps)


        # Compute shortest paths for each rover
        for rover in all_rovers:
            # Ensure all waypoints are considered nodes in the graph for BFS,
            # even if a specific rover cannot traverse to/from them.
            # This ensures the distance is correctly reported as infinity.
            graph_for_rover = {wp: set() for wp in all_waypoints}
            if rover in self.rover_traversal_graph:
                 for wp, neighbors in self.rover_traversal_graph[rover].items():
                      graph_for_rover[wp].update(neighbors)

            self.rover_shortest_paths[rover] = compute_all_pairs_shortest_paths(graph_for_rover)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Data structures for state information
        current_locations = {} # obj -> wp
        rover_contents = {} # rover -> set of (have_X_analysis wp) or (have_image obj mode) facts
        store_status = {} # store -> 'empty' or 'full'
        camera_calibration_status = {} # (camera, rover) -> True/False
        soil_samples_at = set() # wps with soil samples
        rock_samples_at = set() # wps with rock samples
        communicated_data = set() # communicated facts

        # Parse state facts
        for fact in state:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == 'at':
                obj, loc = parts[1], parts[2]
                current_locations[obj] = loc
            elif pred == 'have_soil_analysis':
                rover, wp = parts[1], parts[2]
                rover_contents.setdefault(rover, set()).add(fact)
            elif pred == 'have_rock_analysis':
                rover, wp = parts[1], parts[2]
                rover_contents.setdefault(rover, set()).add(fact)
            elif pred == 'have_image':
                rover, obj, mode = parts[1], parts[2], parts[3]
                rover_contents.setdefault(rover, set()).add(fact)
            elif pred == 'empty':
                store = parts[1]
                store_status[store] = 'empty'
            elif pred == 'full':
                store = parts[1]
                store_status[store] = 'full'
            elif pred == 'calibrated':
                camera, rover = parts[1], parts[2]
                camera_calibration_status[(camera, rover)] = True
            elif pred == 'at_soil_sample':
                wp = parts[1]
                soil_samples_at.add(wp)
            elif pred == 'at_rock_sample':
                wp = parts[1]
                rock_samples_at.add(wp)
            elif pred == 'communicated_soil_data':
                communicated_data.add(fact)
            elif pred == 'communicated_rock_data':
                communicated_data.add(fact)
            elif pred == 'communicated_image_data':
                communicated_data.add(fact)

        # Default calibration status to False if not explicitly True in state
        # We only care about cameras on rovers that exist and are equipped for imaging
        for rover in self.rover_cameras:
             if 'imaging' in self.rover_equipment.get(rover, set()):
                  for camera in self.rover_cameras[rover]:
                       if (camera, rover) not in camera_calibration_status:
                            camera_calibration_status[(camera, rover)] = False

        # Default store status to full if not explicitly empty (safer assumption)
        for rover, store in self.rover_stores.items():
             if store not in store_status:
                  store_status[store] = 'full'


        total_cost = 0

        # Iterate through goals and sum costs for unachieved ones
        for goal in self.goals:
            if goal in communicated_data:
                continue # Goal already achieved

            # Parse the goal fact
            parts = get_parts(goal)
            pred = parts[0]

            if pred == 'communicated_soil_data':
                wp_sample = parts[1]
                goal_cost = 1 # Cost for the final communicate action

                # Check if sample data is already collected by any rover
                have_data = False
                rover_with_data = None
                for rover, contents in rover_contents.items():
                    if f'(have_soil_analysis {rover} {wp_sample})' in contents:
                        have_data = True
                        rover_with_data = rover # Found a rover with the data
                        break

                if not have_data:
                    # Need to sample
                    # Check if sample is still available at the waypoint
                    sample_available_at_wp = f'(at_soil_sample {wp_sample})' in soil_samples_at

                    if not sample_available_at_wp:
                         # Goal requires communicating data from a sample at wp_sample,
                         # but the sample is gone from the waypoint and no rover has the data.
                         # This goal is unreachable.
                         return float('inf')

                    goal_cost += 1 # Cost for sample_soil action

                    # Find an equipped rover and min cost to navigate to sample location
                    min_nav_to_sample = float('inf')
                    best_sampler_rover = None
                    for rover in self.rover_equipment:
                        if 'soil' in self.rover_equipment[rover] and rover in current_locations:
                            current_wp = current_locations[rover]
                            if rover in self.rover_shortest_paths and current_wp in self.rover_shortest_paths[rover] and wp_sample in self.rover_shortest_paths[rover][current_wp]:
                                nav_cost = self.rover_shortest_paths[rover][current_wp][wp_sample]
                                if nav_cost < min_nav_to_sample:
                                    min_nav_to_sample = nav_cost
                                    best_sampler_rover = rover

                    if min_nav_to_sample == float('inf'):
                         # Cannot reach sample location with any equipped rover
                         return float('inf')

                    goal_cost += min_nav_to_sample

                    # Check if the chosen sampler rover needs to drop a sample first
                    # Assuming one store per rover.
                    if best_sampler_rover and self.rover_stores.get(best_sampler_rover) in store_status and store_status.get(self.rover_stores.get(best_sampler_rover)) == 'full':
                         goal_cost += 1 # Cost for drop action

                    # The sampler rover will have the data
                    rover_with_data = best_sampler_rover

                # Need to navigate the rover with data to a communication point
                if rover_with_data and rover_with_data in current_locations:
                    current_wp = current_locations[rover_with_data]
                    min_nav_to_comm = float('inf')
                    if rover_with_data in self.rover_shortest_paths and current_wp in self.rover_shortest_paths[rover_with_data]:
                        for comm_wp in self.communication_wps:
                            if comm_wp in self.rover_shortest_paths[rover_with_data][current_wp]:
                                min_nav_to_comm = min(min_nav_to_comm, self.rover_shortest_paths[rover_with_data][current_wp][comm_wp])

                    if min_nav_to_comm == float('inf'):
                         # Cannot reach any communication point
                         return float('inf')

                    goal_cost += min_nav_to_comm
                elif not rover_with_data:
                     # Should not happen if logic is correct, but indicates unreachability
                     return float('inf')

                total_cost += goal_cost

            elif pred == 'communicated_rock_data':
                 wp_sample = parts[1]
                 goal_cost = 1 # Cost for the final communicate action

                 have_data = False
                 rover_with_data = None
                 for rover, contents in rover_contents.items():
                     if f'(have_rock_analysis {rover} {wp_sample})' in contents:
                         have_data = True
                         rover_with_data = rover
                         break

                 if not have_data:
                     sample_available_at_wp = f'(at_rock_sample {wp_sample})' in rock_samples_at
                     if not sample_available_at_wp:
                          return float('inf')

                     goal_cost += 1 # Cost for sample_rock action

                     min_nav_to_sample = float('inf')
                     best_sampler_rover = None
                     for rover in self.rover_equipment:
                         if 'rock' in self.rover_equipment[rover] and rover in current_locations:
                             current_wp = current_locations[rover]
                             if rover in self.rover_shortest_paths and current_wp in self.rover_shortest_paths[rover] and wp_sample in self.rover_shortest_paths[rover][current_wp]:
                                 nav_cost = self.rover_shortest_paths[rover][current_wp][wp_sample]
                                 if nav_cost < min_nav_to_sample:
                                     min_nav_to_sample = nav_cost
                                     best_sampler_rover = rover

                     if min_nav_to_sample == float('inf'):
                          return float('inf')

                     goal_cost += min_nav_to_sample

                     if best_sampler_rover and self.rover_stores.get(best_sampler_rover) in store_status and store_status.get(self.rover_stores.get(best_sampler_rover)) == 'full':
                          goal_cost += 1 # Cost for drop action

                     rover_with_data = best_sampler_rover

                 if rover_with_data and rover_with_data in current_locations:
                     current_wp = current_locations[rover_with_data]
                     min_nav_to_comm = float('inf')
                     if rover_with_data in self.rover_shortest_paths and current_wp in self.rover_shortest_paths[rover_with_data]:
                         for comm_wp in self.communication_wps:
                             if comm_wp in self.rover_shortest_paths[rover_with_data][current_wp]:
                                 min_nav_to_comm = min(min_nav_to_comm, self.rover_shortest_paths[rover_with_data][current_wp][comm_wp])

                     if min_nav_to_comm == float('inf'):
                          return float('inf')

                     goal_cost += min_nav_to_comm
                 elif not rover_with_data:
                      return float('inf')

                 total_cost += goal_cost

            elif pred == 'communicated_image_data':
                 objective = parts[1]
                 mode = parts[2]
                 goal_cost = 1 # Cost for the final communicate action

                 have_data = False
                 rover_with_data = None
                 for rover, contents in rover_contents.items():
                     if f'(have_image {rover} {objective} {mode})' in contents:
                         have_data = True
                         rover_with_data = rover
                         break

                 if not have_data:
                     goal_cost += 1 # Cost for take_image action

                     min_nav_to_image_wp = float('inf')
                     best_imager_rover = None
                     best_camera = None
                     # image_wp = None # Not strictly needed for cost calculation

                     suitable_rovers = []
                     for rover in self.rover_equipment:
                          if 'imaging' in self.rover_equipment.get(rover, set()):
                               for camera in self.rover_cameras.get(rover, set()):
                                    if mode in self.camera_modes.get(camera, set()):
                                         suitable_rovers.append((rover, camera))

                     if not suitable_rovers:
                          # No rover can take this image in this mode
                          return float('inf')

                     possible_image_wps = self.objective_visibility.get(objective, set())
                     if not possible_image_wps:
                          # Objective not visible from anywhere
                          return float('inf')

                     # Find the best rover/camera and the best waypoint to take the image from
                     for rover, camera in suitable_rovers:
                          if rover in current_locations:
                               current_wp = current_locations[rover]
                               if rover in self.rover_shortest_paths and current_wp in self.rover_shortest_paths[rover]:
                                    for img_wp in possible_image_wps:
                                         if img_wp in self.rover_shortest_paths[rover][current_wp]:
                                              nav_cost = self.rover_shortest_paths[rover][current_wp][img_wp]
                                              if nav_cost < min_nav_to_image_wp:
                                                   min_nav_to_image_wp = nav_cost
                                                   best_imager_rover = rover
                                                   best_camera = camera
                                                   # image_wp = img_wp # Store the chosen image waypoint

                     if min_nav_to_image_wp == float('inf'):
                          # Cannot reach any image waypoint for this objective with any suitable rover
                          return float('inf')

                     goal_cost += min_nav_to_image_wp

                     # Check if the chosen camera needs calibration *in the current state*
                     if best_imager_rover and best_camera and not camera_calibration_status.get((best_camera, best_imager_rover), False):
                          goal_cost += 1 # Cost for calibrate action

                          cal_target = self.camera_calibration_target.get(best_camera)
                          if not cal_target:
                               # Camera has no calibration target defined
                               return float('inf')

                          possible_cal_wps = self.calibration_target_visibility.get(cal_target, set())
                          if not possible_cal_wps:
                               # Calibration target not visible from anywhere
                               return float('inf')

                          min_nav_to_cal_wp = float('inf')
                          if best_imager_rover in current_locations: # Use the same rover
                               current_wp = current_locations[best_imager_rover]
                               if best_imager_rover in self.rover_shortest_paths and current_wp in self.rover_shortest_paths[best_imager_rover]:
                                    for cal_wp in possible_cal_wps:
                                         if cal_wp in self.rover_shortest_paths[best_imager_rover][current_wp]:
                                              min_nav_to_cal_wp = min(min_nav_to_cal_wp, self.rover_shortest_paths[best_imager_rover][current_wp][cal_wp])

                          if min_nav_to_cal_wp == float('inf'):
                               # Cannot reach any calibration waypoint for this camera/target with the chosen rover
                               return float('inf')

                          goal_cost += min_nav_to_cal_wp

                     rover_with_data = best_imager_rover

                 # Need to navigate the rover with data to a communication point
                 if rover_with_data and rover_with_data in current_locations:
                     current_wp = current_locations[rover_with_data]
                     min_nav_to_comm = float('inf')
                     if rover_with_data in self.rover_shortest_paths and current_wp in self.rover_shortest_paths[rover_with_data]:
                         for comm_wp in self.communication_wps:
                             if comm_wp in self.rover_shortest_paths[rover_with_data][current_wp]:
                                 min_nav_to_comm = min(min_nav_to_comm, self.rover_shortest_paths[rover_with_data][current_wp][comm_wp])

                     if min_nav_to_comm == float('inf'):
                          # Cannot reach any communication point
                          return float('inf')

                     goal_cost += min_nav_to_comm
                 elif not rover_with_data:
                      # Should not happen if logic is correct, but indicates unreachability
                      return float('inf')

                 total_cost += goal_cost


        return total_cost
