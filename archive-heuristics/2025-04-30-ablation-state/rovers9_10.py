from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import math
from collections import defaultdict, deque

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    if not fact or fact[0] != '(' or fact[-1] != ')':
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
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def bfs(graph, start_node):
    """
    Performs Breadth-First Search on a graph to find shortest distances
    from a start node to all reachable nodes.

    Args:
        graph: Adjacency list representation {node: {neighbor1, neighbor2, ...}}
        start_node: The node to start the BFS from.

    Returns:
        A dictionary {node: distance} containing shortest distances.
        Distance is math.inf if node is unreachable.
    """
    distances = {node: math.inf for node in graph}
    if start_node not in graph:
         # Start node is not in the graph (e.g., rover cannot traverse from its location)
         return distances

    distances[start_node] = 0
    queue = deque([start_node])

    while queue:
        current_node = queue.popleft()

        if current_node in graph:
            for neighbor in graph[current_node]:
                if distances[neighbor] == math.inf:
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)
    return distances

class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve all
    unmet goal conditions. It sums the estimated costs for each individual
    unachieved goal fact. The cost for a goal fact is estimated by finding
    the minimum cost sequence of actions (sampling/imaging, calibrating,
    communicating) and necessary navigation steps for the most suitable rover.
    Navigation costs are precomputed shortest path distances on the traversal
    graph for each rover.

    # Assumptions
    - Each unachieved goal fact contributes independently to the heuristic cost.
    - For sampling goals, a rover needs to navigate to the sample location,
      sample (potentially drop first), navigate to a communication location,
      and communicate.
    - For imaging goals, a rover needs to navigate to a calibration location
      (if needed), calibrate, navigate to an image location, take the image,
      navigate to a communication location, and communicate. The navigation
      steps for imaging goals are chained (current -> cal_wp -> img_wp -> comm_wp).
    - Navigation costs are shortest path distances on the rover's traversal graph.
    - The cost of an action (sample, drop, calibrate, take_image, communicate) is 1.
    - The cost of navigation is the number of `navigate` actions.
    - If a goal requires a resource (equipped rover, visible waypoint) that
      doesn't exist or is unreachable, the heuristic returns infinity.

    # Heuristic Initialization
    - Parses static facts to identify lander location, rover capabilities,
      camera properties, objective visibility, etc.
    - Builds the traversal graph for each rover based on `can_traverse` facts.
    - Identifies communication waypoints (visible from the lander).
    - Precomputes all-pairs shortest path distances for each rover on its
      traversal graph using BFS.
    - Precomputes minimum distances for each rover from any waypoint to any
      communication waypoint.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all goal facts that are not currently true in the state.
    2. Initialize total heuristic cost to 0.
    3. For each unachieved goal fact:
        a. Determine the type of goal (soil, rock, or image communication).
        b. If it's a soil/rock communication goal for waypoint `W`:
            i. Check if any rover already has the sample from `W`.
            ii. If yes, find such a rover `R`. The cost is navigation from `R`'s
               current location to the nearest communication waypoint + 1 (communicate).
            iii. If no, find the minimum cost over all rovers `R` equipped for
                the task: navigation from `R`'s current location to `W` + 1 (sample)
                + (1 if `R`'s store is full) + navigation from `W` to the nearest
                communication waypoint + 1 (communicate).
            iv. Add the minimum cost found to the total heuristic cost. If no
                suitable rover or path exists, return infinity.
        c. If it's an image communication goal for objective `O` and mode `M`:
            i. Check if any rover already has the image of `O` in `M`.
            ii. If yes, find such a rover `R`. The cost is navigation from `R`'s
               current location to the nearest communication waypoint + 1 (communicate).
            iii. If no, find the minimum cost over all rovers `R` equipped for
                imaging with a camera `I` supporting `M`:
                - Cost starts at 0.
                - Track the rover's assumed current location for navigation chaining, starting with its actual current location.
                - If camera `I` is not calibrated: add 1 (calibrate) + navigation
                  from the rover's assumed current location to the nearest calibration waypoint
                  for `I`. Update the rover's assumed current location to that calibration waypoint.
                - Add 1 (take_image) + navigation from the rover's assumed current location
                  (potentially updated after calibration step) to the nearest
                  image waypoint for `O`. Update the rover's assumed current location to that
                  image waypoint.
                - Add 1 (communicate) + navigation from the rover's assumed current location
                  (potentially updated after image step) to the nearest
                  communication waypoint.
                - Add the minimum cost found over all suitable rovers/cameras
                  to the total heuristic cost. If no suitable rover/camera or
                  path exists, return infinity.
    4. Return the total heuristic cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static
        objects = task.objects

        self.rovers = [obj for obj in objects if obj.startswith('rover')]
        self.waypoints = [obj for obj in objects if obj.startswith('waypoint')]
        self.landers = [obj for obj in objects if obj.startswith('lander')]
        self.stores = [obj for obj in objects if obj.startswith('store')]
        self.cameras = [obj for obj in objects if obj.startswith('camera')]
        self.modes = [obj for obj in objects if obj.startswith('mode')]
        self.objectives = [obj for obj in objects if obj.startswith('objective')]

        # --- Static Information ---
        self.lander_location = None
        self.rover_equipment = defaultdict(set) # rover -> {soil, rock, imaging}
        self.rover_stores = {} # rover -> store
        self.rover_cameras = defaultdict(list) # rover -> [camera1, camera2, ...]
        self.camera_modes = defaultdict(set) # camera -> {mode1, mode2, ...}
        self.camera_calibration_target = {} # camera -> objective
        self.objective_visible_from = defaultdict(set) # objective -> {waypoint1, waypoint2, ...}
        self.calibration_target_visible_from = defaultdict(set) # objective (target) -> {waypoint1, waypoint2, ...}
        self.rover_traversal_graph = defaultdict(lambda: defaultdict(set)) # rover -> {wp -> {neighbor_wp}}
        self.waypoint_visibility = defaultdict(set) # wp -> {visible_wp}

        # Parse static facts
        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue

            pred = parts[0]
            if pred == "at_lander":
                self.lander_location = parts[2]
            elif pred == "equipped_for_soil_analysis":
                self.rover_equipment[parts[1]].add("soil")
            elif pred == "equipped_for_rock_analysis":
                self.rover_equipment[parts[1]].add("rock")
            elif pred == "equipped_for_imaging":
                self.rover_equipment[parts[1]].add("imaging")
            elif pred == "store_of":
                self.rover_stores[parts[2]] = parts[1] # rover -> store
            elif pred == "on_board":
                self.rover_cameras[parts[2]].append(parts[1]) # rover -> camera
            elif pred == "supports":
                self.camera_modes[parts[1]].add(parts[2]) # camera -> mode
            elif pred == "calibration_target":
                self.camera_calibration_target[parts[1]] = parts[2] # camera -> objective
            elif pred == "visible_from":
                self.objective_visible_from[parts[1]].add(parts[2]) # objective -> waypoint
            elif pred == "can_traverse":
                rover, wp1, wp2 = parts[1:]
                self.rover_traversal_graph[rover][wp1].add(wp2)
            elif pred == "visible":
                 wp1, wp2 = parts[1:]
                 self.waypoint_visibility[wp1].add(wp2)

        # Populate calibration_target_visible_from based on objective_visible_from
        for camera, target_objective in self.camera_calibration_target.items():
             if target_objective in self.objective_visible_from:
                 self.calibration_target_visible_from[target_objective].update(
                     self.objective_visible_from[target_objective]
                 )

        # Identify communication waypoints (visible from lander)
        self.comm_waypoints = set()
        if self.lander_location and self.lander_location in self.waypoint_visibility:
             self.comm_waypoints.update(self.waypoint_visibility[self.lander_location])

        # Precompute all-pairs shortest paths for each rover
        self.dist = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: math.inf))) # rover -> {wp1 -> {wp2 -> distance}}
        for rover in self.rovers:
            # Consider all waypoints as potential start nodes for BFS
            graph = self.rover_traversal_graph.get(rover, {})
            for start_wp in self.waypoints:
                 distances = bfs(graph, start_wp)
                 for end_wp, d in distances.items():
                     self.dist[rover][start_wp][end_wp] = d

        # Precompute minimum distance from any waypoint to any communication waypoint for each rover
        self.dist_to_comm = defaultdict(lambda: defaultdict(lambda: math.inf)) # rover -> {wp -> min_distance}
        if self.comm_waypoints: # Only compute if there are communication waypoints
            for rover in self.rovers:
                for start_wp in self.waypoints:
                    min_d = math.inf
                    # Ensure start_wp is a valid key in the distance map for this rover
                    if rover in self.dist and start_wp in self.dist[rover]:
                        for comm_wp in self.comm_waypoints:
                            # Ensure comm_wp is a valid key in the distance map from start_wp
                            if comm_wp in self.dist[rover][start_wp]:
                                min_d = min(min_d, self.dist[rover][start_wp][comm_wp])
                    self.dist_to_comm[rover][start_wp] = min_d

    def min_nav_cost(self, rover, start_wp, target_waypoints_set):
        """Helper to find min nav cost from start_wp to any in target_waypoints_set."""
        if not target_waypoints_set: return math.inf
        min_d = math.inf
        # Ensure start_wp is a valid key in the distance map for this rover
        if rover not in self.dist or start_wp not in self.dist[rover]:
             return math.inf

        for target_wp in target_waypoints_set:
            # Ensure target_wp is a valid key in the distance map from start_wp
            if target_wp in self.dist[rover][start_wp]:
                min_d = min(min_d, self.dist[rover][start_wp][target_wp])
        return min_d

    def find_closest_wp(self, rover, start_wp, target_waypoints_set):
        """Helper to find the waypoint in target_waypoints_set closest to start_wp."""
        if not target_waypoints_set: return None
        min_d = math.inf
        closest_wp = None
        # Ensure start_wp is a valid key in the distance map for this rover
        if rover not in self.dist or start_wp not in self.dist[rover]:
             return None

        for target_wp in target_waypoints_set:
            # Ensure target_wp is a valid key in the distance map from start_wp
            if target_wp in self.dist[rover][start_wp]:
                dist = self.dist[rover][start_wp][target_wp]
                if dist < min_d:
                    min_d = dist
                    closest_wp = target_wp
        return closest_wp # Returns None if unreachable

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # --- State Information ---
        rover_locations = {} # rover -> waypoint
        store_status = {} # store -> "empty" or "full"
        rover_soil_samples = defaultdict(set) # rover -> {waypoint}
        rover_rock_samples = defaultdict(set) # rover -> {waypoint}
        rover_images = defaultdict(set) # rover -> {(objective, mode)}
        camera_calibrated = set() # {(camera, rover)}
        communicated_soil = set() # {waypoint}
        communicated_rock = set() # {waypoint}
        communicated_image = set() # {(objective, mode)}
        soil_samples_remaining = set() # {waypoint}
        rock_samples_remaining = set() # {waypoint}

        # Parse state facts
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            pred = parts[0]
            if pred == "at":
                obj, loc = parts[1:]
                if obj in self.rovers:
                    rover_locations[obj] = loc
            elif pred == "empty":
                store_status[parts[1]] = "empty"
            elif pred == "full":
                store_status[parts[1]] = "full"
            elif pred == "have_soil_analysis":
                rover, wp = parts[1:]
                rover_soil_samples[rover].add(wp)
            elif pred == "have_rock_analysis":
                rover, wp = parts[1:]
                rover_rock_samples[rover].add(wp)
            elif pred == "have_image":
                rover, obj, mode = parts[1:]
                rover_images[rover].add((obj, mode))
            elif pred == "calibrated":
                camera, rover = parts[1:]
                camera_calibrated.add((camera, rover))
            elif pred == "communicated_soil_data":
                communicated_soil.add(parts[1])
            elif pred == "communicated_rock_data":
                communicated_rock.add(parts[1])
            elif pred == "communicated_image_data":
                communicated_image.add((parts[1], parts[2]))
            elif pred == "at_soil_sample":
                 soil_samples_remaining.add(parts[1])
            elif pred == "at_rock_sample":
                 rock_samples_remaining.add(parts[1])

        total_cost = 0

        # Iterate through goals and sum costs for unachieved ones
        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue

            pred = parts[0]

            if pred == "communicated_soil_data":
                wp = parts[1]
                if wp in communicated_soil:
                    continue # Goal already achieved

                min_task_cost = math.inf
                rover_has_sample = None
                for rover in self.rovers:
                    if wp in rover_soil_samples[rover]:
                        rover_has_sample = rover
                        break

                if rover_has_sample is not None:
                    # Need to communicate existing sample
                    r = rover_has_sample
                    current_loc = rover_locations.get(r)
                    if current_loc is None: continue # Rover location unknown? Should not happen.

                    nav_to_comm = self.dist_to_comm[r][current_loc]
                    if nav_to_comm == math.inf: continue # Cannot reach comm location

                    task_cost = nav_to_comm + 1 # navigate + communicate
                    min_task_cost = task_cost

                else:
                    # Need to sample and communicate
                    # Check if sampling is even possible (sample must be at the waypoint)
                    if wp not in soil_samples_remaining:
                         # Sample is gone and no rover has it -> impossible
                         return math.inf # Goal is impossible

                    for rover in self.rovers:
                         if "soil" in self.rover_equipment[rover]:
                             current_loc = rover_locations.get(rover)
                             if current_loc is None: continue

                             task_cost = 0
                             current_loc_for_nav = current_loc

                             # Cost to sample
                             nav_to_sample = self.dist[rover][current_loc_for_nav][wp]
                             if nav_to_sample == math.inf: continue # Rover cannot reach sample location
                             task_cost += nav_to_sample + 1 # navigate + sample_soil
                             current_loc_for_nav = wp # Rover is now at sample location

                             store = self.rover_stores.get(rover)
                             if store and store_status.get(store) == "full":
                                 task_cost += 1 # drop action

                             # Cost to communicate (from sample location)
                             nav_to_comm = self.dist_to_comm[rover][current_loc_for_nav]
                             if nav_to_comm == math.inf: continue # Cannot reach comm location from sample location
                             task_cost += nav_to_comm + 1 # navigate + communicate_soil_data

                             min_task_cost = min(min_task_cost, task_cost)

                if min_task_cost == math.inf:
                    return math.inf # Goal is unreachable

                total_cost += min_task_cost

            elif pred == "communicated_rock_data":
                wp = parts[1]
                if wp in communicated_rock:
                    continue # Goal already achieved

                min_task_cost = math.inf
                rover_has_sample = None
                for rover in self.rovers:
                    if wp in rover_rock_samples[rover]:
                        rover_has_sample = rover
                        break

                if rover_has_sample is not None:
                    # Need to communicate existing sample
                    r = rover_has_sample
                    current_loc = rover_locations.get(r)
                    if current_loc is None: continue

                    nav_to_comm = self.dist_to_comm[r][current_loc]
                    if nav_to_comm == math.inf: continue

                    task_cost = nav_to_comm + 1 # navigate + communicate
                    min_task_cost = min(min_task_cost, task_cost)

                else:
                    # Need to sample and communicate
                    # Check if sampling is even possible
                    if wp not in rock_samples_remaining:
                         # Sample is gone and no rover has it -> impossible
                         return math.inf # Goal is impossible

                    for rover in self.rovers:
                         if "rock" in self.rover_equipment[rover]:
                             current_loc = rover_locations.get(rover)
                             if current_loc is None: continue

                             task_cost = 0
                             current_loc_for_nav = current_loc

                             # Cost to sample
                             nav_to_sample = self.dist[rover][current_loc_for_nav][wp]
                             if nav_to_sample == math.inf: continue
                             task_cost += nav_to_sample + 1 # navigate + sample_rock
                             current_loc_for_nav = wp # Rover is now at sample location

                             store = self.rover_stores.get(rover)
                             if store and store_status.get(store) == "full":
                                 task_cost += 1 # drop action

                             # Cost to communicate (from sample location)
                             nav_to_comm = self.dist_to_comm[rover][current_loc_for_nav]
                             if nav_to_comm == math.inf: continue
                             task_cost += nav_to_comm + 1 # navigate + communicate_rock_data

                             min_task_cost = min(min_task_cost, task_cost)

                if min_task_cost == math.inf:
                    return math.inf # Goal is unreachable

                total_cost += min_task_cost

            elif pred == "communicated_image_data":
                obj, mode = parts[1:]
                if (obj, mode) in communicated_image:
                    continue # Goal already achieved

                min_task_cost = math.inf
                rover_has_image = None
                for rover in self.rovers:
                    if (obj, mode) in rover_images[rover]:
                        rover_has_image = rover
                        break

                if rover_has_image is not None:
                    # Need to communicate existing image
                    r = rover_has_image
                    current_loc = rover_locations.get(r)
                    if current_loc is None: continue

                    nav_to_comm = self.dist_to_comm[r][current_loc]
                    if nav_to_comm == math.inf: continue

                    task_cost = nav_to_comm + 1 # navigate + communicate
                    min_task_cost = min(min_task_cost, task_cost)

                else:
                    # Need to take image and communicate
                    img_waypoints = self.objective_visible_from.get(obj)
                    if not img_waypoints:
                         # No waypoint to view objective from -> impossible
                         return math.inf

                    for rover in self.rovers:
                        if "imaging" in self.rover_equipment[rover]:
                            for camera in self.rover_cameras[rover]:
                                if mode in self.camera_modes[camera]:
                                    current_loc = rover_locations.get(rover)
                                    if current_loc is None: continue

                                    task_cost = 0
                                    current_loc_for_nav = current_loc # Track location during sequence

                                    # Cost to calibrate (if needed)
                                    if (camera, rover) not in camera_calibrated:
                                        cal_target = self.camera_calibration_target.get(camera)
                                        if cal_target is None: continue # Camera has no calibration target

                                        cal_waypoints = self.calibration_target_visible_from.get(cal_target)
                                        if not cal_waypoints: continue # No waypoint to calibrate from

                                        nav_to_cal = self.min_nav_cost(rover, current_loc_for_nav, cal_waypoints)
                                        if nav_to_cal == math.inf: continue

                                        task_cost += nav_to_cal + 1 # navigate + calibrate
                                        # Update location after calibration trip
                                        current_loc_for_nav = self.find_closest_wp(rover, current_loc_for_nav, cal_waypoints)
                                        if current_loc_for_nav is None: continue # Should not happen if nav_to_cal was finite


                                    # Cost to take image
                                    nav_to_img = self.min_nav_cost(rover, current_loc_for_nav, img_waypoints)
                                    if nav_to_img == math.inf: continue

                                    task_cost += nav_to_img + 1 # navigate + take_image
                                    # Update location after image trip
                                    current_loc_for_nav = self.find_closest_wp(rover, current_loc_for_nav, img_waypoints)
                                    if current_loc_for_nav is None: continue # Should not happen if nav_to_img was finite

                                    # Cost to communicate (from image location)
                                    nav_to_comm = self.dist_to_comm[rover][current_loc_for_nav]
                                    if nav_to_comm == math.inf: continue

                                    task_cost += nav_to_comm + 1 # navigate + communicate_image_data

                                    min_task_cost = min(min_task_cost, task_cost)

                if min_task_cost == math.inf:
                    return math.inf # Goal is unreachable

                total_cost += min_task_cost

        return total_cost
