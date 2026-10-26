from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle empty fact string or invalid format gracefully, though PDDL facts are structured.
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
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

def nav_set_to_set(start_wp_set, target_wp_set, rover, can_traverse_graph):
    """
    Finds the shortest path distance for a rover from any waypoint in start_wp_set
    to any waypoint in target_wp_set using BFS.
    Returns float('inf') if no path exists or sets are invalid.
    """
    if not start_wp_set or not target_wp_set:
        return float('inf')

    # Check for overlap
    if start_wp_set.intersection(target_wp_set):
        return 0

    graph = can_traverse_graph.get(rover)
    if not graph:
        return float('inf') # Rover cannot move

    # Start BFS from all valid waypoints in the start set that exist in the graph
    queue = deque([(wp, 0) for wp in start_wp_set if wp in graph])
    visited = set(start_wp_set)

    while queue:
        current_wp, dist = queue.popleft()

        if current_wp in target_wp_set:
            return dist

        if current_wp in graph:
            for neighbor_wp in graph[current_wp]:
                if neighbor_wp not in visited:
                    visited.add(neighbor_wp)
                    queue.append((neighbor_wp, dist + 1))

    return float('inf') # No waypoint in target_wp_set is reachable from start_wp_set


class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the total number of actions required to achieve all
    uncommunicated goal predicates (soil data, rock data, image data). It sums
    the estimated minimum cost for each unachieved goal independently, ignoring
    potential synergies or conflicts between goals.

    # Assumptions
    - The cost of each action (navigate, sample, drop, calibrate, take_image, communicate) is 1.
    - Navigation cost between waypoints is the shortest path distance for the specific rover.
    - The heuristic assumes that if a goal is not yet communicated, the required data must either be acquired or is already held by a rover.
    - For sampling goals, it checks if a sample still exists at the waypoint in the current state.
    - For image goals, it considers the steps of calibration, taking the image, and communication.
    - The heuristic uses a relaxed model where a rover can navigate optimally between sets of relevant waypoints (e.g., from current location to any calibration waypoint, from any calibration waypoint to any image waypoint, from any image waypoint to any communication waypoint).
    - Unreachable goals contribute infinity to the heuristic value.

    # Heuristic Initialization
    The constructor parses the static facts from the task definition to build data structures representing:
    - The lander's location.
    - Waypoints visible from the lander (communication waypoints).
    - The navigation graph for each rover based on 'can_traverse'.
    - Which rovers are equipped for soil, rock, and imaging.
    - Store ownership by rovers.
    - Camera capabilities ('supports' modes).
    - Which cameras are on which rovers.
    - Calibration targets for cameras.
    - Waypoints from which objectives are visible (for imaging and calibration).
    It also parses the goal facts to identify the specific soil, rock, and image data that need to be communicated.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Parse the current state to determine:
       - The current location of each rover.
       - Which rovers have which soil/rock data or images.
       - Which stores are full.
       - Which cameras are calibrated.
       - Which data has already been communicated (goal predicates met).
       - Which soil/rock samples still exist at waypoints.
    2. Initialize the total heuristic value `h` to 0.
    3. For each soil data goal (waypoint `W`) that is not yet communicated:
       - Find the minimum cost among all equipped rovers `R`.
       - If `R` already has `(have_soil_analysis R W)`: Cost = Nav(current_loc(R), nearest_comm_wp) + 1 (communicate).
       - If `R` does not have the data:
         - Check if `(at_soil_sample W)` is true in the current state. If not, this goal is impossible unless data is already held (handled above).
         - Cost = Nav(current_loc(R), W) + (1 if R's store is full) + 1 (sample) + Nav(W, nearest_comm_wp) + 1 (communicate).
       - Add the minimum cost found for this goal to `h`.
    4. For each rock data goal (waypoint `W`) that is not yet communicated:
       - Follow the same logic as for soil data goals, substituting rock predicates and equipped rovers.
       - Add the minimum cost found for this goal to `h`.
    5. For each image data goal (objective `O`, mode `M`) that is not yet communicated:
       - Find the minimum cost among all equipped rovers `R` with a camera `I` supporting mode `M` on board.
       - If `R` already has `(have_image R O M)`: Cost = Nav(current_loc(R), nearest_comm_wp) + 1 (communicate).
       - If `R` does not have the image:
         - Find calibration target `T` for camera `I`. Find calibration waypoints `Cal_WPs` (visible from `T`). Find image waypoints `Img_WPs` (visible from `O`).
         - If `Cal_WPs` or `Img_WPs` is empty, this goal is impossible.
         - Cost = Nav(current_loc(R), any_wp_in_Cal_WPs) + 1 (calibrate) + Nav(any_wp_in_Cal_WPs, any_wp_in_Img_WPs) + 1 (take_image) + Nav(any_wp_in_Img_WPs, any_wp_in_Comm_WPs) + 1 (communicate). The navigation costs between sets are the shortest path distances between the closest points in those sets, calculated using BFS starting from all points in the source set.
       - Add the minimum cost found for this goal (over all suitable R/I and waypoint sets) to `h`.
    6. If any goal's minimum cost was infinite, the total heuristic `h` will be infinite.
    7. Return `h`.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals

        self.lander_wp = None
        self.comm_wps = set()
        self.can_traverse_graph = {} # {rover: {wp: set(neighbor_wps)}}
        self.equipped_soil = set()
        self.equipped_rock = set()
        self.equipped_imaging = set()
        self.store_to_rover = {} # {store: rover}
        self.rover_to_store = {} # {rover: store}
        self.camera_supports_map = {} # {camera: set(modes)}
        self.camera_on_board_map = {} # {camera: rover}
        self.rover_cameras_map = {} # {rover: set(cameras)}
        self.camera_cal_target_map = {} # {camera: objective}
        self.objective_visible_from_map = {} # {objective: set(wps)}

        # Store visible facts temporarily to build comm_wps after finding lander_wp
        visible_facts_list = []

        # Parse static facts
        for fact in task.static:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts
            pred = parts[0]
            if pred == "at_lander":
                if len(parts) > 2: self.lander_wp = parts[2]
            elif pred == "can_traverse":
                if len(parts) > 3:
                    rover, wp1, wp2 = parts[1], parts[2], parts[3]
                    self.can_traverse_graph.setdefault(rover, {}).setdefault(wp1, set()).add(wp2)
            elif pred == "equipped_for_soil_analysis":
                if len(parts) > 1: self.equipped_soil.add(parts[1])
            elif pred == "equipped_for_rock_analysis":
                if len(parts) > 1: self.equipped_rock.add(parts[1])
            elif pred == "equipped_for_imaging":
                if len(parts) > 1: self.equipped_imaging.add(parts[1])
            elif pred == "store_of":
                if len(parts) > 2:
                    store, rover = parts[1], parts[2]
                    self.store_to_rover[store] = rover
                    self.rover_to_store[rover] = store
            elif pred == "supports":
                if len(parts) > 2:
                    camera, mode = parts[1], parts[2]
                    self.camera_supports_map.setdefault(camera, set()).add(mode)
            elif pred == "visible":
                 if len(parts) > 2: visible_facts_list.append((parts[1], parts[2]))
            elif pred == "visible_from":
                if len(parts) > 2:
                    obj, wp = parts[1], parts[2]
                    self.objective_visible_from_map.setdefault(obj, set()).add(wp)
            elif pred == "calibration_target":
                if len(parts) > 2:
                    camera, obj = parts[1], parts[2]
                    self.camera_cal_target_map[camera] = obj
            elif pred == "on_board":
                if len(parts) > 2:
                    camera, rover = parts[1], parts[2]
                    self.camera_on_board_map[camera] = rover
                    self.rover_cameras_map.setdefault(rover, set()).add(camera)

        # Build communication waypoints based on lander_wp and visible facts
        if self.lander_wp:
            for wp1, wp2 in visible_facts_list:
                if wp1 == self.lander_wp:
                    self.comm_wps.add(wp2)
                elif wp2 == self.lander_wp:
                    self.comm_wps.add(wp1)
            # Add lander waypoint itself if it's a valid location for a rover
            # and visible to itself (unlikely but possible in PDDL)
            # The action precondition is (visible ?x ?y) where ?y is lander_wp.
            # If rover is at lander_wp (?x = ?y), need (visible lander_wp lander_wp).
            if (self.lander_wp, self.lander_wp) in visible_facts_list:
                 self.comm_wps.add(self.lander_wp)


        # Parse goal facts
        self.goal_soil_wps = set()
        self.goal_rock_wps = set()
        self.goal_image_oms = set() # set of (obj, mode)
        for goal in task.goals:
            parts = get_parts(goal)
            if not parts: continue # Skip malformed goals
            pred = parts[0]
            if pred == "communicated_soil_data":
                if len(parts) > 1: self.goal_soil_wps.add(parts[1])
            elif pred == "communicated_rock_data":
                if len(parts) > 1: self.goal_rock_wps.add(parts[1])
            elif pred == "communicated_image_data":
                if len(parts) > 2: self.goal_image_oms.add((parts[1], parts[2]))

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Parse current state
        rover_locations = {} # {rover: wp}
        have_soil = set() # {(rover, wp)}
        have_rock = set() # {(rover, wp)}
        have_image = set() # {(rover, obj, mode)}
        stores_full = set() # {store}
        calibrated_cameras = set() # {(camera, rover)}
        communicated_soil = set() # {wp}
        communicated_rock = set() # {wp}
        communicated_image = set() # {(obj, mode)}
        current_soil_samples = set() # {wp}
        current_rock_samples = set() # {wp}

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts
            pred = parts[0]
            if pred == "at":
                # Could be rover or lander, but we only care about rover location
                if len(parts) > 2 and parts[1].startswith("rover"): # Assuming rover names start with "rover"
                     rover_locations[parts[1]] = parts[2]
            elif pred == "have_soil_analysis":
                if len(parts) > 2: have_soil.add((parts[1], parts[2]))
            elif pred == "have_rock_analysis":
                if len(parts) > 2: have_rock.add((parts[1], parts[2]))
            elif pred == "have_image":
                if len(parts) > 3: have_image.add((parts[1], parts[2], parts[3]))
            elif pred == "full":
                if len(parts) > 1: stores_full.add(parts[1])
            elif pred == "calibrated":
                if len(parts) > 2: calibrated_cameras.add((parts[1], parts[2]))
            elif pred == "communicated_soil_data":
                if len(parts) > 1: communicated_soil.add(parts[1])
            elif pred == "communicated_rock_data":
                if len(parts) > 1: communicated_rock.add(parts[1])
            elif pred == "communicated_image_data":
                if len(parts) > 2: communicated_image.add((parts[1], parts[2]))
            elif pred == "at_soil_sample":
                if len(parts) > 1: current_soil_samples.add(parts[1])
            elif pred == "at_rock_sample":
                if len(parts) > 1: current_rock_samples.add(parts[1])

        h = 0

        # Soil Goals
        for soil_wp in self.goal_soil_wps:
            if soil_wp in communicated_soil:
                continue
            min_cost_soil = float('inf')
            data_held = False
            for rover in self.equipped_soil:
                if (rover, soil_wp) in have_soil:
                    data_held = True
                    current_loc = rover_locations.get(rover)
                    if not current_loc: continue # Rover location unknown
                    cost = nav_set_to_set({current_loc}, self.comm_wps, rover, self.can_traverse_graph) + 1
                    min_cost_soil = min(min_cost_soil, cost)

            if not data_held:
                if soil_wp not in current_soil_samples:
                     min_cost_soil = float('inf') # Cannot sample if sample is gone and no one has data
                else:
                    for rover in self.equipped_soil:
                        store = self.rover_to_store.get(rover)
                        if not store: continue # Rover has no store? Problematic instance.
                        store_full_cost = 1 if store in stores_full else 0
                        current_loc = rover_locations.get(rover)
                        if not current_loc: continue # Rover location unknown

                        cost_nav_to_sample = nav_set_to_set({current_loc}, {soil_wp}, rover, self.can_traverse_graph)
                        cost_nav_sample_to_comm = nav_set_to_set({soil_wp}, self.comm_wps, rover, self.can_traverse_graph)

                        # Check if any navigation step is impossible
                        if cost_nav_to_sample == float('inf') or cost_nav_sample_to_comm == float('inf'):
                             cost = float('inf')
                        else:
                             cost = cost_nav_to_sample + store_full_cost + 1 + cost_nav_sample_to_comm + 1

                        min_cost_soil = min(min_cost_soil, cost)
            h += min_cost_soil

        # Rock Goals
        for rock_wp in self.goal_rock_wps:
            if rock_wp in communicated_rock:
                continue
            min_cost_rock = float('inf')
            data_held = False
            for rover in self.equipped_rock:
                if (rover, rock_wp) in have_rock:
                    data_held = True
                    current_loc = rover_locations.get(rover)
                    if not current_loc: continue # Rover location unknown
                    cost = nav_set_to_set({current_loc}, self.comm_wps, rover, self.can_traverse_graph) + 1
                    min_cost_rock = min(min_cost_rock, cost)

            if not data_held:
                if rock_wp not in current_rock_samples:
                     min_cost_rock = float('inf') # Cannot sample if sample is gone and no one has data
                else:
                    for rover in self.equipped_rock:
                        store = self.rover_to_store.get(rover)
                        if not store: continue # Rover has no store?
                        store_full_cost = 1 if store in stores_full else 0
                        current_loc = rover_locations.get(rover)
                        if not current_loc: continue # Rover location unknown

                        cost_nav_to_sample = nav_set_to_set({current_loc}, {rock_wp}, rover, self.can_traverse_graph)
                        cost_nav_sample_to_comm = nav_set_to_set({rock_wp}, self.comm_wps, rover, self.can_traverse_graph)

                        # Check if any navigation step is impossible
                        if cost_nav_to_sample == float('inf') or cost_nav_sample_to_comm == float('inf'):
                             cost = float('inf')
                        else:
                             cost = cost_nav_to_sample + store_full_cost + 1 + cost_nav_sample_to_comm + 1

                        min_cost_rock = min(min_cost_rock, cost)
            h += min_cost_rock


        # Image Goals
        for obj, mode in self.goal_image_oms:
            if (obj, mode) in communicated_image:
                continue
            min_cost_image = float('inf')
            image_held = False
            for rover in self.equipped_imaging:
                 for camera in self.rover_cameras_map.get(rover, set()):
                     if mode in self.camera_supports_map.get(camera, set()):
                         if (rover, obj, mode) in have_image:
                             image_held = True
                             current_loc = rover_locations.get(rover)
                             if not current_loc: continue # Rover location unknown
                             cost = nav_set_to_set({current_loc}, self.comm_wps, rover, self.can_traverse_graph) + 1
                             min_cost_image = min(min_cost_image, cost)

            if not image_held:
                img_wps = self.objective_visible_from_map.get(obj, set())
                if not img_wps:
                    min_cost_image = float('inf') # Cannot take image if objective not visible from anywhere
                else:
                    for rover in self.equipped_imaging:
                        current_loc = rover_locations.get(rover)
                        if not current_loc: continue # Rover location unknown
                        for camera in self.rover_cameras_map.get(rover, set()):
                            if mode in self.camera_supports_map.get(camera, set()):
                                cal_target = self.camera_cal_target_map.get(camera)
                                if not cal_target: continue # Camera has no calibration target
                                cal_wps = self.objective_visible_from_map.get(cal_target, set())
                                if not cal_wps: continue # Cannot calibrate if target not visible from anywhere

                                # Cost to acquire + communicate
                                # Nav(curr->cal) + Calibrate + Nav(cal->img) + TakeImage + Nav(img->comm) + Communicate
                                cost_nav_to_cal = nav_set_to_set({current_loc}, cal_wps, rover, self.can_traverse_graph)
                                cost_nav_cal_to_img = nav_set_to_set(cal_wps, img_wps, rover, self.can_traverse_graph)
                                cost_nav_img_to_comm = nav_set_to_set(img_wps, self.comm_wps, rover, self.can_traverse_graph)

                                # Check if any navigation step is impossible
                                if cost_nav_to_cal == float('inf') or cost_nav_cal_to_img == float('inf') or cost_nav_img_to_comm == float('inf'):
                                     cost = float('inf')
                                else:
                                     cost = cost_nav_to_cal + 1 + cost_nav_cal_to_img + 1 + cost_nav_img_to_comm + 1

                                min_cost_image = min(min_cost_image, cost)
            h += min_cost_image

        # If any goal was unreachable, h will be inf. Otherwise, it's the sum of finite costs.
        return h
