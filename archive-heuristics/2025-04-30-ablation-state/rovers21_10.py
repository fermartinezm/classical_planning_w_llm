import collections
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact string or malformed fact
    if not fact or not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
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

def bfs(graph, start_nodes, target_nodes):
    """
    Performs a Breadth-First Search to find the shortest distance
    from any node in start_nodes to any node in target_nodes.

    Args:
        graph: A dictionary representing the graph {node: [neighbors]}.
        start_nodes: A single node (string) or a set of nodes (set of strings) to start the search from.
        target_nodes: A single node (string) or a set of nodes (set of strings) to reach.

    Returns:
        The shortest distance (number of edges) or None if no target node is reachable.
    """
    if isinstance(start_nodes, str):
        start_nodes = {start_nodes}
    if isinstance(target_nodes, str):
        target_nodes = {target_nodes}

    if not start_nodes or not target_nodes:
        return None # Cannot start or no target

    queue = collections.deque()
    visited = set()

    # Initialize queue with all valid start nodes
    for start_node in start_nodes:
        if start_node in graph:
             queue.append((start_node, 0))
             visited.add(start_node)
        # else: start node is not a valid waypoint in the graph, cannot start from here.

    while queue:
        current_node, dist = queue.popleft()

        if current_node in target_nodes:
            return dist

        # Ensure current node is in the graph before iterating neighbors
        if current_node in graph:
            for neighbor in graph[current_node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))

    return None # Target not reachable


class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to satisfy all
    uncommunicated goal conditions. It considers the steps needed to collect
    data/images (sampling, imaging, calibration) and communicate them,
    including estimated navigation costs using shortest paths on the waypoint graph.

    # Assumptions
    - Navigation between waypoints costs 1 action per edge in the traversable graph for the specific rover.
    - Sampling (soil/rock) costs 1 action.
    - Dropping a sample costs 1 action (if store is full before sampling).
    - Calibration costs 1 action.
    - Taking an image costs 1 action.
    - Communication costs 1 action.
    - Samples (soil/rock) are non-renewable. If (at_soil_sample W) or (at_rock_sample W)
      is not in the state and the data is not held, the goal is considered unachievable
      from this state and a penalty is added.
    - Required equipment, cameras, and visibility conditions must exist for a goal
      to be achievable. If not, a penalty is added.
    - The heuristic sums the estimated costs for each uncommunicated goal independently.
      It does not consider resource contention (e.g., multiple rovers needing the same store)
      or parallelization.
    - For goals requiring data/image collection, the heuristic picks the first suitable
      equipped rover found. For navigation, it finds the shortest path for that rover.

    # Heuristic Initialization
    - Parses goal conditions to identify required communications.
    - Parses static facts to build:
        - Waypoint graph for each rover based on `can_traverse` and `visible`.
        - Lander location(s) and waypoints visible from the lander.
        - Rover capabilities (equipped for soil, rock, imaging).
        - Store ownership.
        - Camera information (on board, supports modes, calibration target).
        - Visibility information (objectives, calibration targets from waypoints).

    # Step-By-Step Thinking for Computing Heuristic
    For each goal fact (e.g., `(communicated_soil_data W)`, `(communicated_rock_data W)`, `(communicated_image_data O M)`) that is NOT yet true in the current state:

    1.  **Check if data/image is already collected/taken:**
        - For soil/rock `W`: Is `(have_soil_analysis R W)` or `(have_rock_analysis R W)` true for any rover `R`?
        - For image `(O M)`: Is `(have_image R O M)` true for any rover `R`?

    2.  **If data/image IS collected/taken:**
        - Add 1 action for `communicate`.
        - Find the rover `R` holding the data/image.
        - Calculate navigation cost for `R` from its current location to the nearest waypoint visible from the lander. Add this cost.
        - If no such waypoint is reachable, add a penalty.

    3.  **If data/image is NOT collected/taken:**
        - **For soil/rock `W`:**
            - Check if the sample `(at_soil_sample W)` or `(at_rock_sample W)` still exists in the state. If not, the goal is likely unachievable; add a penalty and skip this goal.
            - Find a rover `R` equipped for the required analysis (`equipped_for_soil_analysis` or `equipped_for_rock_analysis`). If none exists, add a penalty and skip.
            - Add 1 action for `sample`.
            - Check if the rover's store is full. If yes, add 1 action for `drop`.
            - Calculate navigation cost for `R` from its current location to `W`. Add this cost.
            - Calculate navigation cost for `R` from `W` to the nearest waypoint visible from the lander. Add this cost.
            - If any required navigation is impossible, add a penalty.
        - **For image `(O M)`:**
            - Find a rover `R` equipped for imaging (`equipped_for_imaging`) with a camera `I` on board (`on_board I R`) that supports mode `M` (`supports I M`). If no such combination exists, add a penalty and skip.
            - Find the calibration target `T` for camera `I` (`calibration_target I T`). If none, add a penalty and skip.
            - Find waypoints `W_cal` visible from `T` (`visible_from T W_cal`). If none, add a penalty and skip.
            - Find waypoints `W_img` visible from `O` (`visible_from O W_img`). If none, add a penalty and skip.
            - Add 1 action for `calibrate`.
            - Add 1 action for `take_image`.
            - Add 1 action for `communicate`.
            - Calculate navigation cost for `R` from its current location to the nearest `W_cal`. Add this cost.
            - Calculate navigation cost for `R` from the set of `W_cal` to the set of `W_img`. Add this cost.
            - Calculate navigation cost for `R` from the set of `W_img` to the set of lander-visible waypoints. Add this cost.
            - If any required navigation is impossible, add a penalty.

    4.  **Summing the Costs:**
        - The total heuristic value is the sum of the estimated costs for all uncommunicated goals.
        - If the total cost is 0, the state is a goal state.

    """
    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals

        # --- Parse Static Facts ---
        static_facts = task.static

        self.lander_locations = set()
        self.rover_graph = collections.defaultdict(lambda: collections.defaultdict(list)) # rover -> {wp: [neighbors]}
        self.visible_wps = collections.defaultdict(set) # wp -> {visible_neighbors}
        self.equipped_soil = set()
        self.equipped_rock = set()
        self.equipped_imaging = set()
        self.store_of = {} # store -> rover
        self.camera_on_board = {} # camera -> rover
        self.camera_supports = collections.defaultdict(set) # camera -> {modes}
        self.calibration_target = {} # camera -> objective (target)
        self.objective_visible_from = collections.defaultdict(set) # objective -> {waypoints}
        self.caltarget_visible_from = collections.defaultdict(set) # target -> {waypoints}

        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == "at_lander":
                # (at_lander ?x - lander ?y - waypoint)
                # lander, waypoint = parts[1], parts[2] # We only need the waypoint
                if len(parts) > 2:
                    self.lander_locations.add(parts[2])
            elif predicate == "can_traverse":
                # (can_traverse ?r - rover ?x - waypoint ?y - waypoint)
                if len(parts) > 3:
                    rover, wp1, wp2 = parts[1], parts[2], parts[3]
                    # Add bidirectional edge for the rover
                    self.rover_graph[rover][wp1].append(wp2)
                    self.rover_graph[rover][wp2].append(wp1)
            elif predicate == "visible":
                # (visible ?w - waypoint ?p - waypoint)
                if len(parts) > 2:
                    wp1, wp2 = parts[1], parts[2]
                    # Add bidirectional visibility
                    self.visible_wps[wp1].add(wp2)
                    self.visible_wps[wp2].add(wp1)
            elif predicate == "equipped_for_soil_analysis":
                if len(parts) > 1: self.equipped_soil.add(parts[1])
            elif predicate == "equipped_for_rock_analysis":
                if len(parts) > 1: self.equipped_rock.add(parts[1])
            elif predicate == "equipped_for_imaging":
                if len(parts) > 1: self.equipped_imaging.add(parts[1])
            elif predicate == "store_of":
                if len(parts) > 2: self.store_of[parts[1]] = parts[2] # store -> rover
            elif predicate == "on_board":
                if len(parts) > 2: self.camera_on_board[parts[1]] = parts[2] # camera -> rover
            elif predicate == "supports":
                if len(parts) > 2: self.camera_supports[parts[1]].add(parts[2]) # camera -> mode
            elif predicate == "calibration_target":
                if len(parts) > 2: self.calibration_target[parts[1]] = parts[2] # camera -> target (objective)
            elif predicate == "visible_from":
                # (visible_from ?o - objective ?w - waypoint)
                if len(parts) > 2:
                    obj_or_target, waypoint = parts[1], parts[2]
                    # Check if obj_or_target is a calibration target for any camera
                    is_cal_target = obj_or_target in self.calibration_target.values()
                    if is_cal_target:
                         self.caltarget_visible_from[obj_or_target].add(waypoint)
                    else:
                         self.objective_visible_from[obj_or_target].add(waypoint)


        # Precompute lander visible waypoints (waypoints visible *from* a lander location)
        # The communicate action requires (visible ?x ?y) where ?x is rover loc, ?y is lander loc.
        # So we need waypoints ?x such that (visible ?x ?y) is true for some lander loc ?y.
        self.lander_visible_wps = set()
        for lander_wp in self.lander_locations:
             # Find waypoints visible *to* the lander_wp
             for wp1, visible_neighbors in self.visible_wps.items():
                  if lander_wp in visible_neighbors:
                       self.lander_visible_wps.add(wp1)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        total_cost = 0
        PENALTY = 100 # Large cost for likely unachievable goals

        # --- Parse Current State ---
        rover_at = {} # rover -> waypoint
        store_status = {} # store -> 'empty' or 'full'
        have_soil = set() # {(rover, waypoint)}
        have_rock = set() # {(rover, waypoint)}
        have_image = set() # {(rover, objective, mode)}
        soil_samples_at = set() # {waypoint}
        rock_samples_at = set() # {waypoint}

        # Note: camera calibration status is dynamic but not explicitly tracked
        # in the heuristic calculation steps for image goals (we always add calibrate cost if image not taken)
        # This is a simplification for non-admissibility and speed.

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]
            if predicate == "at":
                # (at ?x - rover ?y - waypoint)
                # Check if the object is a rover (heuristic needs rover locations)
                # A more robust check would involve knowing object types from the task,
                # but checking if the object is a key in self.rover_graph (built from can_traverse)
                if len(parts) > 2 and parts[1] in self.rover_graph:
                     rover_at[parts[1]] = parts[2]
            elif predicate == "empty":
                if len(parts) > 1: store_status[parts[1]] = 'empty'
            elif predicate == "full":
                if len(parts) > 1: store_status[parts[1]] = 'full'
            # elif predicate == "calibrated": # Not strictly needed for this heuristic's logic
            #     pass
            elif predicate == "have_soil_analysis":
                # Ensure fact has correct number of parts before accessing index 2
                if len(parts) > 2:
                    have_soil.add((parts[1], parts[2]))
            elif predicate == "have_rock_analysis":
                 # Ensure fact has correct number of parts
                if len(parts) > 2:
                    have_rock.add((parts[1], parts[2]))
            elif predicate == "have_image":
                 # Ensure fact has correct number of parts
                if len(parts) > 3:
                    have_image.add((parts[1], parts[2], parts[3]))
            elif predicate == "at_soil_sample":
                if len(parts) > 1: soil_samples_at.add(parts[1])
            elif predicate == "at_rock_sample":
                if len(parts) > 1: rock_samples_at.add(parts[1])
            # Other static facts are ignored here as they are in self.__dict__

        # --- Calculate Cost for Each Goal ---
        for goal_str in self.goals:
            goal_parts = get_parts(goal_str)
            if not goal_parts: continue

            goal_predicate = goal_parts[0]

            if goal_predicate == "communicated_soil_data":
                # (communicated_soil_data ?p - waypoint)
                if len(goal_parts) < 2: continue # Malformed goal
                waypoint = goal_parts[1]

                if goal_str in state: continue # Goal already achieved

                # Check if data is collected
                data_collected = False
                rover_with_data = None
                # Find any rover that has the data
                for rover, wp in have_soil:
                    if wp == waypoint:
                        data_collected = True
                        rover_with_data = rover
                        break

                if data_collected:
                    # Data collected, need to communicate
                    total_cost += 1 # Communicate action
                    current_wp = rover_at.get(rover_with_data)
                    if current_wp and self.lander_visible_wps:
                         nav_c = bfs(self.rover_graph.get(rover_with_data, {}), current_wp, self.lander_visible_wps)
                         if nav_c is None: total_cost += PENALTY # Cannot reach lander spot
                         else: total_cost += nav_c
                    else:
                         total_cost += PENALTY # Rover location unknown or no lander spots

                else:
                    # Data not collected, need to sample and communicate
                    if waypoint not in soil_samples_at:
                        # Sample is gone, goal likely impossible
                        total_cost += PENALTY
                        continue

                    # Find a suitable equipped rover that is in the current state
                    equipped_rover = None
                    for rover in self.equipped_soil:
                        if rover in rover_at:
                            equipped_rover = rover
                            break
                    if equipped_rover is None:
                        # No rover can sample soil
                        total_cost += PENALTY
                        continue

                    total_cost += 1 # Sample action
                    total_cost += 1 # Communicate action

                    # Check if drop is needed for this rover's store
                    store = None
                    for s, r in self.store_of.items():
                        if r == equipped_rover:
                            store = s
                            break
                    # Only add drop cost if the specific store for the chosen rover is full
                    if store and store_status.get(store) == 'full':
                        total_cost += 1 # Drop action

                    # Navigation to sample spot
                    current_wp = rover_at.get(equipped_rover)
                    if current_wp:
                        nav_to_sample = bfs(self.rover_graph.get(equipped_rover, {}), current_wp, {waypoint})
                        if nav_to_sample is None: total_cost += PENALTY # Cannot reach sample spot
                        else: total_cost += nav_to_sample
                    else:
                         total_cost += PENALTY # Rover location unknown

                    # Navigation from sample spot to lander spot
                    # Assume the rover goes from the sample spot directly to a lander spot
                    if waypoint and self.lander_visible_wps:
                         nav_to_lander = bfs(self.rover_graph.get(equipped_rover, {}), waypoint, self.lander_visible_wps)
                         if nav_to_lander is None: total_cost += PENALTY # Cannot reach lander spot from sample spot
                         else: total_cost += nav_to_lander
                    else:
                         total_cost += PENALTY # Sample spot unknown or no lander spots


            elif goal_predicate == "communicated_rock_data":
                # (communicated_rock_data ?p - waypoint)
                if len(goal_parts) < 2: continue # Malformed goal
                waypoint = goal_parts[1]

                if goal_str in state: continue # Goal already achieved

                # Check if data is collected
                data_collected = False
                rover_with_data = None
                for rover, wp in have_rock:
                    if wp == waypoint:
                        data_collected = True
                        rover_with_data = rover
                        break

                if data_collected:
                    # Data collected, need to communicate
                    total_cost += 1 # Communicate action
                    current_wp = rover_at.get(rover_with_data)
                    if current_wp and self.lander_visible_wps:
                         nav_c = bfs(self.rover_graph.get(rover_with_data, {}), current_wp, self.lander_visible_wps)
                         if nav_c is None: total_cost += PENALTY # Cannot reach lander spot
                         else: total_cost += nav_c
                    else:
                         total_cost += PENALTY # Rover location unknown or no lander spots

                else:
                    # Data not collected, need to sample and communicate
                    if waypoint not in rock_samples_at:
                        # Sample is gone, goal likely impossible
                        total_cost += PENALTY
                        continue

                    # Find a suitable equipped rover that is in the current state
                    equipped_rover = None
                    for rover in self.equipped_rock:
                        if rover in rover_at:
                            equipped_rover = rover
                            break
                    if equipped_rover is None:
                        # No rover can sample rock
                        total_cost += PENALTY
                        continue

                    total_cost += 1 # Sample action
                    total_cost += 1 # Communicate action

                    # Check if drop is needed for this rover's store
                    store = None
                    for s, r in self.store_of.items():
                        if r == equipped_rover:
                            store = s
                            break
                    # Only add drop cost if the specific store for the chosen rover is full
                    if store and store_status.get(store) == 'full':
                        total_cost += 1 # Drop action

                    # Navigation to sample spot
                    current_wp = rover_at.get(equipped_rover)
                    if current_wp:
                        nav_to_sample = bfs(self.rover_graph.get(equipped_rover, {}), current_wp, {waypoint})
                        if nav_to_sample is None: total_cost += PENALTY # Cannot reach sample spot
                        else: total_cost += nav_to_sample
                    else:
                         total_cost += PENALTY # Rover location unknown

                    # Navigation from sample spot to lander spot
                    if waypoint and self.lander_visible_wps:
                         nav_to_lander = bfs(self.rover_graph.get(equipped_rover, {}), waypoint, self.lander_visible_wps)
                         if nav_to_lander is None: total_cost += PENALTY # Cannot reach lander spot from sample spot
                         else: total_cost += nav_to_lander
                    else:
                         total_cost += PENALTY # Sample spot unknown or no lander spots


            elif goal_predicate == "communicated_image_data":
                # (communicated_image_data ?o - objective ?m - mode)
                if len(goal_parts) < 3: continue # Malformed goal
                objective, mode = goal_parts[1], goal_parts[2]

                if goal_str in state: continue # Goal already achieved

                # Check if image is taken
                image_taken = False
                rover_with_image = None
                for rover, obj, m in have_image:
                    if obj == objective and m == mode:
                        image_taken = True
                        rover_with_image = rover
                        break

                if image_taken:
                    # Image taken, need to communicate
                    total_cost += 1 # Communicate action
                    current_wp = rover_at.get(rover_with_image)
                    if current_wp and self.lander_visible_wps:
                         nav_c = bfs(self.rover_graph.get(rover_with_image, {}), current_wp, self.lander_visible_wps)
                         if nav_c is None: total_cost += PENALTY # Cannot reach lander spot
                         else: total_cost += nav_c
                    else:
                         total_cost += PENALTY # Rover location unknown or no lander spots

                else:
                    # Image not taken, need to calibrate, take image, and communicate
                    total_cost += 1 # Calibrate action
                    total_cost += 1 # Take_image action
                    total_cost += 1 # Communicate action

                    # Find a suitable equipped rover with camera supporting mode that is in the current state
                    equipped_rover = None
                    camera = None
                    for rover in self.equipped_imaging:
                        if rover in rover_at:
                            for cam, r_cam in self.camera_on_board.items():
                                if r_cam == rover and mode in self.camera_supports.get(cam, set()):
                                    equipped_rover = rover
                                    camera = cam
                                    break
                        if equipped_rover: break # Found one

                    if equipped_rover is None:
                        # No rover/camera combo can take this image
                        total_cost += PENALTY
                        continue

                    # Find calibration waypoint(s)
                    cal_target = self.calibration_target.get(camera)
                    if cal_target is None:
                         # Camera has no calibration target
                         total_cost += PENALTY
                         continue
                    cal_wps = self.caltarget_visible_from.get(cal_target, set())
                    if not cal_wps:
                         # No waypoint to calibrate from
                         total_cost += PENALTY
                         continue

                    # Find image waypoint(s)
                    img_wps = self.objective_visible_from.get(objective, set())
                    if not img_wps:
                         # No waypoint to take image from
                         total_cost += PENALTY
                         continue

                    # Calculate navigation costs
                    current_wp = rover_at.get(equipped_rover)
                    if current_wp:
                        # Nav from current to calibration spot
                        nav_to_cal = bfs(self.rover_graph.get(equipped_rover, {}), current_wp, cal_wps)
                        if nav_to_cal is None: total_cost += PENALTY # Cannot reach calibration spot
                        else: total_cost += nav_to_cal

                        # Nav from calibration spot to image spot
                        # BFS from the set of cal_wps to the set of img_wps
                        nav_cal_img = bfs(self.rover_graph.get(equipped_rover, {}), cal_wps, img_wps)
                        if nav_cal_img is None: total_cost += PENALTY # Cannot reach image spot from calibration spot
                        else: total_cost += nav_cal_img

                        # Nav from image spot to lander spot
                        # BFS from the set of img_wps to the set of lander_visible_wps
                        nav_img_lander = bfs(self.rover_graph.get(equipped_rover, {}), img_wps, self.lander_visible_wps)
                        if nav_img_lander is None: total_cost += PENALTY # Cannot reach lander spot from image spot
                        else: total_cost += nav_img_lander
                    else:
                         total_cost += PENALTY # Rover location unknown


        return total_cost
