from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import defaultdict, deque
import math # For infinity

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
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def bfs(graph, start):
    """
    Performs Breadth-First Search to find shortest distances from a start node.

    Args:
        graph: Adjacency list representation of the graph (dict: node -> list of neighbors).
        start: The starting node.

    Returns:
        A dictionary mapping each reachable node to its shortest distance from start.
        Nodes not reachable are not included.
    """
    distances = {start: 0}
    queue = deque([start])
    while queue:
        u = queue.popleft()
        if u in graph: # Ensure the node exists in the graph keys
            for v in graph[u]:
                if v not in distances:
                    distances[v] = distances[u] + 1
                    queue.append(v)
    return distances

class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve all goal
    conditions. It calculates the minimum cost for each unachieved goal fact
    independently, considering the capabilities and current state of each rover,
    and sums these minimum costs. It estimates costs based on movement distances
    (calculated via BFS on traversal graphs) and necessary actions like taking
    samples, images, calibrating, and communicating data.

    # Assumptions
    - The cost of each action (move, take_sample, drop_sample, calibrate,
      take_image, communicate_data) is 1.
    - Rovers can only traverse paths defined by `can_traverse` facts.
    - Lander location is static.
    - Sample locations (`at_soil_sample`, `at_rock_sample`) are static.
    - Objective visibility (`visible_from`) is static.
    - Camera properties (`on_board`, `supports`, `calibration_target`) are static.
    - Rover capabilities (`equipped_for_...`) are static.
    - The heuristic does not account for resource contention (e.g., multiple
      rovers needing the same sample or camera, store usage conflicts beyond
      needing an empty store for a new sample).
    - The heuristic calculates the cost for each goal independently and sums them,
      which is non-admissible but can be informative.
    - If a rover already possesses sample data from a waypoint, it is assumed
      to be currently located at that waypoint for the purpose of calculating
      communication cost. If a rover already possesses image data, it is assumed
      to be at its current location for communication cost calculation (as the
      specific image waypoint is not tracked in the state).

    # Heuristic Initialization
    The constructor pre-processes static facts to build data structures:
    - Identify the lander's location.
    - Map rovers to their capabilities (soil, rock, imaging).
    - Map stores to their respective rovers.
    - Map waypoints to the types of samples present.
    - Map objectives to waypoints from which they are visible.
    - Build traversal graphs for each rover based on `can_traverse` facts.
    - Pre-calculate all-pairs shortest path distances for each rover using BFS
      on their respective traversal graphs. Collect all unique waypoints involved
      in traversal graphs to ensure BFS covers all relevant locations.
    - Map rovers to cameras on board, and cameras to their supported modes and
      calibration targets.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify all goal facts that are not currently true in the state.
    2. Initialize the total heuristic cost to 0.
    3. For each unachieved goal fact:
        a. Determine the type of goal (soil, rock, or image communication).
        b. Find all rovers capable of achieving this type of goal based on static facts.
        c. If no capable rover exists or required static conditions (like sample presence or visible waypoints) are not met, the goal is unreachable; add a large cost (infinity) and continue to the next goal.
        d. Calculate the minimum estimated cost for *any* capable rover to achieve this specific goal fact, starting from its current state.
           - **For Soil/Rock Data (`communicated_soil_data ?w` or `communicated_rock_data ?w`):**
             - If the required sample data (`have_soil_analysis ?r ?w` or `have_rock_analysis ?r ?w`) is already in the state for some rover `?r`, the cost to get the data is 0, and the rover is assumed to be at `?w`.
             - Otherwise, for each capable rover `?r`:
               - Cost to get data = Distance from `?r`'s current location to `?w` + (1 if `?r`'s store is not empty) + 1 (take sample).
               - If `?r` cannot reach `?w`, cost is infinity for this rover path.
             - Cost to communicate = Distance from `?w` (where sample was taken or assumed location if data exists) to lander location + 1 (communicate).
             - If `?r` cannot reach the lander from `?w`, cost is infinity for this rover path.
             - Total cost for this rover = Cost to get data + Cost to communicate.
             - The cost for this goal is the minimum total cost over all capable rovers.
           - **For Image Data (`communicated_image_data ?o ?m`):**
             - If the required image data (`have_image ?r ?o ?m`) is already in the state for some rover `?r`, the cost to get the data is 0, and the rover is assumed to be at its current location.
             - Otherwise, for each capable rover `?r` with a camera `?c` supporting mode `?m` and targeting `?o`:
               - Find waypoints `?w_visible` from which `?o` is visible (static check). If none, this path is impossible.
               - Minimum cost to get image over all `?w_img` in `?w_visible`:
                 - Cost = (1 if `?c` is not calibrated on `?r` in the current state) + Distance from `?r`'s current location to `?w_img` + 1 (take image).
                 - If `?r` cannot reach `?w_img`, cost is infinity for this waypoint/camera path.
               - Minimum cost to get image is the minimum over suitable cameras `?c` and visible waypoints `?w_img`.
             - Cost to communicate = Distance from the chosen `?w_img` (where image was taken) to lander location + 1 (communicate).
             - If `?r` cannot reach the lander from `?w_img`, cost is infinity for this rover path.
             - Total cost for this rover = Minimum cost to get image + Cost to communicate.
             - The cost for this goal is the minimum total cost over all suitable rovers/cameras/waypoints.
        e. Add the minimum estimated cost for this goal to the total heuristic cost.
    4. Return the total heuristic cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # --- Pre-process Static Facts ---
        self.lander_location = None
        self.rover_capabilities = defaultdict(set) # rover -> {soil, rock, imaging}
        self.rover_stores = {} # store -> rover
        self.waypoint_samples = defaultdict(set) # waypoint -> {soil, rock}
        self.rover_traversal_graph = defaultdict(lambda: defaultdict(list)) # rover -> waypoint -> list of neighbors
        self.rover_distances = {} # rover -> waypoint -> waypoint -> distance
        self.rover_cameras = defaultdict(list) # rover -> list of cameras
        self.camera_info = defaultdict(lambda: {'modes': set(), 'target': None}) # camera -> {modes: {mode}, target: objective}
        self.objective_visibility = defaultdict(set) # objective -> {waypoint}

        # Collect all waypoints mentioned in can_traverse facts
        all_traversal_waypoints = set()

        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'at_lander':
                self.lander_location = parts[2]
            elif parts[0] == 'equipped_for_soil_analysis':
                self.rover_capabilities[parts[1]].add('soil')
            elif parts[0] == 'equipped_for_rock_analysis':
                self.rover_capabilities[parts[1]].add('rock')
            elif parts[0] == 'equipped_for_imaging':
                self.rover_capabilities[parts[1]].add('imaging')
            elif parts[0] == 'store_of':
                self.rover_stores[parts[1]] = parts[2]
            elif parts[0] == 'at_rock_sample':
                self.waypoint_samples[parts[1]].add('rock')
            elif parts[0] == 'at_soil_sample':
                self.waypoint_samples[parts[1]].add('soil')
            elif parts[0] == 'can_traverse':
                rover, w1, w2 = parts[1], parts[2], parts[3]
                self.rover_traversal_graph[rover][w1].append(w2)
                all_traversal_waypoints.add(w1)
                all_traversal_waypoints.add(w2)
            elif parts[0] == 'on_board':
                camera, rover = parts[1], parts[2]
                self.rover_cameras[rover].append(camera)
            elif parts[0] == 'supports':
                camera, mode = parts[1], parts[2]
                self.camera_info[camera]['modes'].add(mode)
            elif parts[0] == 'calibration_target':
                camera, objective = parts[1], parts[2]
                self.camera_info[camera]['target'] = objective
            elif parts[0] == 'visible_from':
                objective, waypoint = parts[1], parts[2]
                self.objective_visibility[objective].add(waypoint)

        # Pre-calculate distances for each rover
        for rover, graph in self.rover_traversal_graph.items():
            self.rover_distances[rover] = {}
            # Run BFS from every waypoint that is part of this rover's graph
            # or any waypoint mentioned in can_traverse for this rover.
            # Using all_traversal_waypoints ensures we cover all nodes in the graph.
            for start_waypoint in all_traversal_waypoints:
                 # Only run BFS if the start_waypoint is actually connected in this rover's graph
                 # (i.e., it's a key in the adjacency list or a neighbor)
                 if start_waypoint in graph or any(start_waypoint in neighbors for neighbors in graph.values()):
                     self.rover_distances[rover][start_waypoint] = bfs(graph, start_waypoint)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Check if goal is reached
        if self.goals <= state:
            return 0

        # --- Extract Current State Information ---
        current_rover_locations = {} # rover -> waypoint
        current_rover_stores_empty = defaultdict(bool) # rover -> bool (True if empty)
        current_rover_calibrated_cameras = defaultdict(set) # rover -> {camera}
        current_have_soil = set() # (rover, waypoint)
        current_have_rock = set() # (rover, waypoint)
        current_have_image = set() # (rover, objective, mode)
        communicated_soil = set() # waypoint
        communicated_rock = set() # waypoint
        communicated_image = set() # (objective, mode)

        # Initialize all stores as not empty by default, then update if 'empty' fact exists
        for store, rover in self.rover_stores.items():
             current_rover_stores_empty[rover] = False # Assume not empty unless proven otherwise

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at':
                # Could be rover or lander, only care about rovers here
                if parts[1] in self.rover_capabilities: # Check if it's a rover
                    current_rover_locations[parts[1]] = parts[2]
            elif parts[0] == 'empty':
                 # Find which rover this store belongs to
                 store_name = parts[1]
                 if store_name in self.rover_stores:
                     current_rover_stores_empty[self.rover_stores[store_name]] = True
            elif parts[0] == 'calibrated':
                camera, rover = parts[1], parts[2]
                current_rover_calibrated_cameras[rover].add(camera)
            elif parts[0] == 'have_soil_analysis':
                current_have_soil.add((parts[1], parts[2]))
            elif parts[0] == 'have_rock_analysis':
                current_have_rock.add((parts[1], parts[2]))
            elif parts[0] == 'have_image':
                current_have_image.add((parts[1], parts[2], parts[3]))
            elif parts[0] == 'communicated_soil_data':
                communicated_soil.add(parts[1])
            elif parts[0] == 'communicated_rock_data':
                communicated_rock.add(parts[1])
            elif parts[0] == 'communicated_image_data':
                communicated_image.add((parts[1], parts[2]))

        total_cost = 0

        # --- Calculate Cost for Unachieved Goals ---
        for goal in self.goals:
            parts = get_parts(goal)
            goal_predicate = parts[0]

            if goal_predicate == 'communicated_soil_data':
                waypoint = parts[1]
                if waypoint in communicated_soil:
                    continue # Goal already met

                # Find rovers capable of soil analysis
                capable_rovers = [r for r, caps in self.rover_capabilities.items() if 'soil' in caps]
                if not capable_rovers:
                    total_cost += math.inf # Impossible goal
                    continue

                # Check if soil sample exists at the waypoint (static check)
                if 'soil' not in self.waypoint_samples[waypoint]:
                     total_cost += math.inf # Impossible goal
                     continue

                min_goal_cost = math.inf

                for rover in capable_rovers:
                    if rover not in current_rover_locations: continue # Rover location unknown

                    rover_current_loc = current_rover_locations[rover]
                    rover_cost = 0

                    # Cost to get sample data
                    if (rover, waypoint) in current_have_soil:
                        cost_get_data = 0
                        loc_after_data = waypoint # Assume rover is at the sample waypoint if data is already collected
                    else:
                        # Need to move to waypoint, take sample
                        dist_to_sample = self.rover_distances.get(rover, {}).get(rover_current_loc, {}).get(waypoint, math.inf)
                        if dist_to_sample == math.inf: continue # Cannot reach sample waypoint

                        cost_get_data = dist_to_sample + 1 # move + take_sample
                        if not current_rover_stores_empty[rover]:
                            cost_get_data += 1 # Need to drop sample first (simplified)
                        loc_after_data = waypoint # Now at sample waypoint

                    # Cost to communicate data
                    dist_to_lander = self.rover_distances.get(rover, {}).get(loc_after_data, {}).get(self.lander_location, math.inf)
                    if dist_to_lander == math.inf: continue # Cannot reach lander

                    cost_communicate = dist_to_lander + 1 # move + communicate

                    rover_cost = cost_get_data + cost_communicate
                    min_goal_cost = min(min_goal_cost, rover_cost)

                total_cost += min_goal_cost

            elif goal_predicate == 'communicated_rock_data':
                waypoint = parts[1]
                if waypoint in communicated_rock:
                    continue # Goal already met

                # Find rovers capable of rock analysis
                capable_rovers = [r for r, caps in self.rover_capabilities.items() if 'rock' in caps]
                if not capable_rovers:
                    total_cost += math.inf # Impossible goal
                    continue

                # Check if rock sample exists at the waypoint (static check)
                if 'rock' not in self.waypoint_samples[waypoint]:
                     total_cost += math.inf # Impossible goal
                     continue

                min_goal_cost = math.inf

                for rover in capable_rovers:
                    if rover not in current_rover_locations: continue # Rover location unknown

                    rover_current_loc = current_rover_locations[rover]
                    rover_cost = 0

                    # Cost to get sample data
                    if (rover, waypoint) in current_have_rock:
                        cost_get_data = 0
                        loc_after_data = waypoint # Assume rover is at the sample waypoint if data is already collected
                    else:
                        # Need to move to waypoint, take sample
                        dist_to_sample = self.rover_distances.get(rover, {}).get(rover_current_loc, {}).get(waypoint, math.inf)
                        if dist_to_sample == math.inf: continue # Cannot reach sample waypoint

                        cost_get_data = dist_to_sample + 1 # move + take_sample
                        if not current_rover_stores_empty[rover]:
                            cost_get_data += 1 # Need to drop sample first (simplified)
                        loc_after_data = waypoint # Now at sample waypoint

                    # Cost to communicate data
                    dist_to_lander = self.rover_distances.get(rover, {}).get(loc_after_data, {}).get(self.lander_location, math.inf)
                    if dist_to_lander == math.inf: continue # Cannot reach lander

                    cost_communicate = dist_to_lander + 1 # move + communicate

                    rover_cost = cost_get_data + cost_communicate
                    min_goal_cost = min(min_goal_cost, rover_cost)

                total_cost += min_goal_cost

            elif goal_predicate == 'communicated_image_data':
                objective, mode = parts[1], parts[2]
                if (objective, mode) in communicated_image:
                    continue # Goal already met

                # Find waypoints visible from the objective (static check)
                visible_waypoints = list(self.objective_visibility[objective])
                if not visible_waypoints:
                    total_cost += math.inf # Impossible goal
                    continue

                min_goal_cost = math.inf

                # Find rovers capable of imaging with suitable camera
                suitable_rovers_cameras = [] # list of (rover, camera)
                for rover, caps in self.rover_capabilities.items():
                    if 'imaging' in caps:
                        for camera in self.rover_cameras[rover]:
                            if mode in self.camera_info[camera]['modes'] and self.camera_info[camera]['target'] == objective:
                                suitable_rovers_cameras.append((rover, camera))

                if not suitable_rovers_cameras:
                    total_cost += math.inf # Impossible goal
                    continue


                for rover, camera in suitable_rovers_cameras:
                    if rover not in current_rover_locations: continue # Rover location unknown

                    rover_current_loc = current_rover_locations[rover]

                    # Cost to get image data
                    if (rover, objective, mode) in current_have_image:
                        cost_get_data = 0
                        # If already have image, assume rover is at its current location
                        loc_after_data = rover_current_loc
                    else:
                        # Need to calibrate (if not already) and take image at a visible waypoint
                        cost_calibrate = 0
                        if camera not in current_rover_calibrated_cameras[rover]:
                            # Cost for calibration action
                            cost_calibrate = 1

                        min_cost_move_take_image = math.inf
                        best_image_waypoint = None

                        # Find the closest visible waypoint to take the image from
                        for img_wp in visible_waypoints:
                            dist_to_img_wp = self.rover_distances.get(rover, {}).get(rover_current_loc, {}).get(img_wp, math.inf)
                            if dist_to_img_wp == math.inf: continue # Cannot reach image waypoint

                            cost_move_take = dist_to_img_wp + 1 # move + take_image
                            if cost_move_take < min_cost_move_take_image:
                                min_cost_move_take_image = cost_move_take
                                best_image_waypoint = img_wp

                        if min_cost_move_take_image == math.inf: continue # Cannot reach any visible waypoint

                        cost_get_data = cost_calibrate + min_cost_move_take_image
                        loc_after_data = best_image_waypoint # Now at the chosen image waypoint

                    # Cost to communicate data
                    # Need location after getting data.
                    loc_for_communication = loc_after_data

                    dist_to_lander = self.rover_distances.get(rover, {}).get(loc_for_communication, {}).get(self.lander_location, math.inf)
                    if dist_to_lander == math.inf: continue # Cannot reach lander

                    cost_communicate = dist_to_lander + 1 # move + communicate

                    rover_cost = cost_get_data + cost_communicate
                    min_goal_cost = min(min_goal_cost, rover_cost)

                total_cost += min_goal_cost

        # If any goal was impossible, total_cost will be infinity
        return total_cost
