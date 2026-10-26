# Assuming heuristic_base.py exists and contains the Heuristic class
# from heuristics.heuristic_base import Heuristic

# If running standalone for testing, use the dummy class
class Heuristic:
    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static
    def __call__(self, node):
        raise NotImplementedError # This should be implemented by subclasses

# Helper functions
from fnmatch import fnmatch
from collections import deque

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    if not fact or not isinstance(fact, str) or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def bfs(graph, start_node, all_nodes):
    """
    Performs BFS to find shortest distances from start_node to all other nodes.
    graph: adjacency list {node: [neighbor1, neighbor2, ...]}
    start_node: the node to start BFS from
    all_nodes: list or set of all possible nodes in the graph
    Returns: {node: distance}
    """
    distances = {node: float('inf') for node in all_nodes}
    if start_node not in all_nodes:
         # Start node is not a known waypoint, cannot compute distances
         return distances # All distances remain inf

    distances[start_node] = 0
    queue = deque([start_node])

    while queue:
        current_node = queue.popleft()

        if current_node in graph:
            for neighbor in graph[current_node]:
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)
    return distances


class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the total number of actions and minimum required
    travel distance to satisfy all unachieved goal conditions. It sums the
    estimated cost for each unachieved goal independently, assuming resources
    can be optimally assigned to minimize the cost for that single goal.

    # Assumptions
    - Waypoint traversal is symmetric and uniform for all rovers, based on
      the 'visible' predicate. Shortest paths are precomputed using BFS.
    - Store capacity is not a bottleneck (i.e., an empty store is always
      available or can be made available cheaply, which is ignored).
    - Calibration persists once achieved (or can be re-achieved cheaply,
      costed as 1 action).
    - A rover can communicate all its held data items in one trip to the lander,
      but each item requires a separate 'communicate' action at the lander.
    - The static facts provided in the problem instance are sufficient to
      determine if a goal is achievable (e.g., if a camera can calibrate
      for a required objective). If a goal appears impossible based on static
      facts, its estimated cost is infinity.

    # Heuristic Initialization
    The constructor pre-processes static facts to build data structures:
    - Waypoint graph and all-pairs shortest path distances.
    - Lander location.
    - Mappings for rover equipment, camera capabilities (on-board, supports mode,
      calibration target), and objective visibility from waypoints.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify all goal conditions that are not yet satisfied in the current state.
    2. Initialize the total heuristic cost to 0.
    3. For each unsatisfied goal condition:
        a. Estimate the minimum cost to achieve *only* this goal, ignoring other goals.
        b. If the goal is `(communicated_soil_data ?w)`:
            i. Check if any rover currently has the soil sample from `?w` (`have_soil_sample ?r ?w`).
            ii. If yes: Find the rover `?r` with the sample that minimizes `Distance(?r_loc, lander_loc) + 1` (move to lander + communicate). Add this minimum cost to the total.
            iii. If no: Find the minimum cost for any soil-equipped rover `?r` to collect the sample and communicate it. This cost is `Distance(?r_loc, ?w) + 1` (move + sample) + `Distance(?w, lander_loc) + 1` (move + communicate). Find the minimum over all soil-equipped rovers and add it to the total. If no soil-equipped rover exists or path is blocked, the cost is infinity.
        c. If the goal is `(communicated_rock_data ?w)`: Follow the same logic as for soil data, using rock-equipped rovers and checking for `have_rock_sample`.
        d. If the goal is `(communicated_image_data ?o ?m)`:
            i. Check if any rover currently has the image of `?o` in mode `?m` (`have_image ?r ?o ?m`).
            ii. If yes: Find the rover `?r` with the image that minimizes `Distance(?r_loc, lander_loc) + 1` (move to lander + communicate). Add this minimum cost to the total.
            iii. If no: Find the minimum cost for any imaging-equipped rover `?r` with a camera `?c` supporting mode `?m` that can calibrate for objective `?o`, using a waypoint `?img_w` from which `?o` is visible. This cost is `Distance(?r_loc, ?img_w) + 1` (move + calibrate) + `1` (take image) + `Distance(?img_w, lander_loc) + 1` (move + communicate). Find the minimum over all valid `(rover, camera, waypoint)` combinations and add it to the total. If no such combination exists or path is blocked, the cost is infinity.
    4. Return the total heuristic cost. A large finite number is used for infinity if necessary.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # --- Pre-process Static Facts ---
        self.lander_loc = None
        self.rover_equip = {} # {rover: {equipment_type}}
        # self.rover_stores = {} # {rover: store} # Not used in simplified heuristic
        self.camera_on_board = {} # {camera: rover}
        self.camera_supports = {} # {camera: {mode}}
        self.camera_calibrates_for_obj = {} # {camera: {objective}}
        self.obj_visible_from_wp = {} # {objective: {waypoint}}
        self.waypoints = set()
        waypoint_graph = {} # Adjacency list for BFS

        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == "at_lander":
                # (at_lander general waypoint1)
                if len(parts) > 2:
                    self.lander_loc = parts[2]
                    self.waypoints.add(self.lander_loc)
            elif predicate == "equipped_for_soil_analysis":
                # (equipped_for_soil_analysis rover4)
                if len(parts) > 1:
                    rover = parts[1]
                    self.rover_equip.setdefault(rover, set()).add('soil')
            elif predicate == "equipped_for_rock_analysis":
                # (equipped_for_rock_analysis rover3)
                if len(parts) > 1:
                    rover = parts[1]
                    self.rover_equip.setdefault(rover, set()).add('rock')
            elif predicate == "equipped_for_imaging":
                # (equipped_for_imaging rover2)
                if len(parts) > 1:
                    rover = parts[1]
                    self.rover_equip.setdefault(rover, set()).add('imaging')
            # elif predicate == "store_of":
            #     # (store_of rover1store rover1)
            #     if len(parts) > 2:
            #         store, rover = parts[1], parts[2]
            #         self.rover_stores[rover] = store # Assuming one store per rover
            elif predicate == "visible":
                # (visible waypoint1 waypoint2)
                if len(parts) > 2:
                    w1, w2 = parts[1], parts[2]
                    self.waypoints.add(w1)
                    self.waypoints.add(w2)
                    waypoint_graph.setdefault(w1, []).append(w2)
                    waypoint_graph.setdefault(w2, []).append(w1) # Assuming visible is symmetric
            # elif predicate == "can_traverse":
            #      # (can_traverse rover1 waypoint1 waypoint2)
            #      # Assuming visible implies can_traverse for all rovers for simplicity
            #      pass # Already handled by 'visible'
            elif predicate == "calibration_target":
                # (calibration_target camera1 objective4)
                if len(parts) > 2:
                    camera, objective = parts[1], parts[2]
                    self.camera_calibrates_for_obj.setdefault(camera, set()).add(objective)
            elif predicate == "on_board":
                # (on_board camera1 rover3)
                if len(parts) > 2:
                    camera, rover = parts[1], parts[2]
                    self.camera_on_board[camera] = rover
            elif predicate == "supports":
                # (supports camera1 low_res)
                if len(parts) > 2:
                    camera, mode = parts[1], parts[2]
                    self.camera_supports.setdefault(camera, set()).add(mode)
            elif predicate == "visible_from":
                # (visible_from objective1 waypoint6)
                if len(parts) > 2:
                    objective, waypoint = parts[1], parts[2]
                    self.obj_visible_from_wp.setdefault(objective, set()).add(waypoint)
                    self.waypoints.add(waypoint) # Add waypoint from visible_from

        # Add waypoints from goals if they are not in static facts (e.g., new sample locations)
        for goal in self.goals:
             goal_parts = get_parts(goal)
             if not goal_parts: continue
             if goal_parts[0] in ["communicated_soil_data", "communicated_rock_data"]:
                  if len(goal_parts) > 1:
                       self.waypoints.add(goal_parts[1])
             # Image goals refer to objectives and modes, not directly waypoints in the goal predicate

        # Ensure all waypoints mentioned in graph edges are in the set
        for wps in waypoint_graph.values():
             for wp in wps:
                 self.waypoints.add(wp)

        # Compute all-pairs shortest paths
        self.distances = {}
        all_waypoints_list = list(self.waypoints) # Use a list for consistent ordering if needed, or just iterate set
        for start_wp in self.waypoints:
            self.distances[start_wp] = bfs(waypoint_graph, start_wp, self.waypoints)

        # Check if lander location is known and reachable from all waypoints (or vice versa)
        # If not, some goals might be impossible. BFS handles unreachable nodes with inf distance.


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # --- Parse State Facts ---
        current_rover_location = {} # {rover: waypoint}
        have_soil_sample = set() # {(rover, waypoint)}
        have_rock_sample = set() # {(rover, waypoint)}
        have_image = set() # {(rover, objective, mode)}
        communicated_soil_data = set() # {waypoint}
        communicated_rock_data = set() # {waypoint}
        communicated_image_data = set() # {(objective, mode)}
        # empty_stores = set() # Not used in simplified heuristic
        # full_stores = set() # Not used in simplified heuristic
        # calibrated_cameras = set() # Not used in simplified complexity

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]
            if predicate == "at":
                # (at rover1 waypoint2)
                if len(parts) > 2 and parts[1].startswith('rover'):
                    current_rover_location[parts[1]] = parts[2]
            elif predicate == "have_soil_sample":
                # (have_soil_sample rover1 waypoint4)
                 if len(parts) > 2:
                    have_soil_sample.add((parts[1], parts[2]))
            elif predicate == "have_rock_sample":
                # (have_rock_sample rover1 waypoint5)
                 if len(parts) > 2:
                    have_rock_sample.add((parts[1], parts[2]))
            elif predicate == "have_image":
                # (have_image rover1 objective1 colour)
                 if len(parts) > 3:
                    have_image.add((parts[1], parts[2], parts[3]))
            elif predicate == "communicated_soil_data":
                # (communicated_soil_data waypoint4)
                 if len(parts) > 1:
                    communicated_soil_data.add(parts[1])
            elif predicate == "communicated_rock_data":
                # (communicated_rock_data waypoint5)
                 if len(parts) > 1:
                    communicated_rock_data.add(parts[1])
            elif predicate == "communicated_image_data":
                # (communicated_image_data objective1 colour)
                 if len(parts) > 2:
                    communicated_image_data.add((parts[1], parts[2]))
            # elif predicate == "empty": empty_stores.add(parts[1])
            # elif predicate == "full": full_stores.add(parts[1])
            # elif predicate == "calibrated": calibrated_cameras.add((parts[1], parts[2]))


        total_cost = 0
        large_finite_value = 1000000 # Used for unreachable goals

        # Check if lander location is known and reachable from anywhere
        lander_reachable = self.lander_loc is not None and self.lander_loc in self.waypoints

        # --- Calculate Cost for Unsatisfied Goals ---
        for goal in self.goals:
            goal_parts = get_parts(goal)
            if not goal_parts: continue

            goal_predicate = goal_parts[0]

            if goal_predicate == "communicated_soil_data":
                # (communicated_soil_data ?w)
                if len(goal_parts) < 2: continue # Malformed goal
                w = goal_parts[1]
                if w not in communicated_soil_data:
                    # Goal not achieved
                    min_goal_cost = float('inf')

                    # Case 1: Sample is already collected by some rover
                    rover_with_sample = None
                    for r, sample_w in have_soil_sample:
                        if sample_w == w:
                            rover_with_sample = r # Found a rover with the sample
                            break # Assume first one is fine for min calculation

                    if rover_with_sample:
                        # Cost = Move to lander + Communicate
                        r_loc = current_rover_location.get(rover_with_sample)
                        if r_loc and lander_reachable and r_loc in self.distances and self.lander_loc in self.distances[r_loc]:
                             min_goal_cost = self.distances[r_loc][self.lander_loc] + 1
                        # else: min_goal_cost remains inf (rover location unknown or lander unreachable)
                    else:
                        # Case 2: Sample needs to be collected and communicated
                        # Cost = Move to W + Sample + Move to Lander + Communicate
                        min_collection_comm_cost = float('inf')
                        for r in self.rover_equip:
                            if 'soil' in self.rover_equip[r]: # Rover is equipped for soil
                                r_loc = current_rover_location.get(r)
                                # Need sample at W, check if W is a known waypoint and paths exist
                                if r_loc and w in self.waypoints and lander_reachable and \
                                   r_loc in self.distances and w in self.distances[r_loc] and \
                                   w in self.distances and self.lander_loc in self.distances[w]:
                                     cost = self.distances[r_loc][w] + 1 + self.distances[w][self.lander_loc] + 1
                                     min_collection_comm_cost = min(min_collection_comm_cost, cost)
                        min_goal_cost = min_collection_comm_cost

                    # Add the minimum cost for this goal
                    total_cost += min_goal_cost if min_goal_cost != float('inf') else large_finite_value

            elif goal_predicate == "communicated_rock_data":
                # (communicated_rock_data ?w)
                if len(goal_parts) < 2: continue # Malformed goal
                w = goal_parts[1]
                if w not in communicated_rock_data:
                    # Goal not achieved
                    min_goal_cost = float('inf')

                    # Case 1: Sample is already collected by some rover
                    rover_with_sample = None
                    for r, sample_w in have_rock_sample:
                        if sample_w == w:
                            rover_with_sample = r
                            break

                    if rover_with_sample:
                        # Cost = Move to lander + Communicate
                        r_loc = current_rover_location.get(rover_with_sample)
                        if r_loc and lander_reachable and r_loc in self.distances and self.lander_loc in self.distances[r_loc]:
                             min_goal_cost = self.distances[r_loc][self.lander_loc] + 1
                    else:
                        # Case 2: Sample needs to be collected and communicated
                        # Cost = Move to W + Sample + Move to Lander + Communicate
                        min_collection_comm_cost = float('inf')
                        for r in self.rover_equip:
                            if 'rock' in self.rover_equip[r]: # Rover is equipped for rock
                                r_loc = current_rover_location.get(r)
                                # Need sample at W, check if W is a known waypoint and paths exist
                                if r_loc and w in self.waypoints and lander_reachable and \
                                   r_loc in self.distances and w in self.distances[r_loc] and \
                                   w in self.distances and self.lander_loc in self.distances[w]:
                                     cost = self.distances[r_loc][w] + 1 + self.distances[w][self.lander_loc] + 1
                                     min_collection_comm_cost = min(min_collection_comm_cost, cost)
                        min_goal_cost = min_collection_comm_cost

                    # Add the minimum cost for this goal
                    total_cost += min_goal_cost if min_goal_cost != float('inf') else large_finite_value

            elif goal_predicate == "communicated_image_data":
                # (communicated_image_data ?o ?m)
                if len(goal_parts) < 3: continue # Malformed goal
                o, m = goal_parts[1], goal_parts[2]
                if (o, m) not in communicated_image_data:
                    # Goal not achieved
                    min_goal_cost = float('inf')

                    # Case 1: Image is already collected by some rover
                    rover_with_image = None
                    for r, img_o, img_m in have_image:
                        if img_o == o and img_m == m:
                            rover_with_image = r
                            break

                    if rover_with_image:
                        # Cost = Move to lander + Communicate
                        r_loc = current_rover_location.get(rover_with_image)
                        if r_loc and lander_reachable and r_loc in self.distances and self.lander_loc in self.distances[r_loc]:
                             min_goal_cost = self.distances[r_loc][self.lander_loc] + 1
                    else:
                        # Case 2: Image needs to be taken and communicated
                        # Cost = Move to Img_W + Calibrate + Take Image + Move to Lander + Communicate
                        min_collection_comm_cost = float('inf')
                        # Iterate through all possible (rover, camera, image_waypoint) combinations
                        for r in self.rover_equip:
                            if 'imaging' in self.rover_equip[r]: # Rover is equipped for imaging
                                r_loc = current_rover_location.get(r)
                                if not r_loc: continue # Skip if rover location unknown

                                for c in self.camera_on_board:
                                    if self.camera_on_board[c] == r: # Camera is on this rover
                                        if m in self.camera_supports.get(c, set()): # Camera supports the mode
                                            if o in self.camera_calibrates_for_obj.get(c, set()): # Camera calibrates for the objective
                                                # Find best waypoint to take the image from
                                                candidate_img_wps = self.obj_visible_from_wp.get(o, set())
                                                for img_w in candidate_img_wps:
                                                    # Check if img_w is a known waypoint and paths exist
                                                    if img_w in self.waypoints and lander_reachable and \
                                                       r_loc in self.distances and img_w in self.distances[r_loc] and \
                                                       img_w in self.distances and self.lander_loc in self.distances[img_w]:
                                                        cost = self.distances[r_loc][img_w] + 1 + 1 + self.distances[img_w][self.lander_loc] + 1
                                                        min_collection_comm_cost = min(min_collection_comm_cost, cost)
                        min_goal_cost = min_collection_comm_cost

                    # Add the minimum cost for this goal
                    total_cost += min_goal_cost if min_goal_cost != float('inf') else large_finite_value

            # Add other goal types if necessary (e.g., (at roverX waypointY)) - Rovers domain examples only show communication goals
            # else:
            #     # Handle other potential goal types if they exist and are not achieved
            #     pass


        # If total_cost is infinity (meaning at least one necessary goal is unreachable),
        # return a large finite value instead of float('inf') for compatibility
        # with some planner implementations.
        return total_cost if total_cost != float('inf') else large_finite_value

