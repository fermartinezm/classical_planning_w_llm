from fnmatch import fnmatch
from collections import deque
import math

# Assume Heuristic base class exists and provides the interface
# from heuristics.heuristic_base import Heuristic

# Note: The Heuristic base class is assumed to be provided by the planning framework.
# If running this code standalone, you would need to define a simple base class
# like the dummy one shown in the thought block. For the final output,
# we assume the framework provides it and remove the dummy definition.


# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        # This should not happen with valid PDDL facts in state/static
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

def build_navigation_graph(static_facts, rover_name):
    """Builds an adjacency list graph for a specific rover based on can_traverse facts."""
    graph = {}
    all_waypoints_in_graph = set()
    for fact in static_facts:
        if match(fact, "can_traverse", rover_name, "*", "*"):
            _, r, from_wp, to_wp = get_parts(fact)
            if from_wp not in graph:
                graph[from_wp] = set()
            graph[from_wp].add(to_wp)
            all_waypoints_in_graph.add(from_wp)
            all_waypoints_in_graph.add(to_wp)

    # Ensure all waypoints mentioned are keys in the graph dictionary
    for wp in all_waypoints_in_graph:
        if wp not in graph:
            graph[wp] = set()

    return graph

def bfs(graph, start_node):
    """Performs BFS to find shortest distances from a start node in a graph."""
    distances = {node: math.inf for node in graph}
    
    # Check if the start_node is a known waypoint in the graph context
    all_nodes_in_graph = set(graph.keys())
    for neighbors in graph.values():
         all_nodes_in_graph.update(neighbors)

    if start_node not in all_nodes_in_graph:
         # Start node is not a known waypoint in the graph context for this rover
         # This might indicate an invalid state or problem setup.
         # We return distances initialized to infinity.
         return distances

    # If start_node is a known waypoint, set its distance to 0 and proceed with BFS
    distances[start_node] = 0
    queue = deque([start_node])

    while queue:
        current_node = queue.popleft()

        if current_node in graph: # Check again, safety
            for neighbor in graph[current_node]:
                if distances[neighbor] == math.inf:
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)
    return distances

def get_rover_location(state, rover_name):
    """Finds the current waypoint of a rover in the state."""
    for fact in state:
        if match(fact, "at", rover_name, "*"):
            return get_parts(fact)[2]
    return None # Should not happen in a valid state if rover exists

class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve each unmet goal fact independently and sums these estimates. It considers navigation costs (precomputed using BFS), sampling/imaging costs, and communication costs. It is non-admissible as it ignores negative interactions and resource constraints (like a rover's store capacity beyond needing it empty for sampling, or multiple goals requiring the same rover simultaneously).

    # Assumptions:
    - The cost of each action (navigate, sample, drop, calibrate, take_image, communicate) is 1.
    - Navigation cost between waypoints for a specific rover is the shortest path distance on the `can_traverse` graph for that rover.
    - To sample, a rover must be equipped, at the sample location, and have an empty store (a drop action is added if the store is full). The sample must exist at the location in the current state.
    - To take an image, a rover must be equipped for imaging, have a camera on board that supports the mode, the camera must be calibrated, and the rover must be at a waypoint visible from the objective. Taking an image uncalibrates the camera. This heuristic assumes calibration is needed every time a new image is taken for a goal.
    - To communicate data (sample or image), a rover must have the data and be at a waypoint visible from the lander's location.
    - Static facts (equipment, store_of, supports, visible, visible_from, calibration_target, on_board, at_lander, can_traverse) are extracted during initialization. Dynamic facts (at, have_soil_analysis, have_rock_analysis, have_image, communicated_*, at_soil_sample, at_rock_sample, empty, full, calibrated) are read from the current state.
    - If a goal requires a static condition that is not met (e.g., no rover is equipped for soil analysis for a soil goal), the cost for that goal is considered infinite.

    # Heuristic Initialization
    - Parses static facts to build data structures:
        - `lander_location`: Waypoint(s) of the lander.
        - `rover_equipment`: Which rovers have which equipment types.
        - `rover_store`: Which store belongs to which rover.
        - `camera_supports`: Which modes each camera supports.
        - `visible_waypoints`: Which waypoints are visible from each other.
        - `visible_from_objective`: Waypoints visible from each objective.
        - `camera_calibration_target`: Calibration target for each camera.
        - `camera_on_board`: Which rover each camera is on.
        - `rover_navigation_graphs`: Adjacency list for `can_traverse` for each rover.
        - `rover_distances`: Precomputed shortest path distances between all waypoint pairs for each rover using BFS.
        - `lander_visible_waypoints`: Waypoints visible from any lander location.
        - `camera_cal_waypoints`: Waypoints visible from the calibration target of each camera.

    # Step-By-Step Thinking for Computing Heuristic
    The heuristic value for a state is the sum of the estimated minimum costs for each goal fact that is not yet achieved in the current state.

    For each unmet goal fact `g`:
    1.  Initialize the minimum cost for `g` (`min_cost_g`) to infinity.
    2.  Identify the type of goal fact (`communicated_soil_data`, `communicated_rock_data`, or `communicated_image_data`).
    3.  **If `g` is `(communicated_soil_data ?w)`:**
        - Check if any rover `r` currently has `(have_soil_analysis ?r ?w)`.
            - If yes: The cost to get the data is 0. The location after getting data is the rover's current location. Calculate the minimum cost to communicate this data from the rover's current location to any lander-visible waypoint (navigation + 1 for communicate action). Update `min_cost_g`.
        - If no rover has the data, check if `(at_soil_sample ?w)` is in the current state.
            - If yes: Iterate through rovers `r` equipped for soil analysis. For each such rover, calculate the cost to sample: navigate from its current location to `?w`, add 1 for drop if its store is full, add 1 for sample_soil. The location after sampling is `?w`. Then, calculate the minimum cost to communicate from `?w` to any lander-visible waypoint (navigation + 1 for communicate). Sum the sampling cost and communication cost. Update `min_cost_g` with the minimum sum found across all suitable rovers.
        - If no rover has the data and the sample is not at `?w`, this goal is considered impossible from this state in this relaxation. `min_cost_g` remains infinity.
    4.  **If `g` is `(communicated_rock_data ?w)`:**
        - Follow the same logic as for soil data, replacing soil-specific predicates/equipment with rock-specific ones.
    5.  **If `g` is `(communicated_image_data ?o ?m)`:**
        - Check if any rover `r` currently has `(have_image ?r ?o ?m)`.
            - If yes: The cost to get the image is 0. The location after getting the image is the rover's current location. Calculate the minimum cost to communicate this data from the rover's current location to any lander-visible waypoint (navigation + 1 for communicate). Update `min_cost_g`.
        - If no rover has the image: Iterate through rovers `r` equipped for imaging. For each such rover, iterate through cameras `i` on board `r` that support mode `m`. Find the calibration target `t` for `i`. Find waypoints `cal_wps` visible from `t` and `img_wps` visible from `o`. If suitable `cal_wps` and `img_wps` exist:
            - Calculate the minimum cost to get the image using this rover/camera: Find the best pair of `cal_wp` and `img_wp`. The cost is navigate from rover's current location to `cal_wp`, +1 for calibrate, navigate from `cal_wp` to `img_wp`, +1 for take_image. The location after imaging is `img_wp`.
            - Calculate the minimum cost to communicate from `img_wp` to any lander-visible waypoint (navigation + 1 for communicate).
            - Sum the image cost and communication cost. Update `min_cost_g` with the minimum sum found across all suitable rovers/cameras/waypoints.
        - If no way to get the image and communicate it is found, `min_cost_g` remains infinity.
    6.  Add `min_cost_g` to the total heuristic value `h`. If `min_cost_g` is infinity, add `math.inf`.

    The final heuristic value is `h`.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        self.lander_location = set()
        self.rover_equipment = {} # {rover: {soil: bool, rock: bool, imaging: bool}}
        self.rover_store = {} # {rover: store}
        self.camera_supports = {} # {camera: set(mode)}
        self.visible_waypoints = {} # {wp: set(visible_wp)}
        self.visible_from_objective = {} # {objective: set(wp)}
        self.camera_calibration_target = {} # {camera: objective}
        self.camera_on_board = {} # {camera: rover}
        self.rover_navigation_graphs = {} # {rover: {from_wp: set(to_wp)}}
        self.rover_distances = {} # {rover: {from_wp: {to_wp: dist}}}
        self.lander_visible_waypoints = set()
        self.camera_cal_waypoints = {} # {camera: set(cal_wp)}

        all_rovers_set = set()

        # --- Parse Static Facts ---
        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]

            if predicate == "at_lander":
                self.lander_location.add(parts[2])
            elif predicate == "equipped_for_soil_analysis":
                rover = parts[1]
                if rover not in self.rover_equipment: self.rover_equipment[rover] = {}
                self.rover_equipment[rover]['soil'] = True
                all_rovers_set.add(rover)
            elif predicate == "equipped_for_rock_analysis":
                rover = parts[1]
                if rover not in self.rover_equipment: self.rover_equipment[rover] = {}
                self.rover_equipment[rover]['rock'] = True
                all_rovers_set.add(rover)
            elif predicate == "equipped_for_imaging":
                rover = parts[1]
                if rover not in self.rover_equipment: self.rover_equipment[rover] = {}
                self.rover_equipment[rover]['imaging'] = True
                all_rovers_set.add(rover)
            elif predicate == "store_of":
                self.rover_store[parts[2]] = parts[1] # rover -> store
                all_rovers_set.add(parts[2])
            elif predicate == "supports":
                camera, mode = parts[1], parts[2]
                if camera not in self.camera_supports: self.camera_supports[camera] = set()
                self.camera_supports[camera].add(mode)
            elif predicate == "visible":
                wp1, wp2 = parts[1], parts[2]
                if wp1 not in self.visible_waypoints: self.visible_waypoints[wp1] = set()
                if wp2 not in self.visible_waypoints: self.visible_waypoints[wp2] = set()
                self.visible_waypoints[wp1].add(wp2)
                self.visible_waypoints[wp2].add(wp1)
            elif predicate == "visible_from":
                objective, wp = parts[1], parts[2]
                if objective not in self.visible_from_objective: self.visible_from_objective[objective] = set()
                self.visible_from_objective[objective].add(wp)
            elif predicate == "calibration_target":
                self.camera_calibration_target[parts[1]] = parts[2] # camera -> objective
            elif predicate == "on_board":
                self.camera_on_board[parts[1]] = parts[2] # camera -> rover
                all_rovers_set.add(parts[2])
            elif predicate == "can_traverse":
                 rover = parts[1]
                 all_rovers_set.add(rover)

        # --- Build Navigation Graphs ---
        for rover in all_rovers_set:
            self.rover_navigation_graphs[rover] = build_navigation_graph(static_facts, rover)

        # --- Precompute Navigation Distances ---
        for rover, graph in self.rover_navigation_graphs.items():
            self.rover_distances[rover] = {}
            # BFS from every waypoint that is a key in the graph
            for start_wp in graph.keys():
                 self.rover_distances[rover][start_wp] = bfs(graph, start_wp)

        # --- Precompute Lander Visible Waypoints ---
        for lander_loc in self.lander_location:
            if lander_loc in self.visible_waypoints:
                self.lander_visible_waypoints.update(self.visible_waypoints[lander_loc])
            self.lander_visible_waypoints.add(lander_loc) # A waypoint is visible from itself

        # --- Store Calibration Target Visibility (Need camera -> set(cal_waypoints)) ---
        for camera, obj in self.camera_calibration_target.items():
             if obj in self.visible_from_objective:
                 self.camera_cal_waypoints[camera] = self.visible_from_objective[obj]
             else:
                 self.camera_cal_waypoints[camera] = set() # No visible waypoints for calibration target


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        h = 0
        infinity = math.inf

        # Helper to get rover location from current state
        def get_current_rover_location(rover_name):
             return get_rover_location(state, rover_name)

        # Helper to get distance for a rover
        def get_dist(rover_name, from_wp, to_wp):
            if rover_name not in self.rover_distances or from_wp not in self.rover_distances[rover_name]:
                 return infinity # Rover cannot navigate from this waypoint or waypoint is unknown
            return self.rover_distances[rover_name][from_wp].get(to_wp, infinity)

        # Iterate through all goal facts
        for goal in self.goals:
            if goal in state:
                continue # Goal already achieved

            parts = get_parts(goal)
            if not parts: continue

            predicate = parts[0]
            min_cost_for_this_goal = infinity

            # --- Goal: Communicated Soil Data ---
            if predicate == "communicated_soil_data":
                waypoint_w = parts[1]
                soil_sample_at_w = f'(at_soil_sample {waypoint_w})' in state

                # Check if data is already collected by any rover
                data_collected_by_rover = None
                for fact in state:
                    if match(fact, "have_soil_analysis", "*", waypoint_w):
                        data_collected_by_rover = get_parts(fact)[1]
                        break

                if data_collected_by_rover:
                    # Data is collected, just need to communicate
                    rover_r = data_collected_by_rover
                    rover_loc = get_current_rover_location(rover_r)
                    if rover_loc:
                        cost_to_communicate = infinity
                        for lander_wp in self.lander_visible_waypoints:
                            comm_dist = get_dist(rover_r, rover_loc, lander_wp)
                            if comm_dist != infinity:
                                cost_to_communicate = min(cost_to_communicate, comm_dist + 1) # +1 for communicate action
                        min_cost_for_this_goal = min(min_cost_for_this_goal, cost_to_communicate)

                elif soil_sample_at_w:
                    # Sample exists, need to sample and communicate
                    for rover_r, equipment in self.rover_equipment.items():
                        if equipment.get('soil'): # Check if rover is equipped
                            rover_loc = get_current_rover_location(rover_r)
                            store = self.rover_store.get(rover_r)

                            if rover_loc and store:
                                # Cost to sample
                                cost_to_sample = get_dist(rover_r, rover_loc, waypoint_w)
                                if cost_to_sample != infinity:
                                    if f'(full {store})' in state:
                                        cost_to_sample += 1 # drop action
                                    cost_to_sample += 1 # sample_soil action
                                    loc_after_sample = waypoint_w # Rover is at sample location

                                    # Cost to communicate from sample location
                                    cost_to_communicate = infinity
                                    for lander_wp in self.lander_visible_waypoints:
                                        comm_dist = get_dist(rover_r, loc_after_sample, lander_wp)
                                        if comm_dist != infinity:
                                            cost_to_communicate = min(cost_to_communicate, comm_dist + 1) # +1 for communicate action

                                    if cost_to_communicate != infinity:
                                        min_cost_for_this_goal = min(min_cost_for_this_goal, cost_to_sample + cost_to_communicate)

            # --- Goal: Communicated Rock Data ---
            elif predicate == "communicated_rock_data":
                waypoint_w = parts[1]
                rock_sample_at_w = f'(at_rock_sample {waypoint_w})' in state

                # Check if data is already collected by any rover
                data_collected_by_rover = None
                for fact in state:
                    if match(fact, "have_rock_analysis", "*", waypoint_w):
                        data_collected_by_rover = get_parts(fact)[1]
                        break

                if data_collected_by_rover:
                    # Data is collected, just need to communicate
                    rover_r = data_collected_by_rover
                    rover_loc = get_current_rover_location(rover_r)
                    if rover_loc:
                        cost_to_communicate = infinity
                        for lander_wp in self.lander_visible_waypoints:
                            comm_dist = get_dist(rover_r, rover_loc, lander_wp)
                            if comm_dist != infinity:
                                cost_to_communicate = min(cost_to_communicate, comm_dist + 1) # +1 for communicate action
                        min_cost_for_this_goal = min(min_cost_for_this_goal, cost_to_communicate)

                elif rock_sample_at_w:
                    # Sample exists, need to sample and communicate
                    for rover_r, equipment in self.rover_equipment.items():
                        if equipment.get('rock'): # Check if rover is equipped
                            rover_loc = get_current_rover_location(rover_r)
                            store = self.rover_store.get(rover_r)

                            if rover_loc and store:
                                # Cost to sample
                                cost_to_sample = get_dist(rover_r, rover_loc, waypoint_w)
                                if cost_to_sample != infinity:
                                    if f'(full {store})' in state:
                                        cost_to_sample += 1 # drop action
                                    cost_to_sample += 1 # sample_rock action
                                    loc_after_sample = waypoint_w # Rover is at sample location

                                    # Cost to communicate from sample location
                                    cost_to_communicate = infinity
                                    for lander_wp in self.lander_visible_waypoints:
                                        comm_dist = get_dist(rover_r, loc_after_sample, lander_wp)
                                        if comm_dist != infinity:
                                            cost_to_communicate = min(cost_to_communicate, comm_dist + 1) # +1 for communicate action

                                    if cost_to_communicate != infinity:
                                        min_cost_for_this_goal = min(min_cost_for_this_goal, cost_to_sample + cost_to_communicate)

            # --- Goal: Communicated Image Data ---
            elif predicate == "communicated_image_data":
                objective_o, mode_m = parts[1], parts[2]

                # Check if image is already collected by any rover
                image_collected_by_rover = None
                for fact in state:
                    if match(fact, "have_image", "*", objective_o, mode_m):
                        image_collected_by_rover = get_parts(fact)[1]
                        break

                if image_collected_by_rover:
                    # Image is collected, just need to communicate
                    rover_r = image_collected_by_rover
                    rover_loc = get_current_rover_location(rover_r)
                    if rover_loc:
                        cost_to_communicate = infinity
                        for lander_wp in self.lander_visible_waypoints:
                            comm_dist = get_dist(rover_r, rover_loc, lander_wp)
                            if comm_dist != infinity:
                                cost_to_communicate = min(cost_to_communicate, comm_dist + 1) # +1 for communicate action
                        min_cost_for_this_goal = min(min_cost_for_this_goal, cost_to_communicate)

                else:
                    # Image not collected, need to take image and communicate
                    img_wps = self.visible_from_objective.get(objective_o, set())
                    if img_wps: # Need at least one waypoint to view the objective from
                        for rover_r, equipment in self.rover_equipment.items():
                            if equipment.get('imaging'): # Check if rover is equipped
                                rover_loc = get_current_rover_location(rover_r)
                                if not rover_loc: continue # Rover location unknown

                                # Find best camera on this rover for this mode
                                best_camera = None
                                for camera, onboard_rover in self.camera_on_board.items():
                                    if onboard_rover == rover_r and mode_m in self.camera_supports.get(camera, set()):
                                        best_camera = camera
                                        break # Assume one camera is enough, pick the first one found

                                if best_camera:
                                    cal_wps = self.camera_cal_waypoints.get(best_camera, set())
                                    if cal_wps: # Need at least one waypoint for calibration target
                                        # Find best sequence: move to cal_wp, calibrate, move to img_wp, take_image
                                        best_image_sequence_cost = infinity
                                        loc_after_image = None

                                        for cal_wp in cal_wps:
                                            for img_wp in img_wps:
                                                cost_seq = get_dist(rover_r, rover_loc, cal_wp)
                                                if cost_seq != infinity:
                                                    cost_seq += 1 # calibrate
                                                    move_to_img_cost = get_dist(rover_r, cal_wp, img_wp)
                                                    if move_to_img_cost != infinity:
                                                        cost_seq += move_to_img_cost
                                                        cost_seq += 1 # take_image
                                                        if cost_seq < best_image_sequence_cost:
                                                            best_image_sequence_cost = cost_seq
                                                            loc_after_image = img_wp # Rover ends up at img_wp

                                        if best_image_sequence_cost != infinity and loc_after_image:
                                            # Cost to communicate from image location
                                            cost_to_communicate = infinity
                                            for lander_wp in self.lander_visible_waypoints:
                                                comm_dist = get_dist(rover_r, loc_after_image, lander_wp)
                                                if comm_dist != infinity:
                                                    cost_to_communicate = min(cost_to_communicate, comm_dist + 1) # +1 for communicate action

                                            if cost_to_communicate != infinity:
                                                min_cost_for_this_goal = min(min_cost_for_this_goal, best_image_sequence_cost + cost_to_communicate)


            # Add the minimum cost found for this goal to the total heuristic
            if min_cost_for_this_goal == infinity:
                h += infinity # Add infinity if the goal is impossible from this state
            else:
                h += min_cost_for_this_goal

        return h
