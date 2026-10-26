from heuristics.heuristic_base import Heuristic
from collections import deque
import re
import math

# Helper function to parse PDDL fact strings
def parse_fact(fact_string):
    """
    Parses a PDDL fact string into a tuple (predicate, arg1, arg2, ...).
    Handles facts with varying numbers of arguments.
    """
    # Remove surrounding brackets and split by spaces
    parts = fact_string[1:-1].split()
    if not parts:
        return None
    predicate = parts[0]
    args = parts[1:]
    return (predicate, *args)

# Helper function for Breadth-First Search (BFS)
def bfs(graph, start_node):
    """
    Performs BFS on a graph represented as an adjacency list to find shortest
    distances from a start node.
    """
    distances = {node: math.inf for node in graph}
    if start_node not in graph:
         # If start_node is not in the graph keys (e.g., isolated waypoint)
         # BFS from here can only reach itself if it exists in the distances map.
         if start_node in distances:
             distances[start_node] = 0
         return distances

    distances[start_node] = 0
    queue = deque([start_node])
    while queue:
        current_node = queue.popleft()
        # Ensure current_node is in graph before accessing neighbors
        if current_node in graph:
            for neighbor in graph.get(current_node, []):
                if distances[neighbor] == math.inf:
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)
    return distances

# Helper function to compute all-pairs shortest paths for each rover
def compute_all_pairs_shortest_paths(waypoints, can_traverse_map):
    """
    Computes shortest path distances between all pairs of waypoints for each
    rover based on its can_traverse graph using BFS.
    """
    all_paths = {}
    # Ensure all waypoints are included in the graph structure for BFS,
    # even if no 'can_traverse' starts from them.
    graphs_with_all_wps = {}
    for rover, graph in can_traverse_map.items():
        graphs_with_all_wps[rover] = {wp: set() for wp in waypoints}
        for wp1, neighbors in graph.items():
             graphs_with_all_wps[rover][wp1] = neighbors

    for rover, graph in graphs_with_all_wps.items():
        all_paths[rover] = {}
        for start_wp in waypoints:
            all_paths[rover][start_wp] = bfs(graph, start_wp)
    return all_paths


class roversHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Rovers domain.

    Summary:
    The heuristic estimates the cost to reach the goal by summing the estimated
    costs for each uncommunicated goal fact. For each uncommunicated goal, it
    estimates the minimum number of actions required to achieve and communicate
    the corresponding data (soil, rock, or image). This estimation considers
    the steps needed (sampling/imaging, calibration, movement, communication)
    and selects the most efficient available rover/camera/store based on
    precomputed shortest path distances between waypoints. It is not admissible
    but aims to be informative for greedy best-first search.

    Assumptions:
    - All action costs are 1.
    - The shortest path between waypoints for each rover is the minimum number of 'navigate' actions.
    - A rover needs an empty store to sample soil or rock. If its store is full, a 'drop' action is needed first (cost 1).
    - A camera needs to be calibrated before taking an image. Calibration requires moving to a waypoint visible from the calibration target (cost to move + 1 calibrate). Taking an image uncalibrates the camera.
    - Communication requires the rover to be at a waypoint visible from the lander's waypoint (cost to move + 1 communicate).
    - There is at least one lander.
    - For simplicity, assumes only one store per rover and one calibration target per camera.
    - Assumes samples (at_soil_sample, at_rock_sample) and objective visibility (visible_from) are static initial facts that might be consumed (samples) or are properties (visibility). The heuristic checks for sample existence in the current state.

    Heuristic Initialization:
    The constructor parses the static facts from the task description to build
    data structures for efficient lookup:
    - Waypoint graph for each rover based on 'can_traverse'.
    - All-pairs shortest path distances between waypoints for each rover using BFS.
    - Lander locations.
    - Rover equipment (soil, rock, imaging).
    - Store ownership.
    - Camera information (on-board, supported modes, calibration target).
    - Visibility between waypoints and objective/target visibility from waypoints.
    - Precomputes the set of 'communication waypoints' (visible from the lander).
    - Infers all possible objects (rovers, waypoints, etc.) from all facts in the task.

    Step-By-Step Thinking for Computing Heuristic:
    1.  Parse the current state into easily queryable data structures (e.g., dictionaries mapping rovers to locations, sets of full stores, sets of collected data, etc.).
    2.  Initialize the total heuristic value `h` to 0.
    3.  Iterate through each goal fact specified in `self.goals`.
    4.  If a goal fact is already true in the current state, skip it.
    5.  If the goal fact is `(communicated_soil_data ?w)`:
        a.  Check if `(have_soil_analysis ?r ?w)` is true for any rover `r` in the current state.
        b.  If yes, the remaining cost is primarily communication. Find the minimum cost among all rovers that have the data: cost is 1 (communicate) + minimum navigation cost for that rover from its current location to any communication waypoint. Add this minimum cost to `h`. If no such rover can reach a communication waypoint, the goal is unreachable (return infinity).
        c.  If no rover has the data, the remaining cost involves sampling and communication. Check if a soil sample still exists at `?w`. If not, the goal is unreachable (return infinity). Otherwise, find the minimum cost among all soil-equipped rovers `r_s` with an empty store `s_s` (or full store needing drop): cost is navigation from `r_s`'s current location to `?w` + (1 if store full else 0) + 1 (sample) + navigation from `?w` to a communication waypoint + 1 (communicate). Add this minimum cost to `h`. If no suitable rover/store combination can achieve this (e.g., cannot reach `?w` or a communication waypoint), the goal is unreachable (return infinity).
    6.  If the goal fact is `(communicated_rock_data ?w)`: Follow a similar logic as for soil data, using rock-equipped rovers and rock samples. Check for rock sample existence.
    7.  If the goal fact is `(communicated_image_data ?o ?m)`:
        a.  Check if `(have_image ?r ?o ?m)` is true for any rover `r` in the current state.
        b.  If yes, the remaining cost is communication. Find the minimum cost among rovers that have the image: cost is 1 (communicate) + minimum navigation cost from current location to a communication waypoint. Add this minimum cost to `h`. If unreachable, return infinity.
        c.  If no rover has the image, the remaining cost involves imaging and communication. Find the minimum cost among imaging-equipped rovers `r_i` with camera `i` supporting mode `m`: cost is navigation from `r_i`'s current location to a waypoint `p` visible from `?o` + 1 (take_image) + navigation from `p` to a communication waypoint + 1 (communicate). If camera `i` is not calibrated, add the cost of calibration: navigation from `r_i`'s current location to a waypoint `w_c` visible from the calibration target + 1 (calibrate) + navigation from `w_c` to `p`. Minimize this total cost over all suitable rovers, cameras, and image waypoints `p` (and calibration waypoints `w_c` if needed). Add this minimum cost to `h`. If unreachable, return infinity.
    8.  Return the total accumulated heuristic value `h`.
    """

    def __init__(self, task):
        super().__init__()
        self.goals = task.goals

        # --- Static Information ---
        self.lander_at = {} # lander -> waypoint
        self.equipped_soil = set() # {rover}
        self.equipped_rock = set() # {rover}
        self.equipped_imaging = set() # {rover}
        self.rover_stores = {} # rover -> store (assuming one per rover)
        self.store_of = {} # store -> rover
        self.rover_cameras = {} # rover -> {camera}
        self.on_board = {} # camera -> rover
        self.supports = {} # camera -> {mode}
        self.calibration_target = {} # camera -> objective
        self.visible = {} # waypoint -> {waypoint} (wp1 visible from wp2 means (visible wp1 wp2))
        self.visible_from = {} # objective/target -> {waypoint} (objective/target visible from wp means (visible_from objective/target wp))
        self.can_traverse = {} # rover -> {waypoint -> {waypoint}}

        # Infer all possible objects from all facts in the task
        self.waypoints = set()
        self.rovers = set()
        self.landers = set()
        self.stores = set()
        self.cameras = set()
        self.modes = set()
        self.objectives = set() # Includes calibration targets which are objectives

        all_facts = set(task.initial_state) | set(task.goals) | set(task.static)
        for fact_str in all_facts:
            parsed = parse_fact(fact_str)
            if not parsed: continue
            pred = parsed[0]
            args = parsed[1:]

            # Add objects based on predicate arguments and types
            if pred in ['at', 'can_traverse']:
                if len(args) > 0: self.rovers.add(args[0])
                if len(args) > 1: self.waypoints.add(args[1])
                if len(args) > 2: self.waypoints.add(args[2])
            elif pred == 'at_lander':
                if len(args) > 0: self.landers.add(args[0])
                if len(args) > 1: self.waypoints.add(args[1])
            elif pred.startswith('equipped_for_'):
                if len(args) > 0: self.rovers.add(args[0])
            elif pred in ['empty', 'full', 'store_of']:
                if len(args) > 0: self.stores.add(args[0])
                if pred == 'store_of' and len(args) > 1: self.rovers.add(args[1])
            elif pred in ['calibrated', 'supports', 'calibration_target', 'on_board']:
                if len(args) > 0: self.cameras.add(args[0])
                if pred in ['calibrated', 'on_board'] and len(args) > 1: self.rovers.add(args[1])
                if pred == 'supports' and len(args) > 1: self.modes.add(args[1])
                if pred == 'calibration_target' and len(args) > 1: self.objectives.add(args[1]) # Calibration targets are objectives
            elif pred == 'visible':
                if len(args) > 0: self.waypoints.add(args[0])
                if len(args) > 1: self.waypoints.add(args[1])
            elif pred in ['have_rock_analysis', 'have_soil_analysis']:
                if len(args) > 0: self.rovers.add(args[0])
                if len(args) > 1: self.waypoints.add(args[1])
            elif pred == 'have_image':
                if len(args) > 0: self.rovers.add(args[0])
                if len(args) > 1: self.objectives.add(args[1])
                if len(args) > 2: self.modes.add(args[2])
            elif pred.startswith('communicated_'):
                if pred in ['communicated_soil_data', 'communicated_rock_data'] and len(args) > 0:
                    self.waypoints.add(args[0])
                elif pred == 'communicated_image_data' and len(args) > 1:
                    self.objectives.add(args[0])
                    self.modes.add(args[1])
            elif pred in ['at_soil_sample', 'at_rock_sample']:
                if len(args) > 0: self.waypoints.add(args[0])
            elif pred == 'visible_from':
                if len(args) > 0: self.objectives.add(args[0]) # Assume anything visible_from is an objective/target
                if len(args) > 1: self.waypoints.add(args[1])


        # Parse static facts into structured data
        for fact_str in task.static:
            parsed = parse_fact(fact_str)
            if not parsed: continue
            pred = parsed[0]
            args = parsed[1:]

            if pred == 'at_lander' and len(args) == 2:
                self.lander_at[args[0]] = args[1]
            elif pred == 'equipped_for_soil_analysis' and len(args) == 1:
                self.equipped_soil.add(args[0])
            elif pred == 'equipped_for_rock_analysis' and len(args) == 1:
                self.equipped_rock.add(args[0])
            elif pred == 'equipped_for_imaging' and len(args) == 1:
                self.equipped_imaging.add(args[0])
            elif pred == 'store_of' and len(args) == 2:
                self.store_of[args[0]] = args[1]
                self.rover_stores[args[1]] = args[0] # Assuming one store per rover
            elif pred == 'supports' and len(args) == 2:
                self.supports.setdefault(args[0], set()).add(args[1])
            elif pred == 'calibration_target' and len(args) == 2:
                self.calibration_target[args[0]] = args[1]
            elif pred == 'on_board' and len(args) == 2:
                self.on_board[args[0]] = args[1]
                self.rover_cameras.setdefault(args[1], set()).add(args[0])
            elif pred == 'visible' and len(args) == 2:
                self.visible.setdefault(args[0], set()).add(args[1])
            elif pred == 'visible_from' and len(args) == 2:
                 self.visible_from.setdefault(args[0], set()).add(args[1])
            elif pred == 'can_traverse' and len(args) == 3:
                rover, wp1, wp2 = args
                self.can_traverse.setdefault(rover, {}).setdefault(wp1, set()).add(wp2)

        # Precompute shortest paths for each rover
        self.rover_shortest_paths = compute_all_pairs_shortest_paths(list(self.waypoints), self.can_traverse)

        # Precompute communication waypoints for each lander
        self.lander_comm_wps = {}
        for lander, lander_wp in self.lander_at.items():
            self.lander_comm_wps[lander] = set()
            # Find all waypoints ?x such that (visible ?x lander_wp) is true
            # This means ?x is visible from lander_wp
            for wp1, visible_neighbors in self.visible.items():
                 if lander_wp in visible_neighbors:
                     self.lander_comm_wps[lander].add(wp1)

        # Assuming there's only one lander for simplicity based on examples
        self.the_lander = list(self.lander_at.keys())[0] if self.lander_at else None
        self.the_lander_wp = self.lander_at.get(self.the_lander) if self.the_lander else None
        self.the_comm_wps = self.lander_comm_wps.get(self.the_lander, set()) if self.the_lander else set()


    def get_rover_location(self, state_rover_at, rover):
        """Gets the current waypoint of a rover from the state."""
        return state_rover_at.get(rover)

    def is_store_full(self, state_store_full, store):
        """Checks if a store is full in the current state."""
        return store in state_store_full

    def is_camera_calibrated(self, state_calibrated_cams, camera, rover):
        """Checks if a camera on a rover is calibrated in the current state."""
        return (camera, rover) in state_calibrated_cams

    def cost_to_move(self, rover, start_wp, end_wp):
        """Gets the precomputed shortest path cost (navigate actions) between two waypoints for a rover."""
        if rover not in self.rover_shortest_paths or start_wp not in self.rover_shortest_paths[rover]:
            return math.inf # Rover cannot move or start_wp is invalid/isolated
        return self.rover_shortest_paths[rover][start_wp].get(end_wp, math.inf)

    def cost_to_reach_comm_wp(self, rover, current_wp):
        """Estimates the minimum navigation cost for a rover to reach any communication waypoint."""
        min_cost = math.inf
        if not self.the_comm_wps: return math.inf # No communication waypoints
        if current_wp is None: return math.inf # Rover location unknown

        for comm_wp in self.the_comm_wps:
            cost = self.cost_to_move(rover, current_wp, comm_wp)
            min_cost = min(min_cost, cost)
        return min_cost

    def __call__(self, node):
        """
        Computes the domain-dependent heuristic value for the given state node.
        """
        state = node.state

        # --- State Information ---
        # Parse state facts into dictionaries/sets for quick lookup
        rover_at = {} # rover -> waypoint
        store_full = set() # {store}
        soil_samples_at = set() # {waypoint} (initial state facts, consumed)
        rock_samples_at = set() # {waypoint} (initial state facts, consumed)
        have_soil = set() # {(rover, waypoint)}
        have_rock = set() # {(rover, waypoint)}
        have_image = set() # {(rover, objective, mode)}
        calibrated_cams = set() # {(camera, rover)}
        communicated_soil = set() # {waypoint}
        communicated_rock = set() # {waypoint}
        communicated_image = set() # {(objective, mode)}

        for fact_str in state:
            parsed = parse_fact(fact_str)
            if not parsed: continue
            pred = parsed[0]
            args = parsed[1:]

            if pred == 'at' and len(args) == 2:
                rover_at[args[0]] = args[1]
            elif pred == 'full' and len(args) == 1:
                store_full.add(args[0])
            elif pred == 'at_soil_sample' and len(args) == 1:
                soil_samples_at.add(args[0])
            elif pred == 'at_rock_sample' and len(args) == 1:
                rock_samples_at.add(args[0])
            elif pred == 'have_soil_analysis' and len(args) == 2:
                have_soil.add((args[0], args[1]))
            elif pred == 'have_rock_analysis' and len(args) == 2:
                have_rock.add((args[0], args[1]))
            elif pred == 'have_image' and len(args) == 3:
                have_image.add((args[0], args[1], args[2]))
            elif pred == 'calibrated' and len(args) == 2:
                calibrated_cams.add((args[0], args[1]))
            elif pred == 'communicated_soil_data' and len(args) == 1:
                communicated_soil.add(args[0])
            elif pred == 'communicated_rock_data' and len(args) == 1:
                communicated_rock.add(args[0])
            elif pred == 'communicated_image_data' and len(args) == 2:
                communicated_image.add((args[0], args[1]))

        # --- Heuristic Calculation ---
        h = 0

        # If goal is reached, heuristic is 0
        if self.goals <= state:
            return 0

        # Handle unreachable state (e.g., no lander, no comm waypoints) if communication goals exist
        has_comm_goal = any(g.startswith('(communicated_') for g in self.goals)
        if has_comm_goal and (not self.the_lander or not self.the_comm_wps):
             return math.inf # Cannot communicate if no lander or comm points

        for goal_str in self.goals:
            if goal_str in state:
                continue # Goal already achieved

            parsed_goal = parse_fact(goal_str)
            if not parsed_goal: continue # Should not happen for valid goals

            pred = parsed_goal[0]
            args = parsed_goal[1:]

            if pred == 'communicated_soil_data' and len(args) == 1:
                soil_wp = args[0]
                goal_data_in_state = False
                for r in self.rovers:
                    if (r, soil_wp) in have_soil:
                        goal_data_in_state = True
                        break

                if goal_data_in_state:
                    # Need to communicate existing data
                    min_comm_cost = math.inf
                    for r in self.rovers:
                         if (r, soil_wp) in have_soil:
                             current_wp = self.get_rover_location(rover_at, r)
                             if current_wp:
                                 cost = self.cost_to_reach_comm_wp(r, current_wp)
                                 if cost != math.inf:
                                     min_comm_cost = min(min_comm_cost, cost + 1) # +1 for communicate action
                    if min_comm_cost == math.inf: return math.inf # Cannot communicate
                    h += min_comm_cost
                else:
                    # Need to sample and communicate
                    # Check if sample still exists
                    if soil_wp not in soil_samples_at:
                         # Sample is gone, cannot achieve this goal
                         return math.inf

                    min_total_cost = math.inf
                    # Find best soil-equipped rover with empty store OR full store (needs drop)
                    suitable_rovers = [r for r in self.rovers if r in self.equipped_soil]
                    if not suitable_rovers: return math.inf # Cannot sample soil

                    for r_s in suitable_rovers:
                        store = self.rover_stores.get(r_s)
                        if not store: continue # Rover has no store? (Shouldn't happen in domain)

                        current_wp = self.get_rover_location(rover_at, r_s)
                        if not current_wp: continue # Rover location unknown

                        # Cost to sample
                        move_to_sample_cost = self.cost_to_move(r_s, current_wp, soil_wp)
                        if move_to_sample_cost == math.inf: continue # Cannot reach sample location

                        drop_cost = 0
                        if self.is_store_full(store_full, store):
                            drop_cost = 1 # Need to drop before sampling

                        sample_cost = 1 # sample_soil action

                        # Cost to communicate (assuming from sample location after sampling)
                        comm_move_cost = self.cost_to_reach_comm_wp(r_s, soil_wp)
                        if comm_move_cost == math.inf: continue # Cannot reach comm wp from sample location

                        communicate_cost = 1 # communicate_soil_data action

                        total_cost = move_to_sample_cost + drop_cost + sample_cost + comm_move_cost + communicate_cost
                        min_total_cost = min(min_total_cost, total_cost)

                    if min_total_cost == math.inf: return math.inf # Cannot sample or communicate
                    h += min_total_cost

            elif pred == 'communicated_rock_data' and len(args) == 1:
                rock_wp = args[0]
                goal_data_in_state = False
                for r in self.rovers:
                    if (r, rock_wp) in have_rock:
                        goal_data_in_state = True
                        break

                if goal_data_in_state:
                    # Need to communicate existing data
                    min_comm_cost = math.inf
                    for r in self.rovers:
                         if (r, rock_wp) in have_rock:
                             current_wp = self.get_rover_location(rover_at, r)
                             if current_wp:
                                 cost = self.cost_to_reach_comm_wp(r, current_wp)
                                 if cost != math.inf:
                                     min_comm_cost = min(min_comm_cost, cost + 1) # +1 for communicate action
                    if min_comm_cost == math.inf: return math.inf # Cannot communicate
                    h += min_comm_cost
                else:
                    # Need to sample and communicate
                    # Check if sample still exists
                    if rock_wp not in rock_samples_at:
                         # Sample is gone, cannot achieve this goal
                         return math.inf

                    min_total_cost = math.inf
                    # Find best rock-equipped rover with empty store OR full store (needs drop)
                    suitable_rovers = [r for r in self.rovers if r in self.equipped_rock]
                    if not suitable_rovers: return math.inf # Cannot sample rock

                    for r_s in suitable_rovers:
                        store = self.rover_stores.get(r_s)
                        if not store: continue # Rover has no store?

                        current_wp = self.get_rover_location(rover_at, r_s)
                        if not current_wp: continue # Rover location unknown

                        # Cost to sample
                        move_to_sample_cost = self.cost_to_move(r_s, current_wp, rock_wp)
                        if move_to_sample_cost == math.inf: continue # Cannot reach sample location

                        drop_cost = 0
                        if self.is_store_full(store_full, store):
                            drop_cost = 1 # Need to drop before sampling

                        sample_cost = 1 # sample_rock action

                        # Cost to communicate (assuming from sample location after sampling)
                        comm_move_cost = self.cost_to_reach_comm_wp(r_s, rock_wp)
                        if comm_move_cost == math.inf: continue # Cannot reach comm wp from sample location

                        communicate_cost = 1 # communicate_rock_data action

                        total_cost = move_to_sample_cost + drop_cost + sample_cost + comm_move_cost + communicate_cost
                        min_total_cost = min(min_total_cost, total_cost)

                    if min_total_cost == math.inf: return math.inf # Cannot sample or communicate
                    h += min_total_cost

            elif pred == 'communicated_image_data' and len(args) == 2:
                objective = args[0]
                mode = args[1]
                goal_data_in_state = False
                for r in self.rovers:
                    if (r, objective, mode) in have_image:
                        goal_data_in_state = True
                        break

                if goal_data_in_state:
                    # Need to communicate existing image
                    min_comm_cost = math.inf
                    for r in self.rovers:
                         if (r, objective, mode) in have_image:
                             current_wp = self.get_rover_location(rover_at, r)
                             if current_wp:
                                 cost = self.cost_to_reach_comm_wp(r, current_wp)
                                 if cost != math.inf:
                                     min_comm_cost = min(min_comm_cost, cost + 1) # +1 for communicate action
                    if min_comm_cost == math.inf: return math.inf # Cannot communicate
                    h += min_comm_cost
                else:
                    # Need to image and communicate
                    min_total_cost = math.inf
                    # Find best imaging-equipped rover with suitable camera and visible waypoint
                    suitable_rovers = [r for r in self.rovers if r in self.equipped_imaging]
                    if not suitable_rovers: return math.inf # Cannot take image

                    for r_i in suitable_rovers:
                        current_wp = self.get_rover_location(rover_at, r_i)
                        if not current_wp: continue # Rover location unknown

                        for camera in self.rover_cameras.get(r_i, set()):
                            if mode in self.supports.get(camera, set()):
                                # Found a suitable rover and camera
                                cal_target = self.calibration_target.get(camera)
                                cal_wps = self.visible_from.get(cal_target, set()) if cal_target else set()

                                for image_wp in self.visible_from.get(objective, set()):
                                    # Cost to reach image waypoint, potentially including calibration detour
                                    cost_to_reach_image_wp_with_cal = math.inf

                                    if not self.is_camera_calibrated(calibrated_cams, camera, r_i):
                                        # Need calibration
                                        if not cal_wps: continue # Need calibration but no calibration waypoints

                                        min_cal_detour_cost = math.inf
                                        for cal_wp in cal_wps:
                                            # Cost: current -> cal_wp -> image_wp + calibrate action
                                            cost_curr_to_cal = self.cost_to_move(r_i, current_wp, cal_wp)
                                            cost_cal_to_image = self.cost_to_move(r_i, cal_wp, image_wp)
                                            if cost_curr_to_cal != math.inf and cost_cal_to_image != math.inf:
                                                 min_cal_detour_cost = min(min_cal_detour_cost, cost_curr_to_cal + 1 + cost_cal_to_image) # +1 for calibrate

                                        if min_cal_detour_cost == math.inf: continue # Cannot perform calibration detour
                                        cost_to_reach_image_wp_with_cal = min_cal_detour_cost
                                    else:
                                        # Already calibrated, just need to move to image_wp
                                        cost_to_reach_image_wp_with_cal = self.cost_to_move(r_i, current_wp, image_wp)

                                    if cost_to_reach_image_wp_with_cal == math.inf: continue # Cannot reach image location (even with calibration)

                                    take_image_cost = 1 # take_image action

                                    # Cost to communicate (assuming from image location after imaging)
                                    comm_move_cost = self.cost_to_reach_comm_wp(r_i, image_wp)
                                    if comm_move_cost == math.inf: continue # Cannot reach comm wp from image location

                                    communicate_cost = 1 # communicate_image_data action

                                    total_cost = cost_to_reach_image_wp_with_cal + take_image_cost + comm_move_cost + communicate_cost
                                    min_total_cost = min(min_total_cost, total_cost)

                    if min_total_cost == math.inf: return math.inf # Cannot image or communicate
                    h += min_total_cost

        return h
