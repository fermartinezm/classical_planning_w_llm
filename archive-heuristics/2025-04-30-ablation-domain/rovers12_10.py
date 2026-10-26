from fnmatch import fnmatch
from collections import deque, defaultdict
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and not empty
    if not isinstance(fact, str) or len(fact) < 2:
        return []
    # Remove outer parentheses and split by whitespace
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

def bfs(graph, start, end):
    """
    Performs a Breadth-First Search to find the shortest path distance
    between two nodes in a graph.

    Args:
        graph: An adjacency list representation of the graph (dict: node -> set of neighbors).
        start: The starting node.
        end: The target node.

    Returns:
        The shortest distance (number of edges) from start to end, or float('inf') if unreachable.
    """
    if start == end:
        return 0
    if start not in graph or end not in graph:
        # Handle cases where start or end might not be in the graph (e.g., invalid state or goal)
        # Although with proper graph building from all waypoints, this should be rare.
        return float('inf')

    queue = deque([(start, 0)])
    visited = {start}

    while queue:
        current_wp, dist = queue.popleft()

        if current_wp == end:
            return dist

        for neighbor in graph.get(current_wp, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))

    return float('inf') # Not reachable

class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the minimum number of actions required to satisfy
    all goal conditions. It calculates the cost for each unsatisfied goal
    independently and sums these costs. For each goal, it finds the minimum
    cost path involving movement, sampling/imaging, and communication,
    considering available rovers and resources.

    # Assumptions
    - The cost of each action (move, take_sample, calibrate, take_image, communicate) is 1.
    - The traversal graph is derived from 'visible' facts and is the same for all rovers.
    - A rover's store must be empty to take a new sample. This heuristic simplifies by assuming
      a full store can eventually be emptied (by reaching the lander and communicating),
      and doesn't explicitly model the cost of freeing the store for sampling goals.
    - Image goals require calibration, which requires finding a calibration target visible
      from some waypoint.
    - Goals are independent and can be pursued in parallel by different rovers. The heuristic
      sums the minimum cost for each goal across all suitable rovers.

    # Heuristic Initialization
    - Extracts static information from the task:
        - The lander's fixed waypoint.
        - Rover capabilities (soil, rock, imaging).
        - Mapping of stores to rovers.
        - Locations of soil and rock samples.
        - Visibility relationships between objectives/calibration targets and waypoints.
        - Camera properties (on-board rover, supported modes, calibration target).
        - The waypoint graph based on 'visible' facts for calculating distances using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1.  Identify all goal conditions that are not yet satisfied in the current state.
    2.  Initialize the total heuristic cost to 0.
    3.  Parse the current state to determine:
        - The current location of each rover.
        - Which samples/images each rover 'have's.
        - Which cameras are calibrated on which rovers.
        - The status (empty/full) of each rover's store (though simplified in cost calculation).
    4.  For each unsatisfied goal:
        - Determine the type of goal (communicated_soil_data, communicated_rock_data, communicated_image_data).
        - Calculate the minimum cost to achieve this specific goal from the current state, considering all available and capable rovers.
        - **For a `communicated_soil_data(?w)` goal:**
            - If the sample `(at_soil_sample ?w)` doesn't exist statically, the goal is impossible; add a large penalty.
            - Otherwise, iterate through rovers equipped for soil analysis:
                - If the rover already `have_soil_analysis(?r, ?w)`: Cost is `Distance(rover_loc, lander_wp) + 1 (communicate)`.
                - If not: Cost is `Distance(rover_loc, ?w) + 1 (take_sample) + Distance(?w, lander_wp) + 1 (communicate)`. (Store constraint is simplified).
            - Take the minimum cost over all suitable rovers. If no suitable rover or path, add a large penalty.
        - **For a `communicated_rock_data(?w)` goal:**
            - Similar logic to soil data, using rock-specific predicates and capabilities.
        - **For a `communicated_image_data(?o, ?m)` goal:**
            - If objective `?o` is not visible from any waypoint statically, the goal is impossible; add a large penalty.
            - Otherwise, iterate through rovers equipped for imaging:
                - Find a camera on this rover supporting mode `?m`. If none, skip rover.
                - Find a waypoint `?w_img` where `?o` is visible. If none, skip rover/camera combination.
                - If the rover already `have_image(?r, ?o, ?m)`: Cost is `Distance(rover_loc, lander_wp) + 1 (communicate)`.
                - If not:
                    - Check if the camera is calibrated on this rover.
                    - If calibrated: Cost is `Distance(rover_loc, ?w_img) + 1 (take_image) + Distance(?w_img, lander_wp) + 1 (communicate)`.
                    - If not calibrated: Find the camera's calibration target `?t` and a waypoint `?w_cal` where `?t` is visible. If none, skip camera. Cost is `Distance(rover_loc, ?w_cal) + 1 (calibrate) + Distance(?w_cal, ?w_img) + 1 (take_image) + Distance(?w_img, lander_wp) + 1 (communicate)`.
                - Take the minimum cost over all suitable rovers, cameras, and waypoints (`?w_img`, `?w_cal`). If no suitable combination, add a large penalty.
        - Add the minimum cost calculated for the current goal to the `total_cost`.
    5.  Return the `total_cost`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information from the task.
        """
        self.goals = task.goals  # Goal conditions (frozenset of fact strings)
        static_facts = task.static  # Facts that are true in every state (frozenset of fact strings)

        # --- Extract Static Information ---
        self.lander_waypoint = None
        self.rover_capabilities = defaultdict(set) # rover -> {capabilities} e.g., {'rover1': {'soil', 'imaging'}}
        self.rover_stores = {} # store -> rover e.g., {'rover1store': 'rover1'}
        self.waypoint_soil_samples = set() # {waypoint names}
        self.waypoint_rock_samples = set() # {waypoint names}
        self.objective_visible_from = defaultdict(set) # objective -> {waypoint names}
        self.camera_on_board = {} # camera -> rover e.g., {'camera1': 'rover1'}
        self.camera_supports = defaultdict(set) # camera -> {mode names} e.g., {'camera1': {'high_res', 'low_res'}}
        self.camera_calibration_target = {} # camera -> objective e.g., {'camera1': 'objective1'}
        self.waypoint_graph = defaultdict(set) # waypoint -> {neighbor waypoint names} (for BFS)
        self.all_waypoints = set() # Keep track of all waypoints seen

        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            args = parts[1:]

            if predicate == 'at_lander' and len(args) == 2:
                # (at_lander ?l ?w)
                self.lander_waypoint = args[1]
            elif predicate == 'equipped_for_soil_analysis' and len(args) == 1:
                # (equipped_for_soil_analysis ?r)
                self.rover_capabilities[args[0]].add('soil')
            elif predicate == 'equipped_for_rock_analysis' and len(args) == 1:
                # (equipped_for_rock_analysis ?r)
                self.rover_capabilities[args[0]].add('rock')
            elif predicate == 'equipped_for_imaging' and len(args) == 1:
                # (equipped_for_imaging ?r)
                self.rover_capabilities[args[0]].add('imaging')
            elif predicate == 'store_of' and len(args) == 2:
                # (store_of ?s ?r)
                self.rover_stores[args[0]] = args[1]
            elif predicate == 'at_soil_sample' and len(args) == 1:
                # (at_soil_sample ?w)
                self.waypoint_soil_samples.add(args[0])
            elif predicate == 'at_rock_sample' and len(args) == 1:
                # (at_rock_sample ?w)
                self.waypoint_rock_samples.add(args[0])
            elif predicate == 'visible_from' and len(args) == 2:
                # (visible_from ?o ?w)
                self.objective_visible_from[args[0]].add(args[1])
                self.all_waypoints.add(args[1]) # Waypoints mentioned here are part of the map
            elif predicate == 'on_board' and len(args) == 2:
                # (on_board ?c ?r)
                self.camera_on_board[args[0]] = args[1]
            elif predicate == 'supports' and len(args) == 2:
                # (supports ?c ?m)
                self.camera_supports[args[0]].add(args[1])
            elif predicate == 'calibration_target' and len(args) == 2:
                # (calibration_target ?c ?t)
                self.camera_calibration_target[args[0]] = args[1]
            elif predicate == 'visible' and len(args) == 2:
                 # (visible ?w1 ?w2) - Build graph from visible facts
                 w1, w2 = args[0], args[1]
                 self.waypoint_graph[w1].add(w2)
                 self.waypoint_graph[w2].add(w1) # Assuming visible is bidirectional
                 self.all_waypoints.add(w1)
                 self.all_waypoints.add(w2)
            # Assuming can_traverse facts are redundant if visible exists and is bidirectional,
            # or that all rovers can traverse all visible paths.

        # Ensure all waypoints mentioned in static facts are in the graph keys, even if isolated
        for wp in self.all_waypoints:
             if wp not in self.waypoint_graph:
                 self.waypoint_graph[wp] = set()


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions to reach a goal state.
        """
        state = node.state  # Current world state (frozenset of fact strings)

        # --- Parse Current State ---
        current_rover_locations = {} # rover -> waypoint
        current_rover_inventory = defaultdict(set) # rover -> {have_facts} e.g., {'rover1': {'(have_soil_analysis rover1 waypoint4)'}}
        current_rover_calibrated_cameras = defaultdict(set) # rover -> {camera names} e.g., {'rover1': {'camera1'}}
        current_rover_store_status = {} # rover -> 'empty' or 'full' e.g., {'rover1': 'empty'}
        current_fact_set = set(state) # Convert frozenset to set for faster lookups

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]
            args = parts[1:]

            if predicate == 'at' and len(args) == 2 and args[0].startswith('rover'):
                # (at ?r ?w)
                current_rover_locations[args[0]] = args[1]
            elif predicate in ['have_soil_analysis', 'have_rock_analysis', 'have_image'] and len(args) >= 2:
                 # (have_soil_analysis ?r ?w), (have_rock_analysis ?r ?w), (have_image ?r ?o ?m)
                 rover = args[0]
                 current_rover_inventory[rover].add(fact) # Store the full fact string
            elif predicate == 'calibrated' and len(args) == 2:
                 # (calibrated ?c ?r)
                 camera, rover = args[0], args[1]
                 current_rover_calibrated_cameras[rover].add(camera)
            elif predicate == 'empty' and len(args) == 1 and args[0].endswith('store'):
                 # (empty ?s)
                 store = args[0]
                 rover = self.rover_stores.get(store)
                 if rover:
                     current_rover_store_status[rover] = 'empty'
            elif predicate == 'full' and len(args) == 1 and args[0].endswith('store'):
                 # (full ?s)
                 store = args[0]
                 rover = self.rover_stores.get(store)
                 if rover:
                     current_rover_store_status[rover] = 'full'

        total_cost = 0
        # Penalty for goals that seem impossible based on static facts or current state limitations
        # (e.g., no equipped rover, no sample, objective not visible)
        large_penalty = 1000

        # Helper function for distance calculation using the pre-built graph
        def get_distance(start, end):
             # Ensure start and end are valid waypoints in our graph
             if start not in self.all_waypoints or end not in self.all_waypoints:
                 return float('inf') # Should not happen with valid PDDL instances

             return bfs(self.waypoint_graph, start, end)

        # --- Calculate Cost for Each Ungoal ---
        for goal in self.goals:
            if goal in current_fact_set:
                continue # Goal already achieved

            parts = get_parts(goal)
            if not parts:
                total_cost += large_penalty # Malformed goal?
                continue

            goal_type = parts[0]

            if goal_type == 'communicated_soil_data' and len(parts) == 2:
                waypoint = parts[1]
                min_goal_cost = float('inf')

                # Check static possibility: Does a soil sample exist at this waypoint?
                if f'(at_soil_sample {waypoint})' not in self.waypoint_soil_samples:
                     min_goal_cost = large_penalty # Cannot sample if no sample exists

                if min_goal_cost != large_penalty: # Only proceed if sampling is statically possible
                    for rover, capabilities in self.rover_capabilities.items():
                        if 'soil' in capabilities:
                            rover_loc = current_rover_locations.get(rover)
                            if rover_loc is None: continue # Rover location unknown

                            if f'(have_soil_analysis {rover} {waypoint})' in current_rover_inventory[rover]:
                                # Already have the sample, just need to communicate
                                dist_to_lander = get_distance(rover_loc, self.lander_waypoint)
                                if dist_to_lander != float('inf'):
                                     cost = dist_to_lander + 1 # move + communicate
                                     min_goal_cost = min(min_goal_cost, cost)
                            else:
                                # Need to take sample and communicate
                                # Simplified: Assume store can be made empty if needed.
                                # A more complex heuristic would check store status and add cost to empty it.
                                dist_to_sample = get_distance(rover_loc, waypoint)
                                dist_sample_to_lander = get_distance(waypoint, self.lander_waypoint)
                                if dist_to_sample != float('inf') and dist_sample_to_lander != float('inf'):
                                     cost = dist_to_sample + 1 + dist_sample_to_lander + 1 # move + take + move + communicate
                                     min_goal_cost = min(min_goal_cost, cost)

                if min_goal_cost == float('inf'):
                    total_cost += large_penalty # Goal seems unreachable by any suitable rover
                else:
                    total_cost += min_goal_cost

            elif goal_type == 'communicated_rock_data' and len(parts) == 2:
                 waypoint = parts[1]
                 min_goal_cost = float('inf')

                 # Check static possibility: Does a rock sample exist at this waypoint?
                 if f'(at_rock_sample {waypoint})' not in self.waypoint_rock_samples:
                     min_goal_cost = large_penalty # Cannot sample if no sample exists

                 if min_goal_cost != large_penalty: # Only proceed if sampling is statically possible
                     for rover, capabilities in self.rover_capabilities.items():
                         if 'rock' in capabilities:
                             rover_loc = current_rover_locations.get(rover)
                             if rover_loc is None: continue

                             if f'(have_rock_analysis {rover} {waypoint})' in current_rover_inventory[rover]:
                                 # Already have the sample, just need to communicate
                                 dist_to_lander = get_distance(rover_loc, self.lander_waypoint)
                                 if dist_to_lander != float('inf'):
                                     cost = dist_to_lander + 1 # move + communicate
                                     min_goal_cost = min(min_goal_cost, cost)
                             else:
                                 # Need to take sample and communicate
                                 # Simplified: Assume store can be made empty if needed.
                                 dist_to_sample = get_distance(rover_loc, waypoint)
                                 dist_sample_to_lander = get_distance(waypoint, self.lander_waypoint)
                                 if dist_to_sample != float('inf') and dist_sample_to_lander != float('inf'):
                                     cost = dist_to_sample + 1 + dist_sample_to_lander + 1 # move + take + move + communicate
                                     min_goal_cost = min(min_goal_cost, cost)

                 if min_goal_cost == float('inf'):
                     total_cost += large_penalty # Goal seems unreachable
                 else:
                     total_cost += min_goal_cost

            elif goal_type == 'communicated_image_data' and len(parts) == 3:
                 objective, mode = parts[1], parts[2]
                 min_goal_cost = float('inf')

                 # Check static possibility: Is objective visible from anywhere?
                 suitable_image_waypoints = self.objective_visible_from.get(objective, set())
                 if not suitable_image_waypoints:
                     min_goal_cost = large_penalty # Cannot image if not visible from any waypoint

                 if min_goal_cost != large_penalty: # Only proceed if imaging is statically possible
                     for rover, capabilities in self.rover_capabilities.items():
                         if 'imaging' in capabilities:
                             rover_loc = current_rover_locations.get(rover)
                             if rover_loc is None: continue

                             # Find a camera on this rover that supports the mode
                             suitable_cameras = [
                                 cam for cam, r in self.camera_on_board.items()
                                 if r == rover and mode in self.camera_supports.get(cam, set())
                             ]
                             if not suitable_cameras: continue # No suitable camera on this rover

                             for camera in suitable_cameras:
                                 # Check if camera is calibrated on this rover
                                 is_calibrated = camera in current_rover_calibrated_cameras[rover]

                                 if f'(have_image {rover} {objective} {mode})' in current_rover_inventory[rover]:
                                     # Already have the image, just need to communicate
                                     dist_to_lander = get_distance(rover_loc, self.lander_waypoint)
                                     if dist_to_lander != float('inf'):
                                         cost = dist_to_lander + 1 # move + communicate
                                         min_goal_cost = min(min_goal_cost, cost)
                                 else:
                                     # Need to take image and communicate
                                     # Iterate through possible waypoints to take the image from
                                     for image_wp in suitable_image_waypoints:
                                         if is_calibrated:
                                             # Already calibrated, just need to move to image_wp, take image, move to lander, communicate
                                             dist_to_image_wp = get_distance(rover_loc, image_wp)
                                             dist_image_to_lander = get_distance(image_wp, self.lander_waypoint)
                                             if dist_to_image_wp != float('inf') and dist_image_to_lander != float('inf'):
                                                 cost = dist_to_image_wp + 1 + dist_image_to_lander + 1 # move + take + move + communicate
                                                 min_goal_cost = min(min_goal_cost, cost)
                                         else:
                                             # Need to calibrate first
                                             cal_target = self.camera_calibration_target.get(camera)
                                             if cal_target is None: continue # Camera has no calibration target

                                             # Find a waypoint where the calibration target is visible
                                             suitable_cal_waypoints = self.objective_visible_from.get(cal_target, set())
                                             if not suitable_cal_waypoints: continue # Cal target not visible from anywhere

                                             # Iterate through possible waypoints to calibrate from
                                             for cal_wp in suitable_cal_waypoints:
                                                 dist_to_cal_wp = get_distance(rover_loc, cal_wp)
                                                 dist_cal_to_image_wp = get_distance(cal_wp, image_wp)
                                                 dist_image_to_lander = get_distance(image_wp, self.lander_waypoint)

                                                 if dist_to_cal_wp != float('inf') and dist_cal_to_image_wp != float('inf') and dist_image_to_lander != float('inf'):
                                                     cost = dist_to_cal_wp + 1 + dist_cal_to_image_wp + 1 + dist_image_to_lander + 1 # move + calibrate + move + take + move + communicate
                                                     min_goal_cost = min(min_goal_cost, cost)

                 if min_goal_cost == float('inf'):
                     total_cost += large_penalty # Goal seems unreachable
                 else:
                     total_cost += min_goal_cost

            else:
                # Handle unexpected goal types or malformed goals
                total_cost += large_penalty


        return total_cost

