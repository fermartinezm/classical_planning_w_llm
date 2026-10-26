# Import necessary modules
from fnmatch import fnmatch
import collections
# Assuming Heuristic base class is available in heuristics.heuristic_base
from heuristics.heuristic_base import Heuristic

# --- Utility Functions ---

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact strings or malformed facts gracefully
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
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

def bfs(adj_list, start_node):
    """Performs Breadth-First Search to find shortest distances from start_node."""
    distances = {node: float('inf') for node in adj_list}
    if start_node in distances: # Ensure start_node is a valid waypoint
        distances[start_node] = 0
        queue = collections.deque([start_node])

        while queue:
            current_node = queue.popleft()

            if current_node in adj_list: # Handle nodes with no outgoing edges
                for neighbor in adj_list[current_node]:
                    if distances[neighbor] == float('inf'):
                        distances[neighbor] = distances[current_node] + 1
                        queue.append(neighbor)
    return distances

def precompute_all_pairs_distances(visible_facts):
    """
    Precomputes shortest path distances between all pairs of waypoints
    based on 'visible' facts using BFS.
    """
    adj_list = collections.defaultdict(list)
    waypoints = set()
    for fact in visible_facts:
        parts = get_parts(fact)
        if len(parts) == 3 and parts[0] == 'visible':
            _, w1, w2 = parts
            adj_list[w1].append(w2)
            waypoints.add(w1)
            waypoints.add(w2)

    # Ensure all waypoints are in the adj_list keys, even if they have no outgoing edges
    for wp in waypoints:
        if wp not in adj_list:
            adj_list[wp] = []

    all_distances = {}
    for start_node in waypoints:
        all_distances[start_node] = bfs(adj_list, start_node)
    return all_distances, list(waypoints) # Return waypoints list for easy iteration

# --- Heuristic Class ---

class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve all
    goal conditions. It sums the estimated cost for each unachieved goal
    fact independently, ignoring most negative interactions and some positive
    interactions (like sharing navigation paths).

    # Assumptions
    - Navigation cost between waypoints is the shortest path distance in the
      graph defined by the 'visible' predicate. It assumes 'can_traverse'
      is symmetric and covers all 'visible' links for all rovers.
    - Stores are always available when needed for sampling (ignores 'full' state).
    - Cameras are always available when needed for imaging (ignores 'calibrated' state
      after taking an image, adding a fixed cost for calibration instead).
    - A suitable communication waypoint (visible from the lander) is always reachable.
    - Assumes solvable instances (finite heuristic value).

    # Heuristic Initialization
    - Extracts goal conditions.
    - Extracts static facts: lander location, rover equipment, camera properties
      (on-board, supports modes, calibration target), objective visibility from
      waypoints, and waypoint visibility ('visible').
    - Precomputes shortest path distances between all pairs of waypoints based
      on the 'visible' graph using BFS.
    - Identifies communication waypoints (visible from the lander).

    # Step-By-Step Thinking for Computing Heuristic
    The heuristic value is the sum of estimated costs for each goal fact
    that is not yet satisfied in the current state.

    For each unachieved goal fact G:

    1.  If G is `(communicated_soil_data W)`:
        - Check if any rover R currently has the soil analysis: `(have_soil_analysis R W)` is true.
            - If yes: The cost is 1 (for the 'communicate_soil_data' action) plus the minimum navigation cost for rover R from its current location to *any* waypoint visible from the lander's location.
            - If no (the soil sample needs to be collected and communicated):
                - Find a rover R that is `equipped_for_soil_analysis`.
                - The cost for this goal is estimated as:
                  1 (for 'sample_soil') + 1 (for 'communicate_soil_data') +
                  minimum navigation cost for R from its current location to waypoint W +
                  minimum navigation cost from waypoint W to *any* waypoint visible from the lander's location.
                - The minimum cost is taken over all suitable rovers R.

    2.  If G is `(communicated_rock_data W)`:
        - Similar logic as for soil data, replacing 'soil' with 'rock' and checking for `equipped_for_rock_analysis`.

    3.  If G is `(communicated_image_data O M)`:
        - Check if any rover R currently has the image: `(have_image R O M)` is true.
            - If yes: The cost is 1 (for 'communicate_image_data') plus the minimum navigation cost for rover R from its current location to *any* waypoint visible from the lander's location.
            - If no (the image needs to be taken and communicated):
                - Find a rover R `equipped_for_imaging` that has a camera I `on_board` supporting mode M (`supports I M`).
                - Find a waypoint P from which objective O is visible (`visible_from O P`).
                - Find the calibration target T for camera I (`calibration_target I T`).
                - Find a waypoint W from which the calibration target T is visible (`visible_from T W`).
                - The cost for this goal is estimated as:
                  1 (for 'calibrate') + 1 (for 'take_image') + 1 (for 'communicate_image_data') +
                  minimum navigation cost for R from its current location to a suitable calibration waypoint W +
                  minimum navigation cost from W to a suitable imaging waypoint P +
                  minimum navigation cost from P to *any* waypoint visible from the lander's location.
                - The minimum cost is taken over all suitable rovers R, cameras I, imaging waypoints P, and calibration waypoints W.

    The total heuristic value is the sum of these minimum costs for all unachieved goal facts. If a goal fact is unreachable (e.g., no rover has the necessary equipment or required waypoints are unreachable), its cost is considered infinite (or a very large number), although the problem description implies solvable instances.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static # Store static facts for lookup
        static_facts = self.static

        # Precompute waypoint distances based on 'visible' graph
        visible_facts = [fact for fact in static_facts if match(fact, "visible", "*", "*")]
        self.all_pairs_distances, self.waypoints = precompute_all_pairs_distances(visible_facts)

        # Extract static mappings
        self.lander_location = None
        lander_name = None
        for fact in static_facts:
            if match(fact, "at_lander", "*", "*"):
                lander_name, self.lander_location = get_parts(fact)[1:]
                break # Assuming only one lander

        self.communication_waypoints = set()
        if self.lander_location:
             for fact in static_facts:
                 # A waypoint X is a communication waypoint if visible from lander location Y
                 if match(fact, "visible", "*", self.lander_location):
                     self.communication_waypoints.add(get_parts(fact)[1])
                 # Also check the symmetric case if visible is not guaranteed symmetric in facts
                 if match(fact, "visible", self.lander_location, "*"):
                      self.communication_waypoints.add(get_parts(fact)[2])


        self.rover_equipment = collections.defaultdict(set)
        for fact in static_facts:
            if match(fact, "equipped_for_soil_analysis", "*"):
                self.rover_equipment[get_parts(fact)[1]].add("soil")
            elif match(fact, "equipped_for_rock_analysis", "*"):
                self.rover_equipment[get_parts(fact)[1]].add("rock")
            elif match(fact, "equipped_for_imaging", "*"):
                self.rover_equipment[get_parts(fact)[1]].add("imaging")

        self.rover_cameras = collections.defaultdict(set) # rover -> {camera}
        self.camera_modes = collections.defaultdict(set) # camera -> {mode}
        self.camera_calibration_target = {} # camera -> target
        for fact in static_facts:
            if match(fact, "on_board", "*", "*"):
                camera, rover = get_parts(fact)[1:]
                self.rover_cameras[rover].add(camera)
            elif match(fact, "supports", "*", "*"):
                camera, mode = get_parts(fact)[1:]
                self.camera_modes[camera].add(mode)
            elif match(fact, "calibration_target", "*", "*"):
                camera, target = get_parts(fact)[1:]
                self.camera_calibration_target[camera] = target

        self.visible_from_waypoints = collections.defaultdict(set) # obj/target -> {waypoint}
        for fact in static_facts:
             if match(fact, "visible_from", "*", "*"):
                 obj_or_target, waypoint = get_parts(fact)[1:]
                 self.visible_from_waypoints[obj_or_target].add(waypoint)

        # Pre-calculate min distance from any waypoint to any communication waypoint
        self.min_dist_to_comm_wp = {}
        for wp in self.waypoints:
            min_dist = float('inf')
            if wp in self.all_pairs_distances:
                for comm_wp in self.communication_waypoints:
                    if comm_wp in self.all_pairs_distances[wp]:
                         min_dist = min(min_dist, self.all_pairs_distances[wp][comm_wp])
            self.min_dist_to_comm_wp[wp] = min_dist


    def get_distance(self, wp1, wp2):
        """Get the precomputed shortest distance between two waypoints."""
        if wp1 is None or wp2 is None: return float('inf') # Cannot calculate distance from/to None
        if wp1 in self.all_pairs_distances and wp2 in self.all_pairs_distances[wp1]:
            return self.all_pairs_distances[wp1][wp2]
        return float('inf') # Not reachable


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        total_cost = 0

        # Track achieved goals for quick lookup
        achieved_goals = set(state)

        # Find current rover locations
        rover_locations = {}
        # Iterate through all 'at' facts and identify rovers
        for fact in state:
            if match(fact, "at", "*", "*"):
                 parts = get_parts(fact)
                 # A simple way to identify rovers is by checking if they are in rover_equipment keys
                 # A more robust way would be to get object types from the task/instance
                 obj_name = parts[1]
                 if obj_name in self.rover_equipment: # Check if this object is a known rover
                     waypoint = parts[2]
                     rover_locations[obj_name] = waypoint


        # Iterate through goal facts
        for goal in self.goals:
            if goal in achieved_goals:
                continue # Goal already achieved

            parts = get_parts(goal)
            if not parts: continue # Skip malformed goals

            predicate = parts[0]

            if predicate == "communicated_soil_data":
                waypoint_w = parts[1]
                min_goal_cost = float('inf')

                # Case 1: Soil analysis already exists
                have_analysis = False
                for fact in state:
                    if match(fact, "have_soil_analysis", "*", waypoint_w):
                        rover_r = get_parts(fact)[1]
                        if rover_r in rover_locations: # Ensure we know the rover's location
                            current_pos_r = rover_locations[rover_r]
                            dist_to_comm = self.min_dist_to_comm_wp.get(current_pos_r, float('inf'))
                            if dist_to_comm != float('inf'):
                                min_goal_cost = min(min_goal_cost, 1 + dist_to_comm) # 1 for communicate
                                have_analysis = True # Found at least one rover with analysis

                # Case 2: Need to sample and communicate
                if not have_analysis:
                    for rover_r, equipment in self.rover_equipment.items():
                        if "soil" in equipment and rover_r in rover_locations:
                            current_pos_r = rover_locations[rover_r]
                            dist_to_sample_wp = self.get_distance(current_pos_r, waypoint_w)
                            dist_from_sample_to_comm = self.min_dist_to_comm_wp.get(waypoint_w, float('inf'))

                            if dist_to_sample_wp != float('inf') and dist_from_sample_to_comm != float('inf'):
                                # Cost: navigate to sample + sample + navigate to comm + communicate
                                cost = dist_to_sample_wp + 1 + dist_from_sample_to_comm + 1
                                min_goal_cost = min(min_goal_cost, cost)

                if min_goal_cost != float('inf'):
                    total_cost += min_goal_cost
                else:
                    # This goal is likely unreachable in this state configuration
                    # Add a large penalty
                    total_cost += 1000 # Arbitrary large cost for unreachable goals

            elif predicate == "communicated_rock_data":
                waypoint_w = parts[1]
                min_goal_cost = float('inf')

                # Case 1: Rock analysis already exists
                have_analysis = False
                for fact in state:
                    if match(fact, "have_rock_analysis", "*", waypoint_w):
                        rover_r = get_parts(fact)[1]
                        if rover_r in rover_locations:
                            current_pos_r = rover_locations[rover_r]
                            dist_to_comm = self.min_dist_to_comm_wp.get(current_pos_r, float('inf'))
                            if dist_to_comm != float('inf'):
                                min_goal_cost = min(min_goal_cost, 1 + dist_to_comm) # 1 for communicate
                                have_analysis = True

                # Case 2: Need to sample and communicate
                if not have_analysis:
                    for rover_r, equipment in self.rover_equipment.items():
                        if "rock" in equipment and rover_r in rover_locations:
                            current_pos_r = rover_locations[rover_r]
                            dist_to_sample_wp = self.get_distance(current_pos_r, waypoint_w)
                            dist_from_sample_to_comm = self.min_dist_to_comm_wp.get(waypoint_w, float('inf'))

                            if dist_to_sample_wp != float('inf') and dist_from_sample_to_comm != float('inf'):
                                # Cost: navigate to sample + sample + navigate to comm + communicate
                                cost = dist_to_sample_wp + 1 + dist_from_sample_to_comm + 1
                                min_goal_cost = min(min_goal_cost, cost)

                if min_goal_cost != float('inf'):
                    total_cost += min_goal_cost
                else:
                    total_cost += 1000 # Penalty

            elif predicate == "communicated_image_data":
                objective_o, mode_m = parts[1:]
                min_goal_cost = float('inf')

                # Case 1: Image already exists
                have_image = False
                for fact in state:
                    if match(fact, "have_image", "*", objective_o, mode_m):
                        rover_r = get_parts(fact)[1]
                        if rover_r in rover_locations:
                            current_pos_r = rover_locations[rover_r]
                            dist_to_comm = self.min_dist_to_comm_wp.get(current_pos_r, float('inf'))
                            if dist_to_comm != float('inf'):
                                min_goal_cost = min(min_goal_cost, 1 + dist_to_comm) # 1 for communicate
                                have_image = True

                # Case 2: Need to take image and communicate
                if not have_image:
                    # Iterate through suitable rovers, cameras, imaging waypoints, calibration waypoints
                    for rover_r, equipment in self.rover_equipment.items():
                        if "imaging" in equipment and rover_r in rover_locations:
                            current_pos_r = rover_locations[rover_r]
                            for camera_i in self.rover_cameras.get(rover_r, set()):
                                if mode_m in self.camera_modes.get(camera_i, set()):
                                    calibration_target_t = self.camera_calibration_target.get(camera_i)
                                    if calibration_target_t:
                                        # Find suitable calibration waypoints W
                                        suitable_cal_wps = self.visible_from_waypoints.get(calibration_target_t, set())
                                        # Find suitable imaging waypoints P
                                        suitable_img_wps = self.visible_from_waypoints.get(objective_o, set())

                                        for cal_wp_w in suitable_cal_wps:
                                            for img_wp_p in suitable_img_wps:
                                                dist_curr_to_cal = self.get_distance(current_pos_r, cal_wp_w)
                                                dist_cal_to_img = self.get_distance(cal_wp_w, img_wp_p)
                                                dist_img_to_comm = self.min_dist_to_comm_wp.get(img_wp_p, float('inf'))

                                                if dist_curr_to_cal != float('inf') and dist_cal_to_img != float('inf') and dist_img_to_comm != float('inf'):
                                                    # Cost: nav to cal + calibrate + nav to img + take image + nav to comm + communicate
                                                    cost = dist_curr_to_cal + 1 + dist_cal_to_img + 1 + dist_img_to_comm + 1
                                                    min_goal_cost = min(min_goal_cost, cost)

                if min_goal_cost != float('inf'):
                    total_cost += min_goal_cost
                else:
                    total_cost += 1000 # Penalty

        # Heuristic should be 0 at goal state.
        # The sum will be 0 if all goals are achieved.
        # Ensure it's non-negative.
        return max(0, total_cost)

