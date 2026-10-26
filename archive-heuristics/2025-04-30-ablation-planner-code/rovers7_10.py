# Assuming heuristics.heuristic_base exists and provides a Heuristic base class
# from heuristics.heuristic_base import Heuristic

# Dummy Heuristic base class for testing purposes if the actual one is not provided
class Heuristic:
    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static
    def __call__(self, node):
        raise NotImplementedError

# Assuming the actual Heuristic base class is available in the environment
# If running in an environment where the base class is provided,
# uncomment the line below and remove the dummy class definition.
# from heuristics.heuristic_base import Heuristic

import collections
from fnmatch import fnmatch
import sys # Import sys for float('inf') if needed, but it's a built-in


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
         # Return empty list or handle error as appropriate for expected input
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

def bfs(graph, start_node, all_nodes):
    """
    Perform BFS to find shortest distances from a start_node to all other nodes
    in a graph represented as an adjacency list.

    Args:
        graph: dict[node, set[neighbor_node]]
        start_node: The node to start BFS from.
        all_nodes: A set of all possible nodes in the graph.

    Returns:
        dict[node, distance]: A dictionary mapping each reachable node to its distance from start_node.
                              Unreachable nodes have distance float('inf').
    """
    distances = {node: float('inf') for node in all_nodes}
    if start_node not in all_nodes:
        # Start node is not in the graph, no paths possible
        return distances

    distances[start_node] = 0
    queue = collections.deque([start_node])

    while queue:
        current_node = queue.popleft()
        current_dist = distances[current_node]

        # Check if current_node is in the graph keys before accessing neighbors
        # A node might be in all_nodes but have no outgoing edges in the graph
        if current_node in graph:
            for neighbor in graph[current_node]:
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = current_dist + 1
                    queue.append(neighbor)

    return distances


class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the total number of actions required to achieve all
    ungratified goal conditions. It calculates the minimum cost for each goal
    independently, considering the necessary steps (sampling/imaging and communication)
    and the navigation costs between relevant waypoints using precomputed shortest paths.

    # Assumptions
    - The heuristic calculates the cost for each ungratified goal independently.
    - It assumes that any rover with the necessary equipment can perform the required
      actions for a goal.
    - Navigation costs are calculated using shortest paths on the static waypoint
      graph for each rover.
    - For sequential steps within a goal (e.g., calibrate -> image -> communicate),
      the navigation cost is calculated based on the rover's location after the previous step
      in that specific goal's plan.
    - If a required waypoint or path is unreachable by any relevant rover, the goal
      is considered unreachable, and the heuristic returns infinity.
    - The heuristic is non-admissible as it ignores potential conflicts and synergies
      between goals and rovers.

    # Heuristic Initialization
    - Parses static facts to identify:
        - Lander location and communication waypoints.
        - Rover capabilities (soil, rock, imaging).
        - Rover stores and cameras.
        - Camera properties (modes, calibration targets).
        - Objective properties (image waypoints, calibration waypoints).
        - Sample locations (soil, rock).
        - Rover traversal graphs (waypoint connectivity).
    - Precomputes all-pairs shortest path distances for each rover on its traversal
      graph using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify all goal conditions that are not yet satisfied in the current state.
    2. Initialize the total heuristic cost to 0.
    3. For each ungratified goal:
        a. Calculate the minimum cost to achieve this specific goal from the current state,
           considering all relevant rovers and necessary intermediate steps (like sampling,
           imaging, calibration, dropping samples).
        b. The cost calculation for a goal involves summing the costs of sequential stages
           (e.g., acquire data -> communicate data). Navigation costs between stages are
           calculated based on the rover's location after the previous stage in the
           minimum-cost plan for *that specific goal*.
        c. The minimum cost for a goal is the minimum cost across all rovers capable
           of achieving that goal.
        d. If a goal is determined to be unreachable from the current state (e.g., required
           waypoint is inaccessible), the minimum cost for that goal is infinity.
        e. Add the minimum cost for this goal to the total heuristic cost.
    4. Return the total heuristic cost. If the total cost is infinity, return infinity.

    Specific Goal Cost Calculation Details:

    - Goal `(communicated_soil_data W)` not in state:
        - Find minimum cost over all soil-equipped rovers R:
            - Cost_R = 0
            - If `(have_soil_analysis R W)` not in state:
                - Add cost for sampling: (maybe drop) + Nav(Current->W) + Sample(1).
                - Rover is at W after sampling.
                - Navigation starts from W for communication.
            - Else (`(have_soil_analysis R W)` is in state):
                - Navigation starts from CurrentR_loc for communication.
            - Add cost for communication: Nav(Location_after_sample->CommWp) + Communicate(1).
            - Minimize over all CommWps visible from lander.
        - Take the minimum Cost_R over all soil-equipped rovers.

    - Goal `(communicated_rock_data W)` not in state: Analogous to soil data.

    - Goal `(communicated_image_data O M)` not in state:
        - Find minimum cost over all imaging-equipped rovers R with camera I supporting mode M:
            - Cost_R = 0
            - If `(have_image R O M)` not in state:
                - Add cost for image acquisition:
                    - If camera I not calibrated: Nav(Current->CalibWp) + Calibrate(1) + Nav(CalibWp->ImageWp) + TakeImage(1). Minimize over CalibWps and ImageWps.
                    - If camera I calibrated: Nav(Current->ImageWp) + TakeImage(1). Minimize over ImageWps.
                - Rover is at ImageWp after taking image.
                - Navigation starts from ImageWp for communication.
            - Else (`(have_image R O M)` is in state):
                - Navigation starts from CurrentR_loc for communication.
            - Add cost for communication: Nav(Location_after_image->CommWp) + Communicate(1).
            - Minimize over all CommWps visible from lander.
        - Take the minimum Cost_R over all relevant rovers.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        super().__init__(task)

        # Store goals
        self.goals = task.goals

        # --- Parse Static Facts ---
        self.lander_loc = None
        self.comm_wps = set()
        self.rover_capabilities = collections.defaultdict(set)
        self.rover_stores = {} # rover -> store
        self.rover_cameras = collections.defaultdict(set) # rover -> set(cameras)
        self.camera_modes = collections.defaultdict(set) # camera -> set(modes)
        self.camera_calibration_targets = {} # camera -> objective
        self.objective_image_wps = collections.defaultdict(set) # objective -> set(waypoints)
        self.objective_calibration_wps = collections.defaultdict(set) # objective -> set(waypoints)
        self.soil_sample_wps = set()
        self.rock_sample_wps = set()
        self.rovers = set()
        self.waypoints = set()
        self.stores = set()
        self.cameras = set()
        self.objectives = set()
        self.modes = set()
        self.landers = set()

        # Identify all objects and their types/relationships from static facts
        for fact in self.static:
             parts = get_parts(fact)
             if not parts: continue # Skip invalid facts

             if parts[0] == 'at_lander' and len(parts) == 3:
                 self.landers.add(parts[1])
                 self.waypoints.add(parts[2])
                 self.lander_loc = parts[2] # Assuming only one lander
             elif parts[0].startswith('equipped_for_') and len(parts) == 2:
                 self.rovers.add(parts[1])
                 self.rover_capabilities[parts[1]].add(parts[0])
             elif parts[0] == 'store_of' and len(parts) == 3:
                 self.stores.add(parts[1])
                 self.rovers.add(parts[2])
                 self.rover_stores[parts[2]] = parts[1]
             elif parts[0] == 'on_board' and len(parts) == 3:
                 self.cameras.add(parts[1])
                 self.rovers.add(parts[2])
                 self.rover_cameras[parts[2]].add(parts[1])
             elif parts[0] == 'supports' and len(parts) == 3:
                 self.cameras.add(parts[1])
                 self.modes.add(parts[2])
                 self.camera_modes[parts[1]].add(parts[2])
             elif parts[0] == 'calibration_target' and len(parts) == 3:
                 self.cameras.add(parts[1])
                 self.objectives.add(parts[2])
                 self.camera_calibration_targets[parts[1]] = parts[2]
             elif parts[0] == 'visible_from' and len(parts) == 3:
                 self.objectives.add(parts[1])
                 self.waypoints.add(parts[2])
                 self.objective_image_wps[parts[1]].add(parts[2])
                 # Calibration targets are also objectives, visible_from applies to them too
                 self.objective_calibration_wps[parts[1]].add(parts[2])
             elif parts[0] == 'at_soil_sample' and len(parts) == 2:
                 self.waypoints.add(parts[1])
                 self.soil_sample_wps.add(parts[1])
             elif parts[0] == 'at_rock_sample' and len(parts) == 2:
                 self.waypoints.add(parts[1])
                 self.rock_sample_wps.add(parts[1])
             elif parts[0] == 'visible' and len(parts) == 3:
                 self.waypoints.add(parts[1])
                 self.waypoints.add(parts[2])
             elif parts[0] == 'can_traverse' and len(parts) == 4:
                 self.rovers.add(parts[1])
                 self.waypoints.add(parts[2])
                 self.waypoints.add(parts[3])
             # Note: 'at' facts for initial rover locations are in the initial state, not static.

        # Identify communication waypoints (visible from lander location)
        if self.lander_loc:
            for fact in self.static:
                if match(fact, "visible", "*", self.lander_loc):
                    self.comm_wps.add(get_parts(fact)[1])
                # Assuming visible is symmetric or defined in both directions if traversable
                if match(fact, "visible", self.lander_loc, "*"):
                     self.comm_wps.add(get_parts(fact)[2])


        # Build traversal graphs for each rover
        self.rover_traversal_graphs = collections.defaultdict(lambda: collections.defaultdict(set))
        for fact in self.static:
            if match(fact, "can_traverse", "*", "*", "*"):
                r, w1, w2 = get_parts(fact)[1:]
                self.rover_traversal_graphs[r][w1].add(w2)

        # Precompute all-pairs shortest paths for each rover
        self.rover_waypoint_distances = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(lambda: float('inf'))))
        for rover in self.rovers:
            graph = self.rover_traversal_graphs.get(rover, {})
            for start_wp in self.waypoints:
                 distances_from_start = bfs(graph, start_wp, self.waypoints)
                 for end_wp, dist in distances_from_start.items():
                     self.rover_waypoint_distances[rover][start_wp][end_wp] = dist


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # --- Parse Current State Facts ---
        rover_locations = {}
        store_status = {} # store -> 'empty' or 'full'
        rover_samples = set() # set of (have_soil_analysis R W) or (have_rock_analysis R W)
        rover_images = set() # set of (have_image R O M)
        camera_calibrated = set() # set of (calibrated I R)
        communicated_data = set() # set of communicated facts

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip invalid facts

            if parts[0] == 'at' and len(parts) == 3 and parts[1] in self.rovers:
                 rover_locations[parts[1]] = parts[2]
            elif parts[0] in ['empty', 'full'] and len(parts) == 2 and parts[1] in self.stores:
                 store_status[parts[1]] = parts[0]
            elif parts[0] in ['have_soil_analysis', 'have_rock_analysis'] and len(parts) == 3:
                 rover_samples.add(fact)
            elif parts[0] == 'have_image' and len(parts) == 4:
                 rover_images.add(fact)
            elif parts[0] == 'calibrated' and len(parts) == 3:
                 camera_calibrated.add(fact)
            elif parts[0].startswith('communicated_') and len(parts) in [2, 3]:
                 communicated_data.add(fact)

        total_cost = 0  # Initialize action cost counter.

        # --- Calculate Cost for Ungratified Goals ---
        for goal in self.goals:
            if goal in communicated_data:
                continue # Goal already achieved

            parts = get_parts(goal)
            if not parts: continue # Skip invalid goal format

            goal_type = parts[0]

            if goal_type == 'communicated_soil_data' and len(parts) == 2:
                waypoint_w = parts[1]
                min_goal_cost = float('inf')

                # Find min cost over all soil-equipped rovers
                for rover in self.rovers:
                    if 'equipped_for_soil_analysis' not in self.rover_capabilities.get(rover, set()):
                        continue # Rover cannot perform soil analysis

                    current_rover_loc = rover_locations.get(rover)
                    if current_rover_loc is None: continue # Rover location unknown

                    cost_r = 0
                    acquire_end_loc = current_rover_loc # Default end loc if no acquisition needed

                    # Cost to acquire sample (if needed)
                    # Check if sample is held by *this* rover
                    sample_acquired_by_this_rover = match(f'(have_soil_analysis {rover} {waypoint_w})', 'have_soil_analysis', rover, waypoint_w) in rover_samples

                    if not sample_acquired_by_this_rover:
                        store_s = self.rover_stores.get(rover)
                        if store_s is None: continue # Rover has no store? Should not happen.

                        if store_status.get(store_s) == 'full':
                            cost_r += 1 # Drop action

                        dist_to_sample = self.rover_waypoint_distances[rover][current_rover_loc][waypoint_w]
                        if dist_to_sample == float('inf'): continue # Cannot reach sample point

                        cost_r += dist_to_sample + 1 # Nav + Sample
                        acquire_end_loc = waypoint_w # Rover is at W after sampling
                    # If sample is already acquired by this rover, acquire_end_loc remains current_rover_loc

                    # Cost to communicate
                    min_dist_from_acquire_end_to_comm = float('inf')
                    if self.comm_wps:
                        min_dist_from_acquire_end_to_comm = min(
                            self.rover_waypoint_distances[rover][acquire_end_loc][comm_wp]
                            for comm_wp in self.comm_wps
                            if self.rover_waypoint_distances[rover][acquire_end_loc][comm_wp] != float('inf')
                        )

                    if min_dist_from_acquire_end_to_comm == float('inf'): continue # Cannot reach any comm point

                    cost_r += min_dist_from_acquire_end_to_comm + 1 # Nav + Communicate
                    min_goal_cost = min(min_goal_cost, cost_r)

                if min_goal_cost == float('inf'):
                    return float('inf') # Goal is unreachable
                total_cost += min_goal_cost

            elif goal_type == 'communicated_rock_data' and len(parts) == 2:
                waypoint_w = parts[1]
                min_goal_cost = float('inf')

                # Find min cost over all rock-equipped rovers
                for rover in self.rovers:
                    if 'equipped_for_rock_analysis' not in self.rover_capabilities.get(rover, set()):
                        continue # Rover cannot perform rock analysis

                    current_rover_loc = rover_locations.get(rover)
                    if current_rover_loc is None: continue # Rover location unknown

                    cost_r = 0
                    acquire_end_loc = current_rover_loc # Default end loc if no acquisition needed

                    # Cost to acquire sample (if needed)
                    # Check if sample is held by *this* rover
                    sample_acquired_by_this_rover = match(f'(have_rock_analysis {rover} {waypoint_w})', 'have_rock_analysis', rover, waypoint_w) in rover_samples

                    if not sample_acquired_by_this_rover:
                        store_s = self.rover_stores.get(rover)
                        if store_s is None: continue # Rover has no store? Should not happen.

                        if store_status.get(store_s) == 'full':
                            cost_r += 1 # Drop action

                        dist_to_sample = self.rover_waypoint_distances[rover][current_rover_loc][waypoint_w]
                        if dist_to_sample == float('inf'): continue # Cannot reach sample point

                        cost_r += dist_to_sample + 1 # Nav + Sample
                        acquire_end_loc = waypoint_w # Rover is at W after sampling
                    # If sample is already acquired by this rover, acquire_end_loc remains current_rover_loc

                    # Cost to communicate
                    min_dist_from_acquire_end_to_comm = float('inf')
                    if self.comm_wps:
                         min_dist_from_acquire_end_to_comm = min(
                            self.rover_waypoint_distances[rover][acquire_end_loc][comm_wp]
                            for comm_wp in self.comm_wps
                            if self.rover_waypoint_distances[rover][acquire_end_loc][comm_wp] != float('inf')
                        )

                    if min_dist_from_acquire_end_to_comm == float('inf'): continue # Cannot reach any comm point

                    cost_r += min_dist_from_acquire_end_to_comm + 1 # Nav + Communicate
                    min_goal_cost = min(min_goal_cost, cost_r)

                if min_goal_cost == float('inf'):
                    return float('inf') # Goal is unreachable
                total_cost += min_goal_cost

            elif goal_type == 'communicated_image_data' and len(parts) == 3:
                objective_o = parts[1]
                mode_m = parts[2]
                min_goal_cost = float('inf')

                # Find min cost over all imaging-equipped rovers with suitable camera
                for rover in self.rovers:
                    if 'equipped_for_imaging' not in self.rover_capabilities.get(rover, set()):
                        continue # Rover cannot perform imaging

                    # Find camera I on board R supporting M
                    camera_i = None
                    for cam in self.rover_cameras.get(rover, set()):
                        if mode_m in self.camera_modes.get(cam, set()):
                            camera_i = cam
                            break
                    if camera_i is None: continue # Rover R cannot take image O M with mode M

                    current_rover_loc = rover_locations.get(rover)
                    if current_rover_loc is None: continue # Rover location unknown

                    cost_r = 0
                    acquire_end_loc = current_rover_loc # Default end loc if no acquisition needed

                    # Cost to acquire image (if needed)
                    # Check if image is held by *this* rover
                    image_acquired_by_this_rover = match(f'(have_image {rover} {objective_o} {mode_m})', 'have_image', rover, objective_o, mode_m) in rover_images

                    if not image_acquired_by_this_rover:
                        image_wps = self.objective_image_wps.get(objective_o, set())
                        if not image_wps: continue # No waypoint to view objective O from

                        calib_target = self.camera_calibration_targets.get(camera_i)
                        calib_wps = self.objective_calibration_wps.get(calib_target, set()) if calib_target else set()
                        is_calibrated = match(f'(calibrated {camera_i} {rover})', 'calibrated', camera_i, rover) in camera_calibrated

                        if not calib_wps and not is_calibrated: continue # Cannot calibrate if no target/waypoint and not already calibrated

                        min_acquire_sequence_cost = float('inf')

                        if not is_calibrated:
                            # Need calibration: Nav(Current->wp_c) + Calib + Nav(wp_c->wp_i) + Image
                            for wp_c in calib_wps:
                                for wp_i in image_wps:
                                    dist1 = self.rover_waypoint_distances[rover][current_rover_loc][wp_c]
                                    dist2 = self.rover_waypoint_distances[rover][wp_c][wp_i]
                                    if dist1 != float('inf') and dist2 != float('inf'):
                                        cost = dist1 + 1 + dist2 + 1
                                        if cost < min_acquire_sequence_cost:
                                            min_acquire_sequence_cost = cost
                                            acquire_end_loc = wp_i # Rover ends at wp_i
                        else:
                            # Already calibrated: Nav(Current->wp_i) + Image
                            for wp_i in image_wps:
                                dist = self.rover_waypoint_distances[rover][current_rover_loc][wp_i]
                                if dist != float('inf'):
                                    cost = dist + 1
                                    if cost < min_acquire_sequence_cost:
                                        min_acquire_sequence_cost = cost
                                        acquire_end_loc = wp_i # Rover ends at wp_i

                        if min_acquire_sequence_cost == float('inf'): continue # Cannot acquire image
                        cost_r += min_acquire_sequence_cost
                    # If image is already acquired by this rover, acquire_end_loc remains current_rover_loc

                    # Cost to communicate
                    min_dist_from_acquire_end_to_comm = float('inf')
                    if self.comm_wps:
                        min_dist_from_acquire_end_to_comm = min(
                            self.rover_waypoint_distances[rover][acquire_end_loc][comm_wp]
                            for comm_wp in self.comm_wps
                            if self.rover_waypoint_distances[rover][acquire_end_loc][comm_wp] != float('inf')
                        )

                    if min_dist_from_acquire_end_to_comm == float('inf'): continue # Cannot reach any comm point

                    cost_r += min_dist_from_acquire_end_to_comm + 1 # Nav + Communicate
                    min_goal_cost = min(min_goal_cost, cost_r)

                if min_goal_cost == float('inf'):
                    return float('inf') # Goal is unreachable
                total_cost += min_goal_cost

        # Return infinity if any goal was unreachable, otherwise return the sum
        return total_cost if total_cost != float('inf') else float('inf')
