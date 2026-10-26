# Need to import Heuristic base class. Assuming it's available in a 'heuristics' package.
# from heuristics.heuristic_base import Heuristic
# For standalone testing or environments where Heuristic base is not provided,
# uncomment the following dummy definition:
# class Heuristic:
#     def __init__(self, task):
#         pass
#     def __call__(self, node):
#         raise NotImplementedError

import fnmatch
import collections

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and remove outer parentheses
    if isinstance(fact, str) and fact.startswith('(') and fact.endswith(')'):
        return fact[1:-1].split()
    # Handle cases that are not valid fact strings, though state should contain valid facts
    return []


# Helper function to match PDDL facts with patterns
def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Check if the number of parts matches the number of arguments
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# Helper for BFS shortest path
def get_shortest_path_distances(graph, start_node):
    """
    Performs BFS to find shortest path distances from start_node in a graph.

    Args:
        graph: Adjacency list representation (dict: node -> set of neighbors).
        start_node: The starting node for BFS.

    Returns:
        A dictionary mapping each reachable node to its distance from start_node.
        Nodes not in the graph or unreachable will not be in the dictionary or have inf distance.
    """
    distances = {node: float('inf') for node in graph}
    if start_node not in graph:
         # If start_node is not a known waypoint in the graph, no movement is possible from here.
         # All distances remain infinity.
         return distances

    distances[start_node] = 0
    queue = collections.deque([start_node])
    visited = {start_node}

    while queue:
        current_node = queue.popleft()

        # Ensure current_node is still valid in the graph (should be if start_node was)
        if current_node in graph:
            for neighbor in graph[current_node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)
    return distances

# Helper to get distance from BFS result
def dist(start_wp, end_wp, distances_map):
    """
    Get the distance from start_wp to end_wp using a precomputed distances map.

    Args:
        start_wp: The starting waypoint (implicitly the source of the distances_map).
        end_wp: The target waypoint.
        distances_map: Dictionary returned by get_shortest_path_distances.

    Returns:
        The shortest distance, or float('inf') if unreachable or not in map.
    """
    # The distances_map is specific to the start_wp used in get_shortest_path_distances
    # We just need to look up the end_wp in that map.
    return distances_map.get(end_wp, float('inf'))

# Helper to find min distance to any waypoint visible from lander
def min_dist_to_visible_from_lander(start_wp, lander_wp, rover_graph, visible_map):
    """
    Find the minimum navigation distance for a rover from start_wp to any waypoint
    that is visible from the lander's waypoint.

    Args:
        start_wp: The rover's starting waypoint.
        lander_wp: The lander's waypoint.
        rover_graph: The navigation graph for the specific rover.
        visible_map: Dictionary mapping waypoints to sets of waypoints visible from them.
                     (wp1 -> {wp2, wp3} means wp2 and wp3 are visible FROM wp1)

    Returns:
        The minimum distance, or float('inf') if no such waypoint is reachable.
    """
    # Find all waypoints X such that (visible X lander_wp) is true.
    # This means X is visible FROM lander_wp.
    lander_visible_wps = visible_map.get(lander_wp, set())

    if not lander_visible_wps:
        return float('inf') # No waypoint is visible from the lander

    distances_from_start = get_shortest_path_distances(rover_graph, start_wp)

    min_d = float('inf')
    for comm_wp in lander_visible_wps:
        d = dist(start_wp, comm_wp, distances_from_start)
        min_d = min(min_d, d)
    return min_d

class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to satisfy all
    unachieved goal conditions. It calculates the minimum estimated cost for
    each unachieved goal independently and sums these minimum costs.
    The cost for each goal involves steps like navigation, sampling/imaging,
    calibration (for images), dropping samples (if store is full), and communication.
    Navigation costs are estimated using shortest path distances on the rover's
    waypoint graph.

    # Assumptions
    - Each unachieved goal can be pursued independently by the most suitable rover.
    - The cost of achieving one goal does not significantly interfere with the cost
      of achieving another, except for shared resources like rover location,
      store capacity, and camera calibration state at the moment the sub-goal
      action (sample, take_image) is performed.
    - Samples (soil/rock) are permanently removed from the waypoint once collected.
    - Camera calibration is consumed by the `take_image` action.
    - The heuristic assumes the problem is solvable and relevant waypoints are reachable.
      Unreachable goals (infinite cost estimates) are ignored in the sum, but the
      heuristic returns a non-zero value if the state is not the goal state.

    # Heuristic Initialization
    The constructor extracts static information from the task:
    - Lander location.
    - Rover equipment (soil, rock, imaging).
    - Store ownership for each rover.
    - Waypoint graph for each rover based on `can_traverse`.
    - Visibility between waypoints (`visible`).
    - Initial locations of soil and rock samples (`at_soil_sample`, `at_rock_sample`).
    - Camera properties: calibration target, which rover it's on, supported modes.
    - Waypoints from which objectives are visible (`visible_from`).
    - The set of goal conditions (`communicated_soil_data`, `communicated_rock_data`, `communicated_image_data`).

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Parse the current state to determine:
       - Current location of each rover.
       - Status of each store (empty/full).
       - Locations of remaining soil/rock samples.
       - Which samples/images each rover possesses.
       - Calibration status of each camera.
       - Which data has already been communicated.
    2. Initialize `total_cost = 0`.
    3. For each goal condition specified in the problem:
       - If the goal condition is already met in the current state, continue to the next goal (cost is 0 for this goal).
       - If the goal is `(communicated_soil_data ?w)`:
         - Find the minimum cost among all soil-equipped rovers to collect the sample at `?w` (if not already collected) and then communicate it from a waypoint visible from the lander.
         - Cost to collect (if needed): Move to `?w` + (1 if store full, for drop) + 1 (sample).
         - Cost to communicate: Move from current location (or `?w` after sampling) to a waypoint visible from the lander + 1 (communicate).
         - The minimum cost for this goal is the minimum sum of collection and communication costs over all suitable rovers.
       - If the goal is `(communicated_rock_data ?w)`:
         - Similar logic to soil data, using rock-equipped rovers and rock samples.
       - If the goal is `(communicated_image_data ?o ?m)`:
         - Find the minimum cost among all imaging-equipped rovers with a camera supporting mode `?m` to take the image of `?o` (if not already taken) and then communicate it.
         - Cost to take image (if needed):
           - If camera is not calibrated: Move to calibration waypoint + 1 (calibrate) + Move to image waypoint + 1 (take_image).
           - If camera is calibrated: Move to image waypoint + 1 (take_image).
           - Choose the best image waypoint (`visible_from ?o ?p`) and calibration waypoint (`visible_from ?t ?w`) for the chosen camera `?i` and its target `?t`.
         - Cost to communicate: Move from current location (or image waypoint after taking image) to a waypoint visible from the lander + 1 (communicate).
         - The minimum cost for this goal is the minimum sum of image-taking and communication costs over all suitable rovers, cameras, and waypoints.
       - Add the minimum finite cost found for the current goal to `total_cost`. If no finite cost is found (goal appears unreachable), it contributes 0 to the sum.
    4. After processing all goals, if the state is not the goal state and `total_cost` is 0 (meaning all unachieved goals had infinite cost estimates), return a large value (e.g., 1000) to distinguish it from the actual goal state. Otherwise, return `total_cost`.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = set(task.goals) # Goal conditions as a set of strings

        # Extract static facts
        static_facts = set(task.static)

        self.lander_location = None
        self.rover_equipment = collections.defaultdict(set) # rover -> {soil, rock, imaging}
        self.store_owner = {} # store -> rover
        self.rover_waypoint_graph = collections.defaultdict(lambda: collections.defaultdict(set)) # rover -> waypoint -> set of adjacent_waypoints
        self.visible_waypoints = collections.defaultdict(set) # waypoint -> set of visible_waypoints (wp1 -> {wp2, wp3} means wp2 and wp3 are visible FROM wp1)
        self.initial_soil_samples = set() # waypoint
        self.initial_rock_samples = set() # waypoint
        self.camera_calibration_target = {} # camera -> objective
        self.camera_on_board = {} # camera -> rover
        self.camera_modes = collections.defaultdict(set) # camera -> set of modes
        self.objective_visible_from = collections.defaultdict(set) # objective -> set of waypoints (objective -> {wp1, wp2} means objective is visible FROM wp1 and wp2)

        # Collect all waypoints mentioned in can_traverse to build initial graph structure
        all_waypoints_in_traverse = set()

        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue # Skip invalid facts

            if parts[0] == 'at_lander':
                self.lander_location = parts[2]
            elif parts[0] == 'equipped_for_soil_analysis':
                self.rover_equipment[parts[1]].add('soil')
            elif parts[0] == 'equipped_for_rock_analysis':
                self.rover_equipment[parts[1]].add('rock')
            elif parts[0] == 'equipped_for_imaging':
                self.rover_equipment[parts[1]].add('imaging')
            elif parts[0] == 'store_of':
                self.store_owner[parts[1]] = parts[2]
            elif parts[0] == 'can_traverse':
                rover, wp1, wp2 = parts[1], parts[2], parts[3]
                self.rover_waypoint_graph[rover][wp1].add(wp2)
                all_waypoints_in_traverse.add(wp1)
                all_waypoints_in_traverse.add(wp2)
            elif parts[0] == 'visible':
                wp1, wp2 = parts[1], parts[2]
                self.visible_waypoints[wp1].add(wp2)
            elif parts[0] == 'at_soil_sample':
                self.initial_soil_samples.add(parts[1])
            elif parts[0] == 'at_rock_sample':
                self.initial_rock_samples.add(parts[1])
            elif parts[0] == 'calibration_target':
                self.camera_calibration_target[parts[1]] = parts[2]
            elif parts[0] == 'on_board':
                self.camera_on_board[parts[1]] = parts[2]
            elif parts[0] == 'supports':
                self.camera_modes[parts[1]].add(parts[2])
            elif parts[0] == 'visible_from':
                self.objective_visible_from[parts[1]].add(parts[2])

        # Ensure all waypoints mentioned in can_traverse are keys in the graph dicts,
        # even if they have no outgoing edges. This prevents BFS from failing if start_node
        # is a waypoint with no outgoing edges.
        for rover in self.rover_waypoint_graph:
             for wp in all_waypoints_in_traverse:
                 if wp not in self.rover_waypoint_graph[rover]:
                     self.rover_waypoint_graph[rover][wp] = set()


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state # Current world state as a frozenset of strings

        # Check if it's a goal state
        if self.goals.issubset(state):
            return 0

        # Parse current state into usable data structures
        rover_current_location = {} # rover -> waypoint
        store_is_full = {} # store -> bool
        soil_sample_at = set(self.initial_soil_samples) # waypoint (start with initial, remove if collected)
        rock_sample_at = set(self.initial_rock_samples) # waypoint (start with initial, remove if collected)
        rover_have_soil = collections.defaultdict(set) # rover -> set of waypoints
        rover_have_rock = collections.defaultdict(set) # rover -> set of waypoints
        rover_have_image = collections.defaultdict(set) # rover -> set of (objective, mode)
        camera_is_calibrated = {} # camera -> bool
        communicated_soil = set() # waypoint
        communicated_rock = set() # waypoint
        communicated_image = set() # (objective, mode)

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip invalid facts

            if parts[0] == 'at' and parts[1] in self.rover_waypoint_graph: # Check if it's a rover location
                 rover_current_location[parts[1]] = parts[2]
            elif parts[0] == 'full':
                 store_is_full[parts[1]] = True
            elif parts[0] == 'empty':
                 store_is_full[parts[1]] = False
            elif parts[0] == 'have_soil_analysis':
                 rover_have_soil[parts[1]].add(parts[2])
                 # If a rover has the sample, it's no longer at the waypoint (in terms of sample availability)
                 if parts[2] in soil_sample_at:
                     soil_sample_at.remove(parts[2])
            elif parts[0] == 'have_rock_analysis':
                 rover_have_rock[parts[1]].add(parts[2])
                 # If a rover has the sample, it's no longer at the waypoint (in terms of sample availability)
                 if parts[2] in rock_sample_at:
                     rock_sample_at.remove(parts[2])
            elif parts[0] == 'have_image':
                 rover_have_image[parts[1]].add((parts[2], parts[3]))
            elif parts[0] == 'calibrated':
                 camera_is_calibrated[parts[1]] = True
            elif parts[0] == 'communicated_soil_data':
                 communicated_soil.add(parts[1])
            elif parts[0] == 'communicated_rock_data':
                 communicated_rock.add(parts[1])
            elif parts[0] == 'communicated_image_data':
                 communicated_image.add((parts[1], parts[2]))

        total_cost = 0
        goals_with_finite_cost = 0 # Count goals for which we found a finite cost estimate

        # Iterate through all goals and estimate cost for unachieved ones
        for goal_fact_str in self.goals:
            goal_parts = get_parts(goal_fact_str)
            if not goal_parts: continue # Skip invalid goal facts

            goal_type = goal_parts[0]

            min_goal_cost = float('inf')

            if goal_type == 'communicated_soil_data':
                waypoint = goal_parts[1]
                if waypoint in communicated_soil:
                    continue # Goal already achieved

                # Find best rover to achieve this goal
                for rover in self.rover_equipment:
                    if 'soil' not in self.rover_equipment[rover]:
                        continue # Rover cannot collect soil

                    current_loc = rover_current_location.get(rover)
                    if current_loc is None: continue # Rover location unknown (shouldn't happen in valid state)

                    cost = 0
                    sample_collected_by_this_rover = waypoint in rover_have_soil.get(rover, set())
                    sample_still_at_waypoint = waypoint in soil_sample_at

                    comm_start_loc = current_loc # Default communication start location is current location

                    if sample_collected_by_this_rover:
                        # Sample is already collected by this rover, just need to communicate
                        pass # Cost to collect is 0
                        # comm_start_loc remains current_loc

                    elif sample_still_at_waypoint:
                        # Sample needs collecting by this rover
                        distances_from_current = get_shortest_path_distances(self.rover_waypoint_graph[rover], current_loc)
                        dist_to_sample = dist(current_loc, waypoint, distances_from_current)
                        if dist_to_sample == float('inf'): continue # Cannot reach sample waypoint

                        cost += dist_to_sample # Move to sample waypoint
                        rover_store = next((s for s, r in self.store_owner.items() if r == rover), None)
                        if rover_store and store_is_full.get(rover_store, False):
                             cost += 1 # Drop action
                        cost += 1 # Sample action
                        comm_start_loc = waypoint # Rover is at sample waypoint after collecting

                    else:
                        # Sample is neither collected by this rover nor at the waypoint.
                        # Assume this rover cannot achieve this goal.
                        continue

                    # Now communicate the sample
                    dist_to_comm_wp = min_dist_to_visible_from_lander(comm_start_loc, self.lander_location, self.rover_waypoint_graph[rover], self.visible_waypoints)
                    if dist_to_comm_wp == float('inf'): continue # Cannot reach a communication point

                    cost += dist_to_comm_wp # Move to communication waypoint
                    cost += 1 # Communicate action

                    min_goal_cost = min(min_goal_cost, cost)

            elif goal_type == 'communicated_rock_data':
                waypoint = goal_parts[1]
                if waypoint in communicated_rock:
                    continue # Goal already achieved

                # Find best rover to achieve this goal
                for rover in self.rover_equipment:
                    if 'rock' not in self.rover_equipment[rover]:
                        continue # Rover cannot collect rock

                    current_loc = rover_current_location.get(rover)
                    if current_loc is None: continue # Rover location unknown

                    cost = 0
                    sample_collected_by_this_rover = waypoint in rover_have_rock.get(rover, set())
                    sample_still_at_waypoint = waypoint in rock_sample_at

                    comm_start_loc = current_loc # Default communication start location is current location

                    if sample_collected_by_this_rover:
                        # Sample is already collected by this rover, just need to communicate
                        pass # Cost to collect is 0
                        # comm_start_loc remains current_loc

                    elif sample_still_at_waypoint:
                        # Sample needs collecting by this rover
                        distances_from_current = get_shortest_path_distances(self.rover_waypoint_graph[rover], current_loc)
                        dist_to_sample = dist(current_loc, waypoint, distances_from_current)
                        if dist_to_sample == float('inf'): continue # Cannot reach sample waypoint

                        cost += dist_to_sample # Move to sample waypoint
                        rover_store = next((s for s, r in self.store_owner.items() if r == rover), None)
                        if rover_store and store_is_full.get(rover_store, False):
                             cost += 1 # Drop action
                        cost += 1 # Sample action
                        comm_start_loc = waypoint # Rover is at sample waypoint after collecting

                    else:
                        # Sample is neither collected by this rover nor at the waypoint.
                        # Assume this rover cannot achieve this goal.
                        continue

                    # Now communicate the sample
                    dist_to_comm_wp = min_dist_to_visible_from_lander(comm_start_loc, self.lander_location, self.rover_waypoint_graph[rover], self.visible_waypoints)
                    if dist_to_comm_wp == float('inf'): continue # Cannot reach a communication point

                    cost += dist_to_comm_wp # Move to communication waypoint
                    cost += 1 # Communicate action

                    min_goal_cost = min(min_goal_cost, cost)

            elif goal_type == 'communicated_image_data':
                objective, mode = goal_parts[1], goal_parts[2]
                if (objective, mode) in communicated_image:
                    continue # Goal already achieved

                # Find best rover/camera combination to achieve this goal
                for rover in self.rover_equipment:
                    if 'imaging' not in self.rover_equipment[rover]:
                        continue # Rover cannot take images

                    current_loc = rover_current_location.get(rover)
                    if current_loc is None: continue # Rover location unknown

                    for camera in self.camera_on_board:
                        if self.camera_on_board[camera] != rover:
                            continue # Camera not on this rover
                        if mode not in self.camera_modes.get(camera, set()):
                            continue # Camera does not support this mode

                        cost = 0
                        image_taken_by_this_rover = (objective, mode) in rover_have_image.get(rover, set())

                        comm_start_loc = current_loc # Default communication start location is current location

                        if image_taken_by_this_rover:
                            # Image is already taken by this rover, just need to communicate
                            pass # Cost to take image is 0
                            # comm_start_loc remains current_loc
                        else:
                            # Image needs taking by this rover/camera
                            image_wps = self.objective_visible_from.get(objective, set())
                            if not image_wps: continue # No waypoint to take image from

                            distances_from_current = get_shortest_path_distances(self.rover_waypoint_graph[rover], current_loc)
                            min_dist_to_image_wp = float('inf')
                            best_image_wp = None
                            for img_wp in image_wps:
                                d = dist(current_loc, img_wp, distances_from_current)
                                if d != float('inf') and d < min_dist_to_image_wp:
                                    min_dist_to_image_wp = d
                                    best_image_wp = img_wp

                            if best_image_wp is None: continue # Cannot reach any image waypoint

                            cost_to_reach_image_wp = min_dist_to_image_wp

                            # Calibration cost if needed
                            calibration_cost = 0
                            needs_calibration = not camera_is_calibrated.get(camera, False)
                            if needs_calibration:
                                cal_target = self.camera_calibration_target.get(camera)
                                if cal_target is None: continue # Camera has no calibration target
                                cal_wps = self.objective_visible_from.get(cal_target, set())
                                if not cal_wps: continue # No waypoint to calibrate from

                                # Find best calibration waypoint reachable from current location
                                min_dist_to_cal_wp = float('inf')
                                best_cal_wp = None
                                for cal_wp in cal_wps:
                                     d = dist(current_loc, cal_wp, distances_from_current) # Use distances from current location
                                     if d != float('inf') and d < min_dist_to_cal_wp:
                                         min_dist_to_cal_wp = d
                                         best_cal_wp = cal_wp
                                if best_cal_wp is None: continue # Cannot reach any calibration waypoint

                                # Sequence: Move to cal_wp, Calibrate, Move to img_wp, Take_image
                                calibration_cost += min_dist_to_cal_wp # Move to calibration waypoint
                                calibration_cost += 1 # Calibrate action
                                distances_from_cal = get_shortest_path_distances(self.rover_waypoint_graph[rover], best_cal_wp)
                                dist_from_cal_to_image = dist(best_cal_wp, best_image_wp, distances_from_cal)
                                if dist_from_cal_to_image == float('inf'): continue # Cannot reach image wp from cal wp
                                calibration_cost += dist_from_cal_to_image # Move from cal waypoint to image waypoint

                                cost += calibration_cost # Add calibration sequence cost
                                comm_start_loc = best_image_wp # Rover ends up at image waypoint after sequence

                            else: # Camera is already calibrated
                                # Sequence: Move to img_wp, Take_image
                                cost += cost_to_reach_image_wp # Move to image waypoint
                                comm_start_loc = best_image_wp # Rover ends up at image waypoint

                            cost += 1 # Take image action

                            # Now communicate
                            dist_to_comm_wp = min_dist_to_visible_from_lander(comm_start_loc, self.lander_location, self.rover_waypoint_graph[rover], self.visible_waypoints)
                            if dist_to_comm_wp == float('inf'): continue # Cannot reach a communication point

                            cost += dist_to_comm_wp # Move to communication waypoint
                            cost += 1 # Communicate action

                        min_goal_cost = min(min_goal_cost, cost)


            # Add the minimum finite cost found for this goal
            if min_goal_cost != float('inf'):
                total_cost += min_goal_cost
                goals_with_finite_cost += 1

        # If not a goal state, but the calculated cost is 0 (meaning all unachieved goals
        # had infinite cost estimates), return a large value.
        # If goals_with_finite_cost is 0, it means no unachieved goal has a finite path estimate.
        # If the state is not the goal state, this should result in a non-zero heuristic.
        # If total_cost is 0, it means either it's the goal state, or all remaining goals are unreachable.
        # We already checked for goal state at the beginning.
        if total_cost == 0 and not self.goals.issubset(state):
             # This implies all remaining goals are unreachable by any equipped rover
             # from their current locations. Return a large value.
             return 1000 # Arbitrary large value

        return total_cost
