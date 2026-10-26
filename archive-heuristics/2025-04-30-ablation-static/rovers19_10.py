from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import collections
import math # For float('inf')

# Helper functions

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact string or malformed facts defensively
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

def bfs(graph, start_node):
    """
    Performs Breadth-First Search on a graph to find shortest distances.

    Args:
        graph: Adjacency list representation (dict: node -> set(neighbors)).
        start_node: The starting node for the BFS.

    Returns:
        A dictionary mapping reachable nodes to their shortest distance from start_node.
        Nodes not reachable are not included.
    """
    distances = {start_node: 0}
    queue = collections.deque([start_node])

    while queue:
        current_node = queue.popleft()
        current_dist = distances[current_node]

        # Check if current_node exists in the graph keys before iterating neighbors
        if current_node in graph:
            for neighbor in graph[current_node]:
                if neighbor not in distances:
                    distances[neighbor] = current_dist + 1
                    queue.append(neighbor)
    return distances

class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve each uncommunicated goal fact (soil data, rock data, image data). The total heuristic value is the sum of these individual goal estimates. For each uncommunicated goal, it finds the minimum cost among all suitable rovers (and cameras for image goals) to collect the data/image and then communicate it.

    # Assumptions
    - The heuristic assumes that each uncommunicated goal needs to be achieved independently by a single rover (or rover/camera pair).
    - It assumes that if a sample location or image/calibration waypoint exists in the initial state, it remains available until sampled/imaged (except for the sample itself which is consumed).
    - It simplifies calibration: if a camera is not calibrated when an image is needed, it estimates the cost to move to the closest calibration point, calibrate, and then move to the image point. It doesn't track calibration state across multiple image goals for the same camera within a single state evaluation.
    - It simplifies store usage: it only adds a 'drop' action cost if the store is currently full when a sample is needed. It doesn't plan for multiple samples requiring multiple drop/communicate cycles.
    - It assumes solvable instances, meaning necessary equipment, waypoints, and communication links exist to achieve the goals. If a goal appears strictly impossible (e.g., no rover has the required equipment, or a required waypoint is unreachable), the heuristic returns infinity.

    # Heuristic Initialization
    The heuristic precomputes the following information from the task's static facts and initial state:
    - Rover capabilities (equipped for soil, rock, imaging).
    - Store ownership for each rover.
    - Camera ownership (on_board) and supported modes.
    - Calibration targets for cameras.
    - Waypoints visible from objectives (for imaging and calibration).
    - Lander locations.
    - Waypoint visibility graph (for communication links).
    - Rover traversal graphs (from can_traverse).
    - Shortest path distances between all reachable pairs of waypoints for each rover using BFS.
    - Set of communication points (waypoints visible from any lander location).
    - Shortest distance from any waypoint to the closest communication point for each rover.
    - Set of suitable calibration waypoints for each camera.
    - Shortest distance from any waypoint to the closest calibration waypoint for each rover/camera pair.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Initialize total heuristic cost to 0.
    2. Parse the current state to quickly look up facts like rover locations, store states, collected data/images, and camera calibration status.
    3. Iterate through each goal fact defined in the task.
    4. If a goal fact is already true in the current state, add 0 to the total cost for this goal.
    5. If a goal fact is not true:
        a.  **For `(communicated_soil_data ?w)`:**
            - Find all rovers equipped for soil analysis.
            - For each suitable rover `r`:
                - Get its current location.
                - Estimate cost to get `(have_soil_analysis ?r ?w)`:
                    - If `(have_soil_analysis ?r ?w)` is already true: Cost is 0. Current location for communication planning is the rover's current location.
                    - If not true: Check if `(at_soil_sample ?w)` is in the state. If not, this path is impossible for this rover; skip to the next rover. If yes:
                        - Cost includes moving to `w` (`dist[r][rover_loc][w]`).
                        - Cost includes the `sample_soil` action (+1).
                        - Check if the rover's store is full. If yes, add a `drop` action cost (+1).
                        - Current location for communication planning is `w`.
                - Estimate cost to communicate `(have_soil_analysis ?r ?w)`:
                    - Cost includes moving from the current location (after sampling or checking `have_data`) to the closest communication point (`dist_to_closest_comm[r][current_loc]`).
                    - Cost includes the `communicate_soil_data` action (+1).
                - The total cost for this rover for this goal is the sum of the above steps.
            - Find the minimum total cost among all suitable rovers. If no suitable rover can achieve the goal (e.g., sample gone, unreachable waypoint), the minimum cost is infinity.
            - Add the minimum cost for this goal to the total heuristic cost. If the minimum cost is infinity, the entire task is likely impossible from this state, return infinity.
        b.  **For `(communicated_rock_data ?w)`:** Follow the same logic as soil data, using rock-specific predicates and equipment.
        c.  **For `(communicated_image_data ?o ?m)`:**
            - Find all rover/camera pairs `(r, i)` where `r` is imaging-equipped, `i` is on `r`, and `i` supports `m`.
            - For each suitable pair `(r, i)`:
                - Get the rover's current location.
                cost_to_get_image = 0
                current_loc_after_image = rover_loc # Default location for communication

                # 1. Cost to get (have_image r o m)
                have_img = (r, o, m) in have_image
                if not have_img:
                    # Need to take image
                    suitable_img_wps = self.objective_visible_from.get(o, set())
                    if not suitable_img_wps:
                        continue # No waypoint to view objective o

                    is_calibrated = (cam, r) in calibrated_cameras
                    suitable_cal_wps = self.camera_cal_waypoints.get(cam, set())

                    if not is_calibrated and not suitable_cal_wps:
                         # Needs calibration but no calibration waypoints for this camera
                         continue # Try next pair

                    # Find the best path to take the image (includes calibration if needed)
                    min_image_path_cost = float('inf')
                    image_taken_at_wp = None # Need to know where the image was taken for communication

                    # Option 1: Use existing calibration (if calibrated)
                    if is_calibrated:
                        # Find closest image waypoint from current location
                        min_dist_to_img_wp = float('inf')
                        closest_img_wp_from_rover = None
                        if r in self.rover_pairwise_distances and rover_loc in self.rover_pairwise_distances[r]:
                            distances_from_rover = self.rover_pairwise_distances[r][rover_loc]
                            for p in suitable_img_wps:
                                if p in distances_from_rover:
                                    if distances_from_rover[p] < min_dist_to_img_wp:
                                        min_dist_to_img_wp = distances_from_rover[p]
                                        closest_img_wp_from_rover = p

                        if closest_img_wp_from_rover is not None and min_dist_to_img_wp != float('inf'):
                            path_cost = min_dist_to_img_wp + 1 # Move + Take Image
                            min_image_path_cost = path_cost
                            image_taken_at_wp = closest_img_wp_from_rover

                    # Option 2: Calibrate first (if calibration is possible)
                    if not is_calibrated and suitable_cal_wps:
                         # Find closest calibration waypoint from current location
                         min_dist_to_cal_wp = float('inf')
                         closest_cal_wp_from_rover = None
                         if r in self.rover_pairwise_distances and rover_loc in self.rover_pairwise_distances[r]:
                              distances_from_rover = self.rover_pairwise_distances[r][rover_loc]
                              for w in suitable_cal_wps:
                                  if w in distances_from_rover:
                                      if distances_from_rover[w] < min_dist_to_cal_wp:
                                          min_dist_to_cal_wp = distances_from_rover[w]
                                          closest_cal_wp_from_rover = w

                         if closest_cal_wp_from_rover is not None and min_dist_to_cal_wp != float('inf'):
                             # Find closest image waypoint from the chosen calibration waypoint
                             min_dist_cal_to_img = float('inf')
                             closest_img_wp_from_cal = None
                             if r in self.rover_pairwise_distances and closest_cal_wp_from_rover in self.rover_pairwise_distances[r]:
                                  distances_from_cal = self.rover_pairwise_distances[r][closest_cal_wp_from_rover]
                                  for p in suitable_img_wps:
                                      if p in distances_from_cal:
                                          if distances_from_cal[p] < min_dist_cal_to_img:
                                              min_dist_cal_to_img = distances_from_cal[p]
                                              closest_img_wp_from_cal = p

                             if closest_img_wp_from_cal is not None and min_dist_cal_to_img != float('inf'):
                                 path_cost = min_dist_to_cal_wp + 1 + min_dist_cal_to_img + 1 # Move + Calibrate + Move + Take Image
                                 if path_cost < min_image_path_cost:
                                      min_image_path_cost = path_cost
                                      image_taken_at_wp = closest_img_wp_from_cal


                    if image_taken_at_wp is None:
                         continue # Cannot reach necessary waypoints for imaging

                    cost_to_get_image = min_image_path_cost
                    current_loc_after_image = image_taken_at_wp # Rover is now at the image location

                # 2. Cost to communicate
                dist_to_comm = self.rover_dist_to_closest_comm_point.get(r, {}).get(current_loc_after_image, float('inf'))
                if dist_to_comm == float('inf'):
                     continue # Cannot reach a communication point

                cost = cost_to_get_image + dist_to_comm + 1 # Get image cost + Move to comm + Communicate

                min_goal_cost = min(min_goal_cost, cost)

                if min_goal_cost == float('inf'):
                    # If no rover/camera can achieve this goal, the task is likely impossible
                    return float('inf')
                total_cost += min_goal_cost

        return total_cost
