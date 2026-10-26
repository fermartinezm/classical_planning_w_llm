from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact."""
    # Handle facts like "(at rover1 waypoint1)"
    if not fact or not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
         # Return empty list for invalid facts
         return []
    return fact[1:-1].split()

def match(fact, *args):
    """Check if a PDDL fact matches a given pattern."""
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def get_distances_from(graph, start_wp):
    """
    Performs BFS from start_wp to find distances to all reachable waypoints.
    Graph is {waypoint: [neighbor1, ...]}.
    Returns a dictionary {waypoint: distance}.
    """
    # Ensure start_wp is a valid key in the graph or has outgoing edges
    if start_wp not in graph:
         # If start_wp is not a key, it cannot traverse anywhere.
         return {}

    distances = {start_wp: 0}
    queue = deque([start_wp])
    while queue:
        current_wp = queue.popleft()
        dist = distances[current_wp]
        # Check if current_wp has neighbors in the graph
        if current_wp in graph:
            for neighbor in graph[current_wp]:
                if neighbor not in distances:
                    distances[neighbor] = dist + 1
                    queue.append(neighbor)
    return distances

def min_dist_to_set(distances_from_start, target_wp_set):
    """
    Given distances from a start waypoint, find the minimum distance to any waypoint in target_wp_set.
    Returns (distance, target_waypoint) or (float('inf'), None).
    """
    min_d = float('inf')
    closest_wp = None
    for target_wp in target_wp_set:
        if target_wp in distances_from_start:
            dist = distances_from_start[target_wp]
            if dist < min_d:
                min_d = dist
                closest_wp = target_wp
    return (min_d, closest_wp)


class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve all goal
    conditions. It sums the estimated minimum cost for each unachieved goal
    fact independently. The cost for each goal is estimated by considering
    the necessary sequence of actions (e.g., navigate, sample/image, communicate)
    and finding the minimum travel cost using BFS over the rover's graph.

    # Assumptions
    - The cost of each action (sample, image, calibrate, drop, communicate) is 1.
    - The cost of navigation is the number of steps (waypoints traversed),
      calculated using BFS on the rover's specific traversability graph.
    - The heuristic calculates the minimum cost for each unachieved goal
      fact independently, considering all capable rovers and relevant locations
      (sample locations, image locations, calibration locations, communication locations).
      It does not account for potential synergies or conflicts when a single
      rover pursues multiple goals or when multiple rovers interact.
    - For image goals, the sequence of locations visited is assumed to be
      (CurrentLoc -> CalLoc -> ImgLoc -> CommLoc) if calibration is needed,
      or (CurrentLoc -> ImgLoc -> CommLoc) if already calibrated. The heuristic
      finds the minimum travel cost for this sequence by finding the best
      intermediate CalLoc and ImgLoc reachable by the rover and summing the
      BFS distances between the sequence points.
    - If a required location (sample, image, cal, comm) is unreachable by any
      capable rover, the cost for that goal is considered infinite for that rover.
      If the minimum cost over all rovers for a goal is infinite, that goal is
      unreachable, and the heuristic will reflect this (by adding infinity).

    # Heuristic Initialization
    - Extracts static facts: rover traversability graphs, lander location,
      waypoints visible from objectives (for imaging and calibration targets),
      calibration targets for cameras, waypoints visible from the lander
      (for communication), rover store mapping, rover equipment, cameras on board,
      and camera modes.
    - Identifies the set of goal facts.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Initialize total heuristic cost to 0.
    2. Identify all goal facts that are not true in the current state.
    3. For each unachieved goal fact:
        a. Calculate the minimum estimated cost to achieve this specific goal fact.
           Initialize minimum cost for this goal to infinity.
        b. Consider all rovers potentially capable of contributing to this goal type.
        c. For each capable rover:
            i. Determine the rover's current location.
            ii. Get the rover's traversability graph. If the rover's location is not in its graph, it's isolated and cannot move; skip this rover.
            iii. Compute distances from the rover's current location to all reachable waypoints using BFS.
            iv. Estimate the cost for this rover based on the type of goal:
                - **Soil/Rock Data (`communicated_soil_data ?w` or `communicated_rock_data ?w`):**
                    - If the rover already has the sample `?w`:
                        - Cost is travel from current location to a communication location + 1 (communicate). Find the minimum travel cost to any communication location visible from the lander using BFS distances from the current location.
                    - If the rover does not have the sample `?w`:
                        - Cost is travel from current location to `?w` + (1 if store is full, for soil/rock) + 1 (sample) + travel from `?w` to a communication location + 1 (communicate). Calculate travel costs using BFS distances from the current location to `?w`, and from `?w` to the closest communication location.
                - **Image Data (`communicated_image_data ?o ?m`):**
                    - If the rover already has the image `(?o, ?m)`:
                        - Cost is travel from current location to a communication location + 1 (communicate). Find the minimum travel cost to any communication location visible from the lander using BFS distances from the current location.
                    - If the rover does not have the image `(?o, ?m)`:
                        - Consider rovers equipped for imaging that have a camera supporting the mode `?m`.
                        - For each such rover/camera:
                            - Sequence assumed: CurrentLoc -> (CalLoc -> ImgLoc if needed) -> ImgLoc -> CommLoc.
                            - Calculate travel and action costs along this sequence:
                                - Start at CurrentLoc. Total travel cost = 0. Calibration action cost = 0.
                                - `graph = self.rover_graphs[rover]`
                                - `current_loc_in_sequence = loc_r`
                                - `sequence_possible = True`

                                - Step 1: Handle Calibration if needed
                                - If camera is not calibrated:
                                    cal_target = self.camera_calibration_target.get(camera)
                                    cal_wps = self.objective_visible_from.get(cal_target, set()) if cal_target else set()

                                    if not cal_wps:
                                        # Cannot calibrate, this rover/camera cannot achieve the goal if calibration is required
                                        sequence_possible = False
                                    else:
                                        # Find closest CalLoc from current location (LocR)
                                        distances_from_current = get_distances_from(graph, current_loc_in_sequence)
                                        (dist_to_cal, cal_wp) = min_dist_to_set(distances_from_current, cal_wps)
                                        if dist_to_cal is float('inf'):
                                            sequence_possible = False # Cannot reach any calibration location
                                        else:
                                            total_travel_cost_for_rover += dist_to_cal
                                            calibration_action_cost = 1 # Calibrate action
                                            current_loc_in_sequence = cal_wp # Rover is now at CalLoc

                                # Step 2: Go to ImgLoc
                                if sequence_possible:
                                    # Find best image location
                                    img_wps = self.objective_visible_from.get(goal_obj, set())
                                    if not img_wps:
                                        sequence_possible = False # Cannot take image of this objective
                                    else:
                                        # Find closest ImgLoc from the current location (either original LocR or chosen CalLoc)
                                        distances_from_current = get_distances_from(graph, current_loc_in_sequence)
                                        (dist_to_img, img_wp) = min_dist_to_set(distances_from_current, img_wps)
                                        if dist_to_img is float('inf'):
                                            sequence_possible = False # Cannot reach any image location from the current location
                                        else:
                                            total_travel_cost_for_rover += dist_to_img
                                            current_loc_in_sequence = img_wp # Rover is now at ImgLoc

                                # Step 3: Go to CommLoc
                                if sequence_possible:
                                    # Find closest CommLoc from ImgLoc
                                    distances_from_current = get_distances_from(graph, current_loc_in_sequence)
                                    (dist_to_comm, _) = min_dist_to_set(distances_from_current, self.lander_comm_wps)
                                    if dist_to_comm is float('inf'):
                                        sequence_possible = False # Cannot reach any communication location from ImgLoc
                                    else:
                                        total_travel_cost_for_rover += dist_to_comm

                                # Calculate total cost for this rover/camera if sequence was possible
                                if sequence_possible:
                                    # Total cost = Travel + Calibrate Action + Take Image Action + Communicate Action
                                    cost = total_travel_cost_for_rover + calibration_action_cost + 1 + 1
                                    min_goal_cost = min(min_goal_cost, cost)


            if min_goal_cost is not float('inf'):
                total_cost += min_goal_cost
            else:
                # Goal is unreachable by any capable rover from this state
                total_cost += float('inf')


        return total_cost
