import collections
from fnmatch import fnmatch

# Assume Heuristic base class is defined elsewhere and imported as:
# from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure we don't match partial facts if pattern is shorter
    if len(args) > len(parts):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the cost to reach the goal state by summing the estimated costs
    for each unachieved goal fact. The cost for each goal fact is estimated based on the
    minimum number of actions (sample/image, calibrate, communicate, drop) and the
    shortest path movement costs required to achieve it using the most suitable rover(s).
    Shortest path costs between waypoints for each rover are precomputed using BFS.

    # Assumptions:
    - The cost of each action (navigate, sample, drop, calibrate, take_image, communicate) is 1.
    - Movement cost between waypoints is the shortest path distance in the graph defined by
      `can_traverse` and `visible` predicates for the specific rover.
    - For sampling goals, the heuristic assumes a soil/rock sample exists at the target waypoint
      if `(at_soil_sample ?w)` or `(at_rock_sample ?w)` is in the static facts.
    - For imaging goals, the heuristic assumes there is a waypoint visible from the objective
      and a calibration target for the camera visible from some waypoint, based on static facts.
    - The heuristic finds the minimum cost over all suitable rovers and relevant waypoints
      (calibration spot, image spot, lander-visible spot).
    - If a goal is impossible (e.g., no sample exists, no equipped rover, no path), its cost is infinity.
      The total heuristic is infinity if any goal is impossible.

    # Heuristic Initialization
    - Extracts all relevant static facts: lander location, rover capabilities, store ownership,
      camera ownership, camera modes, calibration targets, sample locations, visible_from
      locations, and the waypoint graph for each rover.
    - Precomputes shortest path distances between all pairs of waypoints for each rover
      using BFS on the `can_traverse` and `visible` graph.
    - Identifies waypoints visible from the lander.

    # Step-By-Step Thinking for Computing Heuristic
    1.  Initialize total heuristic cost to 0.
    2.  Identify the current location of each rover in the state.
    3.  Identify the current state of stores (full/empty) and camera calibration.
    4.  Identify which soil/rock samples and images have already been collected by which rovers.
    5.  Iterate through each goal fact specified in the task.
    6.  If a goal fact is already true in the current state, its cost is 0. Continue to the next goal.
    7.  If the goal is `(communicated_soil_data ?w)`:
        -   Initialize minimum cost for this goal (`min_goal_cost`) to infinity.
        -   Check if `(at_soil_sample ?w)` is in the static facts. If not, this goal is impossible, `min_goal_cost` remains infinity.
        -   If the sample exists:
            -   Check if `(have_soil_analysis ?r ?w)` is true for any rover `?r` in the current state.
            -   If yes (sample already collected by some `?r_h`):
                -   The cost is 1 (communicate) + shortest path for `?r_h` from its current location to the nearest lander-visible waypoint. Update `min_goal_cost` with this value.
            -   If no (sample not collected):
                -   Iterate through all rovers `?r` equipped for soil analysis that have a store `?s`.
                -   For each such rover `?r`:
                    -   Calculate the cost to sample: shortest path from `?r`'s current location to `?w` + 1 (sample action).
                    -   If `?s` is full in the current state, add 1 (drop action) to the sample cost.
                    -   Calculate the cost to communicate after sampling: shortest path from `?w` to the nearest lander-visible waypoint for `?r` + 1 (communicate action).
                    -   The total cost for this rover is sample cost + communicate cost. Update `min_goal_cost` with the minimum found across all suitable rovers.
        -   Add `min_goal_cost` to the total heuristic cost. If `min_goal_cost` is infinity, the total becomes infinity.
    8.  If the goal is `(communicated_rock_data ?w)`: Follow similar logic as soil data, checking for rock equipment, rock samples, and `have_rock_analysis`.
    9.  If the goal is `(communicated_image_data ?o ?m)`:
        -   Initialize `min_goal_cost` to infinity.
        -   Check if `(visible_from ?o ?p)` exists for some waypoint `?p` in static facts. If not, impossible.
        -   If visible:
            -   Check if `(have_image ?r ?o ?m)` is true for any rover `?r` in the current state.
            -   If yes (image already taken by some `?r_h`):
                -   The cost is 1 (communicate) + shortest path for `?r_h` from its current location to the nearest lander-visible waypoint. Update `min_goal_cost`.
            -   If no (image not taken):
                -   Iterate through all rovers `?r` equipped for imaging that have a camera `?i` on board supporting mode `?m`.
                -   For each such rover/camera `?r`/`?i`:
                    cal_target = self.camera_calibration_targets.get(camera)
                    # Check if calibration is possible for this camera
                    calibration_possible = False
                    cal_wps = set()
                    if cal_target and cal_target in self.calibration_target_visible_waypoints.get(camera, {}):
                         cal_wps = self.calibration_target_visible_waypoints[camera][cal_target]
                         if cal_wps:
                             calibration_possible = True

                    # If calibration is needed but not possible, this rover/camera combo cannot take the image
                    needs_calibration = camera not in rover_calibrated_cameras.get(rover, set())
                    if needs_calibration and not calibration_possible:
                         continue # Try next camera/rover

                    # Check if image can be taken from anywhere
                    image_wps = self.objective_visible_waypoints.get(objective, set())
                    if not image_wps:
                         continue # Cannot take image of this objective

                    current_pos = self.get_current_rover_location(state, rover)
                    if current_pos:
                        # Cost to image: (calibrate if needed) + move to spots + take image action
                        image_action_cost = 1 # take_image action

                        min_imaging_path_cost_found = float('inf')
                        best_img_wp = None # Keep track of the best image waypoint for communication path

                        if needs_calibration:
                            image_action_cost += 1 # calibrate action
                            # Find min_{cal_wp in cal_wps, img_wp in image_wps} (shortest_path(S, cal_wp) + shortest_path(cal_wp, img_wp))
                            for cal_wp in cal_wps:
                                for img_wp in image_wps:
                                    path1_cost = self.get_shortest_path_distance(rover, current_pos, cal_wp)
                                    if path1_cost != float('inf'):
                                        path2_cost = self.get_shortest_path_distance(rover, cal_wp, img_wp)
                                        if path2_cost != float('inf'):
                                            total_path = path1_cost + path2_cost
                                            if total_path < min_imaging_path_cost_found:
                                                min_imaging_path_cost_found = total_path
                                                best_img_wp = img_wp
                        else: # Already calibrated
                            # Find min_{img_wp in image_wps} (shortest_path(S, img_wp))
                            for img_wp in image_wps:
                                path_cost = self.get_shortest_path_distance(rover, current_pos, img_wp)
                                if path_cost < min_imaging_path_cost_found:
                                    min_imaging_path_cost_found = path_cost
                                    best_img_wp = img_wp

                        if best_img_wp and min_imaging_path_cost_found != float('inf'):
                            imaging_cost = image_action_cost + min_imaging_path_cost_found

                            # Communication cost from the best image waypoint
                            communicate_move_cost = self.get_nearest_lander_visible_waypoint_distance(rover, best_img_wp)
                            if communicate_move_cost != float('inf'):
                                communicate_cost = communicate_move_cost + communicate_action_cost
                                min_goal_cost = min(min_goal_cost, imaging_cost + communicate_cost)


        # Add the minimum cost for this goal to the total
            if min_goal_cost != float('inf'):
                total_cost += min_goal_cost
            else:
                # If any goal is impossible, the total heuristic is infinity.
                return float('inf')

        return total_cost
