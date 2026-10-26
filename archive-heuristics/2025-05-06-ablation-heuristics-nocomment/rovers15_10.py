# Note: The problem description implies a Heuristic base class exists,
# but does not provide its definition. We define the class with the
# expected __init__ and __call__ methods. If a base class is provided
# in the environment, the class definition should be updated to inherit from it.

from fnmatch import fnmatch
from collections import deque
import math

def get_parts(fact):
    """
    Parses a PDDL fact string into a list of its components.
    E.g., '(at rover1 waypoint1)' -> ['at', 'rover1', 'waypoint1']
    """
    # Remove surrounding brackets and split by space
    return fact[1:-1].split()

def match(fact, *args):
    """
    Checks if a fact string matches a pattern of arguments.
    Uses fnmatch for wildcard matching.
    E.g., match('(at rover1 waypoint1)', 'at', '*', 'waypoint1') -> True
    """
    parts = get_parts(fact)
    if not args:
        return True # Matches any fact if no args are provided
    if len(args) > len(parts):
        return False # Cannot match if pattern has more parts than fact
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# If inheriting from Heuristic, uncomment the line below and adjust class definition
# class roversHeuristic(Heuristic):
class roversHeuristic:
    """
    Domain-dependent heuristic for the Rovers domain.

    Summary:
        This heuristic estimates the cost to reach the goal by summing up
        the estimated costs for each unachieved goal fact independently.
        For each unachieved goal fact (communicated data), it calculates
        the minimum cost required to achieve that specific fact, considering
        the necessary steps (sampling/imaging, calibration, navigation,
        communication) and choosing the best available rover/camera
        combination. It uses precomputed shortest path distances between
        waypoints for each rover. The heuristic is non-admissible as it
        ignores negative interactions (e.g., consuming samples, store capacity,
        camera uncalibration after use) and assumes steps for different goals
        can be performed in parallel or without interfering with each other's
        prerequisites beyond the immediate action chain.

    Assumptions:
        - The PDDL task is well-formed according to the domain definition.
        - The waypoint graph for each rover is connected enough to potentially
          reach necessary locations (sample sites, image sites, calibration
          sites, communication points). If a necessary location is unreachable
          for all capable rovers, the heuristic returns infinity (represented as -1).
        - There is at least one lander and its location is static.
        - There is at least one communication waypoint reachable by some rover.
        - For imaging goals, there is at least one waypoint visible from the
          target objective, and if calibration is needed, at least one waypoint
          visible from the calibration target.
        - The heuristic assumes the cost of each action (navigate, sample, drop,
          calibrate, take_image, communicate) is 1.

    Heuristic Initialization:
        - Parses static facts to identify objects (rovers, waypoints, etc.),
          rover capabilities (soil, rock, imaging), store ownership, camera
          details (on-board rover, supported modes, calibration target),
          objective visibility from waypoints, lander location, and the
          waypoint graph for each rover based on `can_traverse`.
        - Computes the set of communication waypoints (visible from the lander).
        - Precomputes all-pairs shortest paths between waypoints for each rover
          using BFS, storing distances. Unreachable waypoints have infinite distance.

    Step-By-Step Thinking for Computing Heuristic:
        1. Initialize total heuristic cost `h = 0`.
        2. Parse the current state to quickly access facts like rover locations,
           sample status, image status, store status, and calibration status.
        3. Iterate through each goal fact specified in the task (`self.goals`).
        4. For each goal fact:
            a. If the goal fact is already true in the current state, add 0 cost for this goal and continue to the next goal.
            b. If the goal fact is `(communicated_soil_data ?w)`:
                i. Initialize minimum cost for this goal `min_goal_cost = infinity`.
                ii. Check if `(have_soil_analysis ?r ?w)` is true for any rover `?r` in the current state.
                iii. If a rover already has the sample:
                    - Iterate through rovers that have the sample `?w`.
                    - Calculate cost: 1 (communicate) + shortest path cost from the rover's current location to the closest communication waypoint.
                    - Update `min_goal_cost` with the minimum cost found across applicable rovers.
                iv. If no rover has the sample:
                    - Iterate through all soil-equipped rovers.
                    - For each soil-equipped rover:
                        - Calculate cost: 1 (sample) + 1 (communicate) + shortest path cost from the rover's current location to waypoint `?w` + (1 if rover's store is full, for drop) + shortest path cost from `?w` (where sampling happens) to the closest communication waypoint.
                        - Update `min_goal_cost` with the minimum cost found across applicable rovers.
                v. If `min_goal_cost` is still infinity, the goal is unreachable; return -1 for the total heuristic.
                vi. Add `min_goal_cost` to the total heuristic `h`.
            c. If the goal fact is `(communicated_rock_data ?w)`:
                i. Follow a similar process as for soil data, using rock-specific predicates and capabilities.
            d. If the goal fact is `(communicated_image_data ?o ?m)`:
                i. Initialize minimum cost for this goal `min_goal_cost = infinity`.
                ii. Check if `(have_image ?r ?o ?m)` is true for any rover `?r` in the current state.
                iii. If a rover already has the image:
                    - Iterate through rovers that have the image `(?o ?m)`.
                    - Calculate cost: 1 (communicate) + shortest path cost from the rover's current location to the closest communication waypoint.
                    - Update `min_goal_cost`.
                iv. If no rover has the image:
                    - Iterate through all imaging-equipped rovers.
                    - For each imaging-equipped rover, iterate through its cameras supporting mode `?m`.
                    - For each suitable camera:
                        # Calculate the minimum cost for this rover/camera to get the image and communicate it
                        cost_to_get_image_and_communicate = math.inf

                        r_i_loc = current_rover_locations.get(rover_name)
                        if r_i_loc is None or rover_name not in self.rover_shortest_paths or r_i_loc not in self.rover_shortest_paths[rover_name]: continue

                        image_wps = self.objective_image_wps.get(o, set())
                        if not image_wps: continue # Cannot image this objective

                        comm_wps = self.communication_wps
                        if not comm_wps: continue # No communication points

                        calibrated = match(f'(calibrated {camera_name} {rover_name})', "calibrated", camera_name, rover_name) in current_camera_calibrations

                        if not calibrated:
                            # Need calibration first. Sequence: current -> calib_wp -> image_wp -> comm_wp
                            calib_target = cam_info.get('calib_target')
                            if not calib_target: continue
                            calib_wps = self.objective_calib_wps.get(calib_target, set())
                            if not calib_wps: continue

                            # Find the best sequence of waypoints
                            min_total_path_cost = math.inf

                            for cal_wp in calib_wps:
                                dist_r_i_to_cal = self.rover_shortest_paths[rover_name][r_i_loc].get(cal_wp, math.inf)
                                if dist_r_i_to_cal == math.inf: continue

                                if cal_wp not in self.rover_shortest_paths[rover_name]: continue # Cannot navigate from calib_wp

                                for img_wp in image_wps:
                                    dist_cal_to_img = self.rover_shortest_paths[rover_name][cal_wp].get(img_wp, math.inf)
                                    if dist_cal_to_img == math.inf: continue

                                    if img_wp not in self.rover_shortest_paths[rover_name]: continue # Cannot navigate from image_wp

                                    comm_wp_dists = [self.rover_shortest_paths[rover_name][img_wp].get(comm_wp, math.inf) for comm_wp in comm_wps]
                                    min_dist_img_to_comm = min(comm_wp_dists) if comm_wp_dists else math.inf
                                    if min_dist_img_to_comm == math.inf: continue

                                    total_path = dist_r_i_to_cal + dist_cal_to_img + min_dist_img_to_comm
                                    min_total_path_cost = min(min_total_path_cost, total_path)

                            if min_total_path_cost == math.inf: continue # Cannot complete the sequence

                            cost_to_get_image_and_communicate = min_total_path_cost + 1 # calibrate
                            cost_to_get_image_and_communicate += 1 # take image
                            cost_to_get_image_and_communicate += 1 # communicate


                        else:
                            # Already calibrated. Sequence: current -> image_wp -> comm_wp
                            min_total_path_cost = math.inf

                            for img_wp in image_wps:
                                dist_r_i_to_img = self.rover_shortest_paths[rover_name][r_i_loc].get(img_wp, math.inf)
                                if dist_r_i_to_img == math.inf: continue

                                if img_wp not in self.rover_shortest_paths[rover_name]: continue # Cannot navigate from image_wp

                                comm_wp_dists = [self.rover_shortest_paths[rover_name][img_wp].get(comm_wp, math.inf) for comm_wp in comm_wps]
                                min_dist_img_to_comm = min(comm_wp_dists) if comm_wp_dists else math.inf
                                if min_dist_img_to_comm == math.inf: continue

                                total_path = dist_r_i_to_img + min_dist_img_to_comm
                                min_total_path_cost = min(min_total_path_cost, total_path)

                            if min_total_path_cost == math.inf: continue # Cannot complete the sequence

                            cost_to_get_image_and_communicate = min_total_path_cost + 1 # take image
                            cost_to_get_image_and_communicate += 1 # communicate

                        min_goal_cost = min(min_goal_cost, cost_to_get_image_and_communicate)

                if min_goal_cost == math.inf:
                    return -1 # Goal is unreachable
                h += min_goal_cost

        # If h is 0 and there are goals, it means all goals were in the state.
        # If h is 0 and there are no goals, it's also a goal state.
        # If h is > 0, it's not a goal state.
        # If h is inf (handled by returning -1), it's unreachable.
        return h

