from fnmatch import fnmatch
from collections import deque

# Assume Heuristic base class exists. If not, a dummy class is defined below.
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    # Define a dummy base class if the actual one is not available
    class Heuristic:
        def __init__(self, task):
            pass
        def __call__(self, node):
            raise NotImplementedError


# Helper functions outside the class
def get_parts(fact):
    """Splits a PDDL fact string into its predicate and arguments."""
    return fact[1:-1].split()

def match(fact, *args):
    """Checks if a fact string matches a pattern of predicate and arguments."""
    parts = get_parts(fact)
    # Ensure we don't go out of bounds if fact has fewer parts than args
    if len(parts) < len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def bfs(graph, start_node):
    """Performs BFS to find distances from start_node to all reachable nodes."""
    distances = {node: float('inf') for node in graph}
    if start_node not in graph:
         # Start node not in graph (e.g., isolated waypoint or rover location not in its traverse graph)
         # In a valid problem, the rover's initial location should be in its graph.
         # If an intermediate waypoint is not in the graph, it's unreachable.
         return distances
    distances[start_node] = 0
    queue = deque([start_node])
    while queue:
        current = queue.popleft()
        # Check if current node has neighbors in the graph
        if current in graph:
            for neighbor in graph[current]:
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = distances[current] + 1
                    queue.append(neighbor)
    return distances


class roversHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Rovers domain.

    Summary:
        Estimates the cost to reach the goal state by summing the estimated
        costs for each uncommunicated goal fact. For each uncommunicated goal,
        it calculates the minimum cost required by any capable rover to achieve
        that specific goal, considering the steps needed (navigation, sampling/imaging,
        communication). It uses BFS to estimate navigation costs.

    Assumptions:
        - Goals are treated independently: The cost for each uncommunicated goal
          is calculated as if it were the only goal, and these costs are summed.
          This ignores potential synergies (e.g., a single trip achieving multiple
          objectives) and resource contention beyond the immediate need for an
          empty store for sampling or calibration for imaging.
        - Navigation cost is estimated using shortest path (BFS) on the rover's
          traversal graph.
        - The cost of each action (navigate, sample, drop, calibrate, take_image,
          communicate) is assumed to be 1.
        - The heuristic is non-admissible.
        - If a soil/rock sample is required for a goal and is no longer at its
          initial waypoint and not held by the current rover, that rover cannot
          achieve the goal by sampling from the ground. The heuristic assumes
          reachability if the sample is either at the waypoint or already held
          by *some* rover (handled by checking `have_soil_analysis/rock_analysis`
          for the current rover and `at_soil_sample/rock_sample` at the waypoint).
        - For image goals, if the image is not yet taken, the heuristic assumes
          it must be taken by a rover equipped for imaging with a suitable camera.
          It finds the minimum cost over such rover/camera pairs. If the image
          is already held by *any* rover, the cost is just communication from
          that rover's current location.

    Heuristic Initialization:
        The constructor processes the static facts from the task description to
        build data structures used for efficient lookups during heuristic computation:
        - `all_rovers`, `all_waypoints`, etc.: Sets of all objects of each type.
        - `rover_graph`: Adjacency list representing the waypoint graph for each rover based on `can_traverse`.
        - `rover_equipment`: Maps rovers to their equipment capabilities (soil, rock, imaging).
        - `rover_store`: Maps rovers to their dedicated store object.
        - `lander_location`: The waypoint where the lander is located.
        - `lander_visible_wps`: Set of waypoints visible from the lander location.
        - `camera_info`: Maps cameras to their properties: the rover they are on, the modes they support, and their calibration target objective.
        - `objective_visible_wps`: Maps objectives to the set of waypoints from which they are visible (`visible_from`).
        - `calibration_target_visible_wps`: Maps calibration target objectives to the set of waypoints from which they are visible (reusing `objective_visible_wps`).

    Step-By-Step Thinking for Computing Heuristic:
        1.  Parse the current state (`node.state`) to extract dynamic information:
            - Current location of each rover (`at ?r ?w`).
            - Status of each store (`empty ?s`, `full ?s`).
            - Locations of remaining soil/rock samples on the ground (`at_soil_sample ?w`, `at_rock_sample ?w`).
            - Samples/images collected by each rover (`have_soil_analysis ?r ?w`, `have_rock_analysis ?r ?w`, `have_image ?r ?o ?m`).
            - Calibration status of each camera (`calibrated ?c ?r`).
            - Data already communicated (`communicated_soil_data ?w`, `communicated_rock_data ?w`, `communicated_image_data ?o ?m`).
        2.  Identify the set of uncommunicated goals by comparing `self.goals` with the communicated data in the current state.
        3.  If there are no uncommunicated goals, return 0.
        4.  Initialize `total_cost = 0`.
        5.  For each uncommunicated goal:
            - Initialize `min_goal_cost = float('inf')`.
            - **Soil Goal `(communicated_soil_data ?w)`:**
                wp = goal[1]
                # Identify rovers capable of achieving this goal: those equipped for soil analysis OR those that already have `(have_soil_analysis ?r ?w)`.
                capable_rovers_for_soil = set()
                for r in self.all_rovers:
                     if r in rover_soil_samples and wp in rover_soil_samples[r]:
                          capable_rovers_for_soil.add(r)
                     elif self.rover_equipment.get(r, {}).get('soil', False) and wp in soil_samples_at_wp:
                          capable_rovers_for_soil.add(r)

                for rover in capable_rovers_for_soil:
                     if rover not in rover_location: continue # Should not happen in valid states

                     curr_w = rover_location[rover]
                     rover_cost = 0
                     rover_graph = self.rover_graph.get(rover, {})

                     curr_w_for_comm = curr_w

                     # Cost to get sample (if needed by this rover)
                     if not (rover in rover_soil_samples and wp in rover_soil_samples[rover]):
                         # Rover needs to get the sample. Assume it must sample from the ground.
                         # Check if sample is still at `?w`. If not, this rover cannot sample it from ground.
                         if wp not in soil_samples_at_wp:
                             rover_cost = float('inf') # Cannot sample from ground
                         else:
                             # Navigate to `?w`.
                             dist_map_from_curr = bfs(rover_graph, curr_w)
                             dist_to_sample_wp = dist_map_from_curr.get(wp, float('inf'))
                             if dist_to_sample_wp == float('inf'):
                                 rover_cost = float('inf') # Cannot reach sample waypoint
                             else:
                                 rover_cost += dist_to_sample_wp
                                 # Need empty store.
                                 store = self.rover_store.get(rover)
                                 if store and store_status.get(store) == 'full':
                                     rover_cost += 1 # drop action
                                 # Sample.
                                 rover_cost += 1 # sample_soil
                                 curr_w_for_comm = wp # Rover is now at wp with sample.

                     # If rover_cost is not infinity (meaning it got the sample or already had it):
                     if rover_cost != float('inf'):
                         # Communicate the sample data.
                         # Find target communication waypoint: closest lander-visible WP to `curr_w_for_comm`.
                         dist_map_from_comm_start = bfs(rover_graph, curr_w_for_comm)
                         dist_to_comm_wp = min((dist_map_from_comm_start.get(lw, float('inf')) for lw in self.lander_visible_wps), default=float('inf'))

                         if dist_to_comm_wp == float('inf'):
                             rover_cost = float('inf') # Cannot reach lander-visible waypoint
                         else:
                             rover_cost += dist_to_comm_wp
                             rover_cost += 1 # communicate_soil_data

                     # Update minimum cost for this goal.
                     min_goal_cost = min(min_goal_cost, rover_cost)

            - **Rock Goal `(communicated_rock_data ?w)`:**
                wp = goal[1]
                # Similar logic as soil goal, using rock equipment, rock samples, sample_rock, communicate_rock_data.
                capable_rovers_for_rock = set()
                for r in self.all_rovers:
                     if r in rover_rock_samples and wp in rover_rock_samples[r]:
                          capable_rovers_for_rock.add(r)
                     elif self.rover_equipment.get(r, {}).get('rock', False) and wp in rock_samples_at_wp:
                          capable_rovers_for_rock.add(r)

                min_goal_cost = float('inf') # Reset for this goal

                for rover in capable_rovers_for_rock:
                     if rover not in rover_location: continue

                     curr_w = rover_location[rover]
                     rover_cost = 0
                     rover_graph = self.rover_graph.get(rover, {})
                     curr_w_for_comm = curr_w

                     if not (rover in rover_rock_samples and wp in rover_rock_samples[rover]):
                         if wp not in rock_samples_at_wp:
                             rover_cost = float('inf')
                         else:
                             dist_map_from_curr = bfs(rover_graph, curr_w)
                             dist_to_sample_wp = dist_map_from_curr.get(wp, float('inf'))
                             if dist_to_sample_wp == float('inf'):
                                  rover_cost = float('inf')
                             else:
                                  rover_cost += dist_to_sample_wp
                                  store = self.rover_store.get(rover)
                                  if store and store_status.get(store) == 'full':
                                       rover_cost += 1
                                  rover_cost += 1 # sample_rock
                                  curr_w_for_comm = wp

                     if rover_cost != float('inf'):
                          dist_map_from_comm_start = bfs(rover_graph, curr_w_for_comm)
                          dist_to_comm_wp = min((dist_map_from_comm_start.get(lw, float('inf')) for lw in self.lander_visible_wps), default=float('inf'))

                          if dist_to_comm_wp == float('inf'):
                               rover_cost = float('inf')
                          else:
                               rover_cost += dist_to_comm_wp
                               rover_cost += 1 # communicate_rock_data

                     min_goal_cost = min(min_goal_cost, rover_cost)


            - **Image Goal `(communicated_image_data ?o ?m)`:**
                obj, mode = goal[1], goal[2]

                # Case A: Image is already held by some rover. Find min cost to communicate.
                min_comm_cost_if_held = float('inf')
                # Find min cost to communicate if image is already held by *any* rover
                for rover in self.all_rovers:
                     if rover in rover_images and obj in rover_images[rover] and mode in rover_images[rover][obj]:
                          if rover_images[rover][obj][mode]: # Check if fact is True
                               if rover in rover_location:
                                    curr_w = rover_location[rover]
                                    rover_graph = self.rover_graph.get(rover, {})
                                    dist_map_from_curr = bfs(rover_graph, curr_w)
                                    dist_to_comm_wp = min((dist_map_from_curr.get(lw, float('inf')) for lw in self.lander_visible_wps), default=float('inf'))
                                    if dist_to_comm_wp != float('inf'):
                                         min_comm_cost_if_held = min(min_comm_cost_if_held, dist_to_comm_wp + 1) # navigate + communicate

                min_goal_cost = min_comm_cost_if_held # Initialize min_goal_cost with communication cost if image is already held

                # Case B: Image needs to be taken. Find min cost over capable rover/camera pairs.
                # Only consider this if no rover already holds the image (or if communication from holding rover is impossible)
                # The min cost for the goal is the minimum of (cost to communicate if held) and (min cost to take + communicate if not held).
                # If image is held, min_goal_cost is already initialized with min_comm_cost_if_held.
                # If image is not held, min_comm_cost_if_held is infinity, so min_goal_cost starts at infinity.

                # Calculate min cost to take + communicate if image is NOT held by *any* rover
                # Check if *any* rover has the image
                image_is_held_by_any_rover = False
                for rover in self.all_rovers:
                     if rover in rover_images and obj in rover_images[rover] and mode in rover_images[rover][obj]:
                          if rover_images[rover][obj][mode]:
                               image_is_held_by_any_rover = True
                               break

                if not image_is_held_by_any_rover:
                     min_take_and_comm_cost = float('inf')

                     # Find capable rover/camera pairs for taking the image
                     capable_rover_camera_pairs_to_take = set() # {(rover, camera)}
                     for rover in self.all_rovers:
                          if self.rover_equipment.get(rover, {}).get('imaging', False) and rover in rover_location:
                               for camera in self.all_cameras:
                                     cam_info = self.camera_info.get(camera)
                                     if cam_info and cam_info.get('rover') == rover and cam_info.get('supports', {}).get(mode, False):
                                          capable_rover_camera_pairs_to_take.add((rover, camera))

                     for rover, camera in capable_rover_camera_pairs_to_take:
                          curr_w = rover_location[rover]
                          rover_camera_cost = 0
                          rover_graph = self.rover_graph.get(rover, {})

                          # Cost to take image
                          # Find the closest image waypoint
                          target_img_wps = self.objective_visible_wps.get(obj, set())
                          img_wp = None
                          min_dist_img = float('inf')
                          dist_map_from_curr = bfs(rover_graph, curr_w) # Need fresh BFS from curr_w to find closest img_wp
                          for twp in target_img_wps:
                               d = dist_map_from_curr.get(twp, float('inf'))
                               if d < min_dist_img:
                                    min_dist_img = d
                                    img_wp = twp

                          if img_wp is None or min_dist_img == float('inf'):
                               rover_camera_cost = float('inf') # Cannot reach any image waypoint
                          else:
                               rover_camera_cost += min_dist_img # dist(curr_w, img_wp)

                               # Cost to calibrate (if needed)
                               if not camera_calibrated.get(camera, False):
                                    cam_info = self.camera_info.get(camera)
                                    cal_target = cam_info.get('calibration_target') if cam_info else None
                                    if cal_target:
                                         target_cal_wps = self.calibration_target_visible_wps.get(cal_target, set())
                                         # Find the closest calibration waypoint from img_wp
                                         dist_map_from_img = bfs(rover_graph, img_wp)
                                         cal_wp = None
                                         min_dist_cal = float('inf')
                                         for twp in target_cal_wps:
                                              d = dist_map_from_img.get(twp, float('inf'))
                                              if d < min_dist_cal:
                                                   min_dist_cal = d
                                                   cal_wp = twp

                                         if cal_wp is None or min_dist_cal == float('inf'):
                                              rover_camera_cost = float('inf') # Cannot reach calibration waypoint from img_wp
                                         else:
                                              rover_camera_cost += min_dist_cal # dist(img_wp, cal_wp)
                                              rover_camera_cost += 1 # calibrate action
                                    else:
                                         # Camera has no calibration target? Assume unreachable.
                                         rover_camera_cost = float('inf')


                               if rover_camera_cost != float('inf'):
                                    rover_camera_cost += 1 # take_image action
                                    curr_w_after_image = img_wp # Rover is now at img_wp

                                    # Cost to communicate
                                    # Find target communication waypoint: closest lander-visible WP to curr_w_after_image
                                    dist_map_from_after_image = bfs(rover_graph, curr_w_after_image)
                                    dist_to_comm_wp = min((dist_map_from_after_image.get(lw, float('inf')) for lw in self.lander_visible_wps), default=float('inf'))

                                    if dist_to_comm_wp == float('inf'):
                                         rover_camera_cost = float('inf')
                                    else:
                                         rover_camera_cost += dist_to_comm_wp
                                         rover_camera_cost += 1 # communicate_image_data

                          min_take_and_comm_cost = min(min_take_and_comm_cost, rover_camera_cost)

                     # The min cost for the goal is the minimum of the communication cost (if held) and the min take+communicate cost (if not held)
                     min_goal_cost = min(min_goal_cost, min_take_and_comm_cost)


            # Add the minimum cost for this goal to the total
            if min_goal_cost == float('inf'):
                 # If any uncommunicated goal is unreachable by any capable rover, the whole task is likely unreachable.
                 return float('inf')

            total_cost += min_goal_cost

        return total_cost
