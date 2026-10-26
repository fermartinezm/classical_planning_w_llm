from fnmatch import fnmatch
from collections import defaultdict, deque
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Example: "(at rover1 waypoint1)" -> ["at", "rover1", "waypoint1"]
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

class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to satisfy all
    uncommunicated goal conditions. It breaks down each communication goal
    into necessary steps (sampling/imaging, then communication) and sums
    up the estimated costs for each step, including navigation and
    intermediate actions like dropping samples or calibrating cameras.

    # Assumptions
    - Each uncommunicated goal is treated independently (relaxation).
    - Navigation cost is estimated using precomputed shortest paths on the
      traversable graph for each rover.
    - For goals requiring data/images not yet collected, the heuristic
      assumes the task will be assigned to the first suitable rover found
      (e.g., equipped for the task) that minimizes navigation cost.
    - For communication, it assumes the rover with the data/image will
      navigate to any waypoint visible from the lander that minimizes navigation cost.
    - For imaging, it assumes calibration happens before taking the image,
      and navigation costs are summed sequentially (current -> calibrate_wp -> image_wp).
    - Store capacity is simplified: only one sample can be held at a time,
      and a drop action costs 1 if the store is full when sampling is needed.

    # Heuristic Initialization
    - Parses goal conditions to identify required communications (soil, rock, image).
    - Parses static facts to build:
        - Rover capabilities (soil, rock, imaging equipment).
        - Store ownership.
        - Lander location.
        - Waypoint visibility graph.
        - Rover-specific traversability graphs.
        - Camera information (on-board, supports modes, calibration targets).
        - Objective visibility from waypoints.
        - Waypoints visible from the lander.
    - Precomputes shortest path distances between all pairs of waypoints
      for each rover based on its traversable graph.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic calculates the cost as follows:

    1.  Initialize total heuristic cost to 0.
    2.  Identify all goal conditions that are not yet satisfied in the current state.
    3.  For each unsatisfied goal `G`:
        a.  Initialize goal_cost = 0.
        b.  If `G` is `(communicated_soil_data ?w)`:
            - Add 1 for the `communicate_soil_data` action.
            - Check if `(have_soil_analysis ?r ?w)` exists for any rover `?r` in the state.
            - If NO:
                - Add 1 for the `sample_soil` action.
                - Find the best rover `?r_s` equipped for soil analysis that minimizes navigation cost to `?w`.
                - Find `?r_s`'s current location `?loc_s`.
                - Add `shortest_path_cost(?loc_s, ?w)` for `?r_s` to `goal_cost`.
                - Check if `?r_s`'s store is full. If yes, add 1 for the `drop` action.
                - Estimate communication cost: Find the best communication waypoint `?comm_w` visible from the lander that minimizes navigation cost from `?w` for `?r_s`. Add `shortest_path_cost(?w, ?comm_w)` for `?r_s` to `goal_cost`.
            - If YES (for some `?r_s`):
                - Find `?r_s`'s current location `?loc_s`.
                - Find the best communication waypoint `?comm_w` visible from the lander that minimizes navigation cost from `?loc_s` for `?r_s`. Add `shortest_path_cost(?loc_s, ?comm_w)` for `?r_s` to `goal_cost`.
        c.  If `G` is `(communicated_rock_data ?w)`:
            - Similar logic as soil data, using rock-specific predicates and equipment.
        d.  If `G` is `(communicated_image_data ?o ?m)`:
            - Add 1 for the `communicate_image_data` action.
            - Check if `(have_image ?r ?o ?m)` exists for any rover `?r` in the state.
            - If NO:
                - Add 1 for the `take_image` action.
                - Find the best rover `?r_i` equipped for imaging, with a camera `?c` on board supporting mode `?m`, and the best sequence of waypoints (current -> calibrate_wp -> image_wp) that minimizes total navigation cost.
                - Find `?r_i`'s current location `?loc_i`.
                - Check if `(calibrated ?c ?r_i)` is in the state.
                - If NO:
                    - Add 1 for the `calibrate` action.
                    - Add `shortest_path_cost(?loc_i, ?cal_w)` + `shortest_path_cost(?cal_w, ?image_w)` for the chosen `?r_i`, `?cal_w`, `?image_w` to `goal_cost`.
                    - Estimate communication cost: Find the best communication waypoint `?comm_w` visible from the lander that minimizes navigation cost from `?image_w` for `?r_i`. Add `shortest_path_cost(?image_w, ?comm_w)` for `?r_i` to `goal_cost`.
                - If YES:
                    - Add `shortest_path_cost(?loc_i, ?image_w)` for the chosen `?r_i`, `?image_w` to `goal_cost`.
                    - Estimate communication cost: Find the best communication waypoint `?comm_w` visible from the lander that minimizes navigation cost from `?image_w` for `?r_i`. Add `shortest_path_cost(?image_w, ?comm_w)` for `?r_i` to `goal_cost`.
            - If YES (for some `?r_i`):
                - Find `?r_i`'s current location `?loc_i`.
                - Find the best communication waypoint `?comm_w` visible from the lander that minimizes navigation cost from `?loc_i` for `?r_i`. Add `shortest_path_cost(?loc_i, ?comm_w)` for `?r_i` to `goal_cost`.
        e.  Add `goal_cost` to the total heuristic cost.
    4.  Return the total heuristic cost.

    Note: If any required navigation step is impossible (waypoints unreachable), the shortest path cost will be infinity, and the total heuristic cost will become infinity, indicating an unsolvable state from this point.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # --- Parse Static Facts ---
        self.lander_location = None
        self.rover_equipment = defaultdict(set) # rover -> {soil, rock, imaging}
        self.store_owner = {} # store -> rover
        self.waypoint_graph = defaultdict(lambda: defaultdict(list)) # rover -> {wp1: [wp2, wp3], ...}
        self.waypoint_visibility = defaultdict(set) # wp1 -> {wp2, wp3}
        self.calibration_targets = {} # camera -> objective
        self.camera_on_rover = {} # camera -> rover
        self.camera_modes = defaultdict(set) # camera -> {mode1, mode2}
        self.objective_visible_from = defaultdict(set) # objective -> {wp1, wp2}
        self.waypoint_visible_from_lander = set() # {wp1, wp2}

        # Collect all waypoints and rovers for graph construction
        all_waypoints = set()
        all_rovers = set()

        for fact in static_facts:
            parts = get_parts(fact)
            predicate = parts[0]

            if predicate == "at_lander":
                self.lander_location = parts[2]
            elif predicate.startswith("equipped_for_"):
                equipment_type = predicate.split("_")[2] # soil, rock, imaging
                rover_name = parts[1]
                self.rover_equipment[rover_name].add(equipment_type)
                all_rovers.add(rover_name)
            elif predicate == "store_of":
                store_name, rover_name = parts[1], parts[2]
                self.store_owner[store_name] = rover_name
            elif predicate == "visible":
                wp1, wp2 = parts[1], parts[2]
                self.waypoint_visibility[wp1].add(wp2)
                self.waypoint_visibility[wp2].add(wp1) # Visibility is symmetric
                all_waypoints.add(wp1)
                all_waypoints.add(wp2)
            elif predicate == "can_traverse":
                rover_name, wp1, wp2 = parts[1], parts[2], parts[3]
                # Add edge only if also visible
                # Check both directions of visible predicate as navigate requires visible y z where rover is at y and moves to z
                # and visible is symmetric.
                if wp2 in self.waypoint_visibility.get(wp1, set()):
                     self.waypoint_graph[rover_name][wp1].append(wp2)
                all_rovers.add(rover_name)
                all_waypoints.add(wp1)
                all_waypoints.add(wp2)
            elif predicate == "calibration_target":
                camera, target = parts[1], parts[2]
                self.calibration_targets[camera] = target
            elif predicate == "on_board":
                camera, rover = parts[1], parts[2]
                self.camera_on_rover[camera] = rover
                all_rovers.add(rover)
            elif predicate == "supports":
                camera, mode = parts[1], parts[2]
                self.camera_modes[camera].add(mode)
            elif predicate == "visible_from":
                objective, waypoint = parts[1], parts[2]
                self.objective_visible_from[objective].add(waypoint)
                all_waypoints.add(waypoint)

        # Determine waypoints visible from the lander location
        if self.lander_location:
             self.waypoint_visible_from_lander = self.waypoint_visibility.get(self.lander_location, set())

        # --- Precompute Shortest Paths for each rover ---
        self.rover_distances = {} # rover -> { (start_wp, end_wp): distance }

        for rover in all_rovers:
            self.rover_distances[rover] = {}
            graph = self.waypoint_graph.get(rover, {})

            # Use BFS from each waypoint to find distances to all others
            for start_node in all_waypoints:
                q = deque([(start_node, 0)])
                visited = {start_node}
                self.rover_distances[rover][(start_node, start_node)] = 0

                while q:
                    current_node, dist = q.popleft()

                    # Ensure current_node exists in the graph keys before iterating neighbors
                    if current_node in graph:
                        for neighbor in graph[current_node]:
                            if neighbor not in visited:
                                visited.add(neighbor)
                                self.rover_distances[rover][(start_node, neighbor)] = dist + 1
                                q.append((neighbor, dist + 1))

        # --- Parse Goal Conditions ---
        self.goal_soil = set() # {waypoint}
        self.goal_rock = set() # {waypoint}
        self.goal_image = set() # {(objective, mode)}

        for goal in self.goals:
            parts = get_parts(goal)
            predicate = parts[0]
            if predicate == "communicated_soil_data":
                self.goal_soil.add(parts[1])
            elif predicate == "communicated_rock_data":
                self.goal_rock.add(parts[1])
            elif predicate == "communicated_image_data":
                self.goal_image.add((parts[1], parts[2]))

    def get_distance(self, rover, start_wp, end_wp):
        """Helper to get precomputed distance, returns infinity if unreachable."""
        # Handle cases where start or end waypoint might not be in the precomputed graph
        # (e.g., if a waypoint exists but no rover can traverse to/from it)
        # The BFS populates distances only for reachable nodes.
        # If (start_wp, end_wp) is not in the dictionary, it's unreachable.
        return self.rover_distances.get(rover, {}).get((start_wp, end_wp), float('inf'))


    def find_rover_location(self, state, rover_name):
        """Helper to find the current location of a specific rover."""
        for fact in state:
            if match(fact, "at", rover_name, "*"):
                return get_parts(fact)[2]
        return None # Should not happen in a valid state

    def is_store_full(self, state, rover_name):
        """Helper to check if a rover's store is full."""
        for store, owner in self.store_owner.items():
            if owner == rover_name:
                return f"(full {store})" in state
        return False # Rover has no store? Or store info missing? Assume not full if no store info.

    def find_rover_with_sample(self, state, sample_type, waypoint):
        """Helper to find a rover holding a specific sample."""
        predicate = f"have_{sample_type}_analysis" # e.g., "have_soil_analysis"
        for fact in state:
            if match(fact, predicate, "*", waypoint):
                return get_parts(fact)[1] # Return rover name
        return None

    def find_rover_with_image(self, state, objective, mode):
        """Helper to find a rover holding a specific image."""
        for fact in state:
            if match(fact, "have_image", "*", objective, mode):
                return get_parts(fact)[1] # Return rover name
        return None

    def find_camera_for_image(self, rover_name, mode):
        """Helper to find a camera on a rover that supports a mode."""
        for camera, rover in self.camera_on_rover.items():
            if rover == rover_name and mode in self.camera_modes.get(camera, set()):
                return camera
        return None # Should ideally not happen if goal is reachable

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Check if goal is reached
        if self.goals <= state:
            return 0

        total_heuristic_cost = 0

        # --- Process Soil Goals ---
        for waypoint in self.goal_soil:
            goal_fact = f"(communicated_soil_data {waypoint})"
            if goal_fact not in state:
                goal_cost = 0
                rover_with_sample = self.find_rover_with_sample(state, "soil", waypoint)

                # Need to communicate
                goal_cost += 1 # communicate_soil_data action
                comm_nav_cost = float('inf')

                if rover_with_sample:
                    # Have sample, need to communicate it
                    rover = rover_with_sample
                    current_loc = self.find_rover_location(state, rover)
                    # Find min distance to any communication waypoint
                    min_dist = float('inf')
                    for comm_wp in self.waypoint_visible_from_lander:
                         dist = self.get_distance(rover, current_loc, comm_wp)
                         min_dist = min(min_dist, dist)
                    comm_nav_cost = min_dist
                else:
                    # Don't have sample, need to sample and then communicate
                    goal_cost += 1 # sample_soil action
                    sample_nav_cost = float('inf')
                    drop_cost = 0

                    # Find a suitable rover and its navigation cost to sample location
                    best_rover = None
                    min_sample_dist = float('inf')
                    for rover in self.rover_equipment:
                        if "soil" in self.rover_equipment[rover]:
                            current_loc = self.find_rover_location(state, rover)
                            dist = self.get_distance(rover, current_loc, waypoint)
                            if dist < min_sample_dist:
                                min_sample_dist = dist
                                best_rover = rover

                    if best_rover:
                        sample_nav_cost = min_sample_dist
                        # Check drop cost for this rover
                        if self.is_store_full(state, best_rover):
                            drop_cost = 1

                        # Estimate communication cost after sampling
                        # Assume the same rover communicates from the sample location
                        current_loc_after_sample = waypoint # Rover is at waypoint after sampling
                        min_comm_dist = float('inf')
                        for comm_wp in self.waypoint_visible_from_lander:
                             dist = self.get_distance(best_rover, current_loc_after_sample, comm_wp)
                             min_comm_dist = min(min_comm_dist, dist)
                        comm_nav_cost = min_comm_dist

                    goal_cost += sample_nav_cost + drop_cost + comm_nav_cost

                total_heuristic_cost += goal_cost


        # --- Process Rock Goals ---
        for waypoint in self.goal_rock:
            goal_fact = f"(communicated_rock_data {waypoint})"
            if goal_fact not in state:
                goal_cost = 0
                rover_with_sample = self.find_rover_with_sample(state, "rock", waypoint)

                # Need to communicate
                goal_cost += 1 # communicate_rock_data action
                comm_nav_cost = float('inf')

                if rover_with_sample:
                    # Have sample, need to communicate it
                    rover = rover_with_sample
                    current_loc = self.find_rover_location(state, rover)
                    min_dist = float('inf')
                    for comm_wp in self.waypoint_visible_from_lander:
                         dist = self.get_distance(rover, current_loc, comm_wp)
                         min_dist = min(min_dist, dist)
                    comm_nav_cost = min_dist
                else:
                    # Don't have sample, need to sample and then communicate
                    goal_cost += 1 # sample_rock action
                    sample_nav_cost = float('inf')
                    drop_cost = 0

                    # Find a suitable rover and its navigation cost to sample location
                    best_rover = None
                    min_sample_dist = float('inf')
                    for rover in self.rover_equipment:
                        if "rock" in self.rover_equipment[rover]:
                            current_loc = self.find_rover_location(state, rover)
                            dist = self.get_distance(rover, current_loc, waypoint)
                            if dist < min_sample_dist:
                                min_sample_dist = dist
                                best_rover = rover

                    if best_rover:
                        sample_nav_cost = min_sample_dist
                        # Check drop cost for this rover
                        if self.is_store_full(state, best_rover):
                            drop_cost = 1

                        # Estimate communication cost after sampling
                        current_loc_after_sample = waypoint # Rover is at waypoint after sampling
                        min_comm_dist = float('inf')
                        for comm_wp in self.waypoint_visible_from_lander:
                             dist = self.get_distance(best_rover, current_loc_after_sample, comm_wp)
                             min_comm_dist = min(min_comm_dist, dist)
                        comm_nav_cost = min_comm_dist

                    goal_cost += sample_nav_cost + drop_cost + comm_nav_cost

                total_heuristic_cost += goal_cost

        # --- Process Image Goals ---
        for objective, mode in self.goal_image:
            goal_fact = f"(communicated_image_data {objective} {mode})"
            if goal_fact not in state:
                goal_cost = 0
                rover_with_image = self.find_rover_with_image(state, objective, mode)

                # Need to communicate
                goal_cost += 1 # communicate_image_data action
                comm_nav_cost = float('inf')

                if rover_with_image:
                    # Have image, need to communicate it
                    rover = rover_with_image
                    current_loc = self.find_rover_location(state, rover)
                    min_dist = float('inf')
                    for comm_wp in self.waypoint_visible_from_lander:
                         dist = self.get_distance(rover, current_loc, comm_wp)
                         min_dist = min(min_dist, dist)
                    comm_nav_cost = min_dist
                else:
                    # Don't have image, need to take image and then communicate
                    goal_cost += 1 # take_image action
                    calibrate_cost = 0

                    # Find the best rover, camera, calibration waypoint, and image waypoint
                    best_rover = None
                    best_camera = None
                    chosen_image_wp = None
                    min_total_image_nav_cost = float('inf') # Combined nav cost for cal + image

                    # Find waypoints where the objective is visible from
                    image_wps = self.objective_visible_from.get(objective, set())

                    for rover in self.rover_equipment:
                        if "imaging" in self.rover_equipment[rover]:
                            current_loc = self.find_rover_location(state, rover)
                            camera = self.find_camera_for_image(rover, mode) # Find a camera on this rover for this mode
                            if camera:
                                is_calibrated = f"(calibrated {camera} {rover})" in state
                                cal_target = self.calibration_targets.get(camera)
                                cal_wps = self.objective_visible_from.get(cal_target, set()) if cal_target else set()

                                for image_wp in image_wps:
                                    if not is_calibrated:
                                        # Need to calibrate first
                                        min_cal_to_image_dist = float('inf')
                                        for cal_wp in cal_wps:
                                            dist_curr_to_cal = self.get_distance(rover, current_loc, cal_wp)
                                            dist_cal_to_image = self.get_distance(rover, cal_wp, image_wp)
                                            # Only consider paths where both legs are reachable
                                            if dist_curr_to_cal != float('inf') and dist_cal_to_image != float('inf'):
                                                total_dist = dist_curr_to_cal + dist_cal_to_image
                                                if total_dist < min_cal_to_image_dist:
                                                    min_cal_to_image_dist = total_dist

                                        if min_cal_to_image_dist < min_total_image_nav_cost:
                                            min_total_image_nav_cost = min_cal_to_image_dist
                                            best_rover = rover
                                            best_camera = camera
                                            chosen_image_wp = image_wp

                                    else:
                                        # Already calibrated
                                        dist_curr_to_image = self.get_distance(rover, current_loc, image_wp)
                                        if dist_curr_to_image < min_total_image_nav_cost:
                                            min_total_image_nav_cost = dist_curr_to_image
                                            best_rover = rover
                                            best_camera = camera
                                            chosen_image_wp = image_wp

                    # Add navigation cost for imaging
                    goal_cost += min_total_image_nav_cost

                    if best_rover and best_camera and chosen_image_wp:
                         # Check if calibration was needed for this best path
                         is_calibrated = f"(calibrated {best_camera} {best_rover})" in state
                         if not is_calibrated:
                             calibrate_cost = 1 # Add cost for calibrate action
                         goal_cost += calibrate_cost

                         # Estimate communication cost from the chosen image location
                         current_loc_after_image = chosen_image_wp
                         min_comm_dist = float('inf')
                         for comm_wp in self.waypoint_visible_from_lander:
                              dist = self.get_distance(best_rover, current_loc_after_image, comm_wp)
                              min_comm_dist = min(min_comm_dist, dist)
                         comm_nav_cost = min_comm_dist
                         goal_cost += comm_nav_cost
                    else:
                         # No suitable rover/path found for imaging
                         # The min_total_image_nav_cost would be inf, already added to goal_cost
                         # Ensure comm_nav_cost is also inf
                         comm_nav_cost = float('inf') # Redundant, but explicit

                total_heuristic_cost += goal_cost

        # If any goal component was unreachable, the total cost will be infinity.
        # Return infinity if any part was impossible, otherwise return the sum.
        # A greedy search will handle infinity correctly (won't expand).
        # If total_heuristic_cost is float('inf'), it means at least one goal is unreachable.
        # If it's a sum of finite numbers and infinities, it will be infinity.
        # If all required components were reachable, it will be finite.

        return total_heuristic_cost
