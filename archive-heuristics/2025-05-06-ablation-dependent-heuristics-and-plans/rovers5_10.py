# Necessary imports
from fnmatch import fnmatch
from collections import deque
# Assuming heuristic_base is available in the environment
from heuristics.heuristic_base import Heuristic

# Helper functions
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
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def bfs(start_wp, target_wps, rover_name, rover_nav_graph):
    """Find the shortest path distance from start_wp to any target_wp for a specific rover."""
    if not target_wps:
        return float('inf') # Cannot reach if no targets

    queue = deque([(start_wp, 0)])
    visited = {start_wp}

    while queue:
        curr_wp, dist = queue.popleft()

        if curr_wp in target_wps:
            return dist

        # Get neighbors traversable by this rover
        neighbors = rover_nav_graph.get(rover_name, {}).get(curr_wp, set())

        for next_wp in neighbors:
            if next_wp not in visited:
                visited.add(next_wp)
                queue.append((next_wp, dist + 1))

    return float('inf') # No path found

def find_closest_waypoint(start_wp, target_wps, rover_name, rover_nav_graph):
    """Find the closest waypoint from target_wps reachable from start_wp for a specific rover."""
    if not target_wps:
        return None # Cannot find if no targets

    queue = deque([start_wp])
    visited = {start_wp}

    # If start is already a target, return it
    if start_wp in target_wps:
        return start_wp

    while queue:
        curr_wp = queue.popleft()

        # Get neighbors traversable by this rover
        neighbors = rover_nav_graph.get(rover_name, {}).get(curr_wp, set())

        for next_wp in neighbors:
            if next_wp not in visited:
                visited.add(next_wp)
                # If this neighbor is a target, return it
                if next_wp in target_wps:
                     return next_wp
                # Otherwise, continue BFS
                queue.append(next_wp)

    return None # No target found


class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve all
    unmet goal conditions. It sums the estimated costs for each individual
    unmet goal, assuming each goal can be pursued independently by the
    most suitable available rover. Navigation costs are estimated using
    shortest path (BFS) on the traversable and visible waypoint graph for
    each rover.

    # Assumptions
    - Each unmet goal condition (communicated data) requires a sequence
      of steps: (1) potentially collect sample/image, (2) navigate to
      communication location, (3) communicate.
    - Sampling requires being at the sample location with an equipped
      rover and an empty store. Dropping a full store costs 1 action.
    - Imaging requires being at an image location with an equipped rover
      and a calibrated camera supporting the mode. Calibration requires
      being at a calibration location visible from the target. Calibration
      costs 1 action and is consumed by taking an image.
    - Communication requires being at a location visible from the lander
      with the required data/image.
    - The heuristic sums the minimum estimated cost for each unachieved
      goal fact across all suitable rovers.
    - Navigation cost is the shortest path distance (number of `navigate` actions).
    - If a sample/image is already collected by any rover, the sampling/imaging
      part of the cost for that goal is zero.
    - If a sample was initially present but is no longer on the ground
      and no rover has the data, the heuristic assumes it cannot be
      resampled from the ground (it was consumed).

    # Heuristic Initialization
    - Extracts static facts: lander location, rover equipment, store ownership,
      camera properties (on-board, supported modes, calibration targets),
      objective visibility from waypoints, waypoint visibility, and rover
      traversal capabilities.
    - Builds navigation graphs (adjacency lists) for each rover based on
      `can_traverse` and `visible` facts.
    - Stores the set of initial soil and rock sample locations.
    - Stores the set of goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize total heuristic cost to 0.
    2. Parse the current state to determine:
       - Current location of each rover.
       - Status of each rover's store (empty/full).
       - Which cameras are calibrated on which rovers.
       - Which rovers have which soil/rock samples or images.
       - Which data/images have already been communicated.
       - Which soil/rock samples are still on the ground.
    3. For each goal fact in the task's goals:
       - If the goal fact is already achieved in the current state, add 0 to the total cost.
       - If the goal is `(communicated_soil_data ?w)`:
         - Find all rovers equipped for soil analysis.
         - Calculate the minimum cost for *any* of these rovers to achieve this goal:
           - Cost = (Cost to get sample at ?w) + (Cost to communicate sample from ?w).
           - Cost to get sample:
             - If any rover already has `(have_soil_analysis ?r ?w)`, cost is 0.
             - Otherwise, if `?w` had a soil sample initially:
               - Find the equipped rover closest to `?w`.
               - Cost = (1 if that rover's store is full) + (BFS distance from rover's location to ?w) + 1 (sample_soil).
             - Otherwise (no initial sample or sample lost), this path is impossible for this goal via sampling from ?w.
           - Cost to communicate sample:
             - Find the waypoint `?w_comm` visible from the lander that is closest to the rover's location *after* sampling (or current location if sample was pre-existing).
             - Cost = (BFS distance from rover's location to ?w_comm) + 1 (communicate_soil_data).
           - The minimum cost for this goal is the minimum sum over all suitable rovers. Add this minimum cost to the total.
       - If the goal is `(communicated_rock_data ?w)`: Follow similar logic as soil data, using rock-specific predicates and equipment.
       - If the goal is `(communicated_image_data ?o ?m)`:
         - Find all rovers equipped for imaging with cameras supporting mode `?m`.
         - Calculate the minimum cost for *any* suitable rover/camera combination:
           - Cost = (Cost to get image of ?o in ?m) + (Cost to communicate image).
           - Cost to get image:
             - If any rover already has `(have_image ?r ?o ?m)`, cost is 0.
             - Otherwise:
               - Find a camera `?i` on the rover supporting `?m` with calibration target `?t`.
               - Find a calibration waypoint `?w_cal` visible from `?t` closest to the rover's current location.
               - Find an image waypoint `?w_img` visible from `?o` closest to `?w_cal`.
               - Cost = (BFS distance from rover's location to ?w_cal) + 1 (calibrate) + (BFS distance from ?w_cal to ?w_img) + 1 (take_image).
           - Cost to communicate image:
             - Find the waypoint `?w_comm` visible from the lander closest to the rover's location *after* imaging (or current location if image was pre-existing).
             - Cost = (BFS distance from rover's location to ?w_comm) + 1 (communicate_image_data).
           - The minimum cost for this goal is the minimum sum over all suitable rover/camera combinations. Add this minimum cost to the total.
    4. Return the total accumulated cost.
    """

    def __init__(self, task):
        """Initialize the heuristic."""
        self.goals = task.goals
        self.initial_state = task.initial_state # Needed to check if samples existed initially

        # --- Parse Static Facts ---
        self.lander_location = None
        self.rover_equipment = {} # rover -> set of {'soil', 'rock', 'imaging'}
        self.store_of = {} # store -> rover
        self.rover_cameras = {} # rover -> set of cameras
        self.camera_modes = {} # camera -> set of modes
        self.camera_calibration_target = {} # camera -> objective
        self.objective_visible_from = {} # objective -> set of waypoints
        self.waypoint_visibility = {} # waypoint -> set of visible waypoints (for navigation graph)
        self.rover_can_traverse = {} # rover -> waypoint -> set of traversable neighbors

        # Store initial sample locations
        self.initial_soil_samples = {get_parts(fact)[1] for fact in self.initial_state if match(fact, "at_soil_sample", "*")}
        self.initial_rock_samples = {get_parts(fact)[1] for fact in self.initial_state if match(fact, "at_rock_sample", "*")}


        for fact in task.static:
            parts = get_parts(fact)
            predicate = parts[0]

            if predicate == "at_lander":
                # Assuming only one lander
                self.lander_location = parts[2]
            elif predicate.startswith("equipped_for_"):
                equipment_type = predicate.split("_")[2] # soil, rock, imaging
                rover = parts[1]
                self.rover_equipment.setdefault(rover, set()).add(equipment_type)
            elif predicate == "store_of":
                store, rover = parts[1], parts[2]
                self.store_of[store] = rover
            elif predicate == "on_board":
                camera, rover = parts[1], parts[2]
                self.rover_cameras.setdefault(rover, set()).add(camera)
            elif predicate == "supports":
                camera, mode = parts[1], parts[2]
                self.camera_modes.setdefault(camera, set()).add(mode)
            elif predicate == "calibration_target":
                camera, objective = parts[1], parts[2]
                self.camera_calibration_target[camera] = objective
            elif predicate == "visible_from":
                objective, waypoint = parts[1], parts[2]
                self.objective_visible_from.setdefault(objective, set()).add(waypoint)
            elif predicate == "visible":
                wp1, wp2 = parts[1], parts[2]
                self.waypoint_visibility.setdefault(wp1, set()).add(wp2)
            elif predicate == "can_traverse":
                rover, wp1, wp2 = parts[1], parts[2], parts[3]
                self.rover_can_traverse.setdefault(rover, {}).setdefault(wp1, set()).add(wp2)

        # Build the actual navigation graph for each rover type
        # An edge exists if can_traverse AND visible
        self.rover_nav_graph = {}
        for rover, traverse_map in self.rover_can_traverse.items():
            graph = {}
            for wp1, neighbors in traverse_map.items():
                # Only add edges where visibility also exists
                graph[wp1] = {wp2 for wp2 in neighbors if wp1 in self.waypoint_visibility and wp2 in self.waypoint_visibility[wp1]}
            self.rover_nav_graph[rover] = graph

        # Identify all rovers from static facts
        self.all_rovers = set(self.rover_equipment.keys()) | set(self.rover_cameras.keys()) | set(self.rover_can_traverse.keys())


    def __call__(self, node):
        """Compute the heuristic value for the given state."""
        state = node.state

        # --- Parse Current State ---
        current_rover_location = {} # rover -> waypoint
        current_store_status = {} # store -> 'empty' or 'full'
        current_calibrated_cameras = set() # (camera, rover)
        current_have_soil = set() # (rover, waypoint)
        current_have_rock = set() # (rover, waypoint)
        current_have_image = set() # (rover, objective, mode)
        current_communicated_soil = set() # waypoint
        current_communicated_rock = set() # waypoint
        current_communicated_image = set() # (objective, mode)

        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]

            if predicate == "at":
                obj = parts[1]
                if obj in self.all_rovers: # Check if the object is a known rover
                     current_rover_location[obj] = parts[2]
            elif predicate == "empty":
                current_store_status[parts[1]] = 'empty'
            elif predicate == "full":
                current_store_status[parts[1]] = 'full'
            elif predicate == "calibrated":
                current_calibrated_cameras.add((parts[1], parts[2]))
            elif predicate == "have_soil_analysis":
                current_have_soil.add((parts[1], parts[2]))
            elif predicate == "have_rock_analysis":
                current_have_rock.add((parts[1], parts[2]))
            elif predicate == "have_image":
                current_have_image.add((parts[1], parts[2], parts[3]))
            elif predicate == "communicated_soil_data":
                current_communicated_soil.add(parts[1])
            elif predicate == "communicated_rock_data":
                current_communicated_rock.add(parts[1])
            elif predicate == "communicated_image_data":
                current_communicated_image.add((parts[1], parts[2]))


        total_cost = 0

        # Pre-calculate communication waypoints (visible from lander)
        lander_comm_wps = set()
        if self.lander_location:
             # A waypoint wp is a communication point if the lander location is visible FROM wp
             for wp in self.waypoint_visibility:
                 if self.lander_location in self.waypoint_visibility.get(wp, set()):
                     lander_comm_wps.add(wp)


        # --- Estimate Cost for Each Goal ---
        for goal in self.goals:
            parts = get_parts(goal)
            predicate = parts[0]

            if predicate == "communicated_soil_data":
                w = parts[1]
                if w in current_communicated_soil:
                    continue # Goal already met

                min_goal_cost = float('inf')

                # Find rovers equipped for soil
                equipped_rovers = [r for r, equip in self.rover_equipment.items() if 'soil' in equip]
                if not equipped_rovers:
                    continue # Cannot achieve this goal

                # Check if sample data is already collected by any rover
                sample_already_collected = any((r, w) in current_have_soil for r in equipped_rovers)

                for r in equipped_rovers:
                    if r not in current_rover_location: continue # Rover location unknown? Skip.
                    rover_loc = current_rover_location[r]
                    rover_cost = 0
                    loc_after_sample = rover_loc # Default location if sample is pre-collected

                    if not sample_already_collected:
                        # Need to sample
                        # Check if sample existed initially at this waypoint
                        if w not in self.initial_soil_samples:
                             # Sample never existed or was consumed and not held by any rover
                             # This rover cannot sample it from the ground at w
                             continue # Skip this rover for this goal

                        store = None
                        for s, owner in self.store_of.items():
                            if owner == r:
                                store = s
                                break
                        store_full = (store is not None and current_store_status.get(store) == 'full')

                        rover_cost += (1 if store_full else 0) # Drop cost if needed
                        dist_to_sample = bfs(rover_loc, {w}, r, self.rover_nav_graph)
                        if dist_to_sample == float('inf'): continue # Cannot reach sample location
                        rover_cost += dist_to_sample + 1 # Navigate + Sample
                        loc_after_sample = w # Rover is now at the sample location

                    # Need to communicate
                    if not lander_comm_wps: continue # Cannot communicate if no comm points
                    dist_to_comm = bfs(loc_after_sample, lander_comm_wps, r, self.rover_nav_graph)
                    if dist_to_comm == float('inf'): continue # Cannot reach comm point
                    rover_cost += dist_to_comm + 1 # Navigate + Communicate

                    min_goal_cost = min(min_goal_cost, rover_cost)

                if min_goal_cost != float('inf'):
                    total_cost += min_goal_cost

            elif predicate == "communicated_rock_data":
                w = parts[1]
                if w in current_communicated_rock:
                    continue # Goal already met

                min_goal_cost = float('inf')

                # Find rovers equipped for rock
                equipped_rovers = [r for r, equip in self.rover_equipment.items() if 'rock' in equip]
                if not equipped_rovers:
                    continue # Cannot achieve

                # Check if sample data is already collected by any rover
                sample_already_collected = any((r, w) in current_have_rock for r in equipped_rovers)

                for r in equipped_rovers:
                    if r not in current_rover_location: continue
                    rover_loc = current_rover_location[r]
                    rover_cost = 0
                    loc_after_sample = rover_loc # Default

                    if not sample_already_collected:
                        # Need to sample
                        if w not in self.initial_rock_samples:
                             continue # Cannot sample from ground at w

                        store = None
                        for s, owner in self.store_of.items():
                            if owner == r:
                                store = s
                                break
                        store_full = (store is not None and current_store_status.get(store) == 'full')

                        rover_cost += (1 if store_full else 0) # Drop cost
                        dist_to_sample = bfs(rover_loc, {w}, r, self.rover_nav_graph)
                        if dist_to_sample == float('inf'): continue
                        rover_cost += dist_to_sample + 1 # Navigate + Sample
                        loc_after_sample = w # Rover is now at the sample location

                    # Need to communicate
                    if not lander_comm_wps: continue
                    dist_to_comm = bfs(loc_after_sample, lander_comm_wps, r, self.rover_nav_graph)
                    if dist_to_comm == float('inf'): continue
                    rover_cost += dist_to_comm + 1 # Navigate + Communicate

                    min_goal_cost = min(min_goal_cost, rover_cost)

                if min_goal_cost != float('inf'):
                    total_cost += min_goal_cost

            elif predicate == "communicated_image_data":
                o, m = parts[1], parts[2]
                if (o, m) in current_communicated_image:
                    continue # Goal already met

                min_goal_cost = float('inf')

                # Find rovers equipped for imaging
                imaging_rovers = [r for r, equip in self.rover_equipment.items() if 'imaging' in equip]
                if not imaging_rovers:
                    continue # Cannot achieve

                # Check if image is already collected by any rover
                image_already_collected = any((r, o, m) in current_have_image for r in imaging_rovers)

                for r in imaging_rovers:
                    if r not in current_rover_location: continue
                    rover_loc = current_rover_location[r]

                    # Find cameras on this rover supporting the mode
                    suitable_cameras = [
                        cam for cam in self.rover_cameras.get(r, set())
                        if cam in self.camera_modes and m in self.camera_modes[cam]
                    ]
                    if not suitable_cameras: continue # Rover cannot take this image

                    for i in suitable_cameras:
                        camera_cost = 0
                        loc_after_image = rover_loc # Default

                        if not image_already_collected:
                            # Need to take image
                            cal_target = self.camera_calibration_target.get(i)
                            if not cal_target: continue # Camera has no calibration target
                            cal_wps = self.objective_visible_from.get(cal_target, set())
                            if not cal_wps: continue # No waypoints to calibrate from

                            img_wps = self.objective_visible_from.get(o, set())
                            if not img_wps: continue # No waypoints to image from

                            # Calibration step
                            dist_to_cal = bfs(rover_loc, cal_wps, r, self.rover_nav_graph)
                            if dist_to_cal == float('inf'): continue
                            loc_for_cal = find_closest_waypoint(rover_loc, cal_wps, r, self.rover_nav_graph)
                            if loc_for_cal is None: continue # Should not happen if dist is finite

                            camera_cost += dist_to_cal + 1 # Navigate + Calibrate
                            loc_after_cal = loc_for_cal

                            # Imaging step
                            dist_to_img = bfs(loc_after_cal, img_wps, r, self.rover_nav_graph)
                            if dist_to_img == float('inf'): continue
                            loc_for_img = find_closest_waypoint(loc_after_cal, img_wps, r, self.rover_nav_graph)
                            if loc_for_img is None: continue # Should not happen

                            camera_cost += dist_to_img + 1 # Navigate + Take_image
                            loc_after_image = loc_for_img

                        # Need to communicate
                        if not lander_comm_wps: continue
                        dist_to_comm = bfs(loc_after_image, lander_comm_wps, r, self.rover_nav_graph)
                        if dist_to_comm == float('inf'): continue
                        camera_cost += dist_to_comm + 1 # Navigate + Communicate

                        min_goal_cost = min(min_goal_cost, camera_cost)

                if min_goal_cost != float('inf'):
                    total_cost += min_goal_cost

        return total_cost
