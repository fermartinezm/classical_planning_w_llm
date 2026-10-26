from fnmatch import fnmatch
from collections import defaultdict, deque
import sys

# Assume Heuristic base class is available
# from heuristics.heuristic_base import Heuristic

# Dummy Heuristic base class for standalone testing (remove in final code)
# class Heuristic:
#     def __init__(self, task):
#         self.goals = task.goals
#         self.static = task.static
#     def __call__(self, node):
#         pass

# Dummy Task class for standalone testing (remove in final code)
# class Task:
#     def __init__(self, goals, static):
#         self.goals = goals
#         self.static = static


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
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


def bfs(graph, start_node):
    """Performs BFS to find shortest distances from start_node to all other nodes."""
    distances = {node: float('inf') for node in graph}
    if start_node not in graph:
         return distances # Start node not in graph

    distances[start_node] = 0
    queue = deque([start_node])

    while queue:
        current_node = queue.popleft()

        if current_node in graph:
            for neighbor in graph[current_node]:
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)
    return distances


class roversHeuristic: # Inherit from Heuristic in the actual planning framework
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve all goal
    conditions. It breaks down the problem into individual unachieved goals
    (communicating soil data, rock data, or image data) and estimates the
    minimum actions needed for each, including navigation, sampling/imaging,
    and communication. It sums these individual costs, ignoring potential
    synergies (like one navigation serving multiple goals) but also ignoring
    some resource constraints (like store capacity beyond the need for one empty slot for sampling)
    and complex interactions (like camera recalibration). Navigation costs are
    estimated using precomputed shortest paths on the traversable waypoint graph
    for each rover.

    # Assumptions:
    - The goal conditions only involve `communicated_soil_data`, `communicated_rock_data`,
      and `communicated_image_data`.
    - If a sample (`at_soil_sample` or `at_rock_sample`) is no longer at its initial
      waypoint but the corresponding communication goal is not met and no rover
      currently has the analysis (`have_soil_analysis` or `have_rock_analysis`),
      the goal is considered unreachable from this state (assigned a large penalty).
    - Camera calibration is assumed to be needed before each image capture for
      heuristic calculation purposes, even if the camera is currently calibrated.
      (This simplifies the heuristic but might overestimate).
    - Navigation costs are 1 per step. Other action costs are 1.
    - Rovers can drop samples anywhere if their store is full before sampling.

    # Heuristic Initialization
    - Parses static facts to identify:
        - Lander location.
        - Rover capabilities (equipped for soil, rock, imaging).
        - Store ownership per rover.
        - Camera details (on-board, supported modes, calibration target).
        - Waypoints visible from objectives (`visible_from`).
        - Initial locations of soil and rock samples.
        - All waypoints in the problem.
        - The waypoint graph for each rover based on `can_traverse`.
    - Precomputes all-pairs shortest paths for navigation for each rover using BFS.
    - Identifies communication waypoints (visible from the lander).
    - Parses goal facts to identify the specific soil, rock, and image data that need to be communicated.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Parse the current state facts to determine the location of rovers,
       which rovers have which samples/images, which cameras are calibrated,
       which stores are empty, and which samples/data have been communicated.
    2. Initialize total heuristic cost to 0.
    3. Define a large penalty value for unreachable goals.
    4. For each soil sample waypoint `w` that was initially present and whose
       communication goal `(communicated_soil_data w)` is not met:
       - If any rover `r` currently has `(have_soil_analysis r w)`:
         - Add 1 (communicate) + minimum navigation cost for rover `r` from its
           current location to any communication waypoint.
       - Else (no rover has the analysis):
         - If `(at_soil_sample w)` is still true at waypoint `w`:
           - Find an equipped soil rover `r_soil` with an empty store.
           - If no such rover exists or is reachable: Add penalty.
           - Else: Add 1 (sample) + 1 (communicate) + minimum navigation cost
             for `r_soil` from its location to `w`, plus minimum navigation cost
             from `w` to any communication waypoint. Add 1 if `r_soil`'s store
             is full (for the drop action).
         - Else (`(at_soil_sample w)` is false and no rover has analysis): Add penalty.
    5. Repeat step 4 for rock samples and `(communicated_rock_data w)` goals.
    6. For each image goal `(communicated_image_data o m)` that is not met:
       - If any rover `r` currently has `(have_image r o m)`:
         - Add 1 (communicate) + minimum navigation cost for rover `r` from its
           current location to any communication waypoint.
       - Else (no rover has the image):
         - Find an equipped imaging rover `r_img` with a camera `i` on board
           that supports mode `m`. If multiple, consider the minimum cost over all.
         - If no such rover/camera exists or required waypoints are unreachable: Add penalty.
         - Else: Add 1 (calibrate) + 1 (take_image) + 1 (communicate) +
           minimum navigation cost for `r_img` from its location to any calibration
           waypoint for camera `i`, plus minimum navigation cost from any calibration
           waypoint to any image waypoint for objective `o`, plus minimum navigation
           cost from any image waypoint to any communication waypoint.
    7. Return the total cost. If the cost exceeds the penalty threshold, return the penalty.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        # self.goals = task.goals # Keep task.goals directly for easy access later
        static_facts = task.static

        self.LARGE_PENALTY = 9999 # Use a large number instead of infinity

        # --- Parse Static Facts ---
        self.lander_location = {} # lander -> waypoint
        self.equipped_soil = set() # rovers
        self.equipped_rock = set() # rovers
        self.equipped_imaging = set() # rovers
        self.store_of_rover = {} # rover -> store
        self.cameras_on_board = defaultdict(set) # rover -> set of cameras
        self.camera_supports = defaultdict(set) # camera -> set of modes
        self.camera_calibration_target = {} # camera -> objective
        self.visible_from_objective = defaultdict(set) # objective -> set of waypoints
        self.initial_soil_samples = set() # waypoints
        self.initial_rock_samples = set() # waypoints
        self.all_waypoints = set() # all waypoints
        self.all_rovers = set() # all rovers
        self.all_cameras = set() # all cameras
        self.all_objectives = set() # all objectives
        self.all_modes = set() # all modes
        self.all_landers = set() # all landers
        self.all_stores = set() # all stores

        # For navigation graph
        rover_can_traverse = defaultdict(lambda: defaultdict(set)) # rover -> {wp: set of reachable wps}
        visible_wps = defaultdict(set) # wp -> set of visible wps

        # Collect all objects mentioned in relevant predicates to populate type sets
        location_predicates = {"at", "can_traverse", "visible", "visible_from", "at_lander", "at_soil_sample", "at_rock_sample"}
        type_predicates = {"rover", "waypoint", "store", "camera", "mode", "lander", "objective"}

        for fact in static_facts:
             parts = get_parts(fact)
             predicate = parts[0]

             if predicate in type_predicates:
                  if predicate == "rover": self.all_rovers.add(parts[1])
                  elif predicate == "waypoint": self.all_waypoints.add(parts[1])
                  elif predicate == "store": self.all_stores.add(parts[1])
                  elif predicate == "camera": self.all_cameras.add(parts[1])
                  elif predicate == "mode": self.all_modes.add(parts[1])
                  elif predicate == "lander": self.all_landers.add(parts[1])
                  elif predicate == "objective": self.all_objectives.add(parts[1])

             # Also collect objects from location/action predicates if they weren't explicitly typed
             # This is a fallback and might include non-objects if predicates are complex
             # For this domain, it's safe to assume arguments to these predicates are objects of relevant types
             if predicate in location_predicates:
                  for part in parts[1:]:
                       # Attempt to infer type based on position/predicate if not already known
                       # This is heuristic inference, not guaranteed correct for all PDDL
                       if part.startswith("waypoint"): self.all_waypoints.add(part)
                       elif part.startswith("rover"): self.all_rovers.add(part)
                       elif part.startswith("lander"): self.all_landers.add(part)
                       elif part.startswith("objective"): self.all_objectives.add(part)
                       elif part.startswith("camera"): self.all_cameras.add(part)
                       elif part.startswith("store"): self.all_stores.add(part)
                       elif part in ["colour", "high_res", "low_res"]: self.all_modes.add(part) # Specific modes


        # Reparse static facts to populate structures now that object sets are more complete
        for fact in static_facts:
            parts = get_parts(fact)
            predicate = parts[0]

            if predicate == "at_lander":
                self.lander_location[parts[1]] = parts[2]
            elif predicate == "equipped_for_soil_analysis":
                self.equipped_soil.add(parts[1])
            elif predicate == "equipped_for_rock_analysis":
                self.equipped_rock.add(parts[1])
            elif predicate == "equipped_for_imaging":
                self.equipped_imaging.add(parts[1])
            elif predicate == "store_of":
                self.store_of_rover[parts[2]] = parts[1] # rover -> store
            elif predicate == "on_board":
                self.cameras_on_board[parts[2]].add(parts[1]) # rover -> camera
            elif predicate == "supports":
                self.camera_supports[parts[1]].add(parts[2]) # camera -> mode
            elif predicate == "calibration_target":
                self.camera_calibration_target[parts[1]] = parts[2] # camera -> objective
            elif predicate == "visible_from":
                self.visible_from_objective[parts[1]].add(parts[2]) # objective -> waypoint
            elif predicate == "at_soil_sample":
                self.initial_soil_samples.add(parts[1])
            elif predicate == "at_rock_sample":
                self.initial_rock_samples.add(parts[1])
            elif predicate == "can_traverse":
                rover, wp1, wp2 = parts[1], parts[2], parts[3]
                rover_can_traverse[rover][wp1].add(wp2)
            elif predicate == "visible":
                 wp1, wp2 = parts[1], parts[2]
                 visible_wps[wp1].add(wp2)


        # Build navigation graphs and precompute shortest paths for each rover
        self.rover_shortest_paths = {} # rover -> {start_wp: {end_wp: distance}}
        for rover in self.all_rovers:
            self.rover_shortest_paths[rover] = {}
            graph = rover_can_traverse.get(rover, defaultdict(set))

            # Ensure all known waypoints are nodes in the graph for BFS, even if isolated for this rover
            for wp in self.all_waypoints:
                 if wp not in graph:
                      graph[wp] = set()

            for start_wp in self.all_waypoints:
                self.rover_shortest_paths[rover][start_wp] = bfs(graph, start_wp)

        # Identify communication waypoints (visible from any lander location)
        self.comm_points = set()
        lander_locs = set(self.lander_location.values())
        for lander_loc in lander_locs:
             if lander_loc in visible_wps:
                  self.comm_points.update(visible_wps[lander_loc])
             # A rover can also communicate if it is AT the lander location
             self.comm_points.add(lander_loc)


        # --- Parse Goal Facts ---
        self.goal_soil_samples = set() # waypoints
        self.goal_rock_samples = set() # waypoints
        self.goal_images = set() # (objective, mode) tuples

        # Use task.goals directly
        for goal in task.goals:
            parts = get_parts(goal)
            predicate = parts[0]
            if predicate == "communicated_soil_data":
                self.goal_soil_samples.add(parts[1])
            elif predicate == "communicated_rock_data":
                self.goal_rock_samples.add(parts[1])
            elif predicate == "communicated_image_data":
                self.goal_images.add((parts[1], parts[2]))

    def get_nav_cost(self, rover, start_wp, end_wp):
        """Looks up precomputed distance for a specific rover."""
        if rover not in self.rover_shortest_paths:
             return float('inf') # Rover doesn't exist or has no graph
        return get_nav_cost(self.rover_shortest_paths[rover], start_wp, end_wp)

    def min_nav_cost(self, rover, start_locs, end_locs):
        """Finds minimum navigation cost for a specific rover between any waypoint in start_locs and any in end_locs."""
        if rover not in self.rover_shortest_paths:
             return float('inf') # Rover doesn't exist or has no graph
        return min_nav_cost(self.rover_shortest_paths[rover], start_locs, end_locs)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # --- Parse Current State Facts ---
        at_rover = {} # rover -> waypoint
        have_soil = defaultdict(set) # rover -> set of waypoints
        have_rock = defaultdict(set) # rover -> set of waypoints
        have_image = defaultdict(set) # rover -> set of (objective, mode) tuples
        calibrated_cameras = set() # set of (camera, rover) tuples
        empty_store = set() # set of stores
        at_soil_sample_state = set() # waypoints
        at_rock_sample_state = set() # waypoints
        communicated_soil = set() # waypoints
        communicated_rock = set() # waypoints
        communicated_image = set() # (objective, mode) tuples
        full_store = set() # set of stores

        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]

            if predicate == "at":
                obj, wp = parts[1], parts[2]
                if obj in self.all_rovers: # Check if obj is a rover type
                    at_rover[obj] = wp
            elif predicate == "have_soil_analysis":
                rover, wp = parts[1], parts[2]
                have_soil[rover].add(wp)
            elif predicate == "have_rock_analysis":
                rover, wp = parts[1], parts[2]
                have_rock[rover].add(wp)
            elif predicate == "have_image":
                rover, obj, mode = parts[1], parts[2], parts[3]
                have_image[rover].add((obj, mode))
            elif predicate == "calibrated":
                camera, rover = parts[1], parts[2]
                calibrated_cameras.add((camera, rover))
            elif predicate == "empty":
                empty_store.add(parts[1])
            elif predicate == "full":
                 full_store.add(parts[1])
            elif predicate == "at_soil_sample":
                at_soil_sample_state.add(parts[1])
            elif predicate == "at_rock_sample":
                at_rock_sample_state.add(parts[1])
            elif predicate == "communicated_soil_data":
                communicated_soil.add(parts[1])
            elif predicate == "communicated_rock_data":
                communicated_rock.add(parts[1])
            elif predicate == "communicated_image_data":
                communicated_image.add((parts[1], parts[2]))

        total_cost = 0

        # --- Estimate Cost for Soil Goals ---
        for w in self.goal_soil_samples:
            if w not in communicated_soil:
                # Check if any rover already has the analysis
                rover_has_analysis = None
                for r, waypoints in have_soil.items():
                    if w in waypoints:
                        rover_has_analysis = r
                        break

                if rover_has_analysis:
                    # Need to communicate the existing analysis
                    r = rover_has_analysis
                    current_loc = at_rover.get(r)
                    if current_loc is None: # Rover location unknown? Should not happen in valid states.
                         total_cost += self.LARGE_PENALTY
                         continue

                    nav_cost = self.min_nav_cost(r, current_loc, self.comm_points)
                    if nav_cost == float('inf'):
                        total_cost += self.LARGE_PENALTY # Cannot reach communication point
                    else:
                        total_cost += 1 # communicate_soil_data
                        total_cost += nav_cost
                else:
                    # Need to sample and communicate
                    if w in at_soil_sample_state:
                        min_sample_comm_cost = self.LARGE_PENALTY
                        suitable_rovers = [r for r in self.equipped_soil if r in at_rover] # Only consider rovers whose location is known

                        if not suitable_rovers:
                             total_cost += self.LARGE_PENALTY # No equipped rover
                             continue

                        for r in suitable_rovers:
                            current_loc = at_rover[r]
                            store = self.store_of_rover.get(r)
                            needs_drop = (store is not None and store in full_store)

                            # Cost: (drop if needed) + Nav(current, w) + sample + Nav(w, comm) + communicate
                            current_rover_cost = 0
                            if needs_drop:
                                current_rover_cost += 1 # drop

                            nav_to_sample = self.get_nav_cost(r, current_loc, w)
                            if nav_to_sample == float('inf'): continue # Cannot reach sample point

                            nav_sample_to_comm = self.min_nav_cost(r, w, self.comm_points)
                            if nav_sample_to_comm == float('inf'): continue # Cannot reach comm point

                            current_rover_cost += nav_to_sample + 1 # sample
                            current_rover_cost += nav_sample_to_comm + 1 # communicate

                            min_sample_comm_cost = min(min_sample_comm_cost, current_rover_cost)

                        total_cost += min_sample_comm_cost

                    else:
                        # Sample was taken, but no rover has analysis, and not communicated.
                        # This state implies an issue or unreachable goal.
                        total_cost += self.LARGE_PENALTY

        # --- Estimate Cost for Rock Goals ---
        for w in self.goal_rock_samples:
            if w not in communicated_rock:
                 # Check if any rover already has the analysis
                rover_has_analysis = None
                for r, waypoints in have_rock.items():
                    if w in waypoints:
                        rover_has_analysis = r
                        break

                if rover_has_analysis:
                    # Need to communicate the existing analysis
                    r = rover_has_analysis
                    current_loc = at_rover.get(r)
                    if current_loc is None:
                         total_cost += self.LARGE_PENALTY
                         continue

                    nav_cost = self.min_nav_cost(r, current_loc, self.comm_points)
                    if nav_cost == float('inf'):
                        total_cost += self.LARGE_PENALTY # Cannot reach communication point
                    else:
                        total_cost += 1 # communicate_rock_data
                        total_cost += nav_cost
                else:
                    # Need to sample and communicate
                    if w in at_rock_sample_state:
                        min_sample_comm_cost = self.LARGE_PENALTY
                        suitable_rovers = [r for r in self.equipped_rock if r in at_rover]

                        if not suitable_rovers:
                             total_cost += self.LARGE_PENALTY # No equipped rover
                             continue

                        for r in suitable_rovers:
                            current_loc = at_rover[r]
                            store = self.store_of_rover.get(r)
                            needs_drop = (store is not None and store in full_store)

                            current_rover_cost = 0
                            if needs_drop:
                                current_rover_cost += 1 # drop

                            nav_to_sample = self.get_nav_cost(r, current_loc, w)
                            if nav_to_sample == float('inf'): continue # Cannot reach sample point

                            nav_sample_to_comm = self.min_nav_cost(r, w, self.comm_points)
                            if nav_sample_to_comm == float('inf'): continue # Cannot reach comm point

                            current_rover_cost += nav_to_sample + 1 # sample
                            current_rover_cost += nav_sample_to_comm + 1 # communicate

                            min_sample_comm_cost = min(min_sample_comm_cost, current_rover_cost)

                        total_cost += min_sample_comm_cost
                    else:
                        # Sample was taken, but no rover has analysis, and not communicated.
                        total_cost += self.LARGE_PENALTY


        # --- Estimate Cost for Image Goals ---
        for o, m in self.goal_images:
            if (o, m) not in communicated_image:
                # Check if any rover already has the image
                rover_has_image = None
                for r, images in have_image.items():
                    if (o, m) in images:
                        rover_has_image = r
                        break

                if rover_has_image:
                    # Need to communicate the existing image
                    r = rover_has_image
                    current_loc = at_rover.get(r)
                    if current_loc is None:
                         total_cost += self.LARGE_PENALTY
                         continue

                    nav_cost = self.min_nav_cost(r, current_loc, self.comm_points)
                    if nav_cost == float('inf'):
                        total_cost += self.LARGE_PENALTY # Cannot reach communication point
                    else:
                        total_cost += 1 # communicate_image_data
                        total_cost += nav_cost
                else:
                    # Need to take image and communicate
                    min_image_comm_cost = self.LARGE_PENALTY
                    suitable_rover_camera = [] # List of (rover, camera) tuples

                    for r in self.equipped_imaging:
                        if r not in at_rover: continue # Rover location unknown
                        for i in self.cameras_on_board.get(r, set()):
                            if m in self.camera_supports.get(i, set()):
                                suitable_rover_camera.append((r, i))

                    if not suitable_rover_camera:
                         total_cost += self.LARGE_PENALTY # No suitable rover/camera
                         continue

                    for r, i in suitable_rover_camera:
                        current_loc = at_rover[r]
                        cal_target = self.camera_calibration_target.get(i)
                        if cal_target is None: continue # Camera has no calibration target?

                        cal_wps = self.visible_from_objective.get(cal_target, set())
                        img_wps = self.visible_from_objective.get(o, set())

                        if not cal_wps or not img_wps or not self.comm_points:
                             # Cannot calibrate, image, or communicate from available waypoints
                             continue

                        # Estimate cost for this rover/camera: Nav(current, cal) + cal + Nav(cal, img) + img + Nav(img, comm) + comm
                        # Minimize over all possible cal_wp, img_wp, comm_wp

                        nav_curr_to_cal = self.min_nav_cost(r, current_loc, cal_wps)
                        if nav_curr_to_cal == float('inf'): continue

                        nav_cal_to_img = self.min_nav_cost(r, cal_wps, img_wps)
                        if nav_cal_to_img == float('inf'): continue

                        nav_img_to_comm = self.min_nav_cost(r, img_wps, self.comm_points)
                        if nav_img_to_comm == float('inf'): continue

                        current_rover_camera_cost = nav_curr_to_cal + 1 # calibrate
                        current_rover_camera_cost += nav_cal_to_img + 1 # take_image
                        current_rover_camera_cost += nav_img_to_comm + 1 # communicate

                        min_image_comm_cost = min(min_image_comm_cost, current_rover_camera_cost)

                    total_cost += min_image_comm_cost


        # Cap the penalty
        if total_cost >= self.LARGE_PENALTY:
             return self.LARGE_PENALTY
        return total_cost

