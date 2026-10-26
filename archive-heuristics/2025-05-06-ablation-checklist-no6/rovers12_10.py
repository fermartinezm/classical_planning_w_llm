from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Check if fact is a string and looks like a PDDL fact
    if isinstance(fact, str) and fact.startswith('(') and fact.endswith(')'):
        return fact[1:-1].split()
    return [] # Return empty list for invalid facts

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at rover1 waypoint1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if not parts or len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve all
    unmet goal conditions. It considers the state of each goal (e.g., whether
    a sample is collected, whether an image is taken, whether a camera is
    calibrated) and estimates the minimum actions (sampling/imaging, calibration,
    communication, and navigation) needed to satisfy that specific goal.
    The total heuristic value is the sum of the estimated costs for each
    unmet goal, minimizing the cost over available rovers and methods for
    each individual goal. Navigation costs are estimated using precomputed
    shortest paths between waypoints for each rover.

    # Assumptions:
    - Action costs are uniform (cost = 1 per action).
    - Navigation cost is the shortest path distance in the rover's traverse graph.
    - The heuristic sums costs for individual goals independently, which may
      overestimate the true cost but is faster to compute.
    - If a goal (like communicating a sample) requires a resource (like the sample
      being at a location or held by a rover) that is no longer available in the
      current state, the goal is considered unachievable from this state and
      a large penalty is added. This assumes solvable instances where such
      situations indicate a dead end.
    - There is at least one waypoint visible from the lander for communication.
    - For image goals, there is at least one waypoint visible from the objective
      and from the calibration target (if calibration is needed).

    # Heuristic Initialization
    - Parses all static facts from the task.
    - Identifies rovers, waypoints, lander location, rover capabilities,
      camera information, calibration targets, and visibility relations.
    - Builds rover-specific navigation graphs based on `can_traverse` facts.
    - Precomputes all-pairs shortest paths for each rover using BFS.
    - Stores the set of goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize total heuristic cost `h` to 0.
    2. Extract relevant dynamic facts from the current state (rover locations,
       full stores, calibrated cameras, collected samples/images, sample locations).
    3. Identify a waypoint visible from the lander (communication point). If none
       exists and there are communication goals, add a large penalty and stop.
    4. Iterate through each goal condition specified in the problem:
       a. If the goal is already true in the current state, continue to the next goal.
       b. If the goal is `(communicated_soil_data ?w)`:
          - Find all rovers equipped for soil analysis.
          - For each suitable rover, calculate the minimum cost to achieve this goal:
            - If the rover already has `(have_soil_analysis ?r ?w)`: Cost = 1 (communicate) + navigation cost from rover's current location to the communication point.
            - If the sample `(at_soil_sample ?w)` is at waypoint `?w`: Cost = 1 (sample) + 1 (communicate) + navigation cost from rover's current location to `?w` + navigation cost from `?w` to the communication point. Add 1 if the rover's store is full (for a drop action).
            - If the sample is neither held nor at the waypoint, this goal is unachievable for this sample from this state.
          - Find the minimum cost among all suitable rovers.
          - If a finite minimum cost is found, add it to `h`. Otherwise, add a large penalty (goal unachievable).
       c. If the goal is `(communicated_rock_data ?w)`: Follow similar logic as soil data.
       d. If the goal is `(communicated_image_data ?o ?m)`:
          - Find all rovers equipped for imaging with a camera supporting mode `?m`.
          - For each suitable rover/camera pair, calculate the minimum cost:
            - If the rover already has `(have_image ?r ?o ?m)`: Cost = 1 (communicate) + navigation cost from rover's current location to the communication point.
            - If the rover needs to take the image:
              - Find a waypoint visible from objective `?o`. Find the closest such waypoint to the rover's current location.
              - If the camera is calibrated: Cost = 1 (take_image) + 1 (communicate) + navigation cost from current location to image waypoint + navigation cost from image waypoint to communication point.
              - If the camera is not calibrated: Find the calibration target for the camera. Find a waypoint visible from the calibration target. Find the closest such waypoint to the rover's current location. Cost = 1 (calibrate) + 1 (take_image) + 1 (communicate) + navigation cost from current location to calibration waypoint + navigation cost from calibration waypoint to image waypoint + navigation cost from image waypoint to communication point.
              - If no suitable image waypoint or calibration waypoint is found, this method is not possible.
          - Find the minimum cost among all suitable rover/camera pairs and methods (communicate existing vs. take and communicate).
          - If a finite minimum cost is found, add it to `h`. Otherwise, add a large penalty (goal unachievable).
    5. Return the total heuristic cost `h`.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static_facts = task.static # Store static facts to parse them

        # Data structures for static information
        self.lander_location = None
        self.rovers = set()
        self.rover_capabilities = {} # rover -> set of capabilities ('soil', 'rock', 'imaging')
        self.rover_stores = {} # rover -> store
        self.rover_cameras = {} # rover -> list of cameras
        self.camera_modes = {} # camera -> set of modes
        self.camera_calibration_targets = {} # camera -> objective (target)
        self.objective_visible_waypoints = {} # objective -> set of waypoints visible from
        self.calibration_target_visible_waypoints = {} # objective (target) -> set of waypoints visible from
        self.rover_graphs = {} # rover -> {wp -> set of reachable wps}
        self.general_visible_graph = {} # wp -> set of visible wps
        self.all_waypoints = set() # Collect all waypoints

        # First pass: Identify objects and basic relations
        calibration_targets_objs = set()
        for fact in self.static_facts:
            parts = get_parts(fact)
            if not parts: continue
            pred = parts[0]
            if pred == 'rover': self.rovers.add(parts[1])
            elif pred == 'waypoint': self.all_waypoints.add(parts[1])
            elif pred == 'at_lander': self.lander_location = parts[2]
            elif pred == 'calibration_target': calibration_targets_objs.add(parts[2]) # Store the objective name that is a target

        # Second pass: Build main structures
        for fact in self.static_facts:
            parts = get_parts(fact)
            if not parts: continue
            pred = parts[0]

            if pred == 'equipped_for_soil_analysis':
                self.rover_capabilities.setdefault(parts[1], set()).add('soil')
            elif pred == 'equipped_for_rock_analysis':
                self.rover_capabilities.setdefault(parts[1], set()).add('rock')
            elif pred == 'equipped_for_imaging':
                self.rover_capabilities.setdefault(parts[1], set()).add('imaging')
            elif pred == 'store_of':
                self.rover_stores[parts[2]] = parts[1] # rover -> store
            elif pred == 'on_board':
                self.rover_cameras.setdefault(parts[2], []).append(parts[1]) # rover -> camera
            elif pred == 'supports':
                self.camera_modes.setdefault(parts[1], set()).add(parts[2]) # camera -> mode
            elif pred == 'calibration_target':
                self.camera_calibration_targets[parts[1]] = parts[2] # camera -> target objective
            elif pred == 'visible_from':
                o, w = parts[1:]
                self.objective_visible_waypoints.setdefault(o, set()).add(w)
                if o in calibration_targets_objs:
                     self.calibration_target_visible_waypoints.setdefault(o, set()).add(w)
            elif pred == 'can_traverse':
                r, w1, w2 = parts[1:]
                self.rover_graphs.setdefault(r, {}).setdefault(w1, set()).add(w2)
                # Ensure all waypoints mentioned in can_traverse are in the graph keys,
                # even if they have no outgoing edges yet.
                self.rover_graphs[r].setdefault(w2, set())
            elif pred == 'visible':
                 w1, w2 = parts[1:]
                 self.general_visible_graph.setdefault(w1, set()).add(w2)
                 self.general_visible_graph.setdefault(w2, set()).add(w1) # Assuming visible is symmetric

        # Add any waypoints from static facts that weren't in can_traverse/visible
        for wp in self.all_waypoints:
             for r in self.rovers:
                 self.rover_graphs.setdefault(r, {}).setdefault(wp, set()) # Ensure all wps exist in rover graphs

        # Compute shortest paths for each rover
        self.rover_shortest_paths = {}
        for rover, graph in self.rover_graphs.items():
            self.rover_shortest_paths[rover] = {}
            # BFS from all waypoints in this rover's graph
            waypoints_in_graph = set(graph.keys()).union(*graph.values())
            for start_wp in waypoints_in_graph:
                 distances = self.bfs(graph, start_wp)
                 for end_wp, dist in distances.items():
                     if dist != float('inf'):
                         self.rover_shortest_paths[rover][(start_wp, end_wp)] = dist

        # Store goal conditions
        self.goal_conditions = set(task.goals)

    def bfs(self, graph, start_node):
        """Helper function to perform BFS for shortest path distances."""
        distances = {node: float('inf') for node in graph}
        # Ensure start_node is a known waypoint, even if it has no outgoing edges
        if start_node in self.all_waypoints:
             distances[start_node] = 0
        else:
             # Unknown waypoint, cannot start BFS from here
             return {start_node: float('inf')}

        queue = deque([start_node])

        while queue:
            current_node = queue.popleft()
            # Check if current_node has neighbors in the graph
            if current_node in graph:
                for neighbor in graph[current_node]:
                    if distances.get(neighbor, float('inf')) == float('inf'): # Use .get for safety
                        distances[neighbor] = distances[current_node] + 1
                        queue.append(neighbor)
        return distances

    def get_distance(self, rover, start_wp, end_wp):
        """Lookup precomputed shortest path distance."""
        # Handle cases where rover or waypoints are not in the graph (e.g., isolated waypoints)
        if rover not in self.rover_shortest_paths:
            return float('inf')
        return self.rover_shortest_paths[rover].get((start_wp, end_wp), float('inf'))

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        h = 0

        # Extract relevant state facts efficiently
        current_rover_locations = {}
        rover_stores_full = {}
        rover_calibrated_cameras = {} # rover -> set of calibrated cameras
        rover_have_soil = {} # (rover, wp) -> True
        rover_have_rock = {} # (rover, wp) -> True
        rover_have_image = {} # (rover, obj, mode) -> True
        soil_samples_at_wp = set() # wp -> True
        rock_samples_at_wp = set() # wp -> True

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue
            pred = parts[0]
            if pred == 'at':
                obj, loc = parts[1:]
                if obj in self.rovers:
                    current_rover_locations[obj] = loc
            elif pred == 'full':
                store = parts[1]
                # Find which rover this store belongs to
                for r, s in self.rover_stores.items():
                    if s == store:
                        rover_stores_full[r] = True
                        break
            elif pred == 'calibrated':
                camera, rover = parts[1:]
                rover_calibrated_cameras.setdefault(rover, set()).add(camera)
            elif pred == 'have_soil_analysis':
                rover, wp = parts[1:]
                rover_have_soil[(rover, wp)] = True
            elif pred == 'have_rock_analysis':
                rover, wp = parts[1:]
                rover_have_rock[(rover, wp)] = True
            elif pred == 'have_image':
                rover, obj, mode = parts[1:]
                rover_have_image[(rover, obj, mode)] = True
            elif pred == 'at_soil_sample':
                soil_samples_at_wp.add(parts[1])
            elif pred == 'at_rock_sample':
                rock_samples_at_wp.add(parts[1])

        # Find a lander-visible waypoint (assume at least one exists and is reachable)
        # We need *a* waypoint visible from the lander. Let's find one.
        lander_comm_wp = None
        lander_wp = self.lander_location
        if lander_wp in self.general_visible_graph:
             for wp in self.general_visible_graph[lander_wp]:
                  lander_comm_wp = wp
                  break # Found one

        # If no lander or no visible waypoint from lander, communication is impossible.
        # If there are communication goals, they are unachievable.
        # For solvable instances, this case implies a dead end.
        # Add a large penalty if communication goals exist but cannot be communicated to.
        communication_possible = (lander_comm_wp is not None)

        # Iterate through goal conditions
        for goal_fact in self.goal_conditions:
            if goal_fact in state:
                continue # Goal already achieved

            parts = get_parts(goal_fact)
            if not parts: continue
            pred = parts[0]

            # If communication is impossible and this is a communication goal, penalize
            if not communication_possible and pred.startswith('communicated_'):
                 h += 1000
                 continue

            if pred == 'communicated_soil_data':
                wp = parts[1]
                min_cost_for_goal = float('inf')

                # Find suitable rovers
                suitable_rovers = [r for r in self.rovers if 'soil' in self.rover_capabilities.get(r, set())]

                for r in suitable_rovers:
                    curr_wp = current_rover_locations.get(r)
                    if curr_wp is None: continue # Rover location unknown

                    cost_r = float('inf')
                    has_sample = rover_have_soil.get((r, wp), False)

                    if has_sample:
                        # Option 1: Communicate existing sample
                        nav_cost = self.get_distance(r, curr_wp, lander_comm_wp)
                        if nav_cost != float('inf'):
                            cost_r = min(cost_r, 1 + nav_cost) # communicate + navigate
                    else:
                        # Option 2: Sample and Communicate
                        is_sample_at_wp = wp in soil_samples_at_wp
                        if is_sample_at_wp:
                            nav_cost_to_sample = self.get_distance(r, curr_wp, wp)
                            nav_cost_to_comm = self.get_distance(r, wp, lander_comm_wp)
                            if nav_cost_to_sample != float('inf') and nav_cost_to_comm != float('inf'):
                                sample_comm_cost = 1 + 1 + nav_cost_to_sample + nav_cost_to_comm # sample + communicate + nav1 + nav2
                                # Check if store is full, needs drop before sample
                                store = self.rover_stores.get(r)
                                if store and rover_stores_full.get(r, False):
                                    sample_comm_cost += 1 # drop action
                                cost_r = min(cost_r, sample_comm_cost)

                    min_cost_for_goal = min(min_cost_for_goal, cost_r)

                if min_cost_for_goal != float('inf'):
                    h += min_cost_for_goal
                else:
                    # Goal is unachievable by any suitable rover. Add penalty.
                    h += 1000


            elif pred == 'communicated_rock_data':
                wp = parts[1]
                min_cost_for_goal = float('inf')

                # Find suitable rovers
                suitable_rovers = [r for r in self.rovers if 'rock' in self.rover_capabilities.get(r, set())]

                for r in suitable_rovers:
                    curr_wp = current_rover_locations.get(r)
                    if curr_wp is None: continue

                    cost_r = float('inf')
                    has_sample = rover_have_rock.get((r, wp), False)

                    if has_sample:
                        # Option 1: Communicate existing sample
                        nav_cost = self.get_distance(r, curr_wp, lander_comm_wp)
                        if nav_cost != float('inf'):
                            cost_r = min(cost_r, 1 + nav_cost) # communicate + navigate
                    else:
                        # Option 2: Sample and Communicate
                        is_sample_at_wp = wp in rock_samples_at_wp
                        if is_sample_at_wp:
                            nav_cost_to_sample = self.get_distance(r, curr_wp, wp)
                            nav_cost_to_comm = self.get_distance(r, wp, lander_comm_wp)
                            if nav_cost_to_sample != float('inf') and nav_cost_to_comm != float('inf'):
                                sample_comm_cost = 1 + 1 + nav_cost_to_sample + nav_cost_to_comm # sample + communicate + nav1 + nav2

                                # Check if store is full, needs drop before sample
                                store = self.rover_stores.get(r)
                                if store and rover_stores_full.get(r, False):
                                    sample_comm_cost += 1 # drop action
                                cost_r = min(cost_r, sample_comm_cost)

                    min_cost_for_goal = min(min_cost_for_goal, cost_r)

                if min_cost_for_goal != float('inf'):
                    h += min_cost_for_goal
                else:
                    # Goal is unachievable by any suitable rover. Add penalty.
                    h += 1000


            elif pred == 'communicated_image_data':
                obj, mode = parts[1:]
                min_cost_for_goal = float('inf')

                # Find suitable rovers/cameras
                suitable_rovers_cameras = [] # List of (rover, camera) tuples
                for r in self.rovers:
                    if 'imaging' in self.rover_capabilities.get(r, set()):
                        for cam in self.rover_cameras.get(r, []):
                            if mode in self.camera_modes.get(cam, set()):
                                suitable_rovers_cameras.append((r, cam))

                for r, cam in suitable_rovers_cameras:
                    curr_wp = current_rover_locations.get(r)
                    if curr_wp is None: continue

                    cost_rc = float('inf')
                    has_image = rover_have_image.get((r, obj, mode), False)

                    if has_image:
                        # Option 1: Communicate existing image
                        nav_cost = self.get_distance(r, curr_wp, lander_comm_wp)
                        if nav_cost != float('inf'):
                            cost_rc = min(cost_rc, 1 + nav_cost) # communicate + navigate
                    else:
                        # Option 2: Take image and Communicate
                        image_wps = self.objective_visible_waypoints.get(obj, set())
                        if image_wps:
                            # Find best image waypoint
                            min_nav_cost_to_image = float('inf')
                            best_image_wp = None
                            for img_wp in image_wps:
                                nav_cost = self.get_distance(r, curr_wp, img_wp)
                                if nav_cost != float('inf') and nav_cost < min_nav_cost_to_image:
                                    min_nav_cost_to_image = nav_cost
                                    best_image_wp = img_wp

                            if best_image_wp:
                                nav_cost_from_image_to_comm = self.get_distance(r, best_image_wp, lander_comm_wp)
                                if nav_cost_from_image_to_comm != float('inf'):
                                    take_comm_cost = 1 + 1 + min_nav_cost_to_image + nav_cost_from_image_to_comm # take_image + communicate + nav1 + nav2

                                    # Check if calibrated
                                    is_calibrated = cam in rover_calibrated_cameras.get(r, set())

                                    if not is_calibrated:
                                        # Need to calibrate
                                        cal_target = self.camera_calibration_targets.get(cam)
                                        if cal_target:
                                            cal_wps = self.calibration_target_visible_waypoints.get(cal_target, set())
                                            if cal_wps:
                                                # Find best calibration waypoint
                                                min_nav_cost_to_cal = float('inf')
                                                best_cal_wp = None
                                                for c_wp in cal_wps:
                                                     nav_cost = self.get_distance(r, curr_wp, c_wp)
                                                     if nav_cost != float('inf') and nav_cost < min_nav_cost_to_cal:
                                                         min_nav_cost_to_cal = nav_cost
                                                         best_cal_wp = c_wp

                                                if best_cal_wp:
                                                    nav_cost_cal_to_image = self.get_distance(r, best_cal_wp, best_image_wp)
                                                    if nav_cost_cal_to_image != float('inf'):
                                                         # Update cost: calibrate + take + comm + nav(curr->cal) + nav(cal->image) + nav(image->comm)
                                                         take_comm_cost = 1 + 1 + 1 + min_nav_cost_to_cal + nav_cost_cal_to_image + nav_cost_from_image_to_comm

                                    cost_rc = min(cost_rc, take_comm_cost)

                    min_cost_for_goal = min(min_cost_for_goal, cost_rc)

                if min_cost_for_goal != float('inf'):
                    h += min_cost_for_goal
                else:
                    # Goal is unachievable by any suitable rover/camera. Add penalty.
                    h += 1000

        return h
