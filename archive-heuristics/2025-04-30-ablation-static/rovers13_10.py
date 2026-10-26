from fnmatch import fnmatch
from collections import deque
# Assuming Heuristic base class is available as heuristics.heuristic_base.Heuristic
# from heuristics.heuristic_base import Heuristic

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact string or malformed fact
    if not fact or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at rover1 waypoint1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# BFS for shortest paths
def bfs(graph, start_node):
    """Compute shortest path distances from start_node in a graph."""
    distances = {node: float('inf') for node in graph}
    if start_node not in graph: # Handle cases where start_node is not in the graph (e.g., isolated waypoint)
         return distances
    distances[start_node] = 0
    queue = deque([start_node])
    while queue:
        current_node = queue.popleft()
        # Ensure current_node is in graph before accessing neighbors
        if current_node in graph:
            for neighbor in graph[current_node]:
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)
    return distances

# Define the heuristic class
class roversHeuristic: # Inherit from Heuristic if available
# class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the minimum number of actions required to satisfy
    all goal conditions. It calculates the cost for each unachieved goal
    independently and sums them up. For each goal, it finds the minimum cost
    path considering available rovers, required equipment, sample/image locations,
    calibration targets, and communication waypoints. Navigation costs are
    estimated using precomputed shortest paths on the traversable waypoint graph
    for each rover.

    # Assumptions
    - The problem instance is solvable.
    - Action costs are uniform (1).
    - Samples, once taken, are no longer at the waypoint and are only available
      via the `have_soil_analysis` or `have_rock_analysis` predicates. If a goal
      requires a sample from waypoint W, and no rover has `have_soil_analysis ?r W`,
      it is assumed that `at_soil_sample W` is true and the sample needs to be taken.
    - Images, once taken, are available via the `have_image` predicate. Taking an
      image uncalibrates the camera. Calibration is typically needed before taking
      an image.
    - Communication requires the rover to be at a waypoint visible from the lander.

    # Heuristic Initialization
    - Extracts static facts: lander location, rover equipment, store ownership,
      camera properties (on-board rover, supported modes, calibration target),
      waypoint visibility, objective visibility from waypoints.
    - Extracts initial facts: initial sample locations.
    - Builds traversal graphs for each rover based on `can_traverse` facts.
    - Computes all-pairs shortest paths for each rover's traversal graph using BFS.
    - Identifies communication waypoints for each rover (waypoints reachable by the
      rover that are visible from the lander).

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify all goal conditions that are not yet satisfied in the current state.
    2. Initialize the total heuristic cost to 0.
    3. For each unachieved goal:
        a. Calculate the minimum estimated cost to achieve this specific goal, considering
           all possible suitable rovers and necessary intermediate steps (navigation,
           sampling/imaging, calibration, dropping sample, communication).
        b. If the goal is `(communicated_soil_data ?w)`:
           - Find rovers equipped for soil analysis.
           - For each suitable rover R:
             - Estimate cost: (Nav cost R from current loc to W if sample not held) + (1 for sample if needed) + (1 for drop if store full and sample needed) + (Nav cost R from W (or current loc if sample held) to nearest communication waypoint) + (1 for communicate).
             - Take the minimum cost over all suitable rovers.
        c. If the goal is `(communicated_rock_data ?w)`:
           - Similar logic to soil data, using rock-specific predicates and equipment.
        d. If the goal is `(communicated_image_data ?o ?m)`:
           - Find rovers equipped for imaging with a camera supporting mode M.
           - For each suitable rover R and camera I:
             - Find waypoints P visible from O and waypoints W visible from I's calibration target.
             - For each pair of P and W:
               - Estimate cost: (Nav cost R from current loc to W if camera not calibrated) + (1 for calibrate if needed) + (Nav cost R from W (or current loc if calibrated) to P) + (1 for take_image if needed) + (Nav cost R from P (or current loc if image held) to nearest communication waypoint) + (1 for communicate).
               - Take the minimum cost over all combinations of R, I, P, and W.
        e. Add the minimum estimated cost for this goal to the total heuristic cost. If a goal is determined to be unreachable by any suitable rover/path combination (e.g., no suitable rover, no reachable communication waypoint), add a large value (e.g., 1000).
    4. Return the total heuristic cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static
        initial_state = task.initial_state

        # --- Extract Static Information ---
        self.lander_location = None
        self.rovers = set()
        self.waypoint_names = set() # Use a different name to avoid conflict with graph nodes
        self.stores = set()
        self.cameras = set()
        self.modes = set()
        self.objectives = set()
        self.landers = set()

        # Helper mappings
        self.rover_equip = {} # rover -> set(equipment_type)
        self.rover_store = {} # rover -> store
        self.camera_info = {} # camera -> {'rover': rover, 'modes': set(modes), 'cal_target': objective}
        self.waypoint_visibility = {} # waypoint -> set(visible_waypoints)
        self.objective_visibility = {} # objective -> set(visible_from_waypoints)
        self.rover_traversal_graphs = {} # rover -> Dict[waypoint, set(neighbor_waypoints)]

        # Collect all objects and basic static relations
        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue

            pred = parts[0]
            if pred == 'at_lander':
                lander, wp = parts[1], parts[2]
                self.lander_location = wp # Assuming one lander
                self.landers.add(lander)
                self.waypoint_names.add(wp)
            elif pred == 'can_traverse':
                r, w1, w2 = parts[1], parts[2], parts[3]
                self.rovers.add(r)
                self.waypoint_names.add(w1)
                self.waypoint_names.add(w2)
                self.rover_traversal_graphs.setdefault(r, {}).setdefault(w1, set()).add(w2)
            elif pred.startswith('equipped_for_'):
                r = parts[1]
                self.rovers.add(r)
                equip_type = pred[len('equipped_for_'):] # e.g., 'soil_analysis'
                self.rover_equip.setdefault(r, set()).add(equip_type)
            elif pred == 'supports':
                cam, mode = parts[1], parts[2]
                self.cameras.add(cam)
                self.modes.add(mode)
                self.camera_info.setdefault(cam, {}).setdefault('modes', set()).add(mode)
            elif pred == 'visible':
                w1, w2 = parts[1], parts[2]
                self.waypoint_names.add(w1)
                self.waypoint_names.add(w2)
                self.waypoint_visibility.setdefault(w1, set()).add(w2)
            elif pred == 'visible_from':
                obj, wp = parts[1], parts[2]
                self.objectives.add(obj)
                self.waypoint_names.add(wp)
                self.objective_visibility.setdefault(obj, set()).add(wp)
            elif pred == 'store_of':
                store, r = parts[1], parts[2]
                self.stores.add(store)
                self.rovers.add(r)
                self.rover_store[r] = store # Assuming one store per rover
            elif pred == 'calibration_target':
                cam, obj = parts[1], parts[2]
                self.cameras.add(cam)
                self.objectives.add(obj)
                self.camera_info.setdefault(cam, {}).setdefault('cal_target', obj)
            elif pred == 'on_board':
                cam, r = parts[1], parts[2]
                self.cameras.add(cam)
                self.rovers.add(r)
                self.camera_info.setdefault(cam, {}).setdefault('rover', r)

        # Add objects from initial state if not seen in static
        for fact in initial_state:
             parts = get_parts(fact)
             if not parts: continue
             pred = parts[0]
             if pred == 'at':
                 obj_name, wp_name = parts[1], parts[2]
                 if obj_name.startswith('rover'): self.rovers.add(obj_name)
                 elif obj_name.startswith('lander'): self.landers.add(obj_name)
                 self.waypoint_names.add(wp_name)
             elif pred == 'at_soil_sample' or pred == 'at_rock_sample':
                 self.waypoint_names.add(parts[1])
             elif pred == 'empty' or pred == 'full':
                 self.stores.add(parts[1])
             elif pred == 'have_soil_analysis' or pred == 'have_rock_analysis':
                 self.rovers.add(parts[1])
                 self.waypoint_names.add(parts[2])
             elif pred == 'have_image':
                 self.rovers.add(parts[1])
                 self.objectives.add(parts[2])
                 self.modes.add(parts[3])
             elif pred == 'calibrated':
                 self.cameras.add(parts[1])
                 self.rovers.add(parts[2])
             elif pred.startswith('communicated_'):
                 if len(parts) > 1:
                     if parts[1].startswith('waypoint'): self.waypoint_names.add(parts[1])
                     elif parts[1].startswith('objective'): self.objectives.add(parts[1])
                     if len(parts) > 2 and parts[2].startswith('mode'): self.modes.add(parts[2])


        # Ensure all waypoints are included in rover graphs, even if isolated
        for r in self.rovers:
            if r not in self.rover_traversal_graphs:
                 self.rover_traversal_graphs[r] = {}
            for wp in self.waypoint_names:
                 self.rover_traversal_graphs[r].setdefault(wp, set())


        # --- Precompute Shortest Paths ---
        self.rover_shortest_paths = {}
        for rover, graph in self.rover_traversal_graphs.items():
            self.rover_shortest_paths[rover] = {}
            for start_wp in graph:
                 self.rover_shortest_paths[rover][start_wp] = bfs(graph, start_wp)

        # --- Identify Communication Waypoints ---
        self.rover_comm_waypoints = {}
        lander_wp = self.lander_location # Assuming one lander location from static facts
        if lander_wp:
            for rover in self.rovers:
                # Communication waypoints are those reachable by the rover AND visible from the lander
                reachable_wps = set(self.rover_shortest_paths.get(rover, {}).keys())
                comm_wps = set()
                for wp in reachable_wps:
                    # Check if wp is visible from lander_wp
                    if lander_wp in self.waypoint_visibility.get(wp, set()):
                        comm_wps.add(wp)
                self.rover_comm_waypoints[rover] = comm_wps
        else:
             # No lander location found, communication is impossible
             for rover in self.rovers:
                 self.rover_comm_waypoints[rover] = set()


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Check if goal is reached
        if self.goals <= state:
            return 0

        # --- Parse Current State ---
        current_rover_locations = {} # rover -> waypoint
        current_soil_samples_at_wp = set() # waypoints
        current_rock_samples_at_wp = set() # waypoints
        current_have_soil = set() # (rover, waypoint)
        current_have_rock = set() # (rover, waypoint)
        current_have_image = set() # (rover, objective, mode)
        current_full_stores = set() # stores
        current_calibrated_cameras = set() # (camera, rover)
        current_communicated_soil = set() # waypoints
        current_communicated_rock = set() # waypoints
        current_communicated_image = set() # (objective, mode)

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            pred = parts[0]
            if pred == 'at' and parts[1] in self.rovers:
                current_rover_locations[parts[1]] = parts[2]
            elif pred == 'at_soil_sample':
                current_soil_samples_at_wp.add(parts[1])
            elif pred == 'at_rock_sample':
                current_rock_samples_at_wp.add(parts[1])
            elif pred == 'have_soil_analysis':
                current_have_soil.add((parts[1], parts[2]))
            elif pred == 'have_rock_analysis':
                current_have_rock.add((parts[1], parts[2]))
            elif pred == 'have_image':
                current_have_image.add((parts[1], parts[2], parts[3]))
            elif pred == 'full':
                current_full_stores.add(parts[1])
            elif pred == 'calibrated':
                current_calibrated_cameras.add((parts[1], parts[2]))
            elif pred == 'communicated_soil_data':
                current_communicated_soil.add(parts[1])
            elif pred == 'communicated_rock_data':
                current_communicated_rock.add(parts[1])
            elif pred == 'communicated_image_data':
                current_communicated_image.add((parts[1], parts[2]))

        total_cost = 0

        # --- Calculate Cost for Each Unachieved Goal ---
        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue

            pred = parts[0]

            if pred == 'communicated_soil_data':
                w = parts[1]
                if w in current_communicated_soil:
                    continue # Goal already achieved

                min_goal_cost = float('inf')

                # Find rovers equipped for soil analysis
                suitable_rovers = [r for r in self.rovers if 'soil_analysis' in self.rover_equip.get(r, set())]

                if not suitable_rovers:
                    # Problem likely unsolvable if this goal is required and no rover is equipped
                    min_goal_cost = 1000 # Large penalty
                else:
                    for r in suitable_rovers:
                        rover_cost = 0
                        rover_at = current_rover_locations.get(r)
                        if rover_at is None: continue # Rover location unknown (shouldn't happen in valid state)

                        has_sample = (r, w) in current_have_soil

                        loc_after_sample = rover_at # Default location if sample is held or doesn't need sampling

                        if not has_sample:
                            # Need to sample
                            # Assuming sample is at waypoint w if not held by any rover
                            nav_cost_to_sample = self.rover_shortest_paths.get(r, {}).get(rover_at, {}).get(w, float('inf'))
                            if nav_cost_to_sample == float('inf'):
                                # Cannot reach sample location
                                continue # Try next rover/path

                            rover_cost += nav_cost_to_sample + 1 # navigate + sample
                            loc_after_sample = w

                            # Check if store is full, need to drop before sampling
                            store = self.rover_store.get(r)
                            if store and store in current_full_stores:
                                 rover_cost += 1 # drop

                        # Need to communicate
                        comm_wps = self.rover_comm_waypoints.get(r, set())
                        if not comm_wps:
                            # No communication waypoint reachable for this rover
                            continue # Try next rover/path

                        min_nav_cost_to_comm = float('inf')
                        if loc_after_sample in self.rover_shortest_paths.get(r, {}):
                            min_nav_cost_to_comm = min(
                                self.rover_shortest_paths[r][loc_after_sample].get(comm_wp, float('inf'))
                                for comm_wp in comm_wps
                                if self.rover_shortest_paths[r].get(loc_after_sample, {}).get(comm_wp, float('inf')) != float('inf') # Ensure path exists
                            ) if comm_wps else float('inf') # Handle empty comm_wps case

                        if min_nav_cost_to_comm == float('inf'):
                            # Cannot reach any communication waypoint from current location
                            continue # Try next rover/path

                        rover_cost += min_nav_cost_to_comm + 1 # navigate + communicate

                        min_goal_cost = min(min_goal_cost, rover_cost)

                if min_goal_cost != float('inf'):
                    total_cost += min_goal_cost
                else:
                    # Goal seems unreachable by any suitable rover/path
                    total_cost += 1000 # Large penalty


            elif pred == 'communicated_rock_data':
                w = parts[1]
                if w in current_communicated_rock:
                    continue # Goal already achieved

                min_goal_cost = float('inf')

                # Find rovers equipped for rock analysis
                suitable_rovers = [r for r in self.rovers if 'rock_analysis' in self.rover_equip.get(r, set())]

                if not suitable_rovers:
                    min_goal_cost = 1000 # Large penalty
                else:
                    for r in suitable_rovers:
                        rover_cost = 0
                        rover_at = current_rover_locations.get(r)
                        if rover_at is None: continue

                        has_sample = (r, w) in current_have_rock

                        loc_after_sample = rover_at

                        if not has_sample:
                            # Need to sample
                            nav_cost_to_sample = self.rover_shortest_paths.get(r, {}).get(rover_at, {}).get(w, float('inf'))
                            if nav_cost_to_sample == float('inf'):
                                continue

                            rover_cost += nav_cost_to_sample + 1 # navigate + sample
                            loc_after_sample = w

                            store = self.rover_store.get(r)
                            if store and store in current_full_stores:
                                 rover_cost += 1 # drop

                        # Need to communicate
                        comm_wps = self.rover_comm_waypoints.get(r, set())
                        if not comm_wps:
                            continue

                        min_nav_cost_to_comm = float('inf')
                        if loc_after_sample in self.rover_shortest_paths.get(r, {}):
                             min_nav_cost_to_comm = min(
                                self.rover_shortest_paths[r][loc_after_sample].get(comm_wp, float('inf'))
                                for comm_wp in comm_wps
                                if self.rover_shortest_paths[r].get(loc_after_sample, {}).get(comm_wp, float('inf')) != float('inf') # Ensure path exists
                            ) if comm_wps else float('inf') # Handle empty comm_wps case


                        if min_nav_cost_to_comm == float('inf'):
                            continue

                        rover_cost += min_nav_cost_to_comm + 1 # navigate + communicate

                        min_goal_cost = min(min_goal_cost, rover_cost)

                if min_goal_cost != float('inf'):
                    total_cost += min_goal_cost
                else:
                    total_cost += 1000 # Large penalty


            elif pred == 'communicated_image_data':
                o, m = parts[1], parts[2]
                if (o, m) in current_communicated_image:
                    continue # Goal already achieved

                min_goal_cost = float('inf')

                # Find rovers equipped for imaging with a camera supporting mode m
                suitable_rover_camera_pairs = []
                for r in self.rovers:
                    if 'imaging' in self.rover_equip.get(r, set()):
                        for cam, info in self.camera_info.items():
                            if info.get('rover') == r and m in info.get('modes', set()):
                                suitable_rover_camera_pairs.append((r, cam))

                if not suitable_rover_camera_pairs:
                    min_goal_cost = 1000 # Large penalty
                else:
                    # Find waypoints visible from objective o
                    image_wps = self.objective_visibility.get(o, set())
                    if not image_wps:
                         # Cannot take image
                         min_goal_cost = 1000 # Large penalty
                    else:
                        for r, cam in suitable_rover_camera_pairs:
                            rover_at = current_rover_locations.get(r)
                            if rover_at is None: continue

                            has_image = (r, o, m) in current_have_image

                            loc_after_image = rover_at # Default location if image is held

                            if not has_image:
                                # Need to take image
                                is_calibrated = (cam, r) in current_calibrated_cameras

                                # Find calibration target and waypoints visible from it
                                cal_target = self.camera_info.get(cam, {}).get('cal_target')
                                cal_wps = self.objective_visibility.get(cal_target, set()) if cal_target else set()

                                if not cal_target or (not cal_wps and not is_calibrated):
                                     # Cannot calibrate or no calibration target/waypoint
                                     # If not calibrated and cannot calibrate, cannot take image
                                     if not is_calibrated:
                                         continue # Try next rover/camera pair

                                min_image_path_cost = float('inf')

                                # Iterate through possible image waypoints
                                for p in image_wps:
                                    # Iterate through possible calibration waypoints (only if calibration is needed)
                                    # If not calibrated, must calibrate at a cal_wp. If calibrated, no cal step needed (w=None)
                                    cal_options = list(cal_wps) if not is_calibrated else [None]

                                    for w in cal_options:
                                        path_cost = 0
                                        current_loc_for_cal = rover_at
                                        current_loc_for_image = rover_at

                                        if not is_calibrated:
                                            # Need to calibrate
                                            if w is None: continue # Should not happen if not calibrated

                                            nav_cost_to_cal = self.rover_shortest_paths.get(r, {}).get(current_loc_for_cal, {}).get(w, float('inf'))
                                            if nav_cost_to_cal == float('inf'):
                                                continue

                                            path_cost += nav_cost_to_cal + 1 # navigate + calibrate
                                            current_loc_for_image = w # Rover is now at calibration waypoint

                                        # Need to take image
                                        nav_cost_to_image = self.rover_shortest_paths.get(r, {}).get(current_loc_for_image, {}).get(p, float('inf'))
                                        if nav_cost_to_image == float('inf'):
                                            continue

                                        path_cost += nav_cost_to_image + 1 # navigate + take_image
                                        loc_after_image = p # Rover is now at image waypoint

                                        # Need to communicate
                                        comm_wps = self.rover_comm_waypoints.get(r, set())
                                        if not comm_wps:
                                            continue

                                        min_nav_cost_to_comm = float('inf')
                                        if loc_after_image in self.rover_shortest_paths.get(r, {}):
                                             min_nav_cost_to_comm = min(
                                                self.rover_shortest_paths[r][loc_after_image].get(comm_wp, float('inf'))
                                                for comm_wp in comm_wps
                                                if self.rover_shortest_paths[r].get(loc_after_image, {}).get(comm_wp, float('inf')) != float('inf') # Ensure path exists
                                            ) if comm_wps else float('inf') # Handle empty comm_wps case


                                        if min_nav_cost_to_comm == float('inf'):
                                            continue

                                        path_cost += min_nav_cost_to_comm + 1 # navigate + communicate

                                        min_image_path_cost = min(min_image_path_cost, path_cost)

                                if min_image_path_cost != float('inf'):
                                     min_goal_cost = min(min_goal_cost, min_image_path_cost)
                                # else: this specific rover/camera/waypoint combo cannot achieve the goal, try others

                            # If image is already held by this rover
                            if has_image:
                                # Need to communicate
                                comm_wps = self.rover_comm_waypoints.get(r, set())
                                if not comm_wps:
                                    continue

                                min_nav_cost_to_comm = float('inf')
                                if rover_at in self.rover_shortest_paths.get(r, {}):
                                     min_nav_cost_to_comm = min(
                                        self.rover_shortest_paths[r][rover_at].get(comm_wp, float('inf'))
                                        for comm_wp in comm_wps
                                        if self.rover_shortest_paths[r].get(rover_at, {}).get(comm_wp, float('inf')) != float('inf') # Ensure path exists
                                    ) if comm_wps else float('inf') # Handle empty comm_wps case


                                if min_nav_cost_to_comm != float('inf'):
                                    rover_cost = min_nav_cost_to_comm + 1 # navigate + communicate
                                    min_goal_cost = min(min_goal_cost, rover_cost)
                                # else: this specific rover cannot communicate from its current location, try others


                if min_goal_cost != float('inf'):
                    total_cost += min_goal_cost
                else:
                    total_cost += 1000 # Large penalty


        return total_cost
