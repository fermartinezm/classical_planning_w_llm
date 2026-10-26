import collections
from fnmatch import fnmatch
# Assuming heuristics.heuristic_base exists in the environment
from heuristics.heuristic_base import Heuristic

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at rover1 waypoint1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure parts has enough elements to match args
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def bfs_shortest_paths(graph, start_node, all_possible_nodes):
    """
    Computes shortest path distances from a start_node to all other nodes
    in a graph using BFS.

    Args:
        graph: A dictionary representing the graph {node: set(neighbors)}.
        start_node: The node to start the BFS from.
        all_possible_nodes: A set of all nodes that could potentially be in the graph.

    Returns:
        A dictionary {node: distance} containing shortest distances.
        Returns float('inf') for unreachable nodes.
    """
    distances = {node: float('inf') for node in all_possible_nodes}

    if start_node not in all_possible_nodes:
         return distances # Start node is not a valid node

    distances[start_node] = 0
    queue = collections.deque([start_node])

    while queue:
        current_node = queue.popleft()

        # Use .get(current_node, set()) to handle nodes that might be in all_possible_nodes
        # but have no outgoing edges defined in the graph dictionary.
        for neighbor in graph.get(current_node, set()):
            if neighbor in distances and distances[neighbor] == float('inf'): # Check if neighbor is a valid node and not visited
                distances[neighbor] = distances[current_node] + 1
                queue.append(neighbor)

    return distances


class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve all
    goal conditions. It sums the estimated costs for each unachieved goal
    independently. For each goal (communicating soil, rock, or image data),
    it estimates the cost of performing the necessary steps (sampling/imaging,
    navigating to a communication point, and communicating), considering
    the current state and the capabilities/locations of the rovers.
    Navigation costs are estimated using precomputed shortest path distances
    on the waypoint graph for each rover.

    # Assumptions
    - The heuristic assumes that for each goal, there exists at least one
      rover capable of performing the necessary tasks (sampling/imaging and
      navigation). If no such rover exists or the required locations
      (sample/rock sites, image/calibration waypoints, communication points)
      are unreachable, the heuristic adds a large penalty.
    - It simplifies the problem by estimating the cost for each unachieved
      goal independently, potentially overestimating the total cost if
      actions contribute to multiple goals.
    - It assumes action costs are uniform (1).
    - It assumes that if a sample/image is already collected, the same rover
      that collected it (if known from state) will be used for communication.
    - It assumes that if a camera needs calibration, the rover will navigate
      to the *closest* suitable calibration waypoint and then from there to
      the *closest* suitable image waypoint.

    # Heuristic Initialization
    The initialization phase precomputes static information from the task,
    which remains constant throughout the search:
    - The lander's waypoint.
    - The set of waypoints visible from the lander (communication points).
    - Rover capabilities (soil, rock, imaging).
    - Store ownership for each rover.
    - Initial locations of soil and rock samples (from the initial state).
    - Waypoints from which objectives are visible (for imaging).
    - Camera information (on-board rover, supported modes, calibration target).
    - Waypoints from which calibration targets are visible (derived from objective visibility).
    - The navigation graph for each rover based on `can_traverse` facts.
    - Shortest path distances between all pairs of waypoints for each rover
      using BFS.
    - Minimum distances from any waypoint to any communication waypoint for
      each rover.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic is computed as follows:

    1.  Initialize `total_cost` to 0.
    2.  Extract relevant dynamic information from the current state:
        - Current location of each rover.
        - Status of each store (full or empty).
        - Which soil/rock samples have been collected by which rover.
        - Which images have been taken by which rover/camera/mode.
        - Which cameras are calibrated on which rovers.
    3.  Identify which goal facts are not yet achieved.
    4.  For each unachieved goal fact `g`:
        - If `g` is `(communicated_soil_data ?w)`:
            - Estimate cost. Check if `(have_soil_analysis ?r ?w)` exists for any rover `r`.
            - If not collected: Find the equipped rover `r_s` that minimizes (nav cost to `?w` + store drop cost + sample action cost). Add this minimum cost. The conceptual location of `r_s` is now `?w`.
            - If collected by rover `r_h`: The conceptual rover is `r_h`, conceptual location is its current location.
            - Add navigation cost for the conceptual rover from its conceptual location to the nearest communication waypoint.
            - Add 1 for the `communicate_soil_data` action.
            - Add a large penalty if any step was impossible (e.g., no equipped rover, unreachable waypoint).
            - Add this estimated cost to `total_cost`.
        - If `g` is `(communicated_rock_data ?w)`: Similar logic as for soil data.
        - If `g` is `(communicated_image_data ?o ?m)`:
            - Estimate cost. Check if `(have_image ?r ?o ?m)` exists for any suitable rover `r`.
            - If not taken: Find the suitable rover/camera combo (`r_i`, `c_i`) that minimizes the cost to take the image. This cost includes:
                - If `c_i` is not calibrated: Nav cost from `r_i`'s current location to the closest calibration waypoint for `c_i`, plus 1 for `calibrate`. Conceptual location is the calibration waypoint.
                - Nav cost from the conceptual location (after calibration or initial) to the closest image waypoint for `?o`, plus 1 for `take_image`. Conceptual location is the image waypoint.
                Add this minimum cost. The conceptual rover is `r_i`, conceptual location is the image waypoint.
            - If taken by rover `r_h`: The conceptual rover is `r_h`, conceptual location is its current location.
            - Add navigation cost for the conceptual rover from its conceptual location to the nearest communication waypoint.
            - Add 1 for the `communicate_image_data` action.
            - Add a large penalty if any step was impossible.
            - Add this estimated cost to `total_cost`.
    5.  Return `total_cost`.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static
        initial_state = task.initial_state # Need initial state for sample locations

        # --- Precomputation ---

        self.lander_waypoint = None
        self.communication_waypoint_set = set()
        self.rovers = set()
        self.stores = set()
        self.cameras = set()
        self.modes = set()
        self.objectives = set()
        self.waypoints = set() # Collect all waypoints mentioned anywhere

        self.rover_capabilities = collections.defaultdict(lambda: {'soil': False, 'rock': False, 'imaging': False})
        self.store_of_rover = {} # {rover: store}
        self.initial_soil_samples_at = set() # Waypoints with soil samples initially
        self.initial_rock_samples_at = set() # Waypoints with rock samples initially
        self.objective_visible_from = collections.defaultdict(set) # {objective: set(waypoint)}
        self.camera_on_board = {} # {camera: rover}
        self.camera_supports = collections.defaultdict(set) # {camera: set(mode)}
        self.camera_cal_target = {} # {camera: objective}
        self.cal_target_visible_from = collections.defaultdict(set) # {camera: set(waypoint)}

        # Parse static facts and initial state for static/initial info
        for fact in static_facts | initial_state:
            parts = get_parts(fact)
            if not parts: continue # Skip empty facts

            predicate = parts[0]

            if predicate == 'at_lander':
                # Assuming only one lander
                if len(parts) > 2: self.lander_waypoint = parts[2]
            elif predicate == 'rover':
                 if len(parts) > 1: self.rovers.add(parts[1])
            elif predicate == 'store':
                 if len(parts) > 1: self.stores.add(parts[1])
            elif predicate == 'camera':
                 if len(parts) > 1: self.cameras.add(parts[1])
            elif predicate == 'mode':
                 if len(parts) > 1: self.modes.add(parts[1])
            elif predicate == 'objective':
                 if len(parts) > 1: self.objectives.add(parts[1])
            elif predicate == 'waypoint':
                 if len(parts) > 1: self.waypoints.add(parts[1])
            elif predicate == 'equipped_for_soil_analysis':
                if len(parts) > 1: self.rover_capabilities[parts[1]]['soil'] = True
            elif predicate == 'equipped_for_rock_analysis':
                if len(parts) > 1: self.rover_capabilities[parts[1]]['rock'] = True
            elif predicate == 'equipped_for_imaging':
                if len(parts) > 1: self.rover_capabilities[parts[1]]['imaging'] = True
            elif predicate == 'store_of':
                if len(parts) > 2: self.store_of_rover[parts[2]] = parts[1] # rover -> store
            elif predicate == 'at_soil_sample' and fact in initial_state:
                 if len(parts) > 1: self.initial_soil_samples_at.add(parts[1])
            elif predicate == 'at_rock_sample' and fact in initial_state:
                 if len(parts) > 1: self.initial_rock_samples_at.add(parts[1])
            elif predicate == 'visible_from':
                 if len(parts) > 2: self.objective_visible_from[parts[1]].add(parts[2]) # objective -> waypoint
            elif predicate == 'on_board':
                 if len(parts) > 2: self.camera_on_board[parts[1]] = parts[2] # camera -> rover
            elif predicate == 'supports':
                 if len(parts) > 2: self.camera_supports[parts[1]].add(parts[2]) # camera -> mode
            elif predicate == 'calibration_target':
                 if len(parts) > 2: self.camera_cal_target[parts[1]] = parts[2] # camera -> objective
            elif predicate == 'at' and len(parts) > 2 and parts[1] in self.rovers:
                 # Also collect waypoints from initial rover locations
                 self.waypoints.add(parts[2])
            elif predicate == 'at_lander' and len(parts) > 2:
                 # Also collect lander waypoint
                 self.waypoints.add(parts[2])
            elif predicate == 'can_traverse' and len(parts) > 3:
                 # Also collect waypoints from can_traverse
                 self.waypoints.add(parts[2])
                 self.waypoints.add(parts[3])
            elif predicate == 'visible' and len(parts) > 2:
                 # Also collect waypoints from visible
                 self.waypoints.add(parts[1])
                 self.waypoints.add(parts[2])


        # Now that we know calibration targets, populate cal_target_visible_from
        for cam, obj in self.camera_cal_target.items():
             if obj in self.objective_visible_from:
                 # Calibration for camera 'cam' using target 'obj' can happen
                 # at any waypoint 'w' where 'obj' is visible from 'w'.
                 self.cal_target_visible_from[cam].update(self.objective_visible_from[obj])


        # Build rover navigation graphs
        self.rover_waypoint_graph = collections.defaultdict(lambda: collections.defaultdict(set))
        for fact in static_facts:
            if match(fact, "can_traverse", "*", "*", "*"):
                _, rover, wp_from, wp_to = get_parts(fact)
                self.rover_waypoint_graph[rover][wp_from].add(wp_to)

        # Precompute shortest paths for each rover
        self.rover_distances = {}
        for rover in self.rovers:
            self.rover_distances[rover] = {}
            # Use the comprehensive set of all known waypoints
            all_waypoints_for_rover = self.waypoints

            for start_wp in all_waypoints_for_rover:
                 self.rover_distances[rover][start_wp] = bfs_shortest_paths(self.rover_waypoint_graph.get(rover, {}), start_wp, all_waypoints_for_rover)


        # Precompute minimum distances to communication waypoints
        # Ensure lander waypoint is known before computing communication waypoints
        if self.lander_waypoint:
             for fact in static_facts:
                 if match(fact, "visible", "*", self.lander_waypoint):
                     self.communication_waypoint_set.add(get_parts(fact)[1])

        self.rover_min_dist_to_comm = {}
        for rover in self.rovers:
            self.rover_min_dist_to_comm[rover] = {}
            all_waypoints = self.waypoints # Compute for all known waypoints
            for from_wp in all_waypoints:
                min_dist = float('inf')
                if self.communication_waypoint_set: # Only compute if communication waypoints exist
                    min_dist = min((self.rover_distances[rover][from_wp].get(comm_wp, float('inf')) for comm_wp in self.communication_waypoint_set), default=float('inf'))
                self.rover_min_dist_to_comm[rover][from_wp] = min_dist


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # --- Extract Dynamic State Information ---
        rover_at = {} # {rover: waypoint}
        store_is_full = {} # {store: bool}
        have_soil = set() # {(rover, waypoint)}
        have_rock = set() # {(rover, waypoint)}
        have_image = set() # {(rover, objective, mode)}
        calibrated_cams = set() # {(camera, rover)}

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]

            if predicate == 'at' and len(parts) > 2 and parts[1] in self.rovers:
                rover_at[parts[1]] = parts[2]
            elif predicate == 'full' and len(parts) > 1 and parts[1] in self.stores:
                store_is_full[parts[1]] = True
            elif predicate == 'empty' and len(parts) > 1 and parts[1] in self.stores:
                 store_is_full[parts[1]] = False # Explicitly mark empty
            elif predicate == 'have_soil_analysis' and len(parts) > 2:
                # Ensure the rover is a known rover
                if parts[1] in self.rovers:
                    have_soil.add((parts[1], parts[2]))
            elif predicate == 'have_rock_analysis' and len(parts) > 2:
                 # Ensure the rover is a known rover
                if parts[1] in self.rovers:
                    have_rock.add((parts[1], parts[2]))
            elif predicate == 'have_image' and len(parts) > 3:
                 # Ensure the rover is a known rover
                if parts[1] in self.rovers:
                    have_image.add((parts[1], parts[2], parts[3]))
            elif predicate == 'calibrated' and len(parts) > 2:
                 # Ensure the camera and rover are known
                if parts[1] in self.cameras and parts[2] in self.rovers:
                    calibrated_cams.add((parts[1], parts[2]))


        # Ensure all stores are accounted for (default to empty if not mentioned)
        for store in self.stores:
             store_is_full.setdefault(store, False)


        # --- Compute Heuristic ---
        total_cost = 0
        IMPOSSIBLE_PENALTY = 1000 # Penalty for steps that seem impossible

        # Track which samples/images are needed for goals but not yet communicated
        needed_soil_samples = set() # {waypoint}
        needed_rock_samples = set() # {waypoint}
        needed_images = set() # {(objective, mode)}

        for goal in self.goals:
            if goal in state:
                continue # Goal already achieved

            parts = get_parts(goal)
            if not parts: continue

            predicate = parts[0]

            if predicate == 'communicated_soil_data' and len(parts) == 2:
                needed_soil_samples.add(parts[1])
            elif predicate == 'communicated_rock_data' and len(parts) == 2:
                needed_rock_samples.add(parts[1])
            elif predicate == 'communicated_image_data' and len(parts) == 3:
                needed_images.add((parts[1], parts[2]))
            # Ignore other potential goal types or malformed goals


        # Estimate cost for needed soil samples
        for wp in needed_soil_samples:
            cost_soil_goal = 0
            sampling_rover = None
            rover_loc_after_task = None # Conceptual location after sampling

            # Check if sample is already collected by any rover
            for r in self.rovers:
                if (r, wp) in have_soil:
                    sampling_rover = r
                    rover_loc_after_task = rover_at.get(r) # Rover is at its current location
                    break # Found a rover with the sample

            if sampling_rover is None: # Not already sampled
                # Need to sample
                min_sample_cost = float('inf')
                best_rover_for_sample = None
                best_rover_loc_after_sample = None

                # Find the best equipped rover to sample
                for r in self.rovers:
                    if self.rover_capabilities[r]['soil']:
                        current_wp = rover_at.get(r)
                        if current_wp is None: continue # Rover location unknown

                        nav_cost = self.rover_distances.get(r, {}).get(current_wp, {}).get(wp, float('inf'))
                        if nav_cost == float('inf'): continue # Cannot reach sample location

                        store = self.store_of_rover.get(r)
                        # Check if store exists and is full
                        store_cost = 1 if store is not None and store_is_full.get(store, False) else 0 # Cost to drop if store is full

                        sample_cost = nav_cost + store_cost + 1 # Nav + Drop (if needed) + Sample

                        if sample_cost < min_sample_cost:
                            min_sample_cost = sample_cost
                            best_rover_for_sample = r
                            best_rover_loc_after_sample = wp # Rover is at sample location after sampling

                if best_rover_for_sample is not None and min_sample_cost != float('inf'):
                    cost_soil_goal += min_sample_cost
                    sampling_rover = best_rover_for_sample
                    rover_loc_after_task = best_rover_loc_after_sample
                else:
                    cost_soil_goal += IMPOSSIBLE_PENALTY # Penalty if sampling seems impossible

            # Need to communicate (either just sampled or already had sample)
            if sampling_rover is not None and rover_loc_after_task is not None:
                 nav_cost_to_comm = self.rover_min_dist_to_comm.get(sampling_rover, {}).get(rover_loc_after_task, float('inf'))
                 if nav_cost_to_comm != float('inf'):
                     cost_soil_goal += nav_cost_to_comm + 1 # Nav to comm + Communicate action
                 else:
                     cost_soil_goal += IMPOSSIBLE_PENALTY # Penalty if communication waypoint unreachable
            elif sampling_rover is not None and rover_loc_after_task is None:
                 # Rover has sample but its current location is unknown? Should not happen.
                 cost_soil_goal += IMPOSSIBLE_PENALTY # Penalty
            else:
                 # No rover capable/having sample found initially or after sampling attempt
                 pass # Penalty already added in sampling step if needed

            total_cost += cost_soil_goal


        # Estimate cost for needed rock samples (similar logic)
        for wp in needed_rock_samples:
            cost_rock_goal = 0
            sampling_rover = None
            rover_loc_after_task = None # Conceptual location after sampling

            # Check if sample is already collected by any rover
            for r in self.rovers:
                if (r, wp) in have_rock:
                    sampling_rover = r
                    rover_loc_after_task = rover_at.get(r) # Rover is at its current location
                    break # Found a rover with the sample

            if sampling_rover is None: # Not already sampled
                # Need to sample
                min_sample_cost = float('inf')
                best_rover_for_sample = None
                best_rover_loc_after_sample = None

                # Find the best equipped rover to sample
                for r in self.rovers:
                    if self.rover_capabilities[r]['rock']:
                        current_wp = rover_at.get(r)
                        if current_wp is None: continue # Rover location unknown

                        nav_cost = self.rover_distances.get(r, {}).get(current_wp, {}).get(wp, float('inf'))
                        if nav_cost == float('inf'): continue # Cannot reach sample location

                        store = self.store_of_rover.get(r)
                        # Check if store exists and is full
                        store_cost = 1 if store is not None and store_is_full.get(store, False) else 0 # Cost to drop if store is full

                        sample_cost = nav_cost + store_cost + 1 # Nav + Drop (if needed) + Sample

                        if sample_cost < min_sample_cost:
                            min_sample_cost = sample_cost
                            best_rover_for_sample = r
                            best_rover_loc_after_sample = wp # Rover is at sample location after sampling

                if best_rover_for_sample is not None and min_sample_cost != float('inf'):
                    cost_rock_goal += min_sample_cost
                    sampling_rover = best_rover_for_sample
                    rover_loc_after_task = best_rover_loc_after_sample
                else:
                    cost_rock_goal += IMPOSSIBLE_PENALTY # Penalty

            # Need to communicate (either just sampled or already had sample)
            if sampling_rover is not None and rover_loc_after_task is not None:
                 nav_cost_to_comm = self.rover_min_dist_to_comm.get(sampling_rover, {}).get(rover_loc_after_task, float('inf'))
                 if nav_cost_to_comm != float('inf'):
                     cost_rock_goal += nav_cost_to_comm + 1 # Nav to comm + Communicate action
                 else:
                     cost_rock_goal += IMPOSSIBLE_PENALTY # Penalty
            elif sampling_rover is not None and rover_loc_after_task is None:
                 cost_rock_goal += IMPOSSIBLE_PENALTY # Penalty
            else:
                 pass # Penalty already added

            total_cost += cost_rock_goal


        # Estimate cost for needed images
        for obj, mode in needed_images:
            cost_image_goal = 0
            imaging_rover = None
            rover_loc_after_task = None # Conceptual location after imaging

            # Check if image is already taken by any rover/camera
            for r in self.rovers:
                # Find cameras on this rover supporting this mode
                suitable_cameras = {c for c in self.cameras if self.camera_on_board.get(c) == r and mode in self.camera_supports.get(c, set())}
                for c in suitable_cameras:
                    if (r, obj, mode) in have_image:
                        imaging_rover = r
                        rover_loc_after_task = rover_at.get(r) # Rover is at its current location
                        break # Found a rover with the image
                if imaging_rover is not None: break # Found the rover

            if imaging_rover is None: # Not already imaged
                # Need to take image
                min_image_cost = float('inf')
                best_rover_for_image = None
                best_rover_loc_after_image = None

                # Find the best equipped rover/camera combo to take the image
                for r in self.rovers:
                    if self.rover_capabilities[r]['imaging']:
                        current_wp = rover_at.get(r)
                        if current_wp is None: continue # Rover location unknown

                        for c in self.cameras:
                            if self.camera_on_board.get(c) == r and mode in self.camera_supports.get(c, set()):
                                # This rover/camera can potentially take the image
                                cost_this_combo = 0
                                current_wp_after_cal = current_wp # Start location for imaging nav

                                # Check calibration
                                calibrated = (c, r) in calibrated_cams
                                current_wp_for_cal = current_wp # Start location for calibration nav

                                if not calibrated:
                                    # Need to calibrate
                                    cal_target_obj = self.camera_cal_target.get(c)
                                    if cal_target_obj is None: continue # Camera has no calibration target

                                    cal_wps = self.cal_target_visible_from.get(c, set()) # Waypoints visible from the target
                                    if not cal_wps: continue # No waypoints to calibrate from

                                    min_nav_to_cal = float('inf')
                                    closest_cal_wp = None
                                    # Find the closest calibration waypoint from the current location
                                    if current_wp_for_cal in self.rover_distances.get(r, {}):
                                        for cal_wp in cal_wps:
                                             dist = self.rover_distances[r][current_wp_for_cal].get(cal_wp, float('inf'))
                                             if dist < min_nav_to_cal:
                                                  min_nav_to_cal = dist
                                                  closest_cal_wp = cal_wp

                                    if closest_cal_wp is None or min_nav_to_cal == float('inf'): continue # Cannot reach any cal waypoint

                                    cost_this_combo += min_nav_to_cal + 1 # Nav + Calibrate
                                    current_wp_after_cal = closest_cal_wp # Rover is now conceptually at this waypoint
                                else:
                                    # Already calibrated, start imaging nav from current location
                                    current_wp_after_cal = current_wp

                                # Need to navigate to image waypoint
                                image_wps = self.objective_visible_from.get(obj, set())
                                if not image_wps: continue # No waypoints to view objective from

                                min_nav_to_image = float('inf')
                                closest_image_wp = None
                                # Find the closest image waypoint from the location after calibration (or current)
                                if current_wp_after_cal is not None and current_wp_after_cal in self.rover_distances.get(r, {}):
                                    for img_wp in image_wps:
                                         dist = self.rover_distances[r][current_wp_after_cal].get(img_wp, float('inf'))
                                         if dist < min_nav_to_image:
                                              min_nav_to_image = dist
                                              closest_image_wp = img_wp
                                else:
                                     # Should not happen if current_wp was valid and cal nav was possible, but safety check
                                     pass # min_nav_to_image remains inf

                                if closest_image_wp is None or min_nav_to_image == float('inf'): continue # Cannot reach any image waypoint

                                cost_this_combo += min_nav_to_image + 1 # Nav + Take_image
                                rover_at_after_image = closest_image_wp # Rover is now conceptually at this waypoint

                                # If we reached here, this combo is possible. Check if it's the best so far.
                                if cost_this_combo < min_image_cost:
                                    min_image_cost = cost_this_combo
                                    best_rover_for_image = r
                                    best_rover_loc_after_image = rover_at_after_image

                if best_rover_for_image is not None and min_image_cost != float('inf'):
                    cost_image_goal += min_image_cost
                    imaging_rover = best_rover_for_image
                    rover_loc_after_task = best_rover_loc_after_image
                else:
                    cost_image_goal += IMPOSSIBLE_PENALTY # Penalty

            # Need to communicate (either just took image or already had image)
            if imaging_rover is not None and rover_loc_after_task is not None:
                 nav_cost_to_comm = self.rover_min_dist_to_comm.get(imaging_rover, {}).get(rover_loc_after_task, float('inf'))
                 if nav_cost_to_comm != float('inf'):
                     cost_image_goal += nav_cost_to_comm + 1 # Nav to comm + Communicate action
                 else:
                     cost_image_goal += IMPOSSIBLE_PENALTY # Penalty if communication waypoint unreachable
            elif imaging_rover is not None and rover_loc_after_task is None:
                 cost_image_goal += IMPOSSIBLE_PENALTY # Penalty
            else:
                 pass # Penalty already added

            total_cost += cost_image_goal

        return total_cost
