from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque
# import sys # Not needed if using float('inf')

# Helper functions to parse PDDL facts represented as strings
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
    # Match parts against args using fnmatch. zip stops at the shortest iterable.
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the total number of actions required to satisfy all
    uncommunicated goal conditions. It sums the estimated cost for each individual
    uncommunicated goal fact (soil data, rock data, image data). The cost for
    each item is estimated based on whether the data is already held by a rover
    or needs to be acquired (sampled or imaged) and then communicated. Navigation
    costs are estimated using precomputed shortest path distances between waypoints.

    # Assumptions
    - The cost of each action (navigate, sample, drop, calibrate, take_image, communicate) is 1.
    - Navigation is possible between waypoints connected by `can_traverse` for a rover. We assume `can_traverse` is symmetric and the same for all rovers for navigation cost calculation.
    - If a sample (soil/rock) is required for a goal but is no longer present at its initial location in the current state and no rover holds it, the goal is considered unachievable from this state (cost infinity).
    - If an image is required for a goal but the objective has no visible waypoints, or the required camera cannot be calibrated, the goal is considered unachievable (cost infinity).
    - Resource constraints like store capacity (beyond needing a drop before sampling) or camera calibration state after taking an image are simplified or ignored in the cost estimation for acquiring multiple items. The heuristic estimates the *minimum* cost for *one* item assuming resources are available when needed for that specific item.
    - The heuristic sums costs for individual goals, ignoring potential positive or negative interactions (e.g., navigating to a waypoint that serves multiple purposes, or needing to drop a sample to take another).

    # Heuristic Initialization
    The heuristic precomputes static information from the task:
    - Identifies all goal facts related to communication of soil, rock, and image data.
    - Determines rover capabilities (soil, rock, imaging).
    - Maps stores to their owning rovers.
    - Finds the lander's location and identifies communication waypoints (visible from the lander).
    - Builds a navigation graph based on `can_traverse` facts and computes all-pairs shortest path distances between waypoints using BFS.
    - Maps objectives to waypoints from which they are visible.
    - Maps cameras to their calibration targets and calibration targets to waypoints from which they are visible.
    - Maps cameras to supported modes and rovers to cameras on board.
    - Records initial locations of soil and rock samples.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic calculates the estimated cost as follows:

    1.  Initialize total heuristic `h = 0`.
    2.  Parse the current state to get the locations of rovers, which data/images rovers currently hold, the status of stores (full/empty), and the presence of soil/rock samples at waypoints, and camera calibration status.
    3.  For each soil data goal `(communicated_soil_data ?w)` that is not yet satisfied:
        - Estimate the cost to satisfy this goal (`item_h`).
        - Check if `(have_soil_analysis ?r ?w)` is true for any rover `?r`.
        - If data is held by rover `r_holding`: `item_h = min_nav_cost(rover_locations[r_holding], comm_waypoint) + 1 (communicate)`. Minimize over all communication waypoints.
        - If data is NOT held:
            - If `(at_soil_sample ?w)` is NOT true in the current state: `item_h = infinity` (sample is gone and not held).
            - If `(at_soil_sample ?w)` IS true: Need to sample and communicate.
                - Find a rover `?r` capable of soil analysis.
                - Cost to acquire = `nav_cost(rover_locations[?r], ?w) + (1 if ?r's store is full) + 1 (sample)`.
                - Cost to communicate from `?w` = `min_nav_cost(?w, comm_waypoint) + 1 (communicate)`. Minimize over all communication waypoints.
                - `item_h = min(Cost to acquire + Cost to communicate)` over all capable rovers `?r`.
        - Add `item_h` to the total heuristic `h`.
    4.  For each rock data goal `(communicated_rock_data ?w)` that is not yet satisfied:
        - Follow the same logic as for soil data goals, substituting rock predicates and capabilities.
    5.  For each image data goal `(communicated_image_data ?o ?m)` that is not yet satisfied:
        - Estimate the cost to satisfy this goal (`item_h`).
        - Check if `(have_image ?r ?o ?m)` is true for any rover `?r`.
        - If data is held by rover `r_holding`: `item_h = min_nav_cost(rover_locations[r_holding], comm_waypoint) + 1 (communicate)`. Minimize over all communication waypoints.
        - If data is NOT held: Need to take image and communicate.
            - Find a rover `?r` capable of imaging.
            - Find a camera `?i` on board `?r` supporting mode `?m`.
            - Find a waypoint `?p` visible from objective `?o`.
            - Cost to acquire image at `?p` with `?i` by `?r`:
                - `nav_cost(rover_locations[?r], ?p)` (navigate to image spot).
                - Calibration cost: If `(calibrated ?i ?r)` is true, cost is 0. Else, find calibration target `?t` for `?i` and waypoint `?w_cal` visible from `?t`. Cost = `nav_cost(?p, ?w_cal) + 1 (calibrate) + nav_cost(?w_cal, ?p)` (navigate to cal spot, calibrate, navigate back). Minimize over `?t`, `?w_cal`. If calibration is impossible, this path is invalid.
                - `+ 1 (take_image)`.
            - Cost to communicate from `?p` = `min_nav_cost(?p, comm_waypoint) + 1 (communicate)`. Minimize over all communication waypoints.
            - `item_h = min(Cost to acquire + Cost to communicate)` over all capable rovers `?r`, cameras `?i`, and image waypoints `?p`.
        - Add `item_h` to the total heuristic `h`.
    6.  If any `item_h` calculation resulted in infinity (meaning a sub-goal like acquiring data or reaching a communication point is impossible from this state), the total heuristic is infinity.
    7.  Return the total heuristic `h`.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        super().__init__(task)

        self.infinity = float('inf')

        # --- Parse Goals ---
        self.goal_soil = set()
        self.goal_rock = set()
        self.goal_images = set() # set of (objective, mode) tuples

        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "communicated_soil_data" and len(parts) == 2:
                self.goal_soil.add(parts[1])
            elif parts[0] == "communicated_rock_data" and len(parts) == 2:
                self.goal_rock.add(parts[1])
            elif parts[0] == "communicated_image_data" and len(parts) == 3:
                self.goal_images.add((parts[1], parts[2]))

        # --- Parse Static Facts ---
        self.rover_capabilities = {} # rover -> set of {'soil', 'rock', 'imaging'}
        self.store_owners = {} # store -> rover
        self.lander_location = None
        self.comm_waypoint_candidates = set() # waypoints visible from lander
        self.nav_graph = {} # waypoint -> set of connected waypoints
        self.all_waypoints = set()
        self.objective_image_waypoints = {} # objective -> set of waypoints
        self.camera_cal_target = {} # camera -> calibration_target
        self.cal_target_cal_waypoints = {} # calibration_target -> set of waypoints
        self.camera_modes = {} # camera -> set of modes
        self.rover_cameras = {} # rover -> set of cameras
        self.initial_soil_samples = set() # waypoints with initial soil samples
        self.initial_rock_samples = set() # waypoints with initial rock samples

        # Temporary structures for graph building
        can_traverse_edges = set() # (w1, w2) tuples

        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "equipped_for_soil_analysis" and len(parts) == 2:
                rover = parts[1]
                self.rover_capabilities.setdefault(rover, set()).add('soil')
            elif parts[0] == "equipped_for_rock_analysis" and len(parts) == 2:
                rover = parts[1]
                self.rover_capabilities.setdefault(rover, set()).add('rock')
            elif parts[0] == "equipped_for_imaging" and len(parts) == 2:
                rover = parts[1]
                self.rover_capabilities.setdefault(rover, set()).add('imaging')
            elif parts[0] == "store_of" and len(parts) == 3:
                store, rover = parts[1], parts[2]
                self.store_owners[store] = rover
            elif parts[0] == "at_lander" and len(parts) == 3:
                # Assuming only one lander
                self.lander_location = parts[2]
            elif parts[0] == "visible" and len(parts) == 3:
                w1, w2 = parts[1], parts[2]
                # Visible is symmetric, but can_traverse defines actual movement
                # We will use can_traverse for the navigation graph
                self.all_waypoints.add(w1)
                self.all_waypoints.add(w2)
            elif parts[0] == "can_traverse" and len(parts) == 4:
                 # Use can_traverse for navigation graph, assuming it implies visible and is symmetric
                 # The example static facts show can_traverse implies visible and is symmetric for rovers
                 r, w1, w2 = parts[1], parts[2], parts[3]
                 # Assuming can_traverse is the same for all rovers and symmetric
                 can_traverse_edges.add(tuple(sorted((w1, w2)))) # Add unique edge regardless of direction/rover
                 self.all_waypoints.add(w1)
                 self.all_waypoints.add(w2)
            elif parts[0] == "visible_from" and len(parts) == 3:
                objective, waypoint = parts[1], parts[2]
                self.objective_image_waypoints.setdefault(objective, set()).add(waypoint)
                self.all_waypoints.add(waypoint)
            elif parts[0] == "calibration_target" and len(parts) == 3:
                camera, target = parts[1], parts[2]
                self.camera_cal_target[camera] = target
            elif parts[0] == "supports" and len(parts) == 3:
                camera, mode = parts[1], parts[2]
                self.camera_modes.setdefault(camera, set()).add(mode)
            elif parts[0] == "on_board" and len(parts) == 3:
                camera, rover = parts[1], parts[2]
                self.rover_cameras.setdefault(rover, set()).add(camera)
            elif parts[0] == "at_soil_sample" and len(parts) == 2:
                self.initial_soil_samples.add(parts[1])
                self.all_waypoints.add(parts[1])
            elif parts[0] == "at_rock_sample" and len(parts) == 2:
                self.initial_rock_samples.add(parts[1])
                self.all_waypoints.add(parts[1])

        # Build the actual navigation graph from can_traverse edges
        self.nav_graph = {w: set() for w in self.all_waypoints}
        for w1, w2 in can_traverse_edges:
             self.nav_graph[w1].add(w2)
             self.nav_graph[w2].add(w1)


        # Identify communication waypoints (visible from lander)
        if self.lander_location:
             for fact in self.static:
                 # Check visible facts involving the lander's location
                 parts = get_parts(fact)
                 if parts[0] == "visible" and len(parts) == 3:
                     w1, w2 = parts[1], parts[2]
                     if w1 == self.lander_location:
                         self.comm_waypoint_candidates.add(w2)
                     if w2 == self.lander_location:
                         self.comm_waypoint_candidates.add(w1)


        # --- Compute Navigation Distances (BFS from all waypoints) ---
        self.nav_distances = {} # (w1, w2) -> distance

        for start_w in self.all_waypoints:
            distances = {w: self.infinity for w in self.all_waypoints}
            distances[start_w] = 0
            queue = deque([start_w])

            while queue:
                curr_w = queue.popleft()

                if curr_w not in self.nav_graph: continue # Should not happen if all_waypoints is correct

                for next_w in self.nav_graph.get(curr_w, set()): # Use .get for safety
                    if distances[next_w] == self.infinity:
                        distances[next_w] = distances[curr_w] + 1
                        queue.append(next_w)

            for end_w in self.all_waypoints:
                 self.nav_distances[(start_w, end_w)] = distances[end_w]

    def get_nav_cost(self, w1, w2):
         """Helper to get navigation cost, returns infinity if path doesn't exist."""
         if w1 is None or w2 is None: return self.infinity # Handle cases where location is unknown
         return self.nav_distances.get((w1, w2), self.infinity)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        infinity = self.infinity

        # --- Parse Current State ---
        rover_locations = {} # rover -> waypoint
        has_soil = set() # (rover, waypoint)
        has_rock = set() # (rover, waypoint)
        has_image = set() # (rover, objective, mode)
        store_full = set() # store
        calibrated_cams = set() # (camera, rover)
        soil_samples_present = set() # waypoint
        rock_samples_present = set() # waypoint

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at" and len(parts) == 3 and parts[1].startswith("rover"):
                rover_locations[parts[1]] = parts[2]
            elif parts[0] == "have_soil_analysis" and len(parts) == 3:
                 has_soil.add((parts[1], parts[2]))
            elif parts[0] == "have_rock_analysis" and len(parts) == 3:
                 has_rock.add((parts[1], parts[2]))
            elif parts[0] == "have_image" and len(parts) == 4:
                 has_image.add((parts[1], parts[2], parts[3]))
            elif parts[0] == "full" and len(parts) == 2:
                 store_full.add(parts[1])
            elif parts[0] == "calibrated" and len(parts) == 3:
                 calibrated_cams.add((parts[1], parts[2]))
            elif parts[0] == "at_soil_sample" and len(parts) == 2:
                 soil_samples_present.add(parts[1])
            elif parts[0] == "at_rock_sample" and len(parts) == 2:
                 rock_samples_present.add(parts[1])

        # Ensure all rovers known from static facts have a location in the state
        # If not, it's an invalid state representation for this heuristic.
        # Assume valid states always have all rover locations.
        # for rover in self.rover_capabilities:
        #      if rover not in rover_locations:
        #           return infinity # Or handle as error


        total_h = 0

        # --- Estimate Cost for Soil Goals ---
        for w in self.goal_soil:
            if f"(communicated_soil_data {w})" not in state:
                item_h = infinity

                # Check if data is already held by any rover
                data_held_by_rover = None
                for r in self.rover_capabilities:
                    if 'soil' in self.rover_capabilities.get(r, set()) and (r, w) in has_soil:
                        data_held_by_rover = r
                        break

                if data_held_by_rover:
                    # Data is held, just need to communicate
                    r_holding = data_held_by_rover
                    current_loc = rover_locations.get(r_holding)
                    if current_loc:
                        min_comm_cost = min(
                            (self.get_nav_cost(current_loc, comm_w) + 1 # navigate + communicate
                             for comm_w in self.comm_waypoint_candidates),
                            default=infinity
                        )
                        item_h = min_comm_cost
                    # else: rover location unknown, item_h remains infinity

                else: # Data is not held by any rover
                    if w not in soil_samples_present:
                        # Sample was needed, is gone, and no one has it. Unachievable from here.
                        item_h = infinity
                    else:
                        # Sample is present, need to acquire and communicate
                        min_acquire_comm_cost = infinity
                        for r in self.rover_capabilities:
                            if 'soil' in self.rover_capabilities.get(r, set()):
                                current_loc = rover_locations.get(r)
                                if current_loc:
                                    # Cost to acquire at w by r
                                    acquire_cost = self.get_nav_cost(current_loc, w) # navigate to sample
                                    if acquire_cost != infinity:
                                        acquire_cost += 1 # sample action
                                        # Check if drop is needed before sampling
                                        rover_store = next((s for s, owner in self.store_owners.items() if owner == r), None)
                                        if rover_store and rover_store in store_full:
                                            acquire_cost += 1 # drop action

                                        # Cost to communicate from w
                                        min_comm_from_sample_loc = min(
                                            (self.get_nav_cost(w, comm_w) + 1 # navigate + communicate
                                             for comm_w in self.comm_waypoint_candidates),
                                            default=infinity
                                        )

                                        if min_comm_from_sample_loc != infinity:
                                            min_acquire_comm_cost = min(min_acquire_comm_cost, acquire_cost + min_comm_from_sample_loc)
                        item_h = min_acquire_comm_cost

                total_h += item_h

        # --- Estimate Cost for Rock Goals ---
        for w in self.goal_rock:
            if f"(communicated_rock_data {w})" not in state:
                item_h = infinity

                # Check if data is already held by any rover
                data_held_by_rover = None
                for r in self.rover_capabilities:
                    if 'rock' in self.rover_capabilities.get(r, set()) and (r, w) in has_rock:
                        data_held_by_rover = r
                        break

                if data_held_by_rover:
                    # Data is held, just need to communicate
                    r_holding = data_held_by_rover
                    current_loc = rover_locations.get(r_holding)
                    if current_loc:
                        min_comm_cost = min(
                            (self.get_nav_cost(current_loc, comm_w) + 1 # navigate + communicate
                             for comm_w in self.comm_waypoint_candidates),
                            default=infinity
                        )
                        item_h = min_comm_cost
                    # else: rover location unknown, item_h remains infinity

                else: # Data is not held by any rover
                    if w not in rock_samples_present:
                        # Sample was needed, is gone, and no one has it. Unachievable from here.
                        item_h = infinity
                    else:
                        # Sample is present, need to acquire and communicate
                        min_acquire_comm_cost = infinity
                        for r in self.rover_capabilities:
                            if 'rock' in self.rover_capabilities.get(r, set()):
                                current_loc = rover_locations.get(r)
                                if current_loc:
                                    # Cost to acquire at w by r
                                    acquire_cost = self.get_nav_cost(current_loc, w) # navigate to sample
                                    if acquire_cost != infinity:
                                        acquire_cost += 1 # sample action
                                        # Check if drop is needed before sampling
                                        rover_store = next((s for s, owner in self.store_owners.items() if owner == r), None)
                                        if rover_store and rover_store in store_full:
                                            acquire_cost += 1 # drop action

                                        # Cost to communicate from w
                                        min_comm_from_sample_loc = min(
                                            (self.get_nav_cost(w, comm_w) + 1 # navigate + communicate
                                             for comm_w in self.comm_waypoint_candidates),
                                            default=infinity
                                        )

                                        if min_comm_from_sample_loc != infinity:
                                            min_acquire_comm_cost = min(min_acquire_comm_cost, acquire_cost + min_comm_from_sample_loc)
                        item_h = min_acquire_comm_cost

                total_h += item_h


        # --- Estimate Cost for Image Goals ---
        for o, m in self.goal_images:
            if f"(communicated_image_data {o} {m})" not in state:
                item_h = infinity

                # Check if data is already held by any rover
                data_held_by_rover = None
                for r in self.rover_capabilities:
                    if 'imaging' in self.rover_capabilities.get(r, set()) and (r, o, m) in has_image:
                        data_held_by_rover = r
                        break

                if data_held_by_rover:
                    # Data is held, just need to communicate
                    r_holding = data_held_by_rover
                    current_loc = rover_locations.get(r_holding)
                    if current_loc:
                        min_comm_cost = min(
                            (self.get_nav_cost(current_loc, comm_w) + 1 # navigate + communicate
                             for comm_w in self.comm_waypoint_candidates),
                            default=infinity
                        )
                        item_h = min_comm_cost
                    # else: rover location unknown, item_h remains infinity

                else: # Data is not held by any rover
                    # Need to acquire image and communicate
                    min_acquire_comm_cost = infinity

                    # Find suitable image waypoints for this objective
                    image_wps = self.objective_image_waypoints.get(o, set())
                    if not image_wps:
                         # Cannot image this objective
                         item_h = infinity # Remains infinity
                    else:
                        for r in self.rover_capabilities:
                            if 'imaging' in self.rover_capabilities.get(r, set()):
                                current_loc_r = rover_locations.get(r)
                                if not current_loc_r: continue # Rover location unknown

                                for i in self.rover_cameras.get(r, set()):
                                    if m in self.camera_modes.get(i, set()):
                                        # Found a capable rover and camera for mode m

                                        for p in image_wps:
                                            # Cost to navigate to image waypoint p
                                            nav_to_image_wp_cost = self.get_nav_cost(current_loc_r, p)
                                            if nav_to_image_wp_cost == infinity: continue # Cannot reach image waypoint

                                            # Cost to calibrate camera i for rover r at waypoint p
                                            cal_cost = 0
                                            if (i, r) not in calibrated_cams:
                                                # Need to calibrate
                                                min_cal_nav_cost = infinity
                                                cal_target = self.camera_cal_target.get(i)
                                                if cal_target:
                                                    cal_wps = self.cal_target_cal_waypoints.get(cal_target, set())
                                                    for w_cal in cal_wps:
                                                        nav_p_to_wcal = self.get_nav_cost(p, w_cal)
                                                        nav_wcal_to_p = self.get_nav_cost(w_cal, p)
                                                        if nav_p_to_wcal != infinity and nav_wcal_to_p != infinity:
                                                            min_cal_nav_cost = min(min_cal_nav_cost, nav_p_to_wcal + 1 + nav_wcal_to_p) # nav + calibrate + nav back
                                                if min_cal_nav_cost == infinity:
                                                    cal_cost = infinity # Cannot calibrate this camera
                                                else:
                                                    cal_cost = min_cal_nav_cost

                                            if cal_cost == infinity: continue # Cannot calibrate this camera for this rover

                                            # Cost to take image
                                            take_image_action_cost = 1

                                            # Cost to communicate from image waypoint p
                                            min_comm_from_image_loc = min(
                                                (self.get_nav_cost(p, comm_w) + 1 # navigate + communicate
                                                 for comm_w in self.comm_waypoint_candidates),
                                                default=infinity
                                            )

                                            if min_comm_from_image_loc != infinity:
                                                 # Total cost for this path (rover, camera, image_wp, cal_wp, comm_wp)
                                                 path_cost = nav_to_image_wp_cost + cal_cost + take_image_action_cost + min_comm_from_image_loc
                                                 min_acquire_comm_cost = min(min_acquire_comm_cost, path_cost)

                        item_h = min_acquire_comm_cost
                total_h += item_h

        # If any item cost was infinity, the total cost is infinity
        if total_h >= infinity:
             return infinity

        return total_h
